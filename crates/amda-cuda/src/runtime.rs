//! CUDA Runtime API (`cuda*`) C ABI exports layered over AMDA contexts.
//!
//! Includes NVIDIA `nvcc` host-runtime registration entrypoints (`__cudaRegisterFatBinary`,
//! `__cudaRegisterFunction`, `__cudaPushCallConfiguration`, `cudaLaunchKernel`, etc.).

#![allow(non_snake_case, clippy::not_unsafe_ptr_arg_deref)]

use crate::driver::{
    cuDeviceGetAttribute, cuDeviceGetCount, cuDriverGetVersion, cuEventCreate, cuEventDestroy_v2,
    cuEventElapsedTime, cuEventQuery, cuEventRecord, cuEventSynchronize, cuInit, cuLaunchKernel,
    cuMemAlloc_v2, cuMemFree_v2, cuMemGetInfo_v2, cuMemcpyDtoD_v2, cuMemcpyDtoH_v2,
    cuMemcpyHtoD_v2, cuModuleGetFunction, cuModuleUnload, cuStreamCreate,
    cuStreamDestroy_v2, cuStreamSynchronize, cuStreamWaitEvent, ensure_initialized_and_context,
    get_device_info, is_device_ptr, isa_hw_metrics, load_module_from_slice, map_runtime_err,
    read_module_image_bytes, reset_device_contexts, validate_stream,
};
use crate::types::{
    CUdeviceptr, CUevent, CUfunction, CUmodule, CUresult, CUstream, CudaDeviceProp, CudaErrorT,
    CudaEventT, CudaMemcpyKind, CudaStreamT, Dim3, FatBinCWrapper,
};
use amda_compiler::fatbin::FATBIN_WRAPPER_MAGIC;
use amda_runtime::{trace_log, Context};
use core::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::CString;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};

const MEMSET_CHUNK_BYTES: usize = 1024 * 1024;
const PINNED_HOST_ALIGN: usize = 4096;
const CUDA_RUNTIME_VERSION: c_int = 12040;

static PINNED_HOST_ALLOCS: OnceLock<Mutex<HashMap<usize, (usize, usize)>>> = OnceLock::new();

fn pinned_host_allocs() -> &'static Mutex<HashMap<usize, (usize, usize)>> {
    PINNED_HOST_ALLOCS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn is_pinned_host_ptr(addr: usize) -> bool {
    if addr == 0 {
        return false;
    }
    let Some(lock) = PINNED_HOST_ALLOCS.get() else {
        return false;
    };
    let Ok(map) = lock.lock() else {
        return false;
    };
    for (&base, &(size, _)) in &*map {
        if addr >= base && addr < base.saturating_add(size) {
            return true;
        }
    }
    false
}

#[derive(Default)]
struct FatbinRegistry {
    next_handle: usize,
    fatbins: HashMap<usize, Vec<u8>>,
    functions: HashMap<usize, (usize, String)>,
    vars: HashMap<usize, (usize, String, usize)>,
    loaded_modules: HashMap<(usize, usize), usize>,
    cached_funcs: HashMap<(usize, usize, String), usize>,
}

impl FatbinRegistry {
    fn new() -> Self {
        Self {
            next_handle: 0x5001,
            ..Self::default()
        }
    }

    fn alloc_handle(&mut self) -> usize {
        let id = self.next_handle;
        self.next_handle = self.next_handle.wrapping_add(1).max(0x5001);
        id
    }
}

static FATBIN_REGISTRY: OnceLock<Mutex<FatbinRegistry>> = OnceLock::new();

fn fatbin_registry() -> &'static Mutex<FatbinRegistry> {
    FATBIN_REGISTRY.get_or_init(|| Mutex::new(FatbinRegistry::new()))
}

#[derive(Clone, Copy)]
struct CallConfig {
    grid_dim: Dim3,
    block_dim: Dim3,
    shared_mem: usize,
    stream: CudaStreamT,
}

thread_local! {
    static CURRENT_DEVICE: Cell<c_int> = const { Cell::new(0) };
    static LAST_ERROR: Cell<CudaErrorT> = const { Cell::new(CudaErrorT::CudaSuccess) };
    static CALL_CONFIG_STACK: RefCell<Vec<CallConfig>> = const { RefCell::new(Vec::new()) };
}

fn map_cu_result(res: CUresult) -> CudaErrorT {
    match res {
        CUresult::CUDA_SUCCESS => CudaErrorT::CudaSuccess,
        CUresult::CUDA_ERROR_INVALID_VALUE => CudaErrorT::CudaErrorInvalidValue,
        CUresult::CUDA_ERROR_OUT_OF_MEMORY => CudaErrorT::CudaErrorMemoryAllocation,
        CUresult::CUDA_ERROR_NOT_INITIALIZED => CudaErrorT::CudaErrorInitializationError,
        CUresult::CUDA_ERROR_NOT_READY => CudaErrorT::CudaErrorNotReady,
        CUresult::CUDA_ERROR_LAUNCH_FAILED => CudaErrorT::CudaErrorLaunchFailure,
        _ => CudaErrorT::CudaErrorUnknown,
    }
}

fn rt_try<F>(f: F) -> CudaErrorT
where
    F: FnOnce() -> CudaErrorT,
{
    let res = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(err) => err,
        Err(_) => CudaErrorT::CudaErrorUnknown,
    };
    if res != CudaErrorT::CudaSuccess {
        LAST_ERROR.set(res);
    }
    res
}

fn ensure_runtime_ctx_arc() -> Result<Arc<Context>, CudaErrorT> {
    let dev = CURRENT_DEVICE.get();
    if dev < 0 {
        return Err(CudaErrorT::CudaErrorInvalidDevice);
    }
    ensure_initialized_and_context(dev as usize).map_err(map_cu_result)
}

fn ensure_runtime_ctx() -> Result<(), CudaErrorT> {
    ensure_runtime_ctx_arc().map(|_| ())
}

fn memcpy_impl(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: CudaMemcpyKind,
) -> CudaErrorT {
    if count == 0 {
        return CudaErrorT::CudaSuccess;
    }
    if dst.is_null() || src.is_null() {
        return CudaErrorT::CudaErrorInvalidValue;
    }
    let resolved = match kind {
        CudaMemcpyKind::Default => {
            let dst_dev = is_device_ptr(dst as u64).is_some();
            let src_dev = is_device_ptr(src as u64).is_some();
            match (dst_dev, src_dev) {
                (true, true) => CudaMemcpyKind::DeviceToDevice,
                (true, false) => CudaMemcpyKind::HostToDevice,
                (false, true) => CudaMemcpyKind::DeviceToHost,
                (false, false) => CudaMemcpyKind::HostToHost,
            }
        }
        other => other,
    };
    match resolved {
        CudaMemcpyKind::HostToHost => {
            unsafe {
                core::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
            }
            CudaErrorT::CudaSuccess
        }
        CudaMemcpyKind::HostToDevice => {
            map_cu_result(cuMemcpyHtoD_v2(dst as CUdeviceptr, src, count))
        }
        CudaMemcpyKind::DeviceToHost => {
            map_cu_result(cuMemcpyDtoH_v2(dst, src as CUdeviceptr, count))
        }
        CudaMemcpyKind::DeviceToDevice => map_cu_result(cuMemcpyDtoD_v2(
            dst as CUdeviceptr,
            src as CUdeviceptr,
            count,
        )),
        CudaMemcpyKind::Default => unreachable!(),
    }
}

fn memset_impl(ctx: &Context, dev_ptr: *mut c_void, value: c_int, count: usize) -> CudaErrorT {
    if count == 0 {
        return CudaErrorT::CudaSuccess;
    }
    if dev_ptr.is_null() {
        return CudaErrorT::CudaErrorInvalidValue;
    }
    let fill_len = count.min(MEMSET_CHUNK_BYTES);
    let buf = vec![(value & 0xFF) as u8; fill_len];
    let base = dev_ptr as u64;
    let mut offset = 0usize;
    while offset < count {
        let step = (count - offset).min(MEMSET_CHUNK_BYTES);
        let Some(addr) = base.checked_add(offset as u64) else {
            return CudaErrorT::CudaErrorInvalidValue;
        };
        if let Err(e) = ctx.copy_htod(addr, &buf[..step]) {
            return map_cu_result(map_runtime_err(e));
        }
        offset += step;
    }
    CudaErrorT::CudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaGetDeviceCount(count: *mut c_int) -> CudaErrorT {
    rt_try(|| {
        let rc = cuInit(0);
        if rc != CUresult::CUDA_SUCCESS {
            return map_cu_result(rc);
        }
        map_cu_result(cuDeviceGetCount(count))
    })
}

#[no_mangle]
pub extern "C" fn cudaSetDevice(device: c_int) -> CudaErrorT {
    rt_try(|| {
        if device < 0 {
            return CudaErrorT::CudaErrorInvalidDevice;
        }
        let mut count: c_int = 0;
        let rc = cuInit(0);
        if rc != CUresult::CUDA_SUCCESS {
            return map_cu_result(rc);
        }
        let rc = cuDeviceGetCount(&mut count);
        if rc != CUresult::CUDA_SUCCESS {
            return map_cu_result(rc);
        }
        if device >= count {
            return CudaErrorT::CudaErrorInvalidDevice;
        }
        CURRENT_DEVICE.set(device);
        match ensure_initialized_and_context(device as usize) {
            Ok(_) => CudaErrorT::CudaSuccess,
            Err(e) => map_cu_result(e),
        }
    })
}

#[no_mangle]
pub extern "C" fn cudaGetDevice(device: *mut c_int) -> CudaErrorT {
    rt_try(|| {
        if device.is_null() {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        unsafe {
            *device = CURRENT_DEVICE.get();
        }
        CudaErrorT::CudaSuccess
    })
}

#[no_mangle]
pub extern "C" fn cudaGetDeviceProperties(prop: *mut CudaDeviceProp, device: c_int) -> CudaErrorT {
    rt_try(|| {
        if prop.is_null() {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        if device < 0 {
            return CudaErrorT::CudaErrorInvalidDevice;
        }
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        let info = match get_device_info(device as usize) {
            Ok(i) => i,
            Err(_) => return CudaErrorT::CudaErrorInvalidDevice,
        };
        let (core_khz, mem_khz, bus_bits, l2_bytes) = isa_hw_metrics(info.isa);
        let mut out = CudaDeviceProp::default();
        let name_bytes = info.name.as_bytes();
        let copy_len = name_bytes.len().min(out.name.len() - 1);
        for (i, &b) in name_bytes[..copy_len].iter().enumerate() {
            out.name[i] = b as c_char;
        }
        let isa_bytes = info.isa.to_string();
        for (i, &b) in isa_bytes.as_bytes().iter().take(12).enumerate() {
            out.uuid[i] = b;
        }
        out.uuid[15] = device as u8;
        out.totalGlobalMem = info.vram_bytes as usize;
        out.sharedMemPerBlock = info.features.lds_bytes_per_cu as usize;
        out.regsPerBlock = 65_536;
        out.warpSize = info.isa.native_wave_size() as c_int;
        out.memPitch = 2_147_483_647;
        out.maxThreadsPerBlock = 1024;
        out.maxThreadsDim = [1024, 1024, 64];
        out.maxGridSize = [2_147_483_647, 65_535, 65_535];
        out.clockRate = core_khz;
        out.totalConstMem = 64 * 1024;
        out.major = 8;
        out.minor = 0;
        out.multiProcessorCount = info.features.compute_units as c_int;
        out.integrated = c_int::from(info.features.unified_memory);
        out.canMapHostMemory = 1;
        out.computeMode = 0;
        out.concurrentKernels = 1;
        out.memoryClockRate = mem_khz;
        out.memoryBusWidth = bus_bits;
        out.l2CacheSize = l2_bytes;
        out.maxThreadsPerMultiProcessor = 2048;
        out.unifiedAddressing = 1;
        out.managedMemory = 1;
        out.asyncEngineCount = 2;
        unsafe {
            *prop = out;
        }
        CudaErrorT::CudaSuccess
    })
}

#[no_mangle]
pub extern "C" fn cudaDeviceGetAttribute(
    value: *mut c_int,
    attr: c_int,
    device: c_int,
) -> CudaErrorT {
    rt_try(|| {
        if value.is_null() {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        if device < 0 {
            return CudaErrorT::CudaErrorInvalidDevice;
        }
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        map_cu_result(cuDeviceGetAttribute(value, attr, device))
    })
}

#[no_mangle]
pub extern "C" fn cudaDriverGetVersion(driverVersion: *mut c_int) -> CudaErrorT {
    rt_try(|| {
        if driverVersion.is_null() {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        map_cu_result(cuDriverGetVersion(driverVersion))
    })
}

#[no_mangle]
pub extern "C" fn cudaRuntimeGetVersion(runtimeVersion: *mut c_int) -> CudaErrorT {
    rt_try(|| {
        if runtimeVersion.is_null() {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        unsafe {
            *runtimeVersion = CUDA_RUNTIME_VERSION;
        }
        CudaErrorT::CudaSuccess
    })
}

#[no_mangle]
pub extern "C" fn cudaDeviceReset() -> CudaErrorT {
    rt_try(|| {
        let ctx = match ensure_runtime_ctx_arc() {
            Ok(c) => c,
            Err(e) => return e,
        };
        if ctx.synchronize().is_err() {
            return CudaErrorT::CudaErrorLaunchFailure;
        }
        let dev = ctx.device_index();
        {
            let mut reg = fatbin_registry().lock().expect("fatbin registry lock");
            reg.loaded_modules.retain(|&(_, d), _| d != dev);
            reg.cached_funcs.retain(|&(_, d, _), _| d != dev);
        }
        match reset_device_contexts(dev) {
            Ok(()) => CudaErrorT::CudaSuccess,
            Err(e) => map_cu_result(e),
        }
    })
}

#[no_mangle]
pub extern "C" fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> CudaErrorT {
    rt_try(|| {
        if devPtr.is_null() {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        let mut dptr: CUdeviceptr = 0;
        let rc = cuMemAlloc_v2(&mut dptr, size);
        if rc != CUresult::CUDA_SUCCESS {
            return map_cu_result(rc);
        }
        unsafe {
            *devPtr = dptr as *mut c_void;
        }
        CudaErrorT::CudaSuccess
    })
}

#[no_mangle]
pub extern "C" fn cudaFree(devPtr: *mut c_void) -> CudaErrorT {
    rt_try(|| {
        if devPtr.is_null() {
            return CudaErrorT::CudaSuccess;
        }
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        map_cu_result(cuMemFree_v2(devPtr as CUdeviceptr))
    })
}

#[no_mangle]
pub extern "C" fn cudaHostAlloc(
    pHost: *mut *mut c_void,
    size: usize,
    _flags: c_uint,
) -> CudaErrorT {
    rt_try(|| {
        if pHost.is_null() {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        let req_bytes = size.max(1);
        let alloc_size = req_bytes
            .checked_add(PINNED_HOST_ALIGN - 1)
            .map(|n| n & !(PINNED_HOST_ALIGN - 1))
            .ok_or(CudaErrorT::CudaErrorMemoryAllocation);
        let alloc_size = match alloc_size {
            Ok(n) => n,
            Err(e) => return e,
        };
        let Ok(layout) = Layout::from_size_align(alloc_size, PINNED_HOST_ALIGN) else {
            return CudaErrorT::CudaErrorMemoryAllocation;
        };
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return CudaErrorT::CudaErrorMemoryAllocation;
        }
        pinned_host_allocs()
            .lock()
            .expect("pinned host alloc lock")
            .insert(ptr as usize, (req_bytes, alloc_size));
        unsafe {
            *pHost = ptr as *mut c_void;
        }
        CudaErrorT::CudaSuccess
    })
}

#[no_mangle]
pub extern "C" fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> CudaErrorT {
    cudaHostAlloc(ptr, size, 0)
}

#[no_mangle]
pub extern "C" fn cudaFreeHost(ptr: *mut c_void) -> CudaErrorT {
    rt_try(|| {
        if ptr.is_null() {
            return CudaErrorT::CudaSuccess;
        }
        let addr = ptr as usize;
        let entry = pinned_host_allocs()
            .lock()
            .expect("pinned host alloc lock")
            .remove(&addr);
        let Some((_, alloc_size)) = entry else {
            return CudaErrorT::CudaErrorInvalidValue;
        };
        let Ok(layout) = Layout::from_size_align(alloc_size, PINNED_HOST_ALIGN) else {
            return CudaErrorT::CudaErrorInvalidValue;
        };
        unsafe {
            dealloc(ptr as *mut u8, layout);
        }
        CudaErrorT::CudaSuccess
    })
}

#[no_mangle]
pub extern "C" fn cudaMemset(devPtr: *mut c_void, value: c_int, count: usize) -> CudaErrorT {
    rt_try(|| {
        if count == 0 {
            return CudaErrorT::CudaSuccess;
        }
        if devPtr.is_null() {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        let ctx = match ensure_runtime_ctx_arc() {
            Ok(c) => c,
            Err(e) => return e,
        };
        memset_impl(&ctx, devPtr, value, count)
    })
}

#[no_mangle]
pub extern "C" fn cudaMemsetAsync(
    devPtr: *mut c_void,
    value: c_int,
    count: usize,
    stream: CudaStreamT,
) -> CudaErrorT {
    rt_try(|| {
        if count == 0 {
            if let Err(e) = ensure_runtime_ctx() {
                return e;
            }
            return match validate_stream(stream as CUstream) {
                Ok(()) => CudaErrorT::CudaSuccess,
                Err(e) => map_cu_result(e),
            };
        }
        if devPtr.is_null() {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        let ctx = match ensure_runtime_ctx_arc() {
            Ok(c) => c,
            Err(e) => return e,
        };
        if let Err(e) = validate_stream(stream as CUstream) {
            return map_cu_result(e);
        }
        memset_impl(&ctx, devPtr, value, count)
    })
}

#[no_mangle]
pub extern "C" fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: CudaMemcpyKind,
) -> CudaErrorT {
    rt_try(|| {
        if count == 0 {
            return CudaErrorT::CudaSuccess;
        }
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        memcpy_impl(dst, src, count, kind)
    })
}

#[no_mangle]
pub extern "C" fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: CudaMemcpyKind,
    stream: CudaStreamT,
) -> CudaErrorT {
    rt_try(|| {
        if count == 0 {
            if let Err(e) = ensure_runtime_ctx() {
                return e;
            }
            return match validate_stream(stream as CUstream) {
                Ok(()) => CudaErrorT::CudaSuccess,
                Err(e) => map_cu_result(e),
            };
        }
        if dst.is_null() || src.is_null() {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        if let Err(e) = validate_stream(stream as CUstream) {
            return map_cu_result(e);
        }
        memcpy_impl(dst, src, count, kind)
    })
}

#[no_mangle]
pub extern "C" fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> CudaErrorT {
    rt_try(|| {
        if free.is_null() || total.is_null() {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        map_cu_result(cuMemGetInfo_v2(free, total))
    })
}

#[no_mangle]
pub extern "C" fn cudaDeviceSynchronize() -> CudaErrorT {
    rt_try(|| {
        let ctx = match ensure_initialized_and_context(CURRENT_DEVICE.get().max(0) as usize) {
            Ok(c) => c,
            Err(e) => return map_cu_result(e),
        };
        match ctx.synchronize() {
            Ok(()) => CudaErrorT::CudaSuccess,
            Err(_) => CudaErrorT::CudaErrorLaunchFailure,
        }
    })
}

#[no_mangle]
pub extern "C" fn cudaStreamCreate(pStream: *mut CudaStreamT) -> CudaErrorT {
    rt_try(|| {
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        map_cu_result(cuStreamCreate(pStream as *mut CUstream, 0))
    })
}

#[no_mangle]
pub extern "C" fn cudaStreamDestroy(stream: CudaStreamT) -> CudaErrorT {
    rt_try(|| {
        if stream.is_null() {
            return CudaErrorT::CudaSuccess;
        }
        map_cu_result(cuStreamDestroy_v2(stream as CUstream))
    })
}

#[no_mangle]
pub extern "C" fn cudaStreamSynchronize(stream: CudaStreamT) -> CudaErrorT {
    rt_try(|| {
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        map_cu_result(cuStreamSynchronize(stream as CUstream))
    })
}

#[no_mangle]
pub extern "C" fn cudaStreamWaitEvent(
    stream: CudaStreamT,
    event: CudaEventT,
    flags: c_uint,
) -> CudaErrorT {
    rt_try(|| {
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        map_cu_result(cuStreamWaitEvent(stream as CUstream, event as CUevent, flags))
    })
}

#[no_mangle]
pub extern "C" fn cudaEventCreate(event: *mut CudaEventT) -> CudaErrorT {
    cudaEventCreateWithFlags(event, 0)
}

#[no_mangle]
pub extern "C" fn cudaEventCreateWithFlags(event: *mut CudaEventT, flags: c_uint) -> CudaErrorT {
    rt_try(|| {
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        map_cu_result(cuEventCreate(event as *mut CUevent, flags))
    })
}

#[no_mangle]
pub extern "C" fn cudaEventDestroy(event: CudaEventT) -> CudaErrorT {
    rt_try(|| {
        if event.is_null() {
            return CudaErrorT::CudaSuccess;
        }
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        map_cu_result(cuEventDestroy_v2(event as CUevent))
    })
}

#[no_mangle]
pub extern "C" fn cudaEventRecord(event: CudaEventT, stream: CudaStreamT) -> CudaErrorT {
    rt_try(|| {
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        map_cu_result(cuEventRecord(event as CUevent, stream as CUstream))
    })
}

#[no_mangle]
pub extern "C" fn cudaEventSynchronize(event: CudaEventT) -> CudaErrorT {
    rt_try(|| {
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        map_cu_result(cuEventSynchronize(event as CUevent))
    })
}

#[no_mangle]
pub extern "C" fn cudaEventQuery(event: CudaEventT) -> CudaErrorT {
    rt_try(|| {
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        map_cu_result(cuEventQuery(event as CUevent))
    })
}

#[no_mangle]
pub extern "C" fn cudaEventElapsedTime(
    ms: *mut f32,
    start: CudaEventT,
    end: CudaEventT,
) -> CudaErrorT {
    rt_try(|| {
        if let Err(e) = ensure_runtime_ctx() {
            return e;
        }
        map_cu_result(cuEventElapsedTime(ms, start as CUevent, end as CUevent))
    })
}

#[no_mangle]
pub extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    numBlocks: *mut c_int,
    _func: *const c_void,
    blockSize: c_int,
    dynamicSMemSize: usize,
) -> CudaErrorT {
    rt_try(|| {
        if numBlocks.is_null() || blockSize <= 0 || blockSize > 1024 {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        let ctx = match ensure_runtime_ctx_arc() {
            Ok(c) => c,
            Err(e) => return e,
        };
        let info = ctx.device_info();
        let lds_per_cu = info.features.lds_bytes_per_cu as usize;
        if dynamicSMemSize > lds_per_cu {
            unsafe {
                *numBlocks = 0;
            }
            return CudaErrorT::CudaSuccess;
        }
        let max_threads_per_cu = 2048usize;
        let by_threads = (max_threads_per_cu / (blockSize as usize)).max(1);
        let by_lds = if dynamicSMemSize > 0 {
            lds_per_cu / dynamicSMemSize
        } else {
            32
        };
        let active = by_threads.min(by_lds).min(32) as c_int;
        unsafe {
            *numBlocks = active;
        }
        CudaErrorT::CudaSuccess
    })
}

#[no_mangle]
pub extern "C" fn cudaOccupancyMaxPotentialBlockSize(
    minGridSize: *mut c_int,
    blockSize: *mut c_int,
    _func: *const c_void,
    dynamicSMemSize: usize,
    blockSizeLimit: c_int,
) -> CudaErrorT {
    rt_try(|| {
        if minGridSize.is_null() || blockSize.is_null() {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        let ctx = match ensure_runtime_ctx_arc() {
            Ok(c) => c,
            Err(e) => return e,
        };
        let info = ctx.device_info();
        let wave = info.isa.native_wave_size() as c_int;
        let cap = if blockSizeLimit > 0 {
            blockSizeLimit.min(1024)
        } else {
            1024
        };
        let chosen_block = if cap >= 256 {
            256
        } else {
            (cap / wave).max(1) * wave
        };
        let lds_per_cu = info.features.lds_bytes_per_cu as usize;
        let by_threads = (2048usize / (chosen_block as usize)).max(1);
        let by_lds = if dynamicSMemSize > 0 {
            lds_per_cu / dynamicSMemSize
        } else {
            32
        };
        let blocks_per_cu = by_threads.min(by_lds).clamp(1, 32);
        let grid = (info.features.compute_units as usize).saturating_mul(blocks_per_cu) as c_int;
        unsafe {
            *blockSize = chosen_block;
            *minGridSize = grid;
        }
        CudaErrorT::CudaSuccess
    })
}

#[no_mangle]
pub extern "C" fn cudaGetLastError() -> CudaErrorT {
    let err = LAST_ERROR.get();
    LAST_ERROR.set(CudaErrorT::CudaSuccess);
    err
}

#[no_mangle]
pub extern "C" fn cudaPeekAtLastError() -> CudaErrorT {
    LAST_ERROR.get()
}

#[no_mangle]
pub extern "C" fn cudaGetErrorString(error: CudaErrorT) -> *const c_char {
    let msg: &'static [u8] = match error {
        CudaErrorT::CudaSuccess => b"no error\0",
        CudaErrorT::CudaErrorInvalidValue => b"invalid argument\0",
        CudaErrorT::CudaErrorMemoryAllocation => b"out of memory\0",
        CudaErrorT::CudaErrorInitializationError => b"initialization error\0",
        CudaErrorT::CudaErrorInvalidDeviceFunction => b"invalid device function\0",
        CudaErrorT::CudaErrorInvalidDevice => b"invalid device ordinal\0",
        CudaErrorT::CudaErrorNotReady => b"device not ready\0",
        CudaErrorT::CudaErrorLaunchFailure => b"unspecified launch failure\0",
        CudaErrorT::CudaErrorUnknown => b"unknown error\0",
    };
    msg.as_ptr() as *const c_char
}

// ============================================================================
// nvcc Host Startup Fatbin Registration & Call Configuration ABI
// ============================================================================

/// Registers an embedded CUDA fatbin wrapper, raw fatbin, ELF, or PTX image without initializing the GPU.
#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinary(fatCubin: *mut c_void) -> *mut *mut c_void {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        if fatCubin.is_null() {
            return core::ptr::null_mut();
        }
        let ptr = fatCubin as *const u8;
        let mut first4 = [0u8; 4];
        let mut reached_nul = false;
        for i in 0..4 {
            let b = *ptr.add(i);
            first4[i] = b;
            if b == 0 {
                if i == 0 {
                    return core::ptr::null_mut();
                }
                if first4[0] != 0x50 && first4[0] != 0x7F {
                    reached_nul = true;
                    break;
                }
            }
        }

        let image_bytes = if !reached_nul && u32::from_le_bytes(first4) == FATBIN_WRAPPER_MAGIC {
            let wrapper = &*(fatCubin as *const FatBinCWrapper);
            if wrapper.data.is_null() {
                return core::ptr::null_mut();
            }
            match read_module_image_bytes(wrapper.data) {
                Ok(b) => b,
                Err(_) => return core::ptr::null_mut(),
            }
        } else {
            match read_module_image_bytes(fatCubin as *const c_void) {
                Ok(b) => b,
                Err(_) => return core::ptr::null_mut(),
            }
        };

        if image_bytes.is_empty() {
            return core::ptr::null_mut();
        }

        let mut reg = fatbin_registry().lock().expect("fatbin registry lock");
        let id = reg.alloc_handle();
        let len = image_bytes.len();
        reg.fatbins.insert(id, image_bytes.to_vec());
        trace_log(
            "cuda-runtime",
            &format!("__cudaRegisterFatBinary handle=0x{id:x} bytes={len}"),
        );
        id as *mut *mut c_void
    }))
    .unwrap_or(core::ptr::null_mut())
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinaryEnd(_fatCubinHandle: *mut *mut c_void) {}

#[no_mangle]
pub extern "C" fn __cudaUnregisterFatBinary(fatCubinHandle: *mut *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if fatCubinHandle.is_null() {
            return;
        }
        let fatbin_id = fatCubinHandle as usize;
        let to_unload: Vec<CUmodule> = {
            let mut reg = fatbin_registry().lock().expect("fatbin registry lock");
            reg.fatbins.remove(&fatbin_id);
            reg.functions.retain(|_, (fid, _)| *fid != fatbin_id);
            reg.vars.retain(|_, (fid, _, _)| *fid != fatbin_id);
            reg.cached_funcs.retain(|(fid, _, _), _| *fid != fatbin_id);

            let keys: Vec<(usize, usize)> = reg
                .loaded_modules
                .keys()
                .filter(|(fid, _)| *fid == fatbin_id)
                .copied()
                .collect();
            let mut mods = Vec::with_capacity(keys.len());
            for k in keys {
                if let Some(m) = reg.loaded_modules.remove(&k) {
                    mods.push(m as CUmodule);
                }
            }
            mods
        };

        for hmod in to_unload {
            let _ = cuModuleUnload(hmod);
        }
    }));
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFunction(
    fatCubinHandle: *mut *mut c_void,
    hostFun: *const c_char,
    deviceFun: *mut c_char,
    deviceName: *const c_char,
    _thread_limit: c_int,
    _tid: *mut c_void,
    _bid: *mut c_void,
    _bDim: *mut c_void,
    _gDim: *mut c_void,
    _wSize: *mut c_int,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        if fatCubinHandle.is_null() || hostFun.is_null() {
            return;
        }
        let fatbin_id = fatCubinHandle as usize;
        let name_from_dev = if !deviceName.is_null() {
            CStr::from_ptr(deviceName)
                .to_str()
                .ok()
                .filter(|s| !s.is_empty())
        } else {
            None
        };
        let kernel_name = if let Some(n) = name_from_dev {
            n.to_owned()
        } else if !deviceFun.is_null() {
            match CStr::from_ptr(deviceFun).to_str() {
                Ok(s) if !s.is_empty() => s.to_owned(),
                _ => return,
            }
        } else {
            return;
        };

        let mut reg = fatbin_registry().lock().expect("fatbin registry lock");
        if reg.fatbins.contains_key(&fatbin_id) {
            reg.functions
                .insert(hostFun as usize, (fatbin_id, kernel_name));
        }
    }));
}

#[no_mangle]
pub extern "C" fn __cudaRegisterVar(
    fatCubinHandle: *mut *mut c_void,
    hostVar: *mut c_char,
    _deviceAddress: *mut c_char,
    deviceName: *const c_char,
    _ext: c_int,
    size: usize,
    _constant: c_int,
    _global: c_int,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        if fatCubinHandle.is_null() || hostVar.is_null() || deviceName.is_null() {
            return;
        }
        let fatbin_id = fatCubinHandle as usize;
        let Ok(name) = CStr::from_ptr(deviceName).to_str() else {
            return;
        };
        let mut reg = fatbin_registry().lock().expect("fatbin registry lock");
        if reg.fatbins.contains_key(&fatbin_id) {
            reg.vars
                .insert(hostVar as usize, (fatbin_id, name.to_owned(), size));
        }
    }));
}

#[no_mangle]
pub extern "C" fn __cudaPushCallConfiguration(
    gridDim: Dim3,
    blockDim: Dim3,
    sharedMem: usize,
    stream: CudaStreamT,
) -> c_uint {
    catch_unwind(AssertUnwindSafe(|| {
        if gridDim.x == 0
            || gridDim.y == 0
            || gridDim.z == 0
            || blockDim.x == 0
            || blockDim.y == 0
            || blockDim.z == 0
        {
            return 1;
        }
        CALL_CONFIG_STACK.with(|stack| {
            stack.borrow_mut().push(CallConfig {
                grid_dim: gridDim,
                block_dim: blockDim,
                shared_mem: sharedMem,
                stream,
            });
        });
        0
    }))
    .unwrap_or(1)
}

#[no_mangle]
pub extern "C" fn __cudaPopCallConfiguration(
    gridDim: *mut Dim3,
    blockDim: *mut Dim3,
    sharedMem: *mut usize,
    stream: *mut CudaStreamT,
) -> CudaErrorT {
    rt_try(|| {
        if gridDim.is_null() || blockDim.is_null() || sharedMem.is_null() || stream.is_null() {
            return CudaErrorT::CudaErrorInvalidValue;
        }
        let cfg = CALL_CONFIG_STACK.with(|stack| stack.borrow_mut().pop());
        let Some(cfg) = cfg else {
            return CudaErrorT::CudaErrorInvalidValue;
        };
        unsafe {
            *gridDim = cfg.grid_dim;
            *blockDim = cfg.block_dim;
            *sharedMem = cfg.shared_mem;
            *stream = cfg.stream;
        }
        CudaErrorT::CudaSuccess
    })
}

#[no_mangle]
pub extern "C" fn cudaLaunchKernel(
    func: *const c_void,
    gridDim: Dim3,
    blockDim: Dim3,
    args: *mut *mut c_void,
    sharedMem: usize,
    stream: CudaStreamT,
) -> CudaErrorT {
    rt_try(|| {
        if func.is_null() {
            return CudaErrorT::CudaErrorInvalidDeviceFunction;
        }
        let ctx = match ensure_runtime_ctx_arc() {
            Ok(c) => c,
            Err(e) => return e,
        };
        if let Err(e) = validate_stream(stream as CUstream) {
            return map_cu_result(e);
        }
        let dev = ctx.device_index();

        // 1. Fast path: resolve cached CUfunction or registered (fatbin_id, kernel_name)
        let (fatbin_id, kernel_name, cached_fn, existing_mod, image_bytes) = {
            let reg = fatbin_registry().lock().expect("fatbin registry lock");
            let Some((fid, kname)) = reg.functions.get(&(func as usize)).cloned() else {
                return CudaErrorT::CudaErrorInvalidDeviceFunction;
            };
            if let Some(&hfunc) = reg.cached_funcs.get(&(fid, dev, kname.clone())) {
                (fid, kname, Some(hfunc as CUfunction), None, None)
            } else if let Some(&hmod) = reg.loaded_modules.get(&(fid, dev)) {
                (fid, kname, None, Some(hmod as CUmodule), None)
            } else {
                let bytes = reg.fatbins.get(&fid).cloned();
                (fid, kname, None, None, bytes)
            }
        };

        let hfunc = if let Some(f) = cached_fn {
            f
        } else {
            // 2. Ensure CUmodule is loaded for `(fatbin_id, dev)` without holding FATBIN_REGISTRY lock
            let hmod = if let Some(m) = existing_mod {
                m
            } else {
                let Some(bytes) = image_bytes else {
                    return CudaErrorT::CudaErrorInvalidDeviceFunction;
                };
                let loaded_mod = match load_module_from_slice(&bytes) {
                    Ok(m) => m,
                    Err(e) => return map_cu_result(e),
                };
                let mut duplicate_mod: Option<CUmodule> = None;
                let chosen = {
                    let mut reg = fatbin_registry().lock().expect("fatbin registry lock");
                    if !reg.fatbins.contains_key(&fatbin_id) {
                        duplicate_mod = Some(loaded_mod);
                        None
                    } else if let Some(&already) = reg.loaded_modules.get(&(fatbin_id, dev)) {
                        duplicate_mod = Some(loaded_mod);
                        Some(already as CUmodule)
                    } else {
                        reg.loaded_modules
                            .insert((fatbin_id, dev), loaded_mod as usize);
                        Some(loaded_mod)
                    }
                };
                if let Some(dup) = duplicate_mod {
                    let _ = cuModuleUnload(dup);
                }
                match chosen {
                    Some(m) => m,
                    None => return CudaErrorT::CudaErrorInvalidDeviceFunction,
                }
            };

            // 3. Resolve CUfunction without holding FATBIN_REGISTRY lock
            let Ok(c_kname) = CString::new(kernel_name.as_str()) else {
                return CudaErrorT::CudaErrorInvalidDeviceFunction;
            };
            let mut resolved_fn: CUfunction = core::ptr::null_mut();
            let rc = cuModuleGetFunction(&mut resolved_fn, hmod, c_kname.as_ptr());
            if rc != CUresult::CUDA_SUCCESS {
                return if rc == CUresult::CUDA_ERROR_NOT_FOUND {
                    CudaErrorT::CudaErrorInvalidDeviceFunction
                } else {
                    map_cu_result(rc)
                };
            }
            let mut reg = fatbin_registry().lock().expect("fatbin registry lock");
            if !reg.fatbins.contains_key(&fatbin_id) {
                return CudaErrorT::CudaErrorInvalidDeviceFunction;
            }
            reg.cached_funcs
                .insert((fatbin_id, dev, kernel_name), resolved_fn as usize);
            resolved_fn
        };

        trace_log(
            "cuda-runtime",
            &format!(
                "cudaLaunchKernel func={:?} grid=({},{},{}) block=({},{},{}) shared={sharedMem}",
                func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z
            ),
        );
        let shared_bytes = u32::try_from(sharedMem).unwrap_or(u32::MAX);
        let rc = cuLaunchKernel(
            hfunc,
            gridDim.x,
            gridDim.y,
            gridDim.z,
            blockDim.x,
            blockDim.y,
            blockDim.z,
            shared_bytes,
            stream as CUstream,
            args,
            core::ptr::null_mut(),
        );
        map_cu_result(rc)
    })
}
