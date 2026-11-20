//! CUDA Driver API (`cu*`) C ABI exports and global handle state.

#![allow(non_snake_case, clippy::not_unsafe_ptr_arg_deref)]

use crate::types::{
    CUcontext, CUdevice, CUdeviceptr, CUevent, CUfunction, CUgraph, CUgraphExec, CUgraphNode,
    CUmodule, CUresult, CUstream, CudaKernelNodeParams, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
    CU_DEVICE_ATTRIBUTE_INTEGRATED, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
    CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST, CU_POINTER_ATTRIBUTE_CONTEXT,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
    CU_POINTER_ATTRIBUTE_HOST_POINTER, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
};
use amda_compiler::fatbin::FATBIN_MAGIC;
use amda_elf::{IsaTarget, KernelMetadata};
use amda_hal::{DeviceInfo, HalError};
use amda_runtime::{
    pack_kernargs, trace_log, Context, Event, ExecutableGraph, ExecutionGraph, KernelFunction,
    LoadedModule, Runtime, RuntimeError, Stream,
};
use core::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::cell::RefCell;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};

const CUDA_DRIVER_VERSION: c_int = 12040;
const MEMSET_CHUNK_BYTES: usize = 1024 * 1024;

struct StoredFunction {
    ctx_id: usize,
    func: KernelFunction,
}

struct StoredModule {
    ctx_id: usize,
    module: LoadedModule,
    func_handles: Vec<usize>,
}

struct StoredStream {
    ctx_id: usize,
    stream: Stream,
}

struct StoredEvent {
    ctx_id: usize,
    event: Event,
}

struct StoredGraph {
    ctx_id: usize,
    graph: ExecutionGraph,
}

struct StoredGraphExec {
    ctx_id: usize,
    exec: ExecutableGraph,
}

struct GlobalState {
    runtime: Option<Arc<Runtime>>,
    next_handle: usize,
    contexts: HashMap<usize, Arc<Context>>,
    modules: HashMap<usize, StoredModule>,
    functions: HashMap<usize, StoredFunction>,
    streams: HashMap<usize, StoredStream>,
    events: HashMap<usize, StoredEvent>,
    graphs: HashMap<usize, StoredGraph>,
    graph_execs: HashMap<usize, StoredGraphExec>,
    primary_contexts: HashMap<usize, usize>,
    primary_ctx_refcounts: HashMap<usize, usize>,
}

impl GlobalState {
    fn new() -> Self {
        Self {
            runtime: None,
            next_handle: 0x1001,
            contexts: HashMap::new(),
            modules: HashMap::new(),
            functions: HashMap::new(),
            streams: HashMap::new(),
            events: HashMap::new(),
            graphs: HashMap::new(),
            graph_execs: HashMap::new(),
            primary_contexts: HashMap::new(),
            primary_ctx_refcounts: HashMap::new(),
        }
    }

    fn alloc_handle(&mut self) -> usize {
        let id = self.next_handle;
        self.next_handle = self.next_handle.wrapping_add(1).max(0x1001);
        id
    }
}

static GLOBAL: OnceLock<Mutex<GlobalState>> = OnceLock::new();

thread_local! {
    static CTX_STACK: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

fn current_ctx_id() -> Option<usize> {
    CTX_STACK.with(|st| st.borrow().last().copied())
}

fn push_ctx_id(id: usize) {
    CTX_STACK.with(|st| st.borrow_mut().push(id));
}

fn pop_ctx_id() -> Option<usize> {
    CTX_STACK.with(|st| st.borrow_mut().pop())
}

fn set_current_ctx_id(id: Option<usize>) {
    CTX_STACK.with(|st| {
        let mut v = st.borrow_mut();
        match id {
            Some(ctx) => {
                if let Some(last) = v.last_mut() {
                    *last = ctx;
                } else {
                    v.push(ctx);
                }
            }
            None => {
                v.pop();
            }
        }
    });
}

fn remove_ctx_from_stack(id: usize) {
    CTX_STACK.with(|st| st.borrow_mut().retain(|&x| x != id));
}

fn global() -> &'static Mutex<GlobalState> {
    GLOBAL.get_or_init(|| Mutex::new(GlobalState::new()))
}

pub(crate) fn ffi_try<F>(f: F) -> CUresult
where
    F: FnOnce() -> Result<(), CUresult>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => CUresult::CUDA_SUCCESS,
        Ok(Err(e)) => e,
        Err(_) => CUresult::CUDA_ERROR_UNKNOWN,
    }
}

pub(crate) fn map_runtime_err(err: RuntimeError) -> CUresult {
    match err {
        RuntimeError::Hal(HalError::DeviceNotFound(_)) => CUresult::CUDA_ERROR_INVALID_VALUE,
        RuntimeError::Hal(HalError::OutOfMemory(_)) => CUresult::CUDA_ERROR_OUT_OF_MEMORY,
        RuntimeError::Hal(HalError::InvalidHandle(_)) => CUresult::CUDA_ERROR_INVALID_HANDLE,
        RuntimeError::Hal(HalError::KernelNotFound(_)) => CUresult::CUDA_ERROR_NOT_FOUND,
        RuntimeError::Hal(HalError::InvalidCodeObject(_)) => CUresult::CUDA_ERROR_INVALID_IMAGE,
        RuntimeError::Hal(HalError::ExecutionError(_))
        | RuntimeError::Hal(HalError::MemoryAccessFault { .. }) => CUresult::CUDA_ERROR_LAUNCH_FAILED,
        RuntimeError::Compile(_) | RuntimeError::Elf(_) => CUresult::CUDA_ERROR_INVALID_IMAGE,
        RuntimeError::InvalidArgument(_) => CUresult::CUDA_ERROR_INVALID_VALUE,
        RuntimeError::InvalidHandle(_) => CUresult::CUDA_ERROR_INVALID_HANDLE,
        RuntimeError::KernelNotFound(_) => CUresult::CUDA_ERROR_NOT_FOUND,
        RuntimeError::NotReady => CUresult::CUDA_ERROR_NOT_READY,
        RuntimeError::Hal(_) => CUresult::CUDA_ERROR_UNKNOWN,
    }
}

fn get_runtime() -> Result<Arc<Runtime>, CUresult> {
    let g = global().lock().expect("global lock");
    g.runtime
        .clone()
        .ok_or(CUresult::CUDA_ERROR_NOT_INITIALIZED)
}

pub(crate) fn ensure_initialized_and_context(device: usize) -> Result<Arc<Context>, CUresult> {
    {
        let mut g = global().lock().expect("global lock");
        if g.runtime.is_none() {
            let rt = Arc::new(Runtime::new().map_err(map_runtime_err)?);
            g.runtime = Some(rt);
        }
    }

    if let Some(ctx_id) = current_ctx_id() {
        let g = global().lock().expect("global lock");
        if let Some(ctx) = g.contexts.get(&ctx_id) {
            if ctx.device_index() == device {
                return Ok(Arc::clone(ctx));
            }
        }
    }

    let mut g = global().lock().expect("global lock");
    if let Some(&ctx_id) = g.primary_contexts.get(&device) {
        if let Some(ctx) = g.contexts.get(&ctx_id).cloned() {
            set_current_ctx_id(Some(ctx_id));
            return Ok(ctx);
        }
    }
    let rt = g
        .runtime
        .clone()
        .ok_or(CUresult::CUDA_ERROR_NOT_INITIALIZED)?;
    let ctx = rt.create_context(device).map_err(map_runtime_err)?;
    let ctx_id = g.alloc_handle();
    g.contexts.insert(ctx_id, Arc::clone(&ctx));
    g.primary_contexts.insert(device, ctx_id);
    g.primary_ctx_refcounts.entry(device).or_insert(1);
    set_current_ctx_id(Some(ctx_id));
    Ok(ctx)
}

pub(crate) fn current_context() -> Result<(usize, Arc<Context>), CUresult> {
    let ctx_id = current_ctx_id().ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)?;
    let g = global().lock().expect("global lock");
    let ctx = g
        .contexts
        .get(&ctx_id)
        .cloned()
        .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)?;
    Ok((ctx_id, ctx))
}

pub(crate) fn get_device_info(dev: usize) -> Result<DeviceInfo, CUresult> {
    let rt = get_runtime()?;
    rt.devices()
        .get(dev)
        .cloned()
        .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)
}

pub(crate) fn is_device_ptr(ptr: u64) -> Option<(usize, usize)> {
    if ptr == 0 {
        return None;
    }
    let g = GLOBAL.get()?.lock().ok()?;
    for (&ctx_id, ctx) in &g.contexts {
        if ctx.contains_device_ptr(ptr).is_some() {
            return Some((ctx_id, ctx.device_index()));
        }
    }
    None
}

pub(crate) fn reset_device_contexts(dev: usize) -> Result<(), CUresult> {
    let mod_ids: Vec<CUmodule> = {
        let g = global().lock().expect("global lock");
        g.modules
            .iter()
            .filter_map(|(&id, m)| {
                g.contexts
                    .get(&m.ctx_id)
                    .filter(|ctx| ctx.device_index() == dev)
                    .map(|_| id as CUmodule)
            })
            .collect()
    };
    for hmod in mod_ids {
        let rc = cuModuleUnload(hmod);
        if rc != CUresult::CUDA_SUCCESS {
            return Err(rc);
        }
    }
    Ok(())
}

pub(crate) fn validate_stream(hStream: CUstream) -> Result<(), CUresult> {
    if hStream.is_null() {
        return Ok(());
    }
    let g = global().lock().expect("global lock");
    if g.streams.contains_key(&(hStream as usize)) {
        Ok(())
    } else {
        Err(CUresult::CUDA_ERROR_INVALID_HANDLE)
    }
}

pub(crate) unsafe fn read_module_image_bytes<'a>(image: *const c_void) -> Result<&'a [u8], CUresult> {
    if image.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let ptr = image as *const u8;
    let mut first4 = [0u8; 4];
    for i in 0..4 {
        let b = *ptr.add(i);
        first4[i] = b;
        if b == 0 && i == 0 {
            return Ok(&[]);
        }
        // Only fatbin (starts with 0x50) or ELF (starts with 0x7F) can safely contain NULs in bytes 1..3.
        if b == 0 && first4[0] != 0x50 && first4[0] != 0x7F {
            let cstr = CStr::from_ptr(image as *const c_char);
            return Ok(cstr.to_bytes());
        }
    }

    // 1. Fatbin container (`0xBA55ED50`)
    if u32::from_le_bytes(first4) == FATBIN_MAGIC {
        let head16 = std::slice::from_raw_parts(ptr, 16);
        let hdr_size = usize::from(u16::from_le_bytes([head16[6], head16[7]]));
        let fat_size = u64::from_le_bytes([
            head16[8], head16[9], head16[10], head16[11], head16[12], head16[13], head16[14],
            head16[15],
        ]) as usize;
        let total = hdr_size
            .checked_add(fat_size)
            .ok_or(CUresult::CUDA_ERROR_INVALID_IMAGE)?;
        return Ok(std::slice::from_raw_parts(ptr, total));
    }

    // 2. ELF64 HSACO image (`\x7FELF`)
    if first4 == [0x7F, b'E', b'L', b'F'] {
        let head64 = std::slice::from_raw_parts(ptr, 64);
        let e_shoff = u64::from_le_bytes([
            head64[40], head64[41], head64[42], head64[43], head64[44], head64[45], head64[46],
            head64[47],
        ]) as usize;
        let e_shentsize = usize::from(u16::from_le_bytes([head64[58], head64[59]]));
        let e_shnum = usize::from(u16::from_le_bytes([head64[60], head64[61]]));
        let total = e_shoff
            .checked_add(e_shentsize.saturating_mul(e_shnum))
            .ok_or(CUresult::CUDA_ERROR_INVALID_IMAGE)?;
        if total < 64 {
            return Err(CUresult::CUDA_ERROR_INVALID_IMAGE);
        }
        return Ok(std::slice::from_raw_parts(ptr, total));
    }

    // 3. NUL-terminated UTF-8 PTX text
    let cstr = CStr::from_ptr(image as *const c_char);
    Ok(cstr.to_bytes())
}

unsafe fn extract_kernel_param_bytes(
    meta: &KernelMetadata,
    kernel_params: *mut *mut c_void,
) -> Result<Vec<u8>, CUresult> {
    if meta.args.is_empty() {
        return Ok(Vec::new());
    }
    if kernel_params.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let mut temp_storage: Vec<Vec<u8>> = Vec::with_capacity(meta.args.len());
    for (idx, arg_meta) in meta.args.iter().enumerate() {
        let ptr = *kernel_params.add(idx);
        if ptr.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let bytes = std::slice::from_raw_parts(ptr as *const u8, arg_meta.size as usize);
        temp_storage.push(bytes.to_vec());
    }
    let refs: Vec<&[u8]> = temp_storage.iter().map(|v| v.as_slice()).collect();
    pack_kernargs(meta, &refs).map_err(map_runtime_err)
}

unsafe fn extract_kernel_param_slices(
    meta: &KernelMetadata,
    kernel_params: *mut *mut c_void,
) -> Result<Vec<Vec<u8>>, CUresult> {
    if meta.args.is_empty() {
        return Ok(Vec::new());
    }
    if kernel_params.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let mut out = Vec::with_capacity(meta.args.len());
    for (idx, arg_meta) in meta.args.iter().enumerate() {
        let ptr = *kernel_params.add(idx);
        if ptr.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let bytes = std::slice::from_raw_parts(ptr as *const u8, arg_meta.size as usize);
        out.push(bytes.to_vec());
    }
    Ok(out)
}

#[no_mangle]
pub extern "C" fn cuInit(flags: c_uint) -> CUresult {
    ffi_try(|| {
        if flags != 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let mut g = global().lock().expect("global lock");
        if g.runtime.is_none() {
            let rt = Arc::new(Runtime::new().map_err(map_runtime_err)?);
            g.runtime = Some(rt);
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuDriverGetVersion(driverVersion: *mut c_int) -> CUresult {
    ffi_try(|| {
        if driverVersion.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        unsafe {
            *driverVersion = CUDA_DRIVER_VERSION;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuDeviceGetCount(count: *mut c_int) -> CUresult {
    ffi_try(|| {
        if count.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let rt = get_runtime()?;
        unsafe {
            *count = rt.devices().len() as c_int;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult {
    ffi_try(|| {
        if device.is_null() || ordinal < 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let rt = get_runtime()?;
        if (ordinal as usize) >= rt.devices().len() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        unsafe {
            *device = ordinal;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    ffi_try(|| {
        if name.is_null() || len <= 0 || dev < 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let rt = get_runtime()?;
        let info = rt
            .devices()
            .get(dev as usize)
            .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
        let bytes = info.name.as_bytes();
        let copy_len = bytes.len().min((len as usize) - 1);
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), name as *mut u8, copy_len);
            *name.add(copy_len) = 0;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult {
    ffi_try(|| {
        if bytes.is_null() || dev < 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let rt = get_runtime()?;
        let info = rt
            .devices()
            .get(dev as usize)
            .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
        unsafe {
            *bytes = info.vram_bytes as usize;
        }
        Ok(())
    })
}

pub(crate) fn isa_hw_metrics(isa: IsaTarget) -> (c_int, c_int, c_int, c_int) {
    // Returns (core_clock_khz, mem_clock_khz, bus_width_bits, l2_cache_bytes).
    match isa {
        IsaTarget::Gfx942 => (2_100_000, 1_300_000, 8192, 256 * 1024 * 1024),
        IsaTarget::Gfx90a | IsaTarget::Gfx908 => (1_700_000, 1_600_000, 4096, 8 * 1024 * 1024),
        IsaTarget::Gfx1201 | IsaTarget::Gfx1200 => (2_970_000, 2_500_000, 256, 64 * 1024 * 1024),
        IsaTarget::Gfx1100 => (2_500_000, 2_500_000, 384, 96 * 1024 * 1024),
        IsaTarget::Gfx1101 => (2_430_000, 2_438_000, 256, 64 * 1024 * 1024),
        IsaTarget::Gfx1102 => (2_655_000, 2_250_000, 128, 32 * 1024 * 1024),
        IsaTarget::Gfx1103 => (2_700_000, 2_800_000, 128, 4 * 1024 * 1024),
        IsaTarget::Gfx1030 => (2_250_000, 2_000_000, 256, 128 * 1024 * 1024),
    }
}

#[no_mangle]
pub extern "C" fn cuDeviceGetAttribute(
    pi: *mut c_int,
    attrib: c_int,
    dev: CUdevice,
) -> CUresult {
    ffi_try(|| {
        if pi.is_null() || dev < 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let rt = get_runtime()?;
        let info = rt
            .devices()
            .get(dev as usize)
            .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
        let (core_khz, mem_khz, bus_bits, l2_bytes) = isa_hw_metrics(info.isa);
        let val = match attrib {
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK => 1024,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X => 1024,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y => 1024,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z => 64,
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X => 2_147_483_647,
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y => 65_535,
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z => 65_535,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK => {
                info.features.lds_bytes_per_cu as c_int
            }
            CU_DEVICE_ATTRIBUTE_WARP_SIZE => info.isa.native_wave_size() as c_int,
            CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK => 65_536,
            CU_DEVICE_ATTRIBUTE_CLOCK_RATE => core_khz,
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT => info.features.compute_units as c_int,
            CU_DEVICE_ATTRIBUTE_INTEGRATED => c_int::from(info.features.unified_memory),
            CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY => 1,
            CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS => 1,
            CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE => mem_khz,
            CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH => bus_bits,
            CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE => l2_bytes,
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR => 2048,
            CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT => 2,
            CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING => 1,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => 8,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => 0,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR => {
                info.features.lds_bytes_per_cu as c_int
            }
            CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY => 1,
            _ => return Err(CUresult::CUDA_ERROR_INVALID_VALUE),
        };
        unsafe {
            *pi = val;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuCtxCreate_v2(
    pctx: *mut CUcontext,
    _flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    ffi_try(|| {
        if pctx.is_null() || dev < 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let rt = get_runtime()?;
        let ctx = rt.create_context(dev as usize).map_err(map_runtime_err)?;
        let mut g = global().lock().expect("global lock");
        let id = g.alloc_handle();
        g.contexts.insert(id, ctx);
        push_ctx_id(id);
        unsafe {
            *pctx = id as CUcontext;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult {
    ffi_try(|| {
        if ctx.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let id = ctx as usize;
        let mut g = global().lock().expect("global lock");
        if g.contexts.remove(&id).is_none() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let primary_devs: Vec<usize> = g
            .primary_contexts
            .iter()
            .filter_map(|(&dev, &cid)| (cid == id).then_some(dev))
            .collect();
        for dev in primary_devs {
            g.primary_contexts.remove(&dev);
            g.primary_ctx_refcounts.remove(&dev);
        }
        remove_ctx_from_stack(id);
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult {
    ffi_try(|| {
        if ctx.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_CONTEXT);
        }
        let id = ctx as usize;
        let g = global().lock().expect("global lock");
        if !g.contexts.contains_key(&id) {
            return Err(CUresult::CUDA_ERROR_INVALID_CONTEXT);
        }
        push_ctx_id(id);
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult {
    ffi_try(|| {
        if !pctx.is_null() {
            unsafe {
                *pctx = core::ptr::null_mut();
            }
        }
        let Some(id) = pop_ctx_id() else {
            return Err(CUresult::CUDA_ERROR_INVALID_CONTEXT);
        };
        if !pctx.is_null() {
            unsafe {
                *pctx = id as CUcontext;
            }
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult {
    ffi_try(|| {
        if ctx.is_null() {
            set_current_ctx_id(None);
            return Ok(());
        }
        let id = ctx as usize;
        let g = global().lock().expect("global lock");
        if !g.contexts.contains_key(&id) {
            return Err(CUresult::CUDA_ERROR_INVALID_CONTEXT);
        }
        set_current_ctx_id(Some(id));
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult {
    ffi_try(|| {
        if pctx.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        unsafe {
            *pctx = current_ctx_id().unwrap_or(0) as CUcontext;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    ffi_try(|| {
        if pctx.is_null() || dev < 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let dev_idx = dev as usize;
        let rt = get_runtime()?;
        if dev_idx >= rt.devices().len() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let mut g = global().lock().expect("global lock");
        let ctx_id = if let Some(&existing) = g.primary_contexts.get(&dev_idx) {
            if g.contexts.contains_key(&existing) {
                let rc = g.primary_ctx_refcounts.entry(dev_idx).or_insert(0);
                *rc += 1;
                existing
            } else {
                let ctx = rt.create_context(dev_idx).map_err(map_runtime_err)?;
                let id = g.alloc_handle();
                g.contexts.insert(id, ctx);
                g.primary_contexts.insert(dev_idx, id);
                g.primary_ctx_refcounts.insert(dev_idx, 1);
                id
            }
        } else {
            let ctx = rt.create_context(dev_idx).map_err(map_runtime_err)?;
            let id = g.alloc_handle();
            g.contexts.insert(id, ctx);
            g.primary_contexts.insert(dev_idx, id);
            g.primary_ctx_refcounts.insert(dev_idx, 1);
            id
        };
        unsafe {
            *pctx = ctx_id as CUcontext;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult {
    ffi_try(|| {
        if dev < 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let dev_idx = dev as usize;
        let rt = get_runtime()?;
        if dev_idx >= rt.devices().len() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let mut g = global().lock().expect("global lock");
        let Some(rc) = g.primary_ctx_refcounts.get_mut(&dev_idx) else {
            return Ok(());
        };
        *rc = rc.saturating_sub(1);
        if *rc == 0 {
            g.primary_ctx_refcounts.remove(&dev_idx);
            if let Some(ctx_id) = g.primary_contexts.remove(&dev_idx) {
                g.contexts.remove(&ctx_id);
                remove_ctx_from_stack(ctx_id);
            }
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuCtxSynchronize() -> CUresult {
    ffi_try(|| {
        let (_, ctx) = current_context()?;
        ctx.synchronize().map_err(map_runtime_err)
    })
}

#[no_mangle]
pub extern "C" fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult {
    ffi_try(|| {
        if free.is_null() || total.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let (_, ctx) = current_context()?;
        let total_bytes = ctx.device_info().vram_bytes as usize;
        let stats = ctx.memory_stats();
        let used = stats.active_bytes.saturating_add(stats.pooled_bytes);
        unsafe {
            *total = total_bytes;
            *free = total_bytes.saturating_sub(used);
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    ffi_try(|| {
        if dptr.is_null() || bytesize == 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let (_, ctx) = current_context()?;
        let ptr = ctx.alloc(bytesize).map_err(map_runtime_err)?;
        unsafe {
            *dptr = ptr;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult {
    ffi_try(|| {
        if dptr == 0 {
            return Ok(());
        }
        let (_, ctx) = current_context()?;
        ctx.free(dptr).map_err(map_runtime_err)
    })
}

#[no_mangle]
pub extern "C" fn cuMemsetD8_v2(dstDevice: CUdeviceptr, uc: u8, N: usize) -> CUresult {
    ffi_try(|| {
        if N == 0 {
            return Ok(());
        }
        if dstDevice == 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let (_, ctx) = current_context()?;
        let chunk_len = N.min(MEMSET_CHUNK_BYTES);
        let buf = vec![uc; chunk_len];
        let mut offset = 0usize;
        while offset < N {
            let step = (N - offset).min(MEMSET_CHUNK_BYTES);
            let addr = dstDevice
                .checked_add(offset as u64)
                .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
            ctx.copy_htod(addr, &buf[..step]).map_err(map_runtime_err)?;
            offset += step;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuMemsetD32_v2(dstDevice: CUdeviceptr, ui: c_uint, N: usize) -> CUresult {
    ffi_try(|| {
        if N == 0 {
            return Ok(());
        }
        if dstDevice == 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let total_bytes = N
            .checked_mul(4)
            .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
        let (_, ctx) = current_context()?;
        let word = ui.to_le_bytes();
        let chunk_words = N.clamp(1, MEMSET_CHUNK_BYTES / 4);
        let mut buf = Vec::with_capacity(chunk_words * 4);
        for _ in 0..chunk_words {
            buf.extend_from_slice(&word);
        }
        let mut offset = 0usize;
        while offset < total_bytes {
            let step = (total_bytes - offset).min(buf.len());
            let addr = dstDevice
                .checked_add(offset as u64)
                .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
            ctx.copy_htod(addr, &buf[..step]).map_err(map_runtime_err)?;
            offset += step;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuMemcpyHtoD_v2(
    dstDevice: CUdeviceptr,
    srcHost: *const c_void,
    byteCount: usize,
) -> CUresult {
    ffi_try(|| {
        if byteCount == 0 {
            return Ok(());
        }
        if dstDevice == 0 || srcHost.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let (_, ctx) = current_context()?;
        let src = unsafe { std::slice::from_raw_parts(srcHost as *const u8, byteCount) };
        ctx.copy_htod(dstDevice, src).map_err(map_runtime_err)
    })
}

#[no_mangle]
pub extern "C" fn cuMemcpyDtoH_v2(
    dstHost: *mut c_void,
    srcDevice: CUdeviceptr,
    byteCount: usize,
) -> CUresult {
    ffi_try(|| {
        if byteCount == 0 {
            return Ok(());
        }
        if dstHost.is_null() || srcDevice == 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let (_, ctx) = current_context()?;
        let dst = unsafe { std::slice::from_raw_parts_mut(dstHost as *mut u8, byteCount) };
        ctx.copy_dtoh(dst, srcDevice).map_err(map_runtime_err)
    })
}

#[no_mangle]
pub extern "C" fn cuMemcpyDtoD_v2(
    dstDevice: CUdeviceptr,
    srcDevice: CUdeviceptr,
    byteCount: usize,
) -> CUresult {
    ffi_try(|| {
        if byteCount == 0 {
            return Ok(());
        }
        if dstDevice == 0 || srcDevice == 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let (_, ctx) = current_context()?;
        ctx.copy_dtod(dstDevice, srcDevice, byteCount)
            .map_err(map_runtime_err)
    })
}

fn stream_context_if_valid(hStream: CUstream) -> Result<Arc<Context>, CUresult> {
    if hStream.is_null() {
        return current_context().map(|(_, ctx)| ctx);
    }
    let g = global().lock().expect("global lock");
    let st = g
        .streams
        .get(&(hStream as usize))
        .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
    g.contexts
        .get(&st.ctx_id)
        .cloned()
        .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)
}

#[no_mangle]
pub extern "C" fn cuMemcpyHtoDAsync_v2(
    dstDevice: CUdeviceptr,
    srcHost: *const c_void,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    ffi_try(|| {
        let ctx = stream_context_if_valid(hStream)?;
        if ByteCount == 0 {
            return Ok(());
        }
        if dstDevice == 0 || srcHost.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let src = unsafe { std::slice::from_raw_parts(srcHost as *const u8, ByteCount) };
        ctx.copy_htod(dstDevice, src).map_err(map_runtime_err)
    })
}

#[no_mangle]
pub extern "C" fn cuMemcpyDtoHAsync_v2(
    dstHost: *mut c_void,
    srcDevice: CUdeviceptr,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    ffi_try(|| {
        let ctx = stream_context_if_valid(hStream)?;
        if ByteCount == 0 {
            return Ok(());
        }
        if dstHost.is_null() || srcDevice == 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let dst = unsafe { std::slice::from_raw_parts_mut(dstHost as *mut u8, ByteCount) };
        ctx.copy_dtoh(dst, srcDevice).map_err(map_runtime_err)
    })
}

#[no_mangle]
pub extern "C" fn cuMemcpyDtoDAsync_v2(
    dstDevice: CUdeviceptr,
    srcDevice: CUdeviceptr,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    ffi_try(|| {
        let ctx = stream_context_if_valid(hStream)?;
        if ByteCount == 0 {
            return Ok(());
        }
        if dstDevice == 0 || srcDevice == 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        ctx.copy_dtod(dstDevice, srcDevice, ByteCount)
            .map_err(map_runtime_err)
    })
}

fn pointer_get_attribute_impl(
    data: *mut c_void,
    attribute: c_int,
    ptr: CUdeviceptr,
) -> Result<(), CUresult> {
    if data.is_null() || ptr == 0 {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let dev_match = is_device_ptr(ptr);
    let pinned_host = dev_match.is_none() && crate::runtime::is_pinned_host_ptr(ptr as usize);
    if dev_match.is_none() && !pinned_host {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    unsafe {
        match attribute {
            CU_POINTER_ATTRIBUTE_CONTEXT => {
                let ctx_val = dev_match
                    .map(|(cid, _)| cid)
                    .or_else(current_ctx_id)
                    .unwrap_or(0);
                *(data as *mut usize) = ctx_val;
            }
            CU_POINTER_ATTRIBUTE_MEMORY_TYPE => {
                *(data as *mut c_int) = if dev_match.is_some() {
                    CU_MEMORYTYPE_DEVICE
                } else {
                    CU_MEMORYTYPE_HOST
                };
            }
            CU_POINTER_ATTRIBUTE_DEVICE_POINTER => {
                *(data as *mut CUdeviceptr) = if dev_match.is_some() { ptr } else { 0 };
            }
            CU_POINTER_ATTRIBUTE_HOST_POINTER => {
                *(data as *mut usize) = if pinned_host { ptr as usize } else { 0 };
            }
            CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL => {
                *(data as *mut c_int) = dev_match
                    .map(|(_, dev)| dev as c_int)
                    .unwrap_or(-2); // CU_DEVICE_CPU
            }
            _ => return Err(CUresult::CUDA_ERROR_INVALID_VALUE),
        }
    }
    Ok(())
}

#[no_mangle]
pub extern "C" fn cuPointerGetAttribute(
    data: *mut c_void,
    attribute: c_int,
    ptr: CUdeviceptr,
) -> CUresult {
    ffi_try(|| pointer_get_attribute_impl(data, attribute, ptr))
}

#[no_mangle]
pub extern "C" fn cuPointerGetAttributes(
    numAttributes: c_uint,
    attributes: *mut c_int,
    data: *mut *mut c_void,
    ptr: CUdeviceptr,
) -> CUresult {
    ffi_try(|| {
        if numAttributes == 0 {
            return Ok(());
        }
        if attributes.is_null() || data.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let n = numAttributes as usize;
        for i in 0..n {
            let attr = unsafe { *attributes.add(i) };
            let slot = unsafe { *data.add(i) };
            pointer_get_attribute_impl(slot, attr, ptr)?;
        }
        Ok(())
    })
}

pub(crate) fn load_module_from_slice(bytes: &[u8]) -> Result<CUmodule, CUresult> {
    let (ctx_id, ctx) = current_context()?;
    trace_log(
        "cuda-driver",
        &format!("cuModuleLoadData ctx=0x{ctx_id:x} bytes={}", bytes.len()),
    );
    let loaded = ctx.load_module(bytes).map_err(map_runtime_err)?;

    let mut g = global().lock().expect("global lock");
    let mod_id = g.alloc_handle();
    g.modules.insert(
        mod_id,
        StoredModule {
            ctx_id,
            module: loaded,
            func_handles: Vec::new(),
        },
    );
    Ok(mod_id as CUmodule)
}

#[no_mangle]
pub extern "C" fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
    ffi_try(|| {
        if module.is_null() || image.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let bytes = unsafe { read_module_image_bytes(image)? };
        let hmod = load_module_from_slice(bytes)?;
        unsafe {
            *module = hmod;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuModuleLoadDataEx(
    module: *mut CUmodule,
    image: *const c_void,
    _numOptions: c_uint,
    _options: *mut c_void,
    _optionValues: *mut *mut c_void,
) -> CUresult {
    cuModuleLoadData(module, image)
}

#[no_mangle]
pub extern "C" fn cuModuleUnload(hmod: CUmodule) -> CUresult {
    ffi_try(|| {
        if hmod.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let mod_id = hmod as usize;
        let stored = {
            let mut g = global().lock().expect("global lock");
            let stored = g
                .modules
                .remove(&mod_id)
                .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
            for fid in &stored.func_handles {
                g.functions.remove(fid);
            }
            stored
        };

        let ctx = {
            let g = global().lock().expect("global lock");
            g.contexts.get(&stored.ctx_id).cloned()
        };
        if let Some(c) = ctx {
            c.unload_module(stored.module).map_err(map_runtime_err)?;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuModuleGetFunction(
    hfunc: *mut CUfunction,
    hmod: CUmodule,
    name: *const c_char,
) -> CUresult {
    ffi_try(|| {
        if hfunc.is_null() || hmod.is_null() || name.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let kname = unsafe {
            CStr::from_ptr(name)
                .to_str()
                .map_err(|_| CUresult::CUDA_ERROR_INVALID_VALUE)?
        };
        let mod_id = hmod as usize;
        let mut g = global().lock().expect("global lock");
        let (ctx_id, func) = {
            let m = g
                .modules
                .get(&mod_id)
                .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
            let func = m.module.get_function(kname).map_err(map_runtime_err)?;
            (m.ctx_id, func)
        };
        let fid = g.alloc_handle();
        g.functions.insert(fid, StoredFunction { ctx_id, func });
        if let Some(m) = g.modules.get_mut(&mod_id) {
            m.func_handles.push(fid);
        }
        unsafe {
            *hfunc = fid as CUfunction;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    kernelParams: *mut *mut c_void,
    _extra: *mut *mut c_void,
) -> CUresult {
    ffi_try(|| {
        if f.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let (ctx, func, stream) = {
            let g = global().lock().expect("global lock");
            let sf = g
                .functions
                .get(&(f as usize))
                .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
            let ctx = g
                .contexts
                .get(&sf.ctx_id)
                .cloned()
                .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)?;
            let stream = if hStream.is_null() {
                None
            } else {
                let st = g
                    .streams
                    .get(&(hStream as usize))
                    .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
                if st.ctx_id != sf.ctx_id {
                    return Err(CUresult::CUDA_ERROR_INVALID_CONTEXT);
                }
                Some(st.stream)
            };
            (ctx, sf.func.clone(), stream)
        };

        let owned_args = unsafe { extract_kernel_param_slices(&func.metadata, kernelParams)? };
        trace_log(
            "cuda-driver",
            &format!(
                "cuLaunchKernel func={} grid=({gridDimX},{gridDimY},{gridDimZ}) block=({blockDimX},{blockDimY},{blockDimZ}) shared={sharedMemBytes}",
                func.name
            ),
        );
        let refs: Vec<&[u8]> = owned_args.iter().map(|v| v.as_slice()).collect();
        ctx.launch_kernel(
            &func,
            [gridDimX, gridDimY, gridDimZ],
            [blockDimX, blockDimY, blockDimZ],
            sharedMemBytes,
            stream,
            &refs,
        )
        .map_err(map_runtime_err)
    })
}

#[no_mangle]
pub extern "C" fn cuStreamCreate(phStream: *mut CUstream, flags: c_uint) -> CUresult {
    ffi_try(|| {
        if phStream.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let (ctx_id, ctx) = current_context()?;
        let stream = ctx.create_stream(flags).map_err(map_runtime_err)?;
        let mut g = global().lock().expect("global lock");
        let id = g.alloc_handle();
        g.streams.insert(id, StoredStream { ctx_id, stream });
        unsafe {
            *phStream = id as CUstream;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult {
    ffi_try(|| {
        if hStream.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let id = hStream as usize;
        let (stored, ctx) = {
            let mut g = global().lock().expect("global lock");
            let stored = g
                .streams
                .remove(&id)
                .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
            let ctx = g.contexts.get(&stored.ctx_id).cloned();
            (stored, ctx)
        };
        if let Some(c) = ctx {
            c.destroy_stream(stored.stream).map_err(map_runtime_err)?;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuStreamSynchronize(hStream: CUstream) -> CUresult {
    ffi_try(|| {
        if hStream.is_null() {
            let (_, ctx) = current_context()?;
            return ctx.synchronize().map_err(map_runtime_err);
        }
        let (ctx, stream) = {
            let g = global().lock().expect("global lock");
            let st = g
                .streams
                .get(&(hStream as usize))
                .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
            let ctx = g
                .contexts
                .get(&st.ctx_id)
                .cloned()
                .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)?;
            (ctx, st.stream)
        };
        ctx.synchronize_stream(stream).map_err(map_runtime_err)
    })
}

#[no_mangle]
pub extern "C" fn cuStreamWaitEvent(
    hStream: CUstream,
    hEvent: CUevent,
    flags: c_uint,
) -> CUresult {
    ffi_try(|| {
        if flags != 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        if hEvent.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let g = global().lock().expect("global lock");
        let ev = g
            .events
            .get(&(hEvent as usize))
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        let (ctx, stream) = if hStream.is_null() {
            let ctx = g
                .contexts
                .get(&ev.ctx_id)
                .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)?;
            (ctx, None)
        } else {
            let st = g
                .streams
                .get(&(hStream as usize))
                .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
            let ctx = g
                .contexts
                .get(&st.ctx_id)
                .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)?;
            (ctx, Some(st.stream))
        };
        ctx.stream_wait_event(stream, &ev.event)
            .map_err(map_runtime_err)
    })
}

#[no_mangle]
pub extern "C" fn cuEventCreate(phEvent: *mut CUevent, flags: c_uint) -> CUresult {
    ffi_try(|| {
        if phEvent.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let (ctx_id, ctx) = current_context()?;
        let event = ctx.create_event(flags).map_err(map_runtime_err)?;
        let mut g = global().lock().expect("global lock");
        let id = g.alloc_handle();
        g.events.insert(id, StoredEvent { ctx_id, event });
        unsafe {
            *phEvent = id as CUevent;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult {
    ffi_try(|| {
        if hEvent.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let id = hEvent as usize;
        let (stored, ctx) = {
            let mut g = global().lock().expect("global lock");
            let stored = g
                .events
                .remove(&id)
                .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
            let ctx = g.contexts.get(&stored.ctx_id).cloned();
            (stored, ctx)
        };
        if let Some(c) = ctx {
            c.destroy_event(stored.event).map_err(map_runtime_err)?;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult {
    ffi_try(|| {
        if hEvent.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let g = global().lock().expect("global lock");
        let ev = g
            .events
            .get(&(hEvent as usize))
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        let ctx = g
            .contexts
            .get(&ev.ctx_id)
            .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)?;
        let stream = if hStream.is_null() {
            None
        } else {
            let st = g
                .streams
                .get(&(hStream as usize))
                .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
            Some(st.stream)
        };
        ctx.record_event(&ev.event, stream).map_err(map_runtime_err)
    })
}

#[no_mangle]
pub extern "C" fn cuEventSynchronize(hEvent: CUevent) -> CUresult {
    ffi_try(|| {
        if hEvent.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let g = global().lock().expect("global lock");
        let ev = g
            .events
            .get(&(hEvent as usize))
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        let ctx = g
            .contexts
            .get(&ev.ctx_id)
            .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)?;
        ctx.synchronize_event(&ev.event).map_err(map_runtime_err)
    })
}

#[no_mangle]
pub extern "C" fn cuEventQuery(hEvent: CUevent) -> CUresult {
    ffi_try(|| {
        if hEvent.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let g = global().lock().expect("global lock");
        let ev = g
            .events
            .get(&(hEvent as usize))
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        let ctx = g
            .contexts
            .get(&ev.ctx_id)
            .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)?;
        let ready = ctx.query_event(&ev.event).map_err(map_runtime_err)?;
        if ready {
            Ok(())
        } else {
            Err(CUresult::CUDA_ERROR_NOT_READY)
        }
    })
}

#[no_mangle]
pub extern "C" fn cuEventElapsedTime(
    pMilliseconds: *mut f32,
    hStart: CUevent,
    hEnd: CUevent,
) -> CUresult {
    ffi_try(|| {
        if pMilliseconds.is_null() || hStart.is_null() || hEnd.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let g = global().lock().expect("global lock");
        let start = g
            .events
            .get(&(hStart as usize))
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        let end = g
            .events
            .get(&(hEnd as usize))
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        let ms = Event::elapsed_ms(&start.event, &end.event).map_err(map_runtime_err)?;
        unsafe {
            *pMilliseconds = ms;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuGraphCreate(phGraph: *mut CUgraph, _flags: c_uint) -> CUresult {
    ffi_try(|| {
        if phGraph.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let (ctx_id, _) = current_context()?;
        let mut g = global().lock().expect("global lock");
        let id = g.alloc_handle();
        g.graphs.insert(
            id,
            StoredGraph {
                ctx_id,
                graph: ExecutionGraph::new(),
            },
        );
        unsafe {
            *phGraph = id as CUgraph;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuGraphAddKernelNode(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    numDependencies: usize,
    nodeParams: *const CudaKernelNodeParams,
) -> CUresult {
    ffi_try(|| {
        if phGraphNode.is_null() || hGraph.is_null() || nodeParams.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let params = unsafe { &*nodeParams };
        if params.func.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }

        let mut deps = Vec::with_capacity(numDependencies);
        if numDependencies > 0 {
            if dependencies.is_null() {
                return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
            }
            let raw_deps = unsafe { std::slice::from_raw_parts(dependencies, numDependencies) };
            for &d in raw_deps {
                // Node handles store `node_idx + 1` to guarantee non-null.
                let idx = (d as usize)
                    .checked_sub(1)
                    .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
                deps.push(idx);
            }
        }

        let mut g = global().lock().expect("global lock");
        let func = g
            .functions
            .get(&(params.func as usize))
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?
            .func
            .clone();

        let kernarg_bytes = unsafe { extract_kernel_param_bytes(&func.metadata, params.kernel_params)? };

        let sg = g
            .graphs
            .get_mut(&(hGraph as usize))
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        let node_idx = sg
            .graph
            .add_kernel_node(
                &func,
                [params.grid_dim_x, params.grid_dim_y, params.grid_dim_z],
                [params.block_dim_x, params.block_dim_y, params.block_dim_z],
                params.shared_mem_bytes,
                kernarg_bytes,
                &deps,
            )
            .map_err(map_runtime_err)?;

        unsafe {
            *phGraphNode = (node_idx + 1) as CUgraphNode;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuGraphInstantiate(
    phGraphExec: *mut CUgraphExec,
    hGraph: CUgraph,
    _flags: u64,
) -> CUresult {
    ffi_try(|| {
        if phGraphExec.is_null() || hGraph.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let (ctx_id, ctx, graph_clone) = {
            let g = global().lock().expect("global lock");
            let sg = g
                .graphs
                .get(&(hGraph as usize))
                .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
            let ctx = g
                .contexts
                .get(&sg.ctx_id)
                .cloned()
                .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)?;
            (sg.ctx_id, ctx, sg.graph.clone())
        };
        let exec = ctx.instantiate_graph(&graph_clone).map_err(map_runtime_err)?;
        let mut g = global().lock().expect("global lock");
        let id = g.alloc_handle();
        g.graph_execs.insert(id, StoredGraphExec { ctx_id, exec });
        unsafe {
            *phGraphExec = id as CUgraphExec;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuGraphLaunch(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult {
    ffi_try(|| {
        if hGraphExec.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let g = global().lock().expect("global lock");
        let exec = g
            .graph_execs
            .get(&(hGraphExec as usize))
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        let ctx = g
            .contexts
            .get(&exec.ctx_id)
            .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)?;
        let stream = if hStream.is_null() {
            None
        } else {
            let st = g
                .streams
                .get(&(hStream as usize))
                .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
            Some(st.stream)
        };
        ctx.launch_graph(&exec.exec, stream)
            .map_err(map_runtime_err)
    })
}

#[no_mangle]
pub extern "C" fn cuGraphDestroy(hGraph: CUgraph) -> CUresult {
    ffi_try(|| {
        if hGraph.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let mut g = global().lock().expect("global lock");
        if g.graphs.remove(&(hGraph as usize)).is_none() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cuGraphExecDestroy(hGraphExec: CUgraphExec) -> CUresult {
    ffi_try(|| {
        if hGraphExec.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let mut g = global().lock().expect("global lock");
        if g.graph_execs.remove(&(hGraphExec as usize)).is_none() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        Ok(())
    })
}

fn resolve_driver_symbol(symbol: &str) -> Option<*mut c_void> {
    let ptr: *const c_void = match symbol {
        "cuInit" => cuInit as *const c_void,
        "cuDriverGetVersion" => cuDriverGetVersion as *const c_void,
        "cuDeviceGetCount" => cuDeviceGetCount as *const c_void,
        "cuDeviceGet" => cuDeviceGet as *const c_void,
        "cuDeviceGetName" => cuDeviceGetName as *const c_void,
        "cuDeviceTotalMem" | "cuDeviceTotalMem_v2" => cuDeviceTotalMem_v2 as *const c_void,
        "cuDeviceGetAttribute" => cuDeviceGetAttribute as *const c_void,
        "cuDevicePrimaryCtxRetain" => cuDevicePrimaryCtxRetain as *const c_void,
        "cuDevicePrimaryCtxRelease" | "cuDevicePrimaryCtxRelease_v2" => {
            cuDevicePrimaryCtxRelease_v2 as *const c_void
        }
        "cuCtxCreate" | "cuCtxCreate_v2" => cuCtxCreate_v2 as *const c_void,
        "cuCtxDestroy" | "cuCtxDestroy_v2" => cuCtxDestroy_v2 as *const c_void,
        "cuCtxPushCurrent" | "cuCtxPushCurrent_v2" => cuCtxPushCurrent_v2 as *const c_void,
        "cuCtxPopCurrent" | "cuCtxPopCurrent_v2" => cuCtxPopCurrent_v2 as *const c_void,
        "cuCtxSetCurrent" => cuCtxSetCurrent as *const c_void,
        "cuCtxGetCurrent" => cuCtxGetCurrent as *const c_void,
        "cuCtxSynchronize" => cuCtxSynchronize as *const c_void,
        "cuMemGetInfo" | "cuMemGetInfo_v2" => cuMemGetInfo_v2 as *const c_void,
        "cuMemAlloc" | "cuMemAlloc_v2" => cuMemAlloc_v2 as *const c_void,
        "cuMemFree" | "cuMemFree_v2" => cuMemFree_v2 as *const c_void,
        "cuMemsetD8" | "cuMemsetD8_v2" => cuMemsetD8_v2 as *const c_void,
        "cuMemsetD32" | "cuMemsetD32_v2" => cuMemsetD32_v2 as *const c_void,
        "cuMemcpyHtoD" | "cuMemcpyHtoD_v2" => cuMemcpyHtoD_v2 as *const c_void,
        "cuMemcpyDtoH" | "cuMemcpyDtoH_v2" => cuMemcpyDtoH_v2 as *const c_void,
        "cuMemcpyDtoD" | "cuMemcpyDtoD_v2" => cuMemcpyDtoD_v2 as *const c_void,
        "cuMemcpyHtoDAsync" | "cuMemcpyHtoDAsync_v2" => cuMemcpyHtoDAsync_v2 as *const c_void,
        "cuMemcpyDtoHAsync" | "cuMemcpyDtoHAsync_v2" => cuMemcpyDtoHAsync_v2 as *const c_void,
        "cuMemcpyDtoDAsync" | "cuMemcpyDtoDAsync_v2" => cuMemcpyDtoDAsync_v2 as *const c_void,
        "cuPointerGetAttribute" => cuPointerGetAttribute as *const c_void,
        "cuPointerGetAttributes" => cuPointerGetAttributes as *const c_void,
        "cuModuleLoadData" => cuModuleLoadData as *const c_void,
        "cuModuleLoadDataEx" => cuModuleLoadDataEx as *const c_void,
        "cuModuleUnload" => cuModuleUnload as *const c_void,
        "cuModuleGetFunction" => cuModuleGetFunction as *const c_void,
        "cuLaunchKernel" => cuLaunchKernel as *const c_void,
        "cuStreamCreate" => cuStreamCreate as *const c_void,
        "cuStreamDestroy" | "cuStreamDestroy_v2" => cuStreamDestroy_v2 as *const c_void,
        "cuStreamSynchronize" => cuStreamSynchronize as *const c_void,
        "cuStreamWaitEvent" => cuStreamWaitEvent as *const c_void,
        "cuEventCreate" => cuEventCreate as *const c_void,
        "cuEventDestroy" | "cuEventDestroy_v2" => cuEventDestroy_v2 as *const c_void,
        "cuEventRecord" => cuEventRecord as *const c_void,
        "cuEventSynchronize" => cuEventSynchronize as *const c_void,
        "cuEventQuery" => cuEventQuery as *const c_void,
        "cuEventElapsedTime" => cuEventElapsedTime as *const c_void,
        "cuGraphCreate" => cuGraphCreate as *const c_void,
        "cuGraphAddKernelNode" => cuGraphAddKernelNode as *const c_void,
        "cuGraphInstantiate" | "cuGraphInstantiateWithFlags" => cuGraphInstantiate as *const c_void,
        "cuGraphLaunch" => cuGraphLaunch as *const c_void,
        "cuGraphDestroy" => cuGraphDestroy as *const c_void,
        "cuGraphExecDestroy" => cuGraphExecDestroy as *const c_void,
        "cuGetProcAddress" => cuGetProcAddress as *const c_void,
        "cuGetProcAddress_v2" => cuGetProcAddress_v2 as *const c_void,
        _ => return None,
    };
    Some(ptr as *mut c_void)
}

#[no_mangle]
pub extern "C" fn cuGetProcAddress(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    _cudaVersion: c_int,
    _flags: u64,
) -> CUresult {
    cuGetProcAddress_v2(symbol, pfn, _cudaVersion, _flags, core::ptr::null_mut())
}

#[no_mangle]
pub extern "C" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    _cudaVersion: c_int,
    _flags: u64,
    symbolStatus: *mut c_int,
) -> CUresult {
    ffi_try(|| {
        if symbol.is_null() || pfn.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }
        let name = unsafe {
            CStr::from_ptr(symbol)
                .to_str()
                .map_err(|_| CUresult::CUDA_ERROR_INVALID_VALUE)?
        };
        if let Some(addr) = resolve_driver_symbol(name) {
            unsafe {
                *pfn = addr;
                if !symbolStatus.is_null() {
                    *symbolStatus = 0; // CU_GET_PROC_ADDRESS_SUCCESS
                }
            }
            Ok(())
        } else {
            unsafe {
                *pfn = core::ptr::null_mut();
                if !symbolStatus.is_null() {
                    *symbolStatus = 1; // CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND
                }
            }
            Err(CUresult::CUDA_ERROR_NOT_FOUND)
        }
    })
}
