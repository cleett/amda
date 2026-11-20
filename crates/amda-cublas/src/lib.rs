//! Drop-in cuBLAS (`cublas*`) C ABI library backed by real AMDA GPU GEMM kernels.

#![allow(non_camel_case_types, non_snake_case, clippy::not_unsafe_ptr_arg_deref)]

pub mod kernels;

use crate::kernels::GEMM_KERNELS_PTX;
use core::ffi::{c_char, c_int, c_longlong, c_uint, c_void};
use std::collections::HashMap;
use std::ffi::CString;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, OnceLock};

pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CudaStreamT = *mut c_void;
pub type CUdeviceptr = u64;

const CUDA_SUCCESS: c_int = 0;

/// Explicit CUDA Driver/Runtime dispatch table for static Rust linkers without `-rdynamic`.
#[derive(Clone, Copy)]
pub struct CudaDispatchTable {
    pub cuda_set_device: extern "C" fn(c_int) -> c_int,
    pub cu_ctx_get_current: extern "C" fn(*mut CUcontext) -> c_int,
    pub cu_module_load_data: extern "C" fn(*mut CUmodule, *const c_void) -> c_int,
    pub cu_module_get_function:
        extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> c_int,
    pub cu_module_unload: extern "C" fn(CUmodule) -> c_int,
    pub cu_launch_kernel: extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> c_int,
}

static CUDA_DISPATCH: OnceLock<CudaDispatchTable> = OnceLock::new();

/// Registers host CUDA symbols when linking statically inside Rust binaries.
pub fn register_cuda_dispatch(table: CudaDispatchTable) {
    let _ = CUDA_DISPATCH.set(table);
}

#[cfg(unix)]
extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[cfg(target_os = "macos")]
const RTLD_DEFAULT_PTR: *mut c_void = (-2isize) as *mut c_void;
#[cfg(not(target_os = "macos"))]
const RTLD_DEFAULT_PTR: *mut c_void = core::ptr::null_mut();

#[cfg(unix)]
unsafe fn resolve_dyn_symbol<T: Copy>(name: &core::ffi::CStr) -> Option<T> {
    let ptr = dlsym(RTLD_DEFAULT_PTR, name.as_ptr());
    if ptr.is_null() {
        None
    } else {
        Some(core::mem::transmute_copy(&ptr))
    }
}

#[cfg(not(unix))]
unsafe fn resolve_dyn_symbol<T: Copy>(_name: &core::ffi::CStr) -> Option<T> {
    None
}

fn resolve_cuda_api() -> Result<&'static CudaDispatchTable, CublasStatusT> {
    if let Some(tbl) = CUDA_DISPATCH.get() {
        return Ok(tbl);
    }

    unsafe {
        if let (Some(sdev), Some(gcur), Some(load), Some(gfn), Some(unld), Some(lnch)) = (
            resolve_dyn_symbol::<extern "C" fn(c_int) -> c_int>(c"cudaSetDevice"),
            resolve_dyn_symbol::<extern "C" fn(*mut CUcontext) -> c_int>(c"cuCtxGetCurrent"),
            resolve_dyn_symbol::<extern "C" fn(*mut CUmodule, *const c_void) -> c_int>(
                c"cuModuleLoadData",
            ),
            resolve_dyn_symbol::<
                extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> c_int,
            >(c"cuModuleGetFunction"),
            resolve_dyn_symbol::<extern "C" fn(CUmodule) -> c_int>(c"cuModuleUnload"),
            resolve_dyn_symbol::<
                extern "C" fn(
                    CUfunction,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    CUstream,
                    *mut *mut c_void,
                    *mut *mut c_void,
                ) -> c_int,
            >(c"cuLaunchKernel"),
        ) {
            let _ = CUDA_DISPATCH.set(CudaDispatchTable {
                cuda_set_device: sdev,
                cu_ctx_get_current: gcur,
                cu_module_load_data: load,
                cu_module_get_function: gfn,
                cu_module_unload: unld,
                cu_launch_kernel: lnch,
            });
            return CUDA_DISPATCH
                .get()
                .ok_or(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED);
        }
    }

    #[cfg(test)]
    {
        extern "C" fn wrap_cuda_set_device(d: c_int) -> c_int {
            amda_cuda::cudaSetDevice(d) as c_int
        }
        extern "C" fn wrap_cu_ctx_get_current(p: *mut CUcontext) -> c_int {
            amda_cuda::cuCtxGetCurrent(p) as c_int
        }
        extern "C" fn wrap_cu_module_load_data(m: *mut CUmodule, i: *const c_void) -> c_int {
            amda_cuda::cuModuleLoadData(m, i) as c_int
        }
        extern "C" fn wrap_cu_module_get_function(
            f: *mut CUfunction,
            m: CUmodule,
            n: *const c_char,
        ) -> c_int {
            amda_cuda::cuModuleGetFunction(f, m, n) as c_int
        }
        extern "C" fn wrap_cu_module_unload(m: CUmodule) -> c_int {
            amda_cuda::cuModuleUnload(m) as c_int
        }
        extern "C" fn wrap_cu_launch_kernel(
            f: CUfunction,
            gx: c_uint,
            gy: c_uint,
            gz: c_uint,
            bx: c_uint,
            by: c_uint,
            bz: c_uint,
            sm: c_uint,
            st: CUstream,
            kp: *mut *mut c_void,
            ex: *mut *mut c_void,
        ) -> c_int {
            amda_cuda::cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sm, st, kp, ex) as c_int
        }
        let _ = CUDA_DISPATCH.set(CudaDispatchTable {
            cuda_set_device: wrap_cuda_set_device,
            cu_ctx_get_current: wrap_cu_ctx_get_current,
            cu_module_load_data: wrap_cu_module_load_data,
            cu_module_get_function: wrap_cu_module_get_function,
            cu_module_unload: wrap_cu_module_unload,
            cu_launch_kernel: wrap_cu_launch_kernel,
        });
        CUDA_DISPATCH
            .get()
            .ok_or(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED)
    }

    #[cfg(not(test))]
    Err(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED)
}

const CUBLAS_VERSION: c_int = 12040;

pub type CublasHandleT = *mut c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum CublasStatusT {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum CublasOperationT {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum CublasMathT {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
    CUBLAS_PEDANTIC_MATH = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3,
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum CudaDataTypeT {
    CUDA_R_32F = 0,
    CUDA_R_64F = 1,
    CUDA_R_16F = 2,
    CUDA_R_8I = 3,
    CUDA_R_16BF = 14,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum CublasComputeTypeT {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum CublasGemmAlgoT {
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
}

#[derive(Clone, Copy)]
struct CachedKernels {
    module: usize,
    fn_f32: usize,
    fn_f16: usize,
    fn_bf16: usize,
}

struct HandleEntry {
    stream: usize,
    math_mode: CublasMathT,
    kernels: Option<CachedKernels>,
}

struct CublasState {
    next_id: usize,
    handles: HashMap<usize, HandleEntry>,
}

impl CublasState {
    fn new() -> Self {
        Self {
            next_id: 0x7001,
            handles: HashMap::new(),
        }
    }

    fn alloc_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(0x7001);
        id
    }
}

static CUBLAS_STATE: OnceLock<Mutex<CublasState>> = OnceLock::new();

fn state() -> &'static Mutex<CublasState> {
    CUBLAS_STATE.get_or_init(|| Mutex::new(CublasState::new()))
}

fn blas_try<F>(f: F) -> CublasStatusT
where
    F: FnOnce() -> Result<(), CublasStatusT>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => CublasStatusT::CUBLAS_STATUS_SUCCESS,
        Ok(Err(e)) => e,
        Err(_) => CublasStatusT::CUBLAS_STATUS_INTERNAL_ERROR,
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut m = mant;
        let mut e = -14i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let exp32 = u32::try_from(e + 127).unwrap_or(0);
        return f32::from_bits(sign | (exp32 << 23) | (m << 13));
    }
    if exp == 31 {
        let nan_bit = if mant != 0 { 0x0040_0000 } else { 0 };
        return f32::from_bits(sign | 0x7F80_0000 | nan_bit | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

fn load_gemm_kernels_if_needed(handle_id: usize) -> Result<(CachedKernels, CudaStreamT), CublasStatusT> {
    let (cached, stream) = {
        let st = state().lock().expect("cublas state lock");
        let entry = st
            .handles
            .get(&handle_id)
            .ok_or(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED)?;
        (entry.kernels, entry.stream as CudaStreamT)
    };
    if let Some(k) = cached {
        return Ok((k, stream));
    }

    let api = resolve_cuda_api()?;
    let mut module: CUmodule = core::ptr::null_mut();
    let rc = (api.cu_module_load_data)(&mut module, GEMM_KERNELS_PTX.as_ptr() as *const c_void);
    if rc != CUDA_SUCCESS {
        return Err(CublasStatusT::CUBLAS_STATUS_EXECUTION_FAILED);
    }

    let get_fn = |name: &str| -> Result<usize, CublasStatusT> {
        let cname = CString::new(name).map_err(|_| CublasStatusT::CUBLAS_STATUS_INTERNAL_ERROR)?;
        let mut func: CUfunction = core::ptr::null_mut();
        if (api.cu_module_get_function)(&mut func, module, cname.as_ptr()) != CUDA_SUCCESS {
            return Err(CublasStatusT::CUBLAS_STATUS_INTERNAL_ERROR);
        }
        Ok(func as usize)
    };

    let loaded = match (
        get_fn("amda_gemm_f32"),
        get_fn("amda_gemm_f16"),
        get_fn("amda_gemm_bf16"),
    ) {
        (Ok(fn_f32), Ok(fn_f16), Ok(fn_bf16)) => CachedKernels {
            module: module as usize,
            fn_f32,
            fn_f16,
            fn_bf16,
        },
        _ => {
            let _ = (api.cu_module_unload)(module);
            return Err(CublasStatusT::CUBLAS_STATUS_INTERNAL_ERROR);
        }
    };

    let mut duplicate = false;
    let chosen = {
        let mut st = state().lock().expect("cublas state lock");
        let Some(entry) = st.handles.get_mut(&handle_id) else {
            let _ = (api.cu_module_unload)(module);
            return Err(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED);
        };
        if let Some(existing) = entry.kernels {
            duplicate = true;
            existing
        } else {
            entry.kernels = Some(loaded);
            loaded
        }
    };
    if duplicate {
        let _ = (api.cu_module_unload)(module);
    }
    Ok((chosen, stream))
}

#[derive(Clone, Copy)]
enum GemmPrecision {
    F32,
    F16 { out_is_f32: u32 },
    Bf16 { out_is_f32: u32 },
}

#[allow(clippy::too_many_arguments)]
fn dispatch_gemm(
    handle: CublasHandleT,
    transa: CublasOperationT,
    transb: CublasOperationT,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f32,
    a_ptr: *const c_void,
    lda: c_int,
    stride_a: c_longlong,
    b_ptr: *const c_void,
    ldb: c_int,
    stride_b: c_longlong,
    beta: f32,
    c_ptr: *mut c_void,
    ldc: c_int,
    stride_c: c_longlong,
    batch_count: c_int,
    prec: GemmPrecision,
) -> Result<(), CublasStatusT> {
    if handle.is_null() {
        return Err(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED);
    }
    if m < 0 || n < 0 || k < 0 || lda <= 0 || ldb <= 0 || ldc <= 0 || batch_count <= 0 {
        return Err(CublasStatusT::CUBLAS_STATUS_INVALID_VALUE);
    }
    if stride_a < 0 || stride_b < 0 || stride_c < 0 {
        return Err(CublasStatusT::CUBLAS_STATUS_INVALID_VALUE);
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k > 0 && (a_ptr.is_null() || b_ptr.is_null()) {
        return Err(CublasStatusT::CUBLAS_STATUS_INVALID_VALUE);
    }
    if c_ptr.is_null() {
        return Err(CublasStatusT::CUBLAS_STATUS_INVALID_VALUE);
    }

    let (kernels, stream) = load_gemm_kernels_if_needed(handle as usize)?;
    let ta_flag: u32 = u32::from(!matches!(transa, CublasOperationT::CUBLAS_OP_N));
    let tb_flag: u32 = u32::from(!matches!(transb, CublasOperationT::CUBLAS_OP_N));

    let mut d_a = a_ptr as CUdeviceptr;
    let mut d_b = b_ptr as CUdeviceptr;
    let mut d_c = c_ptr as CUdeviceptr;
    let mut arg_m = m as u32;
    let mut arg_n = n as u32;
    let mut arg_k = k as u32;
    let mut arg_lda = lda as u32;
    let mut arg_ldb = ldb as u32;
    let mut arg_ldc = ldc as u32;
    let mut arg_sa = u32::try_from(stride_a).map_err(|_| CublasStatusT::CUBLAS_STATUS_INVALID_VALUE)?;
    let mut arg_sb = u32::try_from(stride_b).map_err(|_| CublasStatusT::CUBLAS_STATUS_INVALID_VALUE)?;
    let mut arg_sc = u32::try_from(stride_c).map_err(|_| CublasStatusT::CUBLAS_STATUS_INVALID_VALUE)?;
    let mut arg_ta = ta_flag;
    let mut arg_tb = tb_flag;
    let mut arg_alpha = alpha;
    let mut arg_beta = beta;
    let mut arg_out_f32 = match prec {
        GemmPrecision::F16 { out_is_f32 } | GemmPrecision::Bf16 { out_is_f32 } => out_is_f32,
        GemmPrecision::F32 => 1,
    };

    let (func, use_out_flag) = match prec {
        GemmPrecision::F32 => (kernels.fn_f32 as CUfunction, false),
        GemmPrecision::F16 { .. } => (kernels.fn_f16 as CUfunction, true),
        GemmPrecision::Bf16 { .. } => (kernels.fn_bf16 as CUfunction, true),
    };

    let mut params_16: [*mut c_void; 16] = [
        (&mut d_a as *mut CUdeviceptr) as *mut c_void,
        (&mut d_b as *mut CUdeviceptr) as *mut c_void,
        (&mut d_c as *mut CUdeviceptr) as *mut c_void,
        (&mut arg_m as *mut u32) as *mut c_void,
        (&mut arg_n as *mut u32) as *mut c_void,
        (&mut arg_k as *mut u32) as *mut c_void,
        (&mut arg_lda as *mut u32) as *mut c_void,
        (&mut arg_ldb as *mut u32) as *mut c_void,
        (&mut arg_ldc as *mut u32) as *mut c_void,
        (&mut arg_sa as *mut u32) as *mut c_void,
        (&mut arg_sb as *mut u32) as *mut c_void,
        (&mut arg_sc as *mut u32) as *mut c_void,
        (&mut arg_ta as *mut u32) as *mut c_void,
        (&mut arg_tb as *mut u32) as *mut c_void,
        (&mut arg_alpha as *mut f32) as *mut c_void,
        (&mut arg_beta as *mut f32) as *mut c_void,
    ];
    let mut params_17: [*mut c_void; 17] = [
        params_16[0],
        params_16[1],
        params_16[2],
        params_16[3],
        params_16[4],
        params_16[5],
        params_16[6],
        params_16[7],
        params_16[8],
        params_16[9],
        params_16[10],
        params_16[11],
        params_16[12],
        params_16[13],
        params_16[14],
        params_16[15],
        (&mut arg_out_f32 as *mut u32) as *mut c_void,
    ];

    let param_ptr = if use_out_flag {
        params_17.as_mut_ptr()
    } else {
        params_16.as_mut_ptr()
    };

    let api = resolve_cuda_api()?;
    let gx = (m as u32).div_ceil(16);
    let gy = (n as u32).div_ceil(16);
    let gz = batch_count as u32;
    let rc = (api.cu_launch_kernel)(
        func,
        gx,
        gy,
        gz,
        16,
        16,
        1,
        0,
        stream as CUstream,
        param_ptr,
        core::ptr::null_mut(),
    );
    if rc == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CublasStatusT::CUBLAS_STATUS_EXECUTION_FAILED)
    }
}

#[no_mangle]
pub extern "C" fn cublasCreate_v2(handle: *mut CublasHandleT) -> CublasStatusT {
    blas_try(|| {
        if handle.is_null() {
            return Err(CublasStatusT::CUBLAS_STATUS_INVALID_VALUE);
        }
        let api = resolve_cuda_api()?;
        let mut cur: CUcontext = core::ptr::null_mut();
        if ((api.cu_ctx_get_current)(&mut cur) != CUDA_SUCCESS || cur.is_null())
            && (api.cuda_set_device)(0) != CUDA_SUCCESS
        {
            return Err(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED);
        }
        let mut st = state().lock().expect("cublas state lock");
        let id = st.alloc_id();
        st.handles.insert(
            id,
            HandleEntry {
                stream: 0,
                math_mode: CublasMathT::CUBLAS_DEFAULT_MATH,
                kernels: None,
            },
        );
        unsafe {
            *handle = id as CublasHandleT;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cublasDestroy_v2(handle: CublasHandleT) -> CublasStatusT {
    blas_try(|| {
        if handle.is_null() {
            return Err(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED);
        }
        let entry = {
            let mut st = state().lock().expect("cublas state lock");
            st.handles
                .remove(&(handle as usize))
                .ok_or(CublasStatusT::CUBLAS_STATUS_INVALID_VALUE)?
        };
        if let Some(k) = entry.kernels {
            if let Ok(api) = resolve_cuda_api() {
                let _ = (api.cu_module_unload)(k.module as CUmodule);
            }
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cublasSetStream_v2(
    handle: CublasHandleT,
    streamId: CudaStreamT,
) -> CublasStatusT {
    blas_try(|| {
        if handle.is_null() {
            return Err(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED);
        }
        let mut st = state().lock().expect("cublas state lock");
        let entry = st
            .handles
            .get_mut(&(handle as usize))
            .ok_or(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED)?;
        entry.stream = streamId as usize;
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cublasGetStream_v2(
    handle: CublasHandleT,
    streamId: *mut CudaStreamT,
) -> CublasStatusT {
    blas_try(|| {
        if handle.is_null() {
            return Err(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED);
        }
        if streamId.is_null() {
            return Err(CublasStatusT::CUBLAS_STATUS_INVALID_VALUE);
        }
        let st = state().lock().expect("cublas state lock");
        let entry = st
            .handles
            .get(&(handle as usize))
            .ok_or(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED)?;
        unsafe {
            *streamId = entry.stream as CudaStreamT;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cublasSetMathMode(handle: CublasHandleT, mode: CublasMathT) -> CublasStatusT {
    blas_try(|| {
        if handle.is_null() {
            return Err(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED);
        }
        let mut st = state().lock().expect("cublas state lock");
        let entry = st
            .handles
            .get_mut(&(handle as usize))
            .ok_or(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED)?;
        entry.math_mode = mode;
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cublasGetMathMode(handle: CublasHandleT, mode: *mut CublasMathT) -> CublasStatusT {
    blas_try(|| {
        if handle.is_null() {
            return Err(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED);
        }
        if mode.is_null() {
            return Err(CublasStatusT::CUBLAS_STATUS_INVALID_VALUE);
        }
        let st = state().lock().expect("cublas state lock");
        let entry = st
            .handles
            .get(&(handle as usize))
            .ok_or(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED)?;
        unsafe {
            *mode = entry.math_mode;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cublasGetVersion_v2(handle: CublasHandleT, version: *mut c_int) -> CublasStatusT {
    blas_try(|| {
        if version.is_null() {
            return Err(CublasStatusT::CUBLAS_STATUS_INVALID_VALUE);
        }
        if !handle.is_null() {
            let st = state().lock().expect("cublas state lock");
            if !st.handles.contains_key(&(handle as usize)) {
                return Err(CublasStatusT::CUBLAS_STATUS_NOT_INITIALIZED);
            }
        }
        unsafe {
            *version = CUBLAS_VERSION;
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn cublasSgemm_v2(
    handle: CublasHandleT,
    transa: CublasOperationT,
    transb: CublasOperationT,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    A: *const f32,
    lda: c_int,
    B: *const f32,
    ldb: c_int,
    beta: *const f32,
    C: *mut f32,
    ldc: c_int,
) -> CublasStatusT {
    cublasSgemmStridedBatched(
        handle, transa, transb, m, n, k, alpha, A, lda, 0, B, ldb, 0, beta, C, ldc, 0, 1,
    )
}

#[no_mangle]
pub extern "C" fn cublasSgemmStridedBatched(
    handle: CublasHandleT,
    transa: CublasOperationT,
    transb: CublasOperationT,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    A: *const f32,
    lda: c_int,
    strideA: c_longlong,
    B: *const f32,
    ldb: c_int,
    strideB: c_longlong,
    beta: *const f32,
    C: *mut f32,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
) -> CublasStatusT {
    blas_try(|| {
        if alpha.is_null() || beta.is_null() {
            return Err(CublasStatusT::CUBLAS_STATUS_INVALID_VALUE);
        }
        let a_val = unsafe { *alpha };
        let b_val = unsafe { *beta };
        dispatch_gemm(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            a_val,
            A as *const c_void,
            lda,
            strideA,
            B as *const c_void,
            ldb,
            strideB,
            b_val,
            C as *mut c_void,
            ldc,
            strideC,
            batchCount,
            GemmPrecision::F32,
        )
    })
}

#[no_mangle]
pub extern "C" fn cublasHgemm(
    handle: CublasHandleT,
    transa: CublasOperationT,
    transb: CublasOperationT,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    A: *const c_void,
    lda: c_int,
    B: *const c_void,
    ldb: c_int,
    beta: *const c_void,
    C: *mut c_void,
    ldc: c_int,
) -> CublasStatusT {
    blas_try(|| {
        if alpha.is_null() || beta.is_null() {
            return Err(CublasStatusT::CUBLAS_STATUS_INVALID_VALUE);
        }
        let a_bits = unsafe { *(alpha as *const u16) };
        let b_bits = unsafe { *(beta as *const u16) };
        dispatch_gemm(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            f16_bits_to_f32(a_bits),
            A,
            lda,
            0,
            B,
            ldb,
            0,
            f16_bits_to_f32(b_bits),
            C,
            ldc,
            0,
            1,
            GemmPrecision::F16 { out_is_f32: 0 },
        )
    })
}

#[no_mangle]
pub extern "C" fn cublasGemmEx(
    handle: CublasHandleT,
    transa: CublasOperationT,
    transb: CublasOperationT,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    A: *const c_void,
    Atype: CudaDataTypeT,
    lda: c_int,
    B: *const c_void,
    Btype: CudaDataTypeT,
    ldb: c_int,
    beta: *const c_void,
    C: *mut c_void,
    Ctype: CudaDataTypeT,
    ldc: c_int,
    computeType: CublasComputeTypeT,
    algo: CublasGemmAlgoT,
) -> CublasStatusT {
    cublasGemmStridedBatchedEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A,
        Atype,
        lda,
        0,
        B,
        Btype,
        ldb,
        0,
        beta,
        C,
        Ctype,
        ldc,
        0,
        1,
        computeType,
        algo,
    )
}

#[no_mangle]
pub extern "C" fn cublasGemmStridedBatchedEx(
    handle: CublasHandleT,
    transa: CublasOperationT,
    transb: CublasOperationT,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    A: *const c_void,
    Atype: CudaDataTypeT,
    lda: c_int,
    strideA: c_longlong,
    B: *const c_void,
    Btype: CudaDataTypeT,
    ldb: c_int,
    strideB: c_longlong,
    beta: *const c_void,
    C: *mut c_void,
    Ctype: CudaDataTypeT,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
    computeType: CublasComputeTypeT,
    _algo: CublasGemmAlgoT,
) -> CublasStatusT {
    blas_try(|| {
        if alpha.is_null() || beta.is_null() {
            return Err(CublasStatusT::CUBLAS_STATUS_INVALID_VALUE);
        }
        let (alpha_f32, beta_f32) = match computeType {
            CublasComputeTypeT::CUBLAS_COMPUTE_16F
            | CublasComputeTypeT::CUBLAS_COMPUTE_16F_PEDANTIC => unsafe {
                (
                    f16_bits_to_f32(*(alpha as *const u16)),
                    f16_bits_to_f32(*(beta as *const u16)),
                )
            },
            _ => unsafe { (*(alpha as *const f32), *(beta as *const f32)) },
        };

        let prec = match (Atype, Btype, Ctype) {
            (
                CudaDataTypeT::CUDA_R_32F,
                CudaDataTypeT::CUDA_R_32F,
                CudaDataTypeT::CUDA_R_32F,
            ) => GemmPrecision::F32,
            (
                CudaDataTypeT::CUDA_R_16F,
                CudaDataTypeT::CUDA_R_16F,
                CudaDataTypeT::CUDA_R_16F,
            ) => GemmPrecision::F16 { out_is_f32: 0 },
            (
                CudaDataTypeT::CUDA_R_16F,
                CudaDataTypeT::CUDA_R_16F,
                CudaDataTypeT::CUDA_R_32F,
            ) => GemmPrecision::F16 { out_is_f32: 1 },
            (
                CudaDataTypeT::CUDA_R_16BF,
                CudaDataTypeT::CUDA_R_16BF,
                CudaDataTypeT::CUDA_R_16BF,
            ) => GemmPrecision::Bf16 { out_is_f32: 0 },
            (
                CudaDataTypeT::CUDA_R_16BF,
                CudaDataTypeT::CUDA_R_16BF,
                CudaDataTypeT::CUDA_R_32F,
            ) => GemmPrecision::Bf16 { out_is_f32: 1 },
            _ => return Err(CublasStatusT::CUBLAS_STATUS_NOT_SUPPORTED),
        };

        dispatch_gemm(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha_f32,
            A,
            lda,
            strideA,
            B,
            ldb,
            strideB,
            beta_f32,
            C,
            ldc,
            strideC,
            batchCount,
            prec,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use amda_compiler::{
        compile_llvm_ir_to_hsaco, compile_tier0_native, disassemble_hsaco_with_objdump,
        lower_ptx_to_ir, parse_ptx, validate_llvm_ir, verify_encoded_machine_code, Compiler,
        LlvmToolchain, TierPolicy,
    };
    use amda_cuda::{cudaFree, cudaMalloc, cudaMemcpy, cudaSetDevice, CudaErrorT, CudaMemcpyKind};
    use amda_elf::IsaTarget;

    static BLAS_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn f32_to_f16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let mant = bits & 0x007F_FFFF;
        if exp > 15 {
            return sign | 0x7C00;
        }
        if exp < -14 {
            return sign;
        }
        let e16 = ((exp + 15) as u16) << 10;
        let m16 = ((mant + 0x1000) >> 13) as u16;
        sign | e16 | (m16 & 0x03FF)
    }

    fn f32_to_bf16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let lsb = (bits >> 16) & 1;
        ((bits.wrapping_add(0x7FFF + lsb)) >> 16) as u16
    }

    fn bf16_bits_to_f32(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_col_gemm_batch(
        m: usize,
        n: usize,
        k: usize,
        ta: bool,
        tb: bool,
        alpha: f32,
        a: &[f32],
        lda: usize,
        stride_a: usize,
        b: &[f32],
        ldb: usize,
        stride_b: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
        stride_c: usize,
        batches: usize,
    ) {
        for batch in 0..batches {
            let a_off = batch * stride_a;
            let b_off = batch * stride_b;
            let c_off = batch * stride_c;
            for col in 0..n {
                for row in 0..m {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        let va = if ta {
                            a[a_off + p + row * lda]
                        } else {
                            a[a_off + row + p * lda]
                        };
                        let vb = if tb {
                            b[b_off + col + p * ldb]
                        } else {
                            b[b_off + p + col * ldb]
                        };
                        sum = va.mul_add(vb, sum);
                    }
                    let c_idx = c_off + row + col * ldc;
                    let prev = if beta == 0.0 { 0.0 } else { beta * c[c_idx] };
                    c[c_idx] = alpha * sum + prev;
                }
            }
        }
    }

    #[test]
    fn cublas_gemm_kernels_pass_llvm_and_tier0_verification() {
        let ptx_src = crate::kernels::gemm_kernels_ptx_source();
        let compiler = Compiler::with_cache_dir(None);
        let toolchain = LlvmToolchain::discover();
        let ptx_mod = parse_ptx(ptx_src).expect("gemm PTX parse");
        assert_eq!(ptx_mod.kernels.len(), 3);
        for target in [IsaTarget::Gfx1100, IsaTarget::Gfx942] {
            let art1 = compiler
                .compile_ptx_with_policy(ptx_src, target, TierPolicy::ForceTier1)
                .expect("Tier 1 GEMM compile");
            let llvm = art1.llvm_ir.as_deref().expect("LLVM IR present");
            validate_llvm_ir(llvm).expect("GEMM LLVM IR must validate");
            if toolchain.has_codegen_tools() {
                let hsaco = compile_llvm_ir_to_hsaco(llvm, target)
                    .expect("llc/ld.lld GEMM compilation must succeed");
                assert!(!hsaco.is_empty());
            }

            let ir = lower_ptx_to_ir(&ptx_mod, target).expect("GEMM IR lower");
            let native = compile_tier0_native(&ir, target).expect("Tier 0 GEMM compile");
            assert_eq!(native.kernels.len(), 3);
            for nk in &native.kernels {
                let decoded =
                    verify_encoded_machine_code(&nk.machine_code, target).expect("ISA verify");
                assert!(!decoded.is_empty());
            }

            let art0 = compiler
                .compile_ptx_with_policy(ptx_src, target, TierPolicy::ForceTier0)
                .expect("Tier 0 HSACO compile");
            if let Some(txt) =
                disassemble_hsaco_with_objdump(&art0.hsaco, target).expect("objdump check")
            {
                assert!(!txt.is_empty());
            }
        }
    }

    #[test]
    fn cublas_sgemm_variants_and_strided_batched() {
        let _guard = BLAS_TEST_LOCK.lock().expect("blas lock");
        assert_eq!(cudaSetDevice(0), CudaErrorT::CudaSuccess);

        let mut handle: CublasHandleT = core::ptr::null_mut();
        assert_eq!(cublasCreate_v2(&mut handle), CublasStatusT::CUBLAS_STATUS_SUCCESS);

        let mut ver = 0;
        assert_eq!(
            cublasGetVersion_v2(handle, &mut ver),
            CublasStatusT::CUBLAS_STATUS_SUCCESS
        );
        assert_eq!(ver, CUBLAS_VERSION);

        // Non-square dimensions crossing 16x16 tile boundaries
        let (m, n, k) = (19usize, 17usize, 11usize);
        for (op_a, op_b) in [
            (CublasOperationT::CUBLAS_OP_N, CublasOperationT::CUBLAS_OP_N),
            (CublasOperationT::CUBLAS_OP_T, CublasOperationT::CUBLAS_OP_N),
            (CublasOperationT::CUBLAS_OP_N, CublasOperationT::CUBLAS_OP_T),
            (CublasOperationT::CUBLAS_OP_T, CublasOperationT::CUBLAS_OP_T),
        ] {
            let ta = op_a != CublasOperationT::CUBLAS_OP_N;
            let tb = op_b != CublasOperationT::CUBLAS_OP_N;
            let (lda, a_cols) = if ta { (k, m) } else { (m, k) };
            let (ldb, b_cols) = if tb { (n, k) } else { (k, n) };
            let ldc = m;

            let ha: Vec<f32> = (0..(lda * a_cols))
                .map(|i| ((i % 7) as f32) * 0.25 - 0.5)
                .collect();
            let hb: Vec<f32> = (0..(ldb * b_cols))
                .map(|i| ((i % 5) as f32) * 0.5 - 0.75)
                .collect();
            let hc_init: Vec<f32> = (0..(ldc * n)).map(|i| (i % 3) as f32 * 0.125).collect();
            let mut cpu_c = hc_init.clone();
            let alpha = 1.5f32;
            let beta = 0.5f32;
            cpu_col_gemm_batch(
                m,
                n,
                k,
                ta,
                tb,
                alpha,
                &ha,
                lda,
                0,
                &hb,
                ldb,
                0,
                beta,
                &mut cpu_c,
                ldc,
                0,
                1,
            );

            let (mut d_a, mut d_b, mut d_c) =
                (core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null_mut());
            assert_eq!(cudaMalloc(&mut d_a, ha.len() * 4), CudaErrorT::CudaSuccess);
            assert_eq!(cudaMalloc(&mut d_b, hb.len() * 4), CudaErrorT::CudaSuccess);
            assert_eq!(cudaMalloc(&mut d_c, hc_init.len() * 4), CudaErrorT::CudaSuccess);
            assert_eq!(
                cudaMemcpy(d_a, ha.as_ptr() as *const c_void, ha.len() * 4, CudaMemcpyKind::HostToDevice),
                CudaErrorT::CudaSuccess
            );
            assert_eq!(
                cudaMemcpy(d_b, hb.as_ptr() as *const c_void, hb.len() * 4, CudaMemcpyKind::HostToDevice),
                CudaErrorT::CudaSuccess
            );
            assert_eq!(
                cudaMemcpy(d_c, hc_init.as_ptr() as *const c_void, hc_init.len() * 4, CudaMemcpyKind::HostToDevice),
                CudaErrorT::CudaSuccess
            );

            assert_eq!(
                cublasSgemm_v2(
                    handle,
                    op_a,
                    op_b,
                    m as c_int,
                    n as c_int,
                    k as c_int,
                    &alpha,
                    d_a as *const f32,
                    lda as c_int,
                    d_b as *const f32,
                    ldb as c_int,
                    &beta,
                    d_c as *mut f32,
                    ldc as c_int,
                ),
                CublasStatusT::CUBLAS_STATUS_SUCCESS
            );

            let mut gpu_c = vec![0.0f32; hc_init.len()];
            assert_eq!(
                cudaMemcpy(
                    gpu_c.as_mut_ptr() as *mut c_void,
                    d_c,
                    gpu_c.len() * 4,
                    CudaMemcpyKind::DeviceToHost
                ),
                CudaErrorT::CudaSuccess
            );
            for i in 0..gpu_c.len() {
                assert!(
                    (gpu_c[i] - cpu_c[i]).abs() < 1e-4,
                    "sgemm ({op_a:?},{op_b:?}) mismatch at {i}: gpu={} cpu={}",
                    gpu_c[i],
                    cpu_c[i]
                );
            }
            assert_eq!(cudaFree(d_a), CudaErrorT::CudaSuccess);
            assert_eq!(cudaFree(d_b), CudaErrorT::CudaSuccess);
            assert_eq!(cudaFree(d_c), CudaErrorT::CudaSuccess);
        }

        // Batched SGEMM
        let (bm, bn, bk, batches) = (5usize, 4usize, 6usize, 3usize);
        let sa = bm * bk;
        let sb = bk * bn;
        let sc = bm * bn;
        let ha: Vec<f32> = (0..(batches * sa)).map(|i| (i as f32) * 0.05).collect();
        let hb: Vec<f32> = (0..(batches * sb)).map(|i| ((i % 9) as f32) * 0.1 - 0.3).collect();
        let mut cpu_c = vec![0.0f32; batches * sc];
        let alpha = 1.0f32;
        let beta = 0.0f32;
        cpu_col_gemm_batch(
            bm,
            bn,
            bk,
            false,
            false,
            alpha,
            &ha,
            bm,
            sa,
            &hb,
            bk,
            sb,
            beta,
            &mut cpu_c,
            bm,
            sc,
            batches,
        );

        let (mut d_a, mut d_b, mut d_c) =
            (core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null_mut());
        assert_eq!(cudaMalloc(&mut d_a, ha.len() * 4), CudaErrorT::CudaSuccess);
        assert_eq!(cudaMalloc(&mut d_b, hb.len() * 4), CudaErrorT::CudaSuccess);
        assert_eq!(cudaMalloc(&mut d_c, cpu_c.len() * 4), CudaErrorT::CudaSuccess);
        assert_eq!(
            cudaMemcpy(d_a, ha.as_ptr() as *const c_void, ha.len() * 4, CudaMemcpyKind::HostToDevice),
            CudaErrorT::CudaSuccess
        );
        assert_eq!(
            cudaMemcpy(d_b, hb.as_ptr() as *const c_void, hb.len() * 4, CudaMemcpyKind::HostToDevice),
            CudaErrorT::CudaSuccess
        );
        assert_eq!(
            cublasSgemmStridedBatched(
                handle,
                CublasOperationT::CUBLAS_OP_N,
                CublasOperationT::CUBLAS_OP_N,
                bm as c_int,
                bn as c_int,
                bk as c_int,
                &alpha,
                d_a as *const f32,
                bm as c_int,
                sa as c_longlong,
                d_b as *const f32,
                bk as c_int,
                sb as c_longlong,
                &beta,
                d_c as *mut f32,
                bm as c_int,
                sc as c_longlong,
                batches as c_int,
            ),
            CublasStatusT::CUBLAS_STATUS_SUCCESS
        );
        let mut gpu_c = vec![0.0f32; cpu_c.len()];
        assert_eq!(
            cudaMemcpy(
                gpu_c.as_mut_ptr() as *mut c_void,
                d_c,
                gpu_c.len() * 4,
                CudaMemcpyKind::DeviceToHost
            ),
            CudaErrorT::CudaSuccess
        );
        for i in 0..gpu_c.len() {
            assert!((gpu_c[i] - cpu_c[i]).abs() < 1e-4);
        }

        assert_eq!(cudaFree(d_a), CudaErrorT::CudaSuccess);
        assert_eq!(cudaFree(d_b), CudaErrorT::CudaSuccess);
        assert_eq!(cudaFree(d_c), CudaErrorT::CudaSuccess);
        assert_eq!(cublasDestroy_v2(handle), CublasStatusT::CUBLAS_STATUS_SUCCESS);
    }

    #[test]
    fn cublas_hgemm_and_gemm_ex_f16_bf16() {
        let _guard = BLAS_TEST_LOCK.lock().expect("blas lock");
        assert_eq!(cudaSetDevice(0), CudaErrorT::CudaSuccess);

        let mut handle: CublasHandleT = core::ptr::null_mut();
        assert_eq!(cublasCreate_v2(&mut handle), CublasStatusT::CUBLAS_STATUS_SUCCESS);
        assert_eq!(
            cublasSetMathMode(handle, CublasMathT::CUBLAS_TENSOR_OP_MATH),
            CublasStatusT::CUBLAS_STATUS_SUCCESS
        );

        let (m, n, k) = (8usize, 6usize, 8usize);
        let ha_f32: Vec<f32> = (0..(m * k)).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect();
        let hb_f32: Vec<f32> = (0..(k * n)).map(|i| ((i % 4) as f32 - 1.0) * 0.5).collect();
        let ha_f16: Vec<u16> = ha_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let hb_f16: Vec<u16> = hb_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let ha_bf16: Vec<u16> = ha_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();
        let hb_bf16: Vec<u16> = hb_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();

        let mut cpu_f32 = vec![0.0f32; m * n];
        cpu_col_gemm_batch(
            m,
            n,
            k,
            false,
            false,
            1.0,
            &ha_f32,
            m,
            0,
            &hb_f32,
            k,
            0,
            0.0,
            &mut cpu_f32,
            m,
            0,
            1,
        );

        let (mut d_af16, mut d_bf16, mut d_cf16, mut d_cf32) = (
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        assert_eq!(cudaMalloc(&mut d_af16, ha_f16.len() * 2), CudaErrorT::CudaSuccess);
        assert_eq!(cudaMalloc(&mut d_bf16, hb_f16.len() * 2), CudaErrorT::CudaSuccess);
        assert_eq!(cudaMalloc(&mut d_cf16, m * n * 2), CudaErrorT::CudaSuccess);
        assert_eq!(cudaMalloc(&mut d_cf32, m * n * 4), CudaErrorT::CudaSuccess);
        assert_eq!(
            cudaMemcpy(
                d_af16,
                ha_f16.as_ptr() as *const c_void,
                ha_f16.len() * 2,
                CudaMemcpyKind::HostToDevice
            ),
            CudaErrorT::CudaSuccess
        );
        assert_eq!(
            cudaMemcpy(
                d_bf16,
                hb_f16.as_ptr() as *const c_void,
                hb_f16.len() * 2,
                CudaMemcpyKind::HostToDevice
            ),
            CudaErrorT::CudaSuccess
        );

        // 1. cublasHgemm
        let alpha16 = f32_to_f16_bits(1.0);
        let beta16 = f32_to_f16_bits(0.0);
        assert_eq!(
            cublasHgemm(
                handle,
                CublasOperationT::CUBLAS_OP_N,
                CublasOperationT::CUBLAS_OP_N,
                m as c_int,
                n as c_int,
                k as c_int,
                (&alpha16 as *const u16) as *const c_void,
                d_af16,
                m as c_int,
                d_bf16,
                k as c_int,
                (&beta16 as *const u16) as *const c_void,
                d_cf16,
                m as c_int,
            ),
            CublasStatusT::CUBLAS_STATUS_SUCCESS
        );
        let mut gpu_cf16 = vec![0u16; m * n];
        assert_eq!(
            cudaMemcpy(
                gpu_cf16.as_mut_ptr() as *mut c_void,
                d_cf16,
                gpu_cf16.len() * 2,
                CudaMemcpyKind::DeviceToHost
            ),
            CudaErrorT::CudaSuccess
        );
        for i in 0..cpu_f32.len() {
            let val = f16_bits_to_f32(gpu_cf16[i]);
            assert!((val - cpu_f32[i]).abs() < 1e-2);
        }

        // 2. cublasGemmEx FP16 -> FP32
        let alpha32 = 1.0f32;
        let beta32 = 0.0f32;
        assert_eq!(
            cublasGemmEx(
                handle,
                CublasOperationT::CUBLAS_OP_N,
                CublasOperationT::CUBLAS_OP_N,
                m as c_int,
                n as c_int,
                k as c_int,
                (&alpha32 as *const f32) as *const c_void,
                d_af16,
                CudaDataTypeT::CUDA_R_16F,
                m as c_int,
                d_bf16,
                CudaDataTypeT::CUDA_R_16F,
                k as c_int,
                (&beta32 as *const f32) as *const c_void,
                d_cf32,
                CudaDataTypeT::CUDA_R_32F,
                m as c_int,
                CublasComputeTypeT::CUBLAS_COMPUTE_32F_FAST_16F,
                CublasGemmAlgoT::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            ),
            CublasStatusT::CUBLAS_STATUS_SUCCESS
        );
        let mut gpu_cf32 = vec![0.0f32; m * n];
        assert_eq!(
            cudaMemcpy(
                gpu_cf32.as_mut_ptr() as *mut c_void,
                d_cf32,
                gpu_cf32.len() * 4,
                CudaMemcpyKind::DeviceToHost
            ),
            CudaErrorT::CudaSuccess
        );
        for i in 0..cpu_f32.len() {
            assert!((gpu_cf32[i] - cpu_f32[i]).abs() < 1e-3);
        }

        // 3. cublasGemmStridedBatchedEx BF16 -> BF16 & BF16 -> FP32
        assert_eq!(
            cudaMemcpy(
                d_af16,
                ha_bf16.as_ptr() as *const c_void,
                ha_bf16.len() * 2,
                CudaMemcpyKind::HostToDevice
            ),
            CudaErrorT::CudaSuccess
        );
        assert_eq!(
            cudaMemcpy(
                d_bf16,
                hb_bf16.as_ptr() as *const c_void,
                hb_bf16.len() * 2,
                CudaMemcpyKind::HostToDevice
            ),
            CudaErrorT::CudaSuccess
        );
        assert_eq!(
            cublasGemmStridedBatchedEx(
                handle,
                CublasOperationT::CUBLAS_OP_N,
                CublasOperationT::CUBLAS_OP_N,
                m as c_int,
                n as c_int,
                k as c_int,
                (&alpha32 as *const f32) as *const c_void,
                d_af16,
                CudaDataTypeT::CUDA_R_16BF,
                m as c_int,
                0,
                d_bf16,
                CudaDataTypeT::CUDA_R_16BF,
                k as c_int,
                0,
                (&beta32 as *const f32) as *const c_void,
                d_cf16,
                CudaDataTypeT::CUDA_R_16BF,
                m as c_int,
                0,
                1,
                CublasComputeTypeT::CUBLAS_COMPUTE_32F_FAST_16BF,
                CublasGemmAlgoT::CUBLAS_GEMM_DEFAULT,
            ),
            CublasStatusT::CUBLAS_STATUS_SUCCESS
        );
        assert_eq!(
            cudaMemcpy(
                gpu_cf16.as_mut_ptr() as *mut c_void,
                d_cf16,
                gpu_cf16.len() * 2,
                CudaMemcpyKind::DeviceToHost
            ),
            CudaErrorT::CudaSuccess
        );
        for i in 0..cpu_f32.len() {
            let val = bf16_bits_to_f32(gpu_cf16[i]);
            assert!((val - cpu_f32[i]).abs() < 2e-2);
        }

        assert_eq!(cudaFree(d_af16), CudaErrorT::CudaSuccess);
        assert_eq!(cudaFree(d_bf16), CudaErrorT::CudaSuccess);
        assert_eq!(cudaFree(d_cf16), CudaErrorT::CudaSuccess);
        assert_eq!(cudaFree(d_cf32), CudaErrorT::CudaSuccess);
        assert_eq!(cublasDestroy_v2(handle), CublasStatusT::CUBLAS_STATUS_SUCCESS);
    }
}
