//! C-ABI compatible CUDA Driver and Runtime types and status codes.

use core::ffi::{c_int, c_uint, c_void};

pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphNode = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type CudaStreamT = *mut c_void;
pub type CudaEventT = *mut c_void;

/// CUDA Driver API result code (`CUresult`).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum CUresult {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_UNKNOWN = 999,
}

/// CUDA Runtime API error code (`cudaError_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum CudaErrorT {
    CudaSuccess = 0,
    CudaErrorInvalidValue = 1,
    CudaErrorMemoryAllocation = 2,
    CudaErrorInitializationError = 3,
    CudaErrorInvalidDeviceFunction = 98,
    CudaErrorInvalidDevice = 101,
    CudaErrorNotReady = 600,
    CudaErrorLaunchFailure = 719,
    CudaErrorUnknown = 999,
}

/// CUDA device property record populated by `cudaGetDeviceProperties`.
#[allow(non_snake_case)]
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CudaDeviceProp {
    pub name: [core::ffi::c_char; 256],
    pub uuid: [u8; 16],
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: c_int,
    pub warpSize: c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: c_int,
    pub maxThreadsDim: [c_int; 3],
    pub maxGridSize: [c_int; 3],
    pub clockRate: c_int,
    pub totalConstMem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub multiProcessorCount: c_int,
    pub integrated: c_int,
    pub canMapHostMemory: c_int,
    pub computeMode: c_int,
    pub concurrentKernels: c_int,
    pub memoryClockRate: c_int,
    pub memoryBusWidth: c_int,
    pub l2CacheSize: c_int,
    pub maxThreadsPerMultiProcessor: c_int,
    pub unifiedAddressing: c_int,
    pub managedMemory: c_int,
    pub asyncEngineCount: c_int,
}

impl Default for CudaDeviceProp {
    fn default() -> Self {
        Self {
            name: [0; 256],
            uuid: [0; 16],
            totalGlobalMem: 0,
            sharedMemPerBlock: 0,
            regsPerBlock: 0,
            warpSize: 0,
            memPitch: 0,
            maxThreadsPerBlock: 0,
            maxThreadsDim: [0; 3],
            maxGridSize: [0; 3],
            clockRate: 0,
            totalConstMem: 0,
            major: 0,
            minor: 0,
            multiProcessorCount: 0,
            integrated: 0,
            canMapHostMemory: 0,
            computeMode: 0,
            concurrentKernels: 0,
            memoryClockRate: 0,
            memoryBusWidth: 0,
            l2CacheSize: 0,
            maxThreadsPerMultiProcessor: 0,
            unifiedAddressing: 0,
            managedMemory: 0,
            asyncEngineCount: 0,
        }
    }
}

/// CUDA 3D launch dimension (`dim3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Dim3 {
    pub x: c_uint,
    pub y: c_uint,
    pub z: c_uint,
}

/// NVIDIA `nvcc`-generated fatbin wrapper (`__fatBinC_Wrapper_t`, magic `0x466243B1`).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FatBinCWrapper {
    pub magic: c_uint,
    pub version: c_uint,
    pub data: *const c_void,
    pub filename_or_fatbins: *const c_void,
}

/// Memory transfer direction (`cudaMemcpyKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum CudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

/// `CUDA_KERNEL_NODE_PARAMS` for `cuGraphAddKernelNode`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CudaKernelNodeParams {
    pub func: CUfunction,
    pub grid_dim_x: c_uint,
    pub grid_dim_y: c_uint,
    pub grid_dim_z: c_uint,
    pub block_dim_x: c_uint,
    pub block_dim_y: c_uint,
    pub block_dim_z: c_uint,
    pub shared_mem_bytes: c_uint,
    pub kernel_params: *mut *mut c_void,
    pub extra: *mut *mut c_void,
}

pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: c_int = 2;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: c_int = 3;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: c_int = 4;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: c_int = 5;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: c_int = 6;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: c_int = 7;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
pub const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: c_int = 12;
pub const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: c_int = 13;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_INTEGRATED: c_int = 18;
pub const CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: c_int = 19;
pub const CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: c_int = 31;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: c_int = 36;
pub const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: c_int = 37;
pub const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: c_int = 38;
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: c_int = 39;
pub const CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT: c_int = 40;
pub const CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: c_int = 41;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: c_int = 81;
pub const CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY: c_int = 83;

pub const CU_POINTER_ATTRIBUTE_CONTEXT: c_int = 1;
pub const CU_POINTER_ATTRIBUTE_MEMORY_TYPE: c_int = 2;
pub const CU_POINTER_ATTRIBUTE_DEVICE_POINTER: c_int = 3;
pub const CU_POINTER_ATTRIBUTE_HOST_POINTER: c_int = 4;
pub const CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: c_int = 9;

pub const CU_MEMORYTYPE_HOST: c_int = 1;
pub const CU_MEMORYTYPE_DEVICE: c_int = 2;
pub const CU_MEMORYTYPE_ARRAY: c_int = 3;
pub const CU_MEMORYTYPE_UNIFIED: c_int = 4;
