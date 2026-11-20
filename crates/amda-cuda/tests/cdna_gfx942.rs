//! Integration test exercising `amda-cuda` under the CDNA3 (`gfx942`, Wave64) target profile.

use amda_cuda::{
    cuCtxCreate_v2, cuCtxDestroy_v2, cuDeviceGet, cuDeviceGetAttribute, cuDeviceGetName,
    cuDeviceTotalMem_v2, cuEventCreate, cuEventDestroy_v2, cuEventElapsedTime, cuEventRecord,
    cuEventSynchronize, cuGetProcAddress, cuInit, cuLaunchKernel, cuMemAlloc_v2, cuMemFree_v2,
    cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, cuModuleGetFunction, cuModuleLoadData, cuModuleUnload,
    cuStreamCreate, cuStreamDestroy_v2, cuStreamSynchronize, CUcontext, CUdevice, CUdeviceptr,
    CUevent, CUfunction, CUmodule, CUresult, CUstream, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE,
};
use core::ffi::{c_char, c_int, c_void, CStr};
use std::ffi::CString;

const SAXPY_PTX: &str = concat!(
    ".version 7.8\n",
    ".target sm_80\n",
    ".address_size 64\n",
    ".visible .entry saxpy(\n",
    "    .param .f32 a,\n",
    "    .param .u64 x,\n",
    "    .param .u64 y,\n",
    "    .param .u32 n\n",
    ") {\n",
    "    .reg .pred %p<2>;\n",
    "    .reg .f32 %f<5>;\n",
    "    .reg .b32 %r<6>;\n",
    "    .reg .b64 %rd<8>;\n",
    "    ld.param.f32 %f1, [a];\n",
    "    ld.param.u64 %rd1, [x];\n",
    "    ld.param.u64 %rd2, [y];\n",
    "    ld.param.u32 %r1, [n];\n",
    "    mov.u32 %r2, %ctaid.x;\n",
    "    mov.u32 %r3, %ntid.x;\n",
    "    mov.u32 %r4, %tid.x;\n",
    "    mad.lo.s32 %r5, %r2, %r3, %r4;\n",
    "    setp.ge.s32 %p1, %r5, %r1;\n",
    "    @%p1 bra $L_DONE;\n",
    "    cvta.to.global.u64 %rd3, %rd1;\n",
    "    cvta.to.global.u64 %rd4, %rd2;\n",
    "    mul.wide.s32 %rd5, %r5, 4;\n",
    "    add.s64 %rd6, %rd3, %rd5;\n",
    "    add.s64 %rd7, %rd4, %rd5;\n",
    "    ld.global.f32 %f2, [%rd6];\n",
    "    ld.global.f32 %f3, [%rd7];\n",
    "    fma.rn.f32 %f4, %f1, %f2, %f3;\n",
    "    st.global.f32 [%rd7], %f4;\n",
    "$L_DONE:\n",
    "    ret;\n",
    "}\n\0"
);

#[test]
fn cdna_gfx942_wave64_profile_and_events() {
    std::env::set_var("AMDA_BACKEND", "mock");
    std::env::set_var("AMDA_MOCK_ARCH", "gfx942");

    assert_eq!(cuInit(0), CUresult::CUDA_SUCCESS);

    let mut dev: CUdevice = -1;
    assert_eq!(cuDeviceGet(&mut dev, 0), CUresult::CUDA_SUCCESS);

    let mut name_buf = [0 as c_char; 128];
    assert_eq!(
        cuDeviceGetName(name_buf.as_mut_ptr(), 128, dev),
        CUresult::CUDA_SUCCESS
    );
    let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }.to_str().unwrap();
    assert!(name.contains("MI300X"));

    let mut warp_size: c_int = 0;
    assert_eq!(
        cuDeviceGetAttribute(&mut warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(warp_size, 64);

    let mut cus: c_int = 0;
    assert_eq!(
        cuDeviceGetAttribute(&mut cus, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(cus, 304);

    let mut total_mem: usize = 0;
    assert_eq!(cuDeviceTotalMem_v2(&mut total_mem, dev), CUresult::CUDA_SUCCESS);
    assert_eq!(total_mem, 192 * 1024 * 1024 * 1024);

    // Base symbol lookup via 4-arg cuGetProcAddress
    let sym = CString::new("cuMemAlloc").unwrap();
    let mut pfn: *mut c_void = core::ptr::null_mut();
    assert_eq!(
        cuGetProcAddress(sym.as_ptr(), &mut pfn, 12040, 0),
        CUresult::CUDA_SUCCESS
    );
    assert!(!pfn.is_null());

    let mut ctx: CUcontext = core::ptr::null_mut();
    assert_eq!(cuCtxCreate_v2(&mut ctx, 0, dev), CUresult::CUDA_SUCCESS);

    let mut stream: CUstream = core::ptr::null_mut();
    assert_eq!(cuStreamCreate(&mut stream, 0), CUresult::CUDA_SUCCESS);

    let mut ev_start: CUevent = core::ptr::null_mut();
    let mut ev_end: CUevent = core::ptr::null_mut();
    assert_eq!(cuEventCreate(&mut ev_start, 0), CUresult::CUDA_SUCCESS);
    assert_eq!(cuEventCreate(&mut ev_end, 0), CUresult::CUDA_SUCCESS);

    let mut module: CUmodule = core::ptr::null_mut();
    assert_eq!(
        cuModuleLoadData(&mut module, SAXPY_PTX.as_ptr() as *const c_void),
        CUresult::CUDA_SUCCESS
    );
    let fname = CString::new("saxpy").unwrap();
    let mut func: CUfunction = core::ptr::null_mut();
    assert_eq!(
        cuModuleGetFunction(&mut func, module, fname.as_ptr()),
        CUresult::CUDA_SUCCESS
    );

    let n: u32 = 512;
    let bytes = (n as usize) * core::mem::size_of::<f32>();
    let mut dx: CUdeviceptr = 0;
    let mut dy: CUdeviceptr = 0;
    assert_eq!(cuMemAlloc_v2(&mut dx, bytes), CUresult::CUDA_SUCCESS);
    assert_eq!(cuMemAlloc_v2(&mut dy, bytes), CUresult::CUDA_SUCCESS);

    let hx: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut hy: Vec<f32> = vec![10.0; n as usize];
    assert_eq!(
        cuMemcpyHtoD_v2(dx, hx.as_ptr() as *const c_void, bytes),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(
        cuMemcpyHtoD_v2(dy, hy.as_ptr() as *const c_void, bytes),
        CUresult::CUDA_SUCCESS
    );

    assert_eq!(cuEventRecord(ev_start, stream), CUresult::CUDA_SUCCESS);
    let mut alpha: f32 = 3.0;
    let mut n_arg = n;
    let mut params: [*mut c_void; 4] = [
        (&mut alpha as *mut f32) as *mut c_void,
        (&mut dx as *mut CUdeviceptr) as *mut c_void,
        (&mut dy as *mut CUdeviceptr) as *mut c_void,
        (&mut n_arg as *mut u32) as *mut c_void,
    ];
    assert_eq!(
        cuLaunchKernel(
            func,
            4,
            1,
            1,
            128,
            1,
            1,
            0,
            stream,
            params.as_mut_ptr(),
            core::ptr::null_mut(),
        ),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(cuEventRecord(ev_end, stream), CUresult::CUDA_SUCCESS);
    assert_eq!(cuEventSynchronize(ev_end), CUresult::CUDA_SUCCESS);
    assert_eq!(cuStreamSynchronize(stream), CUresult::CUDA_SUCCESS);

    let mut elapsed_ms = -1.0f32;
    assert_eq!(
        cuEventElapsedTime(&mut elapsed_ms, ev_start, ev_end),
        CUresult::CUDA_SUCCESS
    );
    assert!(elapsed_ms >= 0.0);

    assert_eq!(
        cuMemcpyDtoH_v2(hy.as_mut_ptr() as *mut c_void, dy, bytes),
        CUresult::CUDA_SUCCESS
    );
    for (i, &val) in hy.iter().enumerate() {
        assert_eq!(val, 10.0 + 3.0 * (i as f32));
    }

    assert_eq!(cuEventDestroy_v2(ev_start), CUresult::CUDA_SUCCESS);
    assert_eq!(cuEventDestroy_v2(ev_end), CUresult::CUDA_SUCCESS);
    assert_eq!(cuStreamDestroy_v2(stream), CUresult::CUDA_SUCCESS);
    assert_eq!(cuMemFree_v2(dx), CUresult::CUDA_SUCCESS);
    assert_eq!(cuMemFree_v2(dy), CUresult::CUDA_SUCCESS);
    assert_eq!(cuModuleUnload(module), CUresult::CUDA_SUCCESS);
    assert_eq!(cuCtxDestroy_v2(ctx), CUresult::CUDA_SUCCESS);
}
