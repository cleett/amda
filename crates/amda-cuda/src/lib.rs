//! Drop-in CUDA Driver & Runtime C ABI library backed by AMDA.

pub mod driver;
pub mod runtime;
pub mod types;

pub use driver::*;
pub use runtime::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use core::ffi::{c_char, c_int, c_uint, c_void, CStr};
    use std::ffi::CString;

    const VEC_ADD_PTX: &str = concat!(
        ".version 7.8\n",
        ".target sm_80\n",
        ".address_size 64\n",
        ".visible .entry vector_add(\n",
        "    .param .u64 a,\n",
        "    .param .u64 b,\n",
        "    .param .u64 c,\n",
        "    .param .u32 n\n",
        ") {\n",
        "    .reg .pred %p<2>;\n",
        "    .reg .f32 %f<4>;\n",
        "    .reg .b32 %r<6>;\n",
        "    .reg .b64 %rd<10>;\n",
        "    ld.param.u64 %rd1, [a];\n",
        "    ld.param.u64 %rd2, [b];\n",
        "    ld.param.u64 %rd3, [c];\n",
        "    ld.param.u32 %r1, [n];\n",
        "    mov.u32 %r2, %ctaid.x;\n",
        "    mov.u32 %r3, %ntid.x;\n",
        "    mov.u32 %r4, %tid.x;\n",
        "    mad.lo.s32 %r5, %r2, %r3, %r4;\n",
        "    setp.ge.s32 %p1, %r5, %r1;\n",
        "    @%p1 bra $L_EXIT;\n",
        "    cvta.to.global.u64 %rd4, %rd1;\n",
        "    cvta.to.global.u64 %rd5, %rd2;\n",
        "    cvta.to.global.u64 %rd6, %rd3;\n",
        "    mul.wide.s32 %rd7, %r5, 4;\n",
        "    add.s64 %rd8, %rd4, %rd7;\n",
        "    add.s64 %rd9, %rd5, %rd7;\n",
        "    ld.global.f32 %f1, [%rd8];\n",
        "    ld.global.f32 %f2, [%rd9];\n",
        "    add.f32 %f3, %f1, %f2;\n",
        "    add.s64 %rd8, %rd6, %rd7;\n",
        "    st.global.f32 [%rd8], %f3;\n",
        "$L_EXIT:\n",
        "    ret;\n",
        "}\n\0"
    );

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
    fn cuda_driver_and_runtime_end_to_end() {
        assert_eq!(cuInit(0), CUresult::CUDA_SUCCESS);

        let mut ver: c_int = 0;
        assert_eq!(cuDriverGetVersion(&mut ver), CUresult::CUDA_SUCCESS);
        assert_eq!(ver, 12040);

        let mut count: c_int = 0;
        assert_eq!(cuDeviceGetCount(&mut count), CUresult::CUDA_SUCCESS);
        assert!(count >= 1);

        let mut dev: CUdevice = -1;
        assert_eq!(cuDeviceGet(&mut dev, 0), CUresult::CUDA_SUCCESS);

        let mut name_buf = [0 as c_char; 128];
        assert_eq!(
            cuDeviceGetName(name_buf.as_mut_ptr(), 128, dev),
            CUresult::CUDA_SUCCESS
        );
        let dev_name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }.to_str().unwrap();
        assert!(dev_name.contains("AMD"));

        let mut ctx: CUcontext = core::ptr::null_mut();
        assert_eq!(cuCtxCreate_v2(&mut ctx, 0, dev), CUresult::CUDA_SUCCESS);

        // Dynamic symbol resolution via cuGetProcAddress_v2
        let sym = CString::new("cuLaunchKernel").unwrap();
        let mut pfn: *mut c_void = core::ptr::null_mut();
        let mut status: c_int = -1;
        assert_eq!(
            cuGetProcAddress_v2(sym.as_ptr(), &mut pfn, 12040, 0, &mut status),
            CUresult::CUDA_SUCCESS
        );
        assert!(!pfn.is_null());
        assert_eq!(status, 0);

        type LaunchFn = extern "C" fn(
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
        ) -> CUresult;
        let launch_via_proc: LaunchFn = unsafe { core::mem::transmute(pfn) };

        // 1. Vector add (1024 elements across 4 blocks x 256 threads)
        let mut mod_vadd: CUmodule = core::ptr::null_mut();
        assert_eq!(
            cuModuleLoadData(&mut mod_vadd, VEC_ADD_PTX.as_ptr() as *const c_void),
            CUresult::CUDA_SUCCESS
        );
        let fn_name = CString::new("vector_add").unwrap();
        let mut fn_vadd: CUfunction = core::ptr::null_mut();
        assert_eq!(
            cuModuleGetFunction(&mut fn_vadd, mod_vadd, fn_name.as_ptr()),
            CUresult::CUDA_SUCCESS
        );

        let n: u32 = 1024;
        let byte_len = (n as usize) * core::mem::size_of::<f32>();
        let mut da: CUdeviceptr = 0;
        let mut db: CUdeviceptr = 0;
        let mut dc: CUdeviceptr = 0;
        assert_eq!(cuMemAlloc_v2(&mut da, byte_len), CUresult::CUDA_SUCCESS);
        assert_eq!(cuMemAlloc_v2(&mut db, byte_len), CUresult::CUDA_SUCCESS);
        assert_eq!(cuMemAlloc_v2(&mut dc, byte_len), CUresult::CUDA_SUCCESS);

        let ha: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let hb: Vec<f32> = (0..n).map(|i| (i * 3) as f32).collect();
        let mut hc = vec![0.0f32; n as usize];

        assert_eq!(
            cuMemcpyHtoD_v2(da, ha.as_ptr() as *const c_void, byte_len),
            CUresult::CUDA_SUCCESS
        );
        assert_eq!(
            cuMemcpyHtoD_v2(db, hb.as_ptr() as *const c_void, byte_len),
            CUresult::CUDA_SUCCESS
        );

        let mut arg_n = n;
        let mut params: [*mut c_void; 4] = [
            (&mut da as *mut CUdeviceptr) as *mut c_void,
            (&mut db as *mut CUdeviceptr) as *mut c_void,
            (&mut dc as *mut CUdeviceptr) as *mut c_void,
            (&mut arg_n as *mut u32) as *mut c_void,
        ];
        assert_eq!(
            launch_via_proc(
                fn_vadd,
                4,
                1,
                1,
                256,
                1,
                1,
                0,
                core::ptr::null_mut(),
                params.as_mut_ptr(),
                core::ptr::null_mut()
            ),
            CUresult::CUDA_SUCCESS
        );
        assert_eq!(
            cuMemcpyDtoH_v2(hc.as_mut_ptr() as *mut c_void, dc, byte_len),
            CUresult::CUDA_SUCCESS
        );
        for (i, &val) in hc.iter().enumerate().take(n as usize) {
            assert_eq!(val, (i * 4) as f32);
        }

        // 2. SAXPY + CUDA Graph replay (executed twice: y = y + 2*x twice => y0 + 4*x)
        let mut mod_saxpy: CUmodule = core::ptr::null_mut();
        assert_eq!(
            cuModuleLoadData(&mut mod_saxpy, SAXPY_PTX.as_ptr() as *const c_void),
            CUresult::CUDA_SUCCESS
        );
        let saxpy_name = CString::new("saxpy").unwrap();
        let mut fn_saxpy: CUfunction = core::ptr::null_mut();
        assert_eq!(
            cuModuleGetFunction(&mut fn_saxpy, mod_saxpy, saxpy_name.as_ptr()),
            CUresult::CUDA_SUCCESS
        );

        let mut alpha: f32 = 2.0;
        let mut saxpy_params: [*mut c_void; 4] = [
            (&mut alpha as *mut f32) as *mut c_void,
            (&mut da as *mut CUdeviceptr) as *mut c_void,
            (&mut dc as *mut CUdeviceptr) as *mut c_void,
            (&mut arg_n as *mut u32) as *mut c_void,
        ];

        let mut graph: CUgraph = core::ptr::null_mut();
        assert_eq!(cuGraphCreate(&mut graph, 0), CUresult::CUDA_SUCCESS);
        let mut node: CUgraphNode = core::ptr::null_mut();
        let node_params = CudaKernelNodeParams {
            func: fn_saxpy,
            grid_dim_x: 4,
            grid_dim_y: 1,
            grid_dim_z: 1,
            block_dim_x: 256,
            block_dim_y: 1,
            block_dim_z: 1,
            shared_mem_bytes: 0,
            kernel_params: saxpy_params.as_mut_ptr(),
            extra: core::ptr::null_mut(),
        };
        assert_eq!(
            cuGraphAddKernelNode(&mut node, graph, core::ptr::null(), 0, &node_params),
            CUresult::CUDA_SUCCESS
        );
        let mut graph_exec: CUgraphExec = core::ptr::null_mut();
        assert_eq!(
            cuGraphInstantiate(&mut graph_exec, graph, 0),
            CUresult::CUDA_SUCCESS
        );
        assert_eq!(
            cuGraphLaunch(graph_exec, core::ptr::null_mut()),
            CUresult::CUDA_SUCCESS
        );
        assert_eq!(
            cuGraphLaunch(graph_exec, core::ptr::null_mut()),
            CUresult::CUDA_SUCCESS
        );
        assert_eq!(cuCtxSynchronize(), CUresult::CUDA_SUCCESS);
        assert_eq!(
            cuMemcpyDtoH_v2(hc.as_mut_ptr() as *mut c_void, dc, byte_len),
            CUresult::CUDA_SUCCESS
        );
        for (i, &val) in hc.iter().enumerate().take(n as usize) {
            assert_eq!(val, (i * 8) as f32);
        }

        assert_eq!(cuGraphExecDestroy(graph_exec), CUresult::CUDA_SUCCESS);
        assert_eq!(cuGraphDestroy(graph), CUresult::CUDA_SUCCESS);
        assert_eq!(cuModuleUnload(mod_saxpy), CUresult::CUDA_SUCCESS);
        assert_eq!(cuModuleUnload(mod_vadd), CUresult::CUDA_SUCCESS);
        assert_eq!(cuMemFree_v2(da), CUresult::CUDA_SUCCESS);
        assert_eq!(cuMemFree_v2(db), CUresult::CUDA_SUCCESS);
        assert_eq!(cuMemFree_v2(dc), CUresult::CUDA_SUCCESS);

        // 3. Runtime API sanity check
        let mut rt_ptr: *mut c_void = core::ptr::null_mut();
        assert_eq!(cudaMalloc(&mut rt_ptr, 64), CudaErrorT::CudaSuccess);
        let src_vals = [7u32; 16];
        let mut dst_vals = [0u32; 16];
        assert_eq!(
            cudaMemcpy(
                rt_ptr,
                src_vals.as_ptr() as *const c_void,
                64,
                CudaMemcpyKind::HostToDevice
            ),
            CudaErrorT::CudaSuccess
        );
        assert_eq!(
            cudaMemcpy(
                dst_vals.as_mut_ptr() as *mut c_void,
                rt_ptr,
                64,
                CudaMemcpyKind::DeviceToHost
            ),
            CudaErrorT::CudaSuccess
        );
        assert_eq!(dst_vals, src_vals);
        assert_eq!(cudaFree(rt_ptr), CudaErrorT::CudaSuccess);
        assert_eq!(cudaGetLastError(), CudaErrorT::CudaSuccess);

        assert_eq!(cuCtxDestroy_v2(ctx), CUresult::CUDA_SUCCESS);
    }
}
