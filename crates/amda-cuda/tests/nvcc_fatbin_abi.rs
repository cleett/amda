use amda_compiler::build_lz4_ptx_fatbin;
use amda_compiler::fatbin::FATBIN_WRAPPER_MAGIC;
use amda_cuda::{
    __cudaPopCallConfiguration, __cudaPushCallConfiguration, __cudaRegisterFatBinary,
    __cudaRegisterFatBinaryEnd, __cudaRegisterFunction, __cudaRegisterVar,
    __cudaUnregisterFatBinary, cudaDeviceSynchronize, cudaFree, cudaLaunchKernel, cudaMalloc,
    cudaMemGetInfo, cudaMemcpy, cudaMemcpyAsync, cudaMemset, cudaMemsetAsync, cudaStreamCreate,
    cudaStreamDestroy, cudaStreamSynchronize, CudaErrorT, CudaMemcpyKind, CudaStreamT, Dim3,
    FatBinCWrapper,
};
use core::ffi::{c_void, CStr};
use std::ffi::CString;

const VEC_ADD_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64

    .visible .entry vector_add(
        .param .u64 a,
        .param .u64 b,
        .param .u64 c,
        .param .u32 n
    ) {
        .reg .pred %p<2>;
        .reg .f32 %f<4>;
        .reg .b32 %r<6>;
        .reg .b64 %rd<10>;

        ld.param.u64 %rd1, [a];
        ld.param.u64 %rd2, [b];
        ld.param.u64 %rd3, [c];
        ld.param.u32 %r1, [n];

        mov.u32 %r2, %ctaid.x;
        mov.u32 %r3, %ntid.x;
        mov.u32 %r4, %tid.x;
        mad.lo.s32 %r5, %r2, %r3, %r4;
        setp.ge.s32 %p1, %r5, %r1;
        @%p1 bra $L_EXIT;

        cvta.to.global.u64 %rd4, %rd1;
        cvta.to.global.u64 %rd5, %rd2;
        cvta.to.global.u64 %rd6, %rd3;
        mul.wide.s32 %rd7, %r5, 4;
        add.s64 %rd8, %rd4, %rd7;
        add.s64 %rd9, %rd5, %rd7;
        ld.global.f32 %f1, [%rd8];
        ld.global.f32 %f2, [%rd9];
        add.f32 %f3, %f1, %f2;
        add.s64 %rd8, %rd6, %rd7;
        st.global.f32 [%rd8], %f3;

    $L_EXIT:
        ret;
    }
"#;

extern "C" fn host_vector_add_stub() {}
extern "C" fn host_unregistered_stub() {}
static mut HOST_GLOBAL_VAR: u64 = 0;

#[test]
fn nvcc_lz4_fatbin_registration_and_launch_end_to_end() {
    // 1. Build LZ4-compressed fatbin inside __fatBinC_Wrapper_t
    let fatbin = build_lz4_ptx_fatbin(VEC_ADD_PTX);
    let wrapper = FatBinCWrapper {
        magic: FATBIN_WRAPPER_MAGIC,
        version: 1,
        data: fatbin.as_ptr() as *const c_void,
        filename_or_fatbins: core::ptr::null(),
    };

    // Unregistered function launch must fail cleanly
    let dummy_dim = Dim3 { x: 1, y: 1, z: 1 };
    assert_eq!(
        cudaLaunchKernel(
            host_unregistered_stub as *const c_void,
            dummy_dim,
            dummy_dim,
            core::ptr::null_mut(),
            0,
            core::ptr::null_mut(),
        ),
        CudaErrorT::CudaErrorInvalidDeviceFunction
    );

    // 2. Host static constructor registration sequence (prior to explicit runtime calls)
    let handle = __cudaRegisterFatBinary((&wrapper as *const FatBinCWrapper) as *mut c_void);
    assert!(!handle.is_null());

    let kname = CString::new("vector_add").unwrap();
    let vname = CString::new("device_const_tag").unwrap();
    __cudaRegisterFunction(
        handle,
        host_vector_add_stub as *const core::ffi::c_char,
        kname.as_ptr() as *mut core::ffi::c_char,
        kname.as_ptr(),
        -1,
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    );
    __cudaRegisterVar(
        handle,
        core::ptr::addr_of_mut!(HOST_GLOBAL_VAR) as *mut core::ffi::c_char,
        core::ptr::null_mut(),
        vname.as_ptr(),
        0,
        core::mem::size_of::<u64>(),
        1,
        0,
    );
    __cudaRegisterFatBinaryEnd(handle);

    // 3. Runtime allocation, memset, async memcpy, and memory info
    const N: usize = 2048;
    let bytes = N * core::mem::size_of::<f32>();

    let mut da: *mut c_void = core::ptr::null_mut();
    let mut db: *mut c_void = core::ptr::null_mut();
    let mut dc: *mut c_void = core::ptr::null_mut();
    assert_eq!(cudaMalloc(&mut da, bytes), CudaErrorT::CudaSuccess);
    assert_eq!(cudaMalloc(&mut db, bytes), CudaErrorT::CudaSuccess);
    assert_eq!(cudaMalloc(&mut dc, bytes), CudaErrorT::CudaSuccess);

    let mut free_bytes: usize = 0;
    let mut total_bytes: usize = 0;
    assert_eq!(
        cudaMemGetInfo(&mut free_bytes, &mut total_bytes),
        CudaErrorT::CudaSuccess
    );
    assert!(total_bytes > 0);
    assert!(free_bytes <= total_bytes);

    let mut stream: CudaStreamT = core::ptr::null_mut();
    assert_eq!(cudaStreamCreate(&mut stream), CudaErrorT::CudaSuccess);

    // Fill dc with 0xAA synchronously, verify via DtoH, then zero asynchronously before launch
    assert_eq!(cudaMemset(dc, 0xAA, bytes), CudaErrorT::CudaSuccess);
    let mut check_bytes = vec![0u8; bytes];
    assert_eq!(
        cudaMemcpy(
            check_bytes.as_mut_ptr() as *mut c_void,
            dc,
            bytes,
            CudaMemcpyKind::DeviceToHost,
        ),
        CudaErrorT::CudaSuccess
    );
    assert!(check_bytes.iter().all(|&b| b == 0xAA));
    assert_eq!(
        cudaMemsetAsync(dc, 0, bytes, stream),
        CudaErrorT::CudaSuccess
    );

    let ha: Vec<f32> = (0..N).map(|i| (i as f32) * 1.25).collect();
    let hb: Vec<f32> = (0..N).map(|i| (i as f32) * 0.75 + 2.0).collect();
    let mut hc = vec![0.0f32; N];

    assert_eq!(
        cudaMemcpyAsync(
            da,
            ha.as_ptr() as *const c_void,
            bytes,
            CudaMemcpyKind::HostToDevice,
            stream,
        ),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(
        cudaMemcpyAsync(
            db,
            hb.as_ptr() as *const c_void,
            bytes,
            CudaMemcpyKind::HostToDevice,
            stream,
        ),
        CudaErrorT::CudaSuccess
    );

    // 4. <<<grid, block>>> call-configuration push/pop + cudaLaunchKernel
    let push_grid = Dim3 {
        x: (N as u32) / 256,
        y: 1,
        z: 1,
    };
    let push_block = Dim3 { x: 256, y: 1, z: 1 };
    assert_eq!(
        __cudaPushCallConfiguration(push_grid, push_block, 0, stream),
        0
    );

    let mut grid = Dim3 { x: 0, y: 0, z: 0 };
    let mut block = Dim3 { x: 0, y: 0, z: 0 };
    let mut shared_mem = usize::MAX;
    let mut popped_stream: CudaStreamT = core::ptr::null_mut();
    assert_eq!(
        __cudaPopCallConfiguration(&mut grid, &mut block, &mut shared_mem, &mut popped_stream),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(grid, push_grid);
    assert_eq!(block, push_block);
    assert_eq!(shared_mem, 0);
    assert_eq!(popped_stream, stream);

    let mut n_arg: u32 = N as u32;
    let mut params: [*mut c_void; 4] = [
        (&mut da as *mut *mut c_void) as *mut c_void,
        (&mut db as *mut *mut c_void) as *mut c_void,
        (&mut dc as *mut *mut c_void) as *mut c_void,
        (&mut n_arg as *mut u32) as *mut c_void,
    ];
    assert_eq!(
        cudaLaunchKernel(
            host_vector_add_stub as *const c_void,
            grid,
            block,
            params.as_mut_ptr(),
            shared_mem,
            popped_stream,
        ),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(cudaStreamSynchronize(stream), CudaErrorT::CudaSuccess);
    assert_eq!(cudaDeviceSynchronize(), CudaErrorT::CudaSuccess);

    assert_eq!(
        cudaMemcpy(
            hc.as_mut_ptr() as *mut c_void,
            dc,
            bytes,
            CudaMemcpyKind::DeviceToHost,
        ),
        CudaErrorT::CudaSuccess
    );
    for i in 0..N {
        let expected = ha[i] + hb[i];
        assert!(
            (hc[i] - expected).abs() < 1e-5,
            "mismatch at {i}: got {}, expected {expected}",
            hc[i]
        );
    }

    assert_eq!(cudaFree(da), CudaErrorT::CudaSuccess);
    assert_eq!(cudaFree(db), CudaErrorT::CudaSuccess);
    assert_eq!(cudaFree(dc), CudaErrorT::CudaSuccess);
    assert_eq!(cudaStreamDestroy(stream), CudaErrorT::CudaSuccess);

    // 5. Unregister fatbin and confirm subsequent launch fails with invalid device function
    __cudaUnregisterFatBinary(handle);
    assert_eq!(
        cudaLaunchKernel(
            host_vector_add_stub as *const c_void,
            grid,
            block,
            params.as_mut_ptr(),
            0,
            core::ptr::null_mut(),
        ),
        CudaErrorT::CudaErrorInvalidDeviceFunction
    );

    let err_str = unsafe { CStr::from_ptr(amda_cuda::cudaGetErrorString(CudaErrorT::CudaErrorInvalidDeviceFunction)) }
        .to_str()
        .unwrap();
    assert_eq!(err_str, "invalid device function");
}
