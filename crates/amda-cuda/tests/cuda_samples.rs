use amda_cuda::{
    cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxGetCurrent, cuCtxPopCurrent_v2, cuCtxPushCurrent_v2,
    cuDevicePrimaryCtxRelease_v2, cuDevicePrimaryCtxRetain, cuEventCreate, cuEventDestroy_v2,
    cuEventElapsedTime, cuEventQuery, cuEventRecord, cuEventSynchronize, cuGetProcAddress_v2,
    cuInit, cuMemAlloc_v2, cuMemFree_v2, cuMemGetInfo_v2, cuMemcpyDtoDAsync_v2,
    cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2, cuMemsetD32_v2, cuMemsetD8_v2,
    cuPointerGetAttribute, cuPointerGetAttributes, cuStreamCreate, cuStreamDestroy_v2,
    cuStreamSynchronize, cuStreamWaitEvent, cudaDeviceGetAttribute, cudaDeviceReset,
    cudaDeviceSynchronize, cudaDriverGetVersion, cudaEventCreate, cudaEventCreateWithFlags,
    cudaEventDestroy, cudaEventElapsedTime, cudaEventQuery, cudaEventRecord,
    cudaEventSynchronize, cudaFree, cudaFreeHost, cudaGetDevice, cudaGetDeviceCount,
    cudaGetDeviceProperties, cudaHostAlloc, cudaLaunchKernel, cudaMalloc, cudaMallocHost,
    cudaMemGetInfo, cudaMemcpy, cudaMemcpyAsync, cudaOccupancyMaxActiveBlocksPerMultiprocessor,
    cudaOccupancyMaxPotentialBlockSize, cudaRuntimeGetVersion, cudaSetDevice, cudaStreamCreate,
    cudaStreamDestroy, cudaStreamSynchronize, cudaStreamWaitEvent, __cudaPopCallConfiguration,
    __cudaPushCallConfiguration, __cudaRegisterFatBinary, __cudaRegisterFatBinaryEnd,
    __cudaRegisterFunction, __cudaUnregisterFatBinary, CUcontext, CUdeviceptr, CUevent, CUresult,
    CUstream, CudaDeviceProp, CudaErrorT, CudaEventT, CudaMemcpyKind, CudaStreamT, Dim3,
    FatBinCWrapper, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
    CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST, CU_POINTER_ATTRIBUTE_CONTEXT,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
    CU_POINTER_ATTRIBUTE_HOST_POINTER, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
};
use amda_compiler::fatbin::FATBIN_WRAPPER_MAGIC;
use core::ffi::{c_int, c_uint, c_void, CStr};
use std::ffi::CString;
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn launch_ptx_kernel(
    ptx: &str,
    kernel_name: &str,
    grid: Dim3,
    block: Dim3,
    shared_mem: usize,
    args: &mut [*mut c_void],
) {
    let mut ptx_bytes = ptx.as_bytes().to_vec();
    ptx_bytes.push(0);
    let wrapper = FatBinCWrapper {
        magic: FATBIN_WRAPPER_MAGIC,
        version: 1,
        data: ptx_bytes.as_ptr() as *const c_void,
        filename_or_fatbins: core::ptr::null(),
    };
    let handle =
        __cudaRegisterFatBinary((&wrapper as *const FatBinCWrapper) as *mut c_void);
    assert!(!handle.is_null(), "fatbin registration must succeed");

    let host_fn = Box::into_raw(Box::new(0u8)) as *const core::ffi::c_char;
    let c_name = CString::new(kernel_name).expect("valid CString");
    __cudaRegisterFunction(
        handle,
        host_fn,
        c_name.as_ptr() as *mut core::ffi::c_char,
        c_name.as_ptr(),
        -1,
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
    );
    __cudaRegisterFatBinaryEnd(handle);

    assert_eq!(
        __cudaPushCallConfiguration(grid, block, shared_mem, core::ptr::null_mut()),
        0
    );
    let mut popped_grid = Dim3 { x: 0, y: 0, z: 0 };
    let mut popped_block = Dim3 { x: 0, y: 0, z: 0 };
    let mut popped_shared = 0usize;
    let mut popped_stream: CudaStreamT = core::ptr::null_mut();
    assert_eq!(
        __cudaPopCallConfiguration(
            &mut popped_grid,
            &mut popped_block,
            &mut popped_shared,
            &mut popped_stream
        ),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(
        cudaLaunchKernel(
            host_fn as *const c_void,
            popped_grid,
            popped_block,
            args.as_mut_ptr(),
            popped_shared,
            popped_stream,
        ),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(cudaDeviceSynchronize(), CudaErrorT::CudaSuccess);

    __cudaUnregisterFatBinary(handle);
    unsafe {
        drop(Box::from_raw(host_fn as *mut u8));
    }
}

const MATRIX_MUL_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64

    .visible .entry matrixMul(
        .param .u64 A_ptr,
        .param .u64 B_ptr,
        .param .u64 C_ptr,
        .param .u32 N
    )
    {
        .shared .align 4 .b8 As[1024];
        .shared .align 4 .b8 Bs[1024];
        .reg .pred %p<3>;
        .reg .f32 %f<6>;
        .reg .b32 %r<22>;
        .reg .b64 %rd<10>;

        ld.param.u64 %rd1, [A_ptr];
        ld.param.u64 %rd2, [B_ptr];
        ld.param.u64 %rd3, [C_ptr];
        ld.param.u32 %r1, [N];

        mov.u32 %r2, %ctaid.x;
        mov.u32 %r3, %ctaid.y;
        mov.u32 %r4, %tid.x;
        mov.u32 %r5, %tid.y;

        mul.lo.u32 %r6, %r3, 16;
        add.u32 %r6, %r6, %r5;
        mul.lo.u32 %r7, %r2, 16;
        add.u32 %r7, %r7, %r4;

        mov.f32 %f1, 0f00000000;
        mov.u32 %r8, 0;
        shr.u32 %r9, %r1, 4;

    $L_TILE_LOOP:
        setp.ge.u32 %p1, %r8, %r9;
        @%p1 bra $L_WRITE;

        mul.lo.u32 %r10, %r8, 16;
        add.u32 %r11, %r10, %r4;
        mad.lo.u32 %r12, %r6, %r1, %r11;
        cvta.to.global.u64 %rd4, %rd1;
        mul.wide.u32 %rd5, %r12, 4;
        add.u64 %rd6, %rd4, %rd5;
        ld.global.f32 %f2, [%rd6];

        add.u32 %r13, %r10, %r5;
        mad.lo.u32 %r14, %r13, %r1, %r7;
        cvta.to.global.u64 %rd4, %rd2;
        mul.wide.u32 %rd5, %r14, 4;
        add.u64 %rd6, %rd4, %rd5;
        ld.global.f32 %f3, [%rd6];

        mad.lo.u32 %r15, %r5, 16, %r4;
        mul.lo.u32 %r15, %r15, 4;
        mov.u32 %r16, As;
        add.u32 %r16, %r16, %r15;
        st.shared.f32 [%r16], %f2;
        mov.u32 %r17, Bs;
        add.u32 %r17, %r17, %r15;
        st.shared.f32 [%r17], %f3;
        bar.sync 0;

        mov.u32 %r18, 0;
    $L_K_LOOP:
        setp.ge.u32 %p2, %r18, 16;
        @%p2 bra $L_NEXT_TILE;

        mad.lo.u32 %r19, %r5, 16, %r18;
        mul.lo.u32 %r19, %r19, 4;
        mov.u32 %r16, As;
        add.u32 %r16, %r16, %r19;
        ld.shared.f32 %f4, [%r16];

        mad.lo.u32 %r20, %r18, 16, %r4;
        mul.lo.u32 %r20, %r20, 4;
        mov.u32 %r17, Bs;
        add.u32 %r17, %r17, %r20;
        ld.shared.f32 %f5, [%r17];

        fma.rn.f32 %f1, %f4, %f5, %f1;
        add.u32 %r18, %r18, 1;
        bra $L_K_LOOP;

    $L_NEXT_TILE:
        bar.sync 0;
        add.u32 %r8, %r8, 1;
        bra $L_TILE_LOOP;

    $L_WRITE:
        mad.lo.u32 %r21, %r6, %r1, %r7;
        cvta.to.global.u64 %rd7, %rd3;
        mul.wide.u32 %rd8, %r21, 4;
        add.u64 %rd9, %rd7, %rd8;
        st.global.f32 [%rd9], %f1;
        ret;
    }
"#;

#[test]
fn sample_matrix_mul_tiled_shared() {
    let _guard = TEST_LOCK.lock().expect("test lock");
    assert_eq!(cudaSetDevice(0), CudaErrorT::CudaSuccess);

    let n = 32usize;
    let count = n * n;
    let bytes = count * core::mem::size_of::<f32>();

    let mut h_a = vec![0.0f32; count];
    let mut h_b = vec![0.0f32; count];
    for r in 0..n {
        for c in 0..n {
            h_a[r * n + c] = ((r * 3 + c) % 11) as f32 * 0.25 - 1.0;
            h_b[r * n + c] = ((r + c * 5) % 13) as f32 * 0.125 - 0.5;
        }
    }

    let mut cpu_c = vec![0.0f32; count];
    for r in 0..n {
        for c in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum = h_a[r * n + k].mul_add(h_b[k * n + c], sum);
            }
            cpu_c[r * n + c] = sum;
        }
    }

    let mut d_a: *mut c_void = core::ptr::null_mut();
    let mut d_b: *mut c_void = core::ptr::null_mut();
    let mut d_c: *mut c_void = core::ptr::null_mut();
    assert_eq!(cudaMalloc(&mut d_a, bytes), CudaErrorT::CudaSuccess);
    assert_eq!(cudaMalloc(&mut d_b, bytes), CudaErrorT::CudaSuccess);
    assert_eq!(cudaMalloc(&mut d_c, bytes), CudaErrorT::CudaSuccess);

    assert_eq!(
        cudaMemcpy(d_a, h_a.as_ptr() as *const c_void, bytes, CudaMemcpyKind::HostToDevice),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(
        cudaMemcpy(d_b, h_b.as_ptr() as *const c_void, bytes, CudaMemcpyKind::HostToDevice),
        CudaErrorT::CudaSuccess
    );

    let mut arg_a = d_a as u64;
    let mut arg_b = d_b as u64;
    let mut arg_c = d_c as u64;
    let mut arg_n = n as u32;
    let mut params = [
        (&mut arg_a as *mut u64) as *mut c_void,
        (&mut arg_b as *mut u64) as *mut c_void,
        (&mut arg_c as *mut u64) as *mut c_void,
        (&mut arg_n as *mut u32) as *mut c_void,
    ];

    launch_ptx_kernel(
        MATRIX_MUL_PTX,
        "matrixMul",
        Dim3 { x: 2, y: 2, z: 1 },
        Dim3 { x: 16, y: 16, z: 1 },
        2048,
        &mut params,
    );

    let mut h_c = vec![0.0f32; count];
    assert_eq!(
        cudaMemcpy(h_c.as_mut_ptr() as *mut c_void, d_c, bytes, CudaMemcpyKind::DeviceToHost),
        CudaErrorT::CudaSuccess
    );

    for i in 0..count {
        let diff = (h_c[i] - cpu_c[i]).abs();
        assert!(
            diff < 1e-4,
            "matrixMul mismatch at idx {i}: gpu={} cpu={}",
            h_c[i],
            cpu_c[i]
        );
    }

    assert_eq!(cudaFree(d_a), CudaErrorT::CudaSuccess);
    assert_eq!(cudaFree(d_b), CudaErrorT::CudaSuccess);
    assert_eq!(cudaFree(d_c), CudaErrorT::CudaSuccess);
}

const REDUCTION_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64

    .visible .entry reduction_sum(
        .param .u64 in_ptr,
        .param .u64 out_ptr,
        .param .u32 n
    )
    {
        .shared .align 4 .b8 warp_sums[16];
        .reg .pred %p<2>;
        .reg .f32 %f<8>;
        .reg .b32 %r<12>;
        .reg .b64 %rd<6>;

        ld.param.u64 %rd1, [in_ptr];
        ld.param.u64 %rd2, [out_ptr];
        ld.param.u32 %r1, [n];

        mov.u32 %r2, %ctaid.x;
        mov.u32 %r3, %ntid.x;
        mov.u32 %r4, %tid.x;
        mad.lo.u32 %r5, %r2, %r3, %r4;

        mov.f32 %f1, 0f00000000;
        setp.lt.u32 %p1, %r5, %r1;
        @!%p1 bra $L_SHFL;

        cvta.to.global.u64 %rd3, %rd1;
        mul.wide.u32 %rd4, %r5, 4;
        add.u64 %rd5, %rd3, %rd4;
        ld.global.f32 %f1, [%rd5];

    $L_SHFL:
        shfl.sync.down.b32 %f2, %f1, 16, 31, 0xffffffff;
        add.f32 %f1, %f1, %f2;
        shfl.sync.down.b32 %f2, %f1, 8, 31, 0xffffffff;
        add.f32 %f1, %f1, %f2;
        shfl.sync.down.b32 %f2, %f1, 4, 31, 0xffffffff;
        add.f32 %f1, %f1, %f2;
        shfl.sync.down.b32 %f2, %f1, 2, 31, 0xffffffff;
        add.f32 %f1, %f1, %f2;
        shfl.sync.down.b32 %f2, %f1, 1, 31, 0xffffffff;
        add.f32 %f1, %f1, %f2;

        and.b32 %r6, %r4, 31;
        shr.u32 %r7, %r4, 5;
        setp.eq.u32 %p1, %r6, 0;
        @!%p1 bra $L_BAR;
        mov.u32 %r8, warp_sums;
        mul.lo.u32 %r9, %r7, 4;
        add.u32 %r8, %r8, %r9;
        st.shared.f32 [%r8], %f1;

    $L_BAR:
        bar.sync 0;
        setp.eq.u32 %p1, %r4, 0;
        @!%p1 bra $L_DONE;

        mov.u32 %r8, warp_sums;
        ld.shared.f32 %f3, [%r8];
        ld.shared.f32 %f4, [%r8+4];
        add.f32 %f3, %f3, %f4;
        ld.shared.f32 %f5, [%r8+8];
        add.f32 %f3, %f3, %f5;
        ld.shared.f32 %f6, [%r8+12];
        add.f32 %f3, %f3, %f6;

        cvta.to.global.u64 %rd3, %rd2;
        atom.global.add.f32 %f7, [%rd3], %f3;

    $L_DONE:
        ret;
    }
"#;

#[test]
fn sample_parallel_reduction_warp_shuffle_and_atomic() {
    let _guard = TEST_LOCK.lock().expect("test lock");
    assert_eq!(cudaSetDevice(0), CudaErrorT::CudaSuccess);

    let n = 256usize;
    let h_in: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 5.0) * 0.375 + 1.25).collect();
    let cpu_sum: f32 = h_in.iter().sum();

    let mut d_in: *mut c_void = core::ptr::null_mut();
    let mut d_out: *mut c_void = core::ptr::null_mut();
    assert_eq!(
        cudaMalloc(&mut d_in, n * core::mem::size_of::<f32>()),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(
        cudaMalloc(&mut d_out, core::mem::size_of::<f32>()),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(
        cudaMemcpy(
            d_in,
            h_in.as_ptr() as *const c_void,
            n * core::mem::size_of::<f32>(),
            CudaMemcpyKind::HostToDevice,
        ),
        CudaErrorT::CudaSuccess
    );
    let zero = 0.0f32;
    assert_eq!(
        cudaMemcpy(
            d_out,
            (&zero as *const f32) as *const c_void,
            core::mem::size_of::<f32>(),
            CudaMemcpyKind::HostToDevice,
        ),
        CudaErrorT::CudaSuccess
    );

    let mut arg_in = d_in as u64;
    let mut arg_out = d_out as u64;
    let mut arg_n = n as u32;
    let mut params = [
        (&mut arg_in as *mut u64) as *mut c_void,
        (&mut arg_out as *mut u64) as *mut c_void,
        (&mut arg_n as *mut u32) as *mut c_void,
    ];

    launch_ptx_kernel(
        REDUCTION_PTX,
        "reduction_sum",
        Dim3 { x: 2, y: 1, z: 1 },
        Dim3 { x: 128, y: 1, z: 1 },
        16,
        &mut params,
    );

    let mut gpu_sum = 0.0f32;
    assert_eq!(
        cudaMemcpy(
            (&mut gpu_sum as *mut f32) as *mut c_void,
            d_out,
            core::mem::size_of::<f32>(),
            CudaMemcpyKind::DeviceToHost,
        ),
        CudaErrorT::CudaSuccess
    );

    assert!(
        (gpu_sum - cpu_sum).abs() < 1e-3,
        "reduction sum mismatch: gpu={gpu_sum} cpu={cpu_sum}"
    );

    assert_eq!(cudaFree(d_in), CudaErrorT::CudaSuccess);
    assert_eq!(cudaFree(d_out), CudaErrorT::CudaSuccess);
}

const SOFTMAX_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64

    .visible .entry row_softmax(
        .param .u64 in_ptr,
        .param .u64 out_ptr,
        .param .u64 any_pos_flag_ptr
    )
    {
        .reg .pred %p<3>;
        .reg .f32 %f<10>;
        .reg .b32 %r<8>;
        .reg .b64 %rd<8>;

        ld.param.u64 %rd1, [in_ptr];
        ld.param.u64 %rd2, [out_ptr];
        ld.param.u64 %rd3, [any_pos_flag_ptr];

        mov.u32 %r1, %ctaid.x;
        mov.u32 %r2, %ntid.x;
        mov.u32 %r3, %tid.x;
        mad.lo.u32 %r4, %r1, %r2, %r3;

        cvta.to.global.u64 %rd4, %rd1;
        mul.wide.u32 %rd5, %r4, 4;
        add.u64 %rd6, %rd4, %rd5;
        ld.global.f32 %f1, [%rd6];

        mov.f32 %f2, %f1;
        shfl.sync.bfly.b32 %f3, %f2, 16, 31, 0xffffffff;
        max.f32 %f2, %f2, %f3;
        shfl.sync.bfly.b32 %f3, %f2, 8, 31, 0xffffffff;
        max.f32 %f2, %f2, %f3;
        shfl.sync.bfly.b32 %f3, %f2, 4, 31, 0xffffffff;
        max.f32 %f2, %f2, %f3;
        shfl.sync.bfly.b32 %f3, %f2, 2, 31, 0xffffffff;
        max.f32 %f2, %f2, %f3;
        shfl.sync.bfly.b32 %f3, %f2, 1, 31, 0xffffffff;
        max.f32 %f2, %f2, %f3;

        sub.f32 %f4, %f1, %f2;
        mul.f32 %f5, %f4, 0f3fb8aa3b;
        ex2.approx.f32 %f6, %f5;

        mov.f32 %f7, %f6;
        shfl.sync.bfly.b32 %f3, %f7, 16, 31, 0xffffffff;
        add.f32 %f7, %f7, %f3;
        shfl.sync.bfly.b32 %f3, %f7, 8, 31, 0xffffffff;
        add.f32 %f7, %f7, %f3;
        shfl.sync.bfly.b32 %f3, %f7, 4, 31, 0xffffffff;
        add.f32 %f7, %f7, %f3;
        shfl.sync.bfly.b32 %f3, %f7, 2, 31, 0xffffffff;
        add.f32 %f7, %f7, %f3;
        shfl.sync.bfly.b32 %f3, %f7, 1, 31, 0xffffffff;
        add.f32 %f7, %f7, %f3;

        rcp.rn.f32 %f8, %f7;
        mul.f32 %f9, %f6, %f8;

        cvta.to.global.u64 %rd4, %rd2;
        add.u64 %rd7, %rd4, %rd5;
        st.global.f32 [%rd7], %f9;

        setp.gt.f32 %p1, %f9, 0f00000000;
        vote.sync.any.pred %p2, %p1, 0xffffffff;
        setp.eq.u32 %p1, %r3, 0;
        @!%p1 bra $L_DONE;
        selp.u32 %r5, 1, 0, %p2;
        cvta.to.global.u64 %rd4, %rd3;
        mul.wide.u32 %rd5, %r1, 4;
        add.u64 %rd6, %rd4, %rd5;
        st.global.u32 [%rd6], %r5;

    $L_DONE:
        ret;
    }
"#;

#[test]
fn sample_row_softmax_transcendentals_and_vote() {
    let _guard = TEST_LOCK.lock().expect("test lock");
    assert_eq!(cudaSetDevice(0), CudaErrorT::CudaSuccess);

    let rows = 4usize;
    let cols = 32usize;
    let total = rows * cols;
    let h_in: Vec<f32> = (0..total)
        .map(|i| ((i as f32) * 0.17).sin() * 2.5 + ((i / cols) as f32) * 0.4)
        .collect();

    let mut cpu_out = vec![0.0f32; total];
    for r in 0..rows {
        let row = &h_in[r * cols..(r + 1) * cols];
        let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - max_v).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        for c in 0..cols {
            cpu_out[r * cols + c] = exps[c] / sum_e;
        }
    }

    let bytes = total * core::mem::size_of::<f32>();
    let mut d_in: *mut c_void = core::ptr::null_mut();
    let mut d_out: *mut c_void = core::ptr::null_mut();
    let mut d_flags: *mut c_void = core::ptr::null_mut();
    assert_eq!(cudaMalloc(&mut d_in, bytes), CudaErrorT::CudaSuccess);
    assert_eq!(cudaMalloc(&mut d_out, bytes), CudaErrorT::CudaSuccess);
    assert_eq!(
        cudaMalloc(&mut d_flags, rows * core::mem::size_of::<u32>()),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(
        cudaMemcpy(
            d_in,
            h_in.as_ptr() as *const c_void,
            bytes,
            CudaMemcpyKind::HostToDevice
        ),
        CudaErrorT::CudaSuccess
    );

    let mut arg_in = d_in as u64;
    let mut arg_out = d_out as u64;
    let mut arg_flags = d_flags as u64;
    let mut params = [
        (&mut arg_in as *mut u64) as *mut c_void,
        (&mut arg_out as *mut u64) as *mut c_void,
        (&mut arg_flags as *mut u64) as *mut c_void,
    ];

    launch_ptx_kernel(
        SOFTMAX_PTX,
        "row_softmax",
        Dim3 { x: rows as u32, y: 1, z: 1 },
        Dim3 { x: cols as u32, y: 1, z: 1 },
        0,
        &mut params,
    );

    let mut h_out = vec![0.0f32; total];
    let mut h_flags = vec![0u32; rows];
    assert_eq!(
        cudaMemcpy(
            h_out.as_mut_ptr() as *mut c_void,
            d_out,
            bytes,
            CudaMemcpyKind::DeviceToHost
        ),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(
        cudaMemcpy(
            h_flags.as_mut_ptr() as *mut c_void,
            d_flags,
            rows * core::mem::size_of::<u32>(),
            CudaMemcpyKind::DeviceToHost
        ),
        CudaErrorT::CudaSuccess
    );

    for r in 0..rows {
        assert_eq!(h_flags[r], 1, "vote.sync.any should be true for row {r}");
        let row_sum: f32 = h_out[r * cols..(r + 1) * cols].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-4, "row {r} sum={row_sum}");
        for c in 0..cols {
            let idx = r * cols + c;
            assert!(
                (h_out[idx] - cpu_out[idx]).abs() < 1e-4,
                "softmax mismatch at ({r},{c}): gpu={} cpu={}",
                h_out[idx],
                cpu_out[idx]
            );
        }
    }

    assert_eq!(cudaFree(d_in), CudaErrorT::CudaSuccess);
    assert_eq!(cudaFree(d_out), CudaErrorT::CudaSuccess);
    assert_eq!(cudaFree(d_flags), CudaErrorT::CudaSuccess);
}

const ADVANCED_OPS_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64

    .func (.param .s32 fn_ret) add_bias_and_scale(
        .param .s32 fn_arg
    )
    {
        .reg .s32 %r<4>;
        ld.param.s32 %r0, [fn_arg];
        mul.lo.s32 %r1, %r0, 3;
        add.s32 %r2, %r1, 7;
        st.param.s32 [fn_ret], %r2;
        ret;
    }

    .visible .entry ptx_advanced_ops(
        .param .u64 vec_in_ptr,
        .param .u64 vec_out_ptr,
        .param .u64 float_out_ptr,
        .param .u64 int_out_ptr,
        .param .s32 in_scalar
    )
    {
        .local .align 4 .b8 lscratch[16];
        .reg .f32 %f<10>;
        .reg .b16 %h<4>;
        .reg .b32 %r<14>;
        .reg .b64 %rd<10>;

        ld.param.u64 %rd1, [vec_in_ptr];
        ld.param.u64 %rd2, [vec_out_ptr];
        ld.param.u64 %rd3, [float_out_ptr];
        ld.param.u64 %rd4, [int_out_ptr];
        ld.param.s32 %r1, [in_scalar];

        // 1. Call .func helper
        call (%r2), add_bias_and_scale, (%r1);

        // 2. Vectorized v4 load/store
        cvta.to.global.u64 %rd5, %rd1;
        ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd5];
        cvta.to.global.u64 %rd6, %rd2;
        st.global.v4.f32 [%rd6], {%f0, %f1, %f2, %f3};

        // 3. .local scratch roundtrip + FP16 & BF16 conversions
        st.local.f32 [lscratch], %f0;
        st.local.f32 [lscratch+4], %f1;
        ld.local.f32 %f4, [lscratch];
        ld.local.f32 %f5, [lscratch+4];
        add.f32 %f6, %f4, %f5;

        cvt.rn.f16.f32 %h0, %f2;
        cvt.f32.f16 %f7, %h0;
        cvt.rn.bf16.f32 %h1, %f3;
        cvt.f32.bf16 %f8, %h1;

        cvta.to.global.u64 %rd7, %rd3;
        st.global.f32 [%rd7], %f6;
        st.global.f32 [%rd7+4], %f7;
        st.global.f32 [%rd7+8], %f8;

        // 4. bfe, bfi, prmt, dp4a
        bfe.s32 %r3, -240, 4, 8;
        bfi.b32 %r4, 0xAB, 0x11223344, 8, 8;
        prmt.b32 %r5, 0x81234567, 0x1122337F, 0x9B40;
        dp4a.s32.s32 %r6, 0x0403FE01, 0xFF020304, 10;

        cvta.to.global.u64 %rd8, %rd4;
        st.global.s32 [%rd8], %r2;
        st.global.s32 [%rd8+4], %r3;
        st.global.u32 [%rd8+8], %r4;
        st.global.u32 [%rd8+12], %r5;
        st.global.s32 [%rd8+16], %r6;
        ret;
    }
"#;

#[test]
fn sample_ptx_advanced_ops_inlining_vec4_fp16_bf16_local_dp4a_prmt() {
    let _guard = TEST_LOCK.lock().expect("test lock");
    assert_eq!(cudaSetDevice(0), CudaErrorT::CudaSuccess);

    let h_vec_in = [1.5f32, -2.25f32, 3.5f32, 6.25f32];
    let mut d_vec_in: *mut c_void = core::ptr::null_mut();
    let mut d_vec_out: *mut c_void = core::ptr::null_mut();
    let mut d_flt_out: *mut c_void = core::ptr::null_mut();
    let mut d_int_out: *mut c_void = core::ptr::null_mut();
    assert_eq!(cudaMalloc(&mut d_vec_in, 16), CudaErrorT::CudaSuccess);
    assert_eq!(cudaMalloc(&mut d_vec_out, 16), CudaErrorT::CudaSuccess);
    assert_eq!(cudaMalloc(&mut d_flt_out, 12), CudaErrorT::CudaSuccess);
    assert_eq!(cudaMalloc(&mut d_int_out, 20), CudaErrorT::CudaSuccess);
    assert_eq!(
        cudaMemcpy(
            d_vec_in,
            h_vec_in.as_ptr() as *const c_void,
            16,
            CudaMemcpyKind::HostToDevice
        ),
        CudaErrorT::CudaSuccess
    );

    let mut arg_vin = d_vec_in as u64;
    let mut arg_vout = d_vec_out as u64;
    let mut arg_fout = d_flt_out as u64;
    let mut arg_iout = d_int_out as u64;
    let mut arg_scalar: i32 = 11;
    let mut params = [
        (&mut arg_vin as *mut u64) as *mut c_void,
        (&mut arg_vout as *mut u64) as *mut c_void,
        (&mut arg_fout as *mut u64) as *mut c_void,
        (&mut arg_iout as *mut u64) as *mut c_void,
        (&mut arg_scalar as *mut i32) as *mut c_void,
    ];

    launch_ptx_kernel(
        ADVANCED_OPS_PTX,
        "ptx_advanced_ops",
        Dim3 { x: 1, y: 1, z: 1 },
        Dim3 { x: 1, y: 1, z: 1 },
        0,
        &mut params,
    );

    let mut h_vec_out = [0.0f32; 4];
    let mut h_flt_out = [0.0f32; 3];
    let mut h_int_out = [0i32; 5];
    assert_eq!(
        cudaMemcpy(
            h_vec_out.as_mut_ptr() as *mut c_void,
            d_vec_out,
            16,
            CudaMemcpyKind::DeviceToHost
        ),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(
        cudaMemcpy(
            h_flt_out.as_mut_ptr() as *mut c_void,
            d_flt_out,
            12,
            CudaMemcpyKind::DeviceToHost
        ),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(
        cudaMemcpy(
            h_int_out.as_mut_ptr() as *mut c_void,
            d_int_out,
            20,
            CudaMemcpyKind::DeviceToHost
        ),
        CudaErrorT::CudaSuccess
    );

    assert_eq!(h_vec_out, h_vec_in);
    assert!((h_flt_out[0] - (1.5 - 2.25)).abs() < 1e-6);
    assert!((h_flt_out[1] - 3.5).abs() < 1e-3);
    assert!((h_flt_out[2] - 6.25).abs() < 1e-2);

    // 11 * 3 + 7 = 40
    assert_eq!(h_int_out[0], 40);
    // -240 = 0xFFFFFF10 -> bfe.s32(_, 4, 8) = 0xFFFFFFF1 = -15
    assert_eq!(h_int_out[1], -15);
    // bfi.b32(0xAB, 0x11223344, 8, 8) = 0x1122AB44
    assert_eq!(h_int_out[2] as u32, 0x1122_AB44);
    // prmt.b32(0x81234567, 0x1122337F, 0x9B40) = 0x00FF7F67
    assert_eq!(h_int_out[3] as u32, 0x00FF_7F67);
    // dp4a: 1*4 + (-2)*3 + 3*2 + 4*(-1) + 10 = 4 - 6 + 6 - 4 + 10 = 10
    assert_eq!(h_int_out[4], 10);

    assert_eq!(cudaFree(d_vec_in), CudaErrorT::CudaSuccess);
    assert_eq!(cudaFree(d_vec_out), CudaErrorT::CudaSuccess);
    assert_eq!(cudaFree(d_flt_out), CudaErrorT::CudaSuccess);
    assert_eq!(cudaFree(d_int_out), CudaErrorT::CudaSuccess);
}

#[test]
fn driver_and_runtime_api_sprint2_surface() {
    let _guard = TEST_LOCK.lock().expect("test lock");
    assert_eq!(cuInit(0), CUresult::CUDA_SUCCESS);
    assert_eq!(cudaSetDevice(0), CudaErrorT::CudaSuccess);

    let mut dev_count: c_int = 0;
    assert_eq!(cudaGetDeviceCount(&mut dev_count), CudaErrorT::CudaSuccess);
    assert!(dev_count >= 1);
    let mut cur_dev: c_int = -1;
    assert_eq!(cudaGetDevice(&mut cur_dev), CudaErrorT::CudaSuccess);
    assert_eq!(cur_dev, 0);

    let mut prop = CudaDeviceProp::default();
    assert_eq!(cudaGetDeviceProperties(&mut prop, 0), CudaErrorT::CudaSuccess);
    let name = unsafe { CStr::from_ptr(prop.name.as_ptr()) }
        .to_str()
        .expect("utf8 device name");
    assert!(name.contains("AMD"));
    assert!(prop.multiProcessorCount > 0);
    assert!(prop.totalGlobalMem > 0);
    assert!(prop.warpSize == 32 || prop.warpSize == 64);

    let mut attr_cus: c_int = 0;
    let mut attr_warp: c_int = 0;
    let mut attr_l2: c_int = 0;
    assert_eq!(
        cudaDeviceGetAttribute(&mut attr_cus, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(
        cudaDeviceGetAttribute(&mut attr_warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 0),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(
        cudaDeviceGetAttribute(&mut attr_l2, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 0),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(attr_cus, prop.multiProcessorCount);
    assert_eq!(attr_warp, prop.warpSize);
    assert_eq!(attr_l2, prop.l2CacheSize);

    let mut drv_ver = 0;
    let mut rt_ver = 0;
    assert_eq!(cudaDriverGetVersion(&mut drv_ver), CudaErrorT::CudaSuccess);
    assert_eq!(cudaRuntimeGetVersion(&mut rt_ver), CudaErrorT::CudaSuccess);
    assert!(drv_ver >= 12000 && rt_ver >= 12000);

    // Pinned host allocations + page alignment + pointer attribute lookup + Default memcpy
    let mut h_pinned: *mut c_void = core::ptr::null_mut();
    assert_eq!(cudaMallocHost(&mut h_pinned, 256), CudaErrorT::CudaSuccess);
    assert_eq!((h_pinned as usize) % 4096, 0);
    let mut h_pinned2: *mut c_void = core::ptr::null_mut();
    assert_eq!(cudaHostAlloc(&mut h_pinned2, 256, 0), CudaErrorT::CudaSuccess);

    let mut d_ptr: CUdeviceptr = 0;
    assert_eq!(cuMemAlloc_v2(&mut d_ptr, 128), CUresult::CUDA_SUCCESS);

    let mut mem_type_dev: c_int = 0;
    assert_eq!(
        cuPointerGetAttribute(
            (&mut mem_type_dev as *mut c_int) as *mut c_void,
            CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            d_ptr + 16
        ),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(mem_type_dev, CU_MEMORYTYPE_DEVICE);

    let mut attrs = [
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        CU_POINTER_ATTRIBUTE_HOST_POINTER,
        CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
        CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        CU_POINTER_ATTRIBUTE_CONTEXT,
    ];
    let mut attr_mem = 0i32;
    let mut attr_host: *mut c_void = core::ptr::null_mut();
    let mut attr_dev: CUdeviceptr = 0;
    let mut attr_ord: c_int = 0;
    let mut attr_ctx: CUcontext = core::ptr::null_mut();
    let mut data_ptrs = [
        (&mut attr_mem as *mut c_int) as *mut c_void,
        (&mut attr_host as *mut *mut c_void) as *mut c_void,
        (&mut attr_dev as *mut CUdeviceptr) as *mut c_void,
        (&mut attr_ord as *mut c_int) as *mut c_void,
        (&mut attr_ctx as *mut CUcontext) as *mut c_void,
    ];
    assert_eq!(
        cuPointerGetAttributes(
            attrs.len() as c_uint,
            attrs.as_mut_ptr(),
            data_ptrs.as_mut_ptr(),
            h_pinned as CUdeviceptr
        ),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(attr_mem, CU_MEMORYTYPE_HOST);
    assert_eq!(attr_host, h_pinned);
    assert_eq!(attr_dev, 0);
    assert_eq!(attr_ord, -2);
    assert!(!attr_ctx.is_null());

    // cuMemsetD8_v2 / cuMemsetD32_v2 + Default memcpy + async driver memcpy
    assert_eq!(cuMemsetD8_v2(d_ptr, 0x5A, 64), CUresult::CUDA_SUCCESS);
    assert_eq!(
        cuMemsetD32_v2(d_ptr + 64, 0xCAFE_BABE, 16),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(
        cudaMemcpy(
            h_pinned,
            d_ptr as *const c_void,
            128,
            CudaMemcpyKind::Default
        ),
        CudaErrorT::CudaSuccess
    );
    let pinned_bytes = unsafe { core::slice::from_raw_parts(h_pinned as *const u8, 128) };
    assert!(pinned_bytes[..64].iter().all(|&b| b == 0x5A));
    let words = unsafe { core::slice::from_raw_parts((h_pinned as *const u8).add(64) as *const u32, 16) };
    assert!(words.iter().all(|&w| w == 0xCAFE_BABE));

    let mut d_ptr2: CUdeviceptr = 0;
    assert_eq!(cuMemAlloc_v2(&mut d_ptr2, 128), CUresult::CUDA_SUCCESS);
    let mut cu_stream: CUstream = core::ptr::null_mut();
    assert_eq!(cuStreamCreate(&mut cu_stream, 0), CUresult::CUDA_SUCCESS);
    assert_eq!(
        cuMemcpyHtoDAsync_v2(d_ptr, h_pinned, 128, cu_stream),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(
        cuMemcpyDtoDAsync_v2(d_ptr2, d_ptr, 128, cu_stream),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(
        cuMemcpyDtoHAsync_v2(h_pinned2, d_ptr2, 128, cu_stream),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(cuStreamSynchronize(cu_stream), CUresult::CUDA_SUCCESS);
    let check_words =
        unsafe { core::slice::from_raw_parts((h_pinned2 as *const u8).add(64) as *const u32, 16) };
    assert!(check_words.iter().all(|&w| w == 0xCAFE_BABE));

    // Events + stream wait
    let mut ev1: CudaEventT = core::ptr::null_mut();
    let mut ev2: CudaEventT = core::ptr::null_mut();
    assert_eq!(cudaEventCreate(&mut ev1), CudaErrorT::CudaSuccess);
    assert_eq!(
        cudaEventCreateWithFlags(&mut ev2, 0),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(
        cudaEventRecord(ev1, cu_stream as CudaStreamT),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(cudaEventSynchronize(ev1), CudaErrorT::CudaSuccess);
    assert_eq!(cudaEventQuery(ev1), CudaErrorT::CudaSuccess);

    let mut rt_stream: CudaStreamT = core::ptr::null_mut();
    assert_eq!(cudaStreamCreate(&mut rt_stream), CudaErrorT::CudaSuccess);
    assert_eq!(cudaStreamWaitEvent(rt_stream, ev1, 0), CudaErrorT::CudaSuccess);
    assert_eq!(cudaEventRecord(ev2, rt_stream), CudaErrorT::CudaSuccess);
    assert_eq!(cudaEventSynchronize(ev2), CudaErrorT::CudaSuccess);
    let mut elapsed_ms = -1.0f32;
    assert_eq!(
        cudaEventElapsedTime(&mut elapsed_ms, ev1, ev2),
        CudaErrorT::CudaSuccess
    );
    assert!(elapsed_ms >= 0.0);

    let mut drv_ev: CUevent = core::ptr::null_mut();
    assert_eq!(cuEventCreate(&mut drv_ev, 0), CUresult::CUDA_SUCCESS);
    assert_eq!(cuEventRecord(drv_ev, cu_stream), CUresult::CUDA_SUCCESS);
    assert_eq!(cuEventSynchronize(drv_ev), CUresult::CUDA_SUCCESS);
    assert_eq!(cuEventQuery(drv_ev), CUresult::CUDA_SUCCESS);
    assert_eq!(cuStreamWaitEvent(cu_stream, drv_ev, 0), CUresult::CUDA_SUCCESS);
    let mut drv_ms = -1.0f32;
    assert_eq!(
        cuEventElapsedTime(&mut drv_ms, ev1 as CUevent, drv_ev),
        CUresult::CUDA_SUCCESS
    );
    assert!(drv_ms >= 0.0);
    assert_eq!(cuEventDestroy_v2(drv_ev), CUresult::CUDA_SUCCESS);

    assert_eq!(cudaEventDestroy(ev1), CudaErrorT::CudaSuccess);
    assert_eq!(cudaEventDestroy(ev2), CudaErrorT::CudaSuccess);
    assert_eq!(
        cudaMemcpyAsync(
            h_pinned,
            d_ptr2 as *const c_void,
            64,
            CudaMemcpyKind::Default,
            rt_stream
        ),
        CudaErrorT::CudaSuccess
    );
    assert_eq!(cudaStreamSynchronize(rt_stream), CudaErrorT::CudaSuccess);
    assert_eq!(cudaStreamDestroy(rt_stream), CudaErrorT::CudaSuccess);
    assert_eq!(cuStreamDestroy_v2(cu_stream), CUresult::CUDA_SUCCESS);

    // cuMemGetInfo_v2 & cudaMemGetInfo
    let (mut cu_free, mut cu_total) = (0usize, 0usize);
    let (mut rt_free, mut rt_total) = (0usize, 0usize);
    assert_eq!(
        cuMemGetInfo_v2(&mut cu_free, &mut cu_total),
        CUresult::CUDA_SUCCESS
    );
    assert_eq!(cudaMemGetInfo(&mut rt_free, &mut rt_total), CudaErrorT::CudaSuccess);
    assert_eq!(cu_total, rt_total);
    assert!(cu_free <= cu_total);

    // Occupancy calculators
    let mut active_blocks = 0;
    assert_eq!(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &mut active_blocks,
            core::ptr::null(),
            256,
            1024
        ),
        CudaErrorT::CudaSuccess
    );
    assert!(active_blocks >= 1);
    let (mut min_grid, mut pot_block) = (0, 0);
    assert_eq!(
        cudaOccupancyMaxPotentialBlockSize(
            &mut min_grid,
            &mut pot_block,
            core::ptr::null(),
            0,
            512
        ),
        CudaErrorT::CudaSuccess
    );
    assert!(min_grid >= prop.multiProcessorCount && pot_block > 0 && pot_block <= 512);

    // Context stack & primary ctx retain/release
    let mut orig_ctx: CUcontext = core::ptr::null_mut();
    assert_eq!(cuCtxGetCurrent(&mut orig_ctx), CUresult::CUDA_SUCCESS);
    let mut new_ctx: CUcontext = core::ptr::null_mut();
    assert_eq!(cuCtxCreate_v2(&mut new_ctx, 0, 0), CUresult::CUDA_SUCCESS);
    assert_eq!(cuCtxPushCurrent_v2(orig_ctx), CUresult::CUDA_SUCCESS);
    let mut popped: CUcontext = core::ptr::null_mut();
    assert_eq!(cuCtxPopCurrent_v2(&mut popped), CUresult::CUDA_SUCCESS);
    assert_eq!(popped, orig_ctx);
    assert_eq!(cuCtxDestroy_v2(new_ctx), CUresult::CUDA_SUCCESS);

    let mut primary: CUcontext = core::ptr::null_mut();
    assert_eq!(cuDevicePrimaryCtxRetain(&mut primary, 0), CUresult::CUDA_SUCCESS);
    assert!(!primary.is_null());
    assert_eq!(cuDevicePrimaryCtxRelease_v2(0), CUresult::CUDA_SUCCESS);

    // cuGetProcAddress_v2 checks for Sprint 2 symbols
    for sym in [
        "cuCtxPushCurrent_v2",
        "cuCtxPopCurrent_v2",
        "cuDevicePrimaryCtxRetain",
        "cuDevicePrimaryCtxRelease_v2",
        "cuMemGetInfo_v2",
        "cuMemsetD8_v2",
        "cuMemsetD32_v2",
        "cuMemcpyHtoDAsync_v2",
        "cuMemcpyDtoHAsync_v2",
        "cuMemcpyDtoDAsync_v2",
        "cuStreamWaitEvent",
        "cuEventQuery",
        "cuPointerGetAttribute",
        "cuPointerGetAttributes",
    ] {
        let cs = CString::new(sym).expect("valid sym");
        let mut pfn: *mut c_void = core::ptr::null_mut();
        let mut status: c_int = -1;
        assert_eq!(
            cuGetProcAddress_v2(cs.as_ptr(), &mut pfn, 12040, 0, &mut status),
            CUresult::CUDA_SUCCESS,
            "missing driver symbol {sym}"
        );
        assert!(!pfn.is_null());
        assert_eq!(status, 0);
    }

    assert_eq!(cuMemFree_v2(d_ptr), CUresult::CUDA_SUCCESS);
    assert_eq!(cuMemFree_v2(d_ptr2), CUresult::CUDA_SUCCESS);
    assert_eq!(cudaFreeHost(h_pinned), CudaErrorT::CudaSuccess);
    assert_eq!(cudaFreeHost(h_pinned2), CudaErrorT::CudaSuccess);
    assert_eq!(cudaDeviceReset(), CudaErrorT::CudaSuccess);
}
