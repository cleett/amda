//! Embedded AMDA GPU PTX GEMM kernels powering `libamda_cublas`.

/// Unified multi-entry PTX module containing `amda_gemm_f32`, `amda_gemm_f16`, and `amda_gemm_bf16`.
pub const GEMM_KERNELS_PTX: &str = concat!(
    r#"
    .version 7.8
    .target sm_80
    .address_size 64

    .visible .entry amda_gemm_f32(
        .param .u64 A_ptr,
        .param .u64 B_ptr,
        .param .u64 C_ptr,
        .param .u32 m,
        .param .u32 n,
        .param .u32 k,
        .param .u32 lda,
        .param .u32 ldb,
        .param .u32 ldc,
        .param .u32 stride_a,
        .param .u32 stride_b,
        .param .u32 stride_c,
        .param .u32 trans_a,
        .param .u32 trans_b,
        .param .f32 alpha,
        .param .f32 beta
    )
    {
        .reg .pred %p<6>;
        .reg .f32 %f<8>;
        .reg .b32 %r<28>;
        .reg .b64 %rd<12>;

        ld.param.u64 %rd1, [A_ptr];
        ld.param.u64 %rd2, [B_ptr];
        ld.param.u64 %rd3, [C_ptr];
        ld.param.u32 %r1, [m];
        ld.param.u32 %r2, [n];
        ld.param.u32 %r3, [k];
        ld.param.u32 %r4, [lda];
        ld.param.u32 %r5, [ldb];
        ld.param.u32 %r6, [ldc];
        ld.param.u32 %r7, [stride_a];
        ld.param.u32 %r8, [stride_b];
        ld.param.u32 %r9, [stride_c];
        ld.param.u32 %r10, [trans_a];
        ld.param.u32 %r11, [trans_b];
        ld.param.f32 %f1, [alpha];
        ld.param.f32 %f2, [beta];

        mov.u32 %r12, %ctaid.x;
        mov.u32 %r13, %tid.x;
        mad.lo.u32 %r14, %r12, 16, %r13;
        mov.u32 %r12, %ctaid.y;
        mov.u32 %r13, %tid.y;
        mad.lo.u32 %r15, %r12, 16, %r13;
        mov.u32 %r16, %ctaid.z;

        setp.ge.u32 %p1, %r14, %r1;
        @%p1 bra $L_EXIT;
        setp.ge.u32 %p1, %r15, %r2;
        @%p1 bra $L_EXIT;

        mul.lo.u32 %r17, %r16, %r7;
        mul.lo.u32 %r18, %r16, %r8;
        mul.lo.u32 %r19, %r16, %r9;

        cvta.to.global.u64 %rd4, %rd1;
        cvta.to.global.u64 %rd5, %rd2;
        cvta.to.global.u64 %rd6, %rd3;

        setp.ne.u32 %p2, %r10, 0;
        setp.ne.u32 %p3, %r11, 0;

        mov.f32 %f3, 0f00000000;
        mov.u32 %r20, 0;

    $L_LOOP:
        setp.ge.u32 %p4, %r20, %r3;
        @%p4 bra $L_AFTER_LOOP;

        mad.lo.u32 %r21, %r20, %r4, %r14;
        mad.lo.u32 %r22, %r14, %r4, %r20;
        selp.u32 %r23, %r22, %r21, %p2;
        add.u32 %r23, %r23, %r17;
        mul.wide.u32 %rd7, %r23, 4;
        add.u64 %rd8, %rd4, %rd7;
        ld.global.f32 %f4, [%rd8];

        mad.lo.u32 %r24, %r15, %r5, %r20;
        mad.lo.u32 %r25, %r20, %r5, %r15;
        selp.u32 %r26, %r25, %r24, %p3;
        add.u32 %r26, %r26, %r18;
        mul.wide.u32 %rd9, %r26, 4;
        add.u64 %rd10, %rd5, %rd9;
        ld.global.f32 %f5, [%rd10];

        fma.rn.f32 %f3, %f4, %f5, %f3;
        add.u32 %r20, %r20, 1;
        bra $L_LOOP;

    $L_AFTER_LOOP:
        mul.f32 %f6, %f3, %f1;
        mad.lo.u32 %r27, %r15, %r6, %r14;
        add.u32 %r27, %r27, %r19;
        mul.wide.u32 %rd7, %r27, 4;
        add.u64 %rd11, %rd6, %rd7;

        setp.eq.f32 %p5, %f2, 0f00000000;
        @%p5 bra $L_STORE;
        ld.global.f32 %f7, [%rd11];
        fma.rn.f32 %f6, %f2, %f7, %f6;

    $L_STORE:
        st.global.f32 [%rd11], %f6;

    $L_EXIT:
        ret;
    }

    .visible .entry amda_gemm_f16(
        .param .u64 A_ptr,
        .param .u64 B_ptr,
        .param .u64 C_ptr,
        .param .u32 m,
        .param .u32 n,
        .param .u32 k,
        .param .u32 lda,
        .param .u32 ldb,
        .param .u32 ldc,
        .param .u32 stride_a,
        .param .u32 stride_b,
        .param .u32 stride_c,
        .param .u32 trans_a,
        .param .u32 trans_b,
        .param .f32 alpha,
        .param .f32 beta,
        .param .u32 out_is_f32
    )
    {
        .reg .pred %p<7>;
        .reg .b16 %h<4>;
        .reg .f32 %f<8>;
        .reg .b32 %r<29>;
        .reg .b64 %rd<12>;

        ld.param.u64 %rd1, [A_ptr];
        ld.param.u64 %rd2, [B_ptr];
        ld.param.u64 %rd3, [C_ptr];
        ld.param.u32 %r1, [m];
        ld.param.u32 %r2, [n];
        ld.param.u32 %r3, [k];
        ld.param.u32 %r4, [lda];
        ld.param.u32 %r5, [ldb];
        ld.param.u32 %r6, [ldc];
        ld.param.u32 %r7, [stride_a];
        ld.param.u32 %r8, [stride_b];
        ld.param.u32 %r9, [stride_c];
        ld.param.u32 %r10, [trans_a];
        ld.param.u32 %r11, [trans_b];
        ld.param.f32 %f1, [alpha];
        ld.param.f32 %f2, [beta];
        ld.param.u32 %r28, [out_is_f32];

        mov.u32 %r12, %ctaid.x;
        mov.u32 %r13, %tid.x;
        mad.lo.u32 %r14, %r12, 16, %r13;
        mov.u32 %r12, %ctaid.y;
        mov.u32 %r13, %tid.y;
        mad.lo.u32 %r15, %r12, 16, %r13;
        mov.u32 %r16, %ctaid.z;

        setp.ge.u32 %p1, %r14, %r1;
        @%p1 bra $L_EXIT;
        setp.ge.u32 %p1, %r15, %r2;
        @%p1 bra $L_EXIT;

        mul.lo.u32 %r17, %r16, %r7;
        mul.lo.u32 %r18, %r16, %r8;
        mul.lo.u32 %r19, %r16, %r9;

        cvta.to.global.u64 %rd4, %rd1;
        cvta.to.global.u64 %rd5, %rd2;
        cvta.to.global.u64 %rd6, %rd3;

        setp.ne.u32 %p2, %r10, 0;
        setp.ne.u32 %p3, %r11, 0;
        setp.ne.u32 %p6, %r28, 0;

        mov.f32 %f3, 0f00000000;
        mov.u32 %r20, 0;

    $L_LOOP:
        setp.ge.u32 %p4, %r20, %r3;
        @%p4 bra $L_AFTER_LOOP;

        mad.lo.u32 %r21, %r20, %r4, %r14;
        mad.lo.u32 %r22, %r14, %r4, %r20;
        selp.u32 %r23, %r22, %r21, %p2;
        add.u32 %r23, %r23, %r17;
        mul.wide.u32 %rd7, %r23, 2;
        add.u64 %rd8, %rd4, %rd7;
        ld.global.f16 %h0, [%rd8];
        cvt.f32.f16 %f4, %h0;

        mad.lo.u32 %r24, %r15, %r5, %r20;
        mad.lo.u32 %r25, %r20, %r5, %r15;
        selp.u32 %r26, %r25, %r24, %p3;
        add.u32 %r26, %r26, %r18;
        mul.wide.u32 %rd9, %r26, 2;
        add.u64 %rd10, %rd5, %rd9;
        ld.global.f16 %h1, [%rd10];
        cvt.f32.f16 %f5, %h1;

        fma.rn.f32 %f3, %f4, %f5, %f3;
        add.u32 %r20, %r20, 1;
        bra $L_LOOP;

    $L_AFTER_LOOP:
        mul.f32 %f6, %f3, %f1;
        mad.lo.u32 %r27, %r15, %r6, %r14;
        add.u32 %r27, %r27, %r19;
        @%p6 bra $L_OUT_F32;

        mul.wide.u32 %rd7, %r27, 2;
        add.u64 %rd11, %rd6, %rd7;
        setp.eq.f32 %p5, %f2, 0f00000000;
        @%p5 bra $L_STORE_F16;
        ld.global.f16 %h2, [%rd11];
        cvt.f32.f16 %f7, %h2;
        fma.rn.f32 %f6, %f2, %f7, %f6;

    $L_STORE_F16:
        cvt.rn.f16.f32 %h3, %f6;
        st.global.f16 [%rd11], %h3;
        bra $L_EXIT;

    $L_OUT_F32:
        mul.wide.u32 %rd7, %r27, 4;
        add.u64 %rd11, %rd6, %rd7;
        setp.eq.f32 %p5, %f2, 0f00000000;
        @%p5 bra $L_STORE_F32;
        ld.global.f32 %f7, [%rd11];
        fma.rn.f32 %f6, %f2, %f7, %f6;

    $L_STORE_F32:
        st.global.f32 [%rd11], %f6;

    $L_EXIT:
        ret;
    }

    .visible .entry amda_gemm_bf16(
        .param .u64 A_ptr,
        .param .u64 B_ptr,
        .param .u64 C_ptr,
        .param .u32 m,
        .param .u32 n,
        .param .u32 k,
        .param .u32 lda,
        .param .u32 ldb,
        .param .u32 ldc,
        .param .u32 stride_a,
        .param .u32 stride_b,
        .param .u32 stride_c,
        .param .u32 trans_a,
        .param .u32 trans_b,
        .param .f32 alpha,
        .param .f32 beta,
        .param .u32 out_is_f32
    )
    {
        .reg .pred %p<7>;
        .reg .b16 %h<4>;
        .reg .f32 %f<8>;
        .reg .b32 %r<29>;
        .reg .b64 %rd<12>;

        ld.param.u64 %rd1, [A_ptr];
        ld.param.u64 %rd2, [B_ptr];
        ld.param.u64 %rd3, [C_ptr];
        ld.param.u32 %r1, [m];
        ld.param.u32 %r2, [n];
        ld.param.u32 %r3, [k];
        ld.param.u32 %r4, [lda];
        ld.param.u32 %r5, [ldb];
        ld.param.u32 %r6, [ldc];
        ld.param.u32 %r7, [stride_a];
        ld.param.u32 %r8, [stride_b];
        ld.param.u32 %r9, [stride_c];
        ld.param.u32 %r10, [trans_a];
        ld.param.u32 %r11, [trans_b];
        ld.param.f32 %f1, [alpha];
        ld.param.f32 %f2, [beta];
        ld.param.u32 %r28, [out_is_f32];

        mov.u32 %r12, %ctaid.x;
        mov.u32 %r13, %tid.x;
        mad.lo.u32 %r14, %r12, 16, %r13;
        mov.u32 %r12, %ctaid.y;
        mov.u32 %r13, %tid.y;
        mad.lo.u32 %r15, %r12, 16, %r13;
        mov.u32 %r16, %ctaid.z;

        setp.ge.u32 %p1, %r14, %r1;
        @%p1 bra $L_EXIT;
        setp.ge.u32 %p1, %r15, %r2;
        @%p1 bra $L_EXIT;

        mul.lo.u32 %r17, %r16, %r7;
        mul.lo.u32 %r18, %r16, %r8;
        mul.lo.u32 %r19, %r16, %r9;

        cvta.to.global.u64 %rd4, %rd1;
        cvta.to.global.u64 %rd5, %rd2;
        cvta.to.global.u64 %rd6, %rd3;

        setp.ne.u32 %p2, %r10, 0;
        setp.ne.u32 %p3, %r11, 0;
        setp.ne.u32 %p6, %r28, 0;

        mov.f32 %f3, 0f00000000;
        mov.u32 %r20, 0;

    $L_LOOP:
        setp.ge.u32 %p4, %r20, %r3;
        @%p4 bra $L_AFTER_LOOP;

        mad.lo.u32 %r21, %r20, %r4, %r14;
        mad.lo.u32 %r22, %r14, %r4, %r20;
        selp.u32 %r23, %r22, %r21, %p2;
        add.u32 %r23, %r23, %r17;
        mul.wide.u32 %rd7, %r23, 2;
        add.u64 %rd8, %rd4, %rd7;
        ld.global.bf16 %h0, [%rd8];
        cvt.f32.bf16 %f4, %h0;

        mad.lo.u32 %r24, %r15, %r5, %r20;
        mad.lo.u32 %r25, %r20, %r5, %r15;
        selp.u32 %r26, %r25, %r24, %p3;
        add.u32 %r26, %r26, %r18;
        mul.wide.u32 %rd9, %r26, 2;
        add.u64 %rd10, %rd5, %rd9;
        ld.global.bf16 %h1, [%rd10];
        cvt.f32.bf16 %f5, %h1;

        fma.rn.f32 %f3, %f4, %f5, %f3;
        add.u32 %r20, %r20, 1;
        bra $L_LOOP;

    $L_AFTER_LOOP:
        mul.f32 %f6, %f3, %f1;
        mad.lo.u32 %r27, %r15, %r6, %r14;
        add.u32 %r27, %r27, %r19;
        @%p6 bra $L_OUT_F32;

        mul.wide.u32 %rd7, %r27, 2;
        add.u64 %rd11, %rd6, %rd7;
        setp.eq.f32 %p5, %f2, 0f00000000;
        @%p5 bra $L_STORE_BF16;
        ld.global.bf16 %h2, [%rd11];
        cvt.f32.bf16 %f7, %h2;
        fma.rn.f32 %f6, %f2, %f7, %f6;

    $L_STORE_BF16:
        cvt.rn.bf16.f32 %h3, %f6;
        st.global.bf16 [%rd11], %h3;
        bra $L_EXIT;

    $L_OUT_F32:
        mul.wide.u32 %rd7, %r27, 4;
        add.u64 %rd11, %rd6, %rd7;
        setp.eq.f32 %p5, %f2, 0f00000000;
        @%p5 bra $L_STORE_F32;
        ld.global.f32 %f7, [%rd11];
        fma.rn.f32 %f6, %f2, %f7, %f6;

    $L_STORE_F32:
        st.global.f32 [%rd11], %f6;

    $L_EXIT:
        ret;
    }
"#,
    "\0"
);

/// Clean UTF-8 PTX source slice without trailing NUL for Rust PTX parser consumers.
pub fn gemm_kernels_ptx_source() -> &'static str {
    GEMM_KERNELS_PTX.trim_end_matches('\0')
}
