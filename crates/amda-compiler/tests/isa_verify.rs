use amda_compiler::{
    compile_llvm_ir_to_hsaco, compile_tier0_native, disassemble_hsaco_with_objdump, lower_ptx_to_ir,
    parse_ptx, validate_llvm_ir, verify_encoded_machine_code, CompileTier, Compiler,
    LlvmToolchain, TierPolicy, WaveMode,
};
use amda_elf::IsaTarget;

const VECTOR_ADD_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64

    .visible .entry vector_add(
        .param .u64 vector_add_param_0,
        .param .u64 vector_add_param_1,
        .param .u64 vector_add_param_2,
        .param .u32 vector_add_param_3
    )
    {
        .reg .pred %p<2>;
        .reg .f32 %f<4>;
        .reg .b32 %r<6>;
        .reg .b64 %rd<10>;

        ld.param.u64 %rd1, [vector_add_param_0];
        ld.param.u64 %rd2, [vector_add_param_1];
        ld.param.u64 %rd3, [vector_add_param_2];
        ld.param.u32 %r1, [vector_add_param_3];

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

const WARP_REDUCE_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64

    .visible .entry warp_reduce(
        .param .u64 in_ptr,
        .param .u64 out_ptr,
        .param .u32 n
    )
    {
        .shared .align 4 .b8 smem[128];
        .reg .pred %p<2>;
        .reg .f32 %f<8>;
        .reg .b32 %r<10>;
        .reg .b64 %rd<6>;

        ld.param.u64 %rd1, [in_ptr];
        ld.param.u64 %rd2, [out_ptr];
        ld.param.u32 %r1, [n];

        mov.u32 %r2, %ctaid.x;
        mov.u32 %r3, %ntid.x;
        mov.u32 %r4, %tid.x;
        mad.lo.s32 %r5, %r2, %r3, %r4;

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
        @!%p1 bra $L_SYNC;
        mov.u32 %r8, smem;
        mul.lo.u32 %r9, %r7, 4;
        add.u32 %r8, %r8, %r9;
        st.shared.f32 [%r8], %f1;

    $L_SYNC:
        bar.sync 0;
        setp.eq.u32 %p1, %r4, 0;
        @!%p1 bra $L_END;

        mov.u32 %r8, smem;
        ld.shared.f32 %f3, [%r8];
        cvta.to.global.u64 %rd3, %rd2;
        atom.global.add.f32 %f4, [%rd3], %f3;

    $L_END:
        ret;
    }
"#;

const ROW_SOFTMAX_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64

    .visible .entry row_softmax(
        .param .u64 in_ptr,
        .param .u64 out_ptr,
        .param .u64 flag_ptr
    )
    {
        .reg .pred %p<3>;
        .reg .f32 %f<10>;
        .reg .b32 %r<8>;
        .reg .b64 %rd<8>;

        ld.param.u64 %rd1, [in_ptr];
        ld.param.u64 %rd2, [out_ptr];
        ld.param.u64 %rd3, [flag_ptr];

        mov.u32 %r1, %ctaid.x;
        mov.u32 %r2, %ntid.x;
        mov.u32 %r3, %tid.x;
        mad.lo.s32 %r4, %r1, %r2, %r3;

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

const SAXPY_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64

    .visible .entry saxpy(
        .param .f32 saxpy_param_a,
        .param .u64 saxpy_param_x,
        .param .u64 saxpy_param_y,
        .param .u32 saxpy_param_n
    )
    {
        .reg .pred %p<2>;
        .reg .f32 %f<5>;
        .reg .b32 %r<6>;
        .reg .b64 %rd<8>;

        ld.param.f32 %f1, [saxpy_param_a];
        ld.param.u64 %rd1, [saxpy_param_x];
        ld.param.u64 %rd2, [saxpy_param_y];
        ld.param.u32 %r1, [saxpy_param_n];

        mov.u32 %r2, %ctaid.x;
        mov.u32 %r3, %ntid.x;
        mov.u32 %r4, %tid.x;
        mad.lo.s32 %r5, %r2, %r3, %r4;
        setp.ge.s32 %p1, %r5, %r1;
        @%p1 bra $L_DONE;

        cvta.to.global.u64 %rd3, %rd1;
        cvta.to.global.u64 %rd4, %rd2;
        mul.wide.s32 %rd5, %r5, 4;
        add.s64 %rd6, %rd3, %rd5;
        add.s64 %rd7, %rd4, %rd5;
        ld.global.f32 %f2, [%rd6];
        ld.global.f32 %f3, [%rd7];
        fma.rn.f32 %f4, %f1, %f2, %f3;
        st.global.f32 [%rd7], %f4;

    $L_DONE:
        ret;
    }
"#;

const SHARED_REDUCE_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64

    .visible .entry shared_reduce(
        .param .u64 in_ptr,
        .param .u64 out_ptr
    )
    {
        .shared .align 4 .b8 smem[1024];
        .reg .f32 %f<4>;
        .reg .b32 %r<6>;
        .reg .b64 %rd<7>;

        ld.param.u64 %rd1, [in_ptr];
        ld.param.u64 %rd2, [out_ptr];

        mov.u32 %r1, %tid.x;
        mov.u32 %r2, smem;
        mul.lo.s32 %r3, %r1, 4;
        add.s32 %r4, %r2, %r3;

        cvta.to.global.u64 %rd3, %rd1;
        mul.wide.s32 %rd4, %r1, 4;
        add.s64 %rd5, %rd3, %rd4;
        ld.global.f32 %f1, [%rd5];
        st.shared.f32 [%r4], %f1;
        bar.sync 0;

        ld.shared.f32 %f2, [%r4];
        cvta.to.global.u64 %rd6, %rd2;
        add.s64 %rd5, %rd6, %rd4;
        st.global.f32 [%rd5], %f2;
        ret;
    }
"#;

#[test]
fn verify_tier0_isa_and_tier1_llvm_ir_across_targets() {
    let targets = [IsaTarget::Gfx1100, IsaTarget::Gfx1103, IsaTarget::Gfx942];
    let kernels = [
        ("vector_add", VECTOR_ADD_PTX, false, false),
        ("saxpy", SAXPY_PTX, false, false),
        ("shared_reduce", SHARED_REDUCE_PTX, true, false),
        ("warp_reduce", WARP_REDUCE_PTX, true, true),
        ("row_softmax", ROW_SOFTMAX_PTX, false, true),
    ];

    let compiler = Compiler::with_cache_dir(None);
    let toolchain = LlvmToolchain::discover();

    for &target in &targets {
        for &(name, ptx_src, is_shared, uses_warp) in &kernels {
            // 1. Tier 1 SSA LLVM IR verification
            let art1 = compiler
                .compile_ptx_with_policy(ptx_src, target, TierPolicy::ForceTier1)
                .expect("Tier 1 compile should succeed");
            assert_eq!(art1.tier, CompileTier::Tier1Optimizing);
            let llvm = art1.llvm_ir.as_deref().expect("Tier 1 must produce LLVM IR");
            validate_llvm_ir(llvm).unwrap_or_else(|e| {
                panic!("LLVM IR validation failed for {name} on {target}: {e}\n{llvm}")
            });

            assert!(llvm.contains("alloca "));
            assert!(llvm.contains("addrspace(5)"));
            assert!(llvm.contains("load "));
            assert!(llvm.contains("store "));
            assert!(llvm.contains("ret void"));

            match name {
                "vector_add" => {
                    assert!(llvm.contains("fadd float"));
                    assert!(llvm.contains("icmp "));
                    assert!(llvm.contains("br i1"));
                }
                "saxpy" => {
                    assert!(llvm.contains("call float @llvm.fma.f32"));
                    assert!(llvm.contains("icmp "));
                    assert!(llvm.contains("br i1"));
                }
                "shared_reduce" => {
                    assert!(llvm.contains("call void @llvm.amdgcn.s.barrier()"));
                    assert!(llvm.contains("addrspace(3)"));
                }
                "warp_reduce" => {
                    assert!(llvm.contains("call i32 @llvm.amdgcn.ds.bpermute("));
                    assert!(llvm.contains("call void @llvm.amdgcn.s.barrier()"));
                    assert!(llvm.contains("atomicrmw fadd ptr addrspace(1)"));
                }
                "row_softmax" => {
                    assert!(llvm.contains("call float @llvm.exp2.f32("));
                    assert!(llvm.contains("fdiv float 1.0,"));
                    assert!(llvm.contains("call i32 @llvm.amdgcn.ds.bpermute("));
                    if target.native_wave_size() == 32 {
                        assert!(llvm.contains("call i32 @llvm.amdgcn.ballot.i32("));
                    } else {
                        assert!(llvm.contains("call i64 @llvm.amdgcn.ballot.i64("));
                    }
                }
                _ => unreachable!(),
            }

            if uses_warp && !target.is_rdna() {
                assert!(
                    llvm.contains("EmulatedWave32OnWave64"),
                    "missing Wave32-on-Wave64 legalization on {target} for {name}"
                );
            }

            if toolchain.has_codegen_tools() {
                let hsaco = compile_llvm_ir_to_hsaco(llvm, target)
                    .unwrap_or_else(|e| panic!("llc/ld.lld failed for {name} on {target}: {e}"));
                assert!(!hsaco.is_empty());
            }

            // 2. Tier 0 Native ISA & Binary Decoder verification
            let ptx = parse_ptx(ptx_src).expect("PTX parse should succeed");
            let ir = lower_ptx_to_ir(&ptx, target).expect("IR lowering should succeed");
            if uses_warp && !target.is_rdna() {
                assert_eq!(ir.kernels[0].wave_mode, WaveMode::EmulatedWave32OnWave64);
            }
            let native = compile_tier0_native(&ir, target).expect("Tier 0 codegen should succeed");
            assert_eq!(native.kernels.len(), 1);
            let k = &native.kernels[0];
            assert_eq!(k.name, name);

            let decoded = verify_encoded_machine_code(&k.machine_code, target).unwrap_or_else(|e| {
                panic!(
                    "binary machine code verification failed for {name} on {target}: {e}\n{}",
                    k.disasm
                )
            });
            assert!(!decoded.is_empty());
            assert!(k.disasm.contains("s_endpgm"));

            if !is_shared && !uses_warp {
                // 64-bit VOP3B carry chain for pointer arithmetic
                assert!(
                    k.disasm.contains("v_add_co_u32"),
                    "missing v_add_co_u32 for {name} on {target}:\n{}",
                    k.disasm
                );
                assert!(
                    k.disasm.contains("v_add_co_ci_u32"),
                    "missing v_add_co_ci_u32 for {name} on {target}:\n{}",
                    k.disasm
                );

                // EXEC-mask SIMT divergence save / mask / branch / restore
                if target.native_wave_size() == 32 {
                    assert!(k.disasm.contains("s_mov_b32 s9, exec_lo"));
                    assert!(k.disasm.contains("s_andn2_b32 exec_lo, exec_lo, vcc_lo"));
                    assert!(k.disasm.contains("s_cbranch_execz"));
                    assert!(k.disasm.contains("s_or_b32 exec_lo, exec_lo, s9"));
                } else {
                    assert!(k.disasm.contains("s_mov_b64 s[8:9], exec"));
                    assert!(k.disasm.contains("s_andn2_b64 exec, exec, vcc"));
                    assert!(k.disasm.contains("s_cbranch_execz"));
                    assert!(k.disasm.contains("s_or_b64 exec, exec, s[8:9]"));
                }
            } else if name == "shared_reduce" {
                assert!(k.disasm.contains("ds_write_b32"));
                assert!(k.disasm.contains("ds_read_b32"));
                assert!(k.disasm.contains("s_barrier"));
            } else if name == "warp_reduce" {
                assert!(k.disasm.contains("ds_bpermute_b32"));
                assert!(k.disasm.contains("global_atomic_add_u32"));
                assert!(k.disasm.contains("s_barrier"));
            } else if name == "row_softmax" {
                assert!(k.disasm.contains("v_exp_f32"));
                assert!(k.disasm.contains("v_rcp_f32"));
                assert!(k.disasm.contains("ds_bpermute_b32"));
            }

            // 3. Tier 0 HSACO packaging + optional llvm-objdump verification
            let art0 = compiler
                .compile_ptx_with_policy(ptx_src, target, TierPolicy::ForceTier0)
                .expect("Tier 0 HSACO compile should succeed");
            assert_eq!(art0.tier, CompileTier::Tier0Fast);
            let objdump_res = disassemble_hsaco_with_objdump(&art0.hsaco, target)
                .unwrap_or_else(|e| panic!("llvm-objdump rejected {name} on {target}: {e}"));
            if let Some(txt) = objdump_res {
                assert!(!txt.is_empty());
            }
        }
    }
}
