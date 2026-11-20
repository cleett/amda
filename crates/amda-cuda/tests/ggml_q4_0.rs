use amda_compiler::{
    compile_tier0_native, lower_ptx_to_ir, parse_ptx, validate_llvm_ir,
    verify_encoded_machine_code, Compiler, TierPolicy, WaveMode,
};
use amda_elf::IsaTarget;
use amda_hal::MockBackend;
use amda_runtime::Runtime;
use std::sync::Arc;

/// GGML `block_q4_0` packed size (2B f16 scale + 16B packed nibbles for 32 weights).
const Q4_0_BLOCK_BYTES: usize = 18;
/// GGML `block_q8_1` packed size (2B f16 scale + 2B f16 weighted sum + 32B i8 quants).
const Q8_1_BLOCK_BYTES: usize = 36;
const QK: usize = 32;

const GGML_Q4_0_MATVEC_PTX: &str = r#"
    .version 7.8
    .target sm_80
    .address_size 64

    .visible .entry ggml_matvec_q4_0_q8_1(
        .param .u64 W_ptr,
        .param .u64 X_ptr,
        .param .u64 Y_ptr,
        .param .u32 k_blocks
    )
    {
        .reg .pred %p<5>;
        .reg .b16 %h<4>;
        .reg .f32 %f<12>;
        .reg .s32 %r<24>;
        .reg .b64 %rd<16>;

        ld.param.u64 %rd1, [W_ptr];
        ld.param.u64 %rd2, [X_ptr];
        ld.param.u64 %rd3, [Y_ptr];
        ld.param.u32 %r1, [k_blocks];

        mov.u32 %r2, %ctaid.x;
        mov.u32 %r3, %tid.x;
        and.b32 %r4, %r3, 3;
        shr.u32 %r5, %r3, 2;

        cvta.to.global.u64 %rd4, %rd1;
        cvta.to.global.u64 %rd5, %rd2;
        cvta.to.global.u64 %rd6, %rd3;

        mul.lo.u32 %r6, %r2, %r1;
        mul.lo.u32 %r6, %r6, 18;
        cvt.u64.u32 %rd7, %r6;
        add.u64 %rd8, %rd4, %rd7;

        mov.f32 %f1, 0f00000000;
        mov.u32 %r7, %r5;

    $L_BLK_LOOP:
        setp.ge.u32 %p1, %r7, %r1;
        @%p1 bra $L_REDUCE;

        mul.lo.u32 %r8, %r7, 18;
        cvt.u64.u32 %rd9, %r8;
        add.u64 %rd10, %rd8, %rd9;
        ld.global.f16 %h0, [%rd10];
        cvt.f32.f16 %f2, %h0;

        mul.lo.u32 %r9, %r7, 36;
        cvt.u64.u32 %rd11, %r9;
        add.u64 %rd12, %rd5, %rd11;
        ld.global.f16 %h1, [%rd12];
        cvt.f32.f16 %f3, %h1;
        ld.global.f16 %h2, [%rd12+2];
        cvt.f32.f16 %f4, %h2;

        mul.lo.u32 %r10, %r4, 4;
        add.u32 %r11, %r10, 2;
        cvt.u64.u32 %rd13, %r11;
        add.u64 %rd14, %rd10, %rd13;
        ld.global.u16 %r12, [%rd14];
        ld.global.u16 %r13, [%rd14+2];
        prmt.b32 %r14, %r12, %r13, 0x5410;

        and.b32 %r15, %r14, 0x0F0F0F0F;
        shr.u32 %r16, %r14, 4;
        and.b32 %r16, %r16, 0x0F0F0F0F;

        add.u32 %r17, %r10, 4;
        cvt.u64.u32 %rd13, %r17;
        add.u64 %rd15, %rd12, %rd13;
        ld.global.u32 %r18, [%rd15];
        ld.global.u32 %r19, [%rd15+16];

        dp4a.s32.s32 %r20, %r15, %r18, 0;
        dp4a.s32.s32 %r20, %r16, %r19, %r20;
        cvt.rn.f32.s32 %f5, %r20;

        mul.f32 %f6, %f2, %f3;
        fma.rn.f32 %f1, %f6, %f5, %f1;

        setp.eq.u32 %p2, %r4, 0;
        @!%p2 bra $L_NEXT_BLK;
        mul.f32 %f7, %f2, %f4;
        mul.f32 %f8, %f7, 0f41000000;
        sub.f32 %f1, %f1, %f8;

    $L_NEXT_BLK:
        add.u32 %r7, %r7, 8;
        bra $L_BLK_LOOP;

    $L_REDUCE:
        shfl.sync.down.b32 %f9, %f1, 16, 31, 0xffffffff;
        add.f32 %f1, %f1, %f9;
        shfl.sync.down.b32 %f9, %f1, 8, 31, 0xffffffff;
        add.f32 %f1, %f1, %f9;
        shfl.sync.down.b32 %f9, %f1, 4, 31, 0xffffffff;
        add.f32 %f1, %f1, %f9;
        shfl.sync.down.b32 %f9, %f1, 2, 31, 0xffffffff;
        add.f32 %f1, %f1, %f9;
        shfl.sync.down.b32 %f9, %f1, 1, 31, 0xffffffff;
        add.f32 %f1, %f1, %f9;

        setp.eq.u32 %p3, %r3, 0;
        @!%p3 bra $L_DONE;
        mul.wide.u32 %rd13, %r2, 4;
        add.u64 %rd14, %rd6, %rd13;
        st.global.f32 [%rd14], %f1;

    $L_DONE:
        ret;
    }
"#;

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

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);
    if exp == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

fn build_q4_0_q8_1_fixture(rows: usize, k_blocks: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    let mut w_bytes = Vec::with_capacity(rows * k_blocks * Q4_0_BLOCK_BYTES);
    let mut x_bytes = Vec::with_capacity(k_blocks * Q8_1_BLOCK_BYTES);

    let mut x_blocks: Vec<(f32, f32, [i8; QK])> = Vec::with_capacity(k_blocks);
    for blk in 0..k_blocks {
        let dx_f16 = f32_to_f16_bits(0.0625 * ((blk as f32) + 1.0));
        let dx = f16_bits_to_f32(dx_f16);
        let mut qs = [0i8; QK];
        let mut qsum = 0i32;
        for (i, q) in qs.iter_mut().enumerate() {
            let v = (((blk * 7 + i * 5) % 19) as i32) - 9;
            *q = v as i8;
            qsum += v;
        }
        let sx_f16 = f32_to_f16_bits(dx * (qsum as f32));
        let sx = f16_bits_to_f32(sx_f16);
        x_bytes.extend_from_slice(&dx_f16.to_le_bytes());
        x_bytes.extend_from_slice(&sx_f16.to_le_bytes());
        for &v in &qs {
            x_bytes.push(v as u8);
        }
        x_blocks.push((dx, sx, qs));
    }

    let mut cpu_y = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row_acc = 0.0f32;
        for (blk, &(dx, sx, ref x_qs)) in x_blocks.iter().enumerate() {
            let dw_f16 = f32_to_f16_bits(0.125 * ((((r + blk) % 5) as f32) - 2.0));
            let dw = f16_bits_to_f32(dw_f16);
            w_bytes.extend_from_slice(&dw_f16.to_le_bytes());
            let mut w_qs = [0u8; QK / 2];
            for (j, byte) in w_qs.iter_mut().enumerate() {
                let q_lo = ((r * 3 + blk * 5 + j) % 16) as u8;
                let q_hi = ((r * 7 + blk * 3 + j + 5) % 16) as u8;
                *byte = q_lo | (q_hi << 4);
            }
            w_bytes.extend_from_slice(&w_qs);

            let mut idot = 0i32;
            for j in 0..(QK / 2) {
                let lo = i32::from(w_qs[j] & 0x0F);
                let hi = i32::from((w_qs[j] >> 4) & 0x0F);
                idot += lo * i32::from(x_qs[j]);
                idot += hi * i32::from(x_qs[j + QK / 2]);
            }
            row_acc += dw * (dx * (idot as f32) - 8.0 * sx);
        }
        cpu_y.push(row_acc);
    }

    (w_bytes, x_bytes, cpu_y)
}

#[test]
fn ggml_q4_0_dp4a_prmt_matvec_matches_cpu_on_rdna3_and_cdna3() {
    let rows = 16usize;
    let k_blocks = 4usize; // K = 128
    let (w_bytes, x_bytes, cpu_y) = build_q4_0_q8_1_fixture(rows, k_blocks);

    // 1. Static verification across Tier 1 LLVM IR & Tier 0 ISA on Wave32 (gfx1100) + Wave64 (gfx942)
    let compiler = Compiler::with_cache_dir(None);
    let toolchain = amda_compiler::LlvmToolchain::discover();
    let ptx_mod = parse_ptx(GGML_Q4_0_MATVEC_PTX).expect("q4_0 PTX parse");
    for target in [IsaTarget::Gfx1100, IsaTarget::Gfx942] {
        let art1 = compiler
            .compile_ptx_with_policy(GGML_Q4_0_MATVEC_PTX, target, TierPolicy::ForceTier1)
            .expect("Tier 1 Q4_0 compile");
        let llvm = art1.llvm_ir.as_deref().expect("LLVM IR present");
        validate_llvm_ir(llvm).expect("Q4_0 LLVM IR validates");
        assert!(llvm.contains("@__amda_prmt_b32"));
        assert!(llvm.contains("@__amda_dp4a"));
        assert!(llvm.contains("@llvm.amdgcn.ds.bpermute"));
        if !target.is_rdna() {
            assert!(llvm.contains("EmulatedWave32OnWave64"));
        }
        if toolchain.has_codegen_tools() {
            let hsaco = amda_compiler::compile_llvm_ir_to_hsaco(llvm, target)
                .expect("llc/ld.lld Q4_0 compilation must succeed");
            assert!(!hsaco.is_empty());
        }

        let ir = lower_ptx_to_ir(&ptx_mod, target).expect("Q4_0 IR lower");
        if !target.is_rdna() {
            assert_eq!(ir.kernels[0].wave_mode, WaveMode::EmulatedWave32OnWave64);
        }
        let native = compile_tier0_native(&ir, target).expect("Tier 0 Q4_0 compile");
        let decoded =
            verify_encoded_machine_code(&native.kernels[0].machine_code, target).expect("ISA verify");
        assert!(!decoded.is_empty());

        let art0 = compiler
            .compile_ptx_with_policy(GGML_Q4_0_MATVEC_PTX, target, TierPolicy::ForceTier0)
            .expect("Tier 0 Q4_0 HSACO");
        if let Some(txt) =
            amda_compiler::disassemble_hsaco_with_objdump(&art0.hsaco, target).expect("objdump")
        {
            assert!(!txt.is_empty());
        }
    }

    // 2. End-to-end execution on MockBackend for both gfx1100 and gfx942
    for target in [IsaTarget::Gfx1100, IsaTarget::Gfx942] {
        let rt = Runtime::with_backend(Arc::new(MockBackend::with_isa(target)));
        let ctx = rt.create_context(0).expect("context");
        let module = ctx
            .load_module(GGML_Q4_0_MATVEC_PTX.as_bytes())
            .expect("load Q4_0 module");
        let func = module
            .get_function("ggml_matvec_q4_0_q8_1")
            .expect("get function");

        let d_w = ctx.alloc(w_bytes.len()).expect("alloc W");
        let d_x = ctx.alloc(x_bytes.len()).expect("alloc X");
        let d_y = ctx.alloc(rows * 4).expect("alloc Y");
        ctx.copy_htod(d_w, &w_bytes).expect("upload W");
        ctx.copy_htod(d_x, &x_bytes).expect("upload X");

        let arg_w = d_w.to_le_bytes();
        let arg_x = d_x.to_le_bytes();
        let arg_y = d_y.to_le_bytes();
        let arg_k = (k_blocks as u32).to_le_bytes();
        let args = [
            arg_w.as_slice(),
            arg_x.as_slice(),
            arg_y.as_slice(),
            arg_k.as_slice(),
        ];

        ctx.launch_kernel(&func, [rows as u32, 1, 1], [32, 1, 1], 0, None, &args)
            .expect("launch ggml_matvec_q4_0_q8_1");
        ctx.synchronize().expect("sync");

        let mut out_raw = vec![0u8; rows * 4];
        ctx.copy_dtoh(&mut out_raw, d_y).expect("download Y");
        for r in 0..rows {
            let gpu_val = f32::from_le_bytes([
                out_raw[r * 4],
                out_raw[r * 4 + 1],
                out_raw[r * 4 + 2],
                out_raw[r * 4 + 3],
            ]);
            assert!(
                (gpu_val - cpu_y[r]).abs() < 1e-3,
                "Q4_0 matvec mismatch on {target} row {r}: gpu={gpu_val} cpu={}",
                cpu_y[r]
            );
        }

        ctx.free(d_w).expect("free W");
        ctx.free(d_x).expect("free X");
        ctx.free(d_y).expect("free Y");
    }
}
