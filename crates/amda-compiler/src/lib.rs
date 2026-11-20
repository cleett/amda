//! AMDA compiler pipeline: Fatbin unpacking, PTX parsing, AMDA IR lowering,
//! BarraCUDA-inspired Tier 0 zero-LLVM native AMDGPU ISA backend, Tier 1 LLVM IR emitter,
//! HSACO packaging, and SHA-256 caching.

pub mod backend;
pub mod cache;
pub mod error;
pub mod fatbin;
pub mod fusion;
pub mod ir;
pub mod llvm_toolchain;
pub mod ptx;
pub mod tier;

pub use backend::{
    allocate_registers, analyze_divergence, compile_tier0_native, verify_encoded_machine_code,
    DecodedInst, Divergence, KernelAllocation, NativeKernelCode, NativeModuleCode, PhysReg,
    RegAllocSummary,
};
pub use cache::{sha256, to_hex, ArtifactCache, CacheStats};
pub use error::CompileError;
pub use fatbin::{
    build_lz4_ptx_fatbin, build_uncompressed_ptx_fatbin, inspect_module_bytes, lz4_compress_block,
    lz4_decompress_block, ExtractedInput,
};
pub use fusion::{
    find_fusible_buffer_params, fuse_elementwise_kernels, pack_fused_kernargs, FusedKernelPlan,
    FusedParamSource,
};
pub use ir::{
    analyze_kernel_fusion, emit_llvm_ir, lower_ptx_to_ir, validate_llvm_ir, AddressSpace,
    AtomicOp, BinOp, CmpOp, FusionCandidate, IrInst, IrKernel, IrModule, IrOp, IrOperand,
    IrParam, IrPredGuard, IrType, ShuffleMode, SpecialReg, UnaryMathOp, VoteMode, WaveMode,
};
pub use llvm_toolchain::{
    compile_llvm_ir_to_hsaco, disassemble_hsaco_with_objdump, LlvmToolchain,
};
pub use ptx::{parse_ptx, PtxModule};
pub use tier::{promote_threshold_from_env, select_tier, CompileTier, TierPolicy};

use amda_elf::{build_hsaco_with_rodata, parse_hsaco, AmdhsaKernelDescriptor, IsaTarget};
use std::path::PathBuf;

/// Complete compilation output produced by [`Compiler`].
#[derive(Debug, Clone)]
pub struct CompiledArtifact {
    /// Selected compilation tier (`Tier0Fast` or `Tier1Optimizing`).
    pub tier: CompileTier,
    /// Packaged AMDGPU HSACO ELF64 binary (with `PT_NOTE`, `.rodata` descriptors, `.text`, and `.amda.ir`).
    pub hsaco: Vec<u8>,
    /// Emitted LLVM IR targeting `amdgcn-amd-amdhsa` (populated on Tier 1).
    pub llvm_ir: Option<String>,
    /// Native AMDGPU disassembly + hex listing from Tier 0 codegen.
    pub native_asm: Option<String>,
    /// Per-kernel divergence-aware SGPR/VGPR allocation summary.
    pub regalloc: Vec<RegAllocSummary>,
    /// Lowered AMDA IR module.
    pub ir: IrModule,
    /// Whether the HSACO binary was served from cache.
    pub cache_hit: bool,
}

/// Top-level compiler driver coordinating parsing, IR lowering, tiered codegen, and caching.
pub struct Compiler {
    cache: ArtifactCache,
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    /// Creates a compiler backed by the environment-configured artifact cache.
    pub fn new() -> Self {
        Self {
            cache: ArtifactCache::from_env(),
        }
    }

    /// Creates a compiler with an explicit disk cache directory.
    pub fn with_cache_dir(cache_dir: Option<PathBuf>) -> Self {
        Self {
            cache: ArtifactCache::new(cache_dir),
        }
    }

    /// Returns artifact cache hit/miss counters.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Compiles UTF-8 PTX assembly using [`TierPolicy::from_env`].
    pub fn compile_ptx(
        &self,
        ptx_src: &str,
        target: IsaTarget,
    ) -> Result<CompiledArtifact, CompileError> {
        self.compile_ptx_with_policy(ptx_src, target, TierPolicy::from_env())
    }

    /// Compiles UTF-8 PTX assembly under an explicit [`TierPolicy`].
    pub fn compile_ptx_with_policy(
        &self,
        ptx_src: &str,
        target: IsaTarget,
        policy: TierPolicy,
    ) -> Result<CompiledArtifact, CompileError> {
        let ptx_mod = parse_ptx(ptx_src)?;
        let ir = lower_ptx_to_ir(&ptx_mod, target)?;
        let tier = select_tier(&ir, policy, 0, false, promote_threshold_from_env());

        let key = ArtifactCache::compute_key(target, tier, ptx_src.as_bytes());
        if let Some(cached_hsaco) = self.cache.get(&key) {
            if let Ok(parsed) = parse_hsaco(&cached_hsaco) {
                if let Some(ir_bytes) = &parsed.amda_ir_payload {
                    if let Ok(cached_ir) = IrModule::from_bytes(ir_bytes) {
                        return self.hydrate_cached_artifact(cached_hsaco, cached_ir, target, tier);
                    }
                }
            }
        }

        let artifact = self.emit_ir_for_tier(ir, target, tier, false)?;
        self.cache.put(key, &artifact.hsaco)?;
        Ok(artifact)
    }

    /// Compiles an already-lowered [`IrModule`] directly at `tier` (used for hot promotion).
    pub fn compile_ir_at_tier(
        &self,
        ir: &IrModule,
        target: IsaTarget,
        tier: CompileTier,
    ) -> Result<CompiledArtifact, CompileError> {
        let ir_bytes = ir.to_bytes();
        let key = ArtifactCache::compute_key(target, tier, &ir_bytes);
        if let Some(cached_hsaco) = self.cache.get(&key) {
            if let Ok(parsed) = parse_hsaco(&cached_hsaco) {
                if let Some(payload) = &parsed.amda_ir_payload {
                    if let Ok(cached_ir) = IrModule::from_bytes(payload) {
                        return self.hydrate_cached_artifact(cached_hsaco, cached_ir, target, tier);
                    }
                }
            }
        }

        let artifact = self.emit_ir_for_tier(ir.clone(), target, tier, false)?;
        self.cache.put(key, &artifact.hsaco)?;
        Ok(artifact)
    }

    fn hydrate_cached_artifact(
        &self,
        hsaco: Vec<u8>,
        ir: IrModule,
        target: IsaTarget,
        tier: CompileTier,
    ) -> Result<CompiledArtifact, CompileError> {
        let native = compile_tier0_native(&ir, target)?;
        let regalloc = native.kernels.iter().map(|k| k.summary.clone()).collect();
        let llvm_ir = match tier {
            CompileTier::Tier0Fast => None,
            CompileTier::Tier1Optimizing => Some(emit_llvm_ir(&ir, target)),
        };
        Ok(CompiledArtifact {
            tier,
            hsaco,
            llvm_ir,
            native_asm: Some(native.full_disasm),
            regalloc,
            ir,
            cache_hit: true,
        })
    }

    fn emit_ir_for_tier(
        &self,
        ir: IrModule,
        target: IsaTarget,
        tier: CompileTier,
        cache_hit: bool,
    ) -> Result<CompiledArtifact, CompileError> {
        let native = compile_tier0_native(&ir, target)?;
        let regalloc = native.kernels.iter().map(|k| k.summary.clone()).collect();
        let llvm_ir = match tier {
            CompileTier::Tier0Fast => None,
            CompileTier::Tier1Optimizing => Some(emit_llvm_ir(&ir, target)),
        };
        let ir_bytes = ir.to_bytes();
        let mut metas: Vec<_> = ir.kernels.iter().map(|k| k.metadata.clone()).collect();

        // When targeting Tier 1 on hosts with external LLVM (`opt` + `llc` + `ld.lld`), upgrade
        // `.text`/`.rodata` to LLVM-optimized AMDGPU machine code while retaining `.amda.ir`.
        let mut custom_text: Option<Vec<u8>> = None;
        let mut custom_rodata: Option<Vec<u8>> = None;
        if let Some(llvm) = llvm_ir.as_deref() {
            if std::env::var("AMDA_DISABLE_EXTERNAL_LLVM").as_deref() != Ok("1")
                && LlvmToolchain::discover().has_codegen_tools()
            {
                if let Ok(llvm_hsaco) = compile_llvm_ir_to_hsaco(llvm, target) {
                    if let Ok(parsed) = parse_hsaco(&llvm_hsaco) {
                        if let Some(tsec) = parsed.sections.iter().find(|s| s.name == ".text") {
                            if !tsec.data.is_empty() {
                                custom_rodata = parsed
                                    .sections
                                    .iter()
                                    .find(|s| s.name == ".rodata")
                                    .map(|rsec| {
                                        let mut rodata = rsec.data.clone();
                                        let t_start = i128::from(tsec.addr);
                                        let t_end = t_start + i128::from(tsec.data.len() as u64);
                                        for (idx, meta) in metas.iter_mut().enumerate() {
                                            let kd_off = idx * 64;
                                            let Some(slice) = rodata.get_mut(kd_off..kd_off + 64)
                                            else {
                                                break;
                                            };
                                            let Ok(raw): Result<[u8; 64], _> = (&*slice).try_into()
                                            else {
                                                break;
                                            };
                                            let mut kd = AmdhsaKernelDescriptor::from_bytes(&raw);
                                            if rsec.addr > 0 && tsec.addr > 0 {
                                                let cand = i128::from(rsec.addr)
                                                    + i128::from(kd_off as u64)
                                                    + i128::from(kd.kernel_code_entry_byte_offset);
                                                if cand >= t_start && cand < t_end {
                                                    kd.kernel_code_entry_byte_offset =
                                                        (cand - t_start) as i64;
                                                    slice.copy_from_slice(&kd.to_bytes());
                                                }
                                            }
                                            meta.group_segment_fixed_size =
                                                kd.group_segment_fixed_size;
                                            meta.private_segment_fixed_size =
                                                kd.private_segment_fixed_size;
                                            meta.kernarg_segment_size = kd.kernarg_size;
                                        }
                                        rodata
                                    });
                                custom_text = Some(tsec.data.clone());
                            }
                        }
                    }
                }
            }
        }

        let hsaco = build_hsaco_with_rodata(
            target,
            &metas,
            custom_rodata.as_deref().or(Some(&native.combined_rodata)),
            custom_text.as_deref().unwrap_or(&native.combined_text),
            Some(&ir_bytes),
        );

        Ok(CompiledArtifact {
            tier,
            hsaco,
            llvm_ir,
            native_asm: Some(native.full_disasm),
            regalloc,
            ir,
            cache_hit,
        })
    }

    /// Compiles arbitrary module bytes passed to `cuModuleLoadData` (PTX, fatbin, or native HSACO).
    pub fn compile_module_bytes(
        &self,
        data: &[u8],
        target: IsaTarget,
    ) -> Result<Vec<u8>, CompileError> {
        Ok(self.compile_module_artifact(data, target)?.hsaco)
    }

    /// Compiles arbitrary module bytes and returns the full [`CompiledArtifact`] when compiled from PTX/fatbin.
    pub fn compile_module_artifact(
        &self,
        data: &[u8],
        target: IsaTarget,
    ) -> Result<CompiledArtifact, CompileError> {
        match inspect_module_bytes(data)? {
            ExtractedInput::Hsaco(bytes) => {
                let parsed = parse_hsaco(bytes)
                    .map_err(|e| CompileError::InvalidFatbin(format!("invalid HSACO: {e}")))?;
                let ir = if let Some(payload) = &parsed.amda_ir_payload {
                    IrModule::from_bytes(payload)?
                } else {
                    IrModule {
                        target: parsed.isa,
                        kernels: Vec::new(),
                    }
                };
                Ok(CompiledArtifact {
                    tier: CompileTier::Tier1Optimizing,
                    hsaco: bytes.to_vec(),
                    llvm_ir: None,
                    native_asm: None,
                    regalloc: Vec::new(),
                    ir,
                    cache_hit: false,
                })
            }
            ExtractedInput::RawPtx(src) => self.compile_ptx(src, target),
            ExtractedInput::FatbinPtx(src) => self.compile_ptx(&src, target),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn sha256_nist_vectors() {
        assert_eq!(
            to_hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            to_hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn divergence_analysis_and_tier0_native_isa_codegen() {
        let ptx = parse_ptx(VECTOR_ADD_PTX).unwrap();
        let ir = lower_ptx_to_ir(&ptx, IsaTarget::Gfx1100).unwrap();
        let div = analyze_divergence(&ir.kernels[0]);
        let uniform_cnt = div.iter().filter(|&&d| d == Divergence::Uniform).count();
        let div_cnt = div.iter().filter(|&&d| d == Divergence::Divergent).count();
        assert!(uniform_cnt >= 6, "params and ctaid/ntid must be uniform");
        assert!(div_cnt >= 6, "tid/index/data must be divergent");

        let gfx11_native = compile_tier0_native(&ir, IsaTarget::Gfx1100).unwrap();
        assert_eq!(gfx11_native.kernels.len(), 1);
        let k11 = &gfx11_native.kernels[0];
        assert!(k11.machine_code.len() > 16);
        assert!(k11.descriptor.is_wave32());
        assert!(k11.disasm.contains("s_load_b64"));
        assert!(k11.disasm.contains("global_load_b32"));
        assert!(k11.disasm.contains("global_store_b32"));
        assert!(k11.disasm.contains("s_endpgm"));
        // Last 4 bytes on GFX11 must be s_endpgm = 0xBFB00000
        assert_eq!(
            &k11.machine_code[k11.machine_code.len() - 4..],
            &0xBFB0_0000u32.to_le_bytes()
        );

        let gfx942_native = compile_tier0_native(&ir, IsaTarget::Gfx942).unwrap();
        let k942 = &gfx942_native.kernels[0];
        assert!(!k942.descriptor.is_wave32());
        // Last 4 bytes on GFX942 must be s_endpgm = 0xBF810000
        assert_eq!(
            &k942.machine_code[k942.machine_code.len() - 4..],
            &0xBF81_0000u32.to_le_bytes()
        );

        let gfx1103_native = compile_tier0_native(&ir, IsaTarget::Gfx1103).unwrap();
        let k1103 = &gfx1103_native.kernels[0];
        assert!(k1103.descriptor.is_wave32());
        assert_eq!(
            &k1103.machine_code[k1103.machine_code.len() - 4..],
            &0xBFB0_0000u32.to_le_bytes()
        );
    }

    #[test]
    fn compile_tier0_and_tier1_with_isolated_cache_keys() {
        let compiler = Compiler::with_cache_dir(None);

        // Auto policy starts cold simple kernel at Tier 0 (zero-LLVM)
        let art0 = compiler
            .compile_ptx_with_policy(VECTOR_ADD_PTX, IsaTarget::Gfx1100, TierPolicy::Auto)
            .expect("Tier 0 vector_add compilation must succeed");
        assert_eq!(art0.tier, CompileTier::Tier0Fast);
        assert!(!art0.cache_hit);
        assert!(art0.llvm_ir.is_none());
        assert!(art0.native_asm.is_some());
        assert!(!art0.regalloc.is_empty());

        // Verify HSACO carries .rodata descriptor and .amda.ir
        let parsed_elf = parse_hsaco(&art0.hsaco).expect("HSACO must parse");
        assert_eq!(parsed_elf.isa, IsaTarget::Gfx1100);
        assert_eq!(parsed_elf.kernel_descriptors.len(), 1);
        assert!(parsed_elf.kernel_descriptors[0].is_wave32());

        // Tier 1 cache key must NOT collide with Tier 0 cache key
        let key0 = ArtifactCache::compute_key(
            IsaTarget::Gfx1100,
            CompileTier::Tier0Fast,
            VECTOR_ADD_PTX.as_bytes(),
        );
        let key1 = ArtifactCache::compute_key(
            IsaTarget::Gfx1100,
            CompileTier::Tier1Optimizing,
            VECTOR_ADD_PTX.as_bytes(),
        );
        assert_ne!(key0, key1);

        // Force Tier 1 must miss Tier 0 cache and emit LLVM IR
        let art1 = compiler
            .compile_ptx_with_policy(VECTOR_ADD_PTX, IsaTarget::Gfx1100, TierPolicy::ForceTier1)
            .expect("Tier 1 compilation must succeed");
        assert_eq!(art1.tier, CompileTier::Tier1Optimizing);
        assert!(!art1.cache_hit);
        let llvm = art1.llvm_ir.as_deref().expect("Tier 1 must emit LLVM IR");
        assert!(llvm.contains("target triple = \"amdgcn-amd-amdhsa\""));
        assert!(llvm.contains("@llvm.amdgcn.workitem.id.x"));
        assert!(llvm.contains("define amdgpu_kernel void @vector_add"));

        // Second Tier 0 compile hits Tier 0 cache
        let art0_hit = compiler
            .compile_ptx_with_policy(VECTOR_ADD_PTX, IsaTarget::Gfx1100, TierPolicy::ForceTier0)
            .expect("cached Tier 0 compile must succeed");
        assert!(art0_hit.cache_hit);
        assert_eq!(art0_hit.tier, CompileTier::Tier0Fast);
        assert_eq!(compiler.cache_stats(), CacheStats { hits: 1, misses: 2 });

        // SAXPY targeting CDNA3 gfx942 under ForceTier1
        let saxpy_art = compiler
            .compile_ptx_with_policy(SAXPY_PTX, IsaTarget::Gfx942, TierPolicy::ForceTier1)
            .expect("saxpy compilation must succeed");
        assert_eq!(saxpy_art.ir.kernels[0].wave_mode, WaveMode::NativeWave64);
        assert!(saxpy_art
            .llvm_ir
            .as_deref()
            .unwrap()
            .contains("\"target-cpu\"=\"gfx942\""));

        // Fusion detection + Auto escalation in fusion graph
        let fusion = analyze_kernel_fusion(&[&art0.ir.kernels[0], &saxpy_art.ir.kernels[0]]);
        assert_eq!(fusion.len(), 1);
        assert_eq!(
            select_tier(&art0.ir, TierPolicy::Auto, 0, true, 16),
            CompileTier::Tier1Optimizing
        );
        assert_eq!(
            select_tier(&art0.ir, TierPolicy::Auto, 16, false, 16),
            CompileTier::Tier1Optimizing
        );
    }

    #[test]
    fn fatbin_unpack_and_compile() {
        let fatbin = build_uncompressed_ptx_fatbin(VECTOR_ADD_PTX);
        let compiler = Compiler::with_cache_dir(None);
        let hsaco = compiler
            .compile_module_bytes(&fatbin, IsaTarget::Gfx1100)
            .expect("uncompressed fatbin must compile");
        let parsed = parse_hsaco(&hsaco).expect("HSACO from fatbin must parse");
        assert_eq!(parsed.kernels[0].name, "vector_add");
    }

    #[test]
    fn lz4_block_roundtrip_and_errors() {
        // Empty input
        let c_empty = lz4_compress_block(b"");
        assert!(c_empty.is_empty());
        assert_eq!(lz4_decompress_block(&c_empty, 0).unwrap(), b"");

        // Short input (< MF_LIMIT)
        let c_short = lz4_compress_block(b"hi");
        assert_eq!(lz4_decompress_block(&c_short, 2).unwrap(), b"hi");

        // Non-repeating ASCII
        let nonrep = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let c_nonrep = lz4_compress_block(nonrep);
        assert_eq!(
            lz4_decompress_block(&c_nonrep, nonrep.len()).unwrap(),
            nonrep
        );

        // Repetitive overlapping RLE data
        let rle = vec![b'A'; 4096];
        let c_rle = lz4_compress_block(&rle);
        assert!(
            c_rle.len() < rle.len() / 10,
            "expected strong RLE compression, got {} vs {}",
            c_rle.len(),
            rle.len()
        );
        assert_eq!(lz4_decompress_block(&c_rle, rle.len()).unwrap(), rle);

        // Full PTX source round-trip
        let ptx_bytes = VECTOR_ADD_PTX.as_bytes();
        let c_ptx = lz4_compress_block(ptx_bytes);
        assert!(
            c_ptx.len() < ptx_bytes.len(),
            "PTX should compress cleanly ({} -> {})",
            ptx_bytes.len(),
            c_ptx.len()
        );
        assert_eq!(
            lz4_decompress_block(&c_ptx, ptx_bytes.len()).unwrap(),
            ptx_bytes
        );

        // Zero back-reference offset must fail (literal len 1 'a', then offset 0)
        assert!(lz4_decompress_block(&[0x10, b'a', 0x00, 0x00], 64).is_err());

        // Out-of-bounds back-reference offset must fail (output len 1, offset 5)
        assert!(lz4_decompress_block(&[0x10, b'a', 0x05, 0x00], 64).is_err());

        // Output exceeding expected_uncompressed_len must fail
        assert!(lz4_decompress_block(&c_ptx, 16).is_err());
    }

    #[test]
    fn lz4_fatbin_compresses_and_compiles() {
        let uncompressed = build_uncompressed_ptx_fatbin(VECTOR_ADD_PTX);
        let compressed = build_lz4_ptx_fatbin(VECTOR_ADD_PTX);
        assert!(
            compressed.len() < uncompressed.len(),
            "LZ4 fatbin ({}) should be smaller than uncompressed ({})",
            compressed.len(),
            uncompressed.len()
        );

        let compiler = Compiler::with_cache_dir(None);
        let hsaco = compiler
            .compile_module_bytes(&compressed, IsaTarget::Gfx1100)
            .expect("LZ4 fatbin must unpack and compile");
        let parsed = parse_hsaco(&hsaco).expect("HSACO from LZ4 fatbin must parse");
        assert_eq!(parsed.kernels[0].name, "vector_add");
    }
}
