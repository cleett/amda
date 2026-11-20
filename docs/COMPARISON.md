# Competitive Analysis: AMDA vs. Market Alternatives & BarraCUDA

This document captures how AMDA compares to existing CUDA-on-AMD stacks and independent compiler efforts such as **BarraCUDA / Booth**, and explains how AMDA combines binary/source compatibility, full runtime ownership, and tiered JIT compilation to avoid their structural ceilings.

---

## 1. Why Existing Market Alternatives Stall Against CUDA

| Approach | Core Model | Why It Loses to CUDA |
| :--- | :--- | :--- |
| **ROCm / HIP** | `HIP -> CLR -> ROCr -> Thunk -> /dev/kfd` + static kernel libraries (`Tensile` / `Composable Kernel`) | Four-layer dispatch latency on small kernels; fragile install matrix; artificial SKU allowlists breaking consumer RDNA cards; gigabytes of pre-tuned kernel tables that degrade on untuned tensor shapes; permanently chasing NVIDIA features. |
| **ZLUDA** | Drop-in `libcuda.so` + PTX JIT translation | Translating already-lowered PTX 1:1 leaves 20–50% of AMD silicon performance behind, and cannot intercept or fuse NVIDIA's closed SASS libraries (`cuBLASLt`, `cuDNN`, FlashAttention-3). |
| **SCALE** | Source-level `.cu` frontend directly to AMDGPU | Produces stronger single-kernel codegen than PTX translation, but still binds developers to CUDA's thread-level SIMT model and relies on ROCm's runtime/library ecosystem underneath. |
| **Triton / Mojo / Tinygrad** | Alternative higher-level DSLs/compilers | Cleaner programming models, but weak drop-in compatibility with the long tail of existing CUDA C++ extensions (`vLLM`, `llama.cpp`, `bitsandbytes`, custom physics/rendering kernels). |

---

## 2. AMDA vs. BarraCUDA / Booth

**BarraCUDA** (created by Zane Hambly and later expanded into **Booth**) is an independent C99 compiler that compiles CUDA (and in Booth, HIP/Triton) straight to native AMDGPU binaries (such as GFX11 / GFX942) without relying on LLVM.

| Dimension | **BarraCUDA / Booth** | **AMDA** |
| :--- | :--- | :--- |
| **Primary Scope** | **Compiler only** (`.cu` / HIP / Triton source → `.hsaco` / machine code). Depends on external HSA/driver loaders at runtime. | **Full vertical stack**: `libamda_cuda` (`cu*` / `cuda*` + `cuGetProcAddress_v2`), slab allocator, streams/events, pre-baked AQL execution graphs, direct `/dev/kfd` HAL, and tiered compiler. |
| **Implementation Language** | Pure **C99** (~15K LOC core). | Pure **Rust** core (memory- and thread-safe runtime, AQL rings, IR, Tier 0 ISA emitter) + thin C++ boundaries only where LLVM/Clang are integrated. |
| **Compatibility Entry** | **Source-level** custom C/CUDA parser subset. Cannot load prebuilt CUDA binaries, PTX modules, or Python wheels directly. | **Hybrid**: **PTX & fatbin JIT** (`crates/amda-compiler`) for unmodified CUDA binaries/Triton/NVRTC + planned **Clang `.cu` frontend** for full C++20 templates. |
| **Codegen Strategy** | **Zero LLVM**: custom SSA IR (`BIR`), divergence-aware VGPR/SGPR register allocation, direct GFX11/GFX942 binary encoding. | **Tiered JIT**: **Tier 0** zero-LLVM direct AMDGPU native backend (`GFX9/10/11/12`) + **Tier 1** optimizing AMDA Tile/LLVM AMDGPU pipeline. |
| **Cross-Kernel Optimization** | Single-module ahead-of-time compilation; cannot observe runtime stream sequences or fuse around math libraries. | **Runtime-compiler co-design**: owns `ExecutionGraph` and library shims so consecutive PTX + library ops can fuse into single AMDGPU kernels. |

### Where BarraCUDA Shines
1. **Zero-LLVM footprint & instant cold compilation**: No massive LLVM build/link dependency and sub-millisecond cold kernel compilation.
2. **Direct native ISA control**: Hand-crafted encoders emit real GFX11/GFX942 instruction streams with explicit SGPR/VGPR awareness immediately.
3. **Lightweight multi-target portability**: Booth cleanly maps a single compact SSA IR across AMD, NVIDIA, Tenstorrent, and CPU targets.

### Where AMDA Wins Long-Term
1. **Production C++ & PTX Compatibility**: Real CUDA codebases (`PyTorch` ATen, `CUTLASS`/`CuTe`, `vLLM`, `llama.cpp`) rely heavily on C++ templates, macros, and pre-lowered PTX/fatbins that a custom C99 parser subset cannot ingest without rewriting user code.
2. **Silicon Hazard & Peak-Performance Scaling**: Across RDNA2/3/4 and CDNA1/2/3/4, hardware hazards (`s_waitcnt`/`s_wait_alu`, dual-issue `VOPD` pairing, MFMA/WMMA pipeline latencies, AGPR occupancy) multiply rapidly; delegating Tier 1 low-level scheduling to LLVM AMDGPU lets AMDA focus engineering effort on high-leverage **Tile IR** and **cross-op fusion**.
3. **Full Execution & Memory Ownership**: Owning the 64-byte AQL ring buffer, `/dev/kfd` doorbells, slab allocator, and `ExecutionGraph` enables sub-microsecond launch latency, GPU-driven dynamic dispatch, and cross-boundary fusion.
4. **Rust Memory & Concurrency Safety**: Eliminates UAF and data-race classes inside multi-threaded driver handle tables, VMM mappings, and lock-free AQL rings.

---

## 3. How AMDA Synthesizes Both: Tiered JIT Compilation

Rather than choosing between BarraCUDA's speed/simplicity and LLVM's peak optimization, AMDA incorporates BarraCUDA's strongest architectural insights directly into [crates/amda-compiler](../crates/amda-compiler):

1. **Tier 0 — Fast Zero-LLVM Native Backend (`CompileTier::Tier0Fast`)**:
   - Runs fixed-point **divergence analysis** (`Uniform` → scalar `SGPR`s, `Divergent` → vector `VGPR`s) and linear-scan allocation in [regalloc.rs](../crates/amda-compiler/src/backend/regalloc.rs).
   - Directly emits real GFX9/10/11/12 binary instructions (SOPP/SOP1/SOP2/SMEM/VOP1/VOP2/VOP3/VOPC/GLOBAL/DS) and 64-byte `amdhsa_kernel_descriptor_t` `.rodata` descriptors in [amdgpu_isa.rs](../crates/amda-compiler/src/backend/amdgpu_isa.rs) in sub-millisecond time without touching LLVM.
2. **Tier 1 — Optimizing Promotion & Fusion (`CompileTier::Tier1Optimizing`)**:
   - [amda-runtime](../crates/amda-runtime/src/context.rs) tracks per-kernel launch frequency and automatically promotes hot kernels (`AMDA_PROMOTE_THRESHOLD`, default `16`), large kernels (`>256` IR instructions), Wave64 warp-emulation kernels, and fusible `ExecutionGraph` kernels to Tier 1 Tile/LLVM IR.
   - Tier 0 and Tier 1 artifacts are keyed separately in the SHA-256 `ArtifactCache` ([cache.rs](../crates/amda-compiler/src/cache.rs)).
