# AMDA Architecture Specification

AMDA is a CUDA-compatible, AMD-native GPU computing stack built from scratch in pure Rust (`std`-only, zero external crate dependencies). Its goal is twofold: deliver zero-ROCm UX, sub-millisecond dispatch, and zero-allowlist AMD SKU coverage today, while establishing a compilation and execution foundation designed to beat CUDA in TCO / tokens-per-dollar on high-VRAM and high-bandwidth AMD silicon (see [ROADMAP.md](ROADMAP.md)).

---

## 1. The 5-Layer Stack

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ Layer 5: CUDA / cuBLAS / NVML C ABI Shims                                  │
│  • libamda_cuda (crates/amda-cuda): libcuda / libcudart + nvcc fatbin ABI  │
│  • libamda_cublas (crates/amda-cublas): SGEMM/HGEMM/GemmEx via GPU PTX     │
│  • libamda_nvml (crates/amda-nvml): libnvidia-ml device & DRM telemetry    │
├────────────────────────────────────────────────────────────────────────────┤
│ Layer 4: Tiered JIT Compiler & Fusion Engine (crates/amda-compiler)        │
│  • Fatbin unpacker (uncompressed + pure-Rust LZ4) & PTX parser/lowerer     │
│  • Extended PTX IR: bf16/f16x2, v2/v4 vectors, .func inlining, .local,     │
│    dp4a/prmt/bfe/bfi, atomics, shfl.sync/vote.sync (Wave32-on-Wave64 CDNA) │
│  • Cross-Kernel IR Fusion pass (fusion.rs): elides intermediate HBM stores │
│  • Tier 0 (zero-LLVM): fixed-point divergence + SGPR/VGPR regalloc -> ISA  │
│  • Tier 1 (Optimizing): AMDGPU SSA LLVM IR + optional llc/ld.lld + SHA-256 │
├────────────────────────────────────────────────────────────────────────────┤
│ Layer 3: AMDGPU HSACO Binary & Metadata Layer (crates/amda-elf)            │
│  • ELF64 ET_DYN / EM_AMDGPU (224) builder & parser (.text, .rodata, .note) │
│  • 64-byte amdhsa_kernel_descriptor_t + MsgPack amdhsa.kernels codec       │
│  • Embedded .amda.ir section for portable JIT & virtual execution          │
├────────────────────────────────────────────────────────────────────────────┤
│ Layer 2: AMD-Native Core Execution Runtime (crates/amda-runtime)           │
│  • Power-of-two slab VRAM allocator, VMM registry, Streams & HSA Events    │
│  • Launch-counter Tier 0 -> Tier 1 hot promotion & AMDA_TRACE=1 telemetry  │
│  • Auto-fusing ExecutionGraph instantiation into pre-baked AQL packets     │
│  • Dual-Backend Differential Oracle (diff.rs)                              │
├────────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Direct Hardware Abstraction Layer (crates/amda-hal)               │
│  • 64B HSA AQL packets (KernelDispatchPacket, BarrierAndPacket)            │
│  • Lock-free user-space AQL ring buffer + coalescing KfdGpuVaAllocator     │
│  • Live Linux /dev/kfd + DRM renderD* GPUVM, AQL queues & user signals     │
│  • Cross-platform Virtual AMD GPU (MockBackend) for RDNA2/3/4 & CDNA1/2/3  │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                    AMD GPUs (RDNA2/3/4 Wave32 & CDNA1/2/3 Wave64)
```

---

## 2. Core Differentiators & Design Pillars

1. **Cross-Boundary IR Fusion (`amda-compiler/src/fusion.rs` + `ExecutionGraph`)**
   - Flows forward register provenance through elementwise PTX kernels, pairs producer stores and consumer loads sharing an intermediate global buffer (`find_fusible_buffer_params`), merges parameter layouts (`pack_fused_kernargs`), bridges the value in an SSA register (`v_bridge`) guarded by `v_prod_wrote`, and compiles a single AQL dispatch packet — eliminating intermediate VRAM/HBM round-trips automatically inside `ExecutionGraph::instantiate_with_compiler`.
2. **Tiered JIT Compilation (Zero-LLVM Fast Path + Optimizing LLVM Promotion)**
   - **Tier 0 (`Tier0Fast`)**: Pure-Rust fixed-point divergence analysis (`Uniform` -> SGPR, `Divergent` -> VGPR), linear-scan register allocation, and direct AMDGPU binary ISA emission (`GFX9/10/11/12` `.text` + `.rodata` descriptor) without invoking LLVM.
   - **Tier 1 (`Tier1Optimizing`)**: Escalates hot kernels (`AMDA_PROMOTE_THRESHOLD`), large kernels, CDNA Wave64 warp-emulation kernels, and fused graph pipelines to SSA AMDGPU LLVM IR (`amdgcn-amd-amdhsa`), optionally invoking external `llc`/`ld.lld` when present on host (`AMDA_DISABLE_EXTERNAL_LLVM=1` overrides).
3. **Hermetic Linux `/dev/kfd` + Direct AQL Execution (`amda-hal/src/kfd.rs`)**
   - Bypasses the multi-layer ROCm stack by opening `/dev/kfd` + `/dev/dri/renderD*` directly, allocating GPUVM apertures (`KfdGpuVaAllocator`), patching `amdhsa_kernel_descriptor_t` entry offsets for loaded HSACOs, ringing hardware AQL queue doorbells, and awaiting coherent 64-byte `HsaSignalStruct` completion with `AMDA_WATCHDOG_MS` timeout protection.
4. **Built-In Correctness & Differential Oracle (`amda-runtime/src/diff.rs` + `amda diff`)**
   - Runs identical PTX modules & host fixtures across Wave32 RDNA (`gfx1100`), APU (`gfx1103`), and Wave64 CDNA (`gfx942`) virtual GPUs (or live hardware vs. `MockBackend`) and validates against CPU reference tolerances down to FP rounding noise.

---

## 3. Crate Dependency Graph

```text
                       ┌──► amda-cublas (C ABI; runtime dlsym / table to host CUDA)
                       │
amda-cli ──────────────┼──► amda-cuda ──┐
                       │                ▼
                       └──► amda-nvml ──► amda-runtime ──► amda-hal ──► amda-compiler ──► amda-elf
```

- **`amda-elf`**: ELF64 `EM_AMDGPU` HSACO reader/writer, 64-byte `AmdhsaKernelDescriptor` (`.rodata`), MsgPack `.note` codec.
- **`amda-compiler`**: Uncompressed + LZ4 fatbin unpacker, PTX parser/lowerer, Wave32-on-Wave64 legalization, cross-kernel IR fusion pass, Tier 0 native ISA emitter + machine code decoder, Tier 1 LLVM IR emitter, SHA-256 cache.
- **`amda-hal`**: `DeviceBackend` trait, 64-byte AQL packet codecs/ring, complete Linux `/dev/kfd` backend, and `MockBackend` supporting cooperative warp shuffle/vote/barrier simulation.
- **`amda-runtime`**: Caching slab allocator, VMM registry, streams/events, hot promotion, auto-fusing `ExecutionGraph`, differential oracle, `AMDA_TRACE=1`.
- **`amda-cuda`**: `libamda_cuda` Driver + Runtime + `nvcc` fatbin registration C ABI.
- **`amda-cublas`**: `libamda_cublas` BLAS C ABI backed by GPU FP32/FP16/BF16 GEMM PTX kernels (process-shared host CUDA resolution via `dlsym` or `register_cuda_dispatch`).
- **`amda-nvml`**: `libamda_nvml` NVML telemetry C ABI with Linux sysfs DRM probes.
- **`amda-cli`**: CLI tool (`info`, `compile`, `demo`, `diff`, `bench`).

---

## 4. Milestone Status & Roadmap

- **Sprint 1 (Complete)**: Workspace architecture, HSACO ELF64/MsgPack codec, LZ4 fatbin codec, PTX -> SSA LLVM IR + optional external `llc`/`ld.lld`, Tier 0 divergence + regalloc + native AMDGPU ISA + verifier, `nvcc` fatbin registration ABI, live Linux `/dev/kfd` execution backend, and `AMDA_TRACE=1` telemetry.
- **Sprint 2 (Complete)**: Expanded PTX ops (`bf16`, `f16x2`, `v2`/`v4`, `.func` inlining, `.local`/`.const`/`.extern .shared`, math/bitwise/atomic/`shfl.sync`/`vote.sync`), Wave32-on-Wave64 CDNA legalization, full CUDA Driver/Runtime surface (`cudaGetDeviceProperties`, pinned host memory, events, occupancy, context stack, `cuPointerGetAttributes`), CUDA sample tests, and `install_dropin_shims.sh`.
- **Sprint 3 (Complete)**: `amda-cublas` (`cublasSgemm`, `cublasHgemm`, `cublasGemmEx`, strided batched GEMM on GPU), `amda-nvml`, GGML Q4_0 × Q8_1 (`dp4a`+`prmt`+`shfl.sync`) quantized MatVec across RDNA3 & CDNA3, and Dual-Backend Differential Oracle (`amda diff`).
- **Sprint 4 (Complete)**: Real Cross-Kernel IR Fusion (`amda-compiler/src/fusion.rs`) + automatic `ExecutionGraph` pipeline fusion, `amda bench` suite + `docs/BENCHMARKS.md`, and standalone C99 Mini-LLM MVP (`examples/c_llama_mvp.c`).
- **Next (Milestones M1–M5)**: Live `/dev/kfd` validation on Instinct MI300X (`gfx942`), CUDA 12 ABI parity + hardware SDMA queues for unmodified `llama.cpp`, MFMA/WMMA/FP8 Tile IR lowering, multi-SKU RDNA/Strix Halo packaging, and source `.cu` frontend + 8× MI300X xGMI scaling. See [ROADMAP.md](ROADMAP.md) for concrete deliverables and exit gates.
