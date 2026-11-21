# AMDA Engineering Roadmap & Milestones

> **Refined Mission**: **Beat NVIDIA CUDA on TCO & tokens-per-dollar** on memory-capacity and bandwidth-bound AI inference, and **beat AMD ROCm on same-GPU decode `tokens/sec` (`$/token`), cloud cold-start TCO, UX, launch latency, and SKU coverage** via a zero-ROCm pure-Rust `/dev/kfd` + HSA AQL runtime.

---

## 1. Strategic Wedge & Non-Goals

Broadly beating the full NVIDIA CUDA ecosystem across every closed library (`cuDNN`, `cuFFT`, `cuSPARSE`, `TensorRT-LLM`, arbitrary `cubin`/SASS wheels, multi-node InfiniBand/RDMA training) with a small team is not achievable. AMDA wins by focusing ruthlessly on where AMD hardware already beats NVIDIA on economics and where stock ROCm leaves real performance and TCO on the table:

1. **Memory-Capacity & Bandwidth Arbitrage vs. CUDA (TCO / `$/tok` Edge)**:
   - A single **192 GB MI300X** (`gfx942`, 5.3 TB/s HBM3, ~$1.99–$2.59/hr) fits 70B BF16, 70B high-batch/long-context FP8, and 100B–235B quantized MoE models on **1 GPU** instead of forcing 2×–4× 80 GB H100s plus multi-GPU tensor-parallel sync overhead. Similarly, **8× MI300X** (1.536 TB HBM3) fits 405B–671B FP8 models on **1 node**.
   - High-VRAM consumer/workstation RDNA (`RX 7900 XTX`, `RX 9060/9070 XT`) and unified-memory APUs (`Radeon 780M`, **128 GB Strix Halo**) run large local models where ROCm breaks or restricts SKUs.
2. **ROCm Displacement & Same-GPU `$/token` Edge (Decode TPS, Fusion, & Cold-Start TCO)**:
   - Direct `/dev/kfd` + pre-baked 64-byte AQL graph dispatch (`<2–5 µs` host launch overhead), eliminating `HIP -> CLR -> ROCr` launch tax across 300–600 per-token decode kernels.
   - Cross-boundary GEMV/epilogue/RMSNorm/RoPE/SwiGLU fusion ([fusion.rs](../crates/amda-compiler/src/fusion.rs)) and runtime JIT specialization for quantized (`Q4`/`Q8`/`FP8`), MoE small-`M` (`M = 1..16`), and untuned shapes that miss static `Tensile` lookup tables.
   - Single `<20 MB` pure-Rust drop-in bundle (`dist/lib` shims, zero `/opt/rocm` userspace dependency, zero GPU allowlists), cutting 10–25 GB ROCm container pull and cold-start billing waste during autoscale bursts. *(Scope realism: on large-batch dense BF16/FP8 prefill `M >= 1024`, target **80–90%** of AMD's hand-scheduled `AITER` / `hipBLASLt` assembly rather than competing on prefill peak FLOPs.)*

### Explicit Non-Goals (MVP → v1.0)
- **No `cuDNN`, `cuFFT`, or `cuSPARSE` re-implementations** before single-node LLM inference wins.
- **No raw NVIDIA SASS (`cubin`) binary translation**—target PTX, high-level BLAS/math ABI interception, and later a direct `.cu` / Tile IR frontend.
- **No multi-node InfiniBand/RDMA training cluster parity** before single-node inference (`M1`–`M4`) proves real hardware speedup and $/token advantage.

---

## 2. Milestone Progression Overview

```text
M0 (Done) ──► M1 (Live MI300X Bringup) ──► M2 (llama.cpp 61-virtual + SDMA) ──► M3a (Decode TCO Win)
                                                                                       │
                     M5 (Clang .cu/Tile + 8x MI300X + Pro) ◄── M4 (All-SKU Package) ◄──┴── M3b (Dense Prefill + FA)
```

---

## 3. Milestones & Quantitative Exit Gates

### M0 — Foundation & Stage 0 Toolchain Validation *(Complete)*
**Goal**: Establish the zero-dependency Rust stack, Tier 0/1 compiler, clean-room CUDA/cuBLAS/NVML ABI shims, and real LLVM 22 static verification.

- **Deliverables (Complete)**:
  - Pure-Rust `std`-only 8-crate workspace ([Cargo.toml](../Cargo.toml), [Cargo.lock](../Cargo.lock)): `amda-elf`, `amda-compiler`, `amda-hal`, `amda-runtime`, `amda-cuda`, `amda-cublas`, `amda-nvml`, `amda-cli`.
  - Tier 0 native AMDGPU ISA backend (`GFX9/10/11/12`, `s[0:1]` dispatch / `s[2:3]` kernarg ABI, CDNA2/3 VGPR gran 8 + `rsrc3`) and Tier 1 SSA LLVM IR + `opt -O2` + `llc` + `ld.lld` + `llvm-objdump` Stage 0 validation.
  - Cross-kernel elementwise IR fusion ([fusion.rs](../crates/amda-compiler/src/fusion.rs)) in `ExecutionGraph`, pure-Rust LZ4 fatbin + `nvcc` registration ABI ([fatbin.rs](../crates/amda-compiler/src/fatbin.rs)), drop-in shims ([install_dropin_shims.sh](../scripts/install_dropin_shims.sh)), Apache-2.0 licensing ([LICENSE](../LICENSE), [CONTRIBUTING.md](../CONTRIBUTING.md)).
- **Verified Status**:
  - **41/41** workspace tests pass under both `MockBackend` and real LLVM 22 Stage 0 (`RUSTFLAGS="-D warnings"`), `x86_64-unknown-linux-gnu` check and `clippy --all-targets -- -D warnings` pass cleanly.

---

### M1 — Stage 1 Physical Silicon Bringup (`MI300X gfx942` Live `/dev/kfd`)
**Goal**: Prove end-to-end execution on real AMD Instinct MI300X (`gfx942`) hardware through AMDA's userspace `/dev/kfd` + AQL stack and eliminate software emulation as the sole truth source.

- **Key Work**:
  - Provision a single-GPU MI300X Droplet (`gfx942`, 192 GB HBM3, 304 CUs, `num_xcc = 8`) on AMD Developer Cloud and run [mi300x_stage1.sh](../scripts/mi300x_stage1.sh).
  - Validate multi-XCC AQL queue creation (`KfdQueueBufferSizes` with 8× XCC EOP/CWSR alignment and 2× `AQL_QUEUE_MEM` VA reservation), `mem_banks` 192 GiB discovery, coherent doorbell ringing, **`GTT | COHERENT | UNCACHED` control rings/signals vs. `VRAM | PUBLIC` HBM allocations**, `read_dispatch_id` wrap backpressure, and `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` flags on live silicon.
  - Verify both **Tier 1 (`AMDA_TIER=1`)** and **Tier 0 (`AMDA_TIER=0`)** compiled `.text` + `AmdhsaKernelDescriptor` execution (`c_vector_add`, `c_nvcc_abi`, `c_llama_mvp`, `amda demo`).
  - Upgrade `signal_wait` in [kfd.rs](../crates/amda-hal/src/kfd.rs) from pure CPU spin/sleep to hybrid short-spin + `AMDKFD_IOC_WAIT_EVENTS` interrupt blocking, and harden CWSR / fault diagnostics cleanly without hanging the node.
  - Add a **Measured Physical MI300X Hardware** section to [BENCHMARKS.md](BENCHMARKS.md).
- **Exit Gate**:
  - **10 consecutive runs** of `./scripts/mi300x_stage1.sh` pass cleanly (`AMDA_TIER=1` and `AMDA_TIER=0`) with **zero** `amdgpu` VM page faults (`UTCL2`) or queue preemption timeouts in `dmesg`.

---

### M2 — Binary Drop-In Parity & Hardware SDMA Engine (`llama.cpp` via PTX)
**Goal**: Run unmodified `llama.cpp` source built with PTX embedded (`-DCMAKE_CUDA_ARCHITECTURES="61-virtual" -DGGML_CUDA_FORCE_CUBLAS=ON`) on MI300X via `LD_LIBRARY_PATH=dist/lib` with hardware DMA memory copies. *(Defers `sm_80` Tensor-Core `mma.sync` / `cp.async` / `mmq` lowering to M3.)*

- **Key Work**:
  - **Exact CUDA 12 ABI Struct Parity**: Expand [CudaDeviceProp](../crates/amda-cuda/src/types.rs) and [amda_cuda.h](../crates/amda-cuda/include/amda_cuda.h) to match NVIDIA CUDA 12 binary struct offsets/sizes exactly (today's 26-field struct will misalign compiled third-party binaries calling `cudaGetDeviceProperties`).
  - **Expand C ABI Surface for `llama.cpp`**: Implement missing Driver/Runtime/cuBLAS symbols observed when linking `llama.cpp` (including `cudaStreamBeginCapture` / `EndCapture`, `cudaGraphExecUpdate`, multi-TU `__cudaRegisterFatBinary`, `cudaMemcpy2D` / `cudaMemcpy2DAsync`, `cudaMallocAsync` / `cudaFreeAsync`, `cudaFuncSetAttribute`, `cudaGetSymbolAddress`, `cublasGemmBatchedEx`, and minimal `cublasLt` entrypoints).
  - **Hardware SDMA Copy Engine**: Replace host-mapped CPU `memcpy` in `KfdBackend::{copy_htod, copy_dtoh, copy_dtod}` ([kfd.rs](../crates/amda-hal/src/kfd.rs))—especially `copy_dtod`'s GPU→host `Vec`→GPU bounce—with real `KFD_IOC_QUEUE_TYPE_SDMA` async hardware DMA packets and signals.
  - **Non-Tensor-Core PTX & Fatbin Completeness**: Add any missing PTX `dp4a`/scalar instructions/modifiers encountered in `61-virtual` `llama.cpp` GGML CUDA kernels and Zlib/Zstd decompression in [fatbin.rs](../crates/amda-compiler/src/fatbin.rs).
- **Exit Gate**:
  - Unmodified **`llama.cpp` source built with `-DCMAKE_CUDA_ARCHITECTURES="61-virtual" -DGGML_CUDA_FORCE_CUBLAS=ON`** (`llama-cli` / `llama-bench`) runs a 7B–8B `Q4_0` GGUF model end-to-end on MI300X via `LD_LIBRARY_PATH=dist/lib` with token output parity against CPU reference.
  - Hardware SDMA copies achieve **>20 GB/s PCIe H2D/D2H** and **>1 TB/s HBM D2D** (eliminating CPU copy bottlenecks).

---

### M3a — Decode-First TCO Win (Quantized/FP8 GEMV, Baseline MFMA/WMMA, & Fusion)
**Goal**: Ship the core memory-bandwidth and dispatch-latency wins first—beating H100 on >80 GB `$/token` and beating stock ROCm on launch- and intermediate-HBM-bound decode without blocking on year-long dense prefill scheduling.

- **Key Work**:
  - **Baseline Tile IR + Matrix Lowering (`amda-compiler`)**: Add first-class `MFMA` (`v_mfma_*` + AGPR allocation) for CDNA (`gfx908`/`90a`/`942`), `WMMA` for RDNA3/4 (`gfx11xx`/`12xx`), native FP8 (`e4m3`/`e5m2`), and `sm_80` `mma.sync`/`mmq` legalization for `llama.cpp` (`50–65%` of `hipBLASLt` baseline LDS-tiled GEMM).
  - **Bandwidth-Saturating Decode Kernels (`amda-cublas` / runtime)**: Replace the scalar 16×16 triple-loop GEMM in [crates/amda-cublas/src/kernels.rs](../crates/amda-cublas/src/kernels.rs) with tuned LDS-tiled MFMA/WMMA GEMMs plus bandwidth-saturating Q4/Q8/FP8 quantized GEMV kernels tailored for small-`M` (`M = 1..16`) and MoE shapes.
  - **Decode Fusion & `<2–5 µs` AQL Graphs ([fusion.rs](../crates/amda-compiler/src/fusion.rs))**: Expand cross-kernel fusion to **GEMV + bias/activation epilogue, RMSNorm, and RoPE/SwiGLU** inside pre-baked AQL `ExecutionGraph` replay (`<2–5 µs` host dispatch overhead per step).
- **Exit Gate**:
  - **Baseline Matrix & Graph Speed**: AMDA baseline LDS-tiled BF16/FP16/FP8 GEMMs reach **≥50–65% of `hipBLASLt`**, with `<2–5 µs` AQL graph launch overhead.
  - **70B Weight-Streaming Decode (MI300X HBM3 Physics)**: On 70B `Q4`/`BF16` decode, achieve **~3.8–4.2 TB/s** sustained weight streaming (**≥98–105% of stock `llama.cpp-hip`**, bounded realistically by MI300X's ~4.1–4.5 TB/s STREAM Triad ceiling) with zero `/opt/rocm` userspace installed, and deliver **≥25–35% better tokens/sec per dollar** on **1× MI300X (@ $1.99/hr)** vs. **2× H100 (80 GB)**.
  - **Small-Model / Fused Multi-Op Decode (7B–14B, MoE small-`M`)**: **Beat stock ROCm (`llama.cpp-hip`) by ≥15–25%** on `tokens/sec` (`$/token`) where AQL launch tax elimination and intermediate HBM fusion dominate.

---

### M3b — Dense Prefill & FlashAttention Optimization
**Goal**: Close the dense compute-bound prefill and long-context attention gap without delaying M3a's decode TCO release.

- **Key Work**:
  - **LLVM MFMA Scheduling & Warp-Pipelining**: Implement wave-group pipelining, `amdgcn_sched_group_barrier` / `iglp_opt` hints, LDS bank-conflict swizzling, and shape autotuning to break LLVM AMDGPU scheduler MFMA stalls (or expose an optional opt-in `hipBLASLt` prefill delegate while native Tile IR matures).
  - **Fused FlashAttention & Full GEMM+Epilogue Fusion**: Add tiled MFMA/WMMA FlashAttention (prefill + GQA decode) and GEMM+epilogue fusion in [fusion.rs](../crates/amda-compiler/src/fusion.rs).
- **Exit Gate**:
  - AMDA `BF16`/`FP16`/`FP8` large dense prefill GEMMs (`M >= 1024`) achieve **≥75–85% of stock `hipBLASLt` / `AITER` TFLOP/s** (natively or with optional prefill delegate), and FlashAttention matches or beats ROCm `llama.cpp` TTFT cleanly.

---

### M4 — Zero-Allowlist Multi-SKU Distribution (`RDNA3/4` & `128 GB Strix Halo`)
**Goal**: Ship a compact two-layer portable runtime that works uniformly across datacenter CDNA, consumer RDNA3/4, and unified-memory APUs—eliminating ROCm's 10–25 GB containers and SKU allowlists.

- **Key Work**:
  - Hardware validation across **4 ISA families** on live Linux `/dev/kfd`: `gfx942` (MI300X/MI325X), `gfx1100` (RX 7900 XTX), `gfx1200`/`1201` (RX 9060/9070 XT), and `gfx1103`/`gfx1151` (Radeon 780M / 128 GB Strix Halo APUs).
  - Implement true **zero-copy unified-memory (`GTT | COHERENT`) pointer aliasing** on APUs (`unified_memory = true`) to skip `HtoD`/`DtoH` staging copies entirely.
  - **Two-Layer Distribution (resolves LLVM size vs. `<20 MB` core)**:
    - **`<20 MB` Core Runtime** (`dist/lib` shims + `/dev/kfd` HAL + pre-compiled embedded HSACOs for built-in MFMA/WMMA/GEMV/Attention kernels + pure-Rust Tier 0 JIT for user PTX; **no `/opt/rocm`**).
    - **Optional Tier 1 LLVM Sidecar**: Discovers host LLVM (`opt`/`llc`/`ld.lld`) or ships an optional stripped AMDGPU-only compiler sidecar (`~50–80 MB`) when dynamic Tier 1 LLVM promotion of arbitrary user PTX is desired.
- **Exit Gate**:
  - The **`<20 MB` core drop-in package** passes `c_llama_mvp` + `llama.cpp` inference benchmarks across CDNA3, RDNA3, RDNA4, and RDNA3/3.5 APU Linux boxes with **zero** `HSA_OVERRIDE_GFX_VERSION`, **zero** `/opt/rocm` userspace libraries, and zero manual SKU overrides.

---

### M5 — Upstream Clang `.cu` / Tile Frontend, 8× MI300X xGMI, & Commercialization *(Post-M3 / Funded)*
**Goal**: Permanently eliminate `nvcc` EULA & closed-SASS walls, unlock 405B–671B FP8 single-node MI300X serving (`~$16–$24/hr` test nodes), and establish clean open-core commercial boundaries.

- **Key Work**:
  - **Upstream Clang AMDGPU / Triton Tile Frontend**: Integrate **upstream Clang's AMDGPU CUDA driver** (and/or Triton/Tile IR lowering)—explicitly *not* a custom C++20 `.cu` parser—so enterprise users compile CUDA/Tile source directly into AMDA without invoking NVIDIA's `nvcc` (resolving both closed-SASS kernels and NVIDIA's `nvcc`-output EULA clause).
  - **Multi-GPU Intra-Node xGMI (`amda-nccl` subset)**: Add multi-device `KFD` peer GPUVM mapping (`map_memory_to_gpu` across multiple `gpu_id`s), P2P DMA, and fast xGMI `AllReduce`/`AllGather` primitives for **8× MI300X** nodes (1.536 TB HBM3).
  - **Open-Core Product Boundary**:
    - *Apache-2.0 Open Core*: `amda-elf`, `amda-hal`, `amda-compiler` Tier 0/1 & fusion core, `amda-runtime`, `amda-cuda`, `amda-nvml`, baseline `amda-cublas`, CLI.
    - *Commercial Enterprise Layer*: Auto-tuned MFMA/FP8/FlashAttention kernel packs, multi-GPU xGMI `amda-nccl`, hardened Droplet/Kubernetes runtime, hang-recovery & profiler tooling.
- **Exit Gate**:
  - Single-node **8× MI300X** serves a **405B–671B FP8** model using zero NVIDIA tools and zero ROCm userspace packages, ready for neocloud OEM distribution or paid enterprise migration pilots.

---

## 4. Constraint & Risk Mitigation Summary

| Structural Constraint | Why Naive Clones Fail | AMDA Mitigation Milestone |
| :--- | :--- | :--- |
| **Closed SASS-only `cubin`s** | PTX translators cannot run precompiled `cubin` math kernels | **M2/M3/M5**: Intercept `cublas`/`cublasLt`/NVML ABI + upstream Clang `.cu` / Tile frontend |
| **Wave32 (`NVIDIA`) vs. Wave64 (`CDNA`)** | Wave-32 ballot/shuffle emulation wastes 50% lanes or adds shifts | **M3a**: Native MFMA/Tile IR & native Wave64 library kernels for heavy math (keep Wave32 legalization for compatibility) |
| **Hopper `WGMMA + TMA` vs. `MFMA + LDS`** | 1:1 PTX `mma` lowering stalls without hardware TMA; LLVM scheduler stalls MFMA chains | **M3a/M3b**: Explicit double-buffered LDS + AGPR in M3a, then warp-pipelining / `sched_group_barrier` or optional `hipBLASLt` prefill delegate in M3b |
| **`/dev/kfd` Hangs & CWSR Complexity** | Missing CWSR traps, uncached/coherent domain bugs, ring overwrites, or CPU copies hang/starve silicon | **M1/M2**: Multi-XCC CWSR/EOP sizing, `GTT \| COHERENT \| UNCACHED` control rings, `read_dispatch_id` backpressure, `WAIT_EVENTS`, hardware SDMA engine |
| **NVIDIA CUDA EULA Clause** | Prohibits translating `nvcc` output to non-NVIDIA platforms | **M0/M5**: Strict clean-room open-core stack today + upstream Clang `.cu` / Tile frontend so enterprise workloads bypass `nvcc` |
