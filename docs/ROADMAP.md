# AMDA Engineering Roadmap & Milestones

> **Refined Mission**: **Beat NVIDIA CUDA on TCO & tokens-per-dollar** on memory-capacity and bandwidth-bound AI inference, and **beat AMD ROCm on UX, launch latency, and SKU coverage** via a zero-ROCm pure-Rust `/dev/kfd` + HSA AQL runtime.

---

## 1. Strategic Wedge & Non-Goals

Broadly beating the full NVIDIA CUDA ecosystem across every closed library (`cuDNN`, `cuFFT`, `cuSPARSE`, `TensorRT-LLM`, arbitrary `cubin`/SASS wheels, multi-node InfiniBand/RDMA training) with a small team is not achievable. AMDA wins by focusing ruthlessly on where AMD hardware already beats NVIDIA on economics:

1. **Memory-Capacity & Bandwidth Arbitrage (TCO / `$/tok` Edge)**:
   - A single **192 GB MI300X** (`gfx942`, 5.3 TB/s HBM3, ~$1.99–$2.59/hr) fits 70B BF16, 70B high-batch/long-context FP8, and 100B–235B quantized MoE models on **1 GPU** instead of forcing 2×–4× 80 GB H100s plus multi-GPU tensor-parallel sync overhead. Similarly, **8× MI300X** (1.536 TB HBM3) fits 405B–671B FP8 models on **1 node**.
   - High-VRAM consumer/workstation RDNA (`RX 7900 XTX`, `RX 9060/9070 XT`) and unified-memory APUs (`Radeon 780M`, **128 GB Strix Halo**) run large local models where ROCm breaks or restricts SKUs.
2. **ROCm Displacement (UX & Latency Edge)**:
   - Single compact pure-Rust drop-in bundle (`dist/lib` shims + direct `/dev/kfd` + HSA AQL ring + Tier 0/1 JIT) with **no `/opt/rocm` userspace dependency**, zero GPU allowlists, sub-millisecond cold Tier 0 compilation, and pre-baked AQL graph dispatch.

### Explicit Non-Goals (MVP → v1.0)
- **No `cuDNN`, `cuFFT`, or `cuSPARSE` re-implementations** before single-node LLM inference wins.
- **No raw NVIDIA SASS (`cubin`) binary translation**—target PTX, high-level BLAS/math ABI interception, and later a direct `.cu` / Tile IR frontend.
- **No multi-node InfiniBand/RDMA training cluster parity** before single-node inference (`M1`–`M4`) proves real hardware speedup and $/token advantage.

---

## 2. Milestone Progression Overview

```text
M0 (Done) ──► M1 (Live MI300X Bringup) ──► M2 (Unmodified llama.cpp + SDMA) ──► M3 (MFMA/WMMA + HBM Win)
                                                                                     │
                                          M5 (Source/Tile + 8x MI300X + Pro) ◄── M4 (All-SKU Package)
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
  - Validate multi-XCC AQL queue creation (`KfdQueueBufferSizes` with 8× XCC EOP/CWSR alignment and 2× `AQL_QUEUE_MEM` VA reservation), `mem_banks` 192 GiB discovery, coherent doorbell ringing, and `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` flags on live silicon.
  - Verify both **Tier 1 (`AMDA_TIER=1`)** and **Tier 0 (`AMDA_TIER=0`)** compiled `.text` + `AmdhsaKernelDescriptor` execution (`c_vector_add`, `c_nvcc_abi`, `c_llama_mvp`, `amda demo`).
  - Upgrade `signal_wait` in [kfd.rs](../crates/amda-hal/src/kfd.rs) from pure CPU spin/sleep to hybrid short-spin + `AMDKFD_IOC_WAIT_EVENTS` interrupt blocking, and harden CWSR / fault diagnostics cleanly without hanging the node.
  - Add a **Measured Physical MI300X Hardware** section to [BENCHMARKS.md](BENCHMARKS.md).
- **Exit Gate**:
  - **10 consecutive runs** of `./scripts/mi300x_stage1.sh` pass cleanly (`AMDA_TIER=1` and `AMDA_TIER=0`) with **zero** `amdgpu` VM page faults (`UTCL2`) or queue preemption timeouts in `dmesg`.

---

### M2 — Binary Drop-In Parity & Hardware SDMA Engine (Unmodified `llama.cpp`)
**Goal**: Run an unmodified pre-compiled CUDA application (`llama.cpp`) on MI300X via `LD_LIBRARY_PATH=dist/lib` with hardware DMA memory copies.

- **Key Work**:
  - **Exact CUDA 12 ABI Struct Parity**: Expand [CudaDeviceProp](../crates/amda-cuda/src/types.rs) and [amda_cuda.h](../crates/amda-cuda/include/amda_cuda.h) to match NVIDIA CUDA 12 binary struct offsets/sizes exactly (today's 26-field struct will misalign compiled third-party binaries calling `cudaGetDeviceProperties`).
  - **Expand C ABI Surface for `llama.cpp`**: Implement missing Driver/Runtime/cuBLAS symbols observed when linking `llama.cpp` (including `cudaStreamBeginCapture` / `EndCapture`, `cudaMemcpy2D` / `cudaMemcpy2DAsync`, `cudaMallocAsync` / `cudaFreeAsync`, `cudaFuncSetAttribute`, `cudaGetSymbolAddress`, `cublasGemmBatchedEx`, and minimal `cublasLt` entrypoints).
  - **Hardware SDMA Copy Engine**: Replace host-mapped CPU `memcpy` in `KfdBackend::{copy_htod, copy_dtoh, copy_dtod}` ([kfd.rs](../crates/amda-hal/src/kfd.rs))—especially `copy_dtod`'s GPU→host `Vec`→GPU bounce—with real `KFD_IOC_QUEUE_TYPE_SDMA` async hardware DMA packets and signals.
  - **PTX 8.x & Fatbin Completeness**: Add any missing PTX instructions/modifiers encountered in `llama.cpp` GGML CUDA kernels and Zlib/Zstd decompression in [fatbin.rs](../crates/amda-compiler/src/fatbin.rs).
- **Exit Gate**:
  - An **unmodified CUDA-built `llama.cpp` (`llama-cli` / `llama-bench`)** runs a 7B–8B `Q4_0` GGUF model end-to-end on MI300X using `LD_LIBRARY_PATH=dist/lib` with exact/tolerant token output parity against CPU reference.
  - Hardware SDMA copies achieve **>20 GB/s PCIe H2D/D2H** and **>1 TB/s HBM D2D** (eliminating CPU copy bottlenecks).

---

### M3 — Matrix Cores (`MFMA` / `WMMA`), FP8, & Bandwidth-Saturating Inference
**Goal**: Turn hardware specs into real-world speedups—closing the performance gap where scalar PTX GEMMs and PTX 1:1 translation lag H100 and `hipBLASLt`, and proving lower `$/token` than H100.

- **Key Work**:
  - **Tile IR + Hardware Matrix Lowering (`amda-compiler`)**: Add first-class `MFMA` (`v_mfma_*` + AGPR allocation) for CDNA (`gfx908`/`90a`/`942`), `WMMA` for RDNA3/4 (`gfx11xx`/`12xx`), and native FP8 (`e4m3`/`e5m2` on `gfx942`/`12xx`), with XCC-aware tile swizzling and double-buffered LDS staging.
  - **High-Throughput GEMM & Attention Pack (`amda-cublas` / runtime library)**: Replace the scalar 16×16 triple-loop GEMM in [crates/amda-cublas/src/kernels.rs](../crates/amda-cublas/src/kernels.rs) with tuned LDS-tiled MFMA/WMMA GEMMs (BF16/FP16/FP8), bandwidth-saturating Q4/Q8/FP8 quantized GEMV kernels, and fused FlashAttention (prefill + GQA decode).
  - **Expanded Cross-Kernel Fusion ([fusion.rs](../crates/amda-compiler/src/fusion.rs))**: Lift elementwise-only restrictions to support **GEMM + bias/activation epilogue fusion**, RMSNorm/RoPE fusion, and shared-memory/reduction fusion inside `ExecutionGraph`.
  - **Low-Latency AQL Graph Replay**: Optimize pre-baked AQL ring packet submission to **<5 µs host dispatch overhead** per step so small-batch LLM decode is hardware-bound rather than CPU-bound.
- **Exit Gate**:
  - AMDA `BF16`/`FP16`/`FP8` GEMMs achieve **≥70–80% of stock `hipBLASLt` TFLOP/s** across core LLM shapes.
  - Single-GPU MI300X decode achieves **≥75% of 5.3 TB/s HBM3 bandwidth (~4.0 TB/s)** on 70B quantized/BF16 workloads.
  - **Headline TCO Proof**: On a workload exceeding 80 GB VRAM (e.g., Llama-3-70B BF16 or high-batch/long-context 70B FP8), **1× MI300X via AMDA (@ $1.99/hr)** delivers **≥25–35% better tokens/sec per dollar** than **2× H100 (80 GB)**, and matches or beats stock ROCm `llama.cpp` with **zero `/opt/rocm` userspace installed**.

---

### M4 — Zero-Allowlist Multi-SKU Distribution (`RDNA3/4` & `128 GB Strix Halo`)
**Goal**: Ship one tiny portable runtime that works uniformly across datacenter CDNA, consumer RDNA3/4, and unified-memory APUs—where ROCm packaging and SKU allowlists create constant user friction.

- **Key Work**:
  - Hardware validation across **4 ISA families** on live Linux `/dev/kfd`: `gfx942` (MI300X/MI325X), `gfx1100` (RX 7900 XTX), `gfx1200`/`1201` (RX 9060/9070 XT), and `gfx1103`/`gfx1151` (Radeon 780M / 128 GB Strix Halo APUs).
  - Implement true **zero-copy unified-memory (`GTT | COHERENT`) pointer aliasing** on APUs (`unified_memory = true`) to skip `HtoD`/`DtoH` staging copies entirely.
  - Package a single reproducible **`<20 MB` tarball / OCI runtime layer** (no multi-GB `/opt/rocm` install) usable by local devs and cloud Droplet images.
- **Exit Gate**:
  - One unified drop-in artifact passes `c_llama_mvp` + `llama.cpp` inference benchmarks across CDNA3, RDNA3, RDNA4, and RDNA3/3.5 APU Linux boxes with **zero** `HSA_OVERRIDE_GFX_VERSION` or manual SKU environment hacks.

---

### M5 — Source/Tile Frontend, 8× MI300X Intra-Node xGMI, & Open-Core Commercialization
**Goal**: Eliminate `nvcc` EULA & closed-SASS walls permanently, unlock 405B–671B FP8 single-node MI300X serving, and establish clean open-core commercial boundaries.

- **Key Work**:
  - **Source / Tile Frontend**: Add direct `.cu` ingestion (Clang AMDGPU frontend) and/or Python/Triton Tile IR lowering so enterprise users compile CUDA/Tile kernels directly into AMDA without invoking NVIDIA's `nvcc` toolchain (resolving both closed-SASS kernels and NVIDIA's `nvcc`-output EULA clause).
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
| **Closed SASS-only `cubin`s** | PTX translators cannot run precompiled `cubin` math kernels | **M2/M3/M5**: Intercept `cublas`/`cublasLt`/NVML ABI + source `.cu`/Tile frontend |
| **Wave32 (`NVIDIA`) vs. Wave64 (`CDNA`)** | Wave-32 ballot/shuffle emulation wastes 50% lanes or adds shifts | **M3**: Native MFMA/Tile IR & native Wave64 library kernels for heavy math (keep Wave32 legalization for compatibility) |
| **Hopper `WGMMA + TMA` vs. `MFMA + LDS`** | 1:1 PTX `mma` lowering stalls without hardware TMA | **M3**: Explicit double-buffered LDS + AGPR + XCC-swizzled MFMA schedules |
| **`/dev/kfd` Hangs & CWSR Complexity** | Missing CWSR traps, pointer races, or CPU copies hang/starve silicon | **M1/M2**: Multi-XCC CWSR/EOP sizing, `WAIT_EVENTS`, hardware SDMA engine, strict watchdog/diagnostics |
| **NVIDIA CUDA EULA Clause** | Prohibits translating `nvcc` output to non-NVIDIA platforms | **M0/M5**: Strict clean-room open-core stack today + `.cu`/Tile source frontend so enterprise workloads bypass `nvcc` |
