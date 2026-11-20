# AMDA Reproducible Benchmark Report

This report records reproducible micro-benchmark measurements produced by `amda-cli bench` (`crates/amda-cli/src/main.rs`) across RDNA3 (`gfx1100`, wave 32) and CDNA3 (`gfx942`, wave 64).

## Test Environment

- **Host OS**: macOS `15.7.9` (`x86_64`)
- **Host CPU**: `Intel(R) Core(TM) i5-8500 CPU @ 3.00GHz`
- **Rust Toolchain**: `rustc 1.94.0 (4a4ef493e 2026-03-02)` (`--release`)
- **Backend**: Virtual AMDA GPU (`MockBackend` software interpreter)
- **Physical Hardware Status**: **Not measured on physical AMD GPU hardware.** All timing figures below measure host compilation, AQL graph instantiation/replay, software ISA/IR emulation, and C-ABI/FFI marshaling overhead in `MockBackend`. Kernel-execution wall-clock timings on real `/dev/kfd` RDNA3/CDNA3 silicon will differ significantly (hardware execution is memory/ALU-bound on device rather than CPU-interpreted).

---

## Reproducing Locally

### Virtual GPU (`MockBackend`, macOS or Linux)

```bash
RUSTFLAGS="-D warnings" cargo run --release -q -p amda-cli -- bench --iters 50 --target gfx1100
RUSTFLAGS="-D warnings" cargo run --release -q -p amda-cli -- bench --iters 50 --target gfx942
```

### Physical AMD GPU on Linux (`/dev/kfd`)

On an `x86_64` Linux host with `/dev/kfd` accessible, real AMD LLVM tools (`llc` and `ld.lld` in `PATH`) for Tier 1 ELF emission, and a supported AMD GPU (`gfx1100`, `gfx1103`, or `gfx942`):

```bash
RUSTFLAGS="-D warnings" cargo run --release -p amda-cli -- info
RUSTFLAGS="-D warnings" cargo run --release -p amda-cli -- bench --iters 50 --target gfx1100
```

Set `AMDA_TRACE=1` to inspect Tier 0/Tier 1 JIT transitions, fusion passes, and dispatch packets. Set `AMDA_DISABLE_FUSION=1` to force unfused graph replay for A/B comparison.

---

## Measured Results (`--iters 50`)

### 1. RDNA3 (`gfx1100`, wave 32)

```text
AMDA Reproducible Benchmark Suite
=================================
Target ISA: gfx1100 (wave 32), Iterations: 50

[1] Tiered JIT Compile Latency (Cold Tier 0 vs Warm Cache Hit vs Cold Tier 1)
KERNEL           COLD_T0_US  WARM_HIT_US   COLD_T1_US      T1/T0
vector_add           3397.7        118.3       1208.8      0.36x
saxpy                 187.4        148.2        944.6      5.04x
row_softmax           152.5        126.4       1599.3     10.48x
q4_0_matvec           314.2        257.2       1701.8      5.42x

[2] Launch & Dispatch Overhead (saxpy, N=2048, 50 launches)
  Eager launch_kernel : 1173.57 us/launch | Pre-baked AQL Graph : 891.16 us/launch | Speedup : 1.32x

[3] Cross-Kernel IR Fusion (vector_add -> relu_scale, N=2048)
  Unfused : steps=2, dispatches=100, global_io=20 B/elem, 1800.83 us/iter
  Fused   : steps=1, dispatches=50, global_io=12 B/elem (-40% HBM traffic), 1851.74 us/iter, max_abs_err=0.0e0

[4] AI Library & Quantized Inference Micro-Benchmarks
  cuBLAS SGEMM (32x32x32 steady)    : 12.723 ms/gemm
  GGML Q4_0 x Q8_1 (16x128 cold+run): 0.914 ms
```

### 2. CDNA3 (`gfx942`, wave 64)

```text
AMDA Reproducible Benchmark Suite
=================================
Target ISA: gfx942 (wave 64), Iterations: 50

[1] Tiered JIT Compile Latency (Cold Tier 0 vs Warm Cache Hit vs Cold Tier 1)
KERNEL           COLD_T0_US  WARM_HIT_US   COLD_T1_US      T1/T0
vector_add            268.7        116.1       1033.4      3.85x
saxpy                  95.4         84.4        878.5      9.21x
row_softmax           140.4        121.8       1083.2      7.72x
q4_0_matvec           505.8        479.4       1237.2      2.45x

[2] Launch & Dispatch Overhead (saxpy, N=2048, 50 launches)
  Eager launch_kernel : 964.03 us/launch | Pre-baked AQL Graph : 745.02 us/launch | Speedup : 1.29x

[3] Cross-Kernel IR Fusion (vector_add -> relu_scale, N=2048)
  Unfused : steps=2, dispatches=100, global_io=20 B/elem, 1561.09 us/iter
  Fused   : steps=1, dispatches=50, global_io=12 B/elem (-40% HBM traffic), 1598.60 us/iter, max_abs_err=0.0e0

[4] AI Library & Quantized Inference Micro-Benchmarks
  cuBLAS SGEMM (32x32x32 steady)    : 12.239 ms/gemm
  GGML Q4_0 x Q8_1 (16x128 cold+run): 1.878 ms
```

---

## Analysis of Each Benchmark Section

### [1] Tiered JIT Compile Latency
- **Cold Tier 0** bypasses external LLVM tools and lowers PTX directly through `amda-compiler`'s internal SSA IR, divergent/uniform register allocator, and native AMDGPU machine-code encoder. Steady cold Tier 0 compilation takes **95.4–505.8 us** (excluding the very first call in the `gfx1100` process, `vector_add` at `3397.7 us`, which absorbs one-time process/allocator initialization).
- **Warm Cache Hit** reuses the compiled module (`84.4–479.4 us` end-to-end cache lookup + launch harness in this suite).
- **Cold Tier 1** runs LLVM IR emission and validation (`878.5–1701.8 us` in `MockBackend` mode, **2.45x–10.48x** slower than steady Tier 0 even before spawning external `llc`/`ld.lld` processes on Linux).

### [2] Eager Dispatch vs Pre-Baked AQL Graphs
- Pre-baking the `ExecutionGraph` eliminates per-launch parameter packing, kernel-object lookup, and submission staging overhead:
  - `gfx1100`: `1173.57 us` -> `891.16 us` (**1.32x speedup**)
  - `gfx942`: `964.03 us` -> `745.02 us` (**1.29x speedup**)

### [3] Cross-Kernel Elementwise IR Fusion (`vector_add -> relu_scale`)
- `amda-compiler::fusion` merges the producer/consumer pair into one fused AMDA IR kernel when instantiated through `ExecutionGraph::instantiate_with_compiler` / `Context::instantiate_graph`:
  - **Packet reduction**: `2` kernel steps (`100` total dispatches over 50 iterations) -> `1` fused step (`50` total dispatches, **-50% AQL dispatches**).
  - **Memory traffic reduction**: Eliminates the intermediate write (`4 B`) + consumer reload (`4 B`), dropping global memory traffic from **`20 B/elem`** (`A + B + tmp_write + tmp_read + scale = 5 * 4 B`) to **`12 B/elem`** (**`-40%` HBM/VRAM traffic**).
  - **Numerical fidelity**: `max_abs_err = 0.0e0` (bit-exact parity against unfused execution).
  - **Note on `us/iter` in `MockBackend`**: Under CPU emulation, unfused (`1800.83 us/iter` on `gfx1100`, `1561.09 us/iter` on `gfx942`) and fused (`1851.74 us/iter` on `gfx1100`, `1598.60 us/iter` on `gfx942`) execute in host RAM without physical VRAM latency or hardware queue submission barriers, so host interpreter loop differences (~2–3%) dominate instead of memory bandwidth savings. On physical AMD GPUs, eliminating 40% global memory traffic and 1 inter-kernel barrier directly improves throughput for memory-bound elementwise chains.

### [4] AI Library & Quantized Inference Micro-Benchmarks
- `cuBLAS SGEMM (32x32x32 steady)`: `12.723 ms/gemm` (`gfx1100`), `12.239 ms/gemm` (`gfx942`) routed through `amda-cublas` `cublasSgemm_v2` executing our column-major PTX `amda_gemm_f32` kernel on the virtual GPU.
- `GGML Q4_0 x Q8_1 (16x128 cold+run)`: `0.914 ms` (`gfx1100`), `1.878 ms` (`gfx942`) running an end-to-end 4-bit quantized matrix-vector multiply using PTX `prmt.b32`, `dp4a.s32.s32`, half-precision scale loads, and warp-shuffle reductions (`shfl.sync.down.b32`). On `gfx942`, wave-64 emulation of warp-32 collectives adds software interpreter work compared to native wave-32 on `gfx1100`.
