# AMDA

A CUDA-compatible, AMD-native GPU computing stack built from scratch in pure Rust (`std`-only, zero external crate dependencies).

AMDA bridges existing CUDA applications, `nvcc` fatbins, and cuBLAS/NVML workloads onto AMD RDNA (`gfx10xx`/`11xx`/`12xx`) and CDNA (`gfx9xx`) GPUs via a direct Linux `/dev/kfd` + HSA AQL runtime, an in-process Virtual AMD GPU (`MockBackend`), and a **tiered JIT + cross-kernel fusion compiler pipeline**:
- **Tier 0 (Zero-LLVM Native Fast Path)**: Fixed-point divergence analysis (`Uniform` -> SGPR, `Divergent` -> VGPR), linear-scan register allocation, and direct AMDGPU binary ISA emission (`GFX9/10/11/12` + 64-byte `amdhsa_kernel_descriptor_t`) in sub-millisecond time.
- **Tier 1 (Optimizing Promotion & Cross-Kernel Fusion Path)**: Automatic promotion of hot/fusible kernels and cross-kernel producer-consumer pipelines into SSA AMDGPU LLVM IR (`amdgcn-amd-amdhsa`) with optional external `llc`/`ld.lld` upgrade and tier-isolated SHA-256 caching.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 5-layer stack architecture, Tiered JIT, Cross-Kernel Fusion, and roadmap.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — Reproducible Tier 0 vs Tier 1 JIT, AQL Graph, Fusion, cuBLAS, and GGML Q4_0 benchmark methodology and numbers.
- [docs/COMPARISON.md](docs/COMPARISON.md) — Competitive analysis vs. ROCm/HIP, ZLUDA, SCALE, and BarraCUDA/Booth.
- [docs/HARDWARE_TESTING.md](docs/HARDWARE_TESTING.md) — AMD GPU buying/rental guide and local LLM capability matrix.

## Workspace Layout

- [`crates/amda-elf`](crates/amda-elf) — AMDGPU HSACO ELF64 parser/builder, 64-byte `AmdhsaKernelDescriptor` (`.rodata`), and MessagePack `PT_NOTE` (`amdhsa.kernels`) codec.
- [`crates/amda-compiler`](crates/amda-compiler) — Fatbin unpacker (uncompressed & pure-Rust LZ4), PTX lexer/parser (`bf16`, `f16x2`, `.v2`/`.v4`, `.func` inlining, `.local`/`.const`/`.shared`, warp `shfl.sync`/`vote.sync`, `dp4a`, `prmt`, `atom`), AMDA IR lowering & cross-kernel IR fusion pass (`fusion.rs`), Tier 0 native AMDGPU ISA backend + binary verifier, Tier 1 LLVM IR emitter, and SHA-256 artifact cache.
- [`crates/amda-hal`](crates/amda-hal) — `DeviceBackend` trait, 64-byte HSA AQL packet codecs & lock-free ring buffer, full Linux `/dev/kfd` + DRM render node ioctl execution backend, and the Virtual AMD GPU (`MockBackend`) supporting Wave32 & Wave64 targets.
- [`crates/amda-runtime`](crates/amda-runtime) — Slab VRAM allocator, VMM reservation registry, streams/events, module & kernarg packer, live Tier 0 -> Tier 1 hot promotion, `AMDA_TRACE=1` structured tracer, automatic graph-level fusion, and Dual-Backend Differential Oracle (`diff.rs`).
- [`crates/amda-cuda`](crates/amda-cuda) — Drop-in CUDA Driver (`cu*`, `cuGetProcAddress_v2`) and Runtime (`cuda*`, `__cudaRegisterFatBinary`, `__cudaPushCallConfiguration`, `cudaLaunchKernel`, pinned host memory, events, occupancy) `cdylib` + `rlib` with [amda_cuda.h](crates/amda-cuda/include/amda_cuda.h).
- [`crates/amda-cublas`](crates/amda-cublas) — Drop-in cuBLAS C ABI (`cublasCreate_v2`, `cublasSgemm_v2`, `cublasHgemm`, `cublasSgemmStridedBatched`, `cublasGemmEx`, `cublasGemmStridedBatchedEx`) backed by real GPU PTX GEMM kernels + [amda_cublas.h](crates/amda-cublas/include/amda_cublas.h).
- [`crates/amda-nvml`](crates/amda-nvml) — Drop-in NVML telemetry C ABI (`nvmlInit_v2`, `nvmlDeviceGetName`, `nvmlDeviceGetMemoryInfo`, `nvmlDeviceGetUUID`, `nvmlDeviceGetTemperature`, `nvmlDeviceGetUtilizationRates`, `nvmlDeviceGetPciInfo_v2`) + [amda_nvml.h](crates/amda-nvml/include/amda_nvml.h).
- [`crates/amda-cli`](crates/amda-cli) — CLI utility (`info`, `compile`, `demo`, `diff`, `bench`).
- [`scripts/install_dropin_shims.sh`](scripts/install_dropin_shims.sh) — Packages `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`, `libnvidia-ml.so.1` (and macOS `.dylib` equivalents).
- [`examples/c_vector_add.c`](examples/c_vector_add.c), [`examples/c_nvcc_abi.c`](examples/c_nvcc_abi.c), [`examples/c_llama_mvp.c`](examples/c_llama_mvp.c) — Standalone C99 examples linking AMDA shared libraries directly.

## Quickstart

Run the full workspace test suite, Clippy lints, and Linux cross-compile check (zero external crates, zero warnings):

```bash
RUSTFLAGS="-D warnings" cargo test --workspace
RUSTFLAGS="-D warnings" cargo check --workspace --target x86_64-unknown-linux-gnu
cargo clippy --workspace --all-targets -- -D warnings
```

Explore CLI commands (with optional `AMDA_TRACE=1` or `AMDA_MOCK_ARCH=gfx942`):

```bash
cargo run -p amda-cli -- info
AMDA_TRACE=1 cargo run -p amda-cli -- demo
cargo run -p amda-cli -- diff --lhs gfx1100 --rhs gfx942
cargo run --release -p amda-cli -- bench --iters 50 --target gfx1100
cargo run -p amda-cli -- compile path/to/kernel.ptx --target gfx1100 --tier 0 --emit asm
```

Build drop-in C shared libraries and run the Mini-LLM C99 example (`cudaMallocHost` + `cublasSgemm_v2` + nvcc PTX `row_softmax` + `nvml`):

```bash
cargo build -p amda-cuda -p amda-cublas -p amda-nvml
./scripts/install_dropin_shims.sh target/debug dist/lib
clang -std=c99 -Wall -Wextra -Werror \
  -Icrates/amda-cuda/include -Icrates/amda-cublas/include -Icrates/amda-nvml/include \
  examples/c_llama_mvp.c -Ldist/lib -lamda_cuda -lamda_cublas -lamda_nvml -lm \
  -Wl,-rpath,$(pwd)/dist/lib -o /tmp/amda_c_llama_mvp && /tmp/amda_c_llama_mvp
```
