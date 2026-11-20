# AMD GPU Testing & Local LLM Hardware Guide

Because AMDA supports two distinct AMD GPU families in [elf64.rs](../crates/amda-elf/src/elf64.rs)—**RDNA3/4 (Wave32 + WMMA)** and **CDNA (Wave64 + MFMA)**—and consumer RX 7900 cloud listings fluctuate, our standardized physical bringup path is:

> **Use a single-GPU AMD Instinct MI300X (`gfx942`, 192 GB HBM3) Droplet on AMD Developer Cloud (`~$1.99–$2.59 / GPU-hr`) for all Stage 1 physical hardware verification, keeping macOS + LLVM 22 (`Stage 0`) and `MockBackend` free for fast daily iteration.**

*(Note: macOS hosts lack `/dev/kfd` and use AMDA's in-process `MockBackend` plus Stage 0 real-LLVM `llc`/`ld.lld`/`llvm-objdump` HSACO verification. Stage 1 physical execution runs on Linux with `/dev/kfd` + `/dev/dri/renderD*`.)*

---

## 1. Local AMD GPUs for Daily `/dev/kfd` & RDNA Testing

| GPU | ISA Target | VRAM | Bus / Bandwidth | Approx. Price | Features | Verdict |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **Radeon RX 7600** | `gfx1102` (RDNA3) | 8 GB | 128-bit / ~288 GB/s | **~$260–$300** *(cheaper used)* | Wave32, WMMA, BF16 | **Cheapest entry point.** Full GFX11 ISA + Linux `/dev/kfd` for driver, AQL, and Tier 0 bringup. Verify recent kernel/ROCm `gfx1102` support. |
| **Radeon RX 9060 XT (16GB)** | `gfx1200` (RDNA4) | 16 GB | 128-bit / ~322 GB/s | **~$450–$500** | Wave32, WMMA, BF16, **FP8** | **Best new-card sweet spot.** First-class RDNA4 KFD/ROCm support, native FP8, and 16 GB VRAM. |
| **Radeon RX 7800 XT** *(used)* | `gfx1101` (RDNA3) | 16 GB | 256-bit / ~624 GB/s | **~$400–$450** | Wave32, WMMA, BF16 | Roughly **2× memory bandwidth** vs. RX 9060 XT at similar price for faster memory-bound kernels/LLMs. |
| **Radeon RX 9070 XT** | `gfx1201` (RDNA4) | 16 GB | 256-bit / ~640–645 GB/s | **~$700–$950+** | Wave32, WMMA, BF16, **FP8** | High-bandwidth RDNA4 + FP8; overkill solely for basic KFD tests unless high throughput is needed. |
| **Radeon RX 7900 XT / XTX** *(used)* | `gfx1100` (RDNA3) | **20–24 GB** | 320–384-bit / **800–960 GB/s** | **~$550–$750** | Wave32, WMMA, BF16 | **Best dual-purpose RDNA3 card** if running larger local LLMs alongside `gfx1100` kernel development. |
| **Radeon 780M APU** *(e.g. Picopulse R7H 24GB)* | `gfx1103` (RDNA3 APU) | **~16–20 GB** *(shared)* | LPDDR5 / ~102–120 GB/s | **All-in-one Mini-PC** | Wave32, WMMA, BF16, **Unified Mem** | **Best all-in-one Linux box** if you have no desktop PCIe host; great for KFD + `unified_memory` dev, slower for LLMs. |

---

## 2. Primary Physical Test Target: MI300X Droplet (`gfx942`) on AMD Developer Cloud

For all physical hardware tests, provision a **single-GPU MI300X Droplet** (192 GB HBM3, `gfx942`, Wave64, MFMA, FP8, 304 CUs, 8 XCCs) on **AMD Developer Cloud** (`~$1.99–$2.59 / GPU-hr`). Once SSHed into the Droplet:

```bash
# Ensure your user can access /dev/kfd and /dev/dri/renderD* (re-login if updated)
groups | grep -qE '\b(render|video)\b' || sudo usermod -aG render,video "$USER"

# Run Stage 1 end-to-end MI300X hardware bringup
./scripts/mi300x_stage1.sh
```

[mi300x_stage1.sh](../scripts/mi300x_stage1.sh) automatically checks `/dev/kfd`, `/dev/dri/renderD*`, `num_xcc`, and `mem_banks`, discovers `opt` / `llc` / `ld.lld` / `llvm-objdump` (`/opt/rocm/llvm/bin` or Rust `ld.lld`), runs `cargo test --workspace`, builds release drop-in CUDA/cuBLAS/NVML shims into `dist/lib`, probes live hardware via `amda info`, progressively validates `c_vector_add` (both `AMDA_TIER=1` and `AMDA_TIER=0`), `c_nvcc_abi`, and `c_llama_mvp`, and completes with `amda demo` against `AMDA_BACKEND=kfd`.

Other cloud providers with intermittent MI300X capacity for comparison:

| Cloud Provider | Approx. MI300X Rate | Notes |
| :--- | :---: | :--- |
| **AMD Developer Cloud** | **~$1.99–$2.59 / GPU-hr** | **Primary target** — 1x MI300X Droplets for Stage 1 AMDA hardware verification. |
| **Vultr (Spot)** | **~$1.85 / GPU-hr** *(~$3.99 on-demand)* | Lowest burst price when preemptible instances are acceptable. |
| **RunPod** | **~$2.39 / GPU-hr** | Low self-serve on-demand rate (console stock fluctuates). |
| **Hot Aisle** | **~$2.99 / GPU-hr** | Minute-metered bare-metal/VM nodes without enterprise sales friction. |

---

## 3. Hardware to Avoid for AMDA Bringup

- **Radeon RX 6000 (`gfx1030`, RDNA2)**: Uses older GFX10 instruction encodings (`s_endpgm = 0xBF810000` vs. `0xBFB00000` on GFX11+) and lacks WMMA, BF16, and FP8.
- **Instinct MI50 / Radeon VII (`gfx906`)**: Lacks MFMA matrix instructions (introduced in `gfx908`) and is dropped from modern ROCm stacks.
- **macOS / eGPU hosts**: Cannot exercise `/dev/kfd` or AQL user-mode queues directly.

---

## 4. Are These Local GPUs Sufficient for Local LLMs?

**Yes**—for **7B to 32B quantized (Q4 / FP8) local LLMs**, depending on VRAM capacity and memory bus bandwidth.

Single-user LLM token generation (decode) is memory-bandwidth bound, roughly following:

$$\text{tok/s} \approx 0.75 \times \frac{\text{Memory Bandwidth (GB/s)}}{\text{Active Model Weights (GB)}}$$

| GPU | VRAM | Bandwidth | Comfortable Q4 Models | Estimated Decode Speed (Q4) |
| :--- | :---: | :---: | :--- | :--- |
| **RX 7600** | 8 GB | 288 GB/s | **7B–8B** (short/medium context) | ~40–45 tok/s (8B) |
| **RX 9060 XT 16GB** | 16 GB | ~322 GB/s | **8B–14B** comfortably, **20B–24B** tight | ~45 tok/s (8B) / ~25 tok/s (14B) |
| **RX 7800 XT** *(used)* | 16 GB | 624 GB/s | **8B–14B** comfortably, **20B–24B** tight | ~85 tok/s (8B) / ~50 tok/s (14B) |
| **RX 9070 XT** | 16 GB | ~640 GB/s | **8B–14B** + **FP8** models | ~90 tok/s (8B) / ~55 tok/s (14B) |
| **RX 7900 XT / XTX** *(used)* | 20–24 GB | 800–960 GB/s | **27B–32B dense** & **30B MoE** (long context) | ~35–45 tok/s (32B dense) / **100+ tok/s (30B-A3B MoE)** |

### Key Trade-Offs for Local LLM Testing
- **8 GB (`RX 7600`)**: Adequate for running 7B–8B models and validating AMDA's inference kernels (`llama.cpp` / `vLLM`), but long context KV caches quickly exhaust 8 GB.
- **16 GB (`RX 9060 XT 16GB` vs. used `RX 7800 XT`)**: Both fit 14B Q4 models with generous KV cache; the **RX 9060 XT** adds RDNA4 + native **FP8** testing, while a **used RX 7800 XT** delivers roughly **2× higher tokens/sec** due to its 256-bit memory bus.
- **20–24 GB (`RX 7900 XT/XTX`)**: Unlocks **27B–32B dense models** and fast **30B MoE** models in VRAM alongside broad `gfx1100` tooling support.
- **70B+ Dense Models**: Do not fit in single-GPU 8–24 GB VRAM (~40+ GB at Q4). Running 70B+ locally requires multi-GPU VRAM or high-capacity unified memory (e.g., 128 GB Ryzen AI Max / Strix Halo APUs, which trade higher capacity for lower ~250 GB/s memory bandwidth).

### RDNA3 APU / Mini-PC (`gfx1103` — Radeon 780M, e.g. Picopulse R7H 24GB)
- **Why it works well for AMDA development**: If you don't already own a Linux desktop with a PCIe x16 slot, a Ryzen 7 Phoenix/Hawk Point mini-PC (`gfx1103`, 12 CUs) gives full GFX11 ISA parity (`s_endpgm = 0xBFB00000`, Wave32, WMMA, BF16), mainline Linux `/dev/kfd` queues, and **coherent `unified_memory = true`** testing in one quiet box over SSH (plus USB4 for future eGPU testing). Unlike stock ROCm (which historically required `HSA_OVERRIDE_GFX_VERSION=11.0.0`), AMDA supports `gfx1103` natively with no SKU allowlist.
- **Linux memory tuning**: Upstream `amdgpu`/TTM defaults cap iGPU GTT allocation at ~50% of system RAM (~12 GB on a 24 GB machine). Raising `ttm.pages_limit` / `amdgpu.gttsize` kernel parameters unlocks **~16–20 GB** of unified GPU memory out of 24 GB soldered LPDDR5.
- **Local LLM expectations**: Bandwidth (~102–120 GB/s dual-channel LPDDR5) and 12 CUs limit decode throughput compared to discrete cards—expect **~14–18 tok/s** on 7B–8B Q4 models and **~8–10 tok/s** on 14B Q4 models. (For comfortable 30B MoE local inference on an APU, 32 GB+ system RAM is recommended so weights + KV cache + OS don't swap.)
