#!/usr/bin/env bash
# Stage 1 physical hardware bringup script for a single-GPU AMD Instinct MI300X (gfx942)
# Droplet on AMD Developer Cloud.
#
# Executes preflight /dev/kfd + sysfs topology checks, discovers LLVM 22/ROCm codegen tools,
# runs Stage 0 verification tests, builds AMDA release drop-in shims, and runs progressive
# live-KFD smoke tests (`info` -> `c_vector_add` Tier 1/0 -> `c_nvcc_abi` -> `c_llama_mvp` -> `amda demo`).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "================================================================"
echo " AMDA Stage 1 Hardware Bringup — MI300X (gfx942)"
echo "================================================================"
echo "Host   : $(uname -a)"
echo "User   : $(id -un) (groups: $(id -Gn))"

# ------------------------------------------------------------------
# [1/6] Kernel /dev/kfd + DRM + sysfs topology preflight
# ------------------------------------------------------------------
echo ""
echo "[1/6] Checking /dev/kfd, /dev/dri/renderD*, and KFD topology..."
if [[ ! -e /dev/kfd ]]; then
    echo "error: /dev/kfd does not exist (amdgpu kernel driver not loaded?)" >&2
    exit 1
fi
if [[ ! -r /dev/kfd || ! -w /dev/kfd ]]; then
    echo "error: insufficient permissions for /dev/kfd (try: sudo usermod -aG render,video \$USER)" >&2
    exit 1
fi

RENDER_NODES=(/dev/dri/renderD*)
if [[ ! -e "${RENDER_NODES[0]}" ]]; then
    echo "error: no /dev/dri/renderD* nodes found" >&2
    exit 1
fi
if [[ ! -r "${RENDER_NODES[0]}" || ! -w "${RENDER_NODES[0]}" ]]; then
    echo "error: insufficient permissions for ${RENDER_NODES[0]} (check render group membership)" >&2
    exit 1
fi

for prop in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [[ -f "${prop}" ]] || continue
    if grep -qE '^gfx_target_version[[:space:]]+90402$' "${prop}"; then
        node_dir="$(dirname "${prop}")"
        echo "Found MI300X (gfx942) topology node: ${node_dir}"
        grep -E '^(gpu_id|simd_count|simd_per_cu|max_waves_per_simd|lds_size_in_kb|gfx_target_version|drm_render_minor|num_xcc|local_mem_size)[[:space:]]' "${prop}" || true
        for bank_prop in "${node_dir}"/mem_banks/*/properties; do
            [[ -f "${bank_prop}" ]] || continue
            if grep -qE '^heap_type[[:space:]]+(1|2)$' "${bank_prop}"; then
                echo "  mem_bank $(basename "$(dirname "${bank_prop}")"): $(grep -E '^(heap_type|size_in_bytes)[[:space:]]' "${bank_prop}" | tr '\n' ' ')"
            fi
        done
    fi
done

# ------------------------------------------------------------------
# [2/6] Discover external LLVM toolchain (opt, llc, ld.lld, llvm-objdump)
# ------------------------------------------------------------------
echo ""
echo "[2/6] Discovering AMDGPU LLVM codegen tools..."
find_llvm_bin() {
    local candidates=("$@")
    local dirs=()
    IFS=':' read -ra path_dirs <<< "${PATH:-}"
    dirs+=("${path_dirs[@]}" "/opt/rocm/llvm/bin" "/usr/local/opt/llvm/bin")
    for d in "${dirs[@]}"; do
        [[ -d "${d}" ]] || continue
        for c in "${candidates[@]}"; do
            if [[ -x "${d}/${c}" ]]; then
                echo "${d}/${c}"
                return 0
            fi
        done
    done
    return 1
}

if [[ -z "${AMDA_LLC:-}" ]]; then
    AMDA_LLC="$(find_llvm_bin llc llc-22 llc-21 llc-20 llc-19 llc-18 llc-17 || true)"
fi
if [[ -z "${AMDA_OPT:-}" ]]; then
    if [[ -n "${AMDA_LLC}" && -x "$(dirname "${AMDA_LLC}")/opt" ]]; then
        AMDA_OPT="$(dirname "${AMDA_LLC}")/opt"
    else
        AMDA_OPT="$(find_llvm_bin opt opt-22 opt-21 opt-20 opt-19 opt-18 opt-17 || true)"
    fi
fi
if [[ -z "${AMDA_LLD:-}" ]]; then
    AMDA_LLD="$(find_llvm_bin ld.lld ld.lld-22 ld.lld-21 ld.lld-20 ld.lld-19 ld.lld-18 ld.lld-17 || true)"
    if [[ -z "${AMDA_LLD}" ]] && command -v rustc >/dev/null 2>&1; then
        RUST_SYSROOT="$(rustc --print sysroot)"
        RUST_HOST="$(rustc -vV | sed -n 's/^host: //p')"
        RUST_LLD="${RUST_SYSROOT}/lib/rustlib/${RUST_HOST}/bin/gcc-ld/ld.lld"
        if [[ -x "${RUST_LLD}" ]]; then
            AMDA_LLD="${RUST_LLD}"
        fi
    fi
fi
if [[ -z "${AMDA_LLVM_OBJDUMP:-}" ]]; then
    AMDA_LLVM_OBJDUMP="$(find_llvm_bin llvm-objdump llvm-objdump-22 llvm-objdump-21 llvm-objdump-20 llvm-objdump-19 llvm-objdump-18 llvm-objdump-17 || true)"
fi

export AMDA_OPT AMDA_LLC AMDA_LLD AMDA_LLVM_OBJDUMP
echo "  AMDA_OPT          = ${AMDA_OPT:-<not found>}"
echo "  AMDA_LLC          = ${AMDA_LLC:-<not found>}"
echo "  AMDA_LLD          = ${AMDA_LLD:-<not found>}"
echo "  AMDA_LLVM_OBJDUMP = ${AMDA_LLVM_OBJDUMP:-<not found>}"

if [[ -z "${AMDA_LLC}" || -z "${AMDA_LLD}" ]]; then
    echo "warning: Tier 1 external LLVM codegen needs both llc and ld.lld (falling back to Tier 0 if absent)" >&2
fi

# ------------------------------------------------------------------
# [3/6] Stage 0 compiler & ISA verification tests
# ------------------------------------------------------------------
echo ""
echo "[3/6] Running cargo test --workspace..."
RUSTFLAGS="-D warnings" cargo test --workspace

# ------------------------------------------------------------------
# [4/6] Build release libraries + drop-in CUDA/cuBLAS/NVML shims
# ------------------------------------------------------------------
echo ""
echo "[4/6] Building release binaries & installing drop-in shims into dist/lib..."
cargo build --release -p amda-cuda -p amda-cublas -p amda-nvml -p amda-cli
./scripts/install_dropin_shims.sh target/release dist/lib

AMDA_CLI="${REPO_ROOT}/target/release/amda"
export LD_LIBRARY_PATH="${REPO_ROOT}/dist/lib:${LD_LIBRARY_PATH:-}"
export AMDA_BACKEND=kfd
export AMDA_WATCHDOG_MS="${AMDA_WATCHDOG_MS:-10000}"
export AMDA_TRACE="${AMDA_TRACE:-1}"

# ------------------------------------------------------------------
# [5/6] Hardware bringup: amda info + progressive C99 tests
# ------------------------------------------------------------------
echo ""
echo "[5/6] Probing MI300X via KfdBackend (amda info)..."
"${AMDA_CLI}" info

CC_BIN="${CC:-}"
if [[ -z "${CC_BIN}" ]]; then
    if command -v clang >/dev/null 2>&1; then
        CC_BIN=clang
    elif command -v gcc >/dev/null 2>&1; then
        CC_BIN=gcc
    else
        CC_BIN=cc
    fi
fi

BUILD_TMP="$(mktemp -d "${TMPDIR:-/tmp}/amda-stage1.XXXXXX")"
trap 'rm -rf "${BUILD_TMP}"' EXIT

CFLAGS=(-std=c99 -O2 -Wall -Wextra -Werror
    -I"${REPO_ROOT}/crates/amda-cuda/include"
    -I"${REPO_ROOT}/crates/amda-cublas/include"
    -I"${REPO_ROOT}/crates/amda-nvml/include")
LDFLAGS=(-L"${REPO_ROOT}/dist/lib" -lamda_cuda -lamda_cublas -lamda_nvml -ldl -lm
    "-Wl,-rpath,${REPO_ROOT}/dist/lib")

echo ""
echo "Compiling C99 hardware bringup examples with ${CC_BIN}..."
"${CC_BIN}" "${CFLAGS[@]}" examples/c_vector_add.c "${LDFLAGS[@]}" -o "${BUILD_TMP}/c_vector_add"
"${CC_BIN}" "${CFLAGS[@]}" examples/c_nvcc_abi.c "${LDFLAGS[@]}" -o "${BUILD_TMP}/c_nvcc_abi"
"${CC_BIN}" "${CFLAGS[@]}" examples/c_llama_mvp.c "${LDFLAGS[@]}" -o "${BUILD_TMP}/c_llama_mvp"

echo ""
echo "  (5a) c_vector_add (Tier 1 LLVM codegen on live MI300X)..."
AMDA_TIER=1 "${BUILD_TMP}/c_vector_add"

echo ""
echo "  (5b) c_vector_add (Tier 0 zero-LLVM native gfx942 ISA on live MI300X)..."
AMDA_TIER=0 "${BUILD_TMP}/c_vector_add"

echo ""
echo "  (5c) c_nvcc_abi (Tier 1 fatbin + __cudaPushCallConfiguration + cudaLaunchKernel)..."
AMDA_TIER=1 "${BUILD_TMP}/c_nvcc_abi"

echo ""
echo "  (5d) c_llama_mvp (Tier 1 cublasSgemm_v2 + row_softmax + NVML telemetry)..."
AMDA_TIER=1 "${BUILD_TMP}/c_llama_mvp"

# ------------------------------------------------------------------
# [6/6] Full AMDA CLI demo on live MI300X
# ------------------------------------------------------------------
echo ""
echo "[6/6] Running full AMDA demo on live MI300X (Driver/Runtime/Hot-Promote/Fusion/Q4_0/cuBLAS)..."
AMDA_TRACE=0 "${AMDA_CLI}" demo

echo ""
echo "================================================================"
echo " Stage 1 MI300X (gfx942) Bringup PASSED"
echo "================================================================"
