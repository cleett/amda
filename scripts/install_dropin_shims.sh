#!/usr/bin/env bash
# Package AMDA shared libraries with standard CUDA / cuBLAS / NVML SONAME symlinks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -ge 1 ]]; then
    BUILD_DIR="$1"
else
    if [[ -f "${REPO_ROOT}/target/release/libamda_cuda.so" || -f "${REPO_ROOT}/target/release/libamda_cuda.dylib" ]]; then
        BUILD_DIR="${REPO_ROOT}/target/release"
    else
        BUILD_DIR="${REPO_ROOT}/target/debug"
    fi
fi

OUT_DIR="${2:-${REPO_ROOT}/dist/lib}"
mkdir -p "${OUT_DIR}"

UNAME_S="$(uname -s)"
LINKED=0

link_shim() {
    local src_path="$1"
    local dst_name="$2"
    if [[ -f "${src_path}" ]]; then
        ln -sf "$(cd "$(dirname "${src_path}")" && pwd)/$(basename "${src_path}")" "${OUT_DIR}/${dst_name}"
        echo "Linked ${OUT_DIR}/${dst_name} -> ${src_path}"
        LINKED=$((LINKED + 1))
    fi
}

if [[ "${UNAME_S}" == "Darwin" ]]; then
    CUDA_LIB="${BUILD_DIR}/libamda_cuda.dylib"
    CUBLAS_LIB="${BUILD_DIR}/libamda_cublas.dylib"
    NVML_LIB="${BUILD_DIR}/libamda_nvml.dylib"

    if [[ -f "${CUDA_LIB}" ]]; then
        link_shim "${CUDA_LIB}" "libamda_cuda.dylib"
        link_shim "${CUDA_LIB}" "libcuda.dylib"
        link_shim "${CUDA_LIB}" "libcuda.1.dylib"
        link_shim "${CUDA_LIB}" "libcudart.dylib"
        link_shim "${CUDA_LIB}" "libcudart.12.dylib"
    fi
    if [[ -f "${CUBLAS_LIB}" ]]; then
        link_shim "${CUBLAS_LIB}" "libamda_cublas.dylib"
        link_shim "${CUBLAS_LIB}" "libcublas.dylib"
        link_shim "${CUBLAS_LIB}" "libcublas.12.dylib"
        link_shim "${CUBLAS_LIB}" "libcublasLt.dylib"
        link_shim "${CUBLAS_LIB}" "libcublasLt.12.dylib"
    fi
    if [[ -f "${NVML_LIB}" ]]; then
        link_shim "${NVML_LIB}" "libamda_nvml.dylib"
        link_shim "${NVML_LIB}" "libnvidia-ml.dylib"
        link_shim "${NVML_LIB}" "libnvidia-ml.1.dylib"
    fi
else
    CUDA_LIB="${BUILD_DIR}/libamda_cuda.so"
    CUBLAS_LIB="${BUILD_DIR}/libamda_cublas.so"
    NVML_LIB="${BUILD_DIR}/libamda_nvml.so"

    if [[ -f "${CUDA_LIB}" ]]; then
        link_shim "${CUDA_LIB}" "libamda_cuda.so"
        link_shim "${CUDA_LIB}" "libcuda.so"
        link_shim "${CUDA_LIB}" "libcuda.so.1"
        link_shim "${CUDA_LIB}" "libcudart.so"
        link_shim "${CUDA_LIB}" "libcudart.so.12"
    fi
    if [[ -f "${CUBLAS_LIB}" ]]; then
        link_shim "${CUBLAS_LIB}" "libamda_cublas.so"
        link_shim "${CUBLAS_LIB}" "libcublas.so"
        link_shim "${CUBLAS_LIB}" "libcublas.so.12"
        link_shim "${CUBLAS_LIB}" "libcublasLt.so"
        link_shim "${CUBLAS_LIB}" "libcublasLt.so.12"
    fi
    if [[ -f "${NVML_LIB}" ]]; then
        link_shim "${NVML_LIB}" "libamda_nvml.so"
        link_shim "${NVML_LIB}" "libnvidia-ml.so"
        link_shim "${NVML_LIB}" "libnvidia-ml.so.1"
    fi
fi

if [[ "${LINKED}" -eq 0 ]]; then
    echo "error: no AMDA shared libraries found in ${BUILD_DIR}" >&2
    exit 1
fi

echo ""
echo "AMDA drop-in CUDA shims installed into: ${OUT_DIR}"
if [[ "${UNAME_S}" == "Darwin" ]]; then
    echo "Run CUDA-linked binaries with:"
    echo "  export DYLD_LIBRARY_PATH=\"${OUT_DIR}:\${DYLD_LIBRARY_PATH:-}\""
else
    echo "Run unmodified CUDA binaries against AMDA with:"
    echo "  export LD_LIBRARY_PATH=\"${OUT_DIR}:\${LD_LIBRARY_PATH:-}\""
fi
