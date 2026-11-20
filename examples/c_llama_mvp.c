#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "amda_cuda.h"
#include "amda_cublas.h"
#include "amda_nvml.h"

#define CHECK_NVML(expr)                                                           \
    do {                                                                           \
        nvmlReturn_t _rc = (expr);                                                 \
        if (_rc != NVML_SUCCESS) {                                                 \
            fprintf(stderr, "NVML error %d (%s) at %s:%d\n",                       \
                    (int)_rc, nvmlErrorString(_rc), __FILE__, __LINE__);           \
            return 1;                                                              \
        }                                                                          \
    } while (0)

#define CHECK_CUDA(expr)                                                           \
    do {                                                                           \
        cudaError_t _err = (expr);                                                 \
        if (_err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %d (%s) at %s:%d\n",                       \
                    (int)_err, cudaGetErrorString(_err), __FILE__, __LINE__);      \
            return 1;                                                              \
        }                                                                          \
    } while (0)

#define CHECK_CUBLAS(expr)                                                         \
    do {                                                                           \
        cublasStatus_t _st = (expr);                                               \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                        \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_st,                \
                    __FILE__, __LINE__);                                           \
            return 1;                                                              \
        }                                                                          \
    } while (0)

static const char *ROW_SOFTMAX_PTX =
    ".version 7.8\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry row_softmax(\n"
    "    .param .u64 in_ptr,\n"
    "    .param .u64 out_ptr\n"
    ") {\n"
    "    .reg .f32 %f<10>;\n"
    "    .reg .b32 %r<6>;\n"
    "    .reg .b64 %rd<8>;\n"
    "    ld.param.u64 %rd1, [in_ptr];\n"
    "    ld.param.u64 %rd2, [out_ptr];\n"
    "    mov.u32 %r1, %ctaid.x;\n"
    "    mov.u32 %r2, %ntid.x;\n"
    "    mov.u32 %r3, %tid.x;\n"
    "    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
    "    cvta.to.global.u64 %rd3, %rd1;\n"
    "    mul.wide.u32 %rd4, %r4, 4;\n"
    "    add.u64 %rd5, %rd3, %rd4;\n"
    "    ld.global.f32 %f1, [%rd5];\n"
    "    mov.f32 %f2, %f1;\n"
    "    shfl.sync.bfly.b32 %f3, %f2, 16, 31, 0xffffffff;\n"
    "    max.f32 %f2, %f2, %f3;\n"
    "    shfl.sync.bfly.b32 %f3, %f2, 8, 31, 0xffffffff;\n"
    "    max.f32 %f2, %f2, %f3;\n"
    "    shfl.sync.bfly.b32 %f3, %f2, 4, 31, 0xffffffff;\n"
    "    max.f32 %f2, %f2, %f3;\n"
    "    shfl.sync.bfly.b32 %f3, %f2, 2, 31, 0xffffffff;\n"
    "    max.f32 %f2, %f2, %f3;\n"
    "    shfl.sync.bfly.b32 %f3, %f2, 1, 31, 0xffffffff;\n"
    "    max.f32 %f2, %f2, %f3;\n"
    "    sub.f32 %f4, %f1, %f2;\n"
    "    mul.f32 %f5, %f4, 0f3fb8aa3b;\n"
    "    ex2.approx.f32 %f6, %f5;\n"
    "    mov.f32 %f7, %f6;\n"
    "    shfl.sync.bfly.b32 %f3, %f7, 16, 31, 0xffffffff;\n"
    "    add.f32 %f7, %f7, %f3;\n"
    "    shfl.sync.bfly.b32 %f3, %f7, 8, 31, 0xffffffff;\n"
    "    add.f32 %f7, %f7, %f3;\n"
    "    shfl.sync.bfly.b32 %f3, %f7, 4, 31, 0xffffffff;\n"
    "    add.f32 %f7, %f7, %f3;\n"
    "    shfl.sync.bfly.b32 %f3, %f7, 2, 31, 0xffffffff;\n"
    "    add.f32 %f7, %f7, %f3;\n"
    "    shfl.sync.bfly.b32 %f3, %f7, 1, 31, 0xffffffff;\n"
    "    add.f32 %f7, %f7, %f3;\n"
    "    rcp.rn.f32 %f8, %f7;\n"
    "    mul.f32 %f9, %f6, %f8;\n"
    "    cvta.to.global.u64 %rd6, %rd2;\n"
    "    add.u64 %rd7, %rd6, %rd4;\n"
    "    st.global.f32 [%rd7], %f9;\n"
    "    ret;\n"
    "}\n";

static void softmax_kernel_host_stub(void) {}

int main(void) {
    /* 1. NVML device telemetry probe */
    CHECK_NVML(nvmlInit_v2());
    unsigned int nvml_cnt = 0;
    CHECK_NVML(nvmlDeviceGetCount_v2(&nvml_cnt));
    nvmlDevice_t nvml_dev = NULL;
    CHECK_NVML(nvmlDeviceGetHandleByIndex_v2(0, &nvml_dev));

    char dev_name[128] = {0};
    char uuid[64] = {0};
    nvmlMemory_t mem_info = {0, 0, 0};
    nvmlPciInfo_t pci_info;
    memset(&pci_info, 0, sizeof(pci_info));
    unsigned int temp_c = 0;
    CHECK_NVML(nvmlDeviceGetName(nvml_dev, dev_name, sizeof(dev_name)));
    CHECK_NVML(nvmlDeviceGetUUID(nvml_dev, uuid, sizeof(uuid)));
    CHECK_NVML(nvmlDeviceGetMemoryInfo(nvml_dev, &mem_info));
    CHECK_NVML(nvmlDeviceGetTemperature(nvml_dev, NVML_TEMPERATURE_GPU, &temp_c));
    CHECK_NVML(nvmlDeviceGetPciInfo_v2(nvml_dev, &pci_info));

    /* 2. CUDA Runtime device properties & pinned host buffers */
    struct cudaDeviceProp_st prop;
    memset(&prop, 0, sizeof(prop));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("AMDA Mini-LLM MVP: %s | UUID=%s | PCI=%s | Temp=%uC | CUs=%d | VRAM=%llu MiB\n",
           dev_name, uuid, pci_info.busId, temp_c,
           prop.multiProcessorCount, mem_info.total / (1024ULL * 1024ULL));

    /* Compute 4 rows x 32 logits via column-major cuBLAS GEMM (m=32, n=4, k=16), then row_softmax */
    const int ROWS = 4;
    const int COLS = 32;
    const int K = 16;
    const size_t x_bytes = (size_t)(COLS * K) * sizeof(float);
    const size_t w_bytes = (size_t)(K * ROWS) * sizeof(float);
    const size_t y_bytes = (size_t)(ROWS * COLS) * sizeof(float);

    void *h_pinned_x = NULL;
    void *h_pinned_w = NULL;
    void *h_pinned_y = NULL;
    CHECK_CUDA(cudaMallocHost(&h_pinned_x, x_bytes));
    CHECK_CUDA(cudaMallocHost(&h_pinned_w, w_bytes));
    CHECK_CUDA(cudaMallocHost(&h_pinned_y, y_bytes));

    float *hx = (float *)h_pinned_x;
    float *hw = (float *)h_pinned_w;
    for (int i = 0; i < COLS * K; i++) {
        hx[i] = ((float)((i % 13) - 6)) * 0.125f;
    }
    for (int i = 0; i < K * ROWS; i++) {
        hw[i] = ((float)((i % 9) - 4)) * 0.25f;
    }

    /* CPU reference: Y_rm[r * COLS + c] = sum_p X_cm[c + p * COLS] * W_cm[p + r * K] */
    float cpu_logits[4 * 32];
    float cpu_probs[4 * 32];
    for (int r = 0; r < ROWS; r++) {
        float max_v = -1e30f;
        for (int c = 0; c < COLS; c++) {
            float sum = 0.0f;
            for (int p = 0; p < K; p++) {
                sum += hx[c + p * COLS] * hw[p + r * K];
            }
            cpu_logits[r * COLS + c] = sum;
            if (sum > max_v) {
                max_v = sum;
            }
        }
        float sum_e = 0.0f;
        for (int c = 0; c < COLS; c++) {
            float ex = expf(cpu_logits[r * COLS + c] - max_v);
            cpu_probs[r * COLS + c] = ex;
            sum_e += ex;
        }
        for (int c = 0; c < COLS; c++) {
            cpu_probs[r * COLS + c] /= sum_e;
        }
    }

    void *d_x = NULL, *d_w = NULL, *d_logits = NULL, *d_probs = NULL;
    CHECK_CUDA(cudaMalloc(&d_x, x_bytes));
    CHECK_CUDA(cudaMalloc(&d_w, w_bytes));
    CHECK_CUDA(cudaMalloc(&d_logits, y_bytes));
    CHECK_CUDA(cudaMalloc(&d_probs, y_bytes));

    CHECK_CUDA(cudaMemcpy(d_x, h_pinned_x, x_bytes, cudaMemcpyDefault));
    CHECK_CUDA(cudaMemcpy(d_w, h_pinned_w, w_bytes, cudaMemcpyDefault));

    cudaEvent_t ev_start = NULL, ev_stop = NULL;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventRecord(ev_start, NULL));

    /* 3. cuBLAS projection layer */
    cublasHandle_t blas = NULL;
    CHECK_CUBLAS(cublasCreate_v2(&blas));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm_v2(
        blas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        COLS,
        ROWS,
        K,
        &alpha,
        (const float *)d_x,
        COLS,
        (const float *)d_w,
        K,
        &beta,
        (float *)d_logits,
        COLS));

    float *gpu_logits = (float *)malloc(y_bytes);
    if (!gpu_logits) {
        return 1;
    }
    CHECK_CUDA(cudaMemcpy(gpu_logits, d_logits, y_bytes, cudaMemcpyDefault));
    float gemm_max_err = 0.0f;
    for (int i = 0; i < ROWS * COLS; i++) {
        float err = fabsf(gpu_logits[i] - cpu_logits[i]);
        if (err > gemm_max_err) {
            gemm_max_err = err;
        }
        if (err > 1e-4f) {
            fprintf(stderr, "cuBLAS logit mismatch at %d: got=%f cpu=%f\n",
                    i, gpu_logits[i], cpu_logits[i]);
            free(gpu_logits);
            return 1;
        }
    }
    free(gpu_logits);

    /* 4. nvcc fatbin PTX kernel launch: row_softmax */
    void **fatbin_handle = __cudaRegisterFatBinary((void *)ROW_SOFTMAX_PTX);
    if (!fatbin_handle) {
        fprintf(stderr, "__cudaRegisterFatBinary failed\n");
        return 1;
    }
    __cudaRegisterFunction(
        fatbin_handle,
        (const char *)softmax_kernel_host_stub,
        "row_softmax",
        "row_softmax",
        -1,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL);
    __cudaRegisterFatBinaryEnd(fatbin_handle);

    dim3 grid = {(unsigned int)ROWS, 1, 1};
    dim3 block = {(unsigned int)COLS, 1, 1};
    void *sm_args[2] = {&d_logits, &d_probs};
    CHECK_CUDA(cudaLaunchKernel(
        (const void *)softmax_kernel_host_stub,
        grid,
        block,
        sm_args,
        0,
        NULL));
    CHECK_CUDA(cudaEventRecord(ev_stop, NULL));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    CHECK_CUDA(cudaMemcpy(h_pinned_y, d_probs, y_bytes, cudaMemcpyDefault));

    const float *gpu_probs = (const float *)h_pinned_y;
    float sm_max_err = 0.0f;
    for (int r = 0; r < ROWS; r++) {
        float row_sum = 0.0f;
        for (int c = 0; c < COLS; c++) {
            float g = gpu_probs[r * COLS + c];
            float p = cpu_probs[r * COLS + c];
            float err = fabsf(g - p);
            if (err > sm_max_err) {
                sm_max_err = err;
            }
            if (err > 1e-4f) {
                fprintf(stderr, "softmax mismatch at (%d,%d): got=%f cpu=%f\n", r, c, g, p);
                return 1;
            }
            row_sum += g;
        }
        if (fabsf(row_sum - 1.0f) > 1e-4f) {
            fprintf(stderr, "softmax row %d sum mismatch: %f\n", r, row_sum);
            return 1;
        }
    }

    printf("SUCCESS: c_llama_mvp (GEMM 4x16 * 16x32 + PTX row_softmax) "
           "gemm_max_err=%.2e softmax_max_err=%.2e elapsed=%.3fms\n",
           gemm_max_err, sm_max_err, elapsed_ms);

    /* 5. Clean shutdown */
    __cudaUnregisterFatBinary(fatbin_handle);
    CHECK_CUBLAS(cublasDestroy_v2(blas));
    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_logits));
    CHECK_CUDA(cudaFree(d_probs));
    CHECK_CUDA(cudaFreeHost(h_pinned_x));
    CHECK_CUDA(cudaFreeHost(h_pinned_w));
    CHECK_CUDA(cudaFreeHost(h_pinned_y));
    CHECK_NVML(nvmlShutdown());
    return 0;
}
