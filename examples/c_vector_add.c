#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "amda_cuda.h"

#define CHECK_CU(expr)                                                           \
    do {                                                                         \
        CUresult _err = (expr);                                                  \
        if (_err != CUDA_SUCCESS) {                                              \
            fprintf(stderr, "CUDA error %d at %s:%d\n", _err, __FILE__, __LINE__); \
            return 1;                                                            \
        }                                                                        \
    } while (0)

typedef CUresult (*cuLaunchKernel_fn)(
    CUfunction, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, CUstream, void **, void **
);

static const char *VECTOR_ADD_PTX =
    ".version 7.8\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry vector_add(\n"
    "    .param .u64 a,\n"
    "    .param .u64 b,\n"
    "    .param .u64 c,\n"
    "    .param .u32 n\n"
    ") {\n"
    "    .reg .pred %p<2>;\n"
    "    .reg .f32 %f<4>;\n"
    "    .reg .b32 %r<6>;\n"
    "    .reg .b64 %rd<10>;\n"
    "    ld.param.u64 %rd1, [a];\n"
    "    ld.param.u64 %rd2, [b];\n"
    "    ld.param.u64 %rd3, [c];\n"
    "    ld.param.u32 %r1, [n];\n"
    "    mov.u32 %r2, %ctaid.x;\n"
    "    mov.u32 %r3, %ntid.x;\n"
    "    mov.u32 %r4, %tid.x;\n"
    "    mad.lo.s32 %r5, %r2, %r3, %r4;\n"
    "    setp.ge.s32 %p1, %r5, %r1;\n"
    "    @%p1 bra $L_EXIT;\n"
    "    cvta.to.global.u64 %rd4, %rd1;\n"
    "    cvta.to.global.u64 %rd5, %rd2;\n"
    "    cvta.to.global.u64 %rd6, %rd3;\n"
    "    mul.wide.s32 %rd7, %r5, 4;\n"
    "    add.s64 %rd8, %rd4, %rd7;\n"
    "    add.s64 %rd9, %rd5, %rd7;\n"
    "    ld.global.f32 %f1, [%rd8];\n"
    "    ld.global.f32 %f2, [%rd9];\n"
    "    add.f32 %f3, %f1, %f2;\n"
    "    add.s64 %rd8, %rd6, %rd7;\n"
    "    st.global.f32 [%rd8], %f3;\n"
    "$L_EXIT:\n"
    "    ret;\n"
    "}\n";

int main(void) {
    CHECK_CU(cuInit(0));

    CUdevice dev = 0;
    CHECK_CU(cuDeviceGet(&dev, 0));

    char name[128];
    CHECK_CU(cuDeviceGetName(name, (int)sizeof(name), dev));
    printf("AMDA C99 Example - Device 0: %s\n", name);

    CUcontext ctx = NULL;
    CHECK_CU(cuCtxCreate_v2(&ctx, 0, dev));

    void *pfn = NULL;
    int sym_status = -1;
    CHECK_CU(cuGetProcAddress_v2("cuLaunchKernel", &pfn, 12040, 0, &sym_status));
    cuLaunchKernel_fn launch = (cuLaunchKernel_fn)pfn;

    CUmodule mod = NULL;
    CHECK_CU(cuModuleLoadData(&mod, VECTOR_ADD_PTX));

    CUfunction fn = NULL;
    CHECK_CU(cuModuleGetFunction(&fn, mod, "vector_add"));

    const unsigned int N = 4096;
    const size_t bytes = (size_t)N * sizeof(float);

    float *ha = (float *)malloc(bytes);
    float *hb = (float *)malloc(bytes);
    float *hc = (float *)calloc(N, sizeof(float));
    if (!ha || !hb || !hc) {
        fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    for (unsigned int i = 0; i < N; i++) {
        ha[i] = (float)i * 0.5f;
        hb[i] = (float)i * 1.5f;
    }

    CUdeviceptr da = 0, db = 0, dc = 0;
    CHECK_CU(cuMemAlloc_v2(&da, bytes));
    CHECK_CU(cuMemAlloc_v2(&db, bytes));
    CHECK_CU(cuMemAlloc_v2(&dc, bytes));

    CHECK_CU(cuMemcpyHtoD_v2(da, ha, bytes));
    CHECK_CU(cuMemcpyHtoD_v2(db, hb, bytes));

    unsigned int n_arg = N;
    void *params[] = { &da, &db, &dc, &n_arg };
    CHECK_CU(launch(fn, 16, 1, 1, 256, 1, 1, 0, NULL, params, NULL));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH_v2(hc, dc, bytes));

    for (unsigned int i = 0; i < N; i++) {
        float expected = ha[i] + hb[i];
        if (fabsf(hc[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at %u: got %f, expected %f\n", i, hc[i], expected);
            return 1;
        }
    }

    printf("SUCCESS: Verified vector_add for N=%u elements via libamda_cuda.\n", N);

    CHECK_CU(cuMemFree_v2(da));
    CHECK_CU(cuMemFree_v2(db));
    CHECK_CU(cuMemFree_v2(dc));
    CHECK_CU(cuModuleUnload(mod));
    CHECK_CU(cuCtxDestroy_v2(ctx));

    free(ha);
    free(hb);
    free(hc);
    return 0;
}
