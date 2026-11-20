#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "amda_cuda.h"

#define CHECK_CUDA(expr)                                                          \
    do {                                                                          \
        cudaError_t _err = (expr);                                                \
        if (_err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Runtime error %d (%s) at %s:%d\n",              \
                    (int)_err, cudaGetErrorString(_err), __FILE__, __LINE__);     \
            return 1;                                                             \
        }                                                                         \
    } while (0)

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

static void write_u16_le(uint8_t *dst, uint16_t v) {
    dst[0] = (uint8_t)(v & 0xFFu);
    dst[1] = (uint8_t)((v >> 8) & 0xFFu);
}

static void write_u32_le(uint8_t *dst, uint32_t v) {
    dst[0] = (uint8_t)(v & 0xFFu);
    dst[1] = (uint8_t)((v >> 8) & 0xFFu);
    dst[2] = (uint8_t)((v >> 16) & 0xFFu);
    dst[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static void write_u64_le(uint8_t *dst, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        dst[i] = (uint8_t)((v >> (i * 8)) & 0xFFu);
    }
}

static size_t emit_ext_len(size_t rem, uint8_t *dst, size_t pos) {
    while (rem >= 255u) {
        dst[pos++] = 255u;
        rem -= 255u;
    }
    dst[pos++] = (uint8_t)rem;
    return pos;
}

static size_t lz4_compress_c99(const uint8_t *src, size_t src_len, uint8_t *dst) {
    if (src_len == 0) {
        return 0;
    }
    const size_t HASH_SIZE = 4096u;
    const size_t MIN_MATCH = 4u;
    const size_t LAST_LITERALS = 5u;
    const size_t MF_LIMIT = 12u;

    size_t out_pos = 0;
    if (src_len < MF_LIMIT) {
        dst[out_pos++] = (uint8_t)((src_len < 15u ? src_len : 15u) << 4);
        if (src_len >= 15u) {
            out_pos = emit_ext_len(src_len - 15u, dst, out_pos);
        }
        memcpy(dst + out_pos, src, src_len);
        return out_pos + src_len;
    }

    size_t table[4096];
    for (size_t i = 0; i < HASH_SIZE; i++) {
        table[i] = (size_t)-1;
    }

    size_t ip = 0;
    size_t anchor = 0;
    size_t match_limit = src_len - LAST_LITERALS;

    while (ip + MIN_MATCH <= match_limit) {
        uint32_t seq = (uint32_t)src[ip]
            | ((uint32_t)src[ip + 1] << 8)
            | ((uint32_t)src[ip + 2] << 16)
            | ((uint32_t)src[ip + 3] << 24);
        size_t h = (size_t)((seq * 2654435761u) >> 20);
        size_t cand = table[h];
        table[h] = ip;

        if (cand != (size_t)-1
            && ip > cand
            && (ip - cand) <= 65535u
            && memcmp(src + cand, src + ip, 4) == 0) {
            size_t mlen = 4u;
            while (ip + mlen < match_limit && src[cand + mlen] == src[ip + mlen]) {
                mlen++;
            }
            size_t lit_len = ip - anchor;
            size_t match_extra = mlen - MIN_MATCH;
            uint8_t token = (uint8_t)(((lit_len < 15u ? lit_len : 15u) << 4)
                | (match_extra < 15u ? match_extra : 15u));
            dst[out_pos++] = token;
            if (lit_len >= 15u) {
                out_pos = emit_ext_len(lit_len - 15u, dst, out_pos);
            }
            if (lit_len > 0) {
                memcpy(dst + out_pos, src + anchor, lit_len);
                out_pos += lit_len;
            }
            write_u16_le(dst + out_pos, (uint16_t)(ip - cand));
            out_pos += 2;
            if (match_extra >= 15u) {
                out_pos = emit_ext_len(match_extra - 15u, dst, out_pos);
            }
            ip += mlen;
            anchor = ip;
        } else {
            ip++;
        }
    }

    if (anchor < src_len) {
        size_t lit_len = src_len - anchor;
        dst[out_pos++] = (uint8_t)((lit_len < 15u ? lit_len : 15u) << 4);
        if (lit_len >= 15u) {
            out_pos = emit_ext_len(lit_len - 15u, dst, out_pos);
        }
        memcpy(dst + out_pos, src + anchor, lit_len);
        out_pos += lit_len;
    }

    return out_pos;
}

/* Host kernel stub mimicking nvcc-generated wrapper target */
static void host_vector_add_kernel(void) {}

int main(void) {
    /* 1. Compress PTX via C99 LZ4 encoder and build 0xBA55ED50 fatbin + __fatBinC_Wrapper_t */
    size_t ptx_raw_len = strlen(VECTOR_ADD_PTX) + 1u; /* include NUL terminator */
    uint8_t *compressed = (uint8_t *)malloc(ptx_raw_len * 2u + 64u);
    if (!compressed) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    size_t comp_len = lz4_compress_c99((const uint8_t *)VECTOR_ADD_PTX, ptx_raw_len, compressed);
    if (comp_len == 0 || comp_len >= ptx_raw_len) {
        fprintf(stderr, "unexpected LZ4 size %zu vs raw %zu\n", comp_len, ptx_raw_len);
        free(compressed);
        return 1;
    }

    size_t padded_len = (comp_len + 7u) & ~(size_t)7u;
    size_t entry_hdr_size = 64u;
    size_t total_entry_size = entry_hdr_size + padded_len;
    size_t fatbin_total = 16u + total_entry_size;
    uint8_t *fatbin = (uint8_t *)calloc(fatbin_total, 1u);
    if (!fatbin) {
        free(compressed);
        return 1;
    }

    /* 16-byte Fatbin header */
    write_u32_le(fatbin + 0, 0xBA55ED50u);
    write_u16_le(fatbin + 4, 1u);
    write_u16_le(fatbin + 6, 16u);
    write_u64_le(fatbin + 8, (uint64_t)total_entry_size);

    /* 64-byte FatBinEntry header (spec offsets) */
    uint8_t *e = fatbin + 16;
    write_u16_le(e + 0, 1u); /* FATBIN_KIND_PTX */
    write_u16_le(e + 2, 1u);
    write_u32_le(e + 4, (uint32_t)entry_hdr_size);
    write_u64_le(e + 8, (uint64_t)padded_len);
    write_u32_le(e + 16, (uint32_t)comp_len);
    write_u64_le(e + 40, 0x2000u); /* FATBIN_FLAG_LZ4 */
    write_u64_le(e + 56, (uint64_t)ptx_raw_len);
    memcpy(fatbin + 16 + entry_hdr_size, compressed, comp_len);
    free(compressed);

    __fatBinC_Wrapper_t wrapper;
    wrapper.magic = 0x466243B1u;
    wrapper.version = 1u;
    wrapper.data = fatbin;
    wrapper.filename_or_fatbins = NULL;

    /* 2. Simulate nvcc C++ static constructor fatbin/kernel registration */
    void **fatbin_handle = __cudaRegisterFatBinary(&wrapper);
    if (!fatbin_handle) {
        fprintf(stderr, "__cudaRegisterFatBinary returned NULL\n");
        free(fatbin);
        return 1;
    }
    char dev_fn_name[] = "vector_add";
    __cudaRegisterFunction(
        fatbin_handle,
        (const char *)(const void *)&host_vector_add_kernel,
        dev_fn_name,
        "vector_add",
        -1,
        NULL, NULL, NULL, NULL, NULL
    );
    __cudaRegisterFatBinaryEnd(fatbin_handle);

    /* 3. Execute vector_add through CUDA Runtime API + call configuration stack */
    const unsigned int N = 4096u;
    const size_t bytes = (size_t)N * sizeof(float);

    float *ha = (float *)malloc(bytes);
    float *hb = (float *)malloc(bytes);
    float *hc = (float *)malloc(bytes);
    if (!ha || !hb || !hc) {
        fprintf(stderr, "host buffer allocation failed\n");
        return 1;
    }
    for (unsigned int i = 0; i < N; i++) {
        ha[i] = (float)i * 0.25f;
        hb[i] = (float)i * 2.75f + 1.0f;
    }

    void *da = NULL, *db = NULL, *dc = NULL;
    CHECK_CUDA(cudaMalloc(&da, bytes));
    CHECK_CUDA(cudaMalloc(&db, bytes));
    CHECK_CUDA(cudaMalloc(&dc, bytes));
    CHECK_CUDA(cudaMemset(dc, 0, bytes));

    cudaStream_t stream = NULL;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaMemcpyAsync(da, ha, bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(db, hb, bytes, cudaMemcpyHostToDevice, stream));

    dim3 grid_cfg = { N / 256u, 1u, 1u };
    dim3 block_cfg = { 256u, 1u, 1u };
    if (__cudaPushCallConfiguration(grid_cfg, block_cfg, 0u, stream) != 0u) {
        fprintf(stderr, "__cudaPushCallConfiguration failed\n");
        return 1;
    }

    dim3 grid = { 0u, 0u, 0u };
    dim3 block = { 0u, 0u, 0u };
    size_t shared_mem = 0u;
    cudaStream_t popped_stream = NULL;
    CHECK_CUDA(__cudaPopCallConfiguration(&grid, &block, &shared_mem, &popped_stream));

    unsigned int n_arg = N;
    void *params[] = { &da, &db, &dc, &n_arg };
    CHECK_CUDA(cudaLaunchKernel(
        (const void *)&host_vector_add_kernel,
        grid,
        block,
        params,
        shared_mem,
        popped_stream
    ));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));

    for (unsigned int i = 0; i < N; i++) {
        float expected = ha[i] + hb[i];
        if (fabsf(hc[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at %u: got %f, expected %f\n", i, hc[i], expected);
            return 1;
        }
    }

    size_t free_mem = 0u, total_mem = 0u;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    if (total_mem == 0u || free_mem > total_mem) {
        fprintf(stderr, "Invalid cudaMemGetInfo result: free=%zu total=%zu\n", free_mem, total_mem);
        return 1;
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(da));
    CHECK_CUDA(cudaFree(db));
    CHECK_CUDA(cudaFree(dc));
    __cudaUnregisterFatBinary(fatbin_handle);

    free(ha);
    free(hb);
    free(hc);
    free(fatbin);

    printf("SUCCESS: Verified nvcc LZ4 fatbin (%zu -> %zu bytes) & cudaLaunchKernel for N=%u elements.\n",
           ptx_raw_len, comp_len, N);
    return 0;
}
