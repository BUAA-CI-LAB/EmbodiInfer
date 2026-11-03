// Preserve the two BF16 rounding points of F.gelu(gate) * up.
// Include the original quantizers for shared scale/layout definitions only.
#include "quantize.cu"

__device__ float gelu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up, int64_t index, const uint16_t* table) {
    float gelu = __bfloat162float(__ushort_as_bfloat16(table[__bfloat16_as_ushort(gate[index])]));
    float rounded = __bfloat162float(__float2bfloat16(gelu));
    if (up == nullptr) return rounded;
    return __bfloat162float(__float2bfloat16(rounded * __bfloat162float(up[index])));
}

template<bool FP8>
__global__ void fused_geglu(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                             void* output, const float* scale, int64_t count, const uint16_t* table) {
    int64_t index = (int64_t(blockIdx.x) * THREADS + threadIdx.x) * 2;
    if (index >= count) return;
    float first = gelu_mul(gate, up, index, table);
    float second = index + 1 < count ? gelu_mul(gate, up, index + 1, table) : 0.0f;
    if constexpr (FP8) {
        uint16_t packed = __nv_cvt_float2_to_fp8x2(
            make_float2(first / *scale, second / *scale), __NV_SATFINITE, __NV_E4M3);
        if (index + 1 < count) reinterpret_cast<uint16_t*>(output)[index / 2] = packed;
        else reinterpret_cast<uint8_t*>(output)[index] = packed & 255;
    } else {
        if (index + 1 < count)
            reinterpret_cast<__nv_bfloat162*>(output)[index / 2] = __floats2bfloat162_rn(first, second);
        else reinterpret_cast<__nv_bfloat16*>(output)[index] = __float2bfloat16(first);
    }
}

__global__ void fused_geglu_fp4(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                 uint8_t* output, uint8_t* scales, uint8_t* blocked,
                                 const float* global, int rows, int columns, const uint16_t* table) {
    int row = blockIdx.y;
    int column_block = blockIdx.x * (THREADS / 8) + threadIdx.x / 8;
    int blocks_per_row = columns / 16;
    if (column_block >= blocks_per_row) return;
    int block = row * blocks_per_row + column_block;
    int lane = threadIdx.x & 7;
    int64_t index = int64_t(row) * columns + column_block * 16 + lane * 2;
    float first = row < rows ? gelu_mul(gate, up, index, table) : 0.0f;
    float second = row < rows ? gelu_mul(gate, up, index + 1, table) : 0.0f;
    float maximum = fmaxf(fabsf(first), fabsf(second));
    for (int delta = 4; delta; delta /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta, 8));
    uint8_t scale_code = __nv_cvt_float_to_fp8(
        fminf(448.0f, maximum / (6.0f * *global)), __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 converted;
    converted.__x = scale_code;
    float denominator = *global * float(converted);
    uint8_t packed = cc_fp4_cast(
        make_float2(denominator > 0.0f ? first / denominator : 0.0f,
                    denominator > 0.0f ? second / denominator : 0.0f));
    if (row < rows) output[index / 2] = packed;
    if (lane == 0) {
        int column_tiles = (blocks_per_row + 3) / 4;
        int offset = ((row / 128) * column_tiles + column_block / 4) * 512
                   + (row % 32) * 16 + ((row % 128) / 32) * 4 + column_block % 4;
        blocked[offset] = scale_code;
        if (row < rows) scales[block] = scale_code;
    }
}

template<bool FP8>
__global__ void fused_geglu_vec8(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                  void* output, const float* scale, int64_t vec_count, const uint16_t* table) {
    int64_t idx8 = int64_t(blockIdx.x) * THREADS + threadIdx.x;
    if (idx8 >= vec_count) return;
    uint4 g4 = reinterpret_cast<const uint4*>(gate)[idx8];
    const uint16_t* g_u16 = reinterpret_cast<const uint16_t*>(&g4);
    float vals[8];
    if (up == nullptr) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            vals[i] = __bfloat162float(__ushort_as_bfloat16(__ldg(table + g_u16[i])));
        }
    } else {
        uint4 u4 = reinterpret_cast<const uint4*>(up)[idx8];
        const __nv_bfloat16* u_bf16 = reinterpret_cast<const __nv_bfloat16*>(&u4);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float g_val = __bfloat162float(__ushort_as_bfloat16(__ldg(table + g_u16[i])));
            vals[i] = __bfloat162float(__float2bfloat16(g_val * __bfloat162float(u_bf16[i])));
        }
    }
    if constexpr (FP8) {
        float inv_s = 1.0f / (*scale);
        uint16_t p[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            p[i] = __nv_cvt_float2_to_fp8x2(
                make_float2(vals[2 * i] * inv_s, vals[2 * i + 1] * inv_s), __NV_SATFINITE, __NV_E4M3);
        }
        reinterpret_cast<uint2*>(output)[idx8] = *reinterpret_cast<const uint2*>(p);
    } else {
        __nv_bfloat162 p[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            p[i] = __floats2bfloat162_rn(vals[2 * i], vals[2 * i + 1]);
        }
        reinterpret_cast<uint4*>(output)[idx8] = *reinterpret_cast<const uint4*>(p);
    }
}

extern "C" int cc_geglu(const void* gate, const void* up, void* output,
                         const void* scale, int64_t count, int bits, const void* table, void* stream) {
    if ((count & 7) == 0) {
        int64_t vec_count = count >> 3;
        int blocks = int((vec_count + THREADS - 1) / THREADS);
        if (bits == 8)
            fused_geglu_vec8<true><<<blocks, THREADS, 0, (cudaStream_t)stream>>>(
                (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
                output, (const float*)scale, vec_count, (const uint16_t*)table);
        else
            fused_geglu_vec8<false><<<blocks, THREADS, 0, (cudaStream_t)stream>>>(
                (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
                output, (const float*)scale, vec_count, (const uint16_t*)table);
        return int(cudaGetLastError());
    }
    int blocks = (count + THREADS * 2 - 1) / (THREADS * 2);
    if (bits == 8)
        fused_geglu<true><<<blocks, THREADS, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
            output, (const float*)scale, count, (const uint16_t*)table);
    else
        fused_geglu<false><<<blocks, THREADS, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
            output, (const float*)scale, count, (const uint16_t*)table);
    return int(cudaGetLastError());
}

extern "C" int cc_geglu_fp4(const void* gate, const void* up, void* output,
                             void* scales, void* blocked, const void* global,
                             int rows, int columns, const void* table, void* stream) {
    fused_geglu_fp4<<<dim3((columns / 16 + THREADS / 8 - 1) / (THREADS / 8), rows), THREADS, 0,
                       (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up, (uint8_t*)output,
        (uint8_t*)scales, (uint8_t*)blocked, (const float*)global, rows, columns, (const uint16_t*)table);
    return int(cudaGetLastError());
}

// BF16-only entry point over row-strided gate/up views, including packed GEMMs.
// Reuse the exact lookup and product-rounding implementation; no encoding.
__global__ void fused_geglu_bf16_strided(const __nv_bfloat16* gate,
        const __nv_bfloat16* up, __nv_bfloat16* output, int64_t count,
        int columns, int64_t row_stride, const uint16_t* table) {
    int64_t index = (int64_t(blockIdx.x) * THREADS + threadIdx.x) * 2;
    if (index >= count) return;
    int64_t offset = (index / columns) * row_stride + index % columns;
    float first = gelu_mul(gate, up, offset, table);
    if (index + 1 < count) {
        int64_t next = ((index + 1) / columns) * row_stride + (index + 1) % columns;
        float second = gelu_mul(gate, up, next, table);
        reinterpret_cast<__nv_bfloat162*>(output)[index / 2] = __floats2bfloat162_rn(first, second);
    } else output[index] = __float2bfloat16(first);
}

extern "C" int cc_geglu_bf16_strided(const void* gate, const void* up, void* output,
        int rows, int columns, int64_t row_stride, const void* table, void* stream) {
    if (rows <= 0 || columns <= 0 || row_stride < columns) return int(cudaErrorInvalidValue);
    int64_t count = int64_t(rows) * columns;
    int blocks = (count + THREADS * 2 - 1) / (THREADS * 2);
    fused_geglu_bf16_strided<<<blocks, THREADS, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
        (__nv_bfloat16*)output, count, columns, row_stride, (const uint16_t*)table);
    return int(cudaGetLastError());
}

template<bool FP8>
__global__ void fused_geglu_packed(const __nv_bfloat16* packed, void* output,
                                   const float* scale, int columns, const uint16_t* table) {
    int col = (int(blockIdx.x) * THREADS + threadIdx.x) * 2;
    if (col >= columns) return;
    int64_t row = blockIdx.y;
    int64_t g_idx = row * (2 * columns) + col;
    int64_t out_idx = row * columns + col;
    float first = gelu_mul(packed, packed + columns, g_idx, table);
    float second = gelu_mul(packed, packed + columns, g_idx + 1, table);
    if constexpr (FP8) {
        float inv_scale = 1.0f / (*scale);
        uint16_t out_packed = __nv_cvt_float2_to_fp8x2(
            make_float2(first * inv_scale, second * inv_scale), __NV_SATFINITE, __NV_E4M3);
        reinterpret_cast<uint16_t*>(output)[out_idx / 2] = out_packed;
    } else {
        reinterpret_cast<__nv_bfloat162*>(output)[out_idx / 2] = __floats2bfloat162_rn(first, second);
    }
}

extern "C" int cc_geglu_packed(const void* packed, void* output, const void* scale,
                               int64_t count, int columns, int bits, const void* table, void* stream) {
    int rows = int(count / columns);
    dim3 grid((columns / 2 + THREADS - 1) / THREADS, rows);
    if (bits == 8)
        fused_geglu_packed<true><<<grid, THREADS, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)packed, output, (const float*)scale, columns, (const uint16_t*)table);
    else
        fused_geglu_packed<false><<<grid, THREADS, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)packed, output, (const float*)scale, columns, (const uint16_t*)table);
    return int(cudaGetLastError());
}
