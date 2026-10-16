// Preserve the two BF16 rounding points of F.gelu(gate) * up.
// Include the original quantizers for shared scale/layout definitions only.
#include "quantize.cu"

__device__ float gelu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up, int64_t index, const uint16_t* table) {
    float gelu = __bfloat162float(__ushort_as_bfloat16(table[__bfloat16_as_ushort(gate[index])]));
    float rounded = __bfloat162float(__float2bfloat16(gelu));
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

extern "C" int cc_geglu(const void* gate, const void* up, void* output,
                         const void* scale, int64_t count, int bits, const void* table, void* stream) {
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
