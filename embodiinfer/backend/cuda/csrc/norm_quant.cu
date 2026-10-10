// Original quantizer definitions preserve FP8/NVFP4 conversions and padded layout.
#include "quantize.cu"

__device__ __forceinline__ float normalized_bf16(
    const __nv_bfloat16* input, const float* mean, const float* scale,
    const float* modulation, int64_t index, int columns, int tokens) {
    float inverse = rsqrtf(__fadd_rn(mean[index / columns], 1.0e-6f));
    float value = __fmul_rn(__bfloat162float(input[index]), inverse);
    int column = index % columns;
    if (modulation != nullptr) {
        int batch = index / (int64_t(tokens) * columns);
        int64_t base = int64_t(batch) * 3 * columns;
        value = __fmul_rn(value, __fadd_rn(1.0f, modulation[base + column]));
        value = __fadd_rn(value, modulation[base + columns + column]);
    } else {
        value = __fmul_rn(value, __fadd_rn(1.0f, scale[column]));
    }
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ void write_norm_gate(
    const float* modulation, __nv_bfloat16* gate, int64_t index,
    int columns, int tokens) {
    if (gate != nullptr && (index / columns) % tokens == 0) {
        int batch = index / (int64_t(tokens) * columns);
        int column = index % columns;
        gate[int64_t(batch) * columns + column] = __float2bfloat16_rn(
            modulation[int64_t(batch) * 3 * columns + 2 * columns + column]);
    }
}

__global__ void norm_fp8(
    const __nv_bfloat16* input, const float* mean, const float* scale,
    const float* modulation, uint8_t* output, __nv_bfloat16* gate,
    const float* global, int64_t count, int columns, int tokens) {
    int64_t index = (int64_t(blockIdx.x) * THREADS + threadIdx.x) * 2;
    if (index >= count) return;
    float first = normalized_bf16(input, mean, scale, modulation, index, columns, tokens);
    float second = index + 1 < count
        ? normalized_bf16(input, mean, scale, modulation, index + 1, columns, tokens) : 0.0f;
    uint16_t packed = __nv_cvt_float2_to_fp8x2(
        make_float2(first / *global, second / *global), __NV_SATFINITE, __NV_E4M3);
    if (index + 1 < count) reinterpret_cast<uint16_t*>(output)[index / 2] = packed;
    else output[index] = packed & 255;
    write_norm_gate(modulation, gate, index, columns, tokens);
    if (index + 1 < count) write_norm_gate(modulation, gate, index + 1, columns, tokens);
}

__global__ void norm_fp4(
    const __nv_bfloat16* input, const float* mean, const float* scale,
    const float* modulation, uint8_t* output, uint8_t* scales,
    uint8_t* blocked, __nv_bfloat16* gate, const float* global,
    int rows, int columns, int tokens) {
    int row = blockIdx.y;
    int column_block = blockIdx.x * (THREADS / 8) + threadIdx.x / 8;
    int blocks_per_row = columns / 16;
    if (column_block >= blocks_per_row) return;
    int block = row * blocks_per_row + column_block;
    int lane = threadIdx.x & 7;
    int64_t index = int64_t(row) * columns + column_block * 16 + lane * 2;
    float first = row < rows
        ? normalized_bf16(input, mean, scale, modulation, index, columns, tokens) : 0.0f;
    float second = row < rows
        ? normalized_bf16(input, mean, scale, modulation, index + 1, columns, tokens) : 0.0f;
    float maximum = fmaxf(fabsf(first), fabsf(second));
    for (int delta = 4; delta; delta /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta, 8));
    float raw_scale = fminf(448.0f, maximum / (6.0f * *global));
    uint8_t scale_code = __nv_cvt_float_to_fp8(raw_scale, __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 converted;
    converted.__x = scale_code;
    float denominator = *global * float(converted);
    uint8_t packed = cc_fp4_cast(
        make_float2(denominator > 0.0f ? first / denominator : 0.0f,
                    denominator > 0.0f ? second / denominator : 0.0f));
    if (row < rows) {
        output[index / 2] = packed;
        write_norm_gate(modulation, gate, index, columns, tokens);
        write_norm_gate(modulation, gate, index + 1, columns, tokens);
    }
    if (lane == 0) {
        int column_tiles = (blocks_per_row + 3) / 4;
        int offset = ((row / 128) * column_tiles + column_block / 4) * 512
                   + (row % 32) * 16 + ((row % 128) / 32) * 4 + column_block % 4;
        blocked[offset] = scale_code;
        if (row < rows) scales[block] = scale_code;
    }
}

extern "C" int cc_norm_quant(
    const void* input, const void* mean, const void* scale, const void* modulation,
    void* output, void* scales, void* blocked, void* gate, const void* global,
    int rows, int columns, int tokens, int bits, void* stream) {
    if (bits == 8) {
        int64_t count = int64_t(rows) * columns;
        norm_fp8<<<(count + THREADS * 2 - 1) / (THREADS * 2), THREADS, 0,
                   (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)input, (const float*)mean, (const float*)scale,
            (const float*)modulation, (uint8_t*)output, (__nv_bfloat16*)gate,
            (const float*)global, count, columns, tokens);
    } else {
            norm_fp4<<<dim3((columns / 16 + THREADS / 8 - 1) / (THREADS / 8), rows), THREADS, 0,
                   (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)input, (const float*)mean, (const float*)scale,
            (const float*)modulation, (uint8_t*)output, (uint8_t*)scales,
            (uint8_t*)blocked, (__nv_bfloat16*)gate, (const float*)global,
            rows, columns, tokens);
    }
    return int(cudaGetLastError());
}
