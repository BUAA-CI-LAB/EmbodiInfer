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


__global__ void norm_fp8_reduce(
    const __nv_bfloat16* inputs, const __nv_bfloat16* update,
    const __nv_bfloat16* residual_gate, const float* scale, const float* modulation,
    __nv_bfloat16* residual, uint8_t* output, __nv_bfloat16* gate,
    const float* global, int columns, int tokens) {
    __shared__ float partials[THREADS];
    int row = blockIdx.x;
    int batch = row / tokens;
    bool write_gate = (gate != nullptr) && (row == batch * tokens);
    int64_t base = int64_t(row) * columns;
    int64_t gate_base = int64_t(batch) * columns;
    int64_t mod_base = int64_t(batch) * 3 * columns;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < columns; col += THREADS) {
        float val = __bfloat162float(inputs[base + col]);
        if (update != nullptr) {
            float delta = __bfloat162float(update[base + col]);
            if (residual_gate != nullptr) {
                float factor = __bfloat162float(residual_gate[gate_base + col]);
                delta = __bfloat162float(__float2bfloat16_rn(__fmul_rn(delta, factor)));
            }
            val = __bfloat162float(__float2bfloat16_rn(__fadd_rn(val, delta)));
        }
        sum = __fadd_rn(sum, __fmul_rn(val, val));
        if (residual != nullptr) residual[base + col] = __float2bfloat16_rn(val);
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            partials[threadIdx.x] = __fadd_rn(partials[threadIdx.x], partials[threadIdx.x + stride]);
        __syncthreads();
    }
    float inverse = rsqrtf(__fadd_rn(__fmul_rn(partials[0], 1.0f / columns), 1.0e-6f));
    float inv_global = 1.0f / (*global);
    const __nv_bfloat16* src = (residual != nullptr) ? residual : inputs;
    for (int col2 = threadIdx.x * 2; col2 < columns; col2 += THREADS * 2) {
        int64_t idx = base + col2;
        float v0 = __fmul_rn(__bfloat162float(src[idx]), inverse);
        float v1 = __fmul_rn(__bfloat162float(src[idx + 1]), inverse);
        if (modulation != nullptr) {
            v0 = __fadd_rn(__fmul_rn(v0, __fadd_rn(1.0f, modulation[mod_base + col2])), modulation[mod_base + columns + col2]);
            v1 = __fadd_rn(__fmul_rn(v1, __fadd_rn(1.0f, modulation[mod_base + col2 + 1])), modulation[mod_base + columns + col2 + 1]);
        } else {
            v0 = __fmul_rn(v0, __fadd_rn(1.0f, scale[col2]));
            v1 = __fmul_rn(v1, __fadd_rn(1.0f, scale[col2 + 1]));
        }
        v0 = __bfloat162float(__float2bfloat16_rn(v0));
        v1 = __bfloat162float(__float2bfloat16_rn(v1));
        uint16_t packed = __nv_cvt_float2_to_fp8x2(
            make_float2(v0 * inv_global, v1 * inv_global), __NV_SATFINITE, __NV_E4M3);
        reinterpret_cast<uint16_t*>(output)[idx / 2] = packed;
        if (write_gate) {
            gate[gate_base + col2] = __float2bfloat16_rn(modulation[mod_base + 2 * columns + col2]);
            gate[gate_base + col2 + 1] = __float2bfloat16_rn(modulation[mod_base + 2 * columns + col2 + 1]);
        }
    }
}

extern "C" int cc_norm_quant_reduce(
    const void* inputs, const void* update, const void* residual_gate,
    const void* scale, const void* modulation, void* residual,
    void* output, void* gate, const void* global,
    int rows, int columns, int tokens, void* stream) {
    norm_fp8_reduce<<<rows, THREADS, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)inputs, (const __nv_bfloat16*)update,
        (const __nv_bfloat16*)residual_gate, (const float*)scale,
        (const float*)modulation, (__nv_bfloat16*)residual,
        (uint8_t*)output, (__nv_bfloat16*)gate, (const float*)global,
        columns, tokens);
    return int(cudaGetLastError());
}

__global__ void layer_norm_fp8_reduce(
    const __nv_bfloat16* inputs, const __nv_bfloat16* update,
    const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    __nv_bfloat16* residual, uint8_t* output,
    float inv_global, int columns) {
    __shared__ float s_sum[THREADS];
    __shared__ float s_sq[THREADS];
    int64_t base = int64_t(blockIdx.x) * columns;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < columns; col += THREADS) {
        float val = __bfloat162float(inputs[base + col]);
        if (update != nullptr) {
            val = __bfloat162float(__float2bfloat16_rn(__fadd_rn(val, __bfloat162float(update[base + col]))));
            residual[base + col] = __float2bfloat16_rn(val);
        }
        sum = __fadd_rn(sum, val);
    }
    s_sum[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            s_sum[threadIdx.x] = __fadd_rn(s_sum[threadIdx.x], s_sum[threadIdx.x + stride]);
        __syncthreads();
    }
    float mean = s_sum[0] / float(columns);
    float sq_sum = 0.0f;
    const __nv_bfloat16* src = (update != nullptr) ? residual : inputs;
    for (int col = threadIdx.x; col < columns; col += THREADS) {
        float diff = __bfloat162float(src[base + col]) - mean;
        sq_sum = __fadd_rn(sq_sum, diff * diff);
    }
    s_sq[threadIdx.x] = sq_sum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            s_sq[threadIdx.x] = __fadd_rn(s_sq[threadIdx.x], s_sq[threadIdx.x + stride]);
        __syncthreads();
    }
    float inv_std = rsqrtf(s_sq[0] / float(columns) + 1.0e-6f);
    if (inv_global > 0.0f) {
        for (int col2 = threadIdx.x * 2; col2 < columns; col2 += THREADS * 2) {
            int64_t idx = base + col2;
            float v0 = (__bfloat162float(src[idx]) - mean) * inv_std * __bfloat162float(weight[col2]) + __bfloat162float(bias[col2]);
            float v1 = (__bfloat162float(src[idx + 1]) - mean) * inv_std * __bfloat162float(weight[col2 + 1]) + __bfloat162float(bias[col2 + 1]);
            v0 = __bfloat162float(__float2bfloat16_rn(v0));
            v1 = __bfloat162float(__float2bfloat16_rn(v1));
            uint16_t packed = __nv_cvt_float2_to_fp8x2(
                make_float2(v0 * inv_global, v1 * inv_global), __NV_SATFINITE, __NV_E4M3);
            reinterpret_cast<uint16_t*>(output)[idx / 2] = packed;
        }
    } else {
        __nv_bfloat16* out_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
        for (int col2 = threadIdx.x * 2; col2 < columns; col2 += THREADS * 2) {
            int64_t idx = base + col2;
            float v0 = (__bfloat162float(src[idx]) - mean) * inv_std * __bfloat162float(weight[col2]) + __bfloat162float(bias[col2]);
            float v1 = (__bfloat162float(src[idx + 1]) - mean) * inv_std * __bfloat162float(weight[col2 + 1]) + __bfloat162float(bias[col2 + 1]);
            reinterpret_cast<__nv_bfloat162*>(out_bf16)[idx / 2] = __floats2bfloat162_rn(v0, v1);
        }
    }
}

extern "C" int cc_layer_norm_fp8(
    const void* inputs, const void* update, const void* weight, const void* bias,
    void* residual, void* output, float inv_global, int rows, int columns, void* stream) {
    layer_norm_fp8_reduce<<<rows, THREADS, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)inputs, (const __nv_bfloat16*)update,
        (const __nv_bfloat16*)weight, (const __nv_bfloat16*)bias,
        (__nv_bfloat16*)residual, (uint8_t*)output, inv_global, columns);
    return int(cudaGetLastError());
}

__global__ void bhsd_to_bshd_strided(
    const __nv_bfloat16* input, void* output, float inv_scale,
    int64_t stride_b, int64_t stride_h, int64_t stride_s,
    int H, int S, int D) {
    int bs = blockIdx.y;
    int h = blockIdx.x;
    int b = bs / S;
    int s = bs - b * S;
    int col2 = threadIdx.x * 2;
    if (col2 >= D) return;
    int64_t src_idx = int64_t(b) * stride_b + int64_t(h) * stride_h + int64_t(s) * stride_s + col2;
    int64_t dst_idx = (int64_t(bs) * H + h) * D + col2;
    if (inv_scale > 0.0f) {
        float v0 = __bfloat162float(input[src_idx]) * inv_scale;
        float v1 = __bfloat162float(input[src_idx + 1]) * inv_scale;
        reinterpret_cast<uint16_t*>(output)[dst_idx / 2] = __nv_cvt_float2_to_fp8x2(
            make_float2(v0, v1), __NV_SATFINITE, __NV_E4M3);
    } else {
        reinterpret_cast<uint32_t*>(output)[dst_idx / 2] =
            reinterpret_cast<const uint32_t*>(input)[src_idx / 2];
    }
}

extern "C" int cc_bhsd_fp8(
    const void* input, void* output, float inv_scale,
    int64_t stride_b, int64_t stride_h, int64_t stride_s,
    int B, int H, int S, int D, void* stream) {
    int threads = (D / 2 <= 64) ? 64 : 128;
    bhsd_to_bshd_strided<<<dim3(H, B * S), threads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, output, inv_scale,
        stride_b, stride_h, stride_s, H, S, D);
    return int(cudaGetLastError());
}
