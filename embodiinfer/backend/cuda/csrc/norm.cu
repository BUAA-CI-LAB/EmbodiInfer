// Preserve FP32 pointwise order and both BF16 residual rounding boundaries.
// The row reduction is an separate numerical path.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int THREADS = 256;

__device__ __forceinline__ float residual_value(
    const __nv_bfloat16* inputs, const __nv_bfloat16* update,
    const __nv_bfloat16* gate, int64_t index, int columns, int tokens) {
    float value = __bfloat162float(inputs[index]);
    if (update == nullptr) return value;
    float delta = __bfloat162float(update[index]);
    if (gate != nullptr) {
        int batch = index / (int64_t(tokens) * columns);
        float factor = __bfloat162float(gate[int64_t(batch) * columns + index % columns]);
        delta = __bfloat162float(__float2bfloat16_rn(__fmul_rn(delta, factor)));
    }
    return __bfloat162float(__float2bfloat16_rn(__fadd_rn(value, delta)));
}

__device__ __forceinline__ float affine_value(
    float value, float reciprocal_rms, const float* scale, const float* modulation,
    int64_t index, int columns, int tokens) {
    float normalized = __fmul_rn(value, reciprocal_rms);
    int column = index % columns;
    if (modulation != nullptr) {
        int batch = index / (int64_t(tokens) * columns);
        int64_t base = int64_t(batch) * 3 * columns;
        normalized = __fmul_rn(normalized, __fadd_rn(1.0f, modulation[base + column]));
        return __fadd_rn(normalized, modulation[base + columns + column]);
    }
    return __fmul_rn(normalized, __fadd_rn(1.0f, scale[column]));
}

__global__ void residual_square(
    const __nv_bfloat16* inputs, const __nv_bfloat16* update,
    const __nv_bfloat16* gate, __nv_bfloat16* residual, float* square,
    int64_t count, int columns, int tokens) {
    int64_t index = int64_t(blockIdx.x) * THREADS + threadIdx.x;
    if (index >= count) return;
    float value = residual_value(inputs, update, gate, index, columns, tokens);
    residual[index] = __float2bfloat16_rn(value);
    square[index] = __fmul_rn(value, value);
}

__global__ void norm_pointwise(
    const __nv_bfloat16* inputs, const float* mean, const float* scale,
    const float* modulation, __nv_bfloat16* output, __nv_bfloat16* output_gate, int64_t count,
    int columns, int tokens) {
    int64_t index = int64_t(blockIdx.x) * THREADS + threadIdx.x;
    if (index >= count) return;
    float inverse = rsqrtf(__fadd_rn(mean[index / columns], 1.0e-6f));
    output[index] = __float2bfloat16_rn(affine_value(
        __bfloat162float(inputs[index]), inverse, scale, modulation,
        index, columns, tokens));
    if (output_gate != nullptr && (index / columns) % tokens == 0) {
        int batch = index / (int64_t(tokens) * columns);
        int column = index % columns;
        output_gate[int64_t(batch) * columns + column] = __float2bfloat16_rn(
            modulation[int64_t(batch) * 3 * columns + 2 * columns + column]);
    }
}

__global__ void norm_reduce(
    const __nv_bfloat16* inputs, const __nv_bfloat16* update,
    const __nv_bfloat16* gate, const float* scale, const float* modulation,
    __nv_bfloat16* residual, __nv_bfloat16* output, __nv_bfloat16* output_gate,
    int columns, int tokens) {
    __shared__ float partials[THREADS];
    int row = blockIdx.x;
    int batch = row / tokens;
    bool write_gate = (output_gate != nullptr) && (row == batch * tokens);
    int64_t base = int64_t(row) * columns;
    int64_t gate_base = int64_t(batch) * columns;
    int64_t mod_base = int64_t(batch) * 3 * columns;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += THREADS) {
        float value = __bfloat162float(inputs[base + column]);
        if (update != nullptr) {
            float delta = __bfloat162float(update[base + column]);
            if (gate != nullptr) {
                float factor = __bfloat162float(gate[gate_base + column]);
                delta = __bfloat162float(__float2bfloat16_rn(__fmul_rn(delta, factor)));
            }
            value = __bfloat162float(__float2bfloat16_rn(__fadd_rn(value, delta)));
        }
        sum = __fadd_rn(sum, __fmul_rn(value, value));
        if (residual != nullptr) residual[base + column] = __float2bfloat16_rn(value);
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            partials[threadIdx.x] = __fadd_rn(partials[threadIdx.x], partials[threadIdx.x + stride]);
        __syncthreads();
    }
    float inverse = rsqrtf(__fadd_rn(__fmul_rn(partials[0], 1.0f / columns), 1.0e-6f));
    const __nv_bfloat16* src = (residual != nullptr) ? residual : inputs;
    for (int column = threadIdx.x; column < columns; column += THREADS) {
        float value = __bfloat162float(src[base + column]);
        float normalized = __fmul_rn(value, inverse);
        if (modulation != nullptr) {
            normalized = __fmul_rn(normalized, __fadd_rn(1.0f, modulation[mod_base + column]));
            normalized = __fadd_rn(normalized, modulation[mod_base + columns + column]);
        } else {
            normalized = __fmul_rn(normalized, __fadd_rn(1.0f, scale[column]));
        }
        output[base + column] = __float2bfloat16_rn(normalized);
        if (write_gate) {
            output_gate[gate_base + column] = __float2bfloat16_rn(
                modulation[mod_base + 2 * columns + column]);
        }
    }
}

extern "C" int cc_norm(
    const void* inputs, const void* mean, const void* scale, const void* modulation,
    void* output, void* output_gate, int rows, int columns, int tokens, int reduce, void* stream) {
    if (reduce)
        norm_reduce<<<rows, THREADS, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)inputs, nullptr, nullptr, (const float*)scale,
            (const float*)modulation, nullptr, (__nv_bfloat16*)output,
            (__nv_bfloat16*)output_gate, columns, tokens);
    else
        norm_pointwise<<<(int64_t(rows) * columns + THREADS - 1) / THREADS, THREADS, 0,
                         (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)inputs, (const float*)mean, (const float*)scale,
            (const float*)modulation, (__nv_bfloat16*)output,
            (__nv_bfloat16*)output_gate, int64_t(rows) * columns,
            columns, tokens);
    return int(cudaGetLastError());
}

extern "C" int cc_residual_square(
    const void* inputs, const void* update, const void* gate, void* residual,
    void* square, int rows, int columns, int tokens, void* stream) {
    residual_square<<<(int64_t(rows) * columns + THREADS - 1) / THREADS, THREADS, 0,
                      (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)inputs, (const __nv_bfloat16*)update,
        (const __nv_bfloat16*)gate, (__nv_bfloat16*)residual, (float*)square,
        int64_t(rows) * columns, columns, tokens);
    return int(cudaGetLastError());
}

extern "C" int cc_residual_norm(
    const void* inputs, const void* update, const void* gate, const void* scale,
    const void* modulation, void* residual, void* output, void* output_gate,
    int rows, int columns, int tokens, void* stream) {
    norm_reduce<<<rows, THREADS, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)inputs, (const __nv_bfloat16*)update,
        (const __nv_bfloat16*)gate, (const float*)scale, (const float*)modulation,
        (__nv_bfloat16*)residual, (__nv_bfloat16*)output,
        (__nv_bfloat16*)output_gate, columns, tokens);
    return int(cudaGetLastError());
}

__global__ void square_bf16(const __nv_bfloat16* input, float* output, int64_t count) {
    int64_t index = int64_t(blockIdx.x) * THREADS + threadIdx.x;
    if (index >= count) return;
    float x = __bfloat162float(input[index]);
    output[index] = __fmul_rn(x, x);
}

__global__ void residual_only(
    const __nv_bfloat16* inputs, const __nv_bfloat16* update, const __nv_bfloat16* gate,
    __nv_bfloat16* output, int64_t count, int columns, int tokens) {
    int64_t index = int64_t(blockIdx.x) * THREADS + threadIdx.x;
    if (index < count)
        output[index] = __float2bfloat16_rn(residual_value(inputs, update, gate, index, columns, tokens));
}

extern "C" int cc_square(const void* inputs, void* output, int64_t count, void* stream) {
    square_bf16<<<(count + THREADS - 1) / THREADS, THREADS, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)inputs, (float*)output, count);
    return int(cudaGetLastError());
}

extern "C" int cc_residual(
    const void* inputs, const void* update, const void* gate, void* output,
    int64_t count, int columns, int tokens, void* stream) {
    residual_only<<<(count + THREADS - 1) / THREADS, THREADS, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)inputs, (const __nv_bfloat16*)update, (const __nv_bfloat16*)gate,
        (__nv_bfloat16*)output, count, columns, tokens);
    return int(cudaGetLastError());
}
