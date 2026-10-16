// Preserve the reference FP32 operation order and explicit BF16 rounding.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// Match the FP32 persistent-warp softmax tree in PyTorch 2.9.1, while
// applying the reference finite mask on load and rounding to BF16 on store.
// Algorithm reference: aten/src/ATen/native/cuda/PersistentSoftmax.cuh.
template<int ITERATIONS>
__global__ void masked_softmax(
    const float* logits, const bool* mask, __nv_bfloat16* probabilities,
    int rows, int tokens, int heads, int columns) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    int lane = threadIdx.x;
    // Compute addresses before the live FP32 row, avoiding repeated
    // integer divisions/register expansion inside the unrolled store loop.
    int mask_row = row % tokens;
    int head = row / tokens;
    probabilities += (int64_t(mask_row) * heads + head) * columns;
    float values[ITERATIONS];
    #pragma unroll
    for (int i = 0; i < ITERATIONS; ++i) {
        int col = lane + i * 32;
        values[i] = col < columns
            ? ((mask == nullptr || mask[int64_t(mask_row) * columns + col])
                ? logits[int64_t(row) * columns + col] : -2.3819763e38f)
            : -INFINITY;
    }
    float maximum = values[0];
    #pragma unroll
    for (int i = 0; i < ITERATIONS; ++i)
        maximum = maximum > values[i] ? maximum : values[i];
    #pragma unroll
    for (int delta = 16; delta; delta /= 2) {
        float other = __shfl_xor_sync(0xffffffff, maximum, delta, 32);
        maximum = maximum < other ? other : maximum;
    }
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < ITERATIONS; ++i) {
        values[i] = expf(__fsub_rn(values[i], maximum));
        sum = __fadd_rn(sum, values[i]);
    }
    #pragma unroll
    for (int delta = 16; delta; delta /= 2)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, delta, 32));
    #pragma unroll
    for (int i = 0; i < ITERATIONS; ++i) {
        int col = lane + i * 32;
        if (col < columns)
            probabilities[col] = __float2bfloat16_rn(values[i] / sum);
    }
}

__device__ __forceinline__ float bf16_round(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__global__ void prepare_qk(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const float* sine, const float* cosine,
    float* q32, __nv_bfloat16* rotated_key, float* k32,
    __nv_bfloat16* cached_value, int batch, int tokens, int qheads,
    int kheads, int width, int prefix, float scale) {
    int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int half = width / 2;
    int64_t queries = int64_t(batch) * tokens * qheads * half;
    bool is_query = index < queries;
    if (!is_query) index -= queries;
    int heads = is_query ? qheads : kheads;
    if (index >= int64_t(batch) * tokens * heads * half) return;
    int column = index % half;
    int64_t row = index / half;
    int64_t source = row * width + column;
    int64_t factor = (row / heads) * half + column;
    const __nv_bfloat16* input = is_query ? query : key;
    float a = __bfloat162float(input[source]);
    float b = __bfloat162float(input[source + half]);
    float s = sine[factor], c = cosine[factor];
    float first = bf16_round(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)));
    float second = bf16_round(__fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s)));
    if (is_query) {
        // Store [B,H,T,D]; expose [B,T,H,D] as a view in Python.
        // B1 is checked at the Python boundary; row fits signed32.
        int query_row = int(row);
        int token = query_row / heads;
        int head = query_row % heads;
        int64_t destination = (int64_t(head) * tokens + token) * width + column;
        q32[destination] = bf16_round(__fmul_rn(first, scale));
        q32[destination + half] = bf16_round(__fmul_rn(second, scale));
    } else {
        int sample = row / (tokens * heads);
        int64_t destination = source + int64_t(sample + 1) * prefix * heads * width;
        rotated_key[destination] = __float2bfloat16_rn(first);
        rotated_key[destination + half] = __float2bfloat16_rn(second);
        if (k32 != nullptr) {
            k32[destination] = first;
            k32[destination + half] = second;
        }
        if (cached_value != nullptr) {
            cached_value[destination] = value[source];
            cached_value[destination + half] = value[source + half];
        }
    }
}

__global__ void load_prefix(
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* full_key, float* k32, __nv_bfloat16* full_value,
    int64_t count, int prefix, int suffix, int columns) {
    int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int sample = index / (int64_t(prefix) * columns);
    int64_t destination = index + int64_t(sample) * suffix * columns;
    full_key[destination] = key[index];
    k32[destination] = __bfloat162float(key[index]);
    full_value[destination] = value[index];
}

extern "C" int ei_prepare_qk(
    const void* query, const void* key, const void* value,
    const void* sine, const void* cosine, void* q32, void* rotated_key,
    void* k32, void* cached_value, int batch, int tokens, int qheads,
    int kheads, int width, int prefix, float scale, void* stream) {
    int64_t pairs = int64_t(batch) * tokens * (qheads + kheads) * (width / 2);
    prepare_qk<<<(pairs + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)query, (const __nv_bfloat16*)key,
        (const __nv_bfloat16*)value, (const float*)sine, (const float*)cosine,
        (float*)q32, (__nv_bfloat16*)rotated_key, (float*)k32,
        (__nv_bfloat16*)cached_value, batch, tokens, qheads, kheads, width, prefix, scale);
    return int(cudaGetLastError());
}

extern "C" int ei_load_prefix(
    const void* key, const void* value, void* full_key, void* k32,
    void* full_value, int batch, int prefix, int suffix, int columns, void* stream) {
    int64_t count = int64_t(batch) * prefix * columns;
    load_prefix<<<(count + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)key, (const __nv_bfloat16*)value,
        (__nv_bfloat16*)full_key, (float*)k32, (__nv_bfloat16*)full_value,
        count, prefix, suffix, columns);
    return int(cudaGetLastError());
}

extern "C" int ei_masked_softmax(
    const void* logits, const void* mask, void* probabilities,
    int batch, int tokens, int heads, int columns, void* stream) {
    if (columns <= 128 || columns > 1024) return int(cudaErrorInvalidValue);
    int rows = batch * tokens * heads;
    dim3 block(32, 4);
    if (columns <= 256)
        masked_softmax<8><<<(rows + 3) / 4, block, 0, (cudaStream_t)stream>>>(
            (const float*)logits, (const bool*)mask, (__nv_bfloat16*)probabilities,
            rows, tokens, heads, columns);
    else if (columns <= 512)
        masked_softmax<16><<<(rows + 3) / 4, block, 0, (cudaStream_t)stream>>>(
            (const float*)logits, (const bool*)mask, (__nv_bfloat16*)probabilities,
            rows, tokens, heads, columns);
    else
        masked_softmax<32><<<(rows + 3) / 4, block, 0, (cudaStream_t)stream>>>(
            (const float*)logits, (const bool*)mask, (__nv_bfloat16*)probabilities,
            rows, tokens, heads, columns);
    return int(cudaGetLastError());
}
