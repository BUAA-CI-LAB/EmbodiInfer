// Compare the iteration/XOR reduction order to the tested Torch revision:
// aten/src/ATen/native/cuda/PersistentSoftmax.cuh (cf30153c).
// Numerical equality is measured, rather than assumed from the algorithm.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

struct SoftmaxLayout {
    int64_t strides[13];
    int heads, groups, queries, keys, rows, query_rows;
};

template <int ITEMS, bool BF16>
__global__ void mask_softmax_warp(
    const float* scores, const bool* mask, void* output, SoftmaxLayout layout) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= layout.rows) return;
    int lane = threadIdx.x;
    int query = layout.query_rows ? row / layout.groups % layout.queries : row % layout.queries;
    int group = layout.query_rows ? row % layout.groups : row / layout.queries % layout.groups;
    int head = row / (layout.queries * layout.groups) % layout.heads;
    int batch = row / (layout.queries * layout.groups * layout.heads);
    const int64_t* s = layout.strides;
    int64_t score_offset = batch * s[0] + head * s[1] + group * s[2] + query * s[3];
    int64_t mask_offset = batch * s[5] + query * s[6];
    int64_t output_offset = batch * s[8] + head * s[9] + group * s[10] + query * s[11];
    float values[ITEMS];
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        int column = lane + item * 32;
        values[item] = column < layout.keys
            ? (mask[mask_offset + column * s[7]]
                ? scores[score_offset + column * s[4]] : -2.3819763e38f)
            : -CUDART_INF_F;
    }
    float maximum = values[0];
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item)
        maximum = maximum > values[item] ? maximum : values[item];
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xffffffff, maximum, offset);
        maximum = maximum < other ? other : maximum;
    }
    float sum = 0.0f;
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        values[item] = expf(__fsub_rn(values[item], maximum));
        sum = __fadd_rn(sum, values[item]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        int column = lane + item * 32;
        if (column < layout.keys) {
            float probability = __fdiv_rn(values[item], sum);
            int64_t index = output_offset + column * s[12];
            if constexpr (BF16)
                static_cast<__nv_bfloat16*>(output)[index] = __float2bfloat16_rn(probability);
            else
                static_cast<float*>(output)[index] = probability;
        }
    }
}

template <int ITEMS>
void launch_softmax(const float* scores, const bool* mask, void* output,
                    SoftmaxLayout layout, int warps, bool bf16, cudaStream_t stream) {
    dim3 threads(32, warps);
    int blocks = (layout.rows + warps - 1) / warps;
    if (bf16)
        mask_softmax_warp<ITEMS, true><<<blocks, threads, 0, stream>>>(scores, mask, output, layout);
    else
        mask_softmax_warp<ITEMS, false><<<blocks, threads, 0, stream>>>(scores, mask, output, layout);
}

extern "C" int cc_mask_softmax(
    const void* scores, const void* mask, void* output, SoftmaxLayout layout,
    int warps, int bf16, void* stream) {
    int items = 1;
    while (items * 32 < layout.keys) items *= 2;
    const float* x = static_cast<const float*>(scores);
    const bool* m = static_cast<const bool*>(mask);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (items) {
        case 1: launch_softmax<1>(x, m, output, layout, warps, bf16, s); break;
        case 2: launch_softmax<2>(x, m, output, layout, warps, bf16, s); break;
        case 4: launch_softmax<4>(x, m, output, layout, warps, bf16, s); break;
        case 8: launch_softmax<8>(x, m, output, layout, warps, bf16, s); break;
        case 16: launch_softmax<16>(x, m, output, layout, warps, bf16, s); break;
        case 32: launch_softmax<32>(x, m, output, layout, warps, bf16, s); break;
        case 64: launch_softmax<64>(x, m, output, layout, warps, bf16, s); break;
        default: return int(cudaErrorInvalidValue);
    }
    return int(cudaGetLastError());
}
