// BF16 -> FP8/NVFP4 quantizers. All work uses the caller's stream.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif
#include <cfloat>
#include <cstdint>


// FP8/BF16 kernels also compile on Ampere/Ada toolkits. Python rejects FP4
// plans on those architectures before any allocation or kernel launch.
__device__ uint8_t cc_fp4_cast(float2 values) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && CUDART_VERSION >= 12080
    return __nv_cvt_float2_to_fp4x2(values, __NV_E2M1, cudaRoundNearest);
#else
    return 0;
#endif
}

constexpr int THREADS = 256;
constexpr int REDUCE_ELEMENTS = 4096;

__device__ float block_max(float value) {
    __shared__ float warps[8];
    for (int delta = 16; delta; delta /= 2)
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, delta));
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x / 32] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warps[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32)
        for (int delta = 16; delta; delta /= 2)
            value = fmaxf(value, __shfl_down_sync(0xffffffff, value, delta));
    return value;
}

__global__ void reduce_input(const __nv_bfloat16* input, float* partials, int64_t count) {
    float maximum = 0.0f;
    int64_t start = int64_t(blockIdx.x) * REDUCE_ELEMENTS;
    for (int i = threadIdx.x; i < REDUCE_ELEMENTS && start + i < count; i += THREADS)
        maximum = fmaxf(maximum, fabsf(__bfloat162float(input[start + i])));
    maximum = block_max(maximum);
    if (threadIdx.x == 0) partials[blockIdx.x] = maximum;
}

__global__ void finish_scale(const float* partials, float* scale, int count, float bound) {
    float maximum = 0.0f;
    for (int i = threadIdx.x; i < count; i += THREADS)
        maximum = fmaxf(maximum, partials[i]);
    maximum = block_max(maximum);
    // A scale of 1 for an all-zero tensor avoids division by zero and encodes zero exactly.
    // ATen scalar division multiplies by a rounded FP32 reciprocal.
    if (threadIdx.x == 0) *scale = maximum > 0.0f ? maximum * (1.0f / bound) : 1.0f;
}

__global__ void encode_fp8(const __nv_bfloat16* input, uint8_t* output,
                           const float* scale, int64_t count) {
    int64_t index = (int64_t(blockIdx.x) * THREADS + threadIdx.x) * 2;
    if (index >= count) return;
    float first = __bfloat162float(input[index]) / *scale;
    float second = index + 1 < count ? __bfloat162float(input[index + 1]) / *scale : 0.0f;
    uint16_t packed = __nv_cvt_float2_to_fp8x2(make_float2(first, second),
                                             __NV_SATFINITE, __NV_E4M3);
    if (index + 1 < count) reinterpret_cast<uint16_t*>(output)[index / 2] = packed;
    else output[index] = packed & 255;
}

__global__ void encode_fp4(const __nv_bfloat16* input, uint8_t* output,
                           uint8_t* scales, uint8_t* blocked, const float* global,
                           int rows, int columns) {
    // Eight threads encode a 16-element block. Full warps execute even for padded rows.
    int row = blockIdx.y;
    int column_block = blockIdx.x * (THREADS / 8) + threadIdx.x / 8;
    int blocks_per_row = columns / 16;
    if (column_block >= blocks_per_row) return;
    int block = row * blocks_per_row + column_block;
    int lane = threadIdx.x & 7;
    int64_t index = int64_t(row) * columns + column_block * 16 + lane * 2;
    float first = row < rows ? __bfloat162float(input[index]) : 0.0f;
    float second = row < rows ? __bfloat162float(input[index + 1]) : 0.0f;
    float maximum = fmaxf(fabsf(first), fabsf(second));
    for (int delta = 4; delta; delta /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta, 8));
    float raw_scale = fminf(448.0f, maximum / (6.0f * *global));
    uint8_t scale_code = __nv_cvt_float_to_fp8(raw_scale, __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 converted;
    converted.__x = scale_code;
    float local_scale = float(converted);
    float denominator = *global * local_scale;
    // A rounded zero block scale represents an all-zero block, including underflow.
    float normalized_first = denominator > 0.0f ? first / denominator : 0.0f;
    float normalized_second = denominator > 0.0f ? second / denominator : 0.0f;
    uint8_t packed = cc_fp4_cast(
        make_float2(normalized_first, normalized_second));
    if (row < rows) output[index / 2] = packed;
    if (lane == 0) {
        int column_tiles = (blocks_per_row + 3) / 4;
        int offset = ((row / 128) * column_tiles + column_block / 4) * 512
                   + (row % 32) * 16 + ((row % 128) / 32) * 4 + column_block % 4;
        blocked[offset] = scale_code;
        if (row < rows) scales[block] = scale_code;
    }
}

__global__ void scale_result(__nv_bfloat16* output, const float* input_scale,
                              const float* weight_scale, int64_t count) {
    int64_t index = int64_t(blockIdx.x) * THREADS + threadIdx.x;
    if (index < count) {
        // Match the existing PyTorch reference: cast the scale product to BF16 first.
        float scale = __bfloat162float(__float2bfloat16(*input_scale * *weight_scale));
        output[index] = __float2bfloat16(__bfloat162float(output[index]) * scale);
    }
}

extern "C" int cc_scale(const void* input, void* partials, void* scale,
                         int64_t count, float bound, void* stream) {
    int blocks = (count + REDUCE_ELEMENTS - 1) / REDUCE_ELEMENTS;
    reduce_input<<<blocks, THREADS, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (float*)partials, count);
    finish_scale<<<1, THREADS, 0, (cudaStream_t)stream>>>(
        (const float*)partials, (float*)scale, blocks, bound);
    return int(cudaGetLastError());
}

extern "C" int cc_fp8(const void* input, void* output, const void* scale,
                       int64_t count, void* stream) {
    encode_fp8<<<(count + THREADS * 2 - 1) / (THREADS * 2), THREADS, 0,
                  (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (uint8_t*)output, (const float*)scale, count);
    return int(cudaGetLastError());
}

extern "C" int cc_fp4(const void* input, void* output, void* scales, void* blocked,
                       const void* global, int rows, int columns, void* stream) {
    encode_fp4<<<dim3((columns / 16 + THREADS / 8 - 1) / (THREADS / 8), rows), THREADS, 0,
                  (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (uint8_t*)output, (uint8_t*)scales,
        (uint8_t*)blocked, (const float*)global, rows, columns);
    return int(cudaGetLastError());
}

extern "C" int cc_rescale(void* output, const void* input_scale,
                           const void* weight_scale, int64_t count, void* stream) {
    scale_result<<<(count + THREADS - 1) / THREADS, THREADS, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)output, (const float*)input_scale, (const float*)weight_scale, count);
    return int(cudaGetLastError());
}
