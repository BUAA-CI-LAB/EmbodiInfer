// Flow-step bridge between expert tower passes in one launch: the FP32 action output projection,
// the Euler update x + v * dt, and the FP32 action input projection of the next step rounded to the
// BF16 tower input.
#include <cuda_bf16.h>

#include <cstdint>

#include "pdl.cuh"

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kWidth = 1024;   // tower width
constexpr int kActions = 32;   // action dimension; one CTA per action column
constexpr int kMaxRows = 16;   // batch * horizon
constexpr int kPer = kWidth / kActions;

// Phase 1: CTA j computes column j of v for every row (one warp per row) and updates x. Phase 2
// (when embeds is set): CTA j computes hidden columns [32j, 32j + 32) of the next embeddings. Each
// phase stages its operands in shared memory with one round of independent loads.
__global__ void __launch_bounds__(kThreads) action_bridge(
    const __nv_bfloat16* __restrict__ hidden, const float* __restrict__ x, float* __restrict__ x_next,
    const float* __restrict__ out_weight, const float* __restrict__ out_bias, float dt,
    const float* __restrict__ in_weight, const float* __restrict__ in_bias, __nv_bfloat16* __restrict__ embeds,
    int rows, unsigned long long* __restrict__ counter) {
  __shared__ __align__(16) __nv_bfloat16 s_hidden[kMaxRows][kWidth];
  __shared__ __align__(16) float s_weight[kWidth];
  __shared__ __align__(16) float s_state[kMaxRows][kActions];
  __shared__ __align__(16) float s_in[kPer][kActions + 4];
  const int column = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;

  // The weights do not depend on the preceding kernel.
  for (int i = threadIdx.x; i < kWidth / 4; i += kThreads)
    reinterpret_cast<float4*>(s_weight)[i] = reinterpret_cast<const float4*>(out_weight + column * kWidth)[i];
  if (embeds != nullptr)
    for (int i = threadIdx.x; i < kPer * kActions / 4; i += kThreads) {
      const int n = i / (kActions / 4), part = i % (kActions / 4);
      *reinterpret_cast<float4*>(&s_in[n][part * 4]) =
          reinterpret_cast<const float4*>(in_weight + (column * kPer + n) * kActions)[part];
    }
  pdl_wait();
  for (int i = threadIdx.x; i < rows * kWidth / 8; i += kThreads)
    reinterpret_cast<uint4*>(&s_hidden[0][0])[i] = reinterpret_cast<const uint4*>(hidden)[i];
  __syncthreads();
  for (int row = warp; row < rows; row += kWarps) {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < kWidth / 32 / 8; ++i) {
      const int k = (i * 32 + lane) * 8;
      const uint4 raw = *reinterpret_cast<const uint4*>(&s_hidden[row][k]);
      const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const float2 h = __bfloat1622float2(pairs[p]);
        sum = fmaf(h.x, s_weight[k + 2 * p], sum);
        sum = fmaf(h.y, s_weight[k + 2 * p + 1], sum);
      }
    }
    for (int offset = 16; offset; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane == 0) {
      const float v = out_bias != nullptr ? sum + out_bias[column] : sum;
      const int64_t index = int64_t(row) * kActions + column;
      x_next[index] = __fadd_rn(x[index], __fmul_rn(v, dt));
    }
  }
  if (embeds == nullptr) {
    pdl_launch_dependents();
    return;
  }
  // Grid-wide barrier: every CTA is resident (small CTAs, one wave), and dependents launch only
  // after it, so they cannot take a slot a waiting CTA needs.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned long long total = gridDim.x;
    const unsigned long long target = (atomicAdd(counter, 1ull) / total + 1) * total;
    while (*reinterpret_cast<volatile unsigned long long*>(counter) < target) __nanosleep(32);
    __threadfence();
  }
  __syncthreads();
  pdl_launch_dependents();
  for (int i = threadIdx.x; i < rows * kActions / 4; i += kThreads)
    reinterpret_cast<float4*>(&s_state[0][0])[i] = __ldcg(reinterpret_cast<const float4*>(x_next) + i);
  __syncthreads();
  for (int item = threadIdx.x; item < rows * kPer; item += kThreads) {
    const int row = item / kPer, n = item % kPer;
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kActions; ++j) sum = fmaf(s_state[row][j], s_in[n][j], sum);
    if (in_bias != nullptr) sum += in_bias[column * kPer + n];
    embeds[int64_t(row) * kWidth + column * kPer + n] = __float2bfloat16_rn(sum);
  }
}

}  // namespace

// x_next[rows, 32] = x + dt * (hidden[rows, 1024] out_weight[32, 1024]^T + out_bias), and when embeds
// is set, embeds[rows, 1024] = bf16(x_next in_weight[1024, 32]^T + in_bias). counter is one
// zero-initialized uint64 private to this stream.
extern "C" int cc_action_bridge(const void* hidden, const void* x, void* x_next, const void* out_weight,
                                const void* out_bias, float dt, const void* in_weight, const void* in_bias,
                                void* embeds, int rows, int width, int actions, void* counter, void* stream) {
  if (rows < 1 || rows > kMaxRows || width != kWidth || actions != kActions) return 1;
  return int(launch_pdl(action_bridge, dim3(kActions), dim3(kThreads), 0, static_cast<cudaStream_t>(stream),
                        static_cast<const __nv_bfloat16*>(hidden), static_cast<const float*>(x),
                        static_cast<float*>(x_next), static_cast<const float*>(out_weight),
                        static_cast<const float*>(out_bias), dt, static_cast<const float*>(in_weight),
                        static_cast<const float*>(in_bias), static_cast<__nv_bfloat16*>(embeds), rows,
                        static_cast<unsigned long long*>(counter)));
}
