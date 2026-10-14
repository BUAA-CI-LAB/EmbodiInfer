#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Keep the original FP32 products and add/subtract separate. The two outputs
// are rounded to BF16 only after rotation, as in the reference _apply_rope.
__global__ void rotate(const __nv_bfloat16* input, const float* sine,
                       const float* cosine, __nv_bfloat16* output,
                       int64_t pairs, int heads, int half_width) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pairs) return;
  int column = index % half_width;
  int64_t row = index / half_width;
  int64_t factor = (row / heads) * half_width + column;
  int64_t offset = row * (2 * half_width) + column;
  float first = __bfloat162float(input[offset]);
  float second = __bfloat162float(input[offset + half_width]);
  float s = sine[factor], c = cosine[factor];
  output[offset] = __float2bfloat16_rn(
      __fsub_rn(__fmul_rn(first, c), __fmul_rn(second, s)));
  output[offset + half_width] = __float2bfloat16_rn(
      __fadd_rn(__fmul_rn(second, c), __fmul_rn(first, s)));
}

extern "C" int cc_rotary(const void* input, const float* sine,
                          const float* cosine, void* output, int64_t pairs,
                          int heads, int half_width, void* stream) {
  if (pairs == 0) return 0;
  rotate<<<(pairs + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(input), sine, cosine,
      static_cast<__nv_bfloat16*>(output), pairs, heads, half_width);
  return cudaGetLastError();
}
