#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "pdl.cuh"

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

__device__ __forceinline__ __nv_bfloat162 round_and_scale2(float a, float b, float scale) {
    __nv_bfloat162 r = __floats2bfloat162_rn(a, b);
    if (scale != 1.0f) {
        float2 f = __bfloat1622float2(r);
        r = __floats2bfloat162_rn(f.x * scale, f.y * scale);
    }
    return r;
}

__global__ void split_qkv_rope_vec4(
    const __nv_bfloat16* __restrict__ qkv,
    const float* __restrict__ cos_ptr,
    const float* __restrict__ sin_ptr,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int T, int QH, int KVH, int D,
    int kv_total_t, int kv_offset_t, int q_bthd, float q_scale)
{
    pdl_wait();
    pdl_launch_dependents();
    int bt = blockIdx.x;
    int tid = threadIdx.x;
    int total_heads = QH + 2 * KVH;
    int half_d = D >> 1;
    int b = bt / T;
    int t = bt - b * T;
    int64_t row_base = int64_t(bt) * (total_heads * D);
    int64_t cs_base = int64_t(bt) * half_d;

    __shared__ float4 s_cos[32];
    __shared__ float4 s_sin[32];
    if (tid < 32) {
        s_cos[tid] = reinterpret_cast<const float4*>(cos_ptr + cs_base)[tid];
        s_sin[tid] = reinterpret_cast<const float4*>(sin_ptr + cs_base)[tid];
    }
    __syncthreads();

    if (tid < (QH + KVH) * 32) {
        int h = tid >> 5;
        int chunk = tid & 31;
        int64_t src = row_base + h * D + chunk * 4;
        uint2 x1_u2 = *reinterpret_cast<const uint2*>(qkv + src);
        uint2 x2_u2 = *reinterpret_cast<const uint2*>(qkv + src + half_d);
        float4 c4 = s_cos[chunk];
        float4 s4 = s_sin[chunk];

        const __nv_bfloat162* x1_b2 = reinterpret_cast<const __nv_bfloat162*>(&x1_u2);
        const __nv_bfloat162* x2_b2 = reinterpret_cast<const __nv_bfloat162*>(&x2_u2);
        float2 x1_0 = __bfloat1622float2(x1_b2[0]);
        float2 x1_1 = __bfloat1622float2(x1_b2[1]);
        float2 x2_0 = __bfloat1622float2(x2_b2[0]);
        float2 x2_1 = __bfloat1622float2(x2_b2[1]);

        uint2 y1_u2, y2_u2;
        __nv_bfloat162* y1_b2 = reinterpret_cast<__nv_bfloat162*>(&y1_u2);
        __nv_bfloat162* y2_b2 = reinterpret_cast<__nv_bfloat162*>(&y2_u2);

        float scale = (h < QH) ? q_scale : 1.0f;
        y1_b2[0] = round_and_scale2(x1_0.x * c4.x - x2_0.x * s4.x, x1_0.y * c4.y - x2_0.y * s4.y, scale);
        y1_b2[1] = round_and_scale2(x1_1.x * c4.z - x2_1.x * s4.z, x1_1.y * c4.w - x2_1.y * s4.w, scale);
        y2_b2[0] = round_and_scale2(x2_0.x * c4.x + x1_0.x * s4.x, x2_0.y * c4.y + x1_0.y * s4.y, scale);
        y2_b2[1] = round_and_scale2(x2_1.x * c4.z + x1_1.x * s4.z, x2_1.y * c4.w + x1_1.y * s4.w, scale);

        __nv_bfloat16* dst;
        if (h < QH) {
            int64_t q_idx = q_bthd ? ((int64_t(b) * T + t) * QH + h) : ((int64_t(b) * QH + h) * T + t);
            dst = q_out + q_idx * D + chunk * 4;
        } else {
            dst = k_out + ((int64_t(b) * KVH + (h - QH)) * kv_total_t + (kv_offset_t + t)) * D + chunk * 4;
        }
        *reinterpret_cast<uint2*>(dst) = y1_u2;
        *reinterpret_cast<uint2*>(dst + half_d) = y2_u2;
    } else if (tid < (QH + KVH) * 32 + KVH * 32) {
        int v_tid = tid - (QH + KVH) * 32;
        int vh = v_tid >> 5;
        int col = (v_tid & 31) * 8;
        int64_t src = row_base + (QH + KVH + vh) * D + col;
        int64_t dst = ((int64_t(b) * KVH + vh) * kv_total_t + (kv_offset_t + t)) * D + col;
        *reinterpret_cast<uint4*>(v_out + dst) = *reinterpret_cast<const uint4*>(qkv + src);
    }
}

extern "C" int cc_split_qkv_rope(
    const void* qkv, const void* cos_ptr, const void* sin_ptr,
    void* q_out, void* k_out, void* v_out,
    int B, int T, int QH, int KVH, int D,
    int kv_total_t, int kv_offset_t, int q_bthd, float q_scale, void* stream)
{
    int threads = (QH + KVH + KVH) * 32;
    return int(launch_pdl(
        split_qkv_rope_vec4, dim3(B * T), dim3(threads), 0, (cudaStream_t)stream,
        (const __nv_bfloat16*)qkv, (const float*)cos_ptr, (const float*)sin_ptr,
        (__nv_bfloat16*)q_out, (__nv_bfloat16*)k_out, (__nv_bfloat16*)v_out,
        T, QH, KVH, D, kv_total_t, kv_offset_t, q_bthd, q_scale));
}
