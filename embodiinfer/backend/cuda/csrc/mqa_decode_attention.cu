// Decode attention for a small block of queries whose heads all share one K/V head (MQA).
// Every query row attends to the same keys, so CTAs split the keys (and the rows into groups of
// up to 48): each computes S^T = K Q^T, a softmax over its chunk and O = P V with tensor-core MMAs
// and stores its partial output. After a grid-wide barrier the CTAs merge the partials and encode
// the NVFP4 input of the output projection, so the whole attention is one launch.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "pdl.cuh"

namespace {

constexpr int kDim = 256;
constexpr int kChunk = 64;   // keys per CTA
constexpr int kTiles = 3;    // 16-row query tiles per CTA
constexpr int kMaxSplits = 16;
constexpr int kMaxHeads = 16;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kRowBytes = kDim * 2 + 16;  // padded BF16 rows: ldmatrix over 8 rows hits distinct banks
constexpr int kSStride = kChunk + 4;      // FP32 logits [row][key]
constexpr int kPStride = kChunk + 8;      // BF16 probabilities [row][key]

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void copy16(uint32_t destination, const void* source, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(destination), "l"(source),
               "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n"); }

template <int kPending>
__device__ __forceinline__ void wait_copies() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}

__device__ __forceinline__ void ldmatrix(uint32_t address, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address));
}

__device__ __forceinline__ void ldmatrix_transposed(uint32_t address, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address));
}

__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack(float low, float high) {
  __nv_bfloat162 pair = __floats2bfloat162_rn(low, high);
  return *reinterpret_cast<uint32_t*>(&pair);
}

struct Layout {
  static constexpr int q = 0;  // reused for the FP16 partial output once S^T is done
  static constexpr int k = q + kTiles * 16 * kRowBytes;
  static constexpr int v = k + kChunk * kRowBytes;
  static constexpr int s = v + kChunk * kRowBytes;
  static constexpr int p = s + kTiles * 16 * kSStride * 4;
  static constexpr int stats = p + kTiles * 16 * kPStride * 2;  // row max, row sum
  static constexpr int bytes = stats + 2 * kTiles * 16 * 4;
};

// Merge the key-split partials of one action step (heads * 256 outputs), round to BF16, and store
// them or encode NVFP4 exactly like bhsd_to_bshd_fp4_strided: 16-wide blocks, E4M3 scales in the
// 128x4 layout, global scale g_scale. scratch holds kMaxHeads * kMaxSplits * 3 floats.
__device__ void combine_step(const __half* __restrict__ partial, const float* __restrict__ stats, int splits,
                             int heads, int steps, int step, int batch, __nv_bfloat16* __restrict__ output,
                             uint8_t* __restrict__ fp4, uint8_t* __restrict__ blocked, float g_scale,
                             float* scratch) {
  float(*split_stats)[kMaxSplits][2] = reinterpret_cast<float(*)[kMaxSplits][2]>(scratch);
  float(*weights)[kMaxSplits] = reinterpret_cast<float(*)[kMaxSplits]>(scratch + kMaxHeads * kMaxSplits * 2);
  const int rows = heads * steps;
  const float* stats_batch = stats + int64_t(batch) * splits * rows * 2;
  __syncthreads();
  for (int i = threadIdx.x; i < heads * splits * 2; i += kThreads) {
    const int head = i / (splits * 2), c = (i / 2) % splits;
    split_stats[head][c][i & 1] = __ldcg(stats_batch + (c * rows + step * heads + head) * 2 + (i & 1));
  }
  __syncthreads();
  if (threadIdx.x < heads) {
    float maximum = -INFINITY;
    for (int c = 0; c < splits; ++c) maximum = fmaxf(maximum, split_stats[threadIdx.x][c][0]);
    float total = 0.f;
    for (int c = 0; c < splits; ++c) {
      const float w = exp2f(split_stats[threadIdx.x][c][0] - maximum) * split_stats[threadIdx.x][c][1];
      weights[threadIdx.x][c] = w;
      total += w;
    }
    const float inverse = 1.f / total;
    for (int c = 0; c < splits; ++c) weights[threadIdx.x][c] *= inverse;
  }
  __syncthreads();
  const int columns = heads * kDim;
  const int token = batch * steps + step;
  for (int element = threadIdx.x * 8; element < columns; element += kThreads * 8) {
    const int head = element / kDim, row = step * heads + head;
    float sum[8] = {};
    const __half* source = partial + (int64_t(batch) * splits * rows + row) * kDim + element % kDim;
    uint4 raw[kMaxSplits];
#pragma unroll
    for (int c = 0; c < kMaxSplits; ++c)
      if (c < splits) raw[c] = __ldcg(reinterpret_cast<const uint4*>(source + int64_t(c) * rows * kDim));
#pragma unroll
    for (int c = 0; c < kMaxSplits; ++c) {
      if (c >= splits) break;
      const __half2* pairs = reinterpret_cast<const __half2*>(&raw[c]);
      const float w = weights[head][c];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(pairs[i]);
        sum[2 * i] += w * f.x;
        sum[2 * i + 1] += w * f.y;
      }
    }
    float values[8];
    float maximum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      values[i] = __bfloat162float(__float2bfloat16_rn(sum[i]));
      maximum = fmaxf(maximum, fabsf(values[i]));
    }
    if (fp4 == nullptr) {
      uint4 packed = make_uint4(pack(values[0], values[1]), pack(values[2], values[3]), pack(values[4], values[5]),
                                pack(values[6], values[7]));
      *reinterpret_cast<uint4*>(output + int64_t(token) * columns + element) = packed;
      continue;
    }
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
    const uint8_t code = __nv_cvt_float_to_fp8(fminf(448.0f, maximum / (6.0f * g_scale)), __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 converted;
    converted.__x = code;
    const float denominator = g_scale * float(converted);
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 pair = make_float2(denominator > 0.0f ? values[2 * i] / denominator : 0.0f,
                                      denominator > 0.0f ? values[2 * i + 1] / denominator : 0.0f);
      packed |= uint32_t(__nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest)) << (8 * i);
    }
    *reinterpret_cast<uint32_t*>(fp4 + (int64_t(token) * columns + element) / 2) = packed;
    if (((element / 8) & 1) == 0) {
      const int block = element / 16, column_tiles = (columns / 16 + 3) / 4;
      blocked[((token / 128) * column_tiles + block / 4) * 512 + (token % 32) * 16 + ((token % 128) / 32) * 4 +
              (block & 3)] = code;
    }
  }
}

// CTA (c, group, b): keys [64c, 64c + 64), query rows [48 * group, ...). Its partial is
// partial[b][c][r][:] = sum_j p_j v_j / sum_j p_j with p_j = 2^(l_j - max_j l_j) and
// l_j = log2(e) * scale * q_r . k_j, plus stats[b][c][r] = (max, sum). Query row r is
// (step r / heads, head r % heads). Keys below stable_keys are not written by the preceding
// kernel, so they load while it drains.
__global__ void __launch_bounds__(kThreads, 1) mqa_attention(
    const __nv_bfloat16* __restrict__ query, int64_t query_batch, int64_t query_step, int64_t query_head,
    int heads, int rows, const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value,
    int64_t kv_batch, int keys, int stable_keys, float scale_log2, __half* __restrict__ partial,
    float* __restrict__ stats_out, unsigned long long* __restrict__ counter, __nv_bfloat16* __restrict__ output,
    uint8_t* __restrict__ fp4, uint8_t* __restrict__ blocked, float g_scale) {
  using L = Layout;
  constexpr int kKeyTiles = kChunk / 16;
  extern __shared__ __align__(128) uint8_t shared[];
  float* logits = reinterpret_cast<float*>(shared + L::s);
  __nv_bfloat16* probabilities = reinterpret_cast<__nv_bfloat16*>(shared + L::p);
  float* stats = reinterpret_cast<float*>(shared + L::stats);
  __half* staged = reinterpret_cast<__half*>(shared + L::q);
  const uint32_t base = shared_address(shared);

  const int split = blockIdx.x, splits = gridDim.x, batch = blockIdx.z;
  const int row_first = blockIdx.y * kTiles * 16;
  const int tiles = min(kTiles, (rows - row_first) / 16);
  const int local_rows = tiles * 16;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int first = split * kChunk;
  const int count = min(kChunk, keys - first);

  {
    // Fixed 16-byte column per thread; rows advance by 8. Rows past the keys are zero-filled.
    const int column = threadIdx.x & 31, row0 = threadIdx.x >> 5;
    const __nv_bfloat16* k = key + batch * kv_batch + int64_t(first) * kDim + column * 8;
    const __nv_bfloat16* v = value + batch * kv_batch + int64_t(first) * kDim + column * 8;
    const int fresh = max(0, min(count, stable_keys - first));  // rows [fresh, count) wait for the producer
    for (int row = row0; row < kChunk; row += 8)
      if (row < fresh || row >= count)
        copy16(base + L::k + row * kRowBytes + column * 16, k + (row < count ? row : 0) * kDim, row < count);
    commit();
    for (int row = row0; row < kChunk; row += 8)
      if (row < fresh || row >= count)
        copy16(base + L::v + row * kRowBytes + column * 16, v + (row < count ? row : 0) * kDim, row < count);
    commit();
    pdl_wait();
    const __nv_bfloat16* q = query + batch * query_batch + column * 8;
    for (int row = row0; row < local_rows; row += 8) {
      const int r = row_first + row;
      copy16(base + L::q + row * kRowBytes + column * 16, q + (r / heads) * query_step + (r % heads) * query_head,
             true);
    }
    for (int row = row0; row < kChunk; row += 8)
      if (row >= fresh && row < count) copy16(base + L::k + row * kRowBytes + column * 16, k + row * kDim, true);
    commit();
    for (int row = row0; row < kChunk; row += 8)
      if (row >= fresh && row < count) copy16(base + L::v + row * kRowBytes + column * 16, v + row * kDim, true);
    commit();
  }
  wait_copies<1>();
  __syncthreads();

  // S^T = K Q^T in units of (16 keys, 16 query rows).
  for (int unit = warp; unit < kKeyTiles * tiles; unit += kWarps) {
    const int key_tile = unit / tiles, row_tile = unit % tiles;
    // Alternating accumulators halve the dependent MMA chain over the 256-wide head.
    float s[2][2][4] = {};
#pragma unroll
    for (int kk = 0; kk < kDim / 16; ++kk) {
      uint32_t a[4], b[4];
      ldmatrix(base + L::k + (key_tile * 16 + (lane & 15)) * kRowBytes + (2 * kk + (lane >> 4)) * 16, a);
      ldmatrix(base + L::q + (row_tile * 16 + (lane & 7) + ((lane >> 4) << 3)) * kRowBytes +
                   (2 * kk + ((lane >> 3) & 1)) * 16,
               b);
      mma(s[kk & 1][0], a, b[0], b[1]);
      mma(s[kk & 1][1], a, b[2], b[3]);
    }
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key_index = key_tile * 16 + g + 8 * (e >> 1);
        const int row = row_tile * 16 + 8 * n + 2 * t + (e & 1);
        logits[row * kSStride + key_index] =
            key_index < count ? (s[0][n][e] + s[1][n][e]) * scale_log2 : -INFINITY;
      }
    }
  }
  __syncthreads();

  // Softmax over the chunk, four lanes per query row (whole warps: local_rows * 4 is a multiple of 64).
  if (threadIdx.x < local_rows * 4) {
    const int row = threadIdx.x >> 2, quarter = threadIdx.x & 3;
    const float* line = logits + row * kSStride + quarter * (kChunk / 4);
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < kChunk / 4; ++i) maximum = fmaxf(maximum, line[i]);
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2));
    float sum = 0.f;
    __nv_bfloat16* out = probabilities + row * kPStride + quarter * (kChunk / 4);
#pragma unroll
    for (int i = 0; i < kChunk / 4; i += 2) {
      const float p0 = exp2f(line[i] - maximum), p1 = exp2f(line[i + 1] - maximum);
      sum += p0 + p1;
      *reinterpret_cast<uint32_t*>(out + i) = pack(p0, p1);
    }
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    if (quarter == 0) {
      stats[2 * row] = maximum;
      stats[2 * row + 1] = sum;
    }
  }
  wait_copies<0>();
  __syncthreads();

  // O = P V: each warp owns 32 output columns for every query row of the group.
  float acc[kTiles][4][4] = {};
#pragma unroll
  for (int kk = 0; kk < kChunk / 16; ++kk) {
    uint32_t b[2][4];
#pragma unroll
    for (int n = 0; n < 2; ++n)
      ldmatrix_transposed(base + L::v + (kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3)) * kRowBytes +
                              ((warp * 32 + n * 16) / 8 + (lane >> 4)) * 16,
                          b[n]);
#pragma unroll
    for (int m = 0; m < kTiles; ++m) {
      if (m >= tiles) break;
      uint32_t a[4];
      ldmatrix(base + L::p + (m * 16 + (lane & 15)) * kPStride * 2 + (2 * kk + (lane >> 4)) * 16, a);
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        mma(acc[m][2 * n], a, b[n][0], b[n][1]);
        mma(acc[m][2 * n + 1], a, b[n][2], b[n][3]);
      }
    }
  }
  // Normalized partials stay within the range of V, so FP16 holds them.
#pragma unroll
  for (int m = 0; m < kTiles; ++m) {
    if (m >= tiles) break;
    const int row = m * 16 + g;
    const float inverse0 = 1.f / stats[2 * row + 1], inverse1 = 1.f / stats[2 * (row + 8) + 1];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int column = warp * 32 + 8 * n + 2 * t;
      *reinterpret_cast<__half2*>(staged + row * (kRowBytes / 2) + column) =
          __floats2half2_rn(acc[m][n][0] * inverse0, acc[m][n][1] * inverse0);
      *reinterpret_cast<__half2*>(staged + (row + 8) * (kRowBytes / 2) + column) =
          __floats2half2_rn(acc[m][n][2] * inverse1, acc[m][n][3] * inverse1);
    }
  }
  __syncthreads();
  __half* out = partial + ((int64_t(batch) * splits + split) * rows + row_first) * kDim;
  for (int i = threadIdx.x; i < local_rows * (kDim / 8); i += kThreads) {
    const int row = i / (kDim / 8), column = i % (kDim / 8);
    *reinterpret_cast<uint4*>(out + row * kDim + column * 8) =
        *reinterpret_cast<const uint4*>(shared + L::q + row * kRowBytes + column * 16);
  }
  float* stats_split = stats_out + ((int64_t(batch) * splits + split) * rows + row_first) * 2;
  for (int i = threadIdx.x; i < 2 * local_rows; i += kThreads) stats_split[i] = stats[i];

  // Grid-wide barrier: the host keeps the grid within one wave, so every CTA is resident, and
  // dependents launch only after it, so they cannot take a slot a waiting CTA needs.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned long long total = gridDim.x * gridDim.y * gridDim.z;
    const unsigned long long target = (atomicAdd(counter, 1ull) / total + 1) * total;
    while (*reinterpret_cast<volatile unsigned long long*>(counter) < target) __nanosleep(32);
    __threadfence();
  }
  __syncthreads();
  pdl_launch_dependents();
  const int cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  const int steps = rows / heads;
  for (int unit = cta; unit < steps * int(gridDim.z); unit += gridDim.x * gridDim.y * gridDim.z)
    combine_step(partial, stats_out, splits, heads, steps, unit % steps, unit / steps, output, fp4, blocked, g_scale,
                 reinterpret_cast<float*>(shared));
}

}  // namespace

// Attention of folded query rows r = step * heads + head over K/V [batch, keys, 256] (contiguous), with
// query element (b, step, head, d) at query + b * query_batch + step * query_step + head * query_head + d.
// Keys below stable_keys must not be written by the kernel preceding this launch.
// partial holds [batch, ceil(keys / 64), rows, 256] FP16, stats [batch, ceil(keys / 64), rows, 2] FP32 and
// counter one zero-initialized uint64 private to this stream. With fp4 == nullptr the output is BF16
// [batch * steps, heads * 256]; otherwise fp4 receives the NVFP4 encoding of that BF16 output with
// block scales in blocked.
extern "C" int cc_mqa_decode_attention(int batch, int rows, int heads, int keys, int stable_keys, const void* query,
                                       int64_t query_batch, int64_t query_step, int64_t query_head, const void* key,
                                       const void* value, int64_t kv_batch, float scale, void* partial, void* stats,
                                       void* counter, void* output, void* fp4, void* blocked, float g_scale, int pdl,
                                       void* stream) {
  if (keys < 1 || keys > kMaxSplits * kChunk || heads < 1 || heads > kMaxHeads || rows % heads || rows % 16 ||
      rows > 128)
    return 1;
  static int sms = 0;
  if (sms == 0) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (cudaFuncSetAttribute(mqa_attention, cudaFuncAttributeMaxDynamicSharedMemorySize, Layout::bytes) !=
        cudaSuccess)
      return 3;
  }
  const int splits = (keys + kChunk - 1) / kChunk;
  const int groups = (rows / 16 + kTiles - 1) / kTiles;
  // The grid-wide barrier needs every CTA resident at once: at most one per SM.
  if (splits * groups * batch > sms) return 2;
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(splits, groups, batch);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = Layout::bytes;
  config.stream = static_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attribute;
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = pdl ? 1 : 0;
  cudaError_t status = cudaLaunchKernelEx(
      &config, mqa_attention, static_cast<const __nv_bfloat16*>(query), query_batch, query_step, query_head, heads,
      rows, static_cast<const __nv_bfloat16*>(key), static_cast<const __nv_bfloat16*>(value), kv_batch, keys,
      stable_keys, scale * 1.4426950408889634f, static_cast<__half*>(partial), static_cast<float*>(stats),
      static_cast<unsigned long long*>(counter), static_cast<__nv_bfloat16*>(output), static_cast<uint8_t*>(fp4),
      static_cast<uint8_t*>(blocked), g_scale);
  return status == cudaSuccess ? 0 : 4;
}
