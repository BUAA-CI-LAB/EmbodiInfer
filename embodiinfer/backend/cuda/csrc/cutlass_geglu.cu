// Prefix GeGLU on Blackwell (SM100/SM110): a swap-AB CUTLASS NVFP4 block-scaled GEMM over 8-row-interleaved
// [gate f..f+7; up f..f+7] weights whose epilogue fuses exact GeGLU and NVFP4 block quantization of the
// product. The weights are the MMA M side (2-SM 256-row tiles) and the tokens are N in 192-wide tiles, so a
// 512 + text token prefix pads to 576 rather than 768 rows. The epilogue writes the packed product and its
// e4m3 block scales in the 128x4 layout the Down GEMM consumes, with an L2 evict_last policy: the prefix
// layers share one output buffer, which then stays resident for each Down GEMM instead of being written back
// under the weight stream.
// Launches are programmatic; without this CUTLASS compiles griddepcontrol.wait out of the kernel.
#define CUTLASS_ENABLE_GDC_FOR_SM100 1
// CUTLASS device asserts otherwise sit in the epilogue's TMEM load path.
#ifndef NDEBUG
#define NDEBUG
#endif
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <map>
#include <mutex>
#include <tuple>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace embodiinfer_cutlass_geglu {
using namespace cute;

CUTLASS_DEVICE float rcp_approx(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

CUTLASS_DEVICE float ex2_approx(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// Exact GELU of 16 values; erf from Abramowitz-Stegun 7.1.26 (|error| <= 1.5e-7). Each stage spans
// all 16 values so the epilogue warps keep independent instructions in flight.
CUTLASS_DEVICE void gelu_erf16(float (&x)[16]) {
  float z[16], t[16], e[16];
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 16; ++j) z[j] = fabsf(x[j]) * 0.70710678118654752f;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 16; ++j) t[j] = rcp_approx(fmaf(0.3275911f, z[j], 1.0f));
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 16; ++j) e[j] = ex2_approx(-z[j] * z[j] * 1.4426950408889634f);
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 16; ++j) {
    float poly = fmaf(fmaf(fmaf(fmaf(1.061405429f, t[j], -1.453152027f), t[j], 1.421413741f), t[j],
                           -0.284496736f), t[j], 0.254829592f) * t[j];
    float erf_z = fmaf(-poly, e[j], 1.0f);
    x[j] = 0.5f * x[j] * (1.0f + copysignf(erf_z, x[j]));
  }
}

constexpr int TileN = 192;
constexpr int EpilogueWarps = 8;

// Each epilogue warp owns one TMEM lane quarter of the CTA's 128 weight rows, i.e. one 16-feature NVFP4
// block; the two warps per quarter take alternate 32-token chunks of the tile.
class GegluNvfp4Epilogue {
 public:
  using DispatchPolicy = cutlass::epilogue::Sm100NoSmem;
  using ElementC = void;
  using ElementD = cutlass::float_e2m1_t;
  using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD = StrideC;
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;
  using EpilogueTile = Shape<_128, _32>;
  using CopyOpT2R = SM100_TMEM_LOAD_32dp32b32x;
  struct ThreadEpilogueOp {
    using ElementOutput = cutlass::float_e2m1_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
  };
  static constexpr int ThreadCount = 32 * EpilogueWarps;
  static constexpr uint32_t TmaTransactionBytes = 0;

  // Per warp, a 32-token chunk's products [token][2 features x 8], padded to 20 floats per token so both the
  // 64-bit stores and the 128-bit row reads are bank-conflict free.
  struct SharedStorage {
    alignas(16) float product[EpilogueWarps][32 * 20];
  };

  struct Arguments {
    uint8_t* out = nullptr;          // [M, features / 2] packed e2m1 pairs
    uint8_t* out_scale = nullptr;    // e4m3 block scales, 128x4 interleaved
    float const* gate_scale = nullptr;
    float const* up_scale = nullptr;
    float const* inv_scale = nullptr;
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) { return args; }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }
  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&, Arguments const&, void*, cudaStream_t,
                                              cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }
  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto shape = append<4>(problem_shape, 1);
    return get<0>(shape) % 256 == 0 && args.out != nullptr && args.out_scale != nullptr;
  }

  CUTLASS_DEVICE GegluNvfp4Epilogue(Params const& params, SharedStorage&) : params(params) {}

  template <bool ReuseTmem = false, class LoadPipeline, class LoadPipelineState, class AccumulatorPipeline,
            class AccumulatorPipelineState, class ProblemShapeMNKL, class CtaTileMNK, class CtaCoordMNKL,
            class AccEngine, class AccLayout>
  CUTLASS_DEVICE auto operator()(LoadPipeline, LoadPipelineState load_pipe_consumer_state,
                                 AccumulatorPipeline acc_pipeline, AccumulatorPipelineState acc_pipe_consumer_state,
                                 ProblemShapeMNKL problem_shape_mnkl, CtaTileMNK cta_tile_mnk,
                                 CtaCoordMNKL cta_coord_mnkl, cute::Tensor<AccEngine, AccLayout> accumulators,
                                 SharedStorage& shared) {
    static_assert(is_tmem<AccEngine>::value, "Accumulator must be TMEM resident.");
    static_assert(!ReuseTmem, "accumulators of this tile do not overlap");
    static_assert(size<0>(CtaTileMNK{}) == 128 && size<1>(CtaTileMNK{}) == TileN,
                  "per-CTA tile is 128 weight rows by TileN tokens");
    constexpr int Chunks = TileN / 32;
    constexpr int Groups = EpilogueWarps / 4;
    int const features = get<0>(problem_shape_mnkl) / 2;
    int const tokens = get<1>(problem_shape_mnkl);
    int const thread_idx = threadIdx.x % ThreadCount;
    int const lane = threadIdx.x & 31;
    int const warp = thread_idx / 32;
    int const quarter = warp & 3;
    int const group = thread_idx / 128;
    int const t0 = lane & 3, t1 = lane >> 2;
    int const block = int(get<0>(cta_coord_mnkl)) * 4 + quarter;
    int const token_base = int(get<1>(cta_coord_mnkl)) * TileN;
    int const column_tiles = (features / 16 + 3) / 4;
    float const gs = *params.gate_scale;
    float const uc = *params.up_scale * *params.inv_scale;
    uint32_t const tmem = accumulators.data().get() + (uint32_t(32 * quarter) << 16);
    float* buffer = shared.product[warp];
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));

    CUTLASS_PRAGMA_NO_UNROLL
    for (int chunk = group; chunk < Chunks; chunk += Groups) {
      // A 16x256b load hands thread t rows t/4 and t/4 + 8 of a 16-lane half, the gate and up of one
      // feature, at tokens 8r + 2(t%4) + {0, 1}: features t/4 (lanes 0-15) and 8 + t/4 (lanes 16-31).
      uint32_t a[16], b[16];
      SM100_TMEM_LOAD_16dp256b4x::copy(tmem + 32 * chunk, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                                       a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
      SM100_TMEM_LOAD_16dp256b4x::copy(tmem + (16u << 16) + 32 * chunk, b[0], b[1], b[2], b[3], b[4], b[5], b[6],
                                       b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
      cutlass::arch::fence_view_async_tmem_load();
      if (chunk + Groups >= Chunks) {
        acc_pipeline.consumer_release(acc_pipe_consumer_state);
        ++acc_pipe_consumer_state;
      }
      int const token0 = token_base + chunk * 32;
      if (token0 >= tokens) continue;
      float g[16];
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < 4; ++r) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 2; ++j) {
          g[2 * r + j] = __uint_as_float(a[4 * r + j]) * gs;
          g[8 + 2 * r + j] = __uint_as_float(b[4 * r + j]) * gs;
        }
      }
      gelu_erf16(g);
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < 4; ++r) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 2; ++j) {
          *reinterpret_cast<float2*>(buffer + (8 * r + 2 * t0 + j) * 20 + 2 * t1) =
              make_float2(g[2 * r + j] * (__uint_as_float(a[4 * r + 2 + j]) * uc),
                          g[8 + 2 * r + j] * (__uint_as_float(b[4 * r + 2 + j]) * uc));
        }
      }
      __syncwarp();
      int const token = token0 + lane;
      if (token < tokens) {
        // Row `lane` holds features [0, 8, 1, 9, ..., 7, 15] of this block.
        float v[16];
        CUTLASS_PRAGMA_UNROLL
        for (int q = 0; q < 4; ++q) {
          float4 const x = *reinterpret_cast<float4 const*>(buffer + lane * 20 + 4 * q);
          v[4 * q] = x.x;
          v[4 * q + 1] = x.y;
          v[4 * q + 2] = x.z;
          v[4 * q + 3] = x.w;
        }
        float prod[16];
        CUTLASS_PRAGMA_UNROLL
        for (int f = 0; f < 8; ++f) {
          prod[f] = v[2 * f];
          prod[8 + f] = v[2 * f + 1];
        }
        float amax = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fabsf(prod[j]));
        uint8_t code = __nv_cvt_float_to_fp8(fminf(amax * (1.0f / 6.0f), 448.0f), __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_e4m3 dequant;
        dequant.__x = code;
        float den = float(dequant);
        float rden = den > 0.0f ? __frcp_rn(den) : 0.0f;
        uint32_t lo = 0, hi = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          lo |= uint32_t(__nv_cvt_float2_to_fp4x2(make_float2(prod[2 * j] * rden, prod[2 * j + 1] * rden),
                                                  __NV_E2M1, cudaRoundNearest)) << (8 * j);
          hi |= uint32_t(__nv_cvt_float2_to_fp4x2(make_float2(prod[8 + 2 * j] * rden, prod[9 + 2 * j] * rden),
                                                  __NV_E2M1, cudaRoundNearest)) << (8 * j);
        }
        uint8_t* dst = params.out + size_t(token) * (features / 2) + block * 8;
        asm volatile("st.global.L2::cache_hint.v2.u32 [%0], {%1, %2}, %3;" ::"l"(dst), "r"(lo), "r"(hi),
                     "l"(policy) : "memory");
        uint8_t* scale_dst = params.out_scale + ((token / 128) * column_tiles + block / 4) * 512 + (token % 32) * 16 +
                             ((token % 128) / 32) * 4 + (block % 4);
        asm volatile("st.global.L2::cache_hint.u8 [%0], %1, %2;" ::"l"(scale_dst), "h"(uint16_t(code)), "l"(policy)
                     : "memory");
      }
      __syncwarp();
    }
    return cute::make_tuple(acc_pipe_consumer_state, load_pipe_consumer_state);
  }

 protected:
  Params const& params;
};

using ElementIn = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using MmaTile = Shape<_256, Int<TileN>, _256>;
using Cluster = Shape<_2, _1, _1>;
using Epilogue = cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<GegluNvfp4Epilogue>;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, ElementIn, cutlass::layout::RowMajor, 32,
    ElementIn, cutlass::layout::ColumnMajor, 32, float, MmaTile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using SfConfig = typename Mainloop::Sm1xxBlkScaledConfig;

struct Plan {
  Gemm gemm;
  void* workspace = nullptr;
  size_t workspace_size = 0;
};

std::mutex plan_mutex;
std::map<std::tuple<int, int, int, int>, Plan*> plans;

}  // namespace embodiinfer_cutlass_geglu

// out[M, F/2] (+ block scales) = NVFP4(GELU(gate_scale * A Wg^T) * up_scale * (A Wu^T) * inv_scale), with
// w = rows [gate 0..7, up 0..7, gate 8..15, up 8..15, ...] of shape [2F, K/2] and its 128x4 interleaved
// block scales.
extern "C" int cc_cutlass_geglu_fp4(
    int M, int K, int F,
    const void* a, const void* a_scale, const void* w, const void* w_scale,
    const void* gate_scale, const void* up_scale, const void* inv_scale,
    void* out, void* out_scale, int pdl, void* stream) {
  using namespace embodiinfer_cutlass_geglu;
  int device = 0;
  cudaGetDevice(&device);
  std::lock_guard<std::mutex> lock(plan_mutex);
  Plan*& plan = plans[std::make_tuple(device, M, K, F)];
  if (plan == nullptr) plan = new Plan{};
  auto stride_A = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {2 * F, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {M, K, 1});
  auto layout_SFA = SfConfig::tile_atom_to_shape_SFA(make_shape(2 * F, M, K, 1));
  auto layout_SFB = SfConfig::tile_atom_to_shape_SFB(make_shape(2 * F, M, K, 1));
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {2 * F, M, K, 1},
      {static_cast<cutlass::float_e2m1_t const*>(w), stride_A, static_cast<cutlass::float_e2m1_t const*>(a), stride_B,
       static_cast<cutlass::float_ue4m3_t const*>(w_scale), layout_SFA,
       static_cast<cutlass::float_ue4m3_t const*>(a_scale), layout_SFB},
      {static_cast<uint8_t*>(out), static_cast<uint8_t*>(out_scale), static_cast<float const*>(gate_scale),
       static_cast<float const*>(up_scale), static_cast<float const*>(inv_scale)}};
  if (plan->gemm.can_implement(args) != cutlass::Status::kSuccess) return 1;
  size_t workspace = Gemm::get_workspace_size(args);
  if (workspace > plan->workspace_size) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &capture);
    if (capture != cudaStreamCaptureStatusNone) return 3;   // warm this shape before graph capture
    if (plan->workspace) cudaFree(plan->workspace);
    if (cudaMalloc(&plan->workspace, workspace) != cudaSuccess) return 4;
    plan->workspace_size = workspace;
  }
  cutlass::Status status = plan->gemm.run(args, plan->workspace, static_cast<cudaStream_t>(stream), nullptr, pdl != 0);
  return status == cutlass::Status::kSuccess ? 0 : 2;
}
