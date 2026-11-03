// NVFP4 projection for a few hundred tokens on Blackwell (SM100/SM110), computed swap-AB: the weights
// are the MMA M side in 2-SM 256-row tiles and the tokens are N in 192-wide tiles. Each weight tile
// then serves every token, where 128-row token tiles re-read the whole weight matrix for a short
// last tile (526 tokens). D is stored column-major, i.e. the row-major [tokens, out] activation, and
// D = bf16(alpha * acc) with no split-K, so the result matches a cuDNN NVFP4 matmul bit for bit.
// Launches are programmatic; without this CUTLASS compiles griddepcontrol.wait out of the kernel.
#define CUTLASS_ENABLE_GDC_FOR_SM100 1
#include <map>
#include <mutex>
#include <tuple>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace embodiinfer_cutlass_projection {
using namespace cute;

using ElementIn = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using MmaTile = Shape<_256, _192, _256>;
using Cluster = Shape<_2, _1, _1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, cutlass::layout::ColumnMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8, cutlass::epilogue::TmaWarpSpecialized2Sm>::CollectiveOp;
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

}  // namespace embodiinfer_cutlass_projection

// out[tokens, n_out] (row-major BF16) = bf16(alpha * x[tokens, K] w[n_out, K]^T) for NVFP4 x, w with
// 128x4 interleaved E4M3 block scales (x padded to 128-row tiles).
extern "C" int cc_cutlass_swap_linear(int tokens, int n_out, int K, const void* x, const void* x_scale,
                                      const void* w, const void* w_scale, const void* alpha, void* out, int pdl,
                                      void* stream) {
  using namespace embodiinfer_cutlass_projection;
  int device = 0;
  cudaGetDevice(&device);
  std::lock_guard<std::mutex> lock(plan_mutex);
  Plan*& plan = plans[std::make_tuple(device, tokens, n_out, K)];
  if (plan == nullptr) plan = new Plan{};
  auto stride_A = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {n_out, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {tokens, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {n_out, tokens, 1});
  auto layout_SFA = SfConfig::tile_atom_to_shape_SFA(make_shape(n_out, tokens, K, 1));
  auto layout_SFB = SfConfig::tile_atom_to_shape_SFB(make_shape(n_out, tokens, K, 1));
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {n_out, tokens, K, 1},
      {static_cast<cutlass::float_e2m1_t const*>(w), stride_A, static_cast<cutlass::float_e2m1_t const*>(x), stride_B,
       static_cast<cutlass::float_ue4m3_t const*>(w_scale), layout_SFA,
       static_cast<cutlass::float_ue4m3_t const*>(x_scale), layout_SFB},
      {{}, nullptr, stride_D, static_cast<cutlass::bfloat16_t*>(out), stride_D}};
  args.epilogue.thread.alpha_ptr = static_cast<float const*>(alpha);
  args.epilogue.thread.beta = 0.0f;
  if (plan->gemm.can_implement(args) != cutlass::Status::kSuccess) return 1;
  size_t workspace = Gemm::get_workspace_size(args);
  if (workspace > plan->workspace_size) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &capture);
    if (capture != cudaStreamCaptureStatusNone) return 3;  // warm this shape before graph capture
    if (plan->workspace) cudaFree(plan->workspace);
    if (cudaMalloc(&plan->workspace, workspace) != cudaSuccess) return 4;
    plan->workspace_size = workspace;
  }
  cutlass::Status status = plan->gemm.run(args, plan->workspace, static_cast<cudaStream_t>(stream), nullptr, pdl != 0);
  return status == cutlass::Status::kSuccess ? 0 : 2;
}
