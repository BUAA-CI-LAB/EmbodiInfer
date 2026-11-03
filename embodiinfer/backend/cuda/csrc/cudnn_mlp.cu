// cuDNN 9.20 runtime-fused block-scaled NVFP4 / FP8 MLP and Linear kernels for SM100/SM110.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cudnn.h>

#include "pdl.cuh"

namespace {

__device__ __forceinline__ uint8_t cudnn_mlp_fp4_cast2(float2 v) {
    return __nv_cvt_float2_to_fp4x2(v, __NV_E2M1, cudaRoundNearest);
}

__global__ void lookup_fp4_packed_vec8_kernel(
    const __nv_bfloat16* __restrict__ packed,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ blocked,
    const float* __restrict__ down_global,
    float u_ratio,
    int rows, int columns,
    const uint16_t* __restrict__ table)
{
    pdl_wait();
    pdl_launch_dependents();
    int row = blockIdx.y;
    int blocks_per_row = columns / 16;
    int col0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (col0 >= columns) return;
    int column_block = col0 >> 4;
    int half = (col0 >> 3) & 1;
    int64_t row_base = int64_t(row) * (2 * columns);
    uint4 g4 = *reinterpret_cast<const uint4*>(packed + row_base + col0);
    uint4 u4 = *reinterpret_cast<const uint4*>(packed + row_base + columns + col0);
    const uint16_t* g16 = reinterpret_cast<const uint16_t*>(&g4);
    const uint16_t* u16 = reinterpret_cast<const uint16_t*>(&u4);
    float prod[8];
    float maximum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float g = __bfloat162float(__ushort_as_bfloat16(__ldg(table + g16[i])));
        float u = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__ushort_as_bfloat16(u16[i])) * u_ratio));
        float p = __bfloat162float(__float2bfloat16_rn(g * u));
        prod[i] = p;
        maximum = fmaxf(maximum, fabsf(p));
    }
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
    float g_scale = *down_global;
    float raw_scale = fminf(448.0f, maximum / (6.0f * g_scale));
    uint8_t scale_code = __nv_cvt_float_to_fp8(raw_scale, __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 converted;
    converted.__x = scale_code;
    float denominator = g_scale * float(converted);
    uint32_t packed4 = 0;
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint8_t byte = cudnn_mlp_fp4_cast2(make_float2(
            denominator > 0.0f ? prod[2 * p] / denominator : 0.0f,
            denominator > 0.0f ? prod[2 * p + 1] / denominator : 0.0f));
        packed4 |= uint32_t(byte) << (8 * p);
    }
    reinterpret_cast<uint32_t*>(output)[(int64_t(row) * columns + col0) >> 3] = packed4;
    if (half == 0) {
        int column_tiles = (blocks_per_row + 3) / 4;
        int offset = ((row / 128) * column_tiles + column_block / 4) * 512
                   + (row % 32) * 16 + ((row % 128) / 32) * 4
                   + (column_block & 3);
        blocked[offset] = scale_code;
    }
}

struct DeviceContext {
    cudnnHandle_t handle = nullptr;
    cudaStream_t tune_stream = nullptr;
    cudaStream_t prefetch_stream = nullptr;
    cudaEvent_t ev_fork = nullptr;
    cudaEvent_t ev_join = nullptr;
    void* workspace = nullptr;
    size_t max_ws = 64ULL * 1024 * 1024;
};

static std::mutex g_mutex;
static std::map<int, DeviceContext> g_ctx;

static DeviceContext& get_ctx(cudaStream_t stream) {
    int dev = 0;
    cudaGetDevice(&dev);
    auto it = g_ctx.find(dev);
    if (it == g_ctx.end()) {
        DeviceContext ctx;
        cudnnCreate(&ctx.handle);
        cudaStreamCreateWithFlags(&ctx.tune_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&ctx.prefetch_stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&ctx.ev_fork, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ctx.ev_join, cudaEventDisableTiming);
        cudaMalloc(&ctx.workspace, ctx.max_ws);
        it = g_ctx.emplace(dev, ctx).first;
    }
    cudnnSetStream(it->second.handle, stream);
    return it->second;
}

__global__ void __launch_bounds__(256) hw_l2_prefetch4_kernel(
    const char* __restrict__ p0, size_t n0,
    const char* __restrict__ p1, size_t n1,
    const char* __restrict__ p2, size_t n2,
    const char* __restrict__ p3, size_t n3)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x * 128;
    for (size_t off = tid * 128; off < n0; off += stride) {
        const char* addr = p0 + off;
        asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(addr));
    }
    for (size_t off = tid * 128; off < n1; off += stride) {
        const char* addr = p1 + off;
        asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(addr));
    }
    for (size_t off = tid * 128; off < n2; off += stride) {
        const char* addr = p2 + off;
        asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(addr));
    }
    for (size_t off = tid * 128; off < n3; off += stride) {
        const char* addr = p3 + off;
        asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(addr));
    }
}

static cudnnBackendDescriptor_t make_tensor(
    int64_t uid, cudnnDataType_t dtype,
    int64_t b, int64_t r, int64_t c,
    int64_t sb, int64_t sr, int64_t sc,
    bool is_virtual, bool f8_reorder = false)
{
    cudnnBackendDescriptor_t t;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &t);
    cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype);
    int64_t dims[3] = {b, r, c};
    int64_t strides[3] = {sb, sr, sc};
    cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 3, dims);
    cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 3, strides);
    cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
    int64_t align = 16;
    cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
    cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &is_virtual);
    if (f8_reorder) {
        cudnnBackendTensorReordering_t mode = CUDNN_TENSOR_REORDERING_F8_128x4;
        cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_REORDERING_MODE, CUDNN_TYPE_TENSOR_REORDERING_MODE, 1, &mode);
    }
    cudnnBackendFinalize(t);
    return t;
}

static cudnnBackendDescriptor_t make_deq_op(
    cudnnBackendDescriptor_t x, cudnnBackendDescriptor_t s, cudnnBackendDescriptor_t y,
    int32_t b0, int32_t b1)
{
    cudnnDataType_t f32 = CUDNN_DATA_FLOAT;
    int32_t blk[2] = {b0, b1};
    cudnnBackendDescriptor_t op;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR, &op);
    cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x);
    cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &s);
    cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y);
    cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &f32);
    cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE, CUDNN_TYPE_INT32, 2, blk);
    cudnnBackendFinalize(op);
    return op;
}

static cudnnBackendDescriptor_t make_mm_op(
    cudnnBackendDescriptor_t a, cudnnBackendDescriptor_t b, cudnnBackendDescriptor_t c)
{
    cudnnDataType_t f32 = CUDNN_DATA_FLOAT;
    cudnnBackendDescriptor_t mmDesc, opMm;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_MATMUL_DESCRIPTOR, &mmDesc);
    cudnnBackendSetAttribute(mmDesc, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &f32);
    cudnnBackendFinalize(mmDesc);

    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, &opMm);
    cudnnBackendSetAttribute(opMm, CUDNN_ATTR_OPERATION_MATMUL_ADESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &a);
    cudnnBackendSetAttribute(opMm, CUDNN_ATTR_OPERATION_MATMUL_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b);
    cudnnBackendSetAttribute(opMm, CUDNN_ATTR_OPERATION_MATMUL_CDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &c);
    cudnnBackendSetAttribute(opMm, CUDNN_ATTR_OPERATION_MATMUL_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &mmDesc);
    cudnnBackendFinalize(opMm);
    return opMm;
}

static cudnnBackendDescriptor_t make_pw_op(
    cudnnPointwiseMode_t mode, cudnnDataType_t math_prec,
    cudnnBackendDescriptor_t x, cudnnBackendDescriptor_t b, cudnnBackendDescriptor_t y)
{
    cudnnBackendDescriptor_t pwDesc, opPw;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &pwDesc);
    cudnnBackendSetAttribute(pwDesc, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &mode);
    cudnnBackendSetAttribute(pwDesc, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &math_prec);
    cudnnBackendFinalize(pwDesc);

    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &opPw);
    cudnnBackendSetAttribute(opPw, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pwDesc);
    cudnnBackendSetAttribute(opPw, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x);
    if (b != nullptr) {
        cudnnBackendSetAttribute(opPw, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b);
    }
    cudnnBackendSetAttribute(opPw, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y);
    cudnnBackendFinalize(opPw);
    return opPw;
}

static cudnnBackendDescriptor_t build_plan_with_preferred_cfg(
    DeviceContext& ctx,
    std::vector<cudnnBackendDescriptor_t>& ops,
    int preferred_idx)
{
    cudnnBackendDescriptor_t opGraph;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opGraph);
    cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &ctx.handle);
    cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, ops.size(), ops.data());
    if (cudnnBackendFinalize(opGraph) != CUDNN_STATUS_SUCCESS) return nullptr;

    cudnnBackendDescriptor_t heur;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heur);
    cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opGraph);
    cudnnBackendHeurMode_t hmode = CUDNN_HEUR_MODE_A;
    cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &hmode);
    if (cudnnBackendFinalize(heur) != CUDNN_STATUS_SUCCESS) return nullptr;

    int64_t count = 0, actual = 0;
    cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr);
    if (count <= 0) return nullptr;
    std::vector<cudnnBackendDescriptor_t> cfgs(count);
    for (int64_t i = 0; i < count; ++i) cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &cfgs[i]);
    cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, count, &actual, cfgs.data());

    std::vector<int64_t> order;
    if (preferred_idx >= 0 && preferred_idx < actual) order.push_back(preferred_idx);
    for (int64_t i = 0; i < actual; ++i) {
        if (i != preferred_idx) order.push_back(i);
    }

    for (int64_t idx : order) {
        cudnnBackendDescriptor_t plan;
        cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan);
        cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &ctx.handle);
        cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cfgs[idx]);
        if (cudnnBackendFinalize(plan) != CUDNN_STATUS_SUCCESS) {
            cudnnBackendDestroyDescriptor(plan);
            continue;
        }
        int64_t ws_size = 0;
        cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, nullptr, &ws_size);
        if (ws_size < 0 || size_t(ws_size) > ctx.max_ws) {
            cudnnBackendDestroyDescriptor(plan);
            continue;
        }
        return plan;
    }
    return nullptr;
}

using DualKey = std::tuple<int, int64_t, int64_t, int64_t, int>;
static std::map<DualKey, cudnnBackendDescriptor_t> g_dual_plans;

using Fc1Key = std::tuple<int, int64_t, int64_t, int64_t>;
static std::map<Fc1Key, cudnnBackendDescriptor_t> g_fc1_plans;

using LinearFp4Key = std::tuple<int, int64_t, int64_t, int64_t, int>;
static std::map<LinearFp4Key, cudnnBackendDescriptor_t> g_linear_fp4_plans;

using LinearFp8Key = std::tuple<int, int64_t, int64_t, int64_t, int>;
static std::map<LinearFp8Key, cudnnBackendDescriptor_t> g_linear_fp8_plans;

using Fc1Fp4Key = std::tuple<int, int64_t, int64_t, int64_t>;
static std::map<Fc1Fp4Key, cudnnBackendDescriptor_t> g_fc1_fp4_plans;

}  // namespace

extern "C" int cc_cudnn_dual_geglu(
    int64_t M, int64_t K, int64_t N, int out_bits,
    void* dA, void* dAs,
    void* dBg, void* dBgs,
    void* dBu, void* dBus,
    void* dGScale, void* dUScale, void* dInvScale,
    void* dOut, void* dOutScale,
    void* raw_stream)
{
    cudaStream_t stream = (cudaStream_t)raw_stream;
    std::lock_guard<std::mutex> lock(g_mutex);
    DeviceContext& ctx = get_ctx(stream);

    int dev = 0;
    cudaGetDevice(&dev);
    DualKey key(dev, M, K, N, out_bits);

    std::vector<int64_t> uids;
    std::vector<void*> ptrs;
    if (out_bits == 4) {
        uids = {1, 2, 4, 5, 7, 8, 27, 28, 29, 30, 31};
        ptrs = {dA, dAs, dBg, dBgs, dBu, dBus, dGScale, dUScale, dInvScale, dOut, dOutScale};
    } else if (out_bits == 8) {
        uids = {1, 2, 4, 5, 7, 8, 27, 28, 29, 30};
        ptrs = {dA, dAs, dBg, dBgs, dBu, dBus, dGScale, dUScale, dInvScale, dOut};
    } else {
        uids = {1, 2, 4, 5, 7, 8, 27, 28, 30};
        ptrs = {dA, dAs, dBg, dBgs, dBu, dBus, dGScale, dUScale, dOut};
    }

    cudnnBackendDescriptor_t plan = nullptr;
    auto it = g_dual_plans.find(key);
    if (it != g_dual_plans.end()) {
        plan = it->second;
    } else {
        int64_t sr_m = ((M + 127) / 128) * 128;
        int64_t sc_k = ((K / 16 + 3) / 4) * 4;
        int64_t sc_n = ((N / 16 + 3) / 4) * 4;
        cudnnDataType_t f32 = CUDNN_DATA_FLOAT;

        auto tA = make_tensor(1, CUDNN_DATA_FP4_E2M1, 1, M, K, M * K, K, 1, false);
        auto tAs = make_tensor(2, CUDNN_DATA_FP8_E4M3, 1, sr_m, sc_k, sr_m * sc_k, sc_k, 1, false, true);
        auto tAdq = make_tensor(3, CUDNN_DATA_FLOAT, 1, M, K, M * K, K, 1, true);
        auto opDeqA = make_deq_op(tA, tAs, tAdq, 1, 16);

        auto tBg = make_tensor(4, CUDNN_DATA_FP4_E2M1, 1, K, N, K * N, 1, K, false);
        auto tBgs = make_tensor(5, CUDNN_DATA_FP8_E4M3, 1, sc_k, N, sc_k * N, 1, sc_k, false, true);
        auto tBgdq = make_tensor(6, CUDNN_DATA_FLOAT, 1, K, N, K * N, 1, K, true);
        auto opDeqBg = make_deq_op(tBg, tBgs, tBgdq, 16, 1);

        auto tBu = make_tensor(7, CUDNN_DATA_FP4_E2M1, 1, K, N, K * N, 1, K, false);
        auto tBus = make_tensor(8, CUDNN_DATA_FP8_E4M3, 1, sc_k, N, sc_k * N, 1, sc_k, false, true);
        auto tBudq = make_tensor(9, CUDNN_DATA_FLOAT, 1, K, N, K * N, 1, K, true);
        auto opDeqBu = make_deq_op(tBu, tBus, tBudq, 16, 1);

        auto tMmG = make_tensor(20, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tScaledG = make_tensor(21, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tGeluG = make_tensor(22, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tMmU = make_tensor(23, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tScaledU = make_tensor(24, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);

        auto tGScale = make_tensor(27, CUDNN_DATA_FLOAT, 1, 1, 1, 1, 1, 1, false);
        auto tUScale = make_tensor(28, CUDNN_DATA_FLOAT, 1, 1, 1, 1, 1, 1, false);

        auto opMmG = make_mm_op(tAdq, tBgdq, tMmG);
        auto opMmU = make_mm_op(tAdq, tBudq, tMmU);
        auto opMulG = make_pw_op(CUDNN_POINTWISE_MUL, f32, tMmG, tGScale, tScaledG);
        auto opGelu = make_pw_op(CUDNN_POINTWISE_GELU_FWD, f32, tScaledG, nullptr, tGeluG);
        auto opMulU = make_pw_op(CUDNN_POINTWISE_MUL, f32, tMmU, tUScale, tScaledU);

        std::vector<cudnnBackendDescriptor_t> ops = {
            opDeqA, opDeqBg, opDeqBu, opMmG, opMmU, opMulG, opGelu, opMulU
        };

        if (out_bits == 4) {
            auto tProd = make_tensor(25, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
            auto tProdScaled = make_tensor(26, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
            auto tInvScale = make_tensor(29, CUDNN_DATA_FLOAT, 1, 1, 1, 1, 1, 1, false);
            auto tOutFp4 = make_tensor(30, CUDNN_DATA_FP4_E2M1, 1, M, N, M * N, N, 1, false);
            auto tOutScale = make_tensor(31, CUDNN_DATA_FP8_E4M3, 1, sr_m, sc_n, sr_m * sc_n, sc_n, 1, false, true);

            auto opMulGU = make_pw_op(CUDNN_POINTWISE_MUL, f32, tGeluG, tScaledU, tProd);
            auto opMulInv = make_pw_op(CUDNN_POINTWISE_MUL, f32, tProd, tInvScale, tProdScaled);

            cudnnBackendDescriptor_t opQuant;
            cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR, &opQuant);
            cudnnBackendSetAttribute(opQuant, CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tProdScaled);
            cudnnBackendSetAttribute(opQuant, CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tOutFp4);
            cudnnBackendSetAttribute(opQuant, CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tOutScale);
            cudnnBackendSetAttribute(opQuant, CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &f32);
            int32_t qblk = 16;
            cudnnBackendSetAttribute(opQuant, CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE, CUDNN_TYPE_INT32, 1, &qblk);
            cudnnBackendFinalize(opQuant);

            ops.push_back(opMulGU);
            ops.push_back(opMulInv);
            ops.push_back(opQuant);
        } else if (out_bits == 8) {
            auto tProd = make_tensor(25, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
            auto tInvScale = make_tensor(29, CUDNN_DATA_FLOAT, 1, 1, 1, 1, 1, 1, false);
            auto tOutFp8 = make_tensor(30, CUDNN_DATA_FP8_E4M3, 1, M, N, M * N, N, 1, false);
            auto opMulGU = make_pw_op(CUDNN_POINTWISE_MUL, f32, tGeluG, tScaledU, tProd);
            auto opMulInv = make_pw_op(CUDNN_POINTWISE_MUL, f32, tProd, tInvScale, tOutFp8);
            ops.push_back(opMulGU);
            ops.push_back(opMulInv);
        } else {
            auto tOutBf16 = make_tensor(30, CUDNN_DATA_BFLOAT16, 1, M, N, M * N, N, 1, false);
            auto opMulGU = make_pw_op(CUDNN_POINTWISE_MUL, f32, tGeluG, tScaledU, tOutBf16);
            ops.push_back(opMulGU);
        }

        // Small M (up to 128 rows) and large M prefer different engine configs.
        int pref = (M <= 128) ? 3 : 0;
        plan = build_plan_with_preferred_cfg(ctx, ops, pref);
        if (plan == nullptr) return 1;
        g_dual_plans[key] = plan;
    }

    cudnnBackendDescriptor_t vpack;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vpack);
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, uids.size(), uids.data());
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, ptrs.size(), ptrs.data());
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ctx.workspace);
    cudnnBackendFinalize(vpack);

    cudnnStatus_t st = cudnnBackendExecute(ctx.handle, plan, vpack);
    cudnnBackendDestroyDescriptor(vpack);
    return (st == CUDNN_STATUS_SUCCESS) ? 0 : int(st);
}

extern "C" int cc_cudnn_fp8_bias_gelu(
    int64_t M, int64_t K, int64_t N,
    void* dA, void* dB, void* dScale, void* dBias, void* dInvScale, void* dOutFp8,
    void* raw_stream)
{
    cudaStream_t stream = (cudaStream_t)raw_stream;
    std::lock_guard<std::mutex> lock(g_mutex);
    DeviceContext& ctx = get_ctx(stream);

    int dev = 0;
    cudaGetDevice(&dev);
    Fc1Key key(dev, M, K, N);

    std::vector<int64_t> uids = {1, 2, 4, 6, 9, 10};
    std::vector<void*> ptrs = {dA, dB, dScale, dBias, dInvScale, dOutFp8};

    cudnnBackendDescriptor_t plan = nullptr;
    auto it = g_fc1_plans.find(key);
    if (it != g_fc1_plans.end()) {
        plan = it->second;
    } else {
        cudnnDataType_t f32 = CUDNN_DATA_FLOAT;
        auto tA = make_tensor(1, CUDNN_DATA_FP8_E4M3, 1, M, K, M * K, K, 1, false);
        auto tB = make_tensor(2, CUDNN_DATA_FP8_E4M3, 1, K, N, K * N, 1, K, false);
        auto tMm = make_tensor(3, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tScale = make_tensor(4, CUDNN_DATA_FLOAT, 1, 1, 1, 1, 1, 1, false);
        auto tScaled = make_tensor(5, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tBias = make_tensor(6, CUDNN_DATA_BFLOAT16, 1, 1, N, N, N, 1, false);
        auto tBiased = make_tensor(7, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tGelu = make_tensor(8, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tInv = make_tensor(9, CUDNN_DATA_FLOAT, 1, 1, 1, 1, 1, 1, false);
        auto tOut = make_tensor(10, CUDNN_DATA_FP8_E4M3, 1, M, N, M * N, N, 1, false);

        auto opMm = make_mm_op(tA, tB, tMm);
        auto opMul = make_pw_op(CUDNN_POINTWISE_MUL, f32, tMm, tScale, tScaled);
        auto opAdd = make_pw_op(CUDNN_POINTWISE_ADD, f32, tScaled, tBias, tBiased);
        auto opGelu = make_pw_op(CUDNN_POINTWISE_GELU_APPROX_TANH_FWD, f32, tBiased, nullptr, tGelu);
        auto opQuant = make_pw_op(CUDNN_POINTWISE_MUL, f32, tGelu, tInv, tOut);

        std::vector<cudnnBackendDescriptor_t> ops = {opMm, opMul, opAdd, opGelu, opQuant};
        plan = build_plan_with_preferred_cfg(ctx, ops, 0);
        if (plan == nullptr) return 1;
        g_fc1_plans[key] = plan;
    }

    cudnnBackendDescriptor_t vpack;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vpack);
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, uids.size(), uids.data());
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, ptrs.size(), ptrs.data());
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ctx.workspace);
    cudnnBackendFinalize(vpack);

    cudnnStatus_t st = cudnnBackendExecute(ctx.handle, plan, vpack);
    cudnnBackendDestroyDescriptor(vpack);
    return (st == CUDNN_STATUS_SUCCESS) ? 0 : int(st);
}

extern "C" int cc_cudnn_nvfp4_linear(
    int64_t M, int64_t K, int64_t N,
    void* dA, void* dAs,
    void* dB, void* dBs,
    void* dScale, void* dBias, void* dOutBf16,
    void* raw_stream)
{
    cudaStream_t stream = (cudaStream_t)raw_stream;
    std::lock_guard<std::mutex> lock(g_mutex);
    DeviceContext& ctx = get_ctx(stream);

    int dev = 0;
    cudaGetDevice(&dev);
    int has_bias = (dBias != nullptr) ? 1 : 0;
    LinearFp4Key key(dev, M, K, N, has_bias);

    std::vector<int64_t> uids;
    std::vector<void*> ptrs;
    if (has_bias) {
        uids = {1, 2, 4, 5, 21, 23, 24};
        ptrs = {dA, dAs, dB, dBs, dScale, dBias, dOutBf16};
    } else {
        uids = {1, 2, 4, 5, 21, 24};
        ptrs = {dA, dAs, dB, dBs, dScale, dOutBf16};
    }

    cudnnBackendDescriptor_t plan = nullptr;
    auto it = g_linear_fp4_plans.find(key);
    if (it != g_linear_fp4_plans.end()) {
        plan = it->second;
    } else {
        int64_t sr_m = ((M + 127) / 128) * 128;
        int64_t sc_k = ((K / 16 + 3) / 4) * 4;
        int64_t sr_n = ((N + 127) / 128) * 128;
        cudnnDataType_t f32 = CUDNN_DATA_FLOAT;

        auto tA = make_tensor(1, CUDNN_DATA_FP4_E2M1, 1, M, K, M * K, K, 1, false);
        auto tAs = make_tensor(2, CUDNN_DATA_FP8_E4M3, 1, sr_m, sc_k, sr_m * sc_k, sc_k, 1, false, true);
        auto tAdq = make_tensor(3, CUDNN_DATA_FLOAT, 1, M, K, M * K, K, 1, true);
        auto opDeqA = make_deq_op(tA, tAs, tAdq, 1, 16);

        auto tB = make_tensor(4, CUDNN_DATA_FP4_E2M1, 1, K, N, K * N, 1, K, false);
        auto tBs = make_tensor(5, CUDNN_DATA_FP8_E4M3, 1, sc_k, sr_n, sc_k * sr_n, 1, sc_k, false, true);
        auto tBdq = make_tensor(6, CUDNN_DATA_FLOAT, 1, K, N, K * N, 1, K, true);
        auto opDeqB = make_deq_op(tB, tBs, tBdq, 16, 1);

        auto tMm = make_tensor(20, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tScale = make_tensor(21, CUDNN_DATA_FLOAT, 1, 1, 1, 1, 1, 1, false);
        auto opMm = make_mm_op(tAdq, tBdq, tMm);

        std::vector<cudnnBackendDescriptor_t> ops = {opDeqA, opDeqB, opMm};
        if (has_bias) {
            auto tScaled = make_tensor(22, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
            auto tBias = make_tensor(23, CUDNN_DATA_BFLOAT16, 1, 1, N, N, N, 1, false);
            auto tOut = make_tensor(24, CUDNN_DATA_BFLOAT16, 1, M, N, M * N, N, 1, false);
            ops.push_back(make_pw_op(CUDNN_POINTWISE_MUL, f32, tMm, tScale, tScaled));
            ops.push_back(make_pw_op(CUDNN_POINTWISE_ADD, f32, tScaled, tBias, tOut));
        } else {
            auto tOut = make_tensor(24, CUDNN_DATA_BFLOAT16, 1, M, N, M * N, N, 1, false);
            ops.push_back(make_pw_op(CUDNN_POINTWISE_MUL, f32, tMm, tScale, tOut));
        }
        int pref = 0;
        if (M == 512 && K == 4352 && N == 1152) pref = 5;
        else if (M == 512) pref = 0;
        else if (M > 128 && (K == 16384 || N == 2560)) pref = 6;
        else if (M > 128) pref = 0;
        // Decode gate/up weights are L2-resident under the side-stream prefetch, which favors the
        // split-K config.
        else if (N == 8192) pref = 0;
        else pref = 0;
        plan = build_plan_with_preferred_cfg(ctx, ops, pref);
        if (plan == nullptr) return 1;
        g_linear_fp4_plans[key] = plan;
    }

    cudnnBackendDescriptor_t vpack;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vpack);
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, uids.size(), uids.data());
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, ptrs.size(), ptrs.data());
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ctx.workspace);
    cudnnBackendFinalize(vpack);

    cudnnStatus_t st = cudnnBackendExecute(ctx.handle, plan, vpack);
    cudnnBackendDestroyDescriptor(vpack);
    return (st == CUDNN_STATUS_SUCCESS) ? 0 : int(st);
}

extern "C" int cc_cudnn_fp8_linear(
    int64_t M, int64_t K, int64_t N,
    void* dA, void* dB,
    void* dScale, void* dBias, void* dOutBf16,
    void* raw_stream)
{
    cudaStream_t stream = (cudaStream_t)raw_stream;
    std::lock_guard<std::mutex> lock(g_mutex);
    DeviceContext& ctx = get_ctx(stream);

    int dev = 0;
    cudaGetDevice(&dev);
    int has_bias = (dBias != nullptr) ? 1 : 0;
    LinearFp8Key key(dev, M, K, N, has_bias);

    std::vector<int64_t> uids;
    std::vector<void*> ptrs;
    if (has_bias) {
        uids = {1, 2, 4, 6, 7};
        ptrs = {dA, dB, dScale, dBias, dOutBf16};
    } else {
        uids = {1, 2, 4, 7};
        ptrs = {dA, dB, dScale, dOutBf16};
    }

    cudnnBackendDescriptor_t plan = nullptr;
    auto it = g_linear_fp8_plans.find(key);
    if (it != g_linear_fp8_plans.end()) {
        plan = it->second;
    } else {
        cudnnDataType_t f32 = CUDNN_DATA_FLOAT;
        auto tA = make_tensor(1, CUDNN_DATA_FP8_E4M3, 1, M, K, M * K, K, 1, false);
        auto tB = make_tensor(2, CUDNN_DATA_FP8_E4M3, 1, K, N, K * N, 1, K, false);
        auto tMm = make_tensor(3, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tScale = make_tensor(4, CUDNN_DATA_FLOAT, 1, 1, 1, 1, 1, 1, false);
        auto opMm = make_mm_op(tA, tB, tMm);

        std::vector<cudnnBackendDescriptor_t> ops = {opMm};
        if (has_bias) {
            auto tScaled = make_tensor(5, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
            auto tBias = make_tensor(6, CUDNN_DATA_BFLOAT16, 1, 1, N, N, N, 1, false);
            auto tOut = make_tensor(7, CUDNN_DATA_BFLOAT16, 1, M, N, M * N, N, 1, false);
            ops.push_back(make_pw_op(CUDNN_POINTWISE_MUL, f32, tMm, tScale, tScaled));
            ops.push_back(make_pw_op(CUDNN_POINTWISE_ADD, f32, tScaled, tBias, tOut));
        } else {
            auto tOut = make_tensor(7, CUDNN_DATA_BFLOAT16, 1, M, N, M * N, N, 1, false);
            ops.push_back(make_pw_op(CUDNN_POINTWISE_MUL, f32, tMm, tScale, tOut));
        }
        int pref = 0;
        if (M == 512 && N == 3456) pref = 2;
        else if (M == 512 && K == 1152 && N == 1152) pref = 1;
        else if (M == 512 && K == 4304) pref = 3;
        plan = build_plan_with_preferred_cfg(ctx, ops, pref);
        if (plan == nullptr) return 1;
        g_linear_fp8_plans[key] = plan;
    }

    cudnnBackendDescriptor_t vpack;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vpack);
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, uids.size(), uids.data());
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, ptrs.size(), ptrs.data());
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ctx.workspace);
    cudnnBackendFinalize(vpack);

    cudnnStatus_t st = cudnnBackendExecute(ctx.handle, plan, vpack);
    cudnnBackendDestroyDescriptor(vpack);
    return (st == CUDNN_STATUS_SUCCESS) ? 0 : int(st);
}

extern "C" int cc_cudnn_fp4_bias_gelu(
    int64_t M, int64_t K, int64_t N,
    void* dA, void* dAs,
    void* dB, void* dBs,
    void* dScale, void* dBias, void* dInvScale,
    void* dOutFp4, void* dOutScale,
    void* raw_stream)
{
    cudaStream_t stream = (cudaStream_t)raw_stream;
    std::lock_guard<std::mutex> lock(g_mutex);
    DeviceContext& ctx = get_ctx(stream);

    int dev = 0;
    cudaGetDevice(&dev);
    Fc1Fp4Key key(dev, M, K, N);

    std::vector<int64_t> uids = {1, 2, 4, 5, 21, 23, 26, 28, 29};
    std::vector<void*> ptrs = {dA, dAs, dB, dBs, dScale, dBias, dInvScale, dOutFp4, dOutScale};

    cudnnBackendDescriptor_t plan = nullptr;
    auto it = g_fc1_fp4_plans.find(key);
    if (it != g_fc1_fp4_plans.end()) {
        plan = it->second;
    } else {
        int64_t sr_m = ((M + 127) / 128) * 128;
        int64_t sc_k = ((K / 16 + 3) / 4) * 4;
        int64_t sr_n = ((N + 127) / 128) * 128;
        int64_t sc_n = ((N / 16 + 3) / 4) * 4;
        cudnnDataType_t f32 = CUDNN_DATA_FLOAT;

        auto tA = make_tensor(1, CUDNN_DATA_FP4_E2M1, 1, M, K, M * K, K, 1, false);
        auto tAs = make_tensor(2, CUDNN_DATA_FP8_E4M3, 1, sr_m, sc_k, sr_m * sc_k, sc_k, 1, false, true);
        auto tAdq = make_tensor(3, CUDNN_DATA_FLOAT, 1, M, K, M * K, K, 1, true);
        auto opDeqA = make_deq_op(tA, tAs, tAdq, 1, 16);

        auto tB = make_tensor(4, CUDNN_DATA_FP4_E2M1, 1, K, N, K * N, 1, K, false);
        auto tBs = make_tensor(5, CUDNN_DATA_FP8_E4M3, 1, sc_k, sr_n, sc_k * sr_n, 1, sc_k, false, true);
        auto tBdq = make_tensor(6, CUDNN_DATA_FLOAT, 1, K, N, K * N, 1, K, true);
        auto opDeqB = make_deq_op(tB, tBs, tBdq, 16, 1);

        auto tMm = make_tensor(20, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tScale = make_tensor(21, CUDNN_DATA_FLOAT, 1, 1, 1, 1, 1, 1, false);
        auto tScaled = make_tensor(22, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tBias = make_tensor(23, CUDNN_DATA_BFLOAT16, 1, 1, N, N, N, 1, false);
        auto tBiased = make_tensor(24, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tGelu = make_tensor(25, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tInv = make_tensor(26, CUDNN_DATA_FLOAT, 1, 1, 1, 1, 1, 1, false);
        auto tPreQ = make_tensor(27, CUDNN_DATA_FLOAT, 1, M, N, M * N, N, 1, true);
        auto tOutFp4 = make_tensor(28, CUDNN_DATA_FP4_E2M1, 1, M, N, M * N, N, 1, false);
        auto tOutScale = make_tensor(29, CUDNN_DATA_FP8_E4M3, 1, sr_m, sc_n, sr_m * sc_n, sc_n, 1, false, true);

        auto opMm = make_mm_op(tAdq, tBdq, tMm);
        auto opMul = make_pw_op(CUDNN_POINTWISE_MUL, f32, tMm, tScale, tScaled);
        auto opAdd = make_pw_op(CUDNN_POINTWISE_ADD, f32, tScaled, tBias, tBiased);
        auto opGelu = make_pw_op(CUDNN_POINTWISE_GELU_APPROX_TANH_FWD, f32, tBiased, nullptr, tGelu);
        auto opMulInv = make_pw_op(CUDNN_POINTWISE_MUL, f32, tGelu, tInv, tPreQ);

        cudnnBackendDescriptor_t opQuant;
        cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR, &opQuant);
        cudnnBackendSetAttribute(opQuant, CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tPreQ);
        cudnnBackendSetAttribute(opQuant, CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tOutFp4);
        cudnnBackendSetAttribute(opQuant, CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tOutScale);
        cudnnBackendSetAttribute(opQuant, CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &f32);
        int32_t qblk = 16;
        cudnnBackendSetAttribute(opQuant, CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE, CUDNN_TYPE_INT32, 1, &qblk);
        cudnnBackendFinalize(opQuant);

        std::vector<cudnnBackendDescriptor_t> ops = {opDeqA, opDeqB, opMm, opMul, opAdd, opGelu, opMulInv, opQuant};
        plan = build_plan_with_preferred_cfg(ctx, ops, 6);
        if (plan == nullptr) return 1;
        g_fc1_fp4_plans[key] = plan;
    }

    cudnnBackendDescriptor_t vpack;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vpack);
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, uids.size(), uids.data());
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, ptrs.size(), ptrs.data());
    cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ctx.workspace);
    cudnnBackendFinalize(vpack);

    cudnnStatus_t st = cudnnBackendExecute(ctx.handle, plan, vpack);
    cudnnBackendDestroyDescriptor(vpack);
    return (st == CUDNN_STATUS_SUCCESS) ? 0 : int(st);
}

extern "C" int cc_cudnn_packed_geglu_fp4(
    int64_t M, int64_t K, int64_t N,
    void* dA, void* dAs,
    void* dB_packed, void* dBs_packed,
    void* dGScale, float u_ratio,
    void* dPackedBf16,
    void* dOutFp4, void* dOutScale,
    void* dDownGlobal, void* dTable,
    void* raw_stream)
{
    int st = cc_cudnn_nvfp4_linear(
        M, K, 2 * N, dA, dAs, dB_packed, dBs_packed, dGScale, nullptr, dPackedBf16, raw_stream);
    if (st != 0) return st;
    cudaStream_t stream = (cudaStream_t)raw_stream;
    dim3 grid(int(N / 2048), int(M));
    return int(launch_pdl(
        lookup_fp4_packed_vec8_kernel, grid, dim3(256), 0, stream,
        (const __nv_bfloat16*)dPackedBf16,
        (uint8_t*)dOutFp4,
        (uint8_t*)dOutScale,
        (const float*)dDownGlobal,
        u_ratio,
        int(M),
        int(N),
        (const uint16_t*)dTable));
}

extern "C" int cc_l2_prefetch_fork(
    const void* p0, size_t n0,
    const void* p1, size_t n1,
    const void* p2, size_t n2,
    const void* p3, size_t n3,
    int nctas, void* raw_stream)
{
    cudaStream_t main_stream = (cudaStream_t)raw_stream;
    std::lock_guard<std::mutex> lock(g_mutex);
    DeviceContext& ctx = get_ctx(main_stream);
    cudaEventRecord(ctx.ev_fork, main_stream);
    cudaStreamWaitEvent(ctx.prefetch_stream, ctx.ev_fork, 0);
    hw_l2_prefetch4_kernel<<<nctas, 256, 0, ctx.prefetch_stream>>>(
        (const char*)p0, n0, (const char*)p1, n1, (const char*)p2, n2, (const char*)p3, n3);
    return int(cudaGetLastError());
}

extern "C" int cc_l2_prefetch_join(void* raw_stream)
{
    cudaStream_t main_stream = (cudaStream_t)raw_stream;
    std::lock_guard<std::mutex> lock(g_mutex);
    DeviceContext& ctx = get_ctx(main_stream);
    cudaEventRecord(ctx.ev_join, ctx.prefetch_stream);
    cudaStreamWaitEvent(main_stream, ctx.ev_join, 0);
    return int(cudaGetLastError());
}
