// BF16 cuBLASLt plans; FP32 accumulation and explicit caller stream.
#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <memory>
#include <vector>

struct Plan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    std::vector<cublasLtMatmulHeuristicResult_t> algorithms;

    ~Plan() {
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (operation) cublasLtMatmulDescDestroy(operation);
        if (handle) cublasLtDestroy(handle);
    }
};

extern "C" size_t ei_lt_version() { return cublasLtGetVersion(); }

extern "C" int ei_lt_create(int m, int n, int k, size_t workspace,
                            int requested, void** result) {
    *result = nullptr;
    if (m <= 0 || n <= 0 || k <= 0 || requested <= 0 || requested > 64) return -1;
    auto plan = std::make_unique<Plan>();
    cublasStatus_t status;
#define CHECK(call) do { status = (call); if (status != CUBLAS_STATUS_SUCCESS) return status; } while (0)
    CHECK(cublasLtCreate(&plan->handle));
    CHECK(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t trans = CUBLAS_OP_T;
    CHECK(cublasLtMatmulDescSetAttribute(plan->operation,
        CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    // Row-major X[M,K] @ W[N,K]^T equals column-major W^T[N,K] @ X[K,M].
    CHECK(cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_16BF, k, n, k));
    CHECK(cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_16BF, k, m, k));
    CHECK(cublasLtMatrixLayoutCreate(&plan->c, CUDA_R_16BF, n, m, n));
    cublasLtMatmulPreference_t preference = nullptr;
    CHECK(cublasLtMatmulPreferenceCreate(&preference));
    status = cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulPreferenceDestroy(preference);
        return status;
    }
    plan->algorithms.resize(requested);
    int count = 0;
    status = cublasLtMatmulAlgoGetHeuristic(plan->handle, plan->operation,
        plan->a, plan->b, plan->c, plan->c, preference, requested,
        plan->algorithms.data(), &count);
    cublasLtMatmulPreferenceDestroy(preference);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    plan->algorithms.resize(count);
    *result = plan.release();
    return 0;
#undef CHECK
}

extern "C" void ei_lt_destroy(void* pointer) { delete static_cast<Plan*>(pointer); }
extern "C" int ei_lt_count(void* pointer) {
    return static_cast<Plan*>(pointer)->algorithms.size();
}

extern "C" int ei_lt_info(void* pointer, int index, int* values,
                          size_t* workspace, float* waves) {
    auto* plan = static_cast<Plan*>(pointer);
    if (index < 0 || index >= int(plan->algorithms.size())) return -1;
    const auto& row = plan->algorithms[index];
    const cublasLtMatmulAlgoConfigAttributes_t keys[] = {
        CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID,
        CUBLASLT_ALGO_CONFIG_STAGES_ID, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION};
    for (int i = 0; i < 7; ++i) {
        size_t written = 0;
        auto status = cublasLtMatmulAlgoConfigGetAttribute(&row.algo, keys[i],
            &values[i], sizeof(int), &written);
        if (status != CUBLAS_STATUS_SUCCESS) return status;
    }
    *workspace = row.workspaceSize;
    *waves = row.wavesCount;
    return row.state;
}

extern "C" int ei_lt_run(void* pointer, int index, const void* weight,
                         const void* inputs, void* output, void* workspace,
                         size_t bytes, void* stream) {
    auto* plan = static_cast<Plan*>(pointer);
    if (index < 0 || index >= int(plan->algorithms.size())) return -1;
    const auto& row = plan->algorithms[index];
    if (bytes < row.workspaceSize) return -2;
    const float alpha = 1.0f, beta = 0.0f;
    return cublasLtMatmul(plan->handle, plan->operation, &alpha,
        weight, plan->a, inputs, plan->b, &beta, output, plan->c,
        output, plan->c, &row.algo, workspace, bytes,
        static_cast<cudaStream_t>(stream));
}
