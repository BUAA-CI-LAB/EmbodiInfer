// Per-instance cuBLAS TF32 math mode; never changes Torch's global precision.
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

extern "C" int ei_qk_create(void** result) {
    *result = nullptr;
    cublasHandle_t handle = nullptr;
    auto status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    status = cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) { cublasDestroy(handle); return status; }
    *result = handle;
    return 0;
}
extern "C" int ei_qk_destroy(void* handle) {
    return cublasDestroy(static_cast<cublasHandle_t>(handle));
}
extern "C" int ei_qk_run(void* handle, const void* q, const void* k, void* scores,
                          int tokens, int heads, int width, void* workspace,
                          size_t workspace_bytes, void* stream) {
    if (tokens <= 0 || heads <= 0 || width <= 0) return -1;
    auto h = static_cast<cublasHandle_t>(handle);
    auto status = cublasSetStream(h, static_cast<cudaStream_t>(stream));
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    // SetStream resets the workspace; bind the graph/stream-owned storage next.
    status = cublasSetWorkspace(h, workspace, workspace_bytes);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    const float alpha = 1.f, beta = 0.f;
    return cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N,
        tokens, tokens, width, &alpha,
        static_cast<const float*>(k), width, 0,
        static_cast<const float*>(q), width, static_cast<long long>(tokens)*width,
        &beta, static_cast<float*>(scores), tokens,
        static_cast<long long>(tokens)*tokens, heads);
}
