// Programmatic dependent launch: let a kernel launch while its predecessor drains.
#pragma once

#include <cuda_runtime.h>

// Blocks until the preceding grid in the stream has completed and its writes are visible.
// A no-op when the kernel was launched without programmatic serialization.
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// Lets the next grid in the stream launch early; it still waits for this grid in pdl_wait().
__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Kernels launched through this helper must call pdl_wait() before reading anything a
// preceding kernel wrote.
template <typename... Params, typename... Args>
static cudaError_t launch_pdl(
    void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args&&... args) {
    static int major = -1;
    if (major < 0) {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    }
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attribute;
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = major >= 9 ? 1 : 0;
    return cudaLaunchKernelEx(&config, kernel, static_cast<Params>(args)...);
}
