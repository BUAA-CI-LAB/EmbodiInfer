// L2 prefetch of upcoming read-only segments (TMA bulk on Hopper and later, line hints on Ampere/Ada),
// and discard of dead ones.
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kMaxSegments = 16;
// 64 KiB requests keep the TMA queue full; many small hints would be dropped.
constexpr unsigned long long kChunk = 65536;

struct SegmentList {
    const char* ptr[kMaxSegments];
    unsigned long long bytes[kMaxSegments];
    int count;
};

__global__ void __launch_bounds__(128) l2_bulk_prefetch(SegmentList list) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const long long threads = (long long)gridDim.x * blockDim.x;
    const long long thread = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long chunk = 0;
    for (int s = 0; s < list.count; ++s) {
        const char* base = list.ptr[s];
        // Bulk requests must be 16-byte multiples; a skipped tail is only a missed hint.
        const unsigned long long bytes = list.bytes[s] & ~15ull;
        for (unsigned long long offset = 0; offset < bytes; offset += kChunk, ++chunk) {
            if (chunk % threads != thread) continue;
            const unsigned long long rest = bytes - offset;
            const uint32_t size = uint32_t(rest < kChunk ? rest : kChunk);
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + offset), "r"(size) : "memory");
        }
    }
#endif
}

// Ampere/Ada have no bulk prefetch, so each thread hints whole 128-byte lines. A few CTAs keep the hint
// rate below what the memory system drops; with many more, less of the segment reaches L2.
__global__ void __launch_bounds__(256) l2_line_prefetch(SegmentList list) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned long long threads = (unsigned long long)gridDim.x * blockDim.x;
    const unsigned long long thread = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (int s = 0; s < list.count; ++s) {
        const unsigned long long first = reinterpret_cast<unsigned long long>(list.ptr[s]) & ~127ull;
        const unsigned long long end = reinterpret_cast<unsigned long long>(list.ptr[s]) + list.bytes[s];
        for (unsigned long long line = first + thread * 128; line < end; line += threads * 128)
            asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(line) : "memory");
    }
#endif
}

// Only whole 128-byte lines inside a segment are dropped, so neighbouring allocations are untouched.
__global__ void __launch_bounds__(256) l2_discard(SegmentList list) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned long long threads = (unsigned long long)gridDim.x * blockDim.x;
    const unsigned long long thread = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (int s = 0; s < list.count; ++s) {
        const unsigned long long base = reinterpret_cast<unsigned long long>(list.ptr[s]);
        const unsigned long long first = (base + 127) & ~127ull;
        const unsigned long long end = (base + list.bytes[s]) & ~127ull;
        for (unsigned long long line = first + thread * 128; line < end; line += threads * 128)
            asm volatile("discard.global.L2 [%0], 128;" ::"l"(line) : "memory");
    }
#endif
}

SegmentList pack(const unsigned long long* pointers, const unsigned long long* sizes, int count) {
    SegmentList list{};
    list.count = count;
    for (int i = 0; i < count; ++i) {
        list.ptr[i] = reinterpret_cast<const char*>(pointers[i]);
        list.bytes[i] = sizes[i];
    }
    return list;
}

}  // namespace

extern "C" int cc_l2_bulk_prefetch(
    const unsigned long long* pointers, const unsigned long long* sizes, int count, int ctas, void* stream) {
    if (count <= 0) return 0;
    if (count > kMaxSegments || ctas <= 0) return int(cudaErrorInvalidValue);
    for (int i = 0; i < count; ++i) {
        if (pointers[i] % 16 != 0) return int(cudaErrorMisalignedAddress);
    }
    int device = 0, major = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status == cudaSuccess) status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (status != cudaSuccess) return int(status);
    if (major >= 9)
        l2_bulk_prefetch<<<ctas, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(pack(pointers, sizes, count));
    else
        l2_line_prefetch<<<ctas, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(pack(pointers, sizes, count));
    return int(cudaGetLastError());
}

// Drop dead segments from L2 without writing them back; their contents become undefined.
extern "C" int cc_l2_discard(
    const unsigned long long* pointers, const unsigned long long* sizes, int count, int ctas, void* stream) {
    if (count <= 0) return 0;
    if (count > kMaxSegments || ctas <= 0) return int(cudaErrorInvalidValue);
    l2_discard<<<ctas, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(pack(pointers, sizes, count));
    return int(cudaGetLastError());
}
