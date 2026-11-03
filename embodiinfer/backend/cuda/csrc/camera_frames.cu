// Camera frames to model images: uint8 [cameras, H, W, 3] -> float32 [cameras, 3, H', W'] in [-1, 1],
// optionally resized and padded exactly like OpenPI's Pillow-based resize_with_pad.
#include <cuda_runtime.h>

#include <cstdint>

namespace {

// Each thread converts one pixel. Built without FMA contraction, so (x / 255) * 2 - 1 rounds
// every step exactly like the host processors.
__global__ void __launch_bounds__(256) normalize_frames(
    const uint8_t* __restrict__ frames, float* __restrict__ images, int pixels, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int camera = index / pixels;
    const int pixel = index - camera * pixels;
    const uint8_t* source = frames + size_t(index) * 3;
    float* target = images + size_t(camera) * 3 * pixels + pixel;
    for (int channel = 0; channel < 3; ++channel)
        target[size_t(channel) * pixels] = float(source[channel]) / 255.0f * 2.0f - 1.0f;
}

// Pillow's 8-bit resample: 22-bit fixed-point taps, half-unit rounding offset, clamp to [0, 255].
constexpr int kPrecisionBits = 22;

__device__ __forceinline__ uint8_t clip8(int sum) {
    const int value = sum >> kPrecisionBits;
    return uint8_t(value < 0 ? 0 : (value > 255 ? 255 : value));
}

// Horizontal pass of every source row: [cameras, height, width, 3] -> [cameras, height, out_width, 3].
__global__ void __launch_bounds__(256) resize_rows(
    const uint8_t* __restrict__ frames, uint8_t* __restrict__ rows, const int* __restrict__ bounds,
    const int* __restrict__ weights, int taps, int height, int width, int out_width, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int x = index % out_width;
    const int row = index / out_width;  // camera * height + y
    const int first = bounds[2 * x], used = bounds[2 * x + 1];
    const int* tap = weights + x * taps;
    const uint8_t* source = frames + (size_t(row) * width + first) * 3;
    int sum0 = 1 << (kPrecisionBits - 1), sum1 = sum0, sum2 = sum0;
    for (int i = 0; i < used; ++i) {
        sum0 += int(source[3 * i + 0]) * tap[i];
        sum1 += int(source[3 * i + 1]) * tap[i];
        sum2 += int(source[3 * i + 2]) * tap[i];
    }
    uint8_t* target = rows + size_t(index) * 3;
    target[0] = clip8(sum0);
    target[1] = clip8(sum1);
    target[2] = clip8(sum2);
}

// Vertical pass into a padded canvas, then (x / 255) * 2 - 1 as float32 [cameras, 3, H, W].
__global__ void __launch_bounds__(256) resize_columns(
    const uint8_t* __restrict__ rows, float* __restrict__ images, const int* __restrict__ bounds,
    const int* __restrict__ weights, int taps, int height, int out_height, int out_width, int canvas_height,
    int canvas_width, int top, int left, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int pixels = canvas_height * canvas_width;
    const int camera = index / pixels;
    const int pixel = index - camera * pixels;
    const int y = pixel / canvas_width - top, x = pixel % canvas_width - left;
    int value0 = 0, value1 = 0, value2 = 0;
    if (y >= 0 && y < out_height && x >= 0 && x < out_width) {
        const int first = bounds[2 * y], used = bounds[2 * y + 1];
        const int* tap = weights + y * taps;
        const uint8_t* source = rows + ((size_t(camera) * height + first) * out_width + x) * 3;
        const size_t stride = size_t(out_width) * 3;
        int sum0 = 1 << (kPrecisionBits - 1), sum1 = sum0, sum2 = sum0;
        for (int i = 0; i < used; ++i) {
            sum0 += int(source[i * stride + 0]) * tap[i];
            sum1 += int(source[i * stride + 1]) * tap[i];
            sum2 += int(source[i * stride + 2]) * tap[i];
        }
        value0 = clip8(sum0);
        value1 = clip8(sum1);
        value2 = clip8(sum2);
    }
    float* target = images + size_t(camera) * 3 * pixels + pixel;
    target[0] = float(value0) / 255.0f * 2.0f - 1.0f;
    target[size_t(pixels)] = float(value1) / 255.0f * 2.0f - 1.0f;
    target[2 * size_t(pixels)] = float(value2) / 255.0f * 2.0f - 1.0f;
}

}  // namespace

extern "C" int cc_resize_frames(
    const void* frames, void* rows, void* images, const void* row_bounds, const void* row_weights, int row_taps,
    const void* column_bounds, const void* column_weights, int column_taps, int cameras, int height, int width,
    int out_height, int out_width, int canvas_height, int canvas_width, int top, int left, void* stream) {
    if (cameras <= 0 || height <= 0 || width <= 0 || out_height <= 0 || out_width <= 0) {
        return int(cudaErrorInvalidValue);
    }
    const auto launch = static_cast<cudaStream_t>(stream);
    const int row_count = cameras * height * out_width;
    resize_rows<<<(row_count + 255) / 256, 256, 0, launch>>>(
        static_cast<const uint8_t*>(frames), static_cast<uint8_t*>(rows), static_cast<const int*>(row_bounds),
        static_cast<const int*>(row_weights), row_taps, height, width, out_width, row_count);
    const int column_count = cameras * canvas_height * canvas_width;
    resize_columns<<<(column_count + 255) / 256, 256, 0, launch>>>(
        static_cast<const uint8_t*>(rows), static_cast<float*>(images), static_cast<const int*>(column_bounds),
        static_cast<const int*>(column_weights), column_taps, height, out_height, out_width, canvas_height,
        canvas_width, top, left, column_count);
    return int(cudaGetLastError());
}

extern "C" int cc_normalize_frames(const void* frames, void* images, int cameras, int pixels, void* stream) {
    if (cameras <= 0 || pixels <= 0) return int(cudaErrorInvalidValue);
    const int count = cameras * pixels;
    normalize_frames<<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t*>(frames), static_cast<float*>(images), pixels, count);
    return int(cudaGetLastError());
}
