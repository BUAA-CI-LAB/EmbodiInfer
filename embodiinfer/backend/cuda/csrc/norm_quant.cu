// Original quantizer definitions preserve FP8/NVFP4 conversions and padded layout.
#include "quantize.cu"
#include "pdl.cuh"

__device__ __forceinline__ float normalized_bf16(
    const __nv_bfloat16* input, const float* mean, const float* scale,
    const float* modulation, int64_t index, int columns, int tokens) {
    float inverse = rsqrtf(__fadd_rn(mean[index / columns], 1.0e-6f));
    float value = __fmul_rn(__bfloat162float(input[index]), inverse);
    int column = index % columns;
    if (modulation != nullptr) {
        int batch = index / (int64_t(tokens) * columns);
        int64_t base = int64_t(batch) * 3 * columns;
        value = __fmul_rn(value, __fadd_rn(1.0f, modulation[base + column]));
        value = __fadd_rn(value, modulation[base + columns + column]);
    } else {
        value = __fmul_rn(value, __fadd_rn(1.0f, scale[column]));
    }
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ void write_norm_gate(
    const float* modulation, __nv_bfloat16* gate, int64_t index,
    int columns, int tokens) {
    if (gate != nullptr && (index / columns) % tokens == 0) {
        int batch = index / (int64_t(tokens) * columns);
        int column = index % columns;
        gate[int64_t(batch) * columns + column] = __float2bfloat16_rn(
            modulation[int64_t(batch) * 3 * columns + 2 * columns + column]);
    }
}

__global__ void norm_fp8(
    const __nv_bfloat16* input, const float* mean, const float* scale,
    const float* modulation, uint8_t* output, __nv_bfloat16* gate,
    const float* global, int64_t count, int columns, int tokens) {
    int64_t index = (int64_t(blockIdx.x) * THREADS + threadIdx.x) * 2;
    if (index >= count) return;
    float first = normalized_bf16(input, mean, scale, modulation, index, columns, tokens);
    float second = index + 1 < count
        ? normalized_bf16(input, mean, scale, modulation, index + 1, columns, tokens) : 0.0f;
    uint16_t packed = __nv_cvt_float2_to_fp8x2(
        make_float2(first / *global, second / *global), __NV_SATFINITE, __NV_E4M3);
    if (index + 1 < count) reinterpret_cast<uint16_t*>(output)[index / 2] = packed;
    else output[index] = packed & 255;
    write_norm_gate(modulation, gate, index, columns, tokens);
    if (index + 1 < count) write_norm_gate(modulation, gate, index + 1, columns, tokens);
}

__global__ void norm_fp4(
    const __nv_bfloat16* input, const float* mean, const float* scale,
    const float* modulation, uint8_t* output, uint8_t* scales,
    uint8_t* blocked, __nv_bfloat16* gate, const float* global,
    int rows, int columns, int tokens) {
    int row = blockIdx.y;
    int column_block = blockIdx.x * (THREADS / 8) + threadIdx.x / 8;
    int blocks_per_row = columns / 16;
    if (column_block >= blocks_per_row) return;
    int block = row * blocks_per_row + column_block;
    int lane = threadIdx.x & 7;
    int64_t index = int64_t(row) * columns + column_block * 16 + lane * 2;
    float first = row < rows
        ? normalized_bf16(input, mean, scale, modulation, index, columns, tokens) : 0.0f;
    float second = row < rows
        ? normalized_bf16(input, mean, scale, modulation, index + 1, columns, tokens) : 0.0f;
    float maximum = fmaxf(fabsf(first), fabsf(second));
    for (int delta = 4; delta; delta /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta, 8));
    float raw_scale = fminf(448.0f, maximum / (6.0f * *global));
    uint8_t scale_code = __nv_cvt_float_to_fp8(raw_scale, __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 converted;
    converted.__x = scale_code;
    float denominator = *global * float(converted);
    uint8_t packed = cc_fp4_cast(
        make_float2(denominator > 0.0f ? first / denominator : 0.0f,
                    denominator > 0.0f ? second / denominator : 0.0f));
    if (row < rows) {
        output[index / 2] = packed;
        write_norm_gate(modulation, gate, index, columns, tokens);
        write_norm_gate(modulation, gate, index + 1, columns, tokens);
    }
    if (lane == 0) {
        int column_tiles = (blocks_per_row + 3) / 4;
        int offset = ((row / 128) * column_tiles + column_block / 4) * 512
                   + (row % 32) * 16 + ((row % 128) / 32) * 4 + column_block % 4;
        blocked[offset] = scale_code;
        if (row < rows) scales[block] = scale_code;
    }
}

extern "C" int cc_norm_quant(
    const void* input, const void* mean, const void* scale, const void* modulation,
    void* output, void* scales, void* blocked, void* gate, const void* global,
    int rows, int columns, int tokens, int bits, void* stream) {
    if (bits == 8) {
        int64_t count = int64_t(rows) * columns;
        norm_fp8<<<(count + THREADS * 2 - 1) / (THREADS * 2), THREADS, 0,
                   (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)input, (const float*)mean, (const float*)scale,
            (const float*)modulation, (uint8_t*)output, (__nv_bfloat16*)gate,
            (const float*)global, count, columns, tokens);
    } else {
            norm_fp4<<<dim3((columns / 16 + THREADS / 8 - 1) / (THREADS / 8), rows), THREADS, 0,
                   (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)input, (const float*)mean, (const float*)scale,
            (const float*)modulation, (uint8_t*)output, (uint8_t*)scales,
            (uint8_t*)blocked, (__nv_bfloat16*)gate, (const float*)global,
            rows, columns, tokens);
    }
    return int(cudaGetLastError());
}


__global__ void norm_fp8_reduce(
    const __nv_bfloat16* inputs, const __nv_bfloat16* update,
    const __nv_bfloat16* residual_gate, const float* scale, const float* modulation,
    __nv_bfloat16* residual, uint8_t* output, __nv_bfloat16* gate,
    const float* global, int columns, int tokens) {
    __shared__ float partials[THREADS];
    int row = blockIdx.x;
    int batch = row / tokens;
    bool write_gate = (gate != nullptr) && (row == batch * tokens);
    int64_t base = int64_t(row) * columns;
    int64_t gate_base = int64_t(batch) * columns;
    int64_t mod_base = int64_t(batch) * 3 * columns;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < columns; col += THREADS) {
        float val = __bfloat162float(inputs[base + col]);
        if (update != nullptr) {
            float delta = __bfloat162float(update[base + col]);
            if (residual_gate != nullptr) {
                float factor = __bfloat162float(residual_gate[gate_base + col]);
                delta = __bfloat162float(__float2bfloat16_rn(__fmul_rn(delta, factor)));
            }
            val = __bfloat162float(__float2bfloat16_rn(__fadd_rn(val, delta)));
        }
        sum = __fadd_rn(sum, __fmul_rn(val, val));
        if (residual != nullptr) residual[base + col] = __float2bfloat16_rn(val);
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            partials[threadIdx.x] = __fadd_rn(partials[threadIdx.x], partials[threadIdx.x + stride]);
        __syncthreads();
    }
    float inverse = rsqrtf(__fadd_rn(__fmul_rn(partials[0], 1.0f / columns), 1.0e-6f));
    float inv_global = 1.0f / (*global);
    const __nv_bfloat16* src = (residual != nullptr) ? residual : inputs;
    for (int col2 = threadIdx.x * 2; col2 < columns; col2 += THREADS * 2) {
        int64_t idx = base + col2;
        float v0 = __fmul_rn(__bfloat162float(src[idx]), inverse);
        float v1 = __fmul_rn(__bfloat162float(src[idx + 1]), inverse);
        if (modulation != nullptr) {
            v0 = __fadd_rn(__fmul_rn(v0, __fadd_rn(1.0f, modulation[mod_base + col2])), modulation[mod_base + columns + col2]);
            v1 = __fadd_rn(__fmul_rn(v1, __fadd_rn(1.0f, modulation[mod_base + col2 + 1])), modulation[mod_base + columns + col2 + 1]);
        } else {
            v0 = __fmul_rn(v0, __fadd_rn(1.0f, scale[col2]));
            v1 = __fmul_rn(v1, __fadd_rn(1.0f, scale[col2 + 1]));
        }
        v0 = __bfloat162float(__float2bfloat16_rn(v0));
        v1 = __bfloat162float(__float2bfloat16_rn(v1));
        uint16_t packed = __nv_cvt_float2_to_fp8x2(
            make_float2(v0 * inv_global, v1 * inv_global), __NV_SATFINITE, __NV_E4M3);
        reinterpret_cast<uint16_t*>(output)[idx / 2] = packed;
        if (write_gate) {
            gate[gate_base + col2] = __float2bfloat16_rn(modulation[mod_base + 2 * columns + col2]);
            gate[gate_base + col2 + 1] = __float2bfloat16_rn(modulation[mod_base + 2 * columns + col2 + 1]);
        }
    }
}

__global__ void norm_fp4_reduce(
    const __nv_bfloat16* inputs, const __nv_bfloat16* update,
    const __nv_bfloat16* residual_gate, const float* scale, const float* modulation,
    __nv_bfloat16* residual, uint8_t* output, uint8_t* scales,
    uint8_t* blocked, __nv_bfloat16* gate, const float* global,
    int rows, int columns, int tokens) {
    __shared__ float partials[THREADS];
    int row = blockIdx.x;
    int batch = row / tokens;
    bool write_gate = (gate != nullptr) && (row == batch * tokens);
    int64_t base = int64_t(row) * columns;
    int64_t gate_base = int64_t(batch) * columns;
    int64_t mod_base = int64_t(batch) * 3 * columns;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < columns; col += THREADS) {
        float val = __bfloat162float(inputs[base + col]);
        if (update != nullptr) {
            float delta = __bfloat162float(update[base + col]);
            if (residual_gate != nullptr) {
                float factor = __bfloat162float(residual_gate[gate_base + col]);
                delta = __bfloat162float(__float2bfloat16_rn(__fmul_rn(delta, factor)));
            }
            val = __bfloat162float(__float2bfloat16_rn(__fadd_rn(val, delta)));
        }
        sum = __fadd_rn(sum, __fmul_rn(val, val));
        if (residual != nullptr) residual[base + col] = __float2bfloat16_rn(val);
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            partials[threadIdx.x] = __fadd_rn(partials[threadIdx.x], partials[threadIdx.x + stride]);
        __syncthreads();
    }
    float inverse = rsqrtf(__fadd_rn(__fmul_rn(partials[0], 1.0f / columns), 1.0e-6f));
    float g_scale = *global;
    const __nv_bfloat16* src = (residual != nullptr) ? residual : inputs;
    int blocks_per_row = columns / 16;
    int lane = threadIdx.x & 7;
    for (int column_block = threadIdx.x / 8; column_block < blocks_per_row; column_block += THREADS / 8) {
        int block = row * blocks_per_row + column_block;
        int col = column_block * 16 + lane * 2;
        int64_t idx = base + col;
        float first = __fmul_rn(__bfloat162float(src[idx]), inverse);
        float second = __fmul_rn(__bfloat162float(src[idx + 1]), inverse);
        if (modulation != nullptr) {
            first = __fadd_rn(__fmul_rn(first, __fadd_rn(1.0f, modulation[mod_base + col])), modulation[mod_base + columns + col]);
            second = __fadd_rn(__fmul_rn(second, __fadd_rn(1.0f, modulation[mod_base + col + 1])), modulation[mod_base + columns + col + 1]);
        } else {
            first = __fmul_rn(first, __fadd_rn(1.0f, scale[col]));
            second = __fmul_rn(second, __fadd_rn(1.0f, scale[col + 1]));
        }
        first = __bfloat162float(__float2bfloat16_rn(first));
        second = __bfloat162float(__float2bfloat16_rn(second));
        float maximum = fmaxf(fabsf(first), fabsf(second));
        for (int delta = 4; delta; delta /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta, 8));
        float raw_scale = fminf(448.0f, maximum / (6.0f * g_scale));
        uint8_t scale_code = __nv_cvt_float_to_fp8(raw_scale, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_e4m3 converted;
        converted.__x = scale_code;
        float denominator = g_scale * float(converted);
        uint8_t packed = cc_fp4_cast(
            make_float2(denominator > 0.0f ? first / denominator : 0.0f,
                        denominator > 0.0f ? second / denominator : 0.0f));
        output[idx / 2] = packed;
        if (write_gate) {
            gate[gate_base + col] = __float2bfloat16_rn(modulation[mod_base + 2 * columns + col]);
            gate[gate_base + col + 1] = __float2bfloat16_rn(modulation[mod_base + 2 * columns + col + 1]);
        }
        if (lane == 0) {
            int column_tiles = (blocks_per_row + 3) / 4;
            int offset = ((row / 128) * column_tiles + column_block / 4) * 512
                       + (row % 32) * 16 + ((row % 128) / 32) * 4 + column_block % 4;
            blocked[offset] = scale_code;
            scales[block] = scale_code;
        }
    }
}


__global__ void norm_fp8_reduce_vec8(
    const __nv_bfloat16* __restrict__ inputs, const __nv_bfloat16* __restrict__ update,
    const __nv_bfloat16* __restrict__ residual_gate, const float* __restrict__ scale,
    const float* __restrict__ modulation, __nv_bfloat16* __restrict__ residual,
    uint8_t* __restrict__ output, __nv_bfloat16* __restrict__ gate,
    const float* __restrict__ global, int columns, int tokens) {
    __shared__ float warp_sums[8];
    pdl_wait();
    pdl_launch_dependents();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int batch = row / tokens;
    bool write_gate = (gate != nullptr) && (row == batch * tokens);
    int64_t base = int64_t(row) * columns;
    int64_t gate_base = int64_t(batch) * columns;
    int64_t mod_base = int64_t(batch) * 3 * columns;
    int col = tid * 8;
    float vals[8];
    float sum = 0.0f;
    if (col < columns) {
        uint4 in_u4 = *reinterpret_cast<const uint4*>(inputs + base + col);
        const __nv_bfloat162* in_b2 = reinterpret_cast<const __nv_bfloat162*>(&in_u4);
        if (update != nullptr) {
            uint4 up_u4 = *reinterpret_cast<const uint4*>(update + base + col);
            const __nv_bfloat162* up_b2 = reinterpret_cast<const __nv_bfloat162*>(&up_u4);
            uint4 rg_u4;
            const __nv_bfloat162* rg_b2 = nullptr;
            if (residual_gate != nullptr) {
                rg_u4 = *reinterpret_cast<const uint4*>(residual_gate + gate_base + col);
                rg_b2 = reinterpret_cast<const __nv_bfloat162*>(&rg_u4);
            }
            uint4 res_u4;
            __nv_bfloat162* res_b2 = reinterpret_cast<__nv_bfloat162*>(&res_u4);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 v = __bfloat1622float2(in_b2[i]);
                float2 d = __bfloat1622float2(up_b2[i]);
                if (rg_b2 != nullptr) {
                    float2 f = __bfloat1622float2(rg_b2[i]);
                    d.x = __bfloat162float(__float2bfloat16_rn(__fmul_rn(d.x, f.x)));
                    d.y = __bfloat162float(__float2bfloat16_rn(__fmul_rn(d.y, f.y)));
                }
                __nv_bfloat16 r0 = __float2bfloat16_rn(__fadd_rn(v.x, d.x));
                __nv_bfloat16 r1 = __float2bfloat16_rn(__fadd_rn(v.y, d.y));
                res_b2[i] = __halves2bfloat162(r0, r1);
                vals[i * 2] = __bfloat162float(r0);
                vals[i * 2 + 1] = __bfloat162float(r1);
                sum = __fadd_rn(sum, __fmul_rn(vals[i * 2], vals[i * 2]));
                sum = __fadd_rn(sum, __fmul_rn(vals[i * 2 + 1], vals[i * 2 + 1]));
            }
            if (residual != nullptr) {
                *reinterpret_cast<uint4*>(residual + base + col) = res_u4;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 v = __bfloat1622float2(in_b2[i]);
                vals[i * 2] = v.x;
                vals[i * 2 + 1] = v.y;
                sum = __fadd_rn(sum, __fmul_rn(v.x, v.x));
                sum = __fadd_rn(sum, __fmul_rn(v.y, v.y));
            }
            if (residual != nullptr) {
                *reinterpret_cast<uint4*>(residual + base + col) = in_u4;
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum;
    __syncthreads();
    if (tid < 8) {
        float wsum = (tid < (blockDim.x >> 5)) ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            wsum = __fadd_rn(wsum, __shfl_xor_sync(0xff, wsum, offset));
        if (tid == 0) warp_sums[0] = rsqrtf(__fadd_rn(__fmul_rn(wsum, 1.0f / columns), 1.0e-6f));
    }
    __syncthreads();
    float inverse = warp_sums[0];
    float inv_global = 1.0f / (*global);
    if (col < columns) {
        uint64_t out_pack = 0;
        uint16_t* out_u16 = reinterpret_cast<uint16_t*>(&out_pack);
        if (modulation != nullptr) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int c = col + i * 2;
                float v0 = __fmul_rn(vals[i * 2], inverse);
                float v1 = __fmul_rn(vals[i * 2 + 1], inverse);
                v0 = __fadd_rn(__fmul_rn(v0, __fadd_rn(1.0f, modulation[mod_base + c])), modulation[mod_base + columns + c]);
                v1 = __fadd_rn(__fmul_rn(v1, __fadd_rn(1.0f, modulation[mod_base + c + 1])), modulation[mod_base + columns + c + 1]);
                v0 = __bfloat162float(__float2bfloat16_rn(v0));
                v1 = __bfloat162float(__float2bfloat16_rn(v1));
                out_u16[i] = __nv_cvt_float2_to_fp8x2(make_float2(v0 * inv_global, v1 * inv_global), __NV_SATFINITE, __NV_E4M3);
                if (write_gate) {
                    gate[gate_base + c] = __float2bfloat16_rn(modulation[mod_base + 2 * columns + c]);
                    gate[gate_base + c + 1] = __float2bfloat16_rn(modulation[mod_base + 2 * columns + c + 1]);
                }
            }
        } else {
            float4 s0 = *reinterpret_cast<const float4*>(scale + col);
            float4 s1 = *reinterpret_cast<const float4*>(scale + col + 4);
            float sc[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float v0 = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(vals[i * 2], inverse), __fadd_rn(1.0f, sc[i * 2]))));
                float v1 = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(vals[i * 2 + 1], inverse), __fadd_rn(1.0f, sc[i * 2 + 1]))));
                out_u16[i] = __nv_cvt_float2_to_fp8x2(make_float2(v0 * inv_global, v1 * inv_global), __NV_SATFINITE, __NV_E4M3);
            }
        }
        *reinterpret_cast<uint64_t*>(output + base + col) = out_pack;
    }
}

__device__ __forceinline__ void store_residual(__nv_bfloat16* dst, uint4 value, bool evict_last) {
    if (!evict_last) {
        *reinterpret_cast<uint4*>(dst) = value;
        return;
    }
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;"
                 ::"l"(dst), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w), "l"(policy) : "memory");
}

__global__ void norm_fp4_reduce_vec8(
    const __nv_bfloat16* __restrict__ inputs, const __nv_bfloat16* __restrict__ update,
    const __nv_bfloat16* __restrict__ residual_gate, const float* __restrict__ scale,
    const float* __restrict__ modulation, __nv_bfloat16* __restrict__ residual,
    uint8_t* __restrict__ output, uint8_t* __restrict__ scales,
    uint8_t* __restrict__ blocked, __nv_bfloat16* __restrict__ gate,
    const float* __restrict__ global, int columns, int tokens, bool residual_evict_last) {
    __shared__ float warp_sums[8];
    pdl_wait();
    pdl_launch_dependents();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int batch = row / tokens;
    bool write_gate = (gate != nullptr) && (row == batch * tokens);
    int64_t base = int64_t(row) * columns;
    int64_t gate_base = int64_t(batch) * columns;
    int64_t mod_base = int64_t(batch) * 3 * columns;
    int col = tid * 8;
    float vals[8];
    float sum = 0.0f;
    if (col < columns) {
        uint4 in_u4 = *reinterpret_cast<const uint4*>(inputs + base + col);
        const __nv_bfloat162* in_b2 = reinterpret_cast<const __nv_bfloat162*>(&in_u4);
        if (update != nullptr) {
            uint4 up_u4 = *reinterpret_cast<const uint4*>(update + base + col);
            const __nv_bfloat162* up_b2 = reinterpret_cast<const __nv_bfloat162*>(&up_u4);
            uint4 rg_u4;
            const __nv_bfloat162* rg_b2 = nullptr;
            if (residual_gate != nullptr) {
                rg_u4 = *reinterpret_cast<const uint4*>(residual_gate + gate_base + col);
                rg_b2 = reinterpret_cast<const __nv_bfloat162*>(&rg_u4);
            }
            uint4 res_u4;
            __nv_bfloat162* res_b2 = reinterpret_cast<__nv_bfloat162*>(&res_u4);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 v = __bfloat1622float2(in_b2[i]);
                float2 d = __bfloat1622float2(up_b2[i]);
                if (rg_b2 != nullptr) {
                    float2 f = __bfloat1622float2(rg_b2[i]);
                    d.x = __bfloat162float(__float2bfloat16_rn(__fmul_rn(d.x, f.x)));
                    d.y = __bfloat162float(__float2bfloat16_rn(__fmul_rn(d.y, f.y)));
                }
                __nv_bfloat16 r0 = __float2bfloat16_rn(__fadd_rn(v.x, d.x));
                __nv_bfloat16 r1 = __float2bfloat16_rn(__fadd_rn(v.y, d.y));
                res_b2[i] = __halves2bfloat162(r0, r1);
                vals[i * 2] = __bfloat162float(r0);
                vals[i * 2 + 1] = __bfloat162float(r1);
                sum = __fadd_rn(sum, __fmul_rn(vals[i * 2], vals[i * 2]));
                sum = __fadd_rn(sum, __fmul_rn(vals[i * 2 + 1], vals[i * 2 + 1]));
            }
            if (residual != nullptr) {
                store_residual(residual + base + col, res_u4, residual_evict_last);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 v = __bfloat1622float2(in_b2[i]);
                vals[i * 2] = v.x;
                vals[i * 2 + 1] = v.y;
                sum = __fadd_rn(sum, __fmul_rn(v.x, v.x));
                sum = __fadd_rn(sum, __fmul_rn(v.y, v.y));
            }
            if (residual != nullptr) {
                store_residual(residual + base + col, in_u4, residual_evict_last);
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum;
    __syncthreads();
    if (tid < 8) {
        float wsum = (tid < (blockDim.x >> 5)) ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            wsum = __fadd_rn(wsum, __shfl_xor_sync(0xff, wsum, offset));
        if (tid == 0) warp_sums[0] = rsqrtf(__fadd_rn(__fmul_rn(wsum, 1.0f / columns), 1.0e-6f));
    }
    __syncthreads();
    float inverse = warp_sums[0];
    float g_scale = *global;
    if (col < columns) {
        float maximum = 0.0f;
        if (modulation != nullptr) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int c = col + i;
                float v = __fmul_rn(vals[i], inverse);
                v = __fadd_rn(__fmul_rn(v, __fadd_rn(1.0f, modulation[mod_base + c])), modulation[mod_base + columns + c]);
                vals[i] = __bfloat162float(__float2bfloat16_rn(v));
                maximum = fmaxf(maximum, fabsf(vals[i]));
                if (write_gate) {
                    gate[gate_base + c] = __float2bfloat16_rn(modulation[mod_base + 2 * columns + c]);
                }
            }
        } else {
            float4 s0 = *reinterpret_cast<const float4*>(scale + col);
            float4 s1 = *reinterpret_cast<const float4*>(scale + col + 4);
            float sc[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v = __fmul_rn(__fmul_rn(vals[i], inverse), __fadd_rn(1.0f, sc[i]));
                vals[i] = __bfloat162float(__float2bfloat16_rn(v));
                maximum = fmaxf(maximum, fabsf(vals[i]));
            }
        }
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
        float raw_scale = fminf(448.0f, maximum / (6.0f * g_scale));
        uint8_t scale_code = __nv_cvt_float_to_fp8(raw_scale, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_e4m3 converted;
        converted.__x = scale_code;
        float denominator = g_scale * float(converted);
        uint32_t packed4 = 0;
        uint8_t* p_bytes = reinterpret_cast<uint8_t*>(&packed4);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            p_bytes[i] = cc_fp4_cast(
                make_float2(denominator > 0.0f ? vals[i * 2] / denominator : 0.0f,
                            denominator > 0.0f ? vals[i * 2 + 1] / denominator : 0.0f));
        }
        *reinterpret_cast<uint32_t*>(output + (base + col) / 2) = packed4;
        if ((tid & 1) == 0) {
            int column_block = tid >> 1;
            int blocks_per_row = columns / 16;
            int column_tiles = (blocks_per_row + 3) / 4;
            int offset = ((row / 128) * column_tiles + column_block / 4) * 512
                       + (row % 32) * 16 + ((row % 128) / 32) * 4 + column_block % 4;
            blocked[offset] = scale_code;
            scales[row * blocks_per_row + column_block] = scale_code;
        }
    }
}

// flags bit 0: store the NVFP4 path's residual with an L2 evict_last policy (the caller discards it later).
extern "C" int cc_norm_quant_reduce(
    const void* inputs, const void* update, const void* residual_gate,
    const void* scale, const void* modulation, void* residual,
    void* output, void* scales, void* blocked, void* gate, const void* global,
    int rows, int columns, int tokens, int bits, int flags, void* stream) {
    if (bits == 8) {
        if (columns == 1024 || columns == 2048) {
            int threads = columns / 8;
            return int(launch_pdl(
                norm_fp8_reduce_vec8, dim3(rows), dim3(threads), 0, (cudaStream_t)stream,
                (const __nv_bfloat16*)inputs, (const __nv_bfloat16*)update,
                (const __nv_bfloat16*)residual_gate, (const float*)scale,
                (const float*)modulation, (__nv_bfloat16*)residual,
                (uint8_t*)output, (__nv_bfloat16*)gate, (const float*)global,
                columns, tokens));
        } else {
            norm_fp8_reduce<<<rows, THREADS, 0, (cudaStream_t)stream>>>(
                (const __nv_bfloat16*)inputs, (const __nv_bfloat16*)update,
                (const __nv_bfloat16*)residual_gate, (const float*)scale,
                (const float*)modulation, (__nv_bfloat16*)residual,
                (uint8_t*)output, (__nv_bfloat16*)gate, (const float*)global,
                columns, tokens);
        }
    } else {
        if (columns == 1024 || columns == 2048) {
            int threads = columns / 8;
            return int(launch_pdl(
                norm_fp4_reduce_vec8, dim3(rows), dim3(threads), 0, (cudaStream_t)stream,
                (const __nv_bfloat16*)inputs, (const __nv_bfloat16*)update,
                (const __nv_bfloat16*)residual_gate, (const float*)scale,
                (const float*)modulation, (__nv_bfloat16*)residual,
                (uint8_t*)output, (uint8_t*)scales, (uint8_t*)blocked,
                (__nv_bfloat16*)gate, (const float*)global,
                columns, tokens, (flags & 1) != 0));
        } else {
            norm_fp4_reduce<<<rows, THREADS, 0, (cudaStream_t)stream>>>(
                (const __nv_bfloat16*)inputs, (const __nv_bfloat16*)update,
                (const __nv_bfloat16*)residual_gate, (const float*)scale,
                (const float*)modulation, (__nv_bfloat16*)residual,
                (uint8_t*)output, (uint8_t*)scales, (uint8_t*)blocked,
                (__nv_bfloat16*)gate, (const float*)global,
                rows, columns, tokens);
        }
    }
    return int(cudaGetLastError());
}

__global__ void layer_norm_fp8_reduce(
    const __nv_bfloat16* __restrict__ inputs, const __nv_bfloat16* __restrict__ update,
    const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ residual, uint8_t* __restrict__ output,
    float inv_global, int columns) {
    __shared__ float warp_sum[8];
    __shared__ float warp_sq[8];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int64_t base = int64_t(row) * columns;
    int col = tid * 8;
    float vals[8];
    float sum = 0.0f;
    float sq_sum = 0.0f;
    if (col < columns) {
        uint4 in_u4 = *reinterpret_cast<const uint4*>(inputs + base + col);
        const __nv_bfloat162* in_b2 = reinterpret_cast<const __nv_bfloat162*>(&in_u4);
        if (update != nullptr) {
            uint4 up_u4 = *reinterpret_cast<const uint4*>(update + base + col);
            const __nv_bfloat162* up_b2 = reinterpret_cast<const __nv_bfloat162*>(&up_u4);
            uint4 res_u4;
            __nv_bfloat162* res_b2 = reinterpret_cast<__nv_bfloat162*>(&res_u4);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 v = __bfloat1622float2(in_b2[i]);
                float2 u = __bfloat1622float2(up_b2[i]);
                __nv_bfloat162 r = __floats2bfloat162_rn(v.x + u.x, v.y + u.y);
                res_b2[i] = r;
                float2 rf = __bfloat1622float2(r);
                vals[i * 2] = rf.x;
                vals[i * 2 + 1] = rf.y;
                sum += rf.x + rf.y;
                sq_sum += rf.x * rf.x + rf.y * rf.y;
            }
            if (residual != nullptr) {
                *reinterpret_cast<uint4*>(residual + base + col) = res_u4;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 v = __bfloat1622float2(in_b2[i]);
                vals[i * 2] = v.x;
                vals[i * 2 + 1] = v.y;
                sum += v.x + v.y;
                sq_sum += v.x * v.x + v.y * v.y;
            }
            if (residual != nullptr) {
                *reinterpret_cast<uint4*>(residual + base + col) = in_u4;
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
        sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, offset);
    }
    if ((tid & 31) == 0) {
        warp_sum[tid >> 5] = sum;
        warp_sq[tid >> 5] = sq_sum;
    }
    __syncthreads();
    if (tid < 8) {
        float ws = (tid < (blockDim.x >> 5)) ? warp_sum[tid] : 0.0f;
        float wq = (tid < (blockDim.x >> 5)) ? warp_sq[tid] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            ws += __shfl_xor_sync(0xff, ws, offset);
            wq += __shfl_xor_sync(0xff, wq, offset);
        }
        if (tid == 0) {
            float m = ws / float(columns);
            float var = fmaxf(0.0f, wq / float(columns) - m * m);
            warp_sum[0] = m;
            warp_sq[0] = rsqrtf(var + 1.0e-6f);
        }
    }
    __syncthreads();
    float mean = warp_sum[0];
    float inv_std = warp_sq[0];
    if (col < columns) {
        uint4 w_u4 = *reinterpret_cast<const uint4*>(weight + col);
        uint4 b_u4 = *reinterpret_cast<const uint4*>(bias + col);
        const __nv_bfloat162* w_b2 = reinterpret_cast<const __nv_bfloat162*>(&w_u4);
        const __nv_bfloat162* b_b2 = reinterpret_cast<const __nv_bfloat162*>(&b_u4);
        if (inv_global > 0.0f) {
            uint64_t out_pack = 0;
            uint16_t* out_u16 = reinterpret_cast<uint16_t*>(&out_pack);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 wf = __bfloat1622float2(w_b2[i]);
                float2 bf = __bfloat1622float2(b_b2[i]);
                float v0 = (vals[i * 2] - mean) * inv_std * wf.x + bf.x;
                float v1 = (vals[i * 2 + 1] - mean) * inv_std * wf.y + bf.y;
                v0 = __bfloat162float(__float2bfloat16_rn(v0));
                v1 = __bfloat162float(__float2bfloat16_rn(v1));
                out_u16[i] = __nv_cvt_float2_to_fp8x2(make_float2(v0 * inv_global, v1 * inv_global), __NV_SATFINITE, __NV_E4M3);
            }
            *reinterpret_cast<uint64_t*>(output + base + col) = out_pack;
        } else {
            uint4 out_u4;
            __nv_bfloat162* out_b2 = reinterpret_cast<__nv_bfloat162*>(&out_u4);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 wf = __bfloat1622float2(w_b2[i]);
                float2 bf = __bfloat1622float2(b_b2[i]);
                float v0 = (vals[i * 2] - mean) * inv_std * wf.x + bf.x;
                float v1 = (vals[i * 2 + 1] - mean) * inv_std * wf.y + bf.y;
                out_b2[i] = __floats2bfloat162_rn(v0, v1);
            }
            *reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(output) + base + col) = out_u4;
        }
    }
}

extern "C" int cc_layer_norm_fp8(
    const void* inputs, const void* update, const void* weight, const void* bias,
    void* residual, void* output, float inv_global, int rows, int columns, void* stream) {
    int threads = (columns <= 1152) ? 160 : 256;
    layer_norm_fp8_reduce<<<rows, threads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)inputs, (const __nv_bfloat16*)update,
        (const __nv_bfloat16*)weight, (const __nv_bfloat16*)bias,
        (__nv_bfloat16*)residual, (uint8_t*)output, inv_global, columns);
    return int(cudaGetLastError());
}

__global__ void bhsd_to_bshd_strided(
    const __nv_bfloat16* __restrict__ input, void* __restrict__ output, float inv_scale,
    int64_t stride_b, int64_t stride_h, int64_t stride_s,
    int H, int S, int D) {
    int bs = blockIdx.x;
    int b = bs / S;
    int s = bs - b * S;
    int total_vec8 = (H * D) / 8;
    for (int idx8 = threadIdx.x; idx8 < total_vec8; idx8 += blockDim.x) {
        int elem = idx8 * 8;
        int h = elem / D;
        int d = elem - h * D;
        int64_t src_idx = int64_t(b) * stride_b + int64_t(h) * stride_h + int64_t(s) * stride_s + d;
        int64_t dst_idx = int64_t(bs) * (H * D) + elem;
        uint4 in_u4 = *reinterpret_cast<const uint4*>(input + src_idx);
        if (inv_scale > 0.0f) {
            const __nv_bfloat162* in_b2 = reinterpret_cast<const __nv_bfloat162*>(&in_u4);
            uint64_t out_pack = 0;
            uint16_t* out_u16 = reinterpret_cast<uint16_t*>(&out_pack);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 vf = __bfloat1622float2(in_b2[i]);
                out_u16[i] = __nv_cvt_float2_to_fp8x2(make_float2(vf.x * inv_scale, vf.y * inv_scale), __NV_SATFINITE, __NV_E4M3);
            }
            *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(output) + dst_idx) = out_pack;
        } else {
            *reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(output) + dst_idx) = in_u4;
        }
    }
}

extern "C" int cc_bhsd_fp8(
    const void* input, void* output, float inv_scale,
    int64_t stride_b, int64_t stride_h, int64_t stride_s,
    int B, int H, int S, int D, void* stream) {
    int vec8 = (H * D) / 8;
    int threads = (vec8 <= 160) ? 160 : 256;
    bhsd_to_bshd_strided<<<B * S, threads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, output, inv_scale,
        stride_b, stride_h, stride_s, H, S, D);
    return int(cudaGetLastError());
}

__global__ void bhsd_to_bshd_fp4_strided(
    const __nv_bfloat16* __restrict__ input, uint8_t* __restrict__ output,
    uint8_t* __restrict__ blocked, float g_scale,
    int64_t stride_b, int64_t stride_h, int64_t stride_s,
    int H, int S, int D) {
    pdl_wait();
    pdl_launch_dependents();
    int row = blockIdx.x;
    int b = row / S;
    int s = row - b * S;
    int columns = H * D;
    int blocks_per_row = columns / 16;
    int column_tiles = (blocks_per_row + 3) / 4;
    for (int idx8 = threadIdx.x; idx8 < columns / 8; idx8 += blockDim.x) {
        int elem = idx8 * 8;
        int h = elem / D;
        int d = elem - h * D;
        int64_t src_idx = int64_t(b) * stride_b + int64_t(h) * stride_h + int64_t(s) * stride_s + d;
        uint4 in_u4 = *reinterpret_cast<const uint4*>(input + src_idx);
        const __nv_bfloat162* in_b2 = reinterpret_cast<const __nv_bfloat162*>(&in_u4);
        float vals[8];
        float maximum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 vf = __bfloat1622float2(in_b2[i]);
            vals[i * 2] = vf.x;
            vals[i * 2 + 1] = vf.y;
            maximum = fmaxf(maximum, fmaxf(fabsf(vf.x), fabsf(vf.y)));
        }
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
        float raw_scale = fminf(448.0f, maximum / (6.0f * g_scale));
        uint8_t scale_code = __nv_cvt_float_to_fp8(raw_scale, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_e4m3 converted;
        converted.__x = scale_code;
        float denominator = g_scale * float(converted);
        uint32_t packed4 = 0;
        uint8_t* p_bytes = reinterpret_cast<uint8_t*>(&packed4);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            p_bytes[i] = cc_fp4_cast(
                make_float2(denominator > 0.0f ? vals[i * 2] / denominator : 0.0f,
                            denominator > 0.0f ? vals[i * 2 + 1] / denominator : 0.0f));
        }
        *reinterpret_cast<uint32_t*>(output + (int64_t(row) * columns + elem) / 2) = packed4;
        if ((idx8 & 1) == 0) {
            int column_block = idx8 >> 1;
            int offset = ((row / 128) * column_tiles + column_block / 4) * 512
                       + (row % 32) * 16 + ((row % 128) / 32) * 4 + (column_block & 3);
            blocked[offset] = scale_code;
        }
    }
}

extern "C" int cc_bhsd_fp4(
    const void* input, void* output, void* blocked, float g_scale,
    int64_t stride_b, int64_t stride_h, int64_t stride_s,
    int B, int H, int S, int D, void* stream) {
    int vec8 = (H * D) / 8;
    int threads = (vec8 <= 160) ? 160 : 256;
    return int(launch_pdl(
        bhsd_to_bshd_fp4_strided, dim3(B * S), dim3(threads), 0, (cudaStream_t)stream,
        (const __nv_bfloat16*)input, (uint8_t*)output, (uint8_t*)blocked, g_scale,
        stride_b, stride_h, stride_s, H, S, D));
}

__global__ void layer_norm_fp4_reduce(
    const __nv_bfloat16* __restrict__ inputs, const __nv_bfloat16* __restrict__ update,
    const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ residual, uint8_t* __restrict__ output,
    uint8_t* __restrict__ blocked, float g_scale, int columns) {
    __shared__ float warp_sum[8];
    __shared__ float warp_sq[8];
    pdl_wait();
    pdl_launch_dependents();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int64_t base = int64_t(row) * columns;
    int col = tid * 8;
    float vals[8];
    float sum = 0.0f;
    float sq_sum = 0.0f;
    if (col < columns) {
        uint4 in_u4 = *reinterpret_cast<const uint4*>(inputs + base + col);
        const __nv_bfloat162* in_b2 = reinterpret_cast<const __nv_bfloat162*>(&in_u4);
        if (update != nullptr) {
            uint4 up_u4 = *reinterpret_cast<const uint4*>(update + base + col);
            const __nv_bfloat162* up_b2 = reinterpret_cast<const __nv_bfloat162*>(&up_u4);
            uint4 res_u4;
            __nv_bfloat162* res_b2 = reinterpret_cast<__nv_bfloat162*>(&res_u4);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 v = __bfloat1622float2(in_b2[i]);
                float2 u = __bfloat1622float2(up_b2[i]);
                __nv_bfloat162 r = __floats2bfloat162_rn(v.x + u.x, v.y + u.y);
                res_b2[i] = r;
                float2 rf = __bfloat1622float2(r);
                vals[i * 2] = rf.x;
                vals[i * 2 + 1] = rf.y;
                sum += rf.x + rf.y;
                sq_sum += rf.x * rf.x + rf.y * rf.y;
            }
            if (residual != nullptr) {
                *reinterpret_cast<uint4*>(residual + base + col) = res_u4;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 v = __bfloat1622float2(in_b2[i]);
                vals[i * 2] = v.x;
                vals[i * 2 + 1] = v.y;
                sum += v.x + v.y;
                sq_sum += v.x * v.x + v.y * v.y;
            }
            if (residual != nullptr) {
                *reinterpret_cast<uint4*>(residual + base + col) = in_u4;
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
        sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, offset);
    }
    if ((tid & 31) == 0) {
        warp_sum[tid >> 5] = sum;
        warp_sq[tid >> 5] = sq_sum;
    }
    __syncthreads();
    if (tid < 8) {
        float ws = (tid < (blockDim.x >> 5)) ? warp_sum[tid] : 0.0f;
        float wq = (tid < (blockDim.x >> 5)) ? warp_sq[tid] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            ws += __shfl_xor_sync(0xff, ws, offset);
            wq += __shfl_xor_sync(0xff, wq, offset);
        }
        if (tid == 0) {
            float m = ws / float(columns);
            float var = fmaxf(0.0f, wq / float(columns) - m * m);
            warp_sum[0] = m;
            warp_sq[0] = rsqrtf(var + 1.0e-6f);
        }
    }
    __syncthreads();
    float mean = warp_sum[0];
    float inv_std = warp_sq[0];
    float maximum = 0.0f;
    if (col < columns) {
        uint4 w_u4 = *reinterpret_cast<const uint4*>(weight + col);
        uint4 b_u4 = *reinterpret_cast<const uint4*>(bias + col);
        const __nv_bfloat162* w_b2 = reinterpret_cast<const __nv_bfloat162*>(&w_u4);
        const __nv_bfloat162* b_b2 = reinterpret_cast<const __nv_bfloat162*>(&b_u4);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 wf = __bfloat1622float2(w_b2[i]);
            float2 bf = __bfloat1622float2(b_b2[i]);
            float v0 = __bfloat162float(__float2bfloat16_rn((vals[i * 2] - mean) * inv_std * wf.x + bf.x));
            float v1 = __bfloat162float(__float2bfloat16_rn((vals[i * 2 + 1] - mean) * inv_std * wf.y + bf.y));
            vals[i * 2] = v0;
            vals[i * 2 + 1] = v1;
            maximum = fmaxf(maximum, fmaxf(fabsf(v0), fabsf(v1)));
        }
    }
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
    if (col < columns) {
        float raw_scale = fminf(448.0f, maximum / (6.0f * g_scale));
        uint8_t scale_code = __nv_cvt_float_to_fp8(raw_scale, __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_e4m3 converted;
        converted.__x = scale_code;
        float denominator = g_scale * float(converted);
        uint32_t packed4 = 0;
        uint8_t* p_bytes = reinterpret_cast<uint8_t*>(&packed4);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            p_bytes[i] = cc_fp4_cast(
                make_float2(denominator > 0.0f ? vals[i * 2] / denominator : 0.0f,
                            denominator > 0.0f ? vals[i * 2 + 1] / denominator : 0.0f));
        }
        *reinterpret_cast<uint32_t*>(output + (base + col) / 2) = packed4;
        if ((tid & 1) == 0) {
            int column_block = tid >> 1;
            int blocks_per_row = columns / 16;
            int column_tiles = (blocks_per_row + 3) / 4;
            int offset = ((row / 128) * column_tiles + column_block / 4) * 512
                       + (row % 32) * 16 + ((row % 128) / 32) * 4 + (column_block & 3);
            blocked[offset] = scale_code;
        }
    }
}

extern "C" int cc_layer_norm_fp4(
    const void* inputs, const void* update, const void* weight, const void* bias,
    void* residual, void* output, void* blocked, float g_scale, int rows, int columns, void* stream) {
    int threads = (columns <= 1152) ? 160 : 256;
    return int(launch_pdl(
        layer_norm_fp4_reduce, dim3(rows), dim3(threads), 0, (cudaStream_t)stream,
        (const __nv_bfloat16*)inputs, (const __nv_bfloat16*)update,
        (const __nv_bfloat16*)weight, (const __nv_bfloat16*)bias,
        (__nv_bfloat16*)residual, (uint8_t*)output, (uint8_t*)blocked, g_scale, columns));
}
