// BF16 GELU lookup with unchanged FP4 block encoding.
#include "quantize.cu"

__global__ void init_gelu(uint16_t* table) {
    int bits = blockIdx.x * blockDim.x + threadIdx.x;
    float value = __bfloat162float(__ushort_as_bfloat16(uint16_t(bits)));
    float gelu = (0.5f * value) * (1.0f + erff(value * 0.7071067811865475244f));
    table[bits] = __bfloat16_as_ushort(__float2bfloat16(gelu));
}

template<int BLOCK, bool LOOKUP, bool SHARED>
__global__ void lookup_fp4(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                          uint8_t* output, uint8_t* scales, uint8_t* blocked,
                          const float* global, int rows, int columns, const uint16_t* table) {
    int row = blockIdx.y;
    int column_block = blockIdx.x * (BLOCK / 8) + threadIdx.x / 8;
    int blocks_per_row = columns / 16;
    if (column_block >= blocks_per_row) return;
    int lane = threadIdx.x & 7;
    int64_t index = int64_t(row) * columns + column_block * 16 + lane * 2;
    uint32_t codes = reinterpret_cast<const uint32_t*>(gate)[index / 2];
    float2 values = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(up)[index / 2]);
    float first, second;
    if constexpr (LOOKUP) {
        first = __bfloat162float(__ushort_as_bfloat16(table[codes & 65535]));
        second = __bfloat162float(__ushort_as_bfloat16(table[codes >> 16]));
    } else {
        float a = __bfloat162float(__ushort_as_bfloat16(codes & 65535));
        float b = __bfloat162float(__ushort_as_bfloat16(codes >> 16));
        first = __bfloat162float(__float2bfloat16(
            (0.5f * a) * (1.0f + erff(a * 0.7071067811865475244f))));
        second = __bfloat162float(__float2bfloat16(
            (0.5f * b) * (1.0f + erff(b * 0.7071067811865475244f))));
    }
    first = __bfloat162float(__float2bfloat16(first * values.x));
    second = __bfloat162float(__float2bfloat16(second * values.y));
    float maximum = fmaxf(fabsf(first), fabsf(second));
    for (int delta = 4; delta; delta /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta, 8));
    uint8_t scale_code = 0;
    float denominator = 0;
    if (!SHARED || lane == 0) {
        scale_code = __nv_cvt_float_to_fp8(
            fminf(448.0f, maximum / (6.0f * *global)), __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_e4m3 converted;
        converted.__x = scale_code;
        denominator = *global * float(converted);
    }
    if constexpr (SHARED) {
        scale_code = __shfl_sync(0xffffffff, unsigned(scale_code), 0, 8);
        denominator = __shfl_sync(0xffffffff, denominator, 0, 8);
    }
    uint8_t packed = cc_fp4_cast(
        make_float2(denominator > 0.0f ? first / denominator : 0.0f,
                    denominator > 0.0f ? second / denominator : 0.0f));
    output[index / 2] = packed;
    if (lane == 0) {
        int column_tiles = (blocks_per_row + 3) / 4;
        int offset = ((row / 128) * column_tiles + column_block / 4) * 512
                   + (row % 32) * 16 + ((row % 128) / 32) * 4 + column_block % 4;
        blocked[offset] = scale_code;
        scales[row * blocks_per_row + column_block] = scale_code;
    }
}

extern "C" int cc_init_gelu(void* table, void* stream) {
    init_gelu<<<256, 256, 0, (cudaStream_t)stream>>>((uint16_t*)table);
    return int(cudaGetLastError());
}

template<int BLOCK>
__global__ void lookup_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                           __nv_bfloat16* output, int64_t count, const uint16_t* table) {
    int64_t index = (int64_t(blockIdx.x) * BLOCK + threadIdx.x) * 2;
    if (index >= count) return;
    float first = __bfloat162float(__ushort_as_bfloat16(
        __ldg(table + __bfloat16_as_ushort(gate[index])))) * __bfloat162float(up[index]);
    if (index + 1 < count) {
        float second = __bfloat162float(__ushort_as_bfloat16(
            __ldg(table + __bfloat16_as_ushort(gate[index + 1]))))
            * __bfloat162float(up[index + 1]);
        reinterpret_cast<__nv_bfloat162*>(output)[index / 2] =
            __floats2bfloat162_rn(first, second);
    } else {
        output[index] = __float2bfloat16(first);
    }
}

template<int BLOCK>
void launch_bf16(const void* gate, const void* up, void* output, int64_t count,
                 const void* table, cudaStream_t stream) {
    lookup_bf16<BLOCK><<<(count + BLOCK * 2 - 1) / (BLOCK * 2), BLOCK, 0, stream>>>(
        (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up, (__nv_bfloat16*)output,
        count, (const uint16_t*)table);
}

extern "C" int cc_lookup_bf16(const void* gate, const void* up, void* output, int64_t count,
                              const void* table, int threads, void* stream) {
    if (threads == 128)
        launch_bf16<128>(gate, up, output, count, table, (cudaStream_t)stream);
    else if (threads == 512)
        launch_bf16<512>(gate, up, output, count, table, (cudaStream_t)stream);
    else
        launch_bf16<256>(gate, up, output, count, table, (cudaStream_t)stream);
    return int(cudaGetLastError());
}


template<int BLOCK>
__global__ void lookup_fp4_multi4(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    uint8_t* __restrict__ blocked,
    const float* __restrict__ global,
    int rows, int columns,
    const uint16_t* __restrict__ table)
{
    int row = blockIdx.y;
    int blocks_per_row = columns / 16;
    int lane = threadIdx.x & 7;
    int subwarp = threadIdx.x >> 3;
    int subwarps_per_cta = BLOCK >> 3;
    int base_col_block = blockIdx.x * subwarps_per_cta * 4 + subwarp * 4;
    float g_scale = *global;
    int column_tiles = (blocks_per_row + 3) / 4;
    int tile_row_off = (row / 128) * column_tiles * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4;

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int column_block = base_col_block + m;
        if (column_block < blocks_per_row) {
            int64_t index = int64_t(row) * columns + column_block * 16 + lane * 2;
            uint32_t codes = reinterpret_cast<const uint32_t*>(gate)[index / 2];
            float2 values = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(up)[index / 2]);
            float first = __bfloat162float(__ushort_as_bfloat16(__ldg(table + (codes & 65535))));
            float second = __bfloat162float(__ushort_as_bfloat16(__ldg(table + (codes >> 16))));
            first = __bfloat162float(__float2bfloat16_rn(first * values.x));
            second = __bfloat162float(__float2bfloat16_rn(second * values.y));
            float maximum = fmaxf(fabsf(first), fabsf(second));
            #pragma unroll
            for (int delta = 4; delta > 0; delta >>= 1)
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta, 8));
            float raw_scale = fminf(448.0f, maximum / (6.0f * g_scale));
            uint8_t scale_code = __nv_cvt_float_to_fp8(
                raw_scale, __NV_SATFINITE, __NV_E4M3);
            __nv_fp8_e4m3 converted;
            converted.__x = scale_code;
            float denominator = g_scale * float(converted);
            uint8_t packed = cc_fp4_cast(
                make_float2(denominator > 0.0f ? first / denominator : 0.0f,
                            denominator > 0.0f ? second / denominator : 0.0f));
            output[index / 2] = packed;
            if (lane == 0) {
                int offset = tile_row_off + (column_block / 4) * 512 + (column_block & 3);
                blocked[offset] = scale_code;
                scales[row * blocks_per_row + column_block] = scale_code;
            }
        }
    }
}

template<int BLOCK>
__global__ void lookup_fp4_vec8(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    uint8_t* __restrict__ blocked,
    const float* __restrict__ global,
    int rows, int columns,
    const uint16_t* __restrict__ table)
{
    // One thread per 8 columns: a single uint4 load per operand, register
    // resident products, and one warp shuffle to finish the 16-wide block
    // maximum. The elementwise math (table GELU, BF16 product rounding,
    // E4M3 block scale, E2M1 cast) matches lookup_fp4_multi4 exactly; only
    // the grouping of the exact fmaxf reduction changes, so outputs are
    // bitwise identical.
    int row = blockIdx.y;
    int blocks_per_row = columns / 16;
    int col0 = (blockIdx.x * BLOCK + threadIdx.x) * 8;
    if (col0 >= columns) return;
    int column_block = col0 >> 4;
    int half = (col0 >> 3) & 1;
    int64_t index = int64_t(row) * columns + col0;
    uint4 g4 = reinterpret_cast<const uint4*>(gate)[index >> 3];
    uint4 u4 = reinterpret_cast<const uint4*>(up)[index >> 3];
    const uint16_t* g16 = reinterpret_cast<const uint16_t*>(&g4);
    const uint16_t* u16 = reinterpret_cast<const uint16_t*>(&u4);
    float prod[8];
    float maximum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float g = __bfloat162float(__ushort_as_bfloat16(__ldg(table + g16[i])));
        float u = __bfloat162float(__ushort_as_bfloat16(u16[i]));
        float p = __bfloat162float(__float2bfloat16_rn(g * u));
        prod[i] = p;
        maximum = fmaxf(maximum, fabsf(p));
    }
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
    float g_scale = *global;
    float raw_scale = fminf(448.0f, maximum / (6.0f * g_scale));
    uint8_t scale_code = __nv_cvt_float_to_fp8(
        raw_scale, __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 converted;
    converted.__x = scale_code;
    float denominator = g_scale * float(converted);
    uint32_t packed4 = 0;
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint8_t byte = cc_fp4_cast(make_float2(
            denominator > 0.0f ? prod[2 * p] / denominator : 0.0f,
            denominator > 0.0f ? prod[2 * p + 1] / denominator : 0.0f));
        packed4 |= uint32_t(byte) << (8 * p);
    }
    reinterpret_cast<uint32_t*>(output)[index >> 3] = packed4;
    if (half == 0) {
        int column_tiles = (blocks_per_row + 3) / 4;
        int offset = ((row / 128) * column_tiles + column_block / 4) * 512
                   + (row % 32) * 16 + ((row % 128) / 32) * 4
                   + (column_block & 3);
        blocked[offset] = scale_code;
        scales[row * blocks_per_row + column_block] = scale_code;
    }
}

template<int BLOCK>
void launch(const void* gate, const void* up, void* output, void* scales, void* blocked,
            const void* global, int rows, int columns, const void* table,
            int mode, cudaStream_t stream) {
    if (mode == 1 && columns >= 4096) {
        if (columns % 2048 == 0) {
            dim3 grid8(columns / 2048, rows);
            lookup_fp4_vec8<256><<<grid8, 256, 0, stream>>>(
                (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
                (uint8_t*)output, (uint8_t*)scales, (uint8_t*)blocked,
                (const float*)global, rows, columns, (const uint16_t*)table);
            return;
        }
        int subwarps = (BLOCK / 8) * 4;
        dim3 grid4((columns / 16 + subwarps - 1) / subwarps, rows);
        lookup_fp4_multi4<BLOCK><<<grid4, BLOCK, 0, stream>>>(
            (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up, (uint8_t*)output,
            (uint8_t*)scales, (uint8_t*)blocked, (const float*)global, rows, columns,
            (const uint16_t*)table);
        return;
    }
    dim3 grid((columns / 16 + BLOCK / 8 - 1) / (BLOCK / 8), rows);
    if (mode == 2)
        lookup_fp4<BLOCK, true, true><<<grid, BLOCK, 0, stream>>>(
            (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up, (uint8_t*)output,
            (uint8_t*)scales, (uint8_t*)blocked, (const float*)global, rows, columns,
            (const uint16_t*)table);
    else if (mode == 1)
        lookup_fp4<BLOCK, true, false><<<grid, BLOCK, 0, stream>>>(
            (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up, (uint8_t*)output,
            (uint8_t*)scales, (uint8_t*)blocked, (const float*)global, rows, columns,
            (const uint16_t*)table);
    else
        lookup_fp4<BLOCK, false, false><<<grid, BLOCK, 0, stream>>>(
            (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up, (uint8_t*)output,
            (uint8_t*)scales, (uint8_t*)blocked, (const float*)global, rows, columns,
            (const uint16_t*)table);
}

extern "C" int cc_lookup_fp4(const void* gate, const void* up, void* output, void* scales,
                            void* blocked, const void* global, int rows, int columns,
                            const void* table, int threads, int mode, void* stream) {
    if (threads == 128)
        launch<128>(gate, up, output, scales, blocked, global, rows, columns,
                    table, mode, (cudaStream_t)stream);
    else if (threads == 512)
        launch<512>(gate, up, output, scales, blocked, global, rows, columns,
                    table, mode, (cudaStream_t)stream);
    else
        launch<256>(gate, up, output, scales, blocked, global, rows, columns,
                    table, mode, (cudaStream_t)stream);
    return int(cudaGetLastError());
}
