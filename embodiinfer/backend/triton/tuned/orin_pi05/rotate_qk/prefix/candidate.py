"""Packed BF16 rotate-half RoPE for the captured Orin query/key layouts."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate_packed_bf16(lo, hi, cos_lo, cos_hi, sin_lo, sin_hi):
    """Rotate two adjacent lanes, rounding every product and sum to BF16."""
    out_lo, out_hi = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b32 zero, one, neg_hi;
            .reg .b32 product_a, product_b, product_c, product_d;
            .reg .b32 result_lo, result_hi;
            mov.b32 zero, 0x00000000;
            mov.b32 one, 0x3f803f80;
            xor.b32 neg_hi, $3, 0x80008000;

            fma.rn.bf16x2 product_a, $2, $4, zero;
            fma.rn.bf16x2 product_b, neg_hi, $6, zero;
            fma.rn.bf16x2 product_c, $3, $5, zero;
            fma.rn.bf16x2 product_d, $2, $7, zero;
            fma.rn.bf16x2 result_lo, product_a, one, product_b;
            fma.rn.bf16x2 result_hi, product_c, one, product_d;

            mov.b32 $0, result_lo;
            mov.b32 $1, result_hi;
        }
        """,
        constraints="=r,=r,r,r,r,r,r,r",
        args=[lo, hi, cos_lo, cos_hi, sin_lo, sin_hi],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    return out_lo, out_hi


@triton.jit
def _rotate_qk_token_packed(Q, K, Cos, Sin, Qout, Kout):
    """Process one token using 32-bit words containing adjacent BF16 values."""
    token = tl.program_id(0)
    heads = tl.arange(0, 8)
    pairs = tl.arange(0, 64)

    # Only pointer interpretation changes; all tensor elements remain BF16.
    q_words = Q.to(tl.pointer_type(tl.uint32))
    k_words = K.to(tl.pointer_type(tl.uint32))
    cos_words = Cos.to(tl.pointer_type(tl.uint32))
    sin_words = Sin.to(tl.pointer_type(tl.uint32))
    qout_words = Qout.to(tl.pointer_type(tl.uint32))
    kout_words = Kout.to(tl.pointer_type(tl.uint32))

    table_offsets = token * 128 + pairs
    cos_lo = tl.load(cos_words + table_offsets)
    cos_hi = tl.load(cos_words + table_offsets + 64)
    sin_lo = tl.load(sin_words + table_offsets)
    sin_hi = tl.load(sin_words + table_offsets + 64)

    # Ten input head slots per token, of which eight contain Q and one K.
    query_offsets = token * 1280 + heads[:, None] * 128 + pairs[None, :]
    query_lo = tl.load(q_words + query_offsets)
    query_hi = tl.load(q_words + query_offsets + 64)
    query_out_lo, query_out_hi = _rotate_packed_bf16(
        query_lo,
        query_hi,
        tl.broadcast_to(cos_lo[None, :], (8, 64)),
        tl.broadcast_to(cos_hi[None, :], (8, 64)),
        tl.broadcast_to(sin_lo[None, :], (8, 64)),
        tl.broadcast_to(sin_hi[None, :], (8, 64)),
    )
    query_out_offsets = token * 1024 + heads[:, None] * 128 + pairs[None, :]
    tl.store(qout_words + query_out_offsets, query_out_lo)
    tl.store(qout_words + query_out_offsets + 64, query_out_hi)

    # K's pointer already includes its BF16 storage offset of 2048 elements.
    key_offsets = token * 1280 + pairs
    key_lo = tl.load(k_words + key_offsets)
    key_hi = tl.load(k_words + key_offsets + 64)
    key_out_lo, key_out_hi = _rotate_packed_bf16(
        key_lo, key_hi, cos_lo, cos_hi, sin_lo, sin_hi
    )
    key_out_offsets = token * 128 + pairs
    tl.store(kout_words + key_out_offsets, key_out_lo)
    tl.store(kout_words + key_out_offsets + 64, key_out_hi)


def run(
    x0: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return BF16 rotated query/key with the captured output strides.

    Accept the recorded query, key, cosine, and sine layouts on one CUDA
    device, with four-byte-aligned logical data pointers. Each output owns
    fresh storage; inputs, including shared query/key storage, are read-only.
    Packed BF16 arithmetic rounds every product and sum to nearest even,
    subject to the frozen FP64 error bounds. Unsupported metadata raises
    before allocation or launch. Use the input device's current stream and
    restore the caller's device without changing numerical settings.
    """
    expected_inputs = (
        ("x0", x0, (1, 8, 672, 256), (1720320, 256, 2560, 1)),
        ("x1", x1, (1, 1, 672, 256), (1720320, 256, 2560, 1)),
        ("x2", x2, (1, 672, 256), (172032, 256, 1)),
        ("x3", x3, (1, 672, 256), (172032, 256, 1)),
    )
    for name, tensor, shape, strides in expected_inputs:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided or tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be a strided BF16 tensor")
        if tensor.device.type != "cuda" or tensor.device != x0.device:
            raise ValueError("rotate_qk inputs must share one CUDA device")
        if tuple(tensor.shape) != shape or tuple(tensor.stride()) != strides:
            raise ValueError(f"{name} has an unsupported shape or stride")
        if tensor.data_ptr() % 4 != 0:
            raise ValueError(f"{name} must have a four-byte-aligned data pointer")

    # The device guard restores the original device and leaves streams intact.
    with torch.cuda.device(x0.device):
        query_out = torch.empty_strided(
            (1, 8, 672, 256),
            (1376256, 256, 2048, 1),
            dtype=torch.bfloat16,
            device=x0.device,
        )
        key_out = torch.empty_strided(
            (1, 1, 672, 256),
            (172032, 172032, 256, 1),
            dtype=torch.bfloat16,
            device=x1.device,
        )
        _rotate_qk_token_packed[(672,)](
            x0,
            x1,
            x2,
            x3,
            query_out,
            key_out,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return query_out, key_out
