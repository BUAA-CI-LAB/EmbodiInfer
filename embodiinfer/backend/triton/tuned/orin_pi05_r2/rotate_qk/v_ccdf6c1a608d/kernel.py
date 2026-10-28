"""Single-launch fused rotate-half RoPE for the captured pi05 Q/K layouts on Orin.

The frozen call is ``registry.rotate_qk`` over a fused QKV slab: the query is
heads 0..7 and the key is head 8 of the same ``[tokens, 10, 256]`` BF16 region,
and both rotated results are written dense. One Triton program owns a block of
tokens, loads that block's cosine/sine rows once, and walks the eight query head
rows plus the key row, so every input element is read once and every output
element is written once from a single kernel launch.
"""

import torch
import triton
import triton.language as tl

# Captured (frozen) layouts, in BF16 element units. See task/replay.json.
_HEAD_DIM = 256  # head width in BF16 elements
_HALF_DIM = 128  # rotate-half distance
_Q_HEADS = 8  # query heads in the fused slab
_Q_TOKEN = 2560  # token stride of the fused QKV input slab
_Q_HEAD = 256  # head stride inside the fused QKV input slab
_TAB_TOKEN = 256  # token stride of the cosine/sine tables
_QO_TOKEN = 2048  # token stride of the dense query output (8 * 256)
_QO_HEAD = 256  # head stride of the dense query output
_KO_TOKEN = 256  # token stride of the dense key output

# Launch configuration. Each program handles one token block for all nine head
# rows; 64 lanes cover a [4, 128] BF16 tile with one 16-byte access per lane.
_BLOCK_T = 4
_NUM_WARPS = 2


@triton.jit
def _rotate_qk_fused(
    Q,
    K,
    Cos,
    Sin,
    QOut,
    KOut,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    Q_HEADS: tl.constexpr,
    HALF: tl.constexpr,
    Q_TOKEN: tl.constexpr,
    Q_HEAD: tl.constexpr,
    TAB_TOKEN: tl.constexpr,
    QO_TOKEN: tl.constexpr,
    QO_HEAD: tl.constexpr,
    KO_TOKEN: tl.constexpr,
):
    """Rotate one token block of Q and K, reading each element exactly once."""
    tok = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    dim = tl.arange(0, HALF)
    keep = tok[:, None] < T

    in_base = tok[:, None] * Q_TOKEN + dim[None, :]
    tab_base = tok[:, None] * TAB_TOKEN + dim[None, :]
    qo_base = tok[:, None] * QO_TOKEN + dim[None, :]
    ko_base = tok[:, None] * KO_TOKEN + dim[None, :]

    # One table read per token serves all nine head rows.
    cos_lo = tl.load(Cos + tab_base, mask=keep, other=0.0).to(tl.float32)
    cos_hi = tl.load(Cos + tab_base + HALF, mask=keep, other=0.0).to(tl.float32)
    sin_lo = tl.load(Sin + tab_base, mask=keep, other=0.0).to(tl.float32)
    sin_hi = tl.load(Sin + tab_base + HALF, mask=keep, other=0.0).to(tl.float32)

    for head in tl.static_range(Q_HEADS):
        src = in_base + head * Q_HEAD
        lo = tl.load(Q + src, mask=keep, other=0.0).to(tl.float32)
        hi = tl.load(Q + src + HALF, mask=keep, other=0.0).to(tl.float32)
        dst = qo_base + head * QO_HEAD
        tl.store(
            QOut + dst,
            (lo * cos_lo - hi * sin_lo).to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=keep,
        )
        tl.store(
            QOut + dst + HALF,
            (hi * cos_hi + lo * sin_hi).to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=keep,
        )

    # Key row: its tensor pointer already carries the recorded storage offset.
    lo = tl.load(K + in_base, mask=keep, other=0.0).to(tl.float32)
    hi = tl.load(K + in_base + HALF, mask=keep, other=0.0).to(tl.float32)
    tl.store(
        KOut + ko_base,
        (lo * cos_lo - hi * sin_lo).to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=keep,
    )
    tl.store(
        KOut + ko_base + HALF,
        (hi * cos_hi + lo * sin_hi).to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=keep,
    )


def run(
    x0: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rotated query/key in the captured dense output layouts.

    ``x0`` is the BHSD query view, ``x1`` the BHSD key view of the same fused
    QKV storage, and ``x2``/``x3`` the ``[1, tokens, 256]`` cosine/sine tables.
    All four are read-only BF16 CUDA tensors. Each output owns fresh storage,
    inputs are never written, and the launch uses the caller's current stream
    and device without touching numerical settings.
    """
    tensors = (x0, x1, x2, x3)
    for name, tensor in zip(("x0", "x1", "x2", "x3"), tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided or tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be a strided BF16 tensor")
        if tensor.device.type != "cuda" or tensor.device != x0.device:
            raise ValueError("rotate_qk inputs must share one CUDA device")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have a contiguous final dimension")

    tokens = int(x0.shape[2])
    # Check exactly the addressing this kernel hard-codes; strides of size-one
    # dimensions address no other element and are left unconstrained.
    if (
        tuple(x0.shape) != (1, _Q_HEADS, tokens, _HEAD_DIM)
        or x0.stride(1) != _Q_HEAD
        or x0.stride(2) != _Q_TOKEN
        or x0.stride(3) != 1
        or tuple(x1.shape) != (1, 1, tokens, _HEAD_DIM)
        or x1.stride(2) != _Q_TOKEN
        or x1.stride(3) != 1
        or tuple(x2.shape) != (1, tokens, _HEAD_DIM)
        or x2.stride(1) != _TAB_TOKEN
        or x2.stride(2) != 1
        or tuple(x3.shape) != (1, tokens, _HEAD_DIM)
        or x3.stride(1) != _TAB_TOKEN
        or x3.stride(2) != 1
    ):
        raise ValueError("rotate_qk requires the captured pi05 Q/K/table layouts")

    with torch.cuda.device(x0.device):
        query_out = torch.empty_strided(
            (1, _Q_HEADS, tokens, _HEAD_DIM),
            (_QO_TOKEN * tokens, _QO_HEAD, _QO_TOKEN, 1),
            dtype=torch.bfloat16,
            device=x0.device,
        )
        # The captured key output carries the same stride on both leading axes
        # (the head axis has extent 1, so it addresses nothing else).
        key_out = torch.empty_strided(
            (1, 1, tokens, _HEAD_DIM),
            (_KO_TOKEN * tokens, _KO_TOKEN * tokens, _KO_TOKEN, 1),
            dtype=torch.bfloat16,
            device=x1.device,
        )
        _rotate_qk_fused[(triton.cdiv(tokens, _BLOCK_T),)](
            x0,
            x1,
            x2,
            x3,
            query_out,
            key_out,
            T=tokens,
            BLOCK_T=_BLOCK_T,
            Q_HEADS=_Q_HEADS,
            HALF=_HALF_DIM,
            Q_TOKEN=_Q_TOKEN,
            Q_HEAD=_Q_HEAD,
            TAB_TOKEN=_TAB_TOKEN,
            QO_TOKEN=_QO_TOKEN,
            QO_HEAD=_QO_HEAD,
            KO_TOKEN=_KO_TOKEN,
            num_warps=_NUM_WARPS,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return query_out, key_out
