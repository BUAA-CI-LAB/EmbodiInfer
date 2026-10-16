"""Four-token paired-half RoPE for the captured pi05 BF16 layout on Orin."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate_qk_paired_kernel(
    Q,
    K,
    Cos,
    Sin,
    QOut,
    KOut,
    BLOCK_T: tl.constexpr,
    HALF_D: tl.constexpr,
):
    head = tl.program_id(1)
    tokens = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    dims = tl.arange(0, HALF_D)
    valid = tokens[:, None] < 50

    # The tensor pointers already include the recorded storage offsets.
    if head < 8:
        input_base = Q + head * 256
        output_base = QOut + head * 12800
    else:
        input_base = K
        output_base = KOut

    input_offsets = tokens[:, None] * 2560 + dims[None, :]
    a = tl.load(input_base + input_offsets, mask=valid, other=0.0).to(tl.float32)
    b = tl.load(input_base + input_offsets + HALF_D, mask=valid, other=0.0).to(tl.float32)

    table_offsets = tokens[:, None] * 256 + dims[None, :]
    c0 = tl.load(Cos + table_offsets, mask=valid, other=0.0).to(tl.float32)
    c1 = tl.load(Cos + table_offsets + HALF_D, mask=valid, other=0.0).to(tl.float32)
    s0 = tl.load(Sin + table_offsets, mask=valid, other=0.0).to(tl.float32)
    s1 = tl.load(Sin + table_offsets + HALF_D, mask=valid, other=0.0).to(tl.float32)

    # Round each product to BF16 before addition, as in the Torch reference.
    lo_first = (a * c0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    lo_second = ((-b) * s0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    hi_first = (b * c1).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    hi_second = (a * s1).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    lo = (lo_first + lo_second).to(tl.bfloat16, fp_downcast_rounding="rtne")
    hi = (hi_first + hi_second).to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(output_base + table_offsets, lo, mask=valid)
    tl.store(output_base + table_offsets + HALF_D, hi, mask=valid)


def run(
    x0: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return independent BF16 Q/K rotations in the frozen production layouts.

    Accept Q [1, 8, 50, 256], K [1, 1, 50, 256], and Cos/Sin [1, 50, 256]
    with the captured strides on one CUDA device. Inputs are read-only, including
    aliased Q/K views; storage offsets are handled by their tensor pointers.
    Unsupported metadata raises ValueError. Arithmetic preserves BF16 product
    rounding, and the launch uses the current stream of the input device.
    """
    expected_shapes = (
        (1, 8, 50, 256),
        (1, 1, 50, 256),
        (1, 50, 256),
        (1, 50, 256),
    )
    expected_strides = (
        (128000, 256, 2560, 1),
        (128000, 256, 2560, 1),
        (12800, 256, 1),
        (12800, 256, 1),
    )
    for tensor, shape, stride in zip((x0, x1, x2, x3), expected_shapes, expected_strides):
        if (
            tensor.shape != shape
            or tensor.stride() != stride
            or tensor.dtype != torch.bfloat16
            or tensor.device.type != "cuda"
            or tensor.device != x0.device
        ):
            raise ValueError("rotate_qk requires the captured BF16 CUDA shapes and strides")

    # Restore the caller's device on exit and leave its stream selection intact.
    with torch.cuda.device(x0.device):
        qout = torch.empty_strided(
            (1, 8, 50, 256),
            (102400, 12800, 256, 1),
            dtype=x0.dtype,
            device=x0.device,
        )
        kout = torch.empty_strided(
            (1, 1, 50, 256),
            (12800, 12800, 256, 1),
            dtype=x1.dtype,
            device=x1.device,
        )
        _rotate_qk_paired_kernel[(13, 9)](
            x0,
            x1,
            x2,
            x3,
            qout,
            kout,
            BLOCK_T=4,
            HALF_D=128,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return qout, kout
