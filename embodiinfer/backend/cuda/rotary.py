"""Strict FP32 rotation with BF16 storage and precomputed Torch factors."""

import ctypes

import torch

from ...layers.registry import OperatorCapabilities
from .native import build_library


class RotaryKernel:
    """Rotate tensors on the current stream without changing any model globals."""

    capabilities = OperatorCapabilities(
        ("cuda",), (torch.bfloat16,), "fp32_rope_bf16_output", rank=4, minimum_sm=(8, 0)
    )

    def __init__(self) -> None:
        """Load the architecture-specific rotation kernel outside graph capture."""
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.library = build_library("rotary", strict=True).library
        pointer = ctypes.c_void_p
        self.library.cc_rotary.argtypes = [pointer] * 4 + [
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
        ]
        self.library.cc_rotary.restype = ctypes.c_int
        self.library.cc_split_qkv_rope.argtypes = (
            [pointer] * 6 + [ctypes.c_int] * 8 + [ctypes.c_float, pointer]
        )
        self.library.cc_split_qkv_rope.restype = ctypes.c_int

    def __call__(self, inputs: torch.Tensor, sine: torch.Tensor, cosine: torch.Tensor) -> torch.Tensor:
        """Apply reference rotation using FP32 sine/cosine and a distinct BF16 output."""
        if inputs.ndim != 4 or inputs.shape[-1] % 2:
            raise ValueError("RoPE requires [batch, length, heads, even width]")
        if inputs.device != self.device or inputs.dtype != torch.bfloat16:
            raise ValueError("RoPE kernel requires CUDA BF16 inputs")
        values = inputs.contiguous()
        sine, cosine = sine.contiguous(), cosine.contiguous()
        half = values.shape[-1] // 2
        expected = (values.shape[0], values.shape[1], 1, half)
        if any(
            factor.shape != expected or factor.dtype != torch.float32 or factor.device != values.device
            for factor in (sine, cosine)
        ):
            raise ValueError("RoPE requires matching FP32 sine/cosine on the input device")
        output = torch.empty_like(values)
        with torch.cuda.device(values.device):
            status = self.library.cc_rotary(
                values.data_ptr(),
                sine.data_ptr(),
                cosine.data_ptr(),
                output.data_ptr(),
                values.numel() // 2,
                values.shape[2],
                half,
                torch.cuda.current_stream(values.device).cuda_stream,
            )
            if status:
                raise RuntimeError(f"RoPE kernel launch failed: {status}")
        return output

    def split_qkv_rope(
        self,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        batch: int,
        tokens: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        *,
        full_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_offset: int = 0,
        q_bthd: bool = False,
        q_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if q_bthd:
            q_buf = torch.empty((batch, tokens, q_heads, head_dim), device=qkv.device, dtype=torch.bfloat16)
        else:
            q_buf = torch.empty((batch, q_heads, tokens, head_dim), device=qkv.device, dtype=torch.bfloat16)
        if full_kv is not None:
            k_buf, v_buf = full_kv
            kv_total_t = k_buf.shape[2]
        else:
            k_buf = torch.empty((batch, kv_heads, tokens, head_dim), device=qkv.device, dtype=torch.bfloat16)
            v_buf = torch.empty((batch, kv_heads, tokens, head_dim), device=qkv.device, dtype=torch.bfloat16)
            kv_total_t = tokens
            kv_offset = 0
        with torch.cuda.device(qkv.device):
            status = self.library.cc_split_qkv_rope(
                qkv.data_ptr(),
                cos.data_ptr(),
                sin.data_ptr(),
                q_buf.data_ptr(),
                k_buf.data_ptr(),
                v_buf.data_ptr(),
                batch,
                tokens,
                q_heads,
                kv_heads,
                head_dim,
                kv_total_t,
                kv_offset,
                1 if q_bthd else 0,
                q_scale,
                torch.cuda.current_stream(qkv.device).cuda_stream,
            )
            if status:
                raise RuntimeError(f"split_qkv_rope kernel launch failed: {status}")
        q_out = q_buf.transpose(1, 2) if q_bthd else q_buf
        return q_out, k_buf, v_buf
