"""Single-launch split-key decode attention for small MQA query blocks with 256-wide heads (SM100+)."""

import ctypes

import torch

from .native import build_library

_LIBRARIES: dict[int, ctypes.CDLL] = {}
_CHUNK = 64
_GROUP_ROWS = 48


def supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """BF16 [B,H,S,256] queries (H <= 16, 16..128 folded rows) over one contiguous [B,1,L,256] K/V head.

    The kernel's grid-wide barrier needs all ceil(L/64) * ceil(rows/48) * B CTAs resident, one per SM.
    """
    if not q.is_cuda or q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    batch, heads, steps, width = q.shape
    rows = heads * steps
    properties = torch.cuda.get_device_properties(q.device)
    ctas = -(-k.shape[2] // _CHUNK) * -(-rows // _GROUP_ROWS) * batch
    return (
        properties.major >= 10
        and width == 256
        and heads <= 16
        and k.shape == v.shape == (batch, 1, k.shape[2], 256)
        and 1 <= k.shape[2] <= 16 * _CHUNK
        and rows % 16 == 0
        and rows <= 128
        and ctas <= properties.multi_processor_count
        and q.stride(3) == 1
        and all(stride % 8 == 0 for stride in q.stride()[:3])
        and q.data_ptr() % 16 == 0
        and k.is_contiguous()
        and v.is_contiguous()
    )


def _library(device: torch.device) -> ctypes.CDLL:
    index = device.index if device.index is not None else torch.cuda.current_device()
    library = _LIBRARIES.get(index)
    if library is None:
        with torch.cuda.device(index):
            library = build_library("mqa_decode_attention", specific=True, native_fp4=True).library
        library.cc_mqa_decode_attention.argtypes = (
            [ctypes.c_int] * 5
            + [ctypes.c_void_p]
            + [ctypes.c_int64] * 3
            + [ctypes.c_void_p] * 2
            + [ctypes.c_int64, ctypes.c_float]
            + [ctypes.c_void_p] * 6
            + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
        )
        library.cc_mqa_decode_attention.restype = ctypes.c_int
        _LIBRARIES[index] = library
    return library


class MqaDecodeAttention:
    """Graph-stable split scratch and barrier counter for one query shape, key count and stream."""

    def __init__(self, q: torch.Tensor, k: torch.Tensor):
        batch, heads, steps, _ = q.shape
        splits = -(-k.shape[2] // _CHUNK)
        self.partial = torch.empty((batch, splits, heads * steps, 256), device=q.device, dtype=torch.float16)
        self.stats = torch.empty((batch, splits, heads * steps, 2), device=q.device, dtype=torch.float32)
        self.counter = torch.zeros(1, device=q.device, dtype=torch.int64)
        self.library = _library(q.device)

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        *,
        output: torch.Tensor | None = None,
        fp4: torch.Tensor | None = None,
        blocked: torch.Tensor | None = None,
        g_scale: float = 0.0,
        stable_keys: int = 0,
    ) -> None:
        """Write BF16 [B*S, H*256] to output, or its NVFP4 encoding to fp4/blocked (programmatic launch).

        Keys below ``stable_keys`` must not be written by the kernel launched just before this one;
        they are loaded before waiting on it.
        """
        batch, heads, steps, _ = q.shape
        status = self.library.cc_mqa_decode_attention(
            batch,
            steps * heads,
            heads,
            k.shape[2],
            stable_keys,
            q.data_ptr(),
            q.stride(0),
            q.stride(2),
            q.stride(1),
            k.data_ptr(),
            v.data_ptr(),
            k.stride(0),
            scale,
            self.partial.data_ptr(),
            self.stats.data_ptr(),
            self.counter.data_ptr(),
            None if output is None else output.data_ptr(),
            None if fp4 is None else fp4.data_ptr(),
            None if blocked is None else blocked.data_ptr(),
            g_scale,
            1,
            torch.cuda.current_stream(q.device).cuda_stream,
        )
        if status != 0:
            raise RuntimeError(f"MQA decode attention launch failed with status {status}")


def mqa_decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Return [B,1,S*H,256] BF16 attention output in step-major folded row order."""
    batch, heads, steps, width = q.shape
    output = torch.empty((batch, 1, steps * heads, width), device=q.device, dtype=q.dtype)
    MqaDecodeAttention(q, k)(q, k, v, scale, output=output)
    return output
