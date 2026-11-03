"""One-launch FP32 action output projection, Euler step and next action input projection."""

import ctypes

import torch

from .native import build_library

_LIBRARIES: dict[int, ctypes.CDLL] = {}


def _library(device: torch.device) -> ctypes.CDLL:
    index = device.index if device.index is not None else torch.cuda.current_device()
    library = _LIBRARIES.get(index)
    if library is None:
        with torch.cuda.device(index):
            library = build_library("action_bridge", specific=True).library
        pointer = ctypes.c_void_p
        library.cc_action_bridge.argtypes = (
            [pointer] * 5 + [ctypes.c_float] + [pointer] * 3 + [ctypes.c_int] * 3 + [pointer] * 2
        )
        library.cc_action_bridge.restype = ctypes.c_int
        _LIBRARIES[index] = library
    return library


class ActionBridge:
    """Flow-step bridge for FP32 action projections; owns its graph-stable barrier counter."""

    def __init__(self, in_proj: torch.nn.Linear, out_proj: torch.nn.Linear):
        if in_proj.weight.dtype != torch.float32 or out_proj.weight.dtype != torch.float32:
            raise ValueError("The action bridge reproduces FP32 action projections only")
        self.in_weight = in_proj.weight.detach().contiguous()
        self.in_bias = None if in_proj.bias is None else in_proj.bias.detach().contiguous()
        self.out_weight = out_proj.weight.detach().contiguous()
        self.out_bias = None if out_proj.bias is None else out_proj.bias.detach().contiguous()
        self.counter = torch.zeros(1, device=self.in_weight.device, dtype=torch.int64)
        self.library = _library(self.in_weight.device)

    @staticmethod
    def supports(in_proj: torch.nn.Linear, out_proj: torch.nn.Linear, state: torch.Tensor) -> bool:
        """FP32 32x1024 projections on the state's CUDA device for at most 16 action rows."""
        return (
            state.is_cuda
            and state.dtype == torch.float32
            and in_proj.weight.dtype == out_proj.weight.dtype == torch.float32
            and tuple(out_proj.weight.shape) == (32, 1024)
            and tuple(in_proj.weight.shape) == (1024, 32)
            and in_proj.weight.device == state.device
            and state.shape[-1] == 32
            and state.numel() // 32 <= 16
        )

    def __call__(
        self, hidden: torch.Tensor, x: torch.Tensor, dt: float, *, last: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return x + dt * out_proj(hidden) and, unless last, bf16(in_proj(x_next)) for the next step."""
        hidden = hidden.contiguous()
        x = x.contiguous()
        x_next = torch.empty_like(x)
        embeds = None if last else torch.empty(hidden.shape, device=hidden.device, dtype=torch.bfloat16)
        status = self.library.cc_action_bridge(
            hidden.data_ptr(),
            x.data_ptr(),
            x_next.data_ptr(),
            self.out_weight.data_ptr(),
            None if self.out_bias is None else self.out_bias.data_ptr(),
            dt,
            self.in_weight.data_ptr(),
            None if self.in_bias is None else self.in_bias.data_ptr(),
            None if embeds is None else embeds.data_ptr(),
            x.numel() // x.shape[-1],
            hidden.shape[-1],
            x.shape[-1],
            self.counter.data_ptr(),
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        if status != 0:
            raise RuntimeError(f"Action bridge launch failed with status {status}")
        return x_next, embeds
