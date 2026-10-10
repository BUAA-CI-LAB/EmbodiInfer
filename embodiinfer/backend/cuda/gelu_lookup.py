"""Device-local BF16 GELU table and fused FP4 epilogues."""

import ctypes
from typing import Literal

from ...layers.quantization import ActivationQuantizer
from .activation import GeluMulFusion
from .native import build_library


class GeluLookup:
    """Own a device-specific 128 KiB table with optional BF16 and FP4 launches."""

    def __init__(self, fusion: GeluMulFusion) -> None:
        """Load the cached library for the current CUDA device."""
        self.fusion, self.original = fusion, fusion.library
        built = build_library("gelu_lookup", strict=False, specific=True, native_fp4=False)
        self.library, self.compiler, self.command = built.library, built.compiler, built.command
        pointer = ctypes.c_void_p
        self.library.cc_init_gelu.argtypes = [pointer, pointer]
        self.library.cc_lookup_fp4.argtypes = (
            [pointer] * 6 + [ctypes.c_int] * 2 + [pointer, ctypes.c_int, ctypes.c_int, pointer]
        )
        self.library.cc_lookup_bf16.argtypes = [
            pointer,
            pointer,
            pointer,
            ctypes.c_int64,
            pointer,
            ctypes.c_int,
            pointer,
        ]
        self.library.cc_init_gelu.restype = ctypes.c_int
        self.library.cc_lookup_fp4.restype = ctypes.c_int
        self.library.cc_lookup_bf16.restype = ctypes.c_int
        self.table = fusion.table
        self.mode, self.threads, self.pointwise = "reference", 256, False

    def select(self, mode: str = "reference", threads: int = 256, *, pointwise: bool = False) -> None:
        """Bind an explicit launch before warmup and graph capture.

        Args:
            mode: Reference, arithmetic, lookup, or lookup_shared.
            threads: CUDA threads per block, one of 128, 256, or 512.
            pointwise: Also use this table launch for BF16 outputs. FP8 retains
                the fusion kernel, and reference mode retains its FP4 launch.
        """
        if mode not in ("reference", "arithmetic", "lookup", "lookup_shared"):
            raise ValueError(f"Unknown lookup mode: {mode}")
        if threads not in (128, 256, 512):
            raise ValueError("Expected 128, 256, or 512 threads")
        if mode == "arithmetic" and self.fusion.approximate != "none":
            raise ValueError("Arithmetic lookup mode implements exact GELU only")
        self.mode, self.threads, self.pointwise = mode, threads, pointwise
        self.fusion.library = self if mode != "reference" or pointwise else self.original

    def cc_geglu(self, *args) -> int:
        """Optionally replace BF16 GELU arithmetic with its device-local table."""
        if self.pointwise and args[5] == 16:
            return self.library.cc_lookup_bf16(
                *args[:3], args[4], self.table.data_ptr(), self.threads, args[-1]
            )
        return self.original.cc_geglu(*args)

    def cc_geglu_fp4(self, *args) -> int:
        """Append the table and launch selection to the original FP4 ABI."""
        if self.mode == "reference":
            return self.original.cc_geglu_fp4(*args)
        mode = {"arithmetic": 0, "lookup": 1, "lookup_shared": 2}[self.mode]
        return self.library.cc_lookup_fp4(*args[:-2], self.table.data_ptr(), self.threads, mode, args[-1])


class LookupGeluMul(GeluMulFusion):
    """Calibrated epilogues with the measured 128-thread FP4 table launch."""

    def __init__(
        self, backend: ActivationQuantizer, *, approximate: Literal["none", "tanh"] = "none"
    ) -> None:
        """Keep the table and ABI adapter alive for every prepared epilogue."""
        super().__init__(backend, approximate=approximate)
        self.lookup = GeluLookup(self)
        self.lookup.select("lookup", 128)
