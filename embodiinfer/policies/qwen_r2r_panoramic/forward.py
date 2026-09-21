"""Panoramic policy specialization of the shared Qwen2.5-VL forward."""

from ...models.qwen25_vl import Qwen25VLNextTokenForward


class QwenR2RPanoramicNextTokenForward(Qwen25VLNextTokenForward):
    """Panoramic policy type for the shared next-token implementation."""


__all__ = ["QwenR2RPanoramicNextTokenForward"]
