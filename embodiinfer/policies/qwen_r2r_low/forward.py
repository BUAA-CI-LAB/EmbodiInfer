"""Low policy compatibility adapter for the shared Qwen2.5-VL forward."""

from __future__ import annotations

from ...models.qwen25_vl import Qwen25VLNextTokenForward


class QwenR2RLowNextTokenForward(Qwen25VLNextTokenForward):
    """Graph-safe full Qwen vision-to-next-token forward."""


__all__ = ["QwenR2RLowNextTokenForward"]
