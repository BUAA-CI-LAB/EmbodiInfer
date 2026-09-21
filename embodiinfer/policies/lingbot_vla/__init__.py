"""LingBot-VLA policy: Qwen2.5-VL-3B + Qwen2 MoT flow-matching action expert.

Importing this package registers the ``lingbot_vla`` builder with the factory."""

from .modeling_lingbot_vla import LingBotVLAPolicy

__all__ = ["LingBotVLAPolicy"]
