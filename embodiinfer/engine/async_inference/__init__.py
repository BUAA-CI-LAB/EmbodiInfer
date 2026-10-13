"""RTC asynchronous inference for action-chunking VLA policies.

:mod:`~embodiinfer.engine.async_inference.rtc`
    **Real-Time Chunking.** Reconcile the *new* chunk with the *old* one by
    holding the committed prefix and guiding the remainder toward it
    (arXiv 2506.07339).

RTC is opt-in per request and travels in one ``metadata["async"]`` block
(:data:`~embodiinfer.engine.async_inference.contracts.ASYNC_SCHEMA`), so a
request without the block behaves exactly as it did before this package existed.

Layering
--------
This is an engine-layer feature domain. It depends only on ``torch``, on the
public :class:`~embodiinfer.policies.base.FlowVLAPolicy` velocity contract, and
on plain mappings of numbers — never on a model name, a checkpoint layout, or a
robot. That is what lets the same arithmetic be reused by pi0.5, GR00T and
LingBot-VLA decoders, and mirrored by EmbodiRun's execution-side scheduler,
which is a separate process behind a versioned HTTP/WirelessComm API.

What these algorithms need from the execution side is *timing knowledge that
only the execution side has*: how many control steps of inference latency
elapsed, and which actions are still committed. That is why the wire contract
carries ingredients (a leftover chunk and a delay) rather
than a verdict, and why the same golden vectors are asserted on both sides.
"""

from .contracts import ASYNC_SCHEMA, AsyncInferencePlan, RTCPlan, parse_async_plan
from .rtc import (
    PrefixAttentionSchedule,
    RTCGuidance,
    RTCGuidanceConfig,
    action_ness,
    build_rtc_guidance,
    clamp_prefix,
    flow_noise_end,
    guidance_strength,
    hard_prefix_mask,
    prefix_weights,
    rtc_guided_velocity,
)
from .serving import batch_rtc_guidance, plan_from_request

__all__ = [
    "ASYNC_SCHEMA",
    "AsyncInferencePlan",
    "PrefixAttentionSchedule",
    "RTCGuidance",
    "RTCGuidanceConfig",
    "RTCPlan",
    "action_ness",
    "batch_rtc_guidance",
    "build_rtc_guidance",
    "clamp_prefix",
    "flow_noise_end",
    "guidance_strength",
    "hard_prefix_mask",
    "parse_async_plan",
    "plan_from_request",
    "prefix_weights",
    "rtc_guided_velocity",
]
