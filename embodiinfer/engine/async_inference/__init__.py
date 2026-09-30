"""Asynchronous-inference algorithms for action-chunking VLA policies.

Two algorithms live here, both about the same physical fact: in an asynchronous
loop the chunk a policy computes is not the chunk the robot is executing. The
robot keeps moving during the forward pass, so a chunk conditioned on the
observation-time state is applied to a different state than the one it was
computed for.

:mod:`~embodiinfer.engine.async_inference.rtc`
    **Real-Time Chunking.** Reconcile the *new* chunk with the *old* one by
    holding the committed prefix and guiding the remainder toward it
    (arXiv 2506.07339).

:mod:`~embodiinfer.engine.async_inference.vlash`
    **VLASH.** Reconcile the *state* instead, by rolling the robot state forward
    under the actions already issued and conditioning on the execution-time
    state (arXiv 2512.01031).

They are complementary rather than exclusive: RTC fixes action continuity given
a stale state, VLASH fixes the state itself. Both are opt-in per request and
travel in one ``metadata["async"]`` block
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
carries ingredients (a leftover chunk, a delay, a pending action buffer) rather
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
from .serving import apply_request_async_state, batch_rtc_guidance, plan_from_request
from .vlash import (
    ActionSpaceSemantics,
    VlashStatePlan,
    apply_vlash_state_plan,
    roll_state_forward,
    roll_state_forward_projected,
)

__all__ = [
    "ASYNC_SCHEMA",
    "ActionSpaceSemantics",
    "AsyncInferencePlan",
    "PrefixAttentionSchedule",
    "RTCGuidance",
    "RTCGuidanceConfig",
    "RTCPlan",
    "VlashStatePlan",
    "action_ness",
    "apply_request_async_state",
    "apply_vlash_state_plan",
    "batch_rtc_guidance",
    "build_rtc_guidance",
    "clamp_prefix",
    "flow_noise_end",
    "guidance_strength",
    "hard_prefix_mask",
    "parse_async_plan",
    "plan_from_request",
    "prefix_weights",
    "roll_state_forward",
    "roll_state_forward_projected",
    "rtc_guided_velocity",
]
