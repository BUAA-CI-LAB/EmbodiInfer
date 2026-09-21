"""Weightless sampling / scheduling math, nested under ``models``.

The outer numeric recipe that drives a network's denoise — diffusion EDM/Karras/2ab
preconditioning here (:mod:`.diffusion`), flow schedule/Euler/SDE its sibling
(:mod:`.flow`). It has no ``nn.Parameter``; an :class:`~embodiinfer.policies.decoder.ActionDecoder`
drives it (the embodiinfer analogue of a Diffusers pipeline calling ``scheduler.step``).

Home rationale: this is *not* a leaf op called inside a ``forward`` (that is
``embodiinfer/layers`` — attention/norm/rope), nor a policy (model-specific glue), so it lives
beside the networks it pairs with under ``models`` — as in vllm-omni's
``diffusion/models/schedulers/`` (Diffusers keeps a top-level ``schedulers/`` only
because it ships ~50 of them; embodiinfer does not).
"""

from .diffusion import karras_sigmas, rectified_flow_scaling, sample_2ab
from .flow import euler_step, sde_coefficients, sde_transition_mean

__all__ = [
    # diffusion (EDM / rectified-flow / Karras / 2ab)
    "rectified_flow_scaling",
    "karras_sigmas",
    "sample_2ab",
    # flow-matching (Euler / flow-SDE)
    "euler_step",
    "sde_coefficients",
    "sde_transition_mean",
]
