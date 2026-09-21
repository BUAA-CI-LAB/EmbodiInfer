"""Reusable, engine-agnostic model components (NOT policies).

This layer holds weight-bearing generative networks organized by model family — the
video diffusion transformers (:mod:`.video_dit`) and the video VAE tokenizers
(:mod:`.video_vae`). They are the embodiinfer analogue of HuggingFace Diffusers' ``models/``:
pure ``forward``/``encode``/``decode`` networks with their own config + checkpoint,
that a :class:`~embodiinfer.policies.base.VLAPolicy` *composes* (the Diffusers ``pipeline`` /
LeRobot ``policy`` role). The hard rule that keeps ``models`` and ``policies`` from
overlapping: **nothing under ``embodiinfer/models`` may import ``embodiinfer.policies`` or
``embodiinfer.engine``, or reference ``Observation`` / ``ActionChunk``** — a model component
is a leaf in the dependency graph, driven by a policy, never the driver.

Weightless op-math (attention kernels) lives in ``embodiinfer/layers`` (the vLLM
``model_executor/layers`` role). Weightless *sampling* math — the outer numeric loop
that drives a network's denoise (diffusion EDM/Karras/2ab, flow schedule/Euler/SDE) —
lives here too under :mod:`.schedulers` (nested under ``models`` rather than a top-level
dir, as in vllm-omni's ``diffusion/models/schedulers/``): it is not a leaf op called
inside a ``forward`` (that is ``layers``), but the recipe an ``ActionDecoder`` follows.

Only the torch-only ``video_dit`` family is re-exported here; the ``video_vae``
tokenizers pull an extra dependency (``einops``, the ``cosmos`` group), so import them
directly (``from embodiinfer.models.video_vae import WanVAE``) to keep the ``video_dit`` import
path dependency-light.
"""

from .video_dit import CosmosPredict2DiT, VideoDiT, available_video_dit, get_video_dit, register_video_dit

__all__ = [
    "VideoDiT",
    "CosmosPredict2DiT",
    "register_video_dit",
    "get_video_dit",
    "available_video_dit",
]
