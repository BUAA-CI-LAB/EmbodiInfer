"""Text encoders (the diffusion ``text_encoder`` slot) — vendored leaves.

A text encoder turns an instruction string into the cross-attention context a video
DiT conditions on. Like the VAE, it is a **vendored perception leaf** (runs once per
unique instruction, cacheable), not an op-level self-hosted forward. Engine-agnostic
(never import ``embodiinfer.policies`` / ``embodiinfer.engine``).
"""

from .t5 import T5TextEncoder

__all__ = ["T5TextEncoder"]
