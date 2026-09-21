"""pi0.5 adapter parity vs LeRobot (box tier: CUDA + lerobot + a checkpoint).

Marked ``pi05`` and skipped unless a checkpoint is provided via the
``VVLA_PI05_CKPT`` env var (so no checkpoint path lives in the repo). Verifies
that our ``Pi05Policy`` reproduces LeRobot's ``PI05Policy`` bit-for-bit with the
``eager`` backend, and losslessly with the fused ``sdpa`` backend (no monkeypatch).
"""

import os

import pytest
import torch

from embodiinfer.policies.config import VLAPolicyConfig
from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy
from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch

pytestmark = pytest.mark.pi05

CKPT = os.environ.get("VVLA_PI05_CKPT")
_skip = pytest.mark.skipif(
    not (CKPT and torch.cuda.is_available()),
    reason="set VVLA_PI05_CKPT and run on CUDA",
)


def _synthetic_batch(cfg, B, lang_len, device, tok_key, mask_key):
    batch = {
        k: torch.rand(B, 3, *cfg.image_resolution, device=device, dtype=torch.float32)
        for k in cfg.image_features
    }
    batch[tok_key] = torch.randint(0, 257152, (B, lang_len), device=device)
    batch[mask_key] = torch.ones(B, lang_len, dtype=torch.bool, device=device)
    batch["observation.state"] = torch.zeros(B, cfg.max_state_dim, device=device, dtype=torch.float32)
    return batch


@_skip
def test_pi05_adapter_matches_lerobot():
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    device, steps, B = "cuda", 10, 2
    ref = PI05Policy.from_pretrained(CKPT).eval().to(torch.float32).to(device)
    c = ref.config

    batch = _synthetic_batch(c, B, 48, device, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)
    images, img_masks = ref._preprocess_images(batch)
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]

    torch.manual_seed(0)
    noise = torch.randn(B, c.chunk_size, c.max_action_dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        ref_actions = ref.model.sample_actions(
            images, img_masks, tokens, masks, noise=noise.clone(), num_steps=steps
        )

    cfg = VLAPolicyConfig(
        name="pi0.5", action_dim=c.max_action_dim, action_horizon=c.chunk_size, default_num_steps=steps
    )
    pi_batch = Pi05Batch.from_lerobot_batch(ref, batch)

    def mine(attention):
        pol = Pi05Policy(cfg, checkpoint=ref, attention=attention)  # shares the loaded weights
        with torch.no_grad():
            return pol.sample_actions(pi_batch, x0=noise.clone(), num_steps=steps)

    # eager reproduces LeRobot's forced-eager path exactly (same kernels/order)
    assert (ref_actions - mine("eager")).abs().max().item() == 0.0
    # sdpa (HF-native dispatch, no monkeypatch) is a lossless fused drop-in
    assert (ref_actions - mine("sdpa")).abs().max().item() < 1e-2
