"""Cosmos Policy (WAM) tests.

CPU: registration + friendly errors + the model-agnostic latent inject/readout and the
diffusion schedule (no weights needed). Box (``cosmos`` mark, gated on ``EMBODIINFER_COSMOS_CKPT``
/ ``EMBODIINFER_COSMOS_VAE`` / ``EMBODIINFER_COSMOS_REF``): build from the real LIBERO Predict2-2B
checkpoint + Wan2.1 VAE and assert the self-hosted DiT + rectified-flow sampler reproduce
cosmos-policy's own inference within the bf16 floor (see docs/proposals/0006).
"""

import os

import pytest
import torch

from embodiinfer.exceptions import EmbodiInferError
from embodiinfer.models.schedulers import karras_sigmas, rectified_flow_scaling
from embodiinfer.policies import available_policies, make_policy
from embodiinfer.policies.cosmos.modeling_cosmos import _inject_vector, read_action_frame, read_value_frame


def test_cosmos_registered():
    assert "cosmos" in available_policies()


def test_cosmos_requires_checkpoint():
    with pytest.raises((ValueError, EmbodiInferError)):
        make_policy("cosmos")


def test_cosmos_action_readout_roundtrip():
    # tile-fill an action chunk into a latent frame, then read it back (mean over tiles).
    B, C, T, H, W = 1, 16, 9, 28, 28
    horizon, adim = 16, 7
    latent = torch.zeros(B, C, T, H, W)
    action = torch.randn(B, horizon * adim)
    latent = _inject_vector(latent, action, frame_idx=4)
    read = read_action_frame(latent, 4, horizon, adim)
    assert read.shape == (B, horizon, adim)
    assert torch.allclose(read.reshape(B, -1), action, atol=1e-5)


def test_cosmos_value_readout():
    latent = torch.zeros(1, 16, 9, 28, 28)
    latent[:, :, 8, :, :] = 0.5
    assert torch.allclose(read_value_frame(latent, 8), torch.tensor([0.5]), atol=1e-6)


def test_cosmos_rectified_flow_scaling():
    sigma = torch.tensor([4.0, 80.0])
    c_skip, c_out, c_in, c_noise = rectified_flow_scaling(sigma)
    t = sigma / (sigma + 1)
    assert torch.allclose(c_skip, 1 - t) and torch.allclose(c_in, 1 - t)
    assert torch.allclose(c_out, -t) and torch.allclose(c_noise, t)


def test_cosmos_karras_schedule():
    sig = karras_sigmas(4.0, 80.0, num_steps=4, rho=7.0, device="cpu")
    assert sig.shape == (5,)
    assert abs(sig[0].item() - 80.0) < 1e-3 and abs(sig[-1].item() - 4.0) < 1e-3
    assert torch.all(sig[:-1] > sig[1:])  # strictly decreasing (reverse schedule)


@pytest.mark.cosmos
def test_cosmos_matches_native_reference():
    dit = os.environ.get("EMBODIINFER_COSMOS_CKPT")
    vae = os.environ.get("EMBODIINFER_COSMOS_VAE")
    ref = os.environ.get("EMBODIINFER_COSMOS_REF")
    if not dit or not vae or not ref:
        pytest.skip(
            "set EMBODIINFER_COSMOS_CKPT + EMBODIINFER_COSMOS_VAE + EMBODIINFER_COSMOS_REF (native reference dump)"
        )
    from embodiinfer.policies.cosmos.modeling_cosmos import _build_cosmos
    from embodiinfer.policies.cosmos.processor_cosmos import CosmosBatch

    data = torch.load(ref)
    pol = _build_cosmos(dit_ckpt=dit, vae_ckpt=vae, attention="sdpa", device="cuda", dtype=torch.bfloat16)
    video = (data["video"].cuda().float() / 127.5 - 1.0).to(torch.bfloat16)
    batch = CosmosBatch(
        pixel_video=video,
        proprio=data["proprio"].cuda(),
        crossattn=data["t5emb"].cuda().to(torch.bfloat16),
        padding_mask=torch.zeros(1, 1, 224, 224).to(torch.bfloat16),
        request_ids=["c0"],
    )
    prefix = pol.encode_prefix(batch)
    latent = pol.sample_latent(prefix, data["x_sigma_max"].cuda(), 5)
    action = pol.read_action(latent)
    assert (action.cpu() - data["action"]).abs().max() < 5e-2  # bf16 cross-kernel floor
    value = pol.read_value(latent)
    assert abs(value.item() - data["value"].item()) < 5e-2
