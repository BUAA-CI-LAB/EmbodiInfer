"""LingBot-VLA policy tests.

CPU: registration + friendly errors (no weights / no lingbotvla package needed). Box
(``lingbot_vla`` mark, gated on ``EMBODIINFER_LINGBOT_VLA_CKPT``): build from a real checkpoint,
run the self-hosted VL prefill + expert denoise, and assert bit-exact/≤ε parity vs the
RLinf ``LingbotvlaActionModel`` eager rollout (num_steps=10, flow_sde) — driven by
``dev/scripts/lingbot_*`` on the box (see docs/proposals/0005).
"""

import os

import pytest
import torch

from embodiinfer.exceptions import EmbodiInferError
from embodiinfer.policies import available_policies, make_policy


def test_lingbot_vla_registered():
    assert "lingbot_vla" in available_policies()


def test_lingbot_vla_requires_checkpoint():
    with pytest.raises((ValueError, EmbodiInferError)):
        make_policy("lingbot_vla")


@pytest.mark.lingbot_vla
def test_lingbot_vla_matches_native_reference():
    ckpt = os.environ.get("EMBODIINFER_LINGBOT_VLA_CKPT")
    ref = os.environ.get("EMBODIINFER_LINGBOT_VLA_REF")
    if not ckpt or not ref:
        pytest.skip("set EMBODIINFER_LINGBOT_VLA_CKPT + EMBODIINFER_LINGBOT_VLA_REF (native reference dump)")
    # Box parity: load the saved reference (inputs + fixed x0 + per-step noise + ref_actions),
    # run the embodiinfer self-hosted forward, assert max|Δa| within the bf16 cross-impl floor.
    data = torch.load(ref)
    pol = make_policy("lingbot_vla", checkpoint=ckpt, attention="sdpa").to("cuda").eval()
    prefix = pol.encode_prefix(data["batch"].to("cuda"))
    x = data["x0"].to("cuda")
    for t_val, dt in pol.flow_schedule(data["num_steps"]):
        t = torch.full((x.shape[0],), t_val, device="cuda", dtype=x.dtype)
        x = x + pol.denoise_step(x, t, prefix) * dt
    assert (x.cpu() - data["ref_actions"]).abs().max() < 6e-2


def test_lingbot_uses_explicit_backbone_for_weights_and_processor(monkeypatch):
    """A pinned local backbone must be honored on both sides of preprocessing."""
    from transformers import AutoProcessor

    from embodiinfer.policies.lingbot_vla import modeling_lingbot_vla as modeling
    from embodiinfer.policies.lingbot_vla.processor_lingbot_vla import LingBotVLABatch

    loaded = []

    def build(checkpoint, backbone_path):
        loaded.append((checkpoint, backbone_path))
        return tuple(torch.nn.Identity() for _ in range(4))

    processor_paths = []
    processor = object()

    def load_processor(path):
        processor_paths.append(path)
        return processor

    monkeypatch.setattr(modeling, "_build_and_load", build)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(LingBotVLABatch, "from_observations", lambda *args: args[2])
    policy = make_policy("lingbot_vla", checkpoint="/weights", backbone_path="/pinned/qwen")
    assert loaded == [("/weights", "/pinned/qwen")]
    assert policy.collate([], []) is processor
    assert processor_paths == ["/pinned/qwen"]
