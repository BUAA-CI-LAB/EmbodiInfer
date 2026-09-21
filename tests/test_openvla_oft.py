"""OpenVLA-OFT policy tests.

CPU: registration + friendly errors (no weights needed). Box (``openvla_oft`` mark,
gated on ``EMBODIINFER_OPENVLA_OFT_CKPT``): build from a real checkpoint and run the full
single forward, asserting the action tokens land in the 256-bin action-token range.
The bit-exact parity vs the canonical model is driven by ``dev/scripts/parity_oft_e2e.py``
on the box (vision max|Δ|=0, 56/56 action-token argmax match).
"""

import os

import pytest
import torch

from embodiinfer.exceptions import EmbodiInferError
from embodiinfer.policies import available_policies, make_policy


def test_openvla_oft_registered():
    assert "openvla_oft" in available_policies()


def test_openvla_oft_requires_checkpoint():
    with pytest.raises((ValueError, EmbodiInferError)):
        make_policy("openvla_oft")


@pytest.mark.openvla_oft
def test_openvla_oft_forward_produces_action_tokens():
    ckpt = os.environ.get("EMBODIINFER_OPENVLA_OFT_CKPT")
    if not ckpt:
        pytest.skip("set EMBODIINFER_OPENVLA_OFT_CKPT to a real OpenVLA-OFT checkpoint")
    from embodiinfer.types import Observation

    pol = make_policy("openvla_oft", checkpoint=ckpt, attention="eager").to("cuda").eval()
    obs = Observation(
        images=torch.rand(3, 224, 224),
        state=torch.zeros(8),
        instruction_tokens=torch.zeros(1, dtype=torch.long),
        instruction="pick up the black bowl",
    )
    batch = pol.collate([obs], ["r0"]).to("cuda", torch.float32)
    prefix = pol.encode_prefix(batch)
    assert prefix.action_logits.shape == (1, pol.n_tokens, pol.vocab_size + 64)
    idxs, _ = pol.head.sample(prefix.action_logits, do_sample=False)
    lo = pol.vocab_size - pol.n_action_bins
    assert torch.all(idxs >= lo) and torch.all(idxs < pol.vocab_size)
    actions = pol.head.tokens_to_actions(idxs)
    assert actions.shape == (1, pol.num_action_chunks, pol.action_dim)
