"""CPU unit tests for the OpenVLA-OFT categorical action head (pure math, no weights)."""

import numpy as np
import torch

from embodiinfer.policies.openvla_oft.head import CategoricalActionHead

VOCAB, NBINS, ADIM, NCHUNK = 32000, 256, 7, 8
NTOK = ADIM * NCHUNK


def _head():
    rng = np.random.default_rng(0)
    q01 = rng.uniform(-1, 0, ADIM)
    q99 = rng.uniform(0, 1, ADIM)
    mask = np.array([True] * (ADIM - 1) + [False])
    return CategoricalActionHead(VOCAB, NBINS, ADIM, NCHUNK, q01, q99, mask)


def _logits(B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, NTOK, VOCAB + 64, generator=g)  # vocab + pad_to_multiple_of


def test_sampled_ids_are_action_bins():
    h = _head()
    idxs, lp = h.sample(
        _logits(), do_sample=True, temperature=1.6, top_k=-1, generator=torch.Generator().manual_seed(3)
    )
    assert idxs.shape == (2, NTOK) and lp.shape == (2, NTOK)
    assert torch.all(idxs >= VOCAB - NBINS) and torch.all(idxs < VOCAB)


def test_greedy_is_argmax_over_bins():
    h = _head()
    logits = _logits()
    idxs, _ = h.sample(logits, do_sample=False)
    masked = h._mask_logits(logits)
    assert torch.equal(idxs, masked.argmax(dim=-1))


def test_recompute_matches_behavior_logprob():
    # behavior log-prob (from sample) must equal recompute at the same temperature/top_k.
    h = _head()
    logits = _logits(seed=7)
    idxs, behavior = h.sample(
        logits, do_sample=True, temperature=1.6, top_k=-1, generator=torch.Generator().manual_seed(9)
    )
    recomputed = h.recompute_logprob(logits, idxs, temperature=1.6, top_k=-1)
    assert torch.allclose(behavior, recomputed, atol=0, rtol=0)


def test_tokens_to_actions_shape_and_gripper_passthrough():
    h = _head()
    idxs = torch.full((3, NTOK), VOCAB - 1, dtype=torch.long)  # id -> discretized 1 -> bin 0
    actions = h.tokens_to_actions(idxs)
    assert actions.shape == (3, NCHUNK, ADIM)
    # last action dim has mask=False -> stays normalised (bin_center[0]); others get unnorm.
    normalized = h.bin_centers[0]
    assert torch.allclose(actions[..., -1], normalized.expand_as(actions[..., -1]))


def test_recompute_logprob_is_differentiable():
    h = _head()
    logits = _logits(seed=5).requires_grad_(True)
    idxs, _ = h.sample(logits.detach(), do_sample=True, temperature=1.0, top_k=-1)
    lp = h.recompute_logprob(logits, idxs, temperature=1.0, top_k=-1)
    lp.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
