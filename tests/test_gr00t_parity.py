"""GR00T N1.7 adapter parity vs Isaac-GR00T (box tier: CUDA + weights + a reference dump).

The adapter is **fully integrated**: the action head is vendored and the backbone
is the stock transformers ``Qwen3VLForConditionalGeneration`` built from
Cosmos-Reason2-2B, all on the single embodiinfer env (transformers>=5). The ground-truth
reference, however, is produced by the official ``gr00t`` package, which pins
transformers 4.57 and drags in a training stack that conflicts with embodiinfer's env —
so it cannot be imported in-process here. Instead a companion box script
(``dev/scripts/gr00t_ref_run.py``, run in a separate ``gr00t_env``) drives the
official modules once with a fixed ``x0`` and ``torch.save``s
``{input_ids, attention_mask, pixel_values, image_grid_thw, state, embodiment_id,
x0, ref_actions, [vl_embeds, state_features]}``. This test loads that dump and
runs the embodiinfer-integrated policy over byte-identical inputs.

Two bounds are checked:

  * **end-to-end** — embodiinfer (transformers>=5) vs gr00t-native (transformers 4.57).
    With the two cross-version alignments in ``modeling_gr00t.py`` (see below) the
    backbone reproduces gr00t's select-layer features (vl_embeds cos ~0.99), so
    the end-to-end delta collapses onto the DiT floor: observed max|Δ| ≈ 1.6e-2
    (rel 0.4%) on the reference dump — bf16 arithmetic, not bit-exact (torch /
    diffusers differ across the two envs).
  * **DiT/action-head isolation** — when fed gr00t-native's *own* prefix
    (``vl_embeds`` + ``state_features``), embodiinfer's denoise loop reproduces the
    native actions to the same bf16 floor (observed max|Δ| ≈ 1.6e-2 — equal to the
    end-to-end value, confirming the backbone contributes negligibly once
    aligned). This is the part embodiinfer owns; the backbone is a run-once black-box
    prefill. Checked only when the dump carries the intermediates.

Two cross-version subtleties the ground-truth check surfaced, both fixed in
``modeling_gr00t.py`` (each was masked by DiT robustness — actions were only
~1.7% off with both bugs present):
  * **pre-/post-final-norm** — gr00t's ``backbone_features`` is the *pre*-final-norm
    residual stream (tf4.57 returns that as ``hidden_states[-1]``; tf>=5 returns
    the *post*-norm tensor there), recovered by dropping the vestigial final norm.
  * **multimodal-RoPE positions** — tf>=5's Qwen3-VL needs ``mm_token_type_ids`` to
    give image tokens their 2D grid positions; driven with a minimal input set it
    falls back to sequential 1D positions (wrong vision RoPE), so ``encode_prefix``
    computes the positions explicitly to match gr00t (tf4.57).

Gating: ``EMBODIINFER_GR00T_CKPT`` (GR00T-N1.7-3B dir), ``EMBODIINFER_COSMOS_PATH``
(Cosmos-Reason2-2B dir), ``EMBODIINFER_GR00T_REF`` (the reference dump), and CUDA.
``test_gr00t_registered`` runs in CI (no weights).
"""

import json
import os

import pytest
import torch

from embodiinfer.policies.factory import available_policies, make_policy


def test_gr00t_registered():
    """The adapter registers itself and rejects a missing checkpoint (CI-safe)."""
    assert "gr00t" in available_policies()
    with pytest.raises(ValueError):
        make_policy("gr00t")  # no checkpoint
    with pytest.raises(ValueError):
        make_policy("gr00t", checkpoint="/nonexistent")  # no cosmos_path


CKPT = os.environ.get("EMBODIINFER_GR00T_CKPT")
COSMOS = os.environ.get("EMBODIINFER_COSMOS_PATH")
REF = os.environ.get("EMBODIINFER_GR00T_REF")
_skip = pytest.mark.skipif(
    not (CKPT and COSMOS and REF and torch.cuda.is_available()),
    reason="set EMBODIINFER_GR00T_CKPT + EMBODIINFER_COSMOS_PATH + EMBODIINFER_GR00T_REF and run on CUDA",
)

_BACKBONE_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")


@_skip
@pytest.mark.gr00t
def test_gr00t_matches_native_reference():
    from embodiinfer.policies.config import VLAPolicyConfig
    from embodiinfer.policies.gr00t.modeling_gr00t import Gr00tPolicy, Gr00tPrefix
    from embodiinfer.policies.gr00t.processor_gr00t import Gr00tBatch

    dev = "cuda"
    with open(os.path.join(CKPT, "config.json")) as f:
        gcfg = json.load(f)
    dtype = getattr(torch, gcfg.get("model_dtype", "bfloat16"))
    N = gcfg["num_inference_timesteps"]
    H, A = gcfg["action_horizon"], gcfg["max_action_dim"]

    ref = torch.load(REF, map_location=dev)
    bi = {
        k: (ref[k].to(dev, dtype) if torch.is_floating_point(ref[k]) else ref[k].to(dev))
        for k in _BACKBONE_KEYS
    }
    state = ref["state"].to(dev, dtype)
    eid = ref["embodiment_id"].to(dev)
    x0 = ref["x0"].to(dev, dtype)
    ref_actions = ref["ref_actions"].to(dev).float()

    cfg = VLAPolicyConfig(
        name="gr00t",
        action_dim=A,
        action_horizon=H,
        default_num_steps=N,
        dtype=gcfg.get("model_dtype", "bfloat16"),
    )
    pol = Gr00tPolicy(cfg, checkpoint=CKPT, cosmos_path=COSMOS, attention="sdpa").to(dev).eval()
    batch = Gr00tBatch(backbone_inputs=bi, state=state, embodiment_id=eid, request_ids=["r0"])

    # ---- end-to-end: embodiinfer (tf>=5) vs gr00t-native (tf4.57), bf16 cross-env bound ----
    with torch.no_grad():
        mine = pol.sample_actions(batch, x0=x0.clone(), num_steps=N).float()
    assert (ref_actions - mine).abs().max().item() < 3e-2

    # ---- isolation: gr00t-native prefix -> embodiinfer DiT reproduces actions to bf16 ----
    if "vl_embeds" in ref and "state_features" in ref:
        attn_m = bi["attention_mask"] == 1
        image = bi["input_ids"] == pol._image_token_id
        prefix = Gr00tPrefix(
            vl_embeds=ref["vl_embeds"].to(dev, dtype),
            image_key_mask=(image & attn_m)[:, None, None, :],
            text_key_mask=(~image & attn_m)[:, None, None, :],
            state_features=ref["state_features"].to(dev, dtype),
            embodiment_id=eid,
        )
        with torch.no_grad():
            x = x0.clone()
            for t_val, dt in pol.flow_schedule(N):
                t = torch.full((x.shape[0],), t_val, device=dev, dtype=dtype)
                x = x + pol.denoise_step(x, t, prefix) * dt
        assert (ref_actions - x.float()).abs().max().item() < 3e-2


def test_gr00t_graph_buffers_follow_live_instruction_length():
    """A later prefill cannot change the allocation for an earlier queued prefix."""
    from embodiinfer.policies.gr00t.modeling_gr00t import Gr00tPolicy, Gr00tPrefix

    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    torch.nn.Module.__init__(policy)
    policy._cached_prefix_len = 99
    variants = set()
    for length in (7, 13):
        prefix = Gr00tPrefix(
            torch.randn(1, length, 4),
            torch.ones(1, 1, 1, length, dtype=torch.bool),
            torch.zeros(1, 1, 1, length, dtype=torch.bool),
            torch.randn(1, 1, 6),
            torch.tensor([2]),
        )
        variant = policy.cuda_graph_variant(prefix)
        assert variant not in variants
        variants.add(variant)
        allocated = policy.allocate_static_prefix_from_live(
            prefix,
            1,
            torch.device("cpu"),
            torch.float32,
            variant,
        )
        policy.copy_prefix_into(allocated, prefix)
        assert allocated.vl_embeds.shape == (1, length, 4)
        torch.testing.assert_close(allocated.vl_embeds, prefix.vl_embeds)
        assert allocated.vl_embeds.data_ptr() != prefix.vl_embeds.data_ptr()
        torch.testing.assert_close(allocated.text_key_mask, prefix.text_key_mask)
        with pytest.raises(ValueError, match="graph shape"):
            policy.allocate_static_prefix_from_live(
                prefix,
                1,
                torch.device("cpu"),
                torch.float32,
                length + 1,
            )
