"""GPU-gated numerical parity against pinned ActiveVLN reference bundles."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.gpu
@pytest.mark.activevln
@torch.inference_mode()
def test_activevln_incremental_logprobs_match_reference():
    checkpoint = os.environ.get("VVLA_ACTIVEVLN_CKPT")
    reference = os.environ.get("VVLA_ACTIVEVLN_REF")
    if not checkpoint or not reference or not torch.cuda.is_available():
        pytest.skip("set VVLA_ACTIVEVLN_CKPT + VVLA_ACTIVEVLN_REF and run on CUDA")

    from safetensors.torch import load_file
    from scripts.activevln.schema import validate_bundle

    from embodiinfer.policies import make_policy
    from embodiinfer.policies.activevln.cache_activevln import ActiveVLNMemory
    from embodiinfer.policies.activevln.modeling_activevln import (
        ACTIVEVLN_REVISION,
        ActiveVLNPrefix,
        ARRecomputeState,
    )
    from embodiinfer.policies.activevln.processor_activevln import ProcessedTurn
    from embodiinfer.policies.activevln.prompt_activevln import actions_to_tensor, parse_r2r_actions

    manifest = validate_bundle(Path(reference))
    if manifest["producer"] != "hf_incremental":
        pytest.skip("embodiinfer incremental parity consumes an hf_incremental reference bundle")
    tensors = load_file(Path(reference) / "tensors.safetensors", device="cuda")
    policy = (
        make_policy(
            "activevln",
            checkpoint=checkpoint,
            revision=ACTIVEVLN_REVISION,
            # ``hf_incremental_generated_v2`` is a sampled reference
            # (temperature/top-p/repetition penalty).  Recomputing its
            # behaviour log-probabilities must retain those transforms;
            # deterministic argmax parity is covered independently below.
            do_sample=True,
        )
        .cuda()
        .eval()
    )

    # This is a numerical inference test, not the differentiable RL recompute
    # path.  The inference-mode decorator prevents the growing-KV token loop
    # from retaining a full autograd graph for every decoder layer.
    for case_index, case in enumerate(manifest["cases"]):
        memory = None
        for turn_index, turn_record in enumerate(case["turns"]):
            prefix = f"cases/{case_index}/turns/{turn_index}"
            processed = ProcessedTurn(
                input_ids=tensors[f"{prefix}/delta_input_ids"],
                attention_mask=tensors[f"{prefix}/delta_attention_mask"],
                pixel_values=tensors[f"{prefix}/delta_pixel_values"],
                image_grid_thw=tensors[f"{prefix}/delta_image_grid_thw"],
                serialized_prompt="",
                prompt_sha256="reference",
            )
            embeds = policy._embed_turn(processed)
            positions = tensors[f"{prefix}/position_ids"]
            old_kv = None if memory is None else memory.visible_kv()
            hidden, new_kv = policy._forward_chunk(embeds, positions, old_kv)
            if memory is None:
                memory = ActiveVLNMemory.from_chunk(
                    new_kv,
                    processed.input_ids,
                    processed.attention_mask,
                    positions,
                    max_length=policy.max_context,
                )
            else:
                memory.append_chunk(
                    new_kv,
                    processed.input_ids,
                    processed.attention_mask,
                    positions,
                )
            active_prefix = ActiveVLNPrefix(memory, policy._lm_head(hidden[:, -1]))
            token_ids = tensors[f"{prefix}/response_token_ids"]
            action_mask = tensors[f"{prefix}/response_action_mask"].bool()
            recomputed = policy.decoder.recompute_logprob(
                active_prefix,
                ARRecomputeState(token_ids, action_mask),
                1,
                0.0,
            )
            expected = tensors[f"{prefix}/response_token_logprobs"]
            torch.testing.assert_close(recomputed.float(), expected.float(), atol=1e-4, rtol=1e-4)

            response_text = turn_record["response_text"]
            action_tensor, parsed_mask = actions_to_tensor(parse_r2r_actions(response_text))
            torch.testing.assert_close(
                action_tensor,
                tensors[f"{prefix}/parsed_action_tensor"].cpu(),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                parsed_mask,
                tensors[f"{prefix}/parsed_action_mask"].cpu(),
                atol=0,
                rtol=0,
            )
            for i in range(token_ids.shape[1]):
                memory, _ = policy.append_token(memory, token_ids[:, i : i + 1])


@pytest.mark.gpu
@pytest.mark.activevln
@torch.inference_mode()
def test_activevln_greedy_raw_observations_match_reference():
    """Exercise processor + decoder, not only teacher-forced re-scoring."""
    checkpoint = os.environ.get("VVLA_ACTIVEVLN_CKPT")
    reference = os.environ.get("VVLA_ACTIVEVLN_GREEDY_REF")
    input_manifest = os.environ.get("VVLA_ACTIVEVLN_INPUTS")
    if not checkpoint or not reference or not input_manifest or not torch.cuda.is_available():
        pytest.skip("set checkpoint, greedy reference, input manifest, and run on CUDA")

    from PIL import Image
    from safetensors.torch import load_file
    from scripts.activevln.schema import validate_bundle

    from embodiinfer.policies import make_policy
    from embodiinfer.policies.activevln.modeling_activevln import ACTIVEVLN_REVISION
    from embodiinfer.types import Observation

    manifest = validate_bundle(Path(reference))
    if manifest["producer"] != "hf_incremental" or manifest["sampling"]["requested"].get("do_sample"):
        pytest.skip("greedy processor/decode parity needs an hf_incremental --greedy bundle")
    input_cases = json.loads(Path(input_manifest).read_text())["cases"]
    assert [case["case_id"] for case in input_cases] == [case["case_id"] for case in manifest["cases"]]
    tensors = load_file(Path(reference) / "tensors.safetensors", device="cuda")
    policy = (
        make_policy(
            "activevln",
            checkpoint=checkpoint,
            revision=ACTIVEVLN_REVISION,
            do_sample=False,
            max_new_tokens=64,
        )
        .cuda()
        .eval()
    )

    for case_index, (input_case, reference_case) in enumerate(
        zip(input_cases, manifest["cases"], strict=True)
    ):
        memory = None
        for turn_index, (input_turn, turn_record) in enumerate(
            zip(input_case["turns"], reference_case["turns"], strict=True)
        ):
            image = Image.open(input_turn["image"]).convert("RGB")
            image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
            observation = Observation(
                images=image_tensor,
                state=torch.empty(0),
                instruction_tokens=torch.empty(0, dtype=torch.long),
                instruction=input_turn["instruction"],
                env_id=case_index,
            )
            prefix_name = f"cases/{case_index}/turns/{turn_index}"

            processed = policy._processor.process_turn(observation, initial=memory is None)
            torch.testing.assert_close(
                processed.input_ids.cuda(), tensors[f"{prefix_name}/delta_input_ids"], atol=0, rtol=0
            )
            torch.testing.assert_close(
                processed.image_grid_thw.cuda(),
                tensors[f"{prefix_name}/delta_image_grid_thw"],
                atol=0,
                rtol=0,
            )

            prefix = policy.encode_prefix(
                policy.collate([observation], [f"{case_index}:{turn_index}"]), memory
            )
            torch.testing.assert_close(
                prefix.memory.token_ids,
                tensors[f"{prefix_name}/context_input_ids"],
                atol=0,
                rtol=0,
            )
            expected_positions = tensors[f"{prefix_name}/position_ids"]
            torch.testing.assert_close(
                prefix.memory.position_ids[:, :, -expected_positions.shape[-1] :],
                expected_positions,
                atol=0,
                rtol=0,
            )

            decoded = policy.decoder.decode(None, prefix, 1, 1, None)
            trace = decoded.traces[0]
            expected_tokens = tensors[f"{prefix_name}/response_token_ids"]
            torch.testing.assert_close(trace.token_ids[None], expected_tokens, atol=0, rtol=0)
            torch.testing.assert_close(
                decoded.actions[0], tensors[f"{prefix_name}/parsed_action_tensor"], atol=0, rtol=0
            )
            torch.testing.assert_close(
                trace.meta["parsed_action_mask"],
                tensors[f"{prefix_name}/parsed_action_mask"].cpu(),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                trace.token_logprobs[None],
                tensors[f"{prefix_name}/response_token_logprobs"],
                atol=2e-4,
                rtol=0,
            )
            assert decoded.next_memory.seq_len == turn_record["seq_len_after_decode"]
            memory = decoded.next_memory


@pytest.mark.gpu
@pytest.mark.activevln
@torch.inference_mode()
def test_activevln_true_weight_incremental_prefix_sharing_and_logprob():
    """Cover incremental parity, prefix sharing and logprob recomputation."""
    checkpoint = os.environ.get("VVLA_ACTIVEVLN_CKPT")
    input_manifest = os.environ.get("VVLA_ACTIVEVLN_INPUTS")
    if not checkpoint or not input_manifest or not torch.cuda.is_available():
        pytest.skip("set VVLA_ACTIVEVLN_CKPT + VVLA_ACTIVEVLN_INPUTS and run on CUDA")

    from PIL import Image

    from embodiinfer.policies import make_policy
    from embodiinfer.policies.activevln.modeling_activevln import (
        ACTIVEVLN_REVISION,
        ActiveVLNPrefix,
        build_mrope_position_ids,
    )
    from embodiinfer.types import Observation

    turn = json.loads(Path(input_manifest).read_text())["cases"][0]["turns"][0]
    image = Image.open(turn["image"]).convert("RGB")
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    observation = Observation(
        images=pixels,
        state=torch.empty(0),
        instruction_tokens=torch.empty(0, dtype=torch.long),
        instruction=turn["instruction"],
        env_id=0,
    )
    policy = (
        make_policy(
            "activevln",
            checkpoint=checkpoint,
            revision=ACTIVEVLN_REVISION,
            do_sample=False,
            max_new_tokens=4,
        )
        .cuda()
        .eval()
    )
    batch = policy.collate([observation], ["contract"]).to("cuda", torch.float32)

    # Same weights/inputs, but different query shapes. Exact argmax is the
    # semantic gate; the calibrated FP32 CUDA reduction-noise ceiling is 5e-4.
    processed = policy._processor.process_turn(observation, initial=True).to("cuda", torch.float32)
    embeds = policy._embed_turn(processed)
    positions, _ = build_mrope_position_ids(
        processed.input_ids,
        processed.image_grid_thw,
        vision_start_token_id=int(policy.qwen.config.vision_start_token_id),
        image_token_id=int(policy.qwen.config.image_token_id),
        spatial_merge_size=int(policy.qwen.config.vision_config.spatial_merge_size),
    )
    split = max(1, embeds.shape[1] // 2)
    full_hidden, _ = policy._forward_chunk(embeds, positions, None)
    _, first_kv = policy._forward_chunk(embeds[:, :split], positions[:, :, :split], None)
    incremental_hidden, _ = policy._forward_chunk(embeds[:, split:], positions[:, :, split:], first_kv)
    full_logits = policy._lm_head(full_hidden[:, split:]).float()
    incremental_logits = policy._lm_head(incremental_hidden).float()
    torch.testing.assert_close(full_logits, incremental_logits, atol=5e-4, rtol=0)
    assert torch.equal(full_logits.argmax(-1), incremental_logits.argmax(-1))

    # L1 sharing computes the multimodal prefix once and clones only the cache.
    shared = policy.encode_prefix(batch, None)
    expanded = shared.expand(3)
    independents = [policy.encode_prefix(batch, None) for _ in range(3)]
    for branch, independent in zip(expanded.branches, independents, strict=True):
        torch.testing.assert_close(branch.next_logits, independent.next_logits, atol=0, rtol=0)
        branch_result = policy.decoder.decode(None, branch, 1, 1, None)
        independent_result = policy.decoder.decode(None, independent, 1, 1, None)
        torch.testing.assert_close(
            branch_result.recompute_state.token_ids,
            independent_result.recompute_state.token_ids,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(branch_result.actions, independent_result.actions, atol=0, rtol=0)

    original = ActiveVLNPrefix(shared.memory.fork(), shared.next_logits.clone())
    decoded = policy.decoder.decode(
        None,
        ActiveVLNPrefix(shared.memory.fork(), shared.next_logits.clone()),
        1,
        1,
        None,
    )
    recomputed = policy.decoder.recompute_logprob(original, decoded.recompute_state, 1, 0.0)
    action_mask = decoded.recompute_state.action_mask.to(dtype=torch.bool)
    torch.testing.assert_close(
        decoded.behavior_logprob[action_mask],
        recomputed[action_mask],
        atol=0,
        rtol=0,
    )


@pytest.mark.gpu
@pytest.mark.activevln
def test_activevln_true_weight_local_grpo_health_step():
    checkpoint = os.environ.get("VVLA_ACTIVEVLN_CKPT")
    input_manifest = os.environ.get("VVLA_ACTIVEVLN_INPUTS")
    if not checkpoint or not input_manifest or not torch.cuda.is_available():
        pytest.skip("set VVLA_ACTIVEVLN_CKPT + VVLA_ACTIVEVLN_INPUTS and run on CUDA")

    from PIL import Image

    from embodiinfer.engine import EngineCore
    from embodiinfer.engine.config import EngineConfig
    from embodiinfer.engine.rollout import GenerationBackend
    from embodiinfer.engine.rollout.demo import GRPOConfig, GRPOTrainer
    from embodiinfer.policies import make_policy
    from embodiinfer.policies.activevln.modeling_activevln import ACTIVEVLN_REVISION
    from embodiinfer.types import Observation

    turn = json.loads(Path(input_manifest).read_text())["cases"][0]["turns"][0]
    image = Image.open(turn["image"]).convert("RGB")
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    observation = Observation(
        pixels,
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        instruction=turn["instruction"],
        env_id="grpo-real",
    )
    policy = (
        make_policy(
            "activevln",
            checkpoint=checkpoint,
            revision=ACTIVEVLN_REVISION,
            do_sample=True,
            max_new_tokens=4,
            temperature=1.0,
            top_p=1.0,
        )
        .cuda()
        .eval()
    )
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.qwen.lm_head.parameters():
        parameter.requires_grad_(True)
    core = EngineCore(
        policy,
        EngineConfig(
            device="cuda",
            dtype="auto",
            max_batch_size=1,
            use_cuda_graph=False,
        ),
    )
    backend = GenerationBackend(core)

    class RankReward:
        def __call__(self, observations, actions):
            assert len(observations) == 1 and actions.shape[:2] == (1, 2)
            return torch.tensor([[1.0, 0.0]])

    optimizer = torch.optim.Adam(policy.qwen.lm_head.parameters(), lr=1e-7)
    trainer = GRPOTrainer(
        backend,
        RankReward(),
        GRPOConfig(group_size=2, lr=1e-7, num_steps=1, max_grad_norm=1.0),
        optimizer=optimizer,
    )
    torch.manual_seed(0)
    stats = trainer.step([observation])
    assert all(np.isfinite(value) for value in stats.values())
    assert stats["ratio_mean"] == pytest.approx(1.0, abs=1e-6)
    assert stats["grad_norm"] > 0
    assert backend.weight_sync.policy_version == 1
    assert not core.has_session_state()
