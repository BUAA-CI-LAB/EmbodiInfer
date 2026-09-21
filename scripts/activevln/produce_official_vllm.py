"""Produce a pinned offline-vLLM ActiveVLN reference bundle.

The offline RequestOutput exposes exact generated token ids and per-token
log-probabilities, including terminal tokens that the OpenAI chat response hides.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from embodiinfer.policies.activevln.prompt_activevln import (
    SYSTEM_PROMPT_R2R,
    actions_to_tensor,
    parse_r2r_actions,
    render_turn_text,
    user_turn_content,
)
from scripts.activevln._common import base_manifest, load_cases, save_bundle
from scripts.activevln.verify_source_pin import _load_lock, verify_checkpoint, verify_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use deterministic argmax decoding for cross-implementation parity.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("official_vllm reference generation requires CUDA")

    lock = _load_lock(Path(__file__).with_name("source_lock.json"))
    verify_source(lock, args.source_root)
    verify_checkpoint(lock, args.checkpoint)
    cases = load_cases(args.input_manifest)

    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        revision=lock["checkpoint"]["revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    llm = LLM(
        model=str(args.checkpoint),
        tokenizer=str(args.checkpoint),
        trust_remote_code=False,
        tensor_parallel_size=1,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 200},
        dtype="float32",
    )
    requested = {
        # vLLM implements greedy decoding when temperature is zero.  Do not
        # leave the checkpoint's sampling top-p active in this mode: its
        # log-probability semantics would no longer match HF argmax decoding.
        "temperature": 0.0 if args.greedy else 0.2,
        "top_p": 1.0 if args.greedy else 0.8,
        "max_tokens": args.max_new_tokens,
        "n": 1,
        "logprobs": 8,
        "repetition_penalty": 1.05,
        "seed": 0,
    }
    params = SamplingParams(**requested)
    generation_config = lock["runner_profiles"]["official_eval_r2r"]["checkpoint_generation_config"]
    effective = dict(generation_config)
    effective.update(requested)
    tensors = {}
    manifest_cases = []

    for case_index, case in enumerate(cases):
        conversation = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_R2R}]}]
        images = []
        turn_records = []
        for turn_index, turn in enumerate(case["turns"]):
            image = Image.open(turn["image"]).convert("RGB")
            images.append(image)
            conversation.append(
                {
                    "role": "user",
                    "content": user_turn_content(turn["instruction"], initial=turn_index == 0),
                }
            )
            prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=images, return_tensors="pt")
            # vLLM owns multimodal placeholder expansion.  Passing the HF
            # processor's already-expanded ``prompt_token_ids`` together with
            # PIL images asks vLLM to expand the image a second time, producing
            # a placeholder/feature count mismatch for Qwen2.5-VL.
            request = {"prompt": prompt, "multi_modal_data": {"image": images}}
            request_output = llm.generate([request], params, use_tqdm=False)[0]
            result = request_output.outputs[0]
            token_ids = torch.tensor([result.token_ids], dtype=torch.long)
            if not result.logprobs or len(result.logprobs) != token_ids.shape[1]:
                raise RuntimeError("vLLM did not return one logprob record per generated token")

            selected_logprobs = []
            top_ids = []
            top_logprobs = []
            for token_id, token_candidates in zip(result.token_ids, result.logprobs):
                selected = token_candidates.get(token_id)
                if selected is None:
                    raise RuntimeError(f"vLLM logprobs omitted selected token {token_id}")
                selected_logprobs.append(float(selected.logprob))
                pairs = sorted(
                    ((int(key), float(value.logprob)) for key, value in token_candidates.items()),
                    key=lambda item: item[1],
                    reverse=True,
                )[:8]
                pairs += [(-1, float("-inf"))] * (8 - len(pairs))
                top_ids.append([item[0] for item in pairs])
                top_logprobs.append([item[1] for item in pairs])

            response = processor.tokenizer.decode(token_ids[0], skip_special_tokens=True).strip()
            parsed = parse_r2r_actions(response)
            action_codes, action_mask = actions_to_tensor(parsed)
            delta_text = render_turn_text(processor, turn["instruction"], initial=turn_index == 0)
            delta_inputs = processor(text=[delta_text], images=[image], return_tensors="pt")
            prefix = f"cases/{case_index}/turns/{turn_index}"
            tensors[f"{prefix}/context_input_ids"] = inputs.input_ids
            tensors[f"{prefix}/context_attention_mask"] = inputs.attention_mask
            tensors[f"{prefix}/delta_input_ids"] = delta_inputs.input_ids
            tensors[f"{prefix}/delta_attention_mask"] = delta_inputs.attention_mask
            for name in ("pixel_values", "image_grid_thw"):
                if name in inputs:
                    tensors[f"{prefix}/context_{name}"] = inputs[name]
                if name in delta_inputs:
                    tensors[f"{prefix}/delta_{name}"] = delta_inputs[name]
            tensors[f"{prefix}/response_token_ids"] = token_ids
            tensors[f"{prefix}/response_action_mask"] = torch.ones_like(token_ids, dtype=torch.bool)
            tensors[f"{prefix}/response_token_logprobs"] = torch.tensor(
                [selected_logprobs], dtype=torch.float32
            )
            tensors[f"{prefix}/topk_token_ids"] = torch.tensor([top_ids], dtype=torch.long)
            tensors[f"{prefix}/topk_logprobs"] = torch.tensor([top_logprobs], dtype=torch.float32)
            tensors[f"{prefix}/parsed_action_tensor"] = action_codes
            tensors[f"{prefix}/parsed_action_mask"] = action_mask
            turn_records.append(
                {
                    "step_idx": turn_index,
                    "tensor_prefix": prefix,
                    "response_text": response,
                    "finish_reason": result.finish_reason,
                    "seq_len_before_decode": len(request_output.prompt_token_ids),
                    "seq_len_after_decode": len(request_output.prompt_token_ids) + len(result.token_ids),
                }
            )
            conversation.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        manifest_cases.append({"case_id": case["case_id"], "turns": turn_records})

    manifest = base_manifest(
        "official_vllm",
        manifest_cases,
        requested_sampling=requested,
        generation_config=generation_config,
        effective_sampling=effective,
    )
    # This is part of the numerical provenance: the visual model cannot use
    # FlashAttention at FP32, so a run explicitly forced to XFormers is not
    # interchangeable with a future bf16/FlashAttention reference.
    manifest["attention_backend"] = os.environ.get("VLLM_ATTENTION_BACKEND", "auto")
    save_bundle(args.output, manifest, tensors)


if __name__ == "__main__":
    main()
