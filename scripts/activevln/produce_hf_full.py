"""Produce a full-history Hugging Face ActiveVLN reference bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.activevln._common import base_manifest, load_cases, save_bundle
from scripts.activevln.verify_source_pin import _load_lock, verify_checkpoint
from embodiinfer.policies.activevln.prompt_activevln import (
    SYSTEM_PROMPT_R2R,
    actions_to_tensor,
    parse_r2r_actions,
    render_turn_text,
    user_turn_content,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="disable sampling for exact-token parity; sampled is the historical default",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("hf_full reference generation requires CUDA")

    lock = _load_lock(Path(__file__).with_name("source_lock.json"))
    verify_checkpoint(lock, args.checkpoint)
    cases = load_cases(args.input_manifest)

    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    revision = lock["checkpoint"]["revision"]
    processor = AutoProcessor.from_pretrained(
        args.checkpoint, revision=revision, local_files_only=True, trust_remote_code=False
    )
    model = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.checkpoint,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype="auto",
        )
        .cuda()
        .eval()
    )
    generation_config = model.generation_config.to_dict()
    requested = {
        "do_sample": not args.greedy,
        "temperature": 0.2,
        "top_p": 0.8,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": 1.05,
    }
    effective = dict(generation_config)
    effective.update(requested)
    manifest_cases = []
    tensors = {}
    torch.manual_seed(0)

    for case_index, case in enumerate(cases):
        conversation = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_R2R}]}]
        images = []
        turn_records = []
        for turn_index, turn in enumerate(case["turns"]):
            images.append(Image.open(turn["image"]).convert("RGB"))
            conversation.append(
                {
                    "role": "user",
                    "content": user_turn_content(turn["instruction"], initial=turn_index == 0),
                }
            )
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors="pt")
            delta_text = render_turn_text(processor, turn["instruction"], initial=turn_index == 0)
            delta_inputs = processor(text=[delta_text], images=[images[-1]], return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            delta_inputs = {key: value.cuda() for key, value in delta_inputs.items()}
            position_ids, rope_deltas = model.get_rope_index(
                inputs["input_ids"],
                inputs.get("image_grid_thw"),
                inputs.get("video_grid_thw"),
                None,
                inputs.get("attention_mask"),
            )
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    **requested,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            prompt_len = inputs["input_ids"].shape[1]
            generated = output.sequences[:, prompt_len:]
            transition = model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            )
            score_stack = torch.stack(output.scores, dim=1)
            score_logprobs = torch.log_softmax(score_stack, dim=-1)
            top_values, top_ids = score_logprobs.topk(min(8, score_logprobs.shape[-1]), dim=-1)
            response = processor.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            parsed = parse_r2r_actions(response)
            action_codes, action_mask = actions_to_tensor(parsed)

            prefix = f"cases/{case_index}/turns/{turn_index}"
            tensors[f"{prefix}/context_input_ids"] = inputs["input_ids"]
            tensors[f"{prefix}/context_attention_mask"] = inputs["attention_mask"]
            tensors[f"{prefix}/delta_input_ids"] = delta_inputs["input_ids"]
            tensors[f"{prefix}/delta_attention_mask"] = delta_inputs["attention_mask"]
            for name in ("pixel_values", "image_grid_thw"):
                if name in inputs:
                    tensors[f"{prefix}/context_{name}"] = inputs[name]
                if name in delta_inputs:
                    tensors[f"{prefix}/delta_{name}"] = delta_inputs[name]
            tensors[f"{prefix}/position_ids"] = position_ids
            tensors[f"{prefix}/rope_deltas"] = rope_deltas
            tensors[f"{prefix}/response_token_ids"] = generated
            tensors[f"{prefix}/response_action_mask"] = torch.ones_like(generated, dtype=torch.bool)
            tensors[f"{prefix}/response_token_logprobs"] = transition
            tensors[f"{prefix}/topk_token_ids"] = top_ids
            tensors[f"{prefix}/topk_logprobs"] = top_values
            tensors[f"{prefix}/parsed_action_tensor"] = action_codes
            tensors[f"{prefix}/parsed_action_mask"] = action_mask
            turn_records.append(
                {
                    "step_idx": turn_index,
                    "tensor_prefix": prefix,
                    "response_text": response,
                    "seq_len_before_decode": prompt_len,
                    "seq_len_after_decode": int(output.sequences.shape[1]),
                }
            )
            conversation.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        manifest_cases.append({"case_id": case["case_id"], "turns": turn_records})

    manifest = base_manifest(
        "hf_full",
        manifest_cases,
        requested_sampling=requested,
        generation_config=generation_config,
        effective_sampling=effective,
    )
    save_bundle(args.output, manifest, tensors)


if __name__ == "__main__":
    main()
