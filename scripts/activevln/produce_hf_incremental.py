"""Produce a stock-HF incremental-cache ActiveVLN reference bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from embodiinfer.policies.activevln.prompt_activevln import (
    actions_to_tensor,
    parse_r2r_actions,
    render_turn_text,
)
from scripts.activevln._common import base_manifest, load_cases, save_bundle
from scripts.activevln.verify_source_pin import _load_lock, verify_checkpoint


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
        raise RuntimeError("hf_incremental reference generation requires CUDA")

    lock = _load_lock(Path(__file__).with_name("source_lock.json"))
    verify_checkpoint(lock, args.checkpoint)
    cases = load_cases(args.input_manifest)

    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.generation.logits_process import (
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopPLogitsWarper,
    )

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
        "use_cache": True,
    }
    effective = dict(generation_config)
    effective.update(requested)
    repetition = RepetitionPenaltyLogitsProcessor(1.05)
    temperature = TemperatureLogitsWarper(0.2)
    top_p = TopPLogitsWarper(0.8)
    eos_ids = {int(x) for x in model.generation_config.eos_token_id}
    tensors = {}
    manifest_cases = []
    torch.manual_seed(0)

    for case_index, case in enumerate(cases):
        cache = None
        history_ids = torch.empty(1, 0, dtype=torch.long, device="cuda")
        next_position = 0
        turn_records = []
        for turn_index, turn in enumerate(case["turns"]):
            text = render_turn_text(processor, turn["instruction"], initial=turn_index == 0)
            image = Image.open(turn["image"]).convert("RGB")
            inputs = processor(text=[text], images=[image], return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            local_positions, _ = model.get_rope_index(
                inputs["input_ids"],
                inputs.get("image_grid_thw"),
                inputs.get("video_grid_thw"),
                None,
                inputs.get("attention_mask"),
            )
            positions = local_positions + next_position
            total_attention = torch.ones(
                1,
                history_ids.shape[1] + inputs["input_ids"].shape[1],
                dtype=torch.long,
                device="cuda",
            )
            model_inputs = dict(inputs)
            model_inputs["attention_mask"] = total_attention
            with torch.no_grad():
                output = model(
                    **model_inputs,
                    position_ids=positions,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
            cache = output.past_key_values
            history_ids = torch.cat([history_ids, inputs["input_ids"]], dim=1)
            context_input_ids = history_ids.clone()
            next_position = int(positions.max().item()) + 1
            logits = output.logits[:, -1]
            response_tokens = []
            response_logprobs = []
            top_ids = []
            top_values = []
            response_positions = []
            for _ in range(args.max_new_tokens):
                effective_logits = repetition(history_ids, logits)
                if not args.greedy:
                    effective_logits = top_p(history_ids, temperature(history_ids, effective_logits))
                log_probs = torch.log_softmax(effective_logits, dim=-1)
                token = (
                    effective_logits.argmax(dim=-1, keepdim=True)
                    if args.greedy
                    else torch.multinomial(torch.softmax(effective_logits, dim=-1), 1)
                )
                response_tokens.append(token)
                response_logprobs.append(log_probs.gather(1, token).squeeze(1))
                values, ids = log_probs.topk(min(8, log_probs.shape[-1]), dim=-1)
                top_ids.append(ids)
                top_values.append(values)
                position = torch.full((3, 1, 1), next_position, dtype=torch.long, device="cuda")
                response_positions.append(position)
                history_ids = torch.cat([history_ids, token], dim=1)
                total_attention = torch.ones_like(history_ids)
                with torch.no_grad():
                    output = model(
                        input_ids=token,
                        attention_mask=total_attention,
                        position_ids=position,
                        past_key_values=cache,
                        use_cache=True,
                        return_dict=True,
                    )
                cache = output.past_key_values
                logits = output.logits[:, -1]
                next_position += 1
                if int(token.item()) in eos_ids:
                    break

            generated = torch.cat(response_tokens, dim=1)
            token_logprobs = torch.stack(response_logprobs, dim=1)
            response = processor.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            parsed = parse_r2r_actions(response)
            action_codes, action_mask = actions_to_tensor(parsed)
            prefix = f"cases/{case_index}/turns/{turn_index}"
            tensors[f"{prefix}/context_input_ids"] = context_input_ids
            tensors[f"{prefix}/context_attention_mask"] = torch.ones_like(context_input_ids)
            tensors[f"{prefix}/delta_input_ids"] = inputs["input_ids"]
            tensors[f"{prefix}/delta_attention_mask"] = inputs["attention_mask"]
            for name in ("pixel_values", "image_grid_thw"):
                if name in inputs:
                    tensors[f"{prefix}/delta_{name}"] = inputs[name]
            tensors[f"{prefix}/position_ids"] = positions
            tensors[f"{prefix}/response_position_ids"] = torch.cat(response_positions, dim=2)
            tensors[f"{prefix}/response_token_ids"] = generated
            tensors[f"{prefix}/response_action_mask"] = torch.ones_like(generated, dtype=torch.bool)
            tensors[f"{prefix}/response_token_logprobs"] = token_logprobs
            tensors[f"{prefix}/topk_token_ids"] = torch.stack(top_ids, dim=1)
            tensors[f"{prefix}/topk_logprobs"] = torch.stack(top_values, dim=1)
            tensors[f"{prefix}/parsed_action_tensor"] = action_codes
            tensors[f"{prefix}/parsed_action_mask"] = action_mask
            turn_records.append(
                {
                    "step_idx": turn_index,
                    "tensor_prefix": prefix,
                    "response_text": response,
                    "seq_len_after_decode": int(history_ids.shape[1]),
                }
            )
        manifest_cases.append({"case_id": case["case_id"], "turns": turn_records})

    manifest = base_manifest(
        "hf_incremental",
        manifest_cases,
        requested_sampling=requested,
        generation_config=generation_config,
        effective_sampling=effective,
    )
    save_bundle(args.output, manifest, tensors)


if __name__ == "__main__":
    main()
