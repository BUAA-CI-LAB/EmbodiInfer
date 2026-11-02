"""Plan startup graph shapes from prompts alone, without navigation or model outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from embodiinfer.backend.triton.gqa_split_attention import PACKED_BLOCK_M
from embodiinfer.policies.streamvln.prompt import render_streamvln_prompt, tokenize_multimodal_prompt


def plan_prefill_shapes(
    processor: Any,
    instructions: Sequence[str],
    *,
    vision_tokens: int,
    batch_size: int,
    token_bucket: int,
    group_size: int,
    workspace_tokens: int,
    kv_pool_tokens: int,
    max_new_tokens: int,
    max_graphs: int = 2048,
) -> dict[str, Any]:
    """Cover initial, ordinary and slow-window prompts, including mixed rows and tails.

    Each graph gets one attainable representative of its tensor-shape key.
    Positions and episode histories are runtime inputs, not census inputs.
    Exact-length planning is bounded to B1; larger batches require buckets to
    avoid enumerating a combinatorial set of attention-program shapes.
    Insufficient KV or graph budget raises rather than silently losing coverage.
    """
    if not instructions or any(not isinstance(text, str) or not text.strip() for text in instructions):
        raise ValueError("prefill planning requires nonempty instructions")
    if any(
        type(value) is not int or value < 1
        for value in (vision_tokens, batch_size, token_bucket, group_size, max_graphs)
    ):
        raise ValueError("prefill planning dimensions must be positive integers")
    if batch_size > 1 and token_bucket == 1:
        raise ValueError("multi-row shape census requires token buckets")
    per_block = PACKED_BLOCK_M // group_size
    if per_block < 1:
        raise ValueError("GQA group does not fit a packed query program")
    phases: dict[str, set[int]] = {"initial": set(), "ordinary": set(), "slow_window": set()}
    for instruction in sorted(set(instructions)):
        for phase, window_start, memory in (
            ("initial", True, False),
            ("ordinary", False, False),
            ("slow_window", True, True),
        ):
            prompt = render_streamvln_prompt(
                instruction,
                window_start=window_start,
                include_memory=memory,
                vision_phrase=processor.vision_phrase,
            )
            ids = tokenize_multimodal_prompt(prompt, processor.tokenizer)
            if int((ids == -200).sum()) != 1 or int((ids == -300).sum()) != int(memory):
                raise ValueError("prompt sentinels differ from the projected-token accounting")
            length = (
                len(ids) - 1 - int(memory) + vision_tokens * (1 + (processor.num_history if memory else 0))
            )
            if length + max_new_tokens > workspace_tokens:
                raise ValueError("prompt and generation reserve exceed the per-row workspace")
            phases[phase].add(length)
    lengths = sorted(set().union(*phases.values()))
    states: dict[int, tuple[int, ...]] = {0: ()}
    representatives = {}
    for count in range(1, batch_size + 1):
        following = {}
        for total, rows in states.items():
            for length in lengths:
                combined = total + length
                if combined + count * max_new_tokens + batch_size - count > kv_pool_tokens:
                    raise ValueError("declared prompt combinations exceed the shared KV budget")
                following.setdefault(combined, (*rows, length))
        for total, rows in following.items():
            physical = -(-total // token_bucket) * token_bucket
            programs = (-(-physical // per_block) + count - 1) if token_bucket > 1 else -(-total // per_block)
            representatives.setdefault((count, physical, programs), list(rows))
        if len(representatives) > max_graphs:
            raise ValueError(f"prefill census exceeds the declared {max_graphs}-graph startup budget")
        states = following
    keys = sorted(representatives)
    return dict(
        vision_tokens_per_frame=vision_tokens,
        instructions=len(instructions),
        phase_lengths={phase: sorted(values) for phase, values in phases.items()},
        token_bucket=token_bucket,
        graph_shapes=len(keys),
        shape_keys=[list(key) for key in keys],
        shapes=[representatives[key] for key in keys],
        scope="Prompt lengths only; zero embeddings; no model output, action, trajectory or outcome used",
    )
