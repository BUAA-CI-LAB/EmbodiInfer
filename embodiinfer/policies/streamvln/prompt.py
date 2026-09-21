"""Deterministic StreamVLN Habitat prompt and symbolic action protocol."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from ...models.streamvln.backbone import IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX

IMAGE_TOKEN = "<image>"
MEMORY_TOKEN = "<memory>"

STOP = 0
FORWARD = 1
LEFT = 2
RIGHT = 3

VISION_PHRASE = "you can see "
SYSTEM_TURN = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
BASE_INSTRUCTION = (
    "You are an autonomous navigation assistant. Your task is to {instruction} "
    "Devise an action sequence to follow the instruction using the four actions: "
    "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 "
    "centimeters, or STOP."
)


@dataclass(frozen=True)
class ParsedActions:
    actions: torch.Tensor
    mask: torch.Tensor
    invalid: bool = False
    truncated: bool = False


def render_streamvln_prompt(
    instruction: str,
    *,
    window_start: bool,
    include_memory: bool,
    vision_phrase: str = VISION_PHRASE,
) -> str:
    """Render the official Qwen chat turns with a deterministic vision phrase."""

    if not instruction.strip():
        raise ValueError("StreamVLN requires a non-empty navigation instruction")
    if not vision_phrase.endswith(" "):
        raise ValueError("vision_phrase must retain its trailing space")
    if include_memory and not window_start:
        raise ValueError("slow memory is inserted only at a fast-window boundary")

    if window_start:
        content = BASE_INSTRUCTION.format(instruction=instruction)
        if include_memory:
            content += f" You have visited these areas {MEMORY_TOKEN}."
        content += f" {vision_phrase}{IMAGE_TOKEN}."
        return f"{SYSTEM_TURN}<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n"
    return f"<|im_start|>user\n{vision_phrase}{IMAGE_TOKEN}.<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n"


def tokenize_multimodal_prompt(prompt: str, tokenizer: Any) -> torch.Tensor:
    """Tokenize text while preserving image and memory as negative sentinels."""

    parts = re.split(f"({re.escape(IMAGE_TOKEN)}|{re.escape(MEMORY_TOKEN)})", prompt)
    token_ids: list[int] = []
    for part in parts:
        if not part:
            continue
        if part == IMAGE_TOKEN:
            token_ids.append(IMAGE_TOKEN_INDEX)
            continue
        if part == MEMORY_TOKEN:
            token_ids.append(MEMORY_TOKEN_INDEX)
            continue
        encoded = tokenizer(part, add_special_tokens=False)
        ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        token_ids.extend(ids)
    return torch.tensor(token_ids, dtype=torch.long)


_SYMBOLS = re.compile(r"STOP|↑|←|→", flags=re.IGNORECASE)
_ACTION_BY_SYMBOL = {
    "STOP": (float(STOP), 0.0),
    "↑": (float(FORWARD), 0.25),
    "←": (float(LEFT), 15.0),
    "→": (float(RIGHT), 15.0),
}


def parse_symbolic_actions(text: str, *, horizon: int = 4) -> ParsedActions:
    if horizon <= 0:
        raise ValueError("action horizon must be positive")
    rows = [_ACTION_BY_SYMBOL[match.group(0).upper()] for match in _SYMBOLS.finditer(text)]
    invalid = not rows
    if invalid:
        rows = [_ACTION_BY_SYMBOL["STOP"]]
    truncated = len(rows) > horizon
    rows = rows[:horizon]
    mask = [True] * len(rows)
    while len(rows) < horizon:
        rows.append(_ACTION_BY_SYMBOL["STOP"])
        mask.append(False)
    return ParsedActions(
        actions=torch.tensor(rows, dtype=torch.float32),
        mask=torch.tensor(mask, dtype=torch.bool),
        invalid=invalid,
        truncated=truncated,
    )


__all__ = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "STOP",
    "VISION_PHRASE",
    "ParsedActions",
    "parse_symbolic_actions",
    "render_streamvln_prompt",
    "tokenize_multimodal_prompt",
]
