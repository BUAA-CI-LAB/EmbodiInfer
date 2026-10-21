"""Pinned R2R/RxR prompt and action grammar from ActiveVLN.

Source (Apache-2.0):
https://github.com/arvillion/ActiveVLN/tree/3a0c63b00e4f42c828cc74c3554afce17641da60

Local lane extension: RxR action space (30/60/90-degree turns) with the official
RxR system prompt, ported from the workspace's parity-stage RxR support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

SYSTEM_PROMPT_R2R = (
    "You are a helpful assistant. "
    "Your goal is to follow the given instruction to reach a specified destination. \n"
    "At each step, you receive a first-person image (starting view if first step (step 1), or "
    "post-action view otherwise). "
    "Your task is to select choose one action from: move forward 25cm, move forward 50cm, "
    "move forward 75cm, turn left 15 degrees, turn left 30 degrees, turn left 45 degrees, "
    "turn right 15 degrees, turn right 30 degrees, turn right 45 degrees, or stop. \n"
    "The instruction will be provided with each observation. You can take multiple actions at each turn. "
)

SYSTEM_PROMPT_RXR = (
    "You are a helpful assistant. "
    "Your goal is to follow the given instruction to reach a specified destination. \n"
    "At each step, you receive a first-person image (starting view if first step (step 1), or "
    "post-action view otherwise). "
    "Your task is to select choose one action from: move forward 25cm, move forward 50cm, "
    "move forward 75cm, turn left 30 degrees, turn left 60 degrees, turn left 90 degrees, "
    "turn right 30 degrees, turn right 60 degrees, turn right 90 degrees, or stop. \n"
    "The instruction will be provided with each observation. You can take multiple actions at each turn. "
)

SYSTEM_PROMPTS = {"r2r": SYSTEM_PROMPT_R2R, "rxr": SYSTEM_PROMPT_RXR}
DEFAULT_TURN_ANGLE = {"r2r": 15, "rxr": 30}

_INITIAL_LABEL = "[Initial Observation]:"
_SUBSEQUENT_LABEL = "After that, the observation is:"
_USER_SUFFIX = (
    "Instruction: {}Decide your next action. You can take up to 3 actions at a time, separated by ','. "
)

PAD_ACTION = -1
STOP_ACTION = 0
FORWARD_ACTION = 1
LEFT_ACTION = 2
RIGHT_ACTION = 3
MAX_ACTIONS = 3

_NUMBER_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class NavigationAction:
    name: str
    value: int | None

    @property
    def action_id(self) -> int:
        return {
            "stop": STOP_ACTION,
            "move forward": FORWARD_ACTION,
            "turn left": LEFT_ACTION,
            "turn right": RIGHT_ACTION,
        }[self.name]


@dataclass(frozen=True)
class ParsedNavigation:
    raw_text: str
    actions: tuple[NavigationAction, ...]
    invalid_fragments: tuple[str, ...]
    truncated: bool

    @property
    def valid(self) -> bool:
        return not self.invalid_fragments and not self.truncated and bool(self.actions)


def user_turn_content(instruction: str, *, initial: bool) -> list[dict]:
    label = _INITIAL_LABEL if initial else _SUBSEQUENT_LABEL
    return [
        {"type": "text", "text": label},
        {"type": "image"},
        {"type": "text", "text": _USER_SUFFIX.format(instruction)},
    ]


def chat_messages(instruction: str, *, initial: bool, action_space: str = "r2r") -> list[dict]:
    messages = []
    if initial:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPTS[action_space]}]}
        )
    messages.append({"role": "user", "content": user_turn_content(instruction, initial=initial)})
    return messages


def render_turn_text(processor, instruction: str, *, initial: bool, action_space: str = "r2r") -> str:
    """Serialize one delta turn without inserting a second system prompt.

    Qwen's chat template supplies a default ``system`` message when called with
    only a user message.  That is useful for standalone chats, but wrong after
    an ActiveVLN assistant response: the system prompt is already cached in the
    episode prefix.  Serialise the subsequent user→assistant boundary directly
    so its token sequence is the exact suffix of a canonical full history.
    """
    if initial:
        return processor.apply_chat_template(
            chat_messages(instruction, initial=True, action_space=action_space),
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        "\n<|im_start|>user\n"
        f"{_SUBSEQUENT_LABEL}<|vision_start|><|image_pad|><|vision_end|>"
        f"{_USER_SUFFIX.format(instruction)}<|im_end|>\n<|im_start|>assistant\n"
    )


def parse_navigation_actions(
    text: str, *, max_actions: int = MAX_ACTIONS, default_turn_angle: int = 15
) -> ParsedNavigation:
    cleaned = text.replace("<image>", "").strip()
    fragments = [part.strip().lower() for part in cleaned.split(",") if part.strip()]
    truncated = len(fragments) > max_actions
    fragments = fragments[:max_actions]
    actions: list[NavigationAction] = []
    invalid: list[str] = []
    for fragment in fragments:
        number = _NUMBER_RE.search(fragment)
        if "stop" in fragment:
            actions.append(NavigationAction("stop", None))
        elif "forward" in fragment:
            actions.append(NavigationAction("move forward", int(number.group()) if number else 25))
        elif "left" in fragment:
            actions.append(
                NavigationAction("turn left", int(number.group()) if number else default_turn_angle)
            )
        elif "right" in fragment:
            actions.append(
                NavigationAction("turn right", int(number.group()) if number else default_turn_angle)
            )
        else:
            invalid.append(fragment)
    return ParsedNavigation(cleaned, tuple(actions), tuple(invalid), truncated)


def parse_r2r_actions(text: str, *, max_actions: int = MAX_ACTIONS) -> ParsedNavigation:
    """Backward-compatible R2R parser: missing turn angles default to 15 degrees."""
    return parse_navigation_actions(text, max_actions=max_actions, default_turn_angle=15)


def actions_to_tensor(parsed: ParsedNavigation) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.zeros(MAX_ACTIONS, 2, dtype=torch.float32)
    actions[:, 0] = PAD_ACTION
    mask = torch.zeros(MAX_ACTIONS, dtype=torch.bool)
    for i, action in enumerate(parsed.actions[:MAX_ACTIONS]):
        actions[i, 0] = action.action_id
        actions[i, 1] = 0 if action.value is None else action.value
        mask[i] = True
    return actions, mask
