from __future__ import annotations

import pytest
import torch

from embodiinfer import EngineConfig, Observation
from embodiinfer.engine import EngineCore
from embodiinfer.engine.rollout import GenerationBackend
from embodiinfer.policies.navida.modeling_navida import (
    NaViDAMemory,
    NaViDAPolicy,
    NaViDARunner,
    _navida_generation_kwargs,
    _parse_navida_actions,
    _select_navida_history,
)
from embodiinfer.policies.qwen_r2r_low.modeling_qwen_r2r_low import (
    NavigationOutputError as LowLevelNavigationOutputError,
)
from embodiinfer.policies.qwen_r2r_low.modeling_qwen_r2r_low import (
    QwenR2RLowPolicy,
    _parse_low_level_action,
)
from embodiinfer.policies.qwen_r2r_low.modeling_qwen_r2r_low import (
    QwenVLNMemory as LowLevelMemory,
)
from embodiinfer.policies.qwen_r2r_low.modeling_qwen_r2r_low import (
    _Qwen25VLRunner as LowLevelRunner,
)
from embodiinfer.policies.qwen_r2r_panoramic.modeling_qwen_r2r_panoramic import (
    NavigationOutputError as PanoramicNavigationOutputError,
)
from embodiinfer.policies.qwen_r2r_panoramic.modeling_qwen_r2r_panoramic import (
    QwenR2RPanoramicMemory as PanoramicMemory,
)
from embodiinfer.policies.qwen_r2r_panoramic.modeling_qwen_r2r_panoramic import (
    QwenR2RPanoramicPolicy,
    parse_panoramic_action,
)
from embodiinfer.policies.qwen_r2r_panoramic.modeling_qwen_r2r_panoramic import (
    QwenR2RPanoramicRunner as PanoramicRunner,
)
from embodiinfer.types import SessionKey


class FakeRunner:
    def __init__(self, profile: str, text: str):
        self.profile = profile
        self.text = text
        self.model = torch.nn.Linear(1, 1)
        self.execute_chunks = 2 if profile == "navida" else 1
        self.graph_modes = []

    def configure_cuda_graph(self, enabled: bool):
        self.graph_modes.append(enabled)
        return enabled and next(self.model.parameters()).device.type == "cuda"

    def infer(self, observation, memory):
        del observation, memory
        return self.text, torch.tensor([7]), []


def obs(images=1):
    return Observation(
        torch.zeros(images, 3, 8, 8),
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        instruction="walk to the doorway",
        metadata={"candidates": [{"relative_angle": 0, "distance": 1}] * (images - 1)},
    )


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize(
    ("profile", "text", "images", "expected"),
    [
        ("low_level", "Move", 1, [[1.0, 0.0]]),
        ("panoramic", "1", 3, [[1.0, 0.0]]),
        (
            "navida",
            "forward 50 cm, turn left 30 degree",
            1,
            [[1.0, 0.25], [1.0, 0.25], [2.0, 15.0], [2.0, 15.0]],
        ),
    ],
)
def test_profiles_run_through_recurrent_engine(use_graph, profile, text, images, expected):
    runner = FakeRunner(profile, text)
    policy_class = {
        "low_level": QwenR2RLowPolicy,
        "panoramic": QwenR2RPanoramicPolicy,
        "navida": NaViDAPolicy,
    }[profile]
    policy = policy_class(profile, runner) if profile == "navida" else policy_class(profile, runner, profile)
    core = EngineCore(policy, EngineConfig(device="cpu", max_batch_size=1, use_cuda_graph=use_graph))
    output = GenerationBackend(core).generate([obs(images)], session_ids=[SessionKey("env", "episode")])[0]
    assert output.actions.tolist() == expected
    assert output.trace.meta["runtime_mode"] == "eager"
    assert output.trace.meta["cuda_graph_confirmed"] is False
    assert runner.graph_modes == [use_graph]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Left", [[2.0, 0.0]]), ("Right", [[3.0, 0.0]]), ("Move", [[1.0, 0.0]]), ("Stop", [[0.0, 0.0]])],
)
def test_low_level_parser_is_strict(text, expected):
    assert _parse_low_level_action(text).tolist() == expected


@pytest.mark.parametrize("text", ["", "Action: Left", "move forward", "turn right"])
def test_low_level_parser_rejects_unknown_text(text):
    with pytest.raises(LowLevelNavigationOutputError):
        _parse_low_level_action(text)


def test_panoramic_parser_distinguishes_stop_range_and_failure():
    assert parse_panoramic_action("Stop", 3).tolist() == [[-1.0, 0.0]]
    assert parse_panoramic_action("2", 3).tolist() == [[2.0, 0.0]]
    for text in ("", "Candidate: 1", "3", "-1"):
        with pytest.raises(PanoramicNavigationOutputError):
            parse_panoramic_action(text, 3)


def test_navida_v2_parser_executes_two_chunks_and_stops():
    assert _parse_navida_actions("forward 75 cm, turn right 45 degree, forward 25 cm").tolist() == [
        [1.0, 0.25],
        [1.0, 0.25],
        [1.0, 0.25],
        [3.0, 15.0],
        [3.0, 15.0],
        [3.0, 15.0],
    ]
    assert _parse_navida_actions("<answer>stop</answer>").tolist() == [[0.0, 0.0]]
    assert _parse_navida_actions("forward, turn left").tolist() == [[1.0, 0.25], [2.0, 15.0]]
    assert _parse_navida_actions("move", rng=__import__("random").Random(41)).shape == (1, 2)


def test_navida_v2_generation_contract():
    assert _navida_generation_kwargs(512) == {
        "do_sample": True,
        "temperature": 0.2,
        "top_k": 50,
        "top_p": 1.0,
        "max_new_tokens": 512,
        "repetition_penalty": 1.05,
        "num_return_sequences": 1,
        "use_cache": True,
    }


def test_low_level_prompt_matches_official_template():
    runner = object.__new__(LowLevelRunner)
    observation = obs()
    content, frames, kinds, label = runner._low_prompt(observation, LowLevelMemory())
    assert content == [
        {
            "type": "text",
            "text": (
                "Route Instruction: walk to the doorway\nCurrent Step: 0\n"
                "Cummulative Distance Traveled: 0.0\nImages from Previous Steps: []"
            ),
        },
        {"type": "text", "text": ("\nActions performed at Previous Steps: []\nCurrent image:")},
        {"type": "image"},
        {
            "type": "text",
            "text": (
                "\nPossible actions: ['Left', 'Right', 'Move', 'Stop']\n"
                "Now predict the next action based on the input you have recived. "
                "Answer on the format: Action: (an the action you choose)"
            ),
        },
    ]
    assert len(frames) == 1
    assert torch.equal(frames[0], observation.images[0])
    assert kinds == ["low"]
    assert label == "Action: "


def test_panoramic_prompt_matches_official_layout():
    runner = object.__new__(PanoramicRunner)
    observation = Observation(
        torch.zeros(2, 3, 8, 8),
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        instruction="go ahead",
        metadata={
            "distance_traveled": 1.5,
            "candidates": [{"relative_angle": -29.6, "distance": 2.334}],
        },
    )
    content, frames, kinds, label = runner._panoramic_prompt(observation, PanoramicMemory())
    texts = "".join(item["text"] for item in content if item["type"] == "text")
    assert texts == (
        "Route instruction: go ahead\nCurrent step: 0\n"
        "Cumulative Distance Traveled: 1.5 meters\n\n"
        "Panorama Images from Previous Steps:[]\n\nCurrent Panorama Image:\n\t"
        "\n\nCandidate Directions:\n\tCandidate: 0:\n"
        "\t\tRelative angle: 30.0 degrees to the Left\n"
        "\t\tDistance: 2.33 meters\n\t\tview: \n\tCandidate: Stop\n\n"
        "Now, analyze the route instruction, your current position, and the available "
        "candidate directions. Select the candidate that best matches the instruction and "
        "helps you continue along the correct path. Answer on the format: Candidate: "
        "(and then the number)"
    )
    assert len(frames) == 2
    assert torch.equal(frames[0], observation.images[0])
    assert torch.equal(frames[1], observation.images[1])
    assert kinds == ["panorama", "candidate"]
    assert label == "Candidate: "


def test_navida_history_sampling_and_first_step_duplication():
    current = torch.tensor([99.0])
    assert _select_navida_history((), current) == [current]
    previous = tuple(torch.tensor([float(index)]) for index in range(9))
    selected = _select_navida_history(previous, current)
    assert [int(frame.item()) for frame in selected] == [0, 1, 2, 3, 5, 6, 7, 8]

    runner = object.__new__(NaViDARunner)
    observation = obs()
    content, frames = runner._navida_prompt(observation, NaViDAMemory())
    assert len(frames) == 2
    assert torch.equal(frames[0], observation.images[0])
    assert torch.equal(frames[1], observation.images[0])
    assert [item["type"] for item in content] == ["text", "image", "text", "image", "text"]
