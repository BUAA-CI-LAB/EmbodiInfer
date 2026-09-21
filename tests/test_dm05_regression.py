from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from embodiinfer import EmbodiInfer, EngineConfig, Observation
from embodiinfer.engine.graph import _flow_state_shape
from embodiinfer.engine.serve.contracts import RawImage, RawPolicyRequest
from embodiinfer.policies.dm05 import DM05Policy
from embodiinfer.policies.dm05.modeling_dm05 import DM05Prefix
from embodiinfer.policies.dm05.processor_dm05 import pack_images
from embodiinfer.policies.dm05.serving import DM05_STATE_DESCRIPTION, DM05_VIEW_ORDER, DM05ServingAdapter
from embodiinfer.types import ActionChunk


class _Layer:
    def __init__(self, value: torch.Tensor):
        self.keys, self.values = value, value + 1


class _Cache:
    def __init__(self, value: torch.Tensor):
        self.layers = [_Layer(value)]

    def batch_repeat_interleave(self, count):
        for layer in self.layers:
            layer.keys = layer.keys.repeat_interleave(count, dim=0)
            layer.values = layer.values.repeat_interleave(count, dim=0)


class _RuntimeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.model = SimpleNamespace(
            config=SimpleNamespace(action_dim=32, chunk_size=2),
            action_in_proj=torch.nn.Identity(),
        )


class _Output:
    def __call__(self, data):
        # Stand-in for the CPU output transform, outside the captured graph.
        data["action"] = data["action"] + data["state"][: data["action"].shape[-1]]
        return data


def _policy(output_dim: int = 7) -> DM05Policy:
    runtime = SimpleNamespace(
        model=_RuntimeModel(),
        image_prompts=["Head"],
        output_transform=_Output(),
    )
    from embodiinfer.policies.dm05.config import DM05PolicyConfig

    return DM05Policy(
        DM05PolicyConfig(
            name="dm05-test",
            action_dim=output_dim,
            action_horizon=2,
            internal_action_dim=32,
            output_action_dim=output_dim,
        ),
        runtime=runtime,
    )


def _prefix(batch: int = 1, width: int = 4) -> DM05Prefix:
    rows = torch.arange(batch * width * 2, dtype=torch.float32).reshape(batch, width, 2)
    return DM05Prefix(
        _Cache(rows),
        torch.arange(batch * width).reshape(batch, width),
        width,
        [np.zeros(14, dtype=np.float32) for _ in range(batch)],
        [{} for _ in range(batch)],
    )


def _with_input_transform(policy):
    received = []

    def transform(data):
        received.append(data)
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "pixel_values": torch.zeros(len(data["images"]), 3, 2, 2),
            "token_type_ids": torch.zeros(1, 3, dtype=torch.long),
        }

    policy.runtime.input_transform = transform
    policy.runtime.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0, padding_side="right")
    )
    policy._dm._compute_prefix_cache = lambda **kwargs: (
        _Cache(torch.zeros(*kwargs["input_ids"].shape, 2)),
        kwargs["input_ids"].shape[1],
    )
    return received


def _observation(output_dim=7, **metadata):
    return Observation(
        images=torch.zeros(1, 3, 8, 8),
        state=torch.arange(output_dim, dtype=torch.float32) / 10,
        instruction_tokens=torch.empty(0, dtype=torch.long),
        instruction="Press the button.",
        metadata=metadata,
    )


def test_dm05_common_observation_runs_through_public_act():
    policy = _policy()
    received = _with_input_transform(policy)
    policy.denoise_step = lambda state, time, prefix: torch.zeros_like(state)
    observation = _observation(robot_type="UR5", speed=0.7, control_mode="cartesian")
    engine = EmbodiInfer(policy, engine_config=EngineConfig(device="cpu", use_cuda_graph=False))
    torch.manual_seed(43)
    result = engine.act(observation, num_steps=2)
    expected = policy.new_noise(1, torch.Generator().manual_seed(43))[0, :, :7] + observation.state
    torch.testing.assert_close(result.actions, expected, rtol=0, atol=0)
    assert received[0]["prompt"] == observation.instruction
    assert received[0]["meta_data"] == {
        "robot_type": "UR5",
        "speed": "0.7",
        "control_mode": "cartesian",
        "state_desc": ["eef"] * 6 + ["gripper"],
    }
    assert received[0]["history_images"] == []
    assert "state_desc" not in observation.metadata  # Defaults do not mutate caller data.


def test_dm05_padded_observation_preserves_original_pixels_and_history():
    policy = _policy()
    policy.runtime.image_prompts = ["Head", "Left wrist", "Right wrist"]
    received = _with_input_transform(policy)
    images = [
        Image.fromarray(np.arange(8 * 32 * 3, dtype=np.uint8).reshape(8, 32, 3)),
        Image.new("RGB", (11, 4), (20, 70, 190)),
        Image.new("RGB", (5, 16), (255, 0, 128)),
    ]
    pixels, sizes = pack_images(images)
    observation = Observation(
        images=pixels,
        state=torch.zeros(7),
        instruction_tokens=torch.empty(0, dtype=torch.long),
        instruction="Pick up the cup.",
        metadata={
            "image_sizes": sizes,
            "history_images": [images[1]],
            "history_placeholder_text": "<unused1>" * 64 + "<unused0>" * 16 + "\n",
        },
    )
    assert observation.images.shape == (3, 3, 16, 32)
    batch = policy.collate([observation], ["r0"])
    assert batch.request_ids == ["r0"]
    for restored, original in zip(received[0]["images"], images, strict=True):
        assert restored.size == original.size
        assert restored.tobytes() == original.tobytes()
    assert received[0]["history_images"][0].tobytes() == images[1].tobytes()
    assert received[0]["history_placeholder_text"] == observation.metadata["history_placeholder_text"]


@pytest.mark.parametrize("sizes", [[], [(0, 8)], [(9, 8)], [(8, True)], ["88"]])
def test_dm05_rejects_invalid_image_sizes_before_opendm(sizes):
    policy = _policy()
    received = _with_input_transform(policy)
    with pytest.raises(ValueError, match="image_sizes"):
        policy.collate([_observation(image_sizes=sizes)], ["r0"])
    assert not received


def test_dm05_http_shared_observations_keep_session_history_and_reset():
    policy = _policy()
    policy.runtime.image_prompts = ["Head", "Left wrist", "Right wrist"]
    policy.runtime.is_history = True
    received = _with_input_transform(policy)

    def execute(batch, **kwargs):
        assert kwargs["num_steps"] == 2
        return [ActionChunk(request_id=batch.request_ids[0], actions=torch.zeros(2, 7))]

    adapter = DM05ServingAdapter(core=SimpleNamespace(policy=policy, policy_version=0, execute=execute))
    images = []
    for index, name in enumerate(DM05_VIEW_ORDER):
        image = Image.new("RGB", (8 + index, 9 - index), (index, 64, 255))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        images.append(RawImage(name, "image/png", buffer.getvalue()))
    request = RawPolicyRequest(
        session_id="a",
        request_id="r0",
        step_id=0,
        instruction="Press the button.",
        state={
            "representation": "eef_xyzrpy_gripper",
            "state_desc": list(DM05_STATE_DESCRIPTION),
            "source": "recorded",
            "units": "m/rad",
            "values": [0.0] * 7,
        },
        images=tuple(images),
        metadata={"num_steps": 2, "synthetic_views": False, "speed": 0.7},
    )
    adapter.infer(request)
    adapter.infer(replace(request, session_id="b", request_id="r1"))
    adapter.infer(replace(request, request_id="r2", step_id=1))
    assert received[0]["history_images"] == received[1]["history_images"] == []
    assert received[0]["history_placeholder_text"] == "<unused1>" * 80
    assert len(received[2]["history_images"]) == 1
    assert received[2]["history_images"][0].tobytes() == received[0]["images"][0].tobytes()
    assert received[2]["history_placeholder_text"] == "<unused1>" * 64 + "<unused0>" * 16 + "\n"
    assert [image.size for image in received[0]["images"]] == [(8, 9), (9, 8), (10, 7)]
    assert received[2]["meta_data"]["speed"] == "0.7"
    adapter.reset("a")
    adapter.infer(replace(request, request_id="r3"))
    assert received[3]["history_images"] == []


def test_dm05_is_registered_without_importing_opendm():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from embodiinfer.policies import available_policies; assert 'dm05' in available_policies()",
        ],
        check=True,
    )


@pytest.mark.parametrize("output_dim", [7, 14])
def test_dm05_internal_32_state_becomes_public_7_or_14(output_dim):
    policy = _policy(output_dim)
    state = torch.zeros(1, 2, 32)
    actions = policy.finalize_actions(state, _prefix())
    assert actions.shape == (1, 2, output_dim)


def test_dm05_decoder_graph_state_is_internal_32_wide():
    policy = _policy()
    assert _flow_state_shape(policy, 3) == (3, 2, 32)
    assert policy.decoder.init_state(3).shape == (3, 2, 32)


def test_graph_defaults_and_standard_full_loop_argument():
    import argparse

    from embodiinfer.engine.serve.factory import add_policy_arguments
    from embodiinfer.policies import make_policy

    policy = make_policy("mock_flow_vla")
    assert _flow_state_shape(policy, 2) == (2, policy.config.action_horizon, policy.config.action_dim)
    parser = argparse.ArgumentParser()
    add_policy_arguments(parser)
    assert parser.parse_args([]).capture_full_loop is False
    assert parser.parse_args(["--capture-full-loop"]).capture_full_loop is True


def test_dm05_uses_the_standard_serving_factory(monkeypatch):
    import argparse

    from embodiinfer.engine.serve import factory

    policy = _policy()
    policy.runtime.image_prompts = ["Head", "Left wrist", "Right wrist"]
    policy.runtime.is_history = True

    def make_policy(name, **kwargs):
        assert name == "dm05"
        assert kwargs["load_device"] == "cpu"
        return policy

    monkeypatch.setattr(factory, "make_policy", make_policy)
    parser = argparse.ArgumentParser()
    factory.add_policy_arguments(parser)
    args = parser.parse_args(["--policy", "dm05", "--device", "cpu", "--no-cuda-graph"])
    adapter = factory.build_serving_adapter(args)
    assert adapter.capabilities()["model"] == "dm05"
    assert adapter.capabilities()["output_action_dim"] == 7


@pytest.mark.parametrize("output_dim", [7, 14])
def test_dm05_shared_decoder_applies_output_conversion_once(output_dim):
    policy = _policy(output_dim)
    policy.denoise_step = lambda state, time, prefix: torch.zeros_like(state)
    prefix = _prefix()
    prefix.states[0][:] = 2
    calls = []

    def transform(data):
        calls.append(data["action"].shape)
        return _Output()(data)

    policy.runtime.output_transform = transform
    actions = policy.decoder.produce_chunk(torch.ones(1, 2, 32), prefix, 3, 1, None)
    torch.testing.assert_close(actions, torch.full((1, 2, output_dim), 3.0), rtol=0, atol=0)
    assert calls == [(2, output_dim)]


@pytest.mark.parametrize("output_dim", [7, 14])
def test_standard_rollout_backend_restores_batch_and_sample_axes(output_dim):
    from embodiinfer.engine.rollout.generation_backend import GenerationBackend

    policy = _policy(output_dim)
    received = _with_input_transform(policy)
    policy.denoise_step = lambda state, time, prefix: state * 0.1
    backend = GenerationBackend(
        SimpleNamespace(policy=policy, device=torch.device("cpu"), dtype=torch.float32, pcfg=policy.config)
    )
    observations = [_observation(output_dim), _observation(output_dim)]
    actions, scores = backend.generate_with_logprob(observations, num_steps=2, num_samples=3)
    assert actions.shape == (2, 3, 2, output_dim)
    assert scores.shape == (2, 3)
    assert len(received) == 2


@pytest.mark.parametrize("output_dim", [7, 14])
def test_dm05_rollout_scores_realized_actions_and_backpropagates(output_dim):
    policy = _policy(output_dim)
    policy.denoise_step = lambda state, time, prefix: state * 0.1 + policy.model.anchor
    prefix = _prefix()
    actions, behavior, trajectory = policy.decoder.sample_with_logprob(
        prefix, 2, 0.1, torch.Generator().manual_seed(42)
    )
    assert actions.shape == (1, 2, output_dim)
    assert trajectory.shape == (1, 3, 2, 32)
    score = policy.decoder.recompute_logprob(prefix, trajectory, 2, 0.1)
    torch.testing.assert_close(score, behavior, rtol=0, atol=1e-5)
    changed = trajectory.clone()
    changed[..., output_dim:] += 100
    torch.testing.assert_close(policy.decoder.recompute_logprob(prefix, changed, 2, 0.1), score)
    score.sum().backward()
    gradient = policy.model.anchor.grad
    assert gradient is not None and torch.isfinite(gradient) and gradient.abs() > 0


def test_dm05_graph_prefix_harness_copies_in_place_on_cpu():
    policy = _policy()
    source = _prefix()
    static = policy.allocate_static_prefix_from_live(
        source, 1, torch.device("cpu"), torch.float32, policy.cuda_graph_variant(source)
    )
    key_ptr, id_ptr = static.cache.layers[0].keys.data_ptr(), static.input_ids.data_ptr()
    replacement = _prefix()
    replacement.input_ids.add_(10)
    policy.copy_prefix_into(static, replacement)
    assert static.cache.layers[0].keys.data_ptr() == key_ptr
    assert static.input_ids.data_ptr() == id_ptr
    torch.testing.assert_close(static.input_ids, replacement.input_ids)


@pytest.mark.gpu
@pytest.mark.parametrize("full_loop", [False, True])
@pytest.mark.parametrize("output_dim", [7, 14])
def test_dm05_cuda_graph_matches_eager(full_loop, output_dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from embodiinfer.engine.graph import GraphManager

    policy = _policy(output_dim).cuda()
    policy.denoise_step = lambda state, time, prefix: (
        state * 0.1 + policy.model.anchor + prefix.cache.layers[0].keys.mean()
    )
    graphs = GraphManager(policy, torch.device("cuda"), torch.float32, full_loop=full_loop)
    noise = policy.new_noise(1, generator=torch.Generator(device="cuda").manual_seed(42))
    with torch.no_grad():
        for width in (4, 6):
            prefix = _prefix(width=width)
            prefix.input_ids = prefix.input_ids.cuda()
            for layer in prefix.cache.layers:
                layer.keys, layer.values = layer.keys.cuda(), layer.values.cuda()
            for update in (0, 1):
                prefix.cache.layers[0].keys.add_(update)
                prefix.states[0][:] = update + 1
                eager = policy.decoder.produce_chunk(noise.clone(), prefix, 2, 1, None)
                replay = policy.decoder.produce_chunk(noise.clone(), prefix, 2, 1, graphs)
                torch.testing.assert_close(replay, eager, rtol=0, atol=0)
    assert len(graphs._graphs) == 2


@pytest.mark.parametrize("horizon", [10, 50])
def test_checkpoint_loader_honors_policy_horizon_and_fused_options(monkeypatch, horizon):
    """The LIBERO ten-action profile must not silently load the ARX5 fifty-action shape."""
    from types import ModuleType

    from embodiinfer.policies.factory import make_policy

    captured = {}

    class ModelConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class InferenceConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def _initialize(self, **kwargs):
            self.model = kwargs["model"]

    module = ModuleType("opendm.exp.dm05_exp")
    module.DM05ModelConfig = ModelConfig
    module.DM05InferenceConfig = InferenceConfig
    monkeypatch.setitem(sys.modules, "opendm.exp.dm05_exp", module)

    def build_model(config):
        model = _RuntimeModel()
        model.model.config.chunk_size = captured["chunk_size"]
        return model

    monkeypatch.setattr(DM05Policy, "_build_base_model", staticmethod(build_model))
    policy = make_policy(
        "dm05",
        checkpoint="/checkpoint",
        norm_stats="/checkpoint/norm_stats.json",
        robot_type="Franka",
        action_horizon=horizon,
        default_num_steps=6,
        image_prompts=["Head", "Left wrist"],
        liger_kernel=True,
    )
    assert policy.config.action_horizon == horizon
    assert captured["chunk_size"] == horizon
    assert captured["liger_kernel"] is True
    assert policy.runtime.diffusion_steps == 6


def test_flow_model_interval_excludes_physical_action_restoration():
    """Splitting timing must preserve the full decode and apply CPU transforms once."""
    policy = _policy()
    prefix = _prefix()
    calls = []
    finalized = []

    def denoise(state, time, prefix):
        calls.append(float(time[0]))
        return torch.ones_like(state)

    class Output:
        def __call__(self, data):
            finalized.append(data["action"].copy())
            return {**data, "action": data["action"] + 3}

    policy.denoise_step = denoise
    policy.runtime.output_transform = Output()
    initial = torch.zeros(1, 2, 32)
    model_actions = policy.decoder.integrate(initial, prefix, 4, 1, None)
    assert len(calls) == 4
    assert finalized == []
    restored = policy.finalize_actions(model_actions, prefix)
    assert len(finalized) == 1
    calls.clear()
    finalized.clear()
    ordinary = policy.decoder.produce_chunk(initial, prefix, 4, 1, None)
    assert len(calls) == 4
    assert len(finalized) == 1
    torch.testing.assert_close(ordinary, restored, rtol=0, atol=0)
