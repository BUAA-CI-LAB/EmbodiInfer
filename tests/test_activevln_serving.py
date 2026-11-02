"""HTTP mapping and continuation failures without a checkpoint or robot."""

from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from embodiinfer.engine.serve.contracts import RawImage, RawPolicyRequest
from embodiinfer.policies.activevln.execution_activevln import ExecutionPrefix
from embodiinfer.policies.activevln.prompt_activevln import actions_to_tensor, parse_r2r_actions
from embodiinfer.policies.activevln.serving import ActiveVLNServingAdapter
from embodiinfer.types import ActionChunk, DecodeTrace, SessionKey


class Core:
    def __init__(self):
        self.policy = SimpleNamespace(is_recurrent=True, action_space="r2r", collate=self.collate)
        self.calls = []
        self.resets = []
        self.fail = False
        self.text = "move forward 25cm, turn left 15 degrees"

    def collate(self, observations, request_ids):
        return SimpleNamespace(observations=observations, request_ids=request_ids)

    def execute(self, batch, *, session_ids):
        self.calls.append((batch, session_ids))
        if self.fail:
            raise RuntimeError("injected model failure")
        parsed = parse_r2r_actions(self.text)
        return [
            ActionChunk(
                request_id=batch.request_ids[0],
                actions=actions_to_tensor(parsed)[0],
                latency_ms=8,
                timing={"e2e_ms": 8, "prefill_ms": 3, "decode_ms": 5},
                policy_version=4,
                trace=DecodeTrace(
                    token_ids=torch.tensor([1, 2]), text=self.text, parsed_actions=parsed, stop_reason="eos"
                ),
            )
        ]

    def reset_sessions(self, sessions):
        self.resets.append(sessions)


def request(step=0, session="one", metadata=None):
    payload = BytesIO()
    Image.new("RGB", (3, 2), color=(255, 64, 0)).save(payload, format="PNG")
    return RawPolicyRequest(
        session_id=session,
        request_id=f"request-{step}",
        step_id=step,
        instruction="Walk to the door.",
        state={"benchmark_slot": 0},
        images=(RawImage("observation.images.rgb", "image/png", payload.getvalue()),),
        metadata=metadata or {},
    )


def test_maps_rgb_actions_and_engine_session():
    core = Core()
    adapter = ActiveVLNServingAdapter(core=core)
    result = adapter.infer(request())
    batch, keys = core.calls[0]
    observation = batch.observations[0]
    assert observation.images.shape == (1, 3, 2, 3)
    assert torch.equal(observation.images[0, :, 0, 0], torch.tensor([1, 64 / 255, 0]))
    assert keys == [SessionKey("one", "http")]
    assert result.actions[0].kind == "discrete_chunk"
    assert result.actions[0].values["rows"] == [[1, 25], [2, 15], [-1, 0]]
    assert result.actions[0].values["parsed_valid"] is True
    assert result.policy_revision == "4"
    assert result.timing["decode_ms"] == 5


def test_prefix_requires_own_history_and_acknowledges_exact_tail():
    core = Core()
    adapter = ActiveVLNServingAdapter(core=core)
    prefix = ExecutionPrefix(0, ((1, 12.5),))
    metadata = {"activevln_execution": prefix.to_wire()}
    with pytest.raises(ValueError, match="consecutive"):
        adapter.infer(request(1, metadata=metadata))
    assert not core.calls
    adapter.infer(request())
    with pytest.raises(ValueError, match="consecutive"):
        adapter.infer(request(1, session="two", metadata=metadata))
    result = adapter.infer(request(1, metadata=metadata))
    assert core.calls[-1][0].observations[0].metadata["activevln_execution"] == prefix.to_wire()
    assert result.actions[0].values["execution_prefix_sha256"] == prefix.digest


def test_failure_does_not_advance_adapter_history_and_reset_clears_it():
    core = Core()
    adapter = ActiveVLNServingAdapter(core=core)
    core.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        adapter.infer(request())
    core.fail = False
    adapter.infer(request())
    with pytest.raises(ValueError, match="instruction"):
        adapter.infer(replace(request(1), instruction="A different destination"))
    adapter.reset("one")
    assert core.resets == [[SessionKey("one", "http")]]
    adapter.infer(replace(request(), instruction="A different destination"))


@pytest.mark.parametrize(
    "images",
    [
        (),
        (RawImage("wrong", "image/png", b"bad"),),
        (RawImage("observation.images.rgb", "image/png", b"bad"),),
    ],
)
def test_bad_images_fail_before_model(images):
    core = Core()
    with pytest.raises(ValueError):
        ActiveVLNServingAdapter(core=core).infer(replace(request(), images=images))
    assert not core.calls


def test_invalid_generated_text_remains_explicitly_invalid():
    core = Core()
    core.text = "not an action"
    result = ActiveVLNServingAdapter(core=core).infer(request())
    assert result.actions[0].values["parsed_valid"] is False


def test_factory_loads_cpu_mappings_then_moves_without_dtype_cast(monkeypatch, tmp_path):
    import sys

    from embodiinfer.policies.activevln import modeling_activevln as model

    calls = []
    moves = []
    qwen = SimpleNamespace(to=lambda device: moves.append(str(device)))
    loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: (calls.append(kwargs), qwen)[1])
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(Qwen2_5_VLForConditionalGeneration=loader)
    )
    monkeypatch.setattr(model, "_validate_qwen", lambda _: None)
    monkeypatch.setattr(model, "ActiveVLNProcessor", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "ActiveVLNPolicy", lambda backbone, *args, **kwargs: backbone)
    assert model._build_activevln(str(tmp_path), load_device="cuda:0") is qwen
    assert calls[0]["device_map"] == {"": "cpu"}
    assert calls[0]["low_cpu_mem_usage"] is True
    assert calls[0]["torch_dtype"] == "auto"
    assert moves == ["cuda:0"]
