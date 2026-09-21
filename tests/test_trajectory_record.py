"""Proposal 0004 trajectory audit and weight-version contracts."""

import pytest
import torch

from embodiinfer import EngineConfig, Observation, TrajectoryRecord, preset_config
from embodiinfer.engine.core import EngineCore
from embodiinfer.engine.rollout.weight_sync import LocalWeightSync, NCCLWeightSync
from embodiinfer.policies.factory import make_policy
from embodiinfer.types import DecodeTrace, collate


def _record(**overrides):
    values = {
        "env_id": "env-3",
        "episode_id": "episode-9",
        "step_idx": 4,
        "policy_version": 2,
        "seed": 123,
        "raw_tokens": [7, 8],
        "token_logprobs": [-0.25, -0.5],
        "parsed_action": [[1.0, 25.0]],
        "executed_action": [[1.0, 25.0]],
        "reward": 1.0,
        "done": False,
        "timing": {"prefill_ms": 2.0, "decode_ms": 3.0, "e2e_ms": 5.0},
    }
    values.update(overrides)
    return TrajectoryRecord(**values)


def test_trajectory_record_has_typed_complete_schema():
    record = _record()
    assert record.raw_tokens.dtype == torch.long
    assert record.token_logprobs.dtype == torch.float32
    record.validate_complete()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"step_idx": -1}, "step_idx"),
        ({"policy_version": -1}, "policy_version"),
        ({"raw_tokens": [[7, 8]]}, "raw_tokens must be 1-D"),
        ({"token_logprobs": [-0.5]}, "one-to-one"),
        ({"token_logprobs": None}, "token_logprobs or recompute_state"),
        ({"timing": {"prefill_ms": 1.0}}, "e2e_ms"),
        ({"timing": {"e2e_ms": -1.0}}, "non-negative"),
    ],
)
def test_trajectory_record_rejects_invalid_engine_fields(overrides, match):
    with pytest.raises(ValueError, match=match):
        _record(**overrides)


def test_trajectory_record_allows_inflight_env_fields_but_not_persistence():
    record = _record(executed_action=None, reward=None, done=None)
    with pytest.raises(ValueError, match="executed_action.*reward.*done"):
        record.validate_complete()


def test_trajectory_record_accepts_recompute_reference_instead_of_logprobs():
    record = _record(token_logprobs=None, recompute_state="rollout-buffer:17")
    record.validate_complete()


def test_local_weight_sync_version_advances_only_after_success():
    policy = torch.nn.Linear(2, 2)
    sync = LocalWeightSync(policy)
    state = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}

    assert sync.policy_version == 0
    sync.update(state)
    assert sync.policy_version == 1

    with pytest.raises(KeyError, match="missing weights"):
        sync.update({"weight": state["weight"]})
    assert sync.policy_version == 1

    with pytest.raises(RuntimeError):
        sync.update({"weight": torch.zeros(3), "bias": state["bias"]})
    assert sync.policy_version == 1


def test_nccl_weight_sync_version_advances_only_after_complete_broadcast(monkeypatch):
    policy = torch.nn.Linear(2, 2)
    sync = NCCLWeightSync(policy)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)

    sync.update()
    assert sync.policy_version == 1

    def fail(*args, **kwargs):
        raise RuntimeError("broadcast failed")

    monkeypatch.setattr(torch.distributed, "broadcast", fail)
    with pytest.raises(RuntimeError, match="broadcast failed"):
        sync.update()
    assert sync.policy_version == 1


def test_engine_stamps_current_policy_version_and_stage_timing():
    policy = make_policy("mock_flow_vla", preset="tiny")
    core = EngineCore(policy, EngineConfig(device="cpu", use_cuda_graph=False))
    sync = LocalWeightSync(policy)
    state = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}
    sync.update(state)

    cfg = preset_config("tiny")
    obs = Observation(
        images=torch.zeros(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
        state=torch.zeros(cfg.state_dim),
        instruction_tokens=torch.zeros(cfg.max_lang_len, dtype=torch.long),
    )
    chunk = core.execute(collate([obs], ["request-1"]))[0]

    assert core.policy_version == chunk.policy_version == 1
    assert chunk.timing.keys() == {"prefill_ms", "decode_ms", "e2e_ms"}
    assert chunk.timing["e2e_ms"] == chunk.latency_ms
    assert chunk.timing["prefill_ms"] + chunk.timing["decode_ms"] == pytest.approx(chunk.timing["e2e_ms"])


def test_trace_cpu_copy_stamps_version_and_timing():
    trace = DecodeTrace(token_ids=torch.tensor([1]), timing={"e2e_ms": 99.0})
    copied = EngineCore._trace_to_cpu(
        trace,
        policy_version=4,
        timing={"prefill_ms": 1.0, "decode_ms": 2.0, "e2e_ms": 3.0},
    )
    assert copied.policy_version == 4
    assert copied.timing == {"prefill_ms": 1.0, "decode_ms": 2.0, "e2e_ms": 3.0}
