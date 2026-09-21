"""Entry-point error messages: typed, catchable via EmbodiInferError, with hints."""

from __future__ import annotations

import pytest
import torch

from embodiinfer import (
    EmbodiInfer,
    EmbodiInferError,
    EngineConfig,
    Observation,
    ObservationError,
    PolicyNotFoundError,
    make_policy,
    preset_config,
)


def test_make_policy_unknown_is_typed_and_suggests():
    with pytest.raises(PolicyNotFoundError) as ei:
        make_policy("mock_flow_vlda")  # near-miss typo
    assert isinstance(ei.value, EmbodiInferError)
    msg = str(ei.value)
    assert "Did you mean 'mock_flow_vla'?" in msg
    assert "Registered:" in msg


def test_preset_config_unknown_lists_available():
    with pytest.raises(ValueError) as ei:
        preset_config("smal")
    msg = str(ei.value)
    assert "Did you mean 'small'?" in msg
    assert "Available:" in msg


def _engine():
    return EmbodiInfer(
        "mock_flow_vla",
        preset="tiny",
        engine_config=EngineConfig(device="cpu", use_cuda_graph=False),
    )


def test_embodiinfer_act_rejects_wrong_camera_count():
    cfg = preset_config("tiny")  # num_cameras == 1
    engine = _engine()
    bad = Observation(
        images=torch.rand(2, 3, cfg.image_size, cfg.image_size),  # 2 cameras, expected 1
        state=torch.rand(cfg.state_dim),
        instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
    )
    with pytest.raises(ObservationError) as ei:
        engine.act(bad)
    assert "expected 1 cameras; got 2" in str(ei.value)


def test_embodiinfer_act_rejects_wrong_state_rank():
    cfg = preset_config("tiny")
    engine = _engine()
    bad = Observation(
        images=torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
        state=torch.rand(cfg.state_dim, 1),  # 2-D state
        instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
    )
    with pytest.raises(ObservationError):
        engine.act(bad)
