"""The shared inference driver remains usable by profiling, capture-ops and verify."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")


def test_driver_replays_recorded_observations_through_the_engine(tmp_path, monkeypatch):
    from scripts.kernel_tuning.discovery.drive import main, save_inputs

    from embodiinfer.engine import EngineCore
    from embodiinfer.policies.mock import preset_config
    from embodiinfer.types import Observation

    cfg = preset_config("tiny")

    def observation():
        return Observation(
            images=torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
            state=torch.rand(cfg.state_dim),
            instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
        )

    inputs = tmp_path / "inputs.pt"
    save_inputs(inputs, "observations", [[observation(), observation()], [observation()]])
    config = tmp_path / "serving.json"
    config.write_text(json.dumps({"policy_kwargs": {"preset": "tiny"}}), encoding="utf-8")
    original_execute = EngineCore.execute
    calls = []

    def execute(self, *args, **kwargs):
        result = original_execute(self, *args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(EngineCore, "execute", execute)
    main(
        [
            "--policy",
            "mock_flow_vla",
            "--device",
            "cpu",
            "--no-cuda-graph",
            "--adapter-config",
            str(config),
            "--inputs",
            str(inputs),
        ]
    )
    assert len(calls) == 3
