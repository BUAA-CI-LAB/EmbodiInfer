"""Read-only developer commands remain usable without Humanize2 or a GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from scripts.kernel_tuning.__main__ import main


def test_check_and_status_do_not_import_an_agent(task_dir: Path, store, monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "hmz.sdk", None)
    assert main(["check", str(task_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["validation"].startswith("structural")
    assert main(["status", str(store.root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "created"


def test_run_requires_an_explicit_agent_before_creating_artifacts(
    task_dir: Path, tmp_path: Path, capsys
) -> None:
    output = tmp_path / "unused"
    assert main(["run", str(task_dir), "--output", str(output)]) == 2
    assert "explicit Humanize2 agent" in capsys.readouterr().err
    assert not output.exists()


def test_bad_yaml_is_a_clear_static_validation_error(task_dir: Path, capsys) -> None:
    (task_dir / "tuning.yaml").write_text("language: [oops\n", encoding="utf-8")
    assert main(["check", str(task_dir)]) == 2
    assert "Invalid tuning.yaml" in capsys.readouterr().err


def test_removed_full_model_capture_is_rejected_before_running_driver(tmp_path, capsys):
    output = tmp_path / "capture"
    with pytest.raises(SystemExit) as exc:
        main(["capture", "--model", "pi05", "--output", str(output), "--", "unused.py"])
    assert exc.value.code == 2
    assert not output.exists()
    assert main(["generate", "--model", "pi05"]) == 2
    assert "run capture-ops --model first" in capsys.readouterr().err


def test_catalog_capture_keeps_its_explicit_entrypoint(tmp_path, monkeypatch, capsys):
    from scripts.kernel_tuning import capture

    calls = []

    def record(command, output, operators):
        calls.append((command, output, [op.name for op in operators]))
        return [{"operator": "gated_residual", "axes": {}, "count": 3}]

    monkeypatch.setattr(capture, "run", record)
    output = tmp_path / "calls.jsonl"
    assert main(["capture", "--catalog", "--output", str(output), "gated_residual", "--", "driver.py"]) == 0
    assert calls == [(["driver.py"], output, ["gated_residual"])]
    assert "Captured 3 calls" in capsys.readouterr().out
