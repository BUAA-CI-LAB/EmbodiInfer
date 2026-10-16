"""Read-only developer commands remain usable without Humanize2 or a GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
