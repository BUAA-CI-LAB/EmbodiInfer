"""Run the real Humanize2 engine on its official fake agents and CPU evaluators."""

from __future__ import annotations

import asyncio
import copy
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("hmz", reason="requires the optional Humanize2 tooling environment")
from hmz.sdk import fakes  # noqa: E402
from scripts.kernel_tuning.artifacts import RunStore, atomic_json, run_lock  # noqa: E402
from scripts.kernel_tuning.contracts import ContractError, read_json  # noqa: E402
from scripts.kernel_tuning.flow import kernel_tuning  # noqa: E402
from scripts.kernel_tuning.runner import evaluate  # noqa: E402


@pytest.fixture(autouse=True)
def windows_journal_io(monkeypatch) -> None:
    """Adapt only hmz's POSIX atomic-file primitive for Windows CPU tests.

    The real framework state, journal reading, sessions, budgets and resume run
    unchanged. Production run/resume explicitly requires POSIX; this shim is not
    shipped as part of the workflow and does not emulate agent/kernel execution.
    """
    if os.name == "nt":
        from hmz.coganchor import atomic

        def writes(at: Path, said, *, mode=None) -> None:
            fd, name = tempfile.mkstemp(dir=at.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    if isinstance(said, str):
                        handle.write(said.encode("utf-8"))
                    elif isinstance(said, bytes):
                        handle.write(said)
                    else:
                        handle.writelines(said)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(name, at)
            finally:
                Path(name).unlink(missing_ok=True)

        monkeypatch.setattr(atomic, "writes", writes)


def fake_coder(store: RunStore, candidate: dict, latencies: list[Any]):
    """Write candidate files only when a fake agent is asked to implement its plan."""

    def answer(prompt: str, **kwargs: Any) -> str:
        scratch = Path(str(kwargs["session"].placement.workdir))
        if "Write PLAN.md" in prompt:
            (scratch / "PLAN.md").write_text(
                "Hypothesis: fewer launches. Validate every workload.", encoding="utf-8"
            )
            return "plan written"
        value = latencies.pop(0)
        if isinstance(value, BaseException):
            raise value
        solution = copy.deepcopy(candidate)
        solution["name"] += "_" + scratch.name
        solution["description"] = str(value)
        atomic_json(scratch / "solution.json", solution)
        return "solution written"

    return fakes.FakeAgentDriver(reply=answer)


def run_fake(store: RunStore, coder, *, resume: bool = False) -> dict:
    """Exercise hmz journaling/session behavior; no real harness is constructed."""
    return asyncio.run(
        fakes.run_fake(
            kernel_tuning,
            "test only",
            agents={"coder": coder},
            local=fakes.FakeEnvDriver(workdir=(store.root / "workspace").as_posix()),
            params={"run_dir": str(store.root)},
            journal=store.root / "fake-hmz.jsonl",
            resume=resume,
        )
    )


def test_full_loop_promotes_then_preserves_failures_and_exports(store, candidate, tmp_path: Path) -> None:
    coder = fake_coder(store, candidate, [8, "crash", 9])
    result = run_fake(store, coder)
    assert result["status"] == "completed"
    assert result["best"] == "0001"
    assert [a["status"] for a in result["attempts"]] == ["promoted", "failed", "rejected"]
    assert len(coder.prompts) == 6
    assert coder.peak == 1
    assert coder.live == 0
    saved = RunStore(store.root)
    assert (saved.root / "candidates/0001/sources/kernel.py").read_bytes() == candidate["sources"][0][
        "content"
    ].encode("utf-8")
    assert read_json(saved.root / "best.json")["candidate"] == "0001"
    exported = saved.export(tmp_path / "export")
    assert (exported / "sources/kernel.py").read_text() == candidate["sources"][0]["content"]
    assert (exported / "evidence/evaluation/result.json").exists()
    assert (exported / "task/definition.json").exists()
    assert read_json(exported / "checksums.json")
    with pytest.raises(ContractError, match="exists"):
        saved.export(exported)


def test_resume_counts_interrupted_attempt_and_keeps_elapsed_budget(store, candidate) -> None:
    coder = fake_coder(store, candidate, [8, asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        run_fake(store, coder)
    interrupted = RunStore(store.root)
    assert interrupted.manifest["status"] == "interrupted"
    assert interrupted.manifest["best"] == "0001"
    assert len(interrupted.manifest["attempts"]) == 2
    elapsed = interrupted.manifest["elapsed_seconds"]
    second = fake_coder(interrupted, candidate, [7])
    resumed = run_fake(interrupted, second, resume=True)
    assert resumed["best"] == "0003"
    assert len(resumed["attempts"]) == 3
    assert resumed["elapsed_seconds"] > elapsed
    assert len(second.prompts) == 2


def test_early_agent_failure_counts_toward_patience(store, candidate) -> None:
    coder = fake_coder(
        store, candidate, [RuntimeError("agent unavailable"), RuntimeError("agent unavailable")]
    )
    result = run_fake(store, coder)
    assert result["stop_reason"] == "patience"
    assert result["best"] is None
    assert len(result["attempts"]) == 2
    with pytest.raises(ContractError, match="No candidate"):
        RunStore(store.root).export(store.root.parent / "export")


def test_worker_timeout_is_failure_and_cleanup_finishes(store, candidate) -> None:
    path = store.root / "test-solution.json"
    candidate["description"] = "timeout"
    atomic_json(path, candidate)
    with pytest.raises(ContractError, match="timed out"):
        asyncio.run(evaluate(store, store.root / "timeout", candidate=path, timeout=1))
    assert not list((store.root / "timeout").glob("result.json"))


def test_environment_change_refuses_measurement_before_agent_turn(store, candidate) -> None:
    store.manifest["environment"] = {"identity": {"hardware": "different"}}
    store.save()
    coder = fake_coder(store, candidate, [])
    with pytest.raises(ContractError, match="environment changed"):
        run_fake(store, coder)
    assert not coder.prompts


def test_run_lock_refuses_concurrent_writer(store) -> None:
    with run_lock(store.root), pytest.raises(ContractError, match="Another process"), run_lock(store.root):
        pass


def test_export_rejects_mutated_winning_source(store, candidate, tmp_path: Path) -> None:
    run_fake(store, fake_coder(store, candidate, [8, 9, 9]))
    saved = RunStore(store.root)
    atomic_json(saved.root / "candidates/0001/solution.json", {"changed": True})
    with pytest.raises(ContractError, match="changed"):
        saved.export(tmp_path / "export")


def test_export_rejects_mutated_winning_evidence(store, candidate, tmp_path: Path) -> None:
    run_fake(store, fake_coder(store, candidate, [8, 9, 9]))
    saved = RunStore(store.root)
    atomic_json(saved.root / "candidates/0001/evaluation/result.json", {"status": "forged"})
    with pytest.raises(ContractError, match="evidence changed"):
        saved.export(tmp_path / "export")


def test_export_stays_available_after_tool_update(store, candidate, tmp_path: Path, monkeypatch) -> None:
    run_fake(store, fake_coder(store, candidate, [8, 9, 9]))
    saved = RunStore(store.root)
    monkeypatch.setattr("scripts.kernel_tuning.artifacts.tool_identity", lambda: "new tool version")
    with pytest.raises(ContractError, match="source changed"):
        saved.verify()
    exported = saved.export(tmp_path / "export")
    assert (exported / "tool/scripts/kernel_tuning/flow.py").is_file()


def test_crash_during_attempt_preparation_remains_counted(store, monkeypatch) -> None:
    def failed_copy(*args, **kwargs):
        raise OSError("interrupted file copy")

    monkeypatch.setattr("scripts.kernel_tuning.artifacts.shutil.copytree", failed_copy)
    with pytest.raises(OSError, match="interrupted"):
        store.begin_attempt()
    resumed = RunStore(store.root)
    assert resumed.manifest["pending"] == "0001"
    resumed.recover()
    resumed.save()
    assert resumed.manifest["attempts"][0]["status"] == "interrupted"
    assert resumed.manifest["streak"] == 1
