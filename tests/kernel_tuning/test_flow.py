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


def run_fake(store: RunStore, coder, *, second=None, resume: bool = False) -> dict:
    """Exercise hmz journaling/session behavior; no real harness is constructed.

    One fake drives both chasers unless ``second`` gives the second chaser its own.
    """
    return asyncio.run(
        fakes.run_fake(
            kernel_tuning,
            "test only",
            agents={"first_chaser": coder, "second_chaser": second or coder},
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
    # The worker's own exception reaches the feedback, not only a log path.
    assert "RuntimeError: simulated compilation failure" in result["attempts"][1]["decision"]["reasons"][0]
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


def test_rejected_source_is_available_to_the_next_planning_turn(store, candidate) -> None:
    """A fresh agent can inspect rejected code without mistaking it for the incumbent."""
    witnessed = []

    def answer(prompt: str, **kwargs: Any) -> str:
        scratch = Path(str(kwargs["session"].placement.workdir))
        if "Write PLAN.md" in prompt:
            if scratch.name == "0002":
                rejected = read_json(scratch / "feedback.json")[0]
                assert rejected["status"] == "rejected"
                assert rejected["solution"]["name"] == candidate["name"] + "_0001"
                assert rejected["solution"]["sources"] == candidate["sources"]
                assert read_json(scratch / "incumbent.json")["name"] != rejected["solution"]["name"]
                witnessed.append(rejected["id"])
            (scratch / "PLAN.md").write_text("Revise the previous implementation.", encoding="utf-8")
        else:
            solution = copy.deepcopy(candidate)
            solution["name"] += "_" + scratch.name
            solution["description"] = {"0001": "12", "0002": "8", "0003": "9"}[scratch.name]
            atomic_json(scratch / "solution.json", solution)
        return "ready"

    result = run_fake(store, fakes.FakeAgentDriver(reply=answer))
    assert witnessed == ["0001"]
    assert result["best"] == "0002"


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
    assert read_json(store.root / "timeout/command.json")["device_lock_wait_seconds"] >= 0


@pytest.mark.skipif(os.name == "nt", reason="the device lock is a POSIX flock")
def test_evaluator_workers_take_turns_on_the_device() -> None:
    """Concurrent runs overlap only agent turns: a second worker starts once the first is done."""
    from scripts.kernel_tuning.runner import device_lock

    order = []

    async def hold(name: str) -> None:
        async with device_lock("cuda:test-lock", poll=0.01):
            order.append(f"{name} start")
            await asyncio.sleep(0.2)
            order.append(f"{name} end")

    async def both() -> None:
        await asyncio.gather(hold("a"), hold("b"))

    asyncio.run(both())
    assert order in (["a start", "a end", "b start", "b end"], ["b start", "b end", "a start", "a end"])


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
    assert (exported / "tool/scripts/kernel_tuning/flow/__init__.py").is_file()


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


def test_chasers_take_alternate_attempts_and_say_so(store, candidate) -> None:
    """Attempt n belongs to CHASERS[(n - 1) % 2]; each plan prompt names its chaser."""
    first = fake_coder(store, candidate, [8, 9])
    second = fake_coder(store, candidate, [9])
    result = run_fake(store, first, second=second)
    assert [a["chaser"] for a in result["attempts"]] == ["first_chaser", "second_chaser", "first_chaser"]
    assert len(first.prompts) == 4
    assert len(second.prompts) == 2
    assert "You are the second_chaser" in second.prompts[0]
    assert "You are the first_chaser" in first.prompts[2]


def test_prompts_state_the_turn_deadline(store) -> None:
    """A turn stopped before its deliverable exists fails, so each prompt says when that is."""
    from scripts.kernel_tuning.prompts import implementation_prompt, plan_prompt

    minutes = f"{store.task.settings.search.agent_timeout_seconds / 60:g} minutes"
    assert minutes in plan_prompt(store.task) and "before PLAN.md" in plan_prompt(store.task)
    assert minutes in implementation_prompt(store.task)
    assert "before solution.json" in implementation_prompt(store.task)


def test_turn_stopped_after_its_deliverable_still_counts(store, candidate) -> None:
    """A stopped turn whose deliverable exists goes on in a fresh session; without one it fails."""

    def answer(prompt: str, **kwargs: Any) -> str:
        scratch = Path(str(kwargs["session"].placement.workdir))
        if "Write PLAN.md" in prompt:
            if scratch.name != "0002":
                (scratch / "PLAN.md").write_text("Fewer launches.", encoding="utf-8")
        else:
            solution = copy.deepcopy(candidate)
            solution["name"] += "_" + scratch.name
            solution["description"] = "8"
            atomic_json(scratch / "solution.json", solution)
        raise TimeoutError

    coder = fakes.FakeAgentDriver(reply=answer)
    result = run_fake(store, coder)
    assert [a["status"] for a in result["attempts"]] == ["promoted", "failed", "rejected"]
    assert result["attempts"][1]["decision"]["reasons"][0].startswith("TimeoutError")
    reply = (store.root / "candidates/0001/plan-reply.txt").read_text(encoding="utf-8")
    assert reply.startswith("Turn stopped") and "PLAN.md written" in reply
    # Attempts 1 and 3 continue in a second session after each stopped plan turn.
    assert len(coder.sessions) == 5


def test_chaser_agents_accept_one_or_two_specs() -> None:
    from scripts.kernel_tuning.__main__ import chaser_agents

    assert chaser_agents("codex/m:high") == {"first_chaser": "codex/m:high", "second_chaser": "codex/m:high"}
    assert chaser_agents("claude@gw/m:high, codex@gw/m:high") == {
        "first_chaser": "claude@gw/m:high",
        "second_chaser": "codex@gw/m:high",
    }
    for bad in ("a/m,b/m,c/m", "a/m,", ""):
        with pytest.raises(ContractError, match="comma"):
            chaser_agents(bad)


def test_profiles_of_incumbent_and_previous_attempt_are_lent(store) -> None:
    """Agents read NCU evidence copies; request/response records stay in the archive."""
    baseline = store.profile_path()
    baseline.mkdir(parents=True)
    (baseline / "details.txt").write_text("baseline counters", encoding="utf-8")
    (baseline / "request.json").write_text("{}", encoding="utf-8")
    _, _, scratch = store.begin_attempt()
    assert (scratch / "profiles/incumbent/details.txt").read_text(encoding="utf-8") == "baseline counters"
    assert not (scratch / "profiles/incumbent/request.json").exists()
    assert not (scratch / "profiles/previous").exists()
    store.finish_attempt("0001", status="failed", decision={"promoted": False, "reasons": ["x"]})
    previous = store.profile_path("0001")
    previous.mkdir(parents=True)
    (previous / "error.json").write_text('{"error": "compile"}', encoding="utf-8")
    _, _, scratch = store.begin_attempt()
    assert (scratch / "profiles/incumbent/details.txt").is_file()
    assert (scratch / "profiles/previous/error.json").is_file()


@pytest.mark.parametrize("unfenced", [False, True])
def test_sessions_run_under_the_recorded_fence(store, candidate, unfenced: bool) -> None:
    """Fenced runs keep hmz's default grant; only a recorded --unfenced run widens it."""
    from scripts.kernel_tuning.permissions import FENCED, UNFENCED

    store.manifest["unfenced"] = unfenced
    store.save()
    coder = fake_coder(store, candidate, [8, 9, 9])
    run_fake(store, coder)
    assert coder.sessions
    expected = UNFENCED if unfenced else FENCED
    assert all(session.permission == expected for session in coder.sessions)


def profiled_store(task_dir: Path, tmp_path: Path, *, available: bool = True) -> RunStore:
    """Freeze a CPU profile adapter before starting either chaser."""
    import yaml
    from scripts.kernel_tuning.contracts import TaskPackage

    config = yaml.safe_load((task_dir / "tuning.yaml").read_text(encoding="utf-8"))
    config["profile"] = True
    (task_dir / "tuning.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    if available:
        path = task_dir / "benchmark.py"
        path.write_text(
            path.read_text(encoding="utf-8")
            + """
def profile(self, solutions, output):
    (output / "details.txt").write_text(solutions["candidate"]["name"])
    return {"details": str(output / "details.txt")}
Adapter.profile = profile
""",
            encoding="utf-8",
        )
    return RunStore.create(TaskPackage.load(task_dir), tmp_path / "profile-runs", "claude/fake:low")


def test_baseline_and_every_candidate_profiles_reach_chasers(task_dir, tmp_path, candidate) -> None:
    store = profiled_store(task_dir, tmp_path)
    result = run_fake(store, fake_coder(store, candidate, [8, 9, 9]))
    for label in (None, "0001", "0002", "0003"):
        assert (store.profile_path(label) / "result.json").is_file()
    baseline = read_json(store.solution_path())["name"]
    assert (store.root / "workspace/0001/profiles/incumbent/details.txt").read_text() == baseline
    assert (store.root / "workspace/0003/profiles/previous/details.txt").read_text().endswith("_0002")
    assert result["best"] == "0001"
    RunStore(store.root).verify()


def test_profile_failure_preserves_promotion_and_reaches_next_agent(task_dir, tmp_path, candidate) -> None:
    store = profiled_store(task_dir, tmp_path, available=False)
    result = run_fake(store, fake_coder(store, candidate, [8, 9, 9]))
    assert result["best"] == "0001"
    for label in (None, "0001", "0002", "0003"):
        assert "does not implement" in read_json(store.profile_path(label) / "error.json")["error"]
    assert (store.root / "workspace/0002/profiles/incumbent/error.json").is_file()
    RunStore(store.root).verify()


def test_interrupted_baseline_profile_retries_on_resume(task_dir, tmp_path, candidate, monkeypatch) -> None:
    store = profiled_store(task_dir, tmp_path)
    interrupted = False

    async def cancel_once(*args, **kwargs):
        nonlocal interrupted
        if kwargs.get("operation") == "profile" and not interrupted:
            interrupted = True
            raise asyncio.CancelledError()
        return await evaluate(*args, **kwargs)

    monkeypatch.setattr("scripts.kernel_tuning.flow.evaluate", cancel_once)
    with pytest.raises(asyncio.CancelledError):
        run_fake(store, fake_coder(store, candidate, []))
    resumed = RunStore(store.root)
    result = run_fake(resumed, fake_coder(resumed, candidate, [8, 9, 9]), resume=True)
    assert (store.profile_path() / "result.json").is_file()
    assert result["best"] == "0001"
