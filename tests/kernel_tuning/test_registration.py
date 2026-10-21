"""CPU registration contracts, Git isolation and real worker failure evidence."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.kernel_tuning.artifacts import REPOSITORY, atomic_json
from scripts.kernel_tuning.contracts import ContractError, read_json
from scripts.kernel_tuning.registration import checks
from scripts.kernel_tuning.registration.cli import add_parser, dispatch
from scripts.kernel_tuning.registration.cli import test_python as parse_python
from scripts.kernel_tuning.registration.finalize import finalize
from scripts.kernel_tuning.registration.sites import locate, validate_review
from scripts.kernel_tuning.registration.spec import RegistrationSpec, load_spec
from scripts.kernel_tuning.registration.workspace import (
    REFERENCE,
    REGISTRY,
    ROUTED_EXAMPLE,
    RegistrationRun,
    check_scope,
    check_unregistered,
    command,
    file_hash,
    git,
    tracked_hashes,
)

SOURCE = "embodiinfer/policies/pi05/modeling_pi05.py"
NEW_TEST = "tests/test_registry_matmul_t.py"


def specification(**overrides) -> RegistrationSpec:
    """Use an existing unfused registry gap without loading a model/checkpoint."""
    value = {
        "model": "pi05",
        "op": "matmul_t",
        "sites": [{"source": SOURCE, "function": "_fused_qkv", "expression": "F.linear(x, weight)"}],
        "budget": {"max_attempts": 5, "max_seconds": 120, "agent_timeout_seconds": 10},
    }
    value.update(overrides)
    return RegistrationSpec.parse(value)


def test_new_targets_selected_repository(clean_repository, tmp_path, monkeypatch, capsys) -> None:
    """An uncommitted controller can create worktrees in a separate clean checkout."""

    async def preflight(*args, **kwargs):
        pass

    monkeypatch.setattr("scripts.kernel_tuning.registration.workspace.check_unregistered", preflight)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    output = tmp_path / "runs"
    args = parser.parse_args(
        [
            "register",
            "new",
            "--repository",
            str(clean_repository),
            "--output",
            str(output),
            "--model",
            "pi05",
            "--op",
            "matmul_t",
            "--site",
            f"{SOURCE}::_fused_qkv::F.linear(x, weight)",
        ]
    )
    assert dispatch(args) == 0
    run = RegistrationRun(next(path.parent for path in output.glob("*/manifest.json")))
    assert Path(git(run.worktree, "rev-parse", "--git-common-dir").strip()) == clean_repository / ".git"
    assert "Run:" in capsys.readouterr().out


def test_baseline_hashes_track_links_without_following_targets(tmp_path, monkeypatch) -> None:
    """POSIX documentation links are valid baseline entries, but still immutable."""
    root = tmp_path / "repo"
    root.mkdir()
    target = tmp_path / "asset"
    target.write_text("first", encoding="utf-8")
    link = root / "asset-link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Creating symlinks requires platform permissions")
    monkeypatch.setattr(
        "scripts.kernel_tuning.registration.workspace.git",
        lambda *args: f"120000 {'0' * 40} 0\tasset-link\0",
    )
    first = tracked_hashes(root)
    target.write_text("second", encoding="utf-8")
    assert tracked_hashes(root) == first
    link.unlink()
    link.symlink_to(tmp_path / "missing")
    assert tracked_hashes(root) != first
    with pytest.raises(ContractError):
        checks.source_path(root, "asset-link")


def test_baseline_hashes_bind_gitlinks_without_initializing_submodules(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    first, second = "1" * 40, "2" * 40
    name = "skills/example"
    git(root, "update-index", "--add", "--cacheinfo", f"160000,{first},{name}")
    expected = {name: f"gitlink:{first}"}
    assert tracked_hashes(root) == expected  # Uninitialized checkout: no directory yet.
    (root / name).mkdir(parents=True)
    (root / name / "SKILL.md").write_text("Submodule contents", encoding="utf-8")
    assert tracked_hashes(root) == expected  # Parent snapshots the gitlink, not its files.
    git(root, "update-index", "--cacheinfo", f"160000,{second},{name}")
    assert tracked_hashes(root) == {name: f"gitlink:{second}"}


@pytest.fixture(scope="module")
def clean_repository(tmp_path_factory) -> Path:
    """Reuse existing commit objects in a disposable clone; create no commits."""
    destination = tmp_path_factory.mktemp("registration-repository") / "repo"
    subprocess.run(
        ["git", "clone", "--shared", "--quiet", str(REPOSITORY), str(destination)],
        check=True,
        capture_output=True,
    )
    return destination


@pytest.fixture
def registration_run(clean_repository, tmp_path, monkeypatch) -> RegistrationRun:
    """Exercise real worktrees; registry import is covered separately without Torch."""

    async def preflight(*args, **kwargs):
        pass

    monkeypatch.setattr("scripts.kernel_tuning.registration.workspace.check_unregistered", preflight)
    run = asyncio.run(
        RegistrationRun.create(clean_repository, tmp_path / "runs", specification(), sys.executable)
    )
    run.prepare("codex/fake:low")
    return run


def write_candidate(run: RegistrationRun) -> None:
    """Construct a valid structural patch, without executing a model."""
    source = (run.worktree / SOURCE).read_text(encoding="utf-8")
    source = source.replace(
        "def _fused_qkv(attn: Any, x: torch.Tensor)",
        "def _fused_qkv(attn: Any, x: torch.Tensor, kernels=None)",
    )
    start, end = source.index("def _fused_qkv("), source.index("def _prepare_fused_projection_weights(")
    source = (
        source[:start]
        + source[start:end].replace("F.linear(x, weight)", '_op(kernels, "matmul_t", False)(x, weight)')
        + source[end:]
    )
    source = source.replace("_fused_qkv(attn, h)", "_fused_qkv(attn, h, kernels=self._kernels)")
    (run.worktree / SOURCE).write_text(source, encoding="utf-8")
    with (run.worktree / REFERENCE).open("a", encoding="utf-8") as stream:
        stream.write(
            '\n\ndef matmul_t(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:\n    """Preserve the original bias-free F.linear expression."""\n    return F.linear(x, weight)\n'
        )
    with (run.worktree / REGISTRY).open("a", encoding="utf-8") as stream:
        stream.write('\nregister_lazy_op("matmul_t", "torch", "embodiinfer.backend.torch.ops", "matmul_t")\n')
    (run.worktree / NEW_TEST).write_text("def test_new_reference():\n    assert True\n", encoding="utf-8")
    (run.worktree / "PLAN.md").write_text("Register the unchanged F.linear expression.", encoding="utf-8")
    atomic_json(
        run.worktree / "REGISTRATION.json",
        {
            "op": "matmul_t",
            "reference": f"{REFERENCE}::matmul_t",
            "sites": [
                {
                    "id": "S1",
                    "file": SOURCE,
                    "function": "_fused_qkv",
                    "call": '_op(kernels, "matmul_t", False)(x, weight)',
                }
            ],
            "preserved": [],
            "tests": [f"{NEW_TEST}::test_new_reference"],
        },
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"op": "linear.v2"},
        {"op": "for"},
        {"op": "pi05_linear"},
        {"op": "_linear"},
        {"budget": {"max_attempts": 0}},
        {"budget": {"max_seconds": True}},
        {"budget": {"unknown": 2}},
        {"tests": ["--help"]},
        {"tests": ["../test.py"]},
        {"sites": []},
        {"unexpected": True},
    ],
)
def test_spec_rejects_invalid_contracts(overrides) -> None:
    with pytest.raises(ContractError):
        specification(**overrides)


def test_yaml_and_cli_share_validation(tmp_path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text(
        f"model: pi05\nop: matmul_t\nsites:\n  - source: {SOURCE}\n    function: _fused_qkv\n    expression: F.linear(x, weight)\n",
        encoding="utf-8",
    )
    yaml_spec = load_spec(path, model=None, op=None, sites=[])
    cli_spec = load_spec(
        None, model="pi05", op="matmul_t", sites=[f"{SOURCE}::_fused_qkv::F.linear(x, weight)"]
    )
    assert yaml_spec == cli_spec
    with pytest.raises(ContractError):
        load_spec(path, model="pi05", op=None, sites=[])
    with pytest.raises(ContractError):
        parse_python(["test_python=python -c bad"])
    assert parse_python([f"test_python={sys.executable}"]) == sys.executable


def test_locator_qualification_guards_and_call_chain(tmp_path) -> None:
    path = tmp_path / SOURCE
    path.parent.mkdir(parents=True)
    path.write_text(
        """def _op(kernels, op, triton): pass
def leaf(x, weight):
    if quantized:
        return module(x)
    if enabled:
        return F.linear(x, weight)
    return x
def middle(x, weight):
    return leaf(x, weight)
class Policy:
    def run(self, x, weight):
        self._kernels = kernels
        return middle(x, weight)
""",
        encoding="utf-8",
    )
    spec = specification(sites=[{"source": SOURCE, "function": "leaf", "expression": "F.linear( x,weight )"}])
    found = locate(tmp_path, spec)
    site = found["sites"][0]
    assert site["line"] == 6 and site["column"] == 15
    assert {g["condition"] for g in site["guards"]} == {"quantized", "enabled"}
    assert {c["function"] for c in site["callers"]} == {"middle", "Policy.run"}
    assert site["kernels"].startswith("add kernels")
    modified = copy.deepcopy(found)
    modified["preserve"].append("Preserve tensor parallel module calls")
    validate_review(modified, found)
    modified["sites"][0]["guards"] = []
    with pytest.raises(ContractError):
        validate_review(modified, found)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\ndef leaf(x, weight):\n    return F.linear(x, weight)\n")
    with pytest.raises(ContractError, match="definitions"):
        locate(tmp_path, spec)


def test_duplicate_expression_is_not_guessed(tmp_path) -> None:
    path = tmp_path / SOURCE
    path.parent.mkdir(parents=True)
    path.write_text(
        "def _op(*args): pass\ndef _fused_qkv(x, weight):\n    a = F.linear(x, weight)\n    return F.linear(x, weight)\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="2 sites"):
        locate(tmp_path, specification())


def test_new_requires_clean_checkout(clean_repository, tmp_path) -> None:
    unrelated = clean_repository / "untracked.txt"
    unrelated.write_text("keep", encoding="utf-8")
    try:
        with pytest.raises(ContractError, match="clean"):
            asyncio.run(RegistrationRun.create(clean_repository, tmp_path, specification(), sys.executable))
        assert unrelated.read_text() == "keep"
    finally:
        unrelated.unlink()


def test_scope_catches_untracked_staged_and_protected_files(registration_run) -> None:
    run = registration_run
    write_candidate(run)
    assert NEW_TEST in check_scope(run)["files"]
    assert "new file mode" in run.patch() and "test_new_reference" in run.patch()
    bad = run.worktree / "scripts/unexpected.py"
    bad.write_text("pass\n", encoding="utf-8")
    with pytest.raises(ContractError, match="out-of-scope"):
        check_scope(run)
    git(run.worktree, "add", "--", "scripts/unexpected.py")
    with pytest.raises(ContractError, match="out-of-scope"):
        check_scope(run)
    git(run.worktree, "restore", "--staged", "--", "scripts/unexpected.py")
    bad.unlink()
    registry = run.worktree / REGISTRY
    registry.write_text(
        registry.read_text(encoding="utf-8").replace("self.mode = ", 'self.mode = "changed" or ', 1),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="KernelSelection"):
        check_scope(run)


def test_existing_test_body_cannot_be_disabled_by_insertions(registration_run) -> None:
    run = registration_run
    tests = run.worktree / "tests/test_kernel_ops.py"
    text = tests.read_text(encoding="utf-8")
    import ast

    function = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef))
    lines = text.splitlines(keepends=True)
    lines.insert(function.body[0].lineno - 1, "    return\n")
    tests.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(ContractError, match="statements changed"):
        check_scope(run)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing_reference", "definitions"),
        ("loop", "top-level"),
        ("old_expression", "original expression"),
        ("triton", "Torch preference"),
        ("guard", "preserve branch"),
        ("backend_import", "standard library"),
        ("global", "KernelSelection"),
        ("old_test", "already existed"),
    ],
)
def test_structure_rejects_misregistration(registration_run, mutation, message) -> None:
    run = registration_run
    write_candidate(run)
    assert checks.check_structure(run)["sites"] == ["S1"]
    if mutation == "missing_reference":
        shutil.copyfile(run.base / REFERENCE, run.worktree / REFERENCE)
    elif mutation == "loop":
        p = run.worktree / REGISTRY
        text = p.read_text(encoding="utf-8")
        line = 'register_lazy_op("matmul_t", "torch", "embodiinfer.backend.torch.ops", "matmul_t")'
        p.write_text(text.replace(line, "for _ in [1]:\n    " + line), encoding="utf-8")
    elif mutation in {"old_expression", "guard", "global"}:
        p = run.worktree / SOURCE
        text = p.read_text(encoding="utf-8")
        if mutation == "old_expression":
            text = text.replace(
                'fused = _op(kernels, "matmul_t", False)(x, weight)',
                'unused = F.linear(x, weight)\n    fused = _op(kernels, "matmul_t", False)(x, weight)',
            )
        elif mutation == "guard":
            text = text.replace("if torch.is_grad_enabled() or attn.training:", "if attn.training:")
        else:
            text = text.replace("x: torch.Tensor, kernels=None)", "x: torch.Tensor)")
        p.write_text(text, encoding="utf-8")
    elif mutation == "backend_import":
        with (run.worktree / REFERENCE).open("a", encoding="utf-8") as stream:
            stream.write("\nfrom embodiinfer.engine import EngineCore\n")
    else:
        p = run.worktree / "REGISTRATION.json"
        document = read_json(p)
        if mutation == "triton":
            document["sites"][0]["call"] = '_op(kernels, "matmul_t", True)(x, weight)'
        else:
            import ast

            name = next(
                n.name
                for n in ast.parse((run.base / "tests/test_kernel_ops.py").read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
            )
            document["tests"] = [f"tests/test_kernel_ops.py::{name}"]
        atomic_json(p, document)
    with pytest.raises(ContractError, match=message):
        checks.check_structure(run)


def test_integrity_and_baseline_overlay_recovery(registration_run) -> None:
    run = registration_run
    target = run.base / NEW_TEST
    data = "def test_overlay(): pass\n"
    target.write_text(data, encoding="utf-8")
    run.manifest["base_overlay"] = {NEW_TEST: file_hash(target)}
    run.save()
    recovered = RegistrationRun(run.root)
    recovered.recover_base()
    recovered.verify()
    assert not target.exists()
    target.write_text("unexpected", encoding="utf-8")
    with pytest.raises(ContractError, match="RUN/base"):
        recovered.verify()


def test_context_example_comes_from_base(registration_run) -> None:
    """The routed example is a BASE snapshot, never a commit that history may lose."""
    context = registration_run.root / "context"
    assert (context / ROUTED_EXAMPLE).read_bytes() == (registration_run.base / ROUTED_EXAMPLE).read_bytes()
    assert not (context / "migration.diff").exists()


def test_spec_tamper_is_fatal(registration_run) -> None:
    run = registration_run
    document = read_json(run.root / "spec.json")
    document["spec"]["op"] = "changed"
    atomic_json(run.root / "spec.json", document)
    with pytest.raises(ContractError, match="specification"):
        run.verify()
    assert read_json(run.root / "manifest.json")["status"] == "failed"
    with pytest.raises(ContractError, match="specification"):
        RegistrationRun(run.root)


def test_finalize_only_exports_verified_bytes(registration_run) -> None:
    run = registration_run
    write_candidate(run)
    with pytest.raises(ContractError, match="fully checked"):
        finalize(run)
    run.manifest.update(status="passed", verified_fingerprint=run.fingerprint())
    head = git(run.worktree, "rev-parse", "HEAD")
    result = finalize(run)
    assert result["commit"] is None and not result["pushed"]
    assert git(run.worktree, "rev-parse", "HEAD") == head
    assert "NOT been run" in (run.root / "SUMMARY.md").read_text(encoding="utf-8")
    with pytest.raises(ContractError, match="requires --commit"):
        finalize(run, push=True)
    with (run.worktree / NEW_TEST).open("a", encoding="utf-8") as stream:
        stream.write("\n# changed after checks\n")
    with pytest.raises(ContractError, match="unchanged"):
        finalize(run)


def minimal_package(root: Path) -> None:
    """Make a CPU-only stand-in to test subprocess/import behavior, not model math."""
    for directory in (
        "embodiinfer",
        "embodiinfer/layers",
        "embodiinfer/backend",
        "embodiinfer/backend/torch",
        "tests",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
        (root / directory / "__init__.py").write_text("", encoding="utf-8")
    (root / REFERENCE).write_text("", encoding="utf-8")
    (root / "embodiinfer/layers/__init__.py").write_text(
        "def available_ops():\n    return {}\n", encoding="utf-8"
    )


def test_registry_preflight_uses_selected_checkout(tmp_path, monkeypatch) -> None:
    minimal_package(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(REPOSITORY))
    asyncio.run(check_unregistered(tmp_path, "linear", sys.executable, tmp_path / "probe.log"))
    (tmp_path / "embodiinfer/layers/__init__.py").write_text(
        "def available_ops():\n    return {'linear': ['torch']}\n", encoding="utf-8"
    )
    with pytest.raises(ContractError, match="already registered"):
        asyncio.run(check_unregistered(tmp_path, "linear", sys.executable, tmp_path / "probe2.log"))


def test_real_registry_resolution_in_isolated_interpreter(registration_run) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    run = registration_run
    asyncio.run(check_unregistered(run.base, "matmul_t", sys.executable, run.root / "real-preflight.log"))
    write_candidate(run)
    result = asyncio.run(checks.registry_probe(run, run.root / "real-registry.log", timeout=30))
    assert result["returncode"] == 0


@pytest.mark.parametrize(
    "body, effective",
    [
        ("assert False", True),
        ("assert True", False),
        ("import pytest; pytest.skip('not tested')", False),
        ("import unavailable_dependency", False),
    ],
)
def test_c5_requires_each_test_to_execute_and_fail(tmp_path, monkeypatch, body, effective) -> None:
    base, worktree = tmp_path / "base", tmp_path / "worktree"
    minimal_package(base)
    minimal_package(worktree)
    (worktree / NEW_TEST).write_text(f"def test_gate():\n    {body}\n", encoding="utf-8")
    run = SimpleNamespace(
        base=base,
        worktree=worktree,
        manifest={"test_python": sys.executable},
        verify=lambda: None,
        save=lambda: None,
    )

    def restore():
        (base / NEW_TEST).unlink(missing_ok=True)
        run.manifest["base_overlay"] = {}

    run.recover_base = restore
    monkeypatch.setattr(checks, "registration", lambda _: {"tests": [f"{NEW_TEST}::test_gate"]})
    if effective:
        result = asyncio.run(checks.baseline_tests(run, tmp_path / "checks", time.monotonic() + 30))
        assert result["commands"][0]["returncode"] == 1
    else:
        with pytest.raises(ContractError, match="C5"):
            asyncio.run(checks.baseline_tests(run, tmp_path / "checks", time.monotonic() + 30))
    assert not (base / NEW_TEST).exists() and not run.manifest["base_overlay"]


def test_c1_failure_stops_later_checks(registration_run, monkeypatch) -> None:
    run = registration_run
    (run.worktree / "unexpected.txt").write_text("bad", encoding="utf-8")
    monkeypatch.setattr(checks, "check_structure", lambda _: pytest.fail("C2 must not run"))
    results, feedback = asyncio.run(checks.run_checks(run, run.root / "attempts/0001", seconds=30))
    assert [r["check"] for r in results] == ["C1"] and feedback["check"] == "C1"


def test_worker_timeout_cleans_process(tmp_path) -> None:
    with pytest.raises(TimeoutError):
        asyncio.run(
            command(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                root=tmp_path,
                output=tmp_path / "timeout.log",
                timeout=0.2,
            )
        )


def test_status_never_loads_agent(registration_run, capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.kernel_tuning.registration.cli.execute",
        lambda *a, **kw: pytest.fail("status started an agent"),
    )
    assert dispatch(SimpleNamespace(registration_command="status", run=registration_run.root)) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"


@pytest.fixture
def humanize(monkeypatch):
    """Use the official fake agents, adapting only POSIX journal IO on Windows."""
    pytest.importorskip("hmz")
    import os
    import tempfile

    from hmz.sdk import fakes

    if os.name == "nt":
        from hmz.coganchor import atomic

        def writes(at, said, *, mode=None):
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
    return fakes


def fake_flow(run, humanize, coder, *, resume=False):
    """Drive the real resumable flow and its framework state/session machinery."""
    from scripts.kernel_tuning.registration.flow import operator_registration

    return asyncio.run(
        humanize.run_fake(
            operator_registration,
            "register test only",
            agents={"coder": coder},
            local=humanize.FakeEnvDriver(workdir=run.root.as_posix()),
            params={"run_dir": str(run.root)},
            journal=run.root / "fake-hmz.jsonl",
            resume=resume,
        )
    )


def scripted_coder(run, humanize, *, interrupt=None, sessions=None):
    """Modify a real worktree only during implementation; collect session handles."""

    def reply(instruction, **kwargs):
        if sessions is not None:
            sessions.append(id(kwargs["session"]))
        assert Path(str(kwargs["session"].placement.workdir)).resolve() == run.worktree
        if "Write PLAN.md before" in instruction:
            (run.worktree / "PLAN.md").write_text("Preserve F.linear arithmetic.", encoding="utf-8")
        else:
            if interrupt is not None:
                raise interrupt
            if not (run.worktree / "REGISTRATION.json").exists():
                write_candidate(run)
        return "done"

    return humanize.FakeAgentDriver(reply=reply)


def test_flow_feedback_repair_same_session_then_success(registration_run, humanize, monkeypatch) -> None:
    run = registration_run
    sessions, calls = [], []

    async def check(candidate, archive, *, seconds):
        assert seconds > 0
        check_scope(candidate)
        checks.check_structure(candidate)
        calls.append(archive)
        failure = (
            {"check": "C3", "summary": "synthetic formatting error", "details": {}}
            if len(calls) == 1
            else None
        )
        if failure:
            atomic_json(archive / "feedback.json", failure)
        return [{"check": "C3", "passed": failure is None, "details": {}}], failure

    monkeypatch.setattr("scripts.kernel_tuning.registration.flow.run_checks", check)
    coder = scripted_coder(run, humanize, sessions=sessions)
    result = fake_flow(run, humanize, coder)
    assert result["status"] == "passed"
    assert len(coder.prompts) == 3 and len(set(sessions)) == 1
    assert [a["status"] for a in result["attempts"]] == ["failed", "passed"]
    assert "synthetic formatting error" in (run.root / "attempts/0002/repair-prompt.txt").read_text(
        encoding="utf-8"
    )
    assert (run.root / "patch.diff").is_file() and (run.root / "SUMMARY.md").is_file()
    assert (run.root / "attempts/0001/patch.diff").is_file()
    assert coder.live == 0


def test_flow_replans_after_two_equal_failures_and_exhausts(registration_run, humanize, monkeypatch) -> None:
    run = registration_run
    sessions = []

    async def check(*args, **kwargs):
        return [{"check": "C2", "passed": False}], {"check": "C2", "summary": "synthetic", "details": {}}

    monkeypatch.setattr("scripts.kernel_tuning.registration.flow.run_checks", check)
    result = fake_flow(run, humanize, scripted_coder(run, humanize, sessions=sessions))
    assert result["status"] == "exhausted" and len(result["attempts"]) == 5
    assert sessions[0] == sessions[1] == sessions[2]
    assert sessions[3] != sessions[2]
    assert (run.root / "attempts/0003/plan-prompt.txt").is_file()
    assert (run.root / "attempts/0004/repair-prompt.txt").is_file()
    assert (run.root / "attempts/0005/plan-prompt.txt").is_file()


def test_resume_keeps_completed_plan_and_budget(registration_run, humanize, monkeypatch) -> None:
    run = registration_run

    async def check(*args, **kwargs):
        return [{"check": "C1", "passed": True, "details": {}}], None

    monkeypatch.setattr("scripts.kernel_tuning.registration.flow.run_checks", check)
    with pytest.raises(asyncio.CancelledError):
        fake_flow(run, humanize, scripted_coder(run, humanize, interrupt=asyncio.CancelledError()))
    interrupted = RegistrationRun(run.root)
    assert interrupted.manifest["status"] == "interrupted"
    assert interrupted.manifest["attempts"][0]["stage"] == "implement"
    elapsed = interrupted.manifest["elapsed_seconds"]
    coder = scripted_coder(run, humanize)
    result = fake_flow(run, humanize, coder, resume=True)
    assert result["status"] == "passed" and len(result["attempts"]) == 1
    assert len(coder.prompts) == 1 and "Implement PLAN.md" in coder.prompts[0]
    assert result["elapsed_seconds"] > elapsed


def test_checks_resume_does_not_repeat_passed_gates(registration_run, monkeypatch) -> None:
    run = registration_run
    write_candidate(run)
    archive = run.root / "attempts/0001"
    atomic_json(
        archive / "checks.json",
        {"fingerprint": run.fingerprint(), "checks": [{"check": "C1", "passed": True}]},
    )
    monkeypatch.setattr(checks, "check_scope", lambda _: pytest.fail("C1 reran"))
    monkeypatch.setattr(
        checks, "check_structure", lambda _: (_ for _ in ()).throw(ContractError("C2 intentionally fails"))
    )
    results, failure = asyncio.run(checks.run_checks(run, archive, seconds=30))
    assert [r["check"] for r in results] == ["C1", "C2"] and failure["check"] == "C2"


def test_structure_rejects_missing_kernel_forwarding(registration_run) -> None:
    run = registration_run
    write_candidate(run)
    source = run.worktree / SOURCE
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "_fused_qkv(attn, h, kernels=self._kernels)", "_fused_qkv(attn, h)"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="not forwarded"):
        checks.check_structure(run)


def test_patch_excludes_staged_artifacts_and_preserves_no_newline(registration_run) -> None:
    run = registration_run
    write_candidate(run)
    (run.worktree / NEW_TEST).write_text("def test_no_newline(): pass", encoding="utf-8")
    git(run.worktree, "add", "--", "PLAN.md", "REGISTRATION.json")
    patch = run.patch()
    assert "b/PLAN.md" not in patch and "b/REGISTRATION.json" not in patch
    assert "\\ No newline at end of file" in patch
    patch_file = run.root / "verify.diff"
    patch_file.write_text(patch, encoding="utf-8")
    git(run.base, "apply", "--check", str(patch_file))


def test_all_gates_run_real_cpu_workers(tmp_path) -> None:
    """Exercise C1–C5 together on a tiny model-free package, with no mocked workers."""
    from scripts.kernel_tuning.contracts import digest

    base, worktree = tmp_path / "base", tmp_path / "worktree"
    minimal_package(base)
    (base / SOURCE).parent.mkdir(parents=True, exist_ok=True)
    (base / SOURCE).write_text(
        """def _op(kernels, op, use_triton):
    return kernels.resolve(op)


def raw(x, weight):
    return x * weight


def compute(x, weight, kernels=None, quantized=False):
    if quantized:
        return 99
    return raw(x, weight)
""",
        encoding="utf-8",
    )
    (base / REFERENCE).write_text(
        "from __future__ import annotations\n\nimport typing\n\nif typing.TYPE_CHECKING:\n    import torch\n",
        encoding="utf-8",
    )
    (base / REGISTRY).write_text(
        """from importlib import import_module

_ops = {}


def register_lazy_op(op, impl, module, function):
    _ops[op] = (module, function)


def available_ops():
    return {op: ["torch"] for op in _ops}


def get_op(op):
    module, function = _ops[op]
    return getattr(import_module(module), function)


class KernelSelection:
    def __init__(self, mode):
        self.mode = mode

    def prepare(self):
        for op in _ops:
            self.resolve(op)

    def resolve(self, op):
        return get_op(op)
""",
        encoding="utf-8",
    )
    (base / "embodiinfer/layers/__init__.py").write_text(
        'from .ops import KernelSelection, available_ops, get_op\n\n__all__ = ["KernelSelection", "available_ops", "get_op"]\n',
        encoding="utf-8",
    )
    (base / "tests/test_kernel_ops.py").write_text(
        "def test_existing():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )
    subprocess.run([sys.executable, "-m", "ruff", "format", str(base)], check=True, capture_output=True)
    shutil.copytree(base, worktree)
    spec = specification(sites=[{"source": SOURCE, "function": "compute", "expression": "raw(x, weight)"}])
    atomic_json(tmp_path / "sites.json", locate(base, spec))
    source = worktree / SOURCE
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "return raw(x, weight)", 'return _op(kernels, "matmul_t", False)(x, weight)'
        ),
        encoding="utf-8",
    )
    with (worktree / REFERENCE).open("a", encoding="utf-8") as stream:
        stream.write(
            '\n\ndef matmul_t(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:\n    """Synthetic multiplication reference used only to test the controller."""\n    return x * weight\n'
        )
    with (worktree / REGISTRY).open("a", encoding="utf-8") as stream:
        stream.write('\nregister_lazy_op("matmul_t", "torch", "embodiinfer.backend.torch.ops", "matmul_t")\n')
    (worktree / NEW_TEST).write_text(
        """from embodiinfer.backend.torch import ops


def test_reference():
    assert hasattr(ops, "matmul_t")
    assert ops.matmul_t(2, 3) == 6


def test_route():
    from embodiinfer.policies.pi05.modeling_pi05 import compute

    calls = []

    class Recording:
        def resolve(self, op):
            calls.append(op)
            return lambda x, w: x * w

    assert compute(2, 3, Recording()) == 6
    assert calls == ["matmul_t"]
""",
        encoding="utf-8",
    )
    atomic_json(
        worktree / "REGISTRATION.json",
        {
            "op": "matmul_t",
            "reference": f"{REFERENCE}::matmul_t",
            "sites": [
                {
                    "id": "S1",
                    "file": SOURCE,
                    "function": "compute",
                    "call": '_op(kernels, "matmul_t", False)(x, weight)',
                }
            ],
            "preserved": [{"site": "S1", "branch": "quantized"}],
            "tests": [f"{NEW_TEST}::test_reference", f"{NEW_TEST}::test_route"],
        },
    )
    subprocess.run([sys.executable, "-m", "ruff", "format", str(worktree)], check=True, capture_output=True)
    changed = [SOURCE, REFERENCE, REGISTRY, NEW_TEST, "REGISTRATION.json"]
    run = SimpleNamespace(
        root=tmp_path,
        base=base,
        worktree=worktree,
        spec=spec,
        base_commit="synthetic",
        manifest={"test_python": sys.executable},
        verify=lambda: None,
        save=lambda: None,
        changes=lambda: changed,
        fingerprint=lambda: digest({n: file_hash(worktree / n) for n in changed}),
    )

    def restore():
        (base / NEW_TEST).unlink(missing_ok=True)
        run.manifest["base_overlay"] = {}

    run.recover_base = restore
    results, feedback = asyncio.run(checks.run_checks(run, tmp_path / "checks", seconds=60))
    assert feedback is None, feedback
    assert [r["check"] for r in results] == ["C1", "C2", "C3", "C4", "C5"]
    assert all(r["passed"] for r in results)
    assert not (base / NEW_TEST).exists()


@pytest.mark.parametrize("unfenced", [False, True])
def test_flow_sessions_run_under_the_recorded_fence(
    registration_run, humanize, monkeypatch, unfenced: bool
) -> None:
    from scripts.kernel_tuning.permissions import FENCED, UNFENCED

    run = registration_run
    run.manifest["unfenced"] = unfenced
    run.save()

    async def check(*args, **kwargs):
        return [{"check": "C1", "passed": True, "details": {}}], None

    monkeypatch.setattr("scripts.kernel_tuning.registration.flow.run_checks", check)
    coder = scripted_coder(run, humanize)
    assert fake_flow(run, humanize, coder)["status"] == "passed"
    expected = UNFENCED if unfenced else FENCED
    assert coder.sessions and all(session.permission == expected for session in coder.sessions)


def test_unfenced_choice_is_frozen_with_the_agent(registration_run) -> None:
    run = registration_run
    assert run.manifest["unfenced"] is False
    with pytest.raises(ContractError, match="frozen"):
        run.prepare("codex/fake:low", unfenced=True)
    run.prepare("codex/fake:low")
