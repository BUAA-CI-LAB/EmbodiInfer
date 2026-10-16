"""Ordered controller checks; candidate claims never substitute for execution."""

from __future__ import annotations

import ast
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from scripts.kernel_tuning.artifacts import atomic_json
from scripts.kernel_tuning.contracts import ContractError, read_json

from .sites import body_nodes, branch_dump, function_node, functions, source_path
from .spec import test_node
from .workspace import REFERENCE, REGISTRY, RegistrationRun, check_scope, command, file_hash, import_guard


def registration(run: RegistrationRun) -> dict[str, Any]:
    """Validate agent output and require newly introduced executable pytest nodes."""
    document = read_json(source_path(run.worktree, "REGISTRATION.json"))
    if not isinstance(document, dict):
        raise ContractError("REGISTRATION.json must be an object")
    if document.get("op") != run.spec.op or document.get("reference") != f"{REFERENCE}::{run.spec.op}":
        raise ContractError("REGISTRATION.json names the wrong operator/reference")
    tests = document.get("tests")
    if (
        not isinstance(tests, list)
        or not tests
        or not all(isinstance(t, str) for t in tests)
        or len(tests) != len(set(tests))
    ):
        raise ContractError("REGISTRATION.json must list unique new test node IDs")
    allowed = {"tests/test_kernel_ops.py", f"tests/test_registry_{run.spec.op}.py"}
    for item in tests:
        parts = test_node(item).split("::")
        if len(parts) not in (2, 3) or parts[0] not in allowed or "[" in item:
            raise ContractError("List whole new test functions (all parametrizations), not files or cases")
        name = ".".join(parts[1:])
        if not parts[-1].startswith("test_"):
            raise ContractError(f"Not an executable test node: {item}")
        tree = ast.parse(source_path(run.worktree, parts[0]).read_text(encoding="utf-8"))
        function_node(tree, name)
        if (run.base / parts[0]).exists():
            old = ast.parse((run.base / parts[0]).read_text(encoding="utf-8"))
            if any(qualified == name for qualified, _ in functions(old)):
                raise ContractError(f"Test already existed at BASE: {item}")
    sites = document.get("sites")
    expected = read_json(run.root / "sites.json")["sites"]
    if (
        not isinstance(sites, list)
        or len(sites) != len(expected)
        or not all(isinstance(s, dict) and isinstance(s.get("id"), str) for s in sites)
        or {s.get("id") for s in sites} != {s["id"] for s in expected}
    ):
        raise ContractError("REGISTRATION.json must describe every selected site exactly once")
    if not isinstance(document.get("preserved"), list):
        raise ContractError("REGISTRATION.json must include preserved branch evidence")
    for item in document["preserved"]:
        if (
            not isinstance(item, dict)
            or item.get("site") not in {s["id"] for s in expected}
            or not isinstance(item.get("branch"), str)
        ):
            raise ContractError("Malformed preserved branch evidence")
    return document


def _annotation(node: ast.expr | None, *, result: bool = False) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return _annotation(ast.parse(node.value, mode="eval").body, result=result)
        return node.value is Ellipsis if result else node.value is None
    if isinstance(node, ast.Name):
        return node.id in ({"Tensor"} if result else {"Tensor", "int", "float", "bool", "str"})
    if isinstance(node, ast.Attribute):
        return ast.unparse(node) == "torch.Tensor"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation(node.left, result=result) and _annotation(node.right, result=result)
    if isinstance(node, ast.Subscript) and ast.unparse(node.value) in (
        {"tuple", "Tuple"} if result else {"Optional", "Union"}
    ):
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return all(_annotation(item, result=result) for item in items)
    return False


def _torch_route(call: ast.AST, op: str) -> bool:
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Call):
        return False
    helper = call.func
    return (
        isinstance(helper.func, ast.Name)
        and helper.func.id == "_op"
        and len(helper.args) == 3
        and ast.unparse(helper.args[0]) in {"kernels", "self._kernels"}
        and isinstance(helper.args[1], ast.Constant)
        and helper.args[1].value == op
        and isinstance(helper.args[2], ast.Constant)
        and helper.args[2].value is False
        and not helper.keywords
    )


def _preserved_expression(node: ast.AST, function: ast.AST, guards: list[dict[str, Any]]) -> bool:
    for parent in body_nodes(function):
        if not isinstance(parent, ast.If):
            continue
        for arm in ("body", "orelse"):
            if not any(node in ast.walk(stmt) for stmt in getattr(parent, arm)):
                continue
            condition = ast.unparse(parent.test)
            if any(g["preserve_body"] and g["condition"] == condition and g["arm"] == arm for g in guards):
                return True
            # The original module call may survive on the non-plain-Linear arm.
            test = parent.test
            if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
                left = test.left
                is_type = (
                    isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "type"
                )
                plain_linear = ast.unparse(test.comparators[0]) in {"nn.Linear", "torch.nn.Linear"}
                if (
                    is_type
                    and plain_linear
                    and (
                        (isinstance(test.ops[0], ast.Is) and arm == "orelse")
                        or (isinstance(test.ops[0], ast.IsNot) and arm == "body")
                    )
                ):
                    return True
    return False


def _check_parameter_chain(tree: ast.AST, function: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    """Reject dropped selections in same-file callers, including newly added parameters."""
    pending = [function]
    seen = set()
    while pending:
        callee = pending.pop()
        if callee.name in seen:
            continue
        seen.add(callee.name)
        positional = [a.arg for a in (*callee.args.posonlyargs, *callee.args.args)]
        if "kernels" not in positional + [a.arg for a in callee.args.kwonlyargs]:
            continue
        for qualified, caller in functions(tree):
            for call in body_nodes(caller):
                if (
                    not isinstance(call, ast.Call)
                    or not isinstance(call.func, ast.Name)
                    or call.func.id != callee.name
                ):
                    continue
                value = next((kw.value for kw in call.keywords if kw.arg == "kernels"), None)
                if value is None and "kernels" in positional and len(call.args) > positional.index("kernels"):
                    value = call.args[positional.index("kernels")]
                if value is None or ast.unparse(value) not in {"kernels", "self._kernels"}:
                    raise ContractError(
                        f"KernelSelection not forwarded by {qualified}:{call.lineno} to {callee.name}"
                    )
                if ast.unparse(value) == "kernels":
                    parameters = {
                        a.arg for a in (*caller.args.posonlyargs, *caller.args.args, *caller.args.kwonlyargs)
                    }
                    if "kernels" not in parameters:
                        raise ContractError(f"KernelSelection is not a parameter of {qualified}")
                    pending.append(caller)


def check_structure(run: RegistrationRun) -> dict[str, Any]:
    """C2 static gate: reference signature, explicit registration and routed sites."""
    document = registration(run)
    reference_tree = ast.parse((run.worktree / REFERENCE).read_text(encoding="utf-8"))
    reference = function_node(reference_tree, run.spec.op)
    if reference not in reference_tree.body or isinstance(reference, ast.AsyncFunctionDef):
        raise ContractError("Reference must be a module-level synchronous function")
    args = (*reference.args.posonlyargs, *reference.args.args, *reference.args.kwonlyargs)
    if (
        not ast.get_docstring(reference)
        or not _annotation(reference.returns, result=True)
        or not all(_annotation(a.annotation) for a in args)
        or reference.args.vararg
        or reference.args.kwarg
    ):
        raise ContractError("Reference requires a contract docstring and tensor/constant annotations")
    for node in ast.walk(reference_tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            )
            if (isinstance(node, ast.ImportFrom) and node.level) or any(
                m.split(".")[0] not in sys.stdlib_module_names | {"torch"} for m in modules
            ):
                raise ContractError("Torch reference may import only torch and the standard library")
    for node in ast.walk(reference):
        if isinstance(node, (ast.AugAssign, ast.Delete)) or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr.endswith("_")
            and not node.func.attr.endswith("__")
        ):
            raise ContractError("Reference must not mutate inputs in place")
        if isinstance(node, ast.Assign) and any(
            isinstance(t, (ast.Subscript, ast.Attribute)) for t in node.targets
        ):
            raise ContractError("Reference must not assign into tensor inputs")
    registry = ast.parse((run.worktree / REGISTRY).read_text(encoding="utf-8"))
    expected = ast.parse(
        f"register_lazy_op({run.spec.op!r}, 'torch', 'embodiinfer.backend.torch.ops', {run.spec.op!r})"
    ).body[0]
    if sum(ast.dump(n) == ast.dump(expected) for n in registry.body) != 1:
        raise ContractError(
            "Operator requires its own top-level register_lazy_op line, outside the Triton loop"
        )
    reported = {item["id"]: item for item in document["sites"]}
    for site in read_json(run.root / "sites.json")["sites"]:
        item = reported[site["id"]]
        if item.get("file") != site["source"] or item.get("function") != site["function"]:
            raise ContractError(f"{site['id']}: reported site does not match reviewed source")
        tree = ast.parse(source_path(run.worktree, site["source"]).read_text(encoding="utf-8"))
        function = function_node(tree, site["function"])
        call = ast.parse(item.get("call", ""), mode="eval").body
        if not _torch_route(call, run.spec.op):
            raise ContractError(f"{site['id']}: use _op(kernels, op, False)(...) with Torch preference")
        nodes = body_nodes(function)
        if not any(isinstance(n, ast.Call) and ast.dump(n) == ast.dump(call) for n in nodes):
            raise ContractError(f"{site['id']}: reported registry call is absent")
        selection = ast.unparse(call.func.args[0])
        args = {a.arg for a in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)}
        if (selection == "kernels" and "kernels" not in args) or (
            selection == "self._kernels" and "self" not in args
        ):
            raise ContractError(f"{site['id']}: KernelSelection must be passed, not a global")
        if selection == "kernels":
            _check_parameter_chain(tree, function)
        for guard in site["guards"]:
            matches = [
                n for n in nodes if isinstance(n, ast.If) and ast.unparse(n.test) == guard["condition"]
            ]
            if not matches or (
                guard["preserve_body"]
                and not any(branch_dump(getattr(n, guard["arm"])) == guard["body"] for n in matches)
            ):
                raise ContractError(f"{site['id']}: preserve branch changed: {guard['condition']}")
        for node in nodes:
            if (
                isinstance(node, ast.expr)
                and ast.unparse(node) == site["expression"]
                and not _preserved_expression(node, function, site["guards"])
            ):
                raise ContractError(
                    f"{site['id']}: original expression remains outside a preserve branch at line {node.lineno}"
                )
    return {"reference": document["reference"], "sites": list(reported), "tests": document["tests"]}


async def registry_probe(run: RegistrationRun, output: Path, timeout: float) -> dict[str, Any]:
    """Check actual public registry resolution in the selected model interpreter."""
    code = import_guard(run.worktree) + (
        "from embodiinfer.layers import available_ops, get_op, KernelSelection\n"
        f"from embodiinfer.backend.torch.ops import {run.spec.op} as reference\n"
        f"assert 'torch' in available_ops()[{run.spec.op!r}]\n"
        f"assert get_op({run.spec.op!r}) is reference\n"
        "KernelSelection('none').prepare()\n"
        f"assert KernelSelection('torch').resolve({run.spec.op!r}) is reference\n"
    )
    result = await command(
        [run.manifest["test_python"], "-c", code], root=run.worktree, output=output, timeout=timeout
    )
    if result["returncode"]:
        raise ContractError(f"Public registry probe failed: {result['output']}")
    return result


async def pytest_nodes(
    run: RegistrationRun, root: Path, nodes: list[str], output: Path, timeout: float
) -> dict[str, Any]:
    """Run only CPU tests with an origin check and machine-readable outcome evidence."""
    report = output.with_suffix(".xml")
    report.unlink(missing_ok=True)
    args = [*nodes, "-q", "-m", "not gpu", "-p", "no:cacheprovider", f"--junitxml={report}"]
    code = import_guard(root) + f"import pytest\nraise SystemExit(pytest.main({args!r}))\n"
    result = await command(
        [run.manifest["test_python"], "-c", code], root=root, output=output, timeout=timeout
    )
    result["tests"] = nodes
    cases = []
    if report.is_file():
        for case in ET.parse(report).iter("testcase"):
            failure = case.find("failure")
            cases.append(
                {
                    "name": case.get("name"),
                    "class": case.get("classname"),
                    "failed": case.find("failure") is not None,
                    "error": case.find("error") is not None,
                    "skipped": case.find("skipped") is not None,
                    "assertion": failure is not None
                    and failure.get("message", "").startswith(("assert ", "AssertionError")),
                }
            )
    result["cases"] = cases
    return result


async def baseline_tests(run: RegistrationRun, output: Path, deadline: float) -> dict[str, Any]:
    """C5: every new test must fail on BASE; restore even on worker cancellation."""
    document = registration(run)
    files = {node.split("::")[0] for node in document["tests"]}
    run.verify()
    run.manifest["base_overlay"] = {name: file_hash(source_path(run.worktree, name)) for name in files}
    run.save()  # Record expected overlay bytes before touching the detached checkout.
    results = []
    try:
        for name in files:
            target = source_path(run.base, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path(run.worktree, name), target)
        for index, node in enumerate(document["tests"]):
            result = await pytest_nodes(
                run, run.base, [node], output / f"baseline-{index}.log", deadline - time.monotonic()
            )
            results.append(result)
            # Collection/import/environment errors are not evidence of an effective test.
            if (
                result["returncode"] != 1
                or not result["cases"]
                or not all(
                    c["failed"] and c["assertion"] and not c["skipped"] and not c["error"]
                    for c in result["cases"]
                )
            ):
                raise ContractError(
                    f"C5: {node} did not fail as an executed test on BASE (exit {result['returncode']}): {result['output']}"
                )
    finally:
        run.recover_base()
        atomic_json(output / "baseline-results.json", results)
    run.verify()
    return {"commands": results}


async def run_checks(
    run: RegistrationRun, archive: Path, *, seconds: float
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Execute C1–C5 in order; reuse passed gates only for unchanged source bytes."""
    deadline = time.monotonic() + seconds
    run.verify()
    identity = run.fingerprint()
    path = archive / "checks.json"
    previous = read_json(path) if path.exists() else {}
    results = previous.get("checks", []) if previous.get("fingerprint") == identity else []
    results = [item for item in results if item["passed"]]
    for check in ("C1", "C2", "C3", "C4", "C5"):
        if any(item["check"] == check for item in results):
            continue
        details: dict[str, Any] = {}
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("Registration duration exhausted")
            if check == "C1":
                details = check_scope(run)
            elif check == "C2":
                details = check_structure(run)
                details["command"] = await registry_probe(
                    run, archive / "registry.log", deadline - time.monotonic()
                )
            elif check == "C3":
                details["commands"] = []
                files = check_scope(run)["files"]
                for label, args in (("lint", ["check"]), ("format", ["format", "--check"])):
                    result = await command(
                        [sys.executable, "-m", "ruff", *args, *files],
                        root=run.worktree,
                        output=archive / f"ruff-{label}.log",
                        timeout=deadline - time.monotonic(),
                    )
                    details["commands"].append(result)
                    if result["returncode"]:
                        raise ContractError(f"Ruff {label} failed: {result['output']}")
            elif check == "C4":
                details["commands"] = []
                # Run declared regression tests together, new nodes separately so that
                # a deselected/skipped parametrization cannot hide among passing tests.
                selections = [list(run.spec.tests), *[[node] for node in registration(run)["tests"]]]
                for index, nodes in enumerate(selections):
                    result = await pytest_nodes(
                        run, run.worktree, nodes, archive / f"tests-{index}.log", deadline - time.monotonic()
                    )
                    details["commands"].append(result)
                    if (
                        result["returncode"]
                        or not result["cases"]
                        or (index and any(c["skipped"] for c in result["cases"]))
                    ):
                        raise ContractError(
                            f"CPU tests failed or new tests were skipped: {nodes}: {result['output']}"
                        )
            else:
                details = await baseline_tests(run, archive, deadline)
            run.verify()
            if run.fingerprint() != identity:
                raise ContractError("Source changed while checks were running")
        except (ContractError, SyntaxError, ValueError, KeyError, TypeError, OSError) as exc:
            results.append({"check": check, "passed": False, "details": details, "summary": str(exc)[:12000]})
            feedback = {"check": check, "summary": str(exc)[:12000], "details": details}
            atomic_json(path, {"fingerprint": identity, "checks": results})
            atomic_json(archive / "feedback.json", feedback)
            return results, feedback
        results.append({"check": check, "passed": True, "details": details})
        atomic_json(path, {"fingerprint": identity, "checks": results})
    return results, None
