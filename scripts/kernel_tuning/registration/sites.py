"""AST call-site discovery and reviewable preservation constraints."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from scripts.kernel_tuning.contracts import ContractError

from .spec import RegistrationSpec


def source_path(root: Path, name: str) -> Path:
    """Resolve a regular file inside a checkout, refusing symlink components."""
    candidate = root / name
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ContractError(f"File escapes checkout: {name}")
    if any(p.is_symlink() for p in (candidate, *candidate.parents) if p != root.parent):
        raise ContractError(f"Symlink is not a registration input: {name}")
    return candidate


def functions(tree: ast.AST) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Enumerate functions with lexical qualification, including class methods."""
    result = []

    def visit(node: ast.AST, parents: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names = (*parents, child.name)
                if not isinstance(child, ast.ClassDef):
                    result.append((".".join(names), child))
                visit(child, names)
            else:
                visit(child, parents)

    visit(tree, ())
    return result


def function_node(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Require exactly one matching function; qualified names disambiguate methods."""
    matches = [
        node
        for qualified, node in functions(tree)
        if qualified == name or ("." not in name and node.name == name)
    ]
    if len(matches) != 1:
        raise ContractError(f"Function {name!r} matches {len(matches)} definitions; qualify it explicitly")
    return matches[0]


def body_nodes(function: ast.AST) -> list[ast.AST]:
    """Walk one function without treating nested definitions as its call sites."""
    result = []

    def visit(node: ast.AST) -> None:
        result.append(node)
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                visit(child)

    visit(function)
    return result


def branch_dump(statements: list[ast.stmt]) -> str:
    """Normalize a preserved branch while ignoring source positions/formatting."""
    return ast.dump(ast.Module(body=statements, type_ignores=[]), include_attributes=False)


def locate(root: Path, spec: RegistrationSpec) -> dict[str, Any]:
    """Locate each exact call, surrounding guards and same-file parameter callers."""
    records = []
    for index, site in enumerate(spec.sites, 1):
        source = source_path(root, site.source).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = function_node(tree, site.function)
        nodes = body_nodes(function)
        matches = [n for n in nodes if isinstance(n, ast.expr) and ast.unparse(n) == site.expression]
        if len(matches) != 1:
            raise ContractError(f"{site.source}::{site.function}: expression matches {len(matches)} sites")
        selected = matches[0]
        guards = []
        for node in nodes:
            if not isinstance(node, ast.If):
                continue
            for arm in ("body", "orelse"):
                statements = getattr(node, arm)
                contains = any(selected in ast.walk(stmt) for stmt in statements)
                sibling_contains = any(
                    selected in ast.walk(stmt)
                    for stmt in getattr(node, "orelse" if arm == "body" else "body")
                )
                early_return = node.lineno < selected.lineno and any(
                    isinstance(n, ast.Return) for stmt in statements for n in ast.walk(stmt)
                )
                if contains or (statements and (sibling_contains or early_return)):
                    guards.append(
                        {
                            "condition": ast.unparse(node.test),
                            "arm": arm,
                            "line": node.lineno,
                            "preserve_body": not contains,
                            "body": branch_dump(statements),
                        }
                    )
        args = [a.arg for a in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)]
        if "kernels" in args:
            access = "kernels"
        elif "self" in args and any(
            isinstance(n, ast.Attribute) and ast.unparse(n) == "self._kernels" for n in ast.walk(tree)
        ):
            access = "self._kernels"
        else:
            access = "add kernels parameter and propagate through the callers below"
        if not any(isinstance(n, ast.FunctionDef) and n.name == "_op" for n in tree.body):
            raise ContractError(f"{site.source} needs an existing _op/KernelSelection integration")
        callers = []
        frontier, seen = {function.name}, set()
        while frontier:
            target = frontier.pop()
            if target in seen:
                continue
            seen.add(target)
            for qualified, caller in functions(tree):
                for call in body_nodes(caller):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == target
                    ):
                        callers.append(
                            {
                                "function": qualified,
                                "callee": target,
                                "line": call.lineno,
                                "call": ast.unparse(call),
                            }
                        )
                        frontier.add(caller.name)
        records.append(
            {
                "id": f"S{index}",
                "source": site.source,
                "function": site.function,
                "expression": site.expression,
                "line": selected.lineno,
                "column": selected.col_offset,
                "end_line": selected.end_lineno,
                "kernels": access,
                "guards": guards,
                "callers": callers,
                "snippet": "\n".join(
                    source.splitlines()[max(0, selected.lineno - 4) : selected.end_lineno + 3]
                ),
            }
        )
    return {"schema_version": 1, "preserve": list(spec.preserve), "sites": records}


def validate_review(review: dict[str, Any], baseline: dict[str, Any]) -> None:
    """Permit review notes/extra guards, but never silently drop selected sites."""
    if not isinstance(review, dict) or not isinstance(review.get("sites"), list):
        raise ContractError("sites.json must contain the reviewed sites")
    if len(review["sites"]) != len(baseline["sites"]):
        raise ContractError("Changing the selected sites requires a new specification")
    for new, old in zip(review["sites"], baseline["sites"], strict=True):
        if any(new.get(k) != old[k] for k in ("id", "source", "function", "expression", "guards")):
            raise ContractError("Cannot remove source identities or baseline guards from sites.json")
    if not isinstance(review.get("preserve"), list) or any(
        not isinstance(x, str) for x in review["preserve"]
    ):
        raise ContractError("sites.json preserve must be a list of review notes")
    if not set(baseline["preserve"]) <= set(review["preserve"]):
        raise ContractError("Cannot remove specification preservation requirements")


def render(review: dict[str, Any]) -> str:
    """Render the source and branch evidence shown before any agent starts."""
    lines = [
        "# Operator registration call sites",
        "",
        "Review sites.json, then run `register run RUN --agent HARNESS/MODEL:EFFORT`.",
        "",
    ]
    for site in review["sites"]:
        lines.extend(
            [
                f"## {site['id']}: {site['source']}:{site['line']} ({site['function']})",
                "",
                "```python",
                site["snippet"],
                "```",
                "",
                f"KernelSelection: {site['kernels']}",
                "",
            ]
        )
        lines.extend(
            f"- Guard ({g['arm']}): `{g['condition']}`; preserve body: {g['preserve_body']}"
            for g in site["guards"]
        )
        lines.extend(f"- Caller: {c['function']}:{c['line']} — `{c['call']}`" for c in site["callers"])
        lines.append("")
    lines.extend(
        ["## Additional preservation requirements", "", *[f"- {item}" for item in review["preserve"]], ""]
    )
    return "\n".join(lines)
