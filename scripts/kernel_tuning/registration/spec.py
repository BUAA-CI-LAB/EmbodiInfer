"""Frozen input contracts for one operator-registration run."""

from __future__ import annotations

import ast
import keyword
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.kernel_tuning.contracts import ContractError


def relative_file(value: Any, *, prefix: str, suffix: str = ".py") -> str:
    """Accept only portable repository-relative files in the owning directory."""
    if not isinstance(value, str) or "\\" in value or ":" in value:
        raise ContractError(f"Expected a relative {prefix} file: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ContractError(f"Unsafe repository path: {value!r}")
    if not value.startswith(prefix + "/") or path.suffix != suffix:
        raise ContractError(f"Expected a {suffix} file under {prefix}: {value!r}")
    return value


def test_node(value: Any) -> str:
    """Validate a pytest node ID without accepting options or external paths."""
    if not isinstance(value, str):
        raise ContractError("Test selectors must be strings")
    parts = value.split("::")
    relative_file(parts[0], prefix="tests")
    if any(not part or "\n" in part for part in parts):
        raise ContractError(f"Invalid pytest node: {value!r}")
    return value


@dataclass(frozen=True)
class Site:
    """An unambiguous expression within a named (optionally qualified) function."""

    source: str
    function: str
    expression: str


@dataclass(frozen=True)
class RegistrationBudget:
    """Hard attempt and wall-clock limits that survive interruption."""

    max_attempts: int = 5
    agent_timeout_seconds: int = 1800
    max_seconds: int = 7200


@dataclass(frozen=True)
class RegistrationSpec:
    """Immutable mathematical objective, source selection, tests and budget."""

    model: str
    op: str
    sites: tuple[Site, ...]
    description: str = ""
    preserve: tuple[str, ...] = ()
    tests: tuple[str, ...] = ("tests/test_kernel_ops.py",)
    budget: RegistrationBudget = field(default_factory=RegistrationBudget)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data for the frozen specification."""
        return asdict(self)

    @classmethod
    def parse(cls, value: Any) -> RegistrationSpec:
        """Reject malformed/unknown settings before creating any worktrees."""
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ContractError("Unknown or malformed registration specification")
        model, op = value.get("model"), value.get("op")
        if not isinstance(model, str) or not model.isidentifier():
            raise ContractError("model must be a canonical policy identifier")
        if not isinstance(op, str) or not op.isidentifier() or keyword.iskeyword(op) or op.startswith("_"):
            raise ContractError("op must be a public Python identifier, not a keyword")
        if model.lower().replace("_", "") in op.lower().replace("_", ""):
            raise ContractError("Name the operator by its mathematics, not its model")
        description = value.get("description", "")
        preserve, tests = value.get("preserve", []), value.get("tests", ["tests/test_kernel_ops.py"])
        if not isinstance(description, str) or not isinstance(preserve, (list, tuple)):
            raise ContractError("description must be text; preserve must be a list of text")
        if any(not isinstance(item, str) or not item.strip() for item in preserve):
            raise ContractError("preserve entries must be nonempty text")
        if not isinstance(tests, (list, tuple)) or not tests:
            raise ContractError("Specify at least one existing CPU test")
        tests = tuple(test_node(item) for item in tests)
        raw_sites = value.get("sites")
        if not isinstance(raw_sites, (list, tuple)) or not raw_sites:
            raise ContractError("Specify at least one call site")
        sites = []
        for raw in raw_sites:
            if not isinstance(raw, dict) or set(raw) != {"source", "function", "expression"}:
                raise ContractError("Each site requires source, function and expression")
            source = relative_file(raw["source"], prefix=f"embodiinfer/policies/{model}")
            function, expression = raw["function"], raw["expression"]
            if not isinstance(function, str) or not all(p.isidentifier() for p in function.split(".")):
                raise ContractError("site.function must be a function name or qualified name")
            try:
                parsed = ast.parse(expression, mode="eval")
            except (SyntaxError, TypeError, ValueError) as exc:
                raise ContractError("site.expression must be a Python expression") from exc
            if not isinstance(parsed.body, ast.Call):
                raise ContractError("A registration site must select a call expression")
            sites.append(Site(source, function, ast.unparse(parsed.body)))
        if len(set(sites)) != len(sites):
            raise ContractError("Duplicate call sites")
        raw_budget = value.get("budget", {})
        if not isinstance(raw_budget, dict) or set(raw_budget) - set(RegistrationBudget.__dataclass_fields__):
            raise ContractError("Unknown or malformed budget")
        if any(type(v) is not int or v <= 0 for v in raw_budget.values()):
            raise ContractError("Budget values must be positive integers")
        return cls(
            model, op, tuple(sites), description, tuple(preserve), tests, RegistrationBudget(**raw_budget)
        )


def load_spec(path: Path | None, *, model: str | None, op: str | None, sites: list[str]) -> RegistrationSpec:
    """Load YAML/JSON or the equivalent repeated --site command-line form."""
    if path is not None:
        if model or op or sites:
            raise ContractError("Use either --spec or --model/--op/--site")
        import yaml

        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ContractError(f"Invalid specification: {exc}") from exc
    else:
        parsed = []
        for item in sites:
            parts = item.split("::", 2)
            if len(parts) != 3:
                raise ContractError("--site expects SOURCE::FUNCTION::EXPRESSION")
            parsed.append(dict(zip(("source", "function", "expression"), parts, strict=True)))
        value = {"model": model, "op": op, "sites": parsed}
    return RegistrationSpec.parse(value)
