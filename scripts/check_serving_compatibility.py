"""Read-only protocol/gitlink preflight before applying an inference patch downstream.

This reports literal wire contracts, without importing either checkout or
changing a submodule. Matching this check is necessary, not a full integration
certification: authentication, payloads, model dependencies and live requests
still need end-to-end tests.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

_SCHEMA = re.compile(r"^(?:vvla|embodiinfer)\.policy\.(.+)\.v\d+$")


def contracts(paths: list[Path]) -> dict[str, list[str | int]]:
    """Collect primary wire literals; request aliases do not prove response compatibility.

    Current servers also accept schema aliases, but emit their primary schema.
    This conservative check does not infer a client's response parser from its
    request alias declarations. It never treats transport tags as interchangeable.
    """
    result: dict[str, set[str | int]] = {}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        alias_nodes = {
            id(child)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id.endswith("_SCHEMA_ALIASES")
                for target in node.targets
            )
            for child in ast.walk(node.value)
        }
        for node in ast.walk(tree):
            if id(node) not in alias_nodes and isinstance(node, ast.Constant) and isinstance(node.value, str):
                match = _SCHEMA.fullmatch(node.value)
                if match:
                    result.setdefault("schema:" + match[1], set()).add(node.value)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id in {"REQUEST_TAG", "RESPONSE_TAG"}
                        and isinstance(node.value.value, int)
                    ):
                        result.setdefault(target.id, set()).add(node.value.value)
    return {name: sorted(values) for name, values in sorted(result.items())}


def compare(server: dict, client: dict) -> dict:
    """Fail closed for unknown client contracts and every conflicting declared field."""
    differences = {
        key: {"server": server.get(key), "client": values}
        for key, values in client.items()
        if key not in server or server[key] != values
    }
    required = {"schema:step", "schema:session"}
    missing = sorted(required - client.keys())
    if "schema:rpc" in client:
        missing += sorted({"REQUEST_TAG", "RESPONSE_TAG"} - client.keys())
    return {
        "compatible_literals": not missing and not differences,
        "missing": missing,
        "differences": differences,
    }


def main() -> int:
    """Compare selected client sources to the current checkout; never update a gitlink."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--client-source", type=Path, action="append", required=True)
    parser.add_argument("--embodirun", type=Path)
    args = parser.parse_args()
    serve = args.infer / "embodiinfer/engine/serve"
    server = contracts([serve / name for name in ("contracts.py", "service.py", "wireless_server.py")])
    client = contracts(args.client_source)
    report = {"server": server, "client": client, **compare(server, client)}
    if args.embodirun is not None:
        gitlink = subprocess.check_output(
            ["git", "-C", str(args.embodirun), "ls-tree", "HEAD", "third_party/embodiinfer"], text=True
        ).strip()
        report["embodirun_gitlink"] = gitlink
        report["infer_revision"] = subprocess.check_output(
            ["git", "-C", str(args.infer), "rev-parse", "HEAD"], text=True
        ).strip()
        report["gitlink_matches_infer"] = bool(gitlink) and gitlink.split()[2] == report["infer_revision"]
    report["passed"] = report["compatible_literals"] and report.get("gitlink_matches_infer", True)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
