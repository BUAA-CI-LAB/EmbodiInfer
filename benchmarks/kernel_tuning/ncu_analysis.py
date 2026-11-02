"""CPU analysis using the NCU skill submodule; a native report is imported only once."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def analyze(
    output: Path,
    *,
    report: Path | None = None,
    capture_path: Path | None = None,
    profile: str = "auto",
) -> dict[str, Any]:
    """Archive every action and return a bounded, unit/availability-aware agent summary.

    ``capture_path`` allows CPU-only re-analysis without an installed native reader.
    Existing output directories are refused so earlier evidence cannot be overwritten.
    """
    if (report is None) == (capture_path is None):
        raise ValueError("Supply exactly one of report or capture_path")
    sys.path.insert(0, str(ROOT / "skills/ncu-report-skill/scripts"))
    from analyze_reports import collect_action
    from extract_stall_hotspots import rollup, write_report
    from ncu_utils import (
        PROFILE_METRICS,
        action_label,
        action_name,
        instance_values,
        load_capture,
        metric_display,
        metric_unit,
        metric_unit_display,
        read_identity,
        select_actions,
        select_profile,
    )
    from plot_timeline import ascii_plot

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    detached = output / "native-capture.json"
    if report is not None:
        capture = load_capture(report.resolve(), output_path=detached, python=sys.executable)
        if capture.get("report", {}).get("sha256") != hashlib.sha256(report.read_bytes()).hexdigest():
            raise ValueError("Native capture does not match the report content")
    else:
        assert capture_path is not None
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        detached.write_text(json.dumps(capture, allow_nan=False), encoding="utf-8")
    if capture.get("schema_version") != 3 or capture.get("status") != "captured_native_report":
        raise ValueError("Expected a complete detached native capture (schema 3)")
    actions = select_actions(capture)
    if not actions or capture.get("structural_errors"):
        raise ValueError("No actions or incomplete action enumeration in NCU capture")
    summaries = []
    conflicts = []
    for action in actions:
        identity = read_identity(action)
        chosen, status, reason = select_profile(profile, identity)
        selection = {
            "requested_profile": profile,
            "chosen_profile": chosen,
            "status": status,
            "reason": reason,
            "identity": identity,
        }
        if status == "conflict":
            conflicts.append(action_label(action))
        slug = collect_action(action, "kernel", output, PROFILE_METRICS[chosen], selection)
        per_line, unmapped, pca = rollup(action)
        stall = output / f"stall_hotspots_{slug}.txt"
        write_report(per_line, unmapped, pca, stall, action_name(action))
        timeline = output / f"pm_timeline_{slug}.txt"
        lines = ["Native PM samples; indices are not timestamps; replay passes are not aligned."]
        for name in sorted(action.get("pm_series") or {}):
            values = instance_values(action, name)
            if values is not None:
                lines.extend(ascii_plot(values, f"{name} {metric_unit_display(action, name)}"))
        if len(lines) == 1:
            lines.append("PM sampling unavailable: no series with instance values.")
        timeline.write_text("\n".join(lines), encoding="utf-8")
        summary = {
            "action": action_label(action),
            "kernel": action_name(action),
            "profile": selection,
            "metrics": {
                name: {"value": metric_display(action, name), "unit": metric_unit(action, name)}
                for name in PROFILE_METRICS[chosen]
            },
            "source_coverage": pca.get("sample_coverage"),
            "metrics_archive": str(output / f"metrics_all_{slug}.json"),
            "stall_hotspots": str(stall),
            "pm_timeline": str(timeline),
        }
        (output / f"summary_{slug}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if len(summaries) < 16:
            summaries.append(summary)
    result = {
        "native_capture": str(detached),
        "analysis_directory": str(output),
        "action_count": len(actions),
        "actions": summaries,
        "summary_truncated": len(actions) > len(summaries),
        "profile_conflicts": conflicts,
        "interpretation_limits": capture.get("interpretation_limits", []),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if conflicts:
        raise ValueError(f"Requested profile conflicts with report device: {conflicts}")
    return result


def main() -> None:
    """Analyze a report with this interpreter's reader, or a detached CPU capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--report", type=Path)
    group.add_argument("--capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("auto", "generic", "orin", "thor", "b200"), default="auto")
    args = parser.parse_args()
    analyze(args.output, report=args.report, capture_path=args.capture, profile=args.profile)


if __name__ == "__main__":
    main()
