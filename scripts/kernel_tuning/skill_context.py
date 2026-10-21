"""Expose repository skill assets to any supported coding-agent harness."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from .contracts import ContractError

ROOT = Path(__file__).resolve().parents[2]


def require_skills() -> None:
    """Reject an uninitialized skill checkout before spending evaluator or agent budget."""
    missing = [
        name
        for name in ("KernelWiki", "ncu-report-skill")
        if not (ROOT / "skills" / name / "SKILL.md").is_file()
    ]
    if missing:
        raise ContractError(
            f"Missing skill submodules: {', '.join(missing)}. From {ROOT}, run: "
            "git submodule update --init --recursive skills/KernelWiki skills/ncu-report-skill"
        )


def skill_instructions() -> str:
    """Give agents explicit, read-only entrypoints without global skill installation."""
    require_skills()
    wiki = ROOT / "skills/KernelWiki"
    ncu = ROOT / "skills/ncu-report-skill"

    def command(path: Path) -> str:
        return shlex.join([sys.executable, "-B", "-X", "utf8", str(path)])

    return f"""
Project skills (shared read-only assets, already available; no installation):
- Read {wiki / "SKILL.md"} for KernelWiki. Query from any working directory:
  {command(wiki / "scripts/query.py")} --architecture <target-sm> --compact
  {command(wiki / "scripts/get_page.py")} <page-id> --include-code
  {command(wiki / "scripts/grep_wiki.py")} <pattern> --only all
  Use environment.json for the actual GPU. Cite the relevant page and its
  confidence in PLAN.md; check architecture restrictions before reusing code.
- Read {ncu / "SKILL.md"} for NCU report interpretation. Read diagnostics.json
  for baseline and prior candidate evidence, including unavailable/error states.
  Evidence points to reports, native-capture.json, typed metrics, source stall
  coverage and PM series. Inspect every relevant action, not just action zero.
  CPU re-analysis of an archived capture (write only inside this attempt):
  {command(ROOT / "benchmarks/kernel_tuning/ncu_analysis.py")} --capture <native-capture.json> --output <new-local-directory> --profile auto
  Missing metrics, failed reads and unavailable source mappings are not zero.
  Retain native units; PM sample indices are not elapsed timestamps.
Only CPU queries and analysis of existing evidence are allowed during agent
turns. The controller alone runs NCU, compiles and executes kernels. These task
constraints take precedence over skill examples that launch or profile kernels.
You may read archived SASS/source evidence, but do not generate new assembly.
Never modify the shared skills, task, diagnostic archives or benchmark settings.
State the evidence and its limitations in PLAN.md; if diagnostics are unavailable,
make a clearly identified hypothesis from source and measured controller feedback.
"""
