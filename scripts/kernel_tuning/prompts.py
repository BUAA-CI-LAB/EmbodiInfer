"""Task-specific coding instructions; the evaluator owns acceptance decisions."""

from __future__ import annotations

import json
from dataclasses import asdict

from .contracts import TaskPackage


def _deadline(task: TaskPackage, deliverable: str) -> str:
    """Say how long a turn may run, so its deliverable exists before the controller stops it."""
    minutes = task.settings.search.agent_timeout_seconds / 60
    return (
        f"This turn is stopped after {minutes:g} minutes, and a turn stopped before {deliverable}\n"
        f"exists fails the attempt: write a complete {deliverable} within the first few minutes,\n"
        "then refine it only while time remains.\n"
    )


def plan_prompt(
    task: TaskPackage, *, role: str = "", skills: tuple[str, ...] = (), profiles: bool = False
) -> str:
    """Ask for a reviewable hypothesis and executable plan before source generation.

    ``role`` names the chaser taking this turn, ``skills`` the KDA skills its session
    carries, and ``profiles`` whether the controller collects Nsight Compute evidence.
    """
    turns = (
        f"You are the {role} of two coding agents taking turns on this operator, one candidate\n"
        "per turn. feedback.json records which chaser produced each earlier attempt. Build on the\n"
        "incumbent whoever wrote it: a structural rewrite and a careful refinement are both valid moves.\n"
        if role
        else ""
    )
    evidence = (
        "profiles/incumbent/ and profiles/previous/ (when present) hold Nsight Compute evidence the\n"
        "controller collected for the incumbent and the previous attempt: read details.txt first;\n"
        "ncu.ncu-rep is the full report. Diagnose the bottleneck from it before choosing a hypothesis.\n"
        if profiles
        else ""
    )
    if skills:
        evidence += (
            f"Your session carries the skills {', '.join(skills)}. Where a skill routes guidance by\n"
            "device, select the target matching the compute capability in environment.json and do\n"
            "not apply instructions or metric names written for another architecture. Apply their\n"
            "analysis method to the evidence provided here; do not collect profiles yourself.\n"
            "task/HARDWARE.md records what has been measured on this GPU.\n"
        )
    replay = (
        "Read task/replay.json for the frozen model, tensor layouts, aliases and mutations. Optimize this one observed call; no neighboring-operator fusion or quantization.\n"
        if "replay.json" in task.hashes
        else ""
    )
    if "numerics.json" in task.hashes:
        replay += "Read task/numerics.json. It freezes per-workload/profile/output bounds against FP64, not byte equality with the production baseline. All elements and all profiles must pass; tuning.yaml thresholds are reporting maxima only.\n"
    return f"""Optimize the operator in task/ for this target machine.
Read task/README.md, task/definition.json, task/workloads.jsonl, task/baseline.py,
task/tuning.yaml, environment.json, incumbent.json, and feedback.json before choosing one change.
Read task/HARDWARE.md when present for target-specific observations and prior experiments.
{turns}{evidence}\
Feedback includes complete source for recent evaluated candidates, including rejected ones.
Compare their implementations with their measured failures before selecting a materially
different configuration or algorithm; do not repeat a rejected implementation by guessing its source.
{replay}\
{_deadline(task, "PLAN.md")}\
Reason from the task files, the incumbent, feedback and any profiles; do not write
numerical simulators, test harnesses or scratch programs in this turn.
Write PLAN.md with: the current bottleneck; one falsifiable optimization hypothesis;
the source/launch configuration changes; correctness hazards; and validation steps.
Describe the validation criteria for the controller; do not compile, run GPU code,
benchmark, or inspect generated assembly during either agent turn. The agent's
command environment may restrict GPU access and compiler cache operations even
when environment.json confirms that the separate evaluator has a working GPU.
Finish this turn when PLAN.md contains that executable plan. The next turn implements it.
The mathematical reference and baseline have different roles: preserve the former's
outputs and beat the latter's latency over every declared workload.
Frozen conditions: {json.dumps(asdict(task.settings), ensure_ascii=False)}
Use only this attempt directory for edits. Task files, environment.json, incumbent.json, feedback.json,
the evaluator, run archives, and precision/timing settings are fixed inputs.
Create candidate source beside PLAN.md; do not add files inside task/.
Do not commit, install dependencies, alter global settings, or start another search.
"""


def implementation_prompt(task: TaskPackage) -> str:
    """Specify a complete official Solution rather than interpreting PASS in prose."""
    schema = {
        "name": "descriptive_candidate_name",
        "definition": task.definition["name"],
        "author": "coding-agent",
        "spec": {
            "language": task.settings.language,
            "target_hardware": list(task.settings.target_hardware),
            "entry_point": "kernel.py::run" if task.settings.language == "triton" else "kernel.cu::run",
            "destination_passing_style": False,
            **({"binding": "torch"} if task.settings.language == "cuda" else {}),
        },
        "sources": [
            {
                "path": "kernel.py" if task.settings.language == "triton" else "kernel.cu",
                "content": "complete source text",
            }
        ],
    }
    replay = (
        "This discovered task requires destination_passing_style=false. Preserve replay.json input/output layouts, aliases and mutations, returning mutated buffers in definition order. No new operator fusion or quantization.\n"
        if "replay.json" in task.hashes
        else ""
    )
    return f"""Implement PLAN.md and write solution.json using the official FlashInfer
Solution structure below.
{_deadline(task, "solution.json")}\
Include every source/helper in sources; relative POSIX
paths and multiple files are supported. Include actual source text, not placeholders.
You may write source files here and serialize them into solution.json.
{json.dumps(schema, indent=2)}
The entry point matches definition.inputs in order. If destination_passing_style
is true, append all definition.outputs as output buffers; otherwise return outputs
in definition order. For CUDA binding=torch, export the symbol with
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m). tvm-ffi binding is also supported when installed.
{replay}\
Preserve dtype and the declared numerical contract; bit_exact includes intermediate rounding, signed zero and NaN
bit patterns. Write the configured operator in {task.settings.language}; target
the declared hardware. Use only the permitted input metadata for specialization.
Return values must depend on the supplied tensors on every call and graph replay.
Launch on the caller's current stream; any helper-stream work must join it before
returning. The caller's device, stream, and numerical settings stay unchanged.
Keep evaluation logic, expected answers, precision, workloads, and metrics untouched.
The controller will compile, check all seeds, measure, and promote from structured
evidence after this turn returns. Do not compile, run GPU code, benchmark, inspect
assembly, or work around agent-environment permissions/cache failures. Static
source and JSON checks are sufficient for this handoff. A failed controller check
will be recorded as feedback for the next candidate.
Finish when solution.json contains the complete implementation and PLAN.md explains
the implemented change. The controller retains this attempt even when it fails.
"""
