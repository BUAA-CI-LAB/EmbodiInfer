"""Task-specific coding instructions; the evaluator owns acceptance decisions."""

from __future__ import annotations

import json
from dataclasses import asdict

from .contracts import TaskPackage


def plan_prompt(task: TaskPackage) -> str:
    """Ask for a reviewable hypothesis and executable plan before source generation."""
    return f"""Optimize the operator in task/ for this target machine.
Read task/README.md, task/definition.json, task/workloads.jsonl, task/baseline.py,
task/tuning.yaml, environment.json, incumbent.json, and feedback.json before choosing one change.
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
    return f"""Implement PLAN.md and write solution.json using the official FlashInfer
Solution structure below. Include every source/helper in sources; relative POSIX
paths and multiple files are supported. Include actual source text, not placeholders.
You may write source files here and serialize them into solution.json.
{json.dumps(schema, indent=2)}
The entry point matches definition.inputs in order. If destination_passing_style
is true, append all definition.outputs as output buffers; otherwise return outputs
in definition order. For CUDA binding=torch, export the symbol with
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m). tvm-ffi binding is also supported when installed.
Preserve dtype and intermediate rounding; bit_exact includes signed zero and NaN
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
