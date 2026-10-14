"""Agent instructions for lossless registration, separate from controller evidence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .workspace import RegistrationRun

RULES = """You are registering ONE mathematical operator, not tuning or fusing kernels.
1. Put the reference in embodiinfer/backend/torch/ops.py with the operator's exact
name, useful type annotations and a contract docstring. Import only torch and the
standard library. Inputs are tensors or int/float/bool/str/None constants; outputs
are tensors or tuples of tensors. Never mutate inputs.
2. Preserve the original arithmetic verbatim: identical ATen calls, order, shapes,
intermediate dtype casts and contiguous calls. Never replace F.linear by x @ w.T,
add/remove casts, or combine/split GEMMs. Do not loosen numerical assertions.
3. Add one top-level register_lazy_op("<op>", "torch",
"embodiinfer.backend.torch.ops", "<op>") line in layers/ops.py. Never add a Torch-only
operator to the loop that registers both Torch and Triton.
4. Use the policy's existing _op(kernels, "<op>", False)(...) helper, with Torch
preference. Pass KernelSelection explicitly through all callers; do not use globals.
5. Keep all preserved guards and branches unchanged. For ordinary nn.Linear
dispatch, use an exact type check (type(module) is nn.Linear); keep QuantizedLinear,
RowParallelLinear, ColumnParallelLinear and other subclasses on their module path.
Do not branch on model names. Preserve existing gradient, attention and parallel paths.
6. Add tests for all three behaviors: CPU random FP32 AND BF16 reference parity
using torch.equal against the original expression; registry dispatch using a fake
tuned KernelProfile that records calls; preserved branches bypassing that profile.
Keep RNG, matmul precision, shapes and steps identical. Do not run GPU code.
7. Allowed edits: the Torch reference module, layers/ops.py, the reviewed site source
files, additions to tests/test_kernel_ops.py or tests/test_registry_<op>.py, plus
PLAN.md and REGISTRATION.json. Never delete/modify existing test lines or statements,
KernelSelection, backend/triton/tuned, scripts, pyproject.toml or uv.lock.
8. You may run Ruff and CPU pytest with the recorded test_python, PYTHONPATH set to
this worktree, and CUDA_VISIBLE_DEVICES empty. Never install dependencies, run git
commit/push, merge, edit Git metadata, or modify the main checkout, base checkout,
run specification, context, controller artifacts or budgets.
The context is read-only task data. Follow repository AGENTS.md within these bounds.
Tests listed in REGISTRATION.json must be NEW functions (not files, old tests or
selected parametrizations). Each must run and fail by assertion on BASE; put imports
of newly added symbols inside tests so a missing symbol can be asserted explicitly.
Collection errors, skips and environment failures are not baseline evidence.
"""


def prompt(run: RegistrationRun, stage: str, feedback: dict | None = None) -> str:
    """Build a self-contained turn, including context for a newly opened session."""
    header = f"""Operator registration: {run.spec.op}
Work only in: {run.worktree}
Read-only context: {run.root / "context"}
CPU test interpreter: {run.manifest["test_python"]}
Frozen specification:
{json.dumps(run.spec.as_dict(), indent=2, ensure_ascii=False)}

{RULES}
Read context/sites.json, AGENTS.md, the source snapshots and test_kernel_ops.py.
The pi0.5 policy snapshot shows call sites already routed through _op and
KernelSelection. Existing PLAN.md and REGISTRATION.json describe previous work.
"""
    if stage == "plan":
        return (
            header
            + """
Write PLAN.md before changing source. Explain the mathematical definition and
its term-by-term correspondence to every selected expression; the tensor/constant
signature; how KernelSelection reaches each caller; preservation of each branch;
and the exact tests you will add. Do not implement yet.
"""
        )
    example = {
        "op": run.spec.op,
        "reference": f"embodiinfer/backend/torch/ops.py::{run.spec.op}",
        "sites": [
            {
                "id": "S1",
                "file": run.spec.sites[0].source,
                "function": run.spec.sites[0].function,
                "call": f'_op(kernels, "{run.spec.op}", False)(x, weight)',
            }
        ],
        "preserved": [{"site": "S1", "branch": "exact original guard"}],
        "tests": [f"tests/test_registry_{run.spec.op}.py::test_reference"],
    }
    instruction = (
        "Implement PLAN.md. "
        if stage == "implement"
        else "Repair the current worktree using the controller feedback below; update PLAN.md as needed. "
    )
    return (
        header
        + instruction
        + "Write REGISTRATION.json covering EVERY selected site and ALL new tests. Schema example:\n"
        + json.dumps(example, indent=2)
        + "\nController feedback:\n"
        + json.dumps(feedback, indent=2, ensure_ascii=False)
    )
