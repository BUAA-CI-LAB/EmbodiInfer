"""Catalog of EmbodiInfer's core Triton operators, rendered into tuning task packages.

Each entry states only what a person must decide: the mathematical reference, the
production kernel call used as the baseline, model-derived workload shapes, and
the numerical contract. ``generate.py`` turns an entry into a complete KDA task.

The ``workloads`` here are estimates from model configurations. Shapes captured
from real runs (``benchmarks/kernel_tuning/captures/*.jsonl``, see
``capture.py``) replace them by default; the estimates remain the fallback for
operators no capture covers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Tolerances count BF16 roundings that the production kernel and the eager
# reference do not share. Each contributes at most the unit roundoff 2**-8
# relative error; the two final output roundings alone account for two steps.
# FP32 reordering is negligible except where outputs cancel toward zero, which
# atol covers. ``test_production_baseline_meets_generated_contract`` reports how
# much of each bound the production kernel uses on the target GPU.
BF16_ATOL = 2**-10


@dataclass(frozen=True)
class Tensor:
    """A FlashInfer TensorSpec: ``shape=None`` declares a scalar argument."""

    shape: tuple[str, ...] | None
    dtype: str


@dataclass(frozen=True)
class Workload:
    """One declared case; tensors not listed in ``scalars``/``fixed`` are random."""

    uuid: str
    axes: dict[str, int]
    scalars: dict[str, int | float] = field(default_factory=dict)
    # Valid structured integer inputs (e.g. segment offsets) cannot be random.
    # They are written into the task as safetensors data.
    fixed: dict[str, tuple[int, ...]] = field(default_factory=dict)
    weight: float = 1.0
    note: str = ""


@dataclass(frozen=True)
class Operator:
    """A production operator plus the frozen contract used to tune it."""

    name: str
    summary: str
    models: tuple[str, ...]
    op_type: str
    source: str  # Repository module holding the production kernel.
    entry: str  # Function imported from ``source`` and called by the baseline.
    call: str  # Baseline return expression over the definition inputs.
    axes: dict[str, int | None]  # None declares a variable axis.
    inputs: dict[str, Tensor]
    outputs: dict[str, Tensor]
    reference: str
    workloads: tuple[Workload, ...]
    notes: str
    constraints: tuple[str, ...] = ()
    precision: dict[str, Any] = field(default_factory=lambda: {"mode": "bit_exact"})
    timing: str = "cuda_graph"


def _tolerance(reason: str, steps: int = 2) -> dict[str, Any]:
    """Tolerance for ``steps`` unshared BF16 roundings (two: only the final casts differ)."""
    return {"mode": "tolerance", "atol": BF16_ATOL, "rtol": steps * 2**-8, "reason": reason}


BF16 = "bfloat16"

# pi0.5 action expert (Gemma 300M): width 1024, MLP 4096, 8 query heads, 1 KV
# head, head_dim 256, 50 action tokens. Prefix = 3 x 256 image + 200 text tokens.
# B=1 is serving; B=8 is a grouped RL rollout.
PI05_BATCHES = (1, 8)
PI05_TOKENS = 50
PI05_PREFIX = 968

# Qwen R2R low-level / panoramic policies (Qwen2.5-VL-3B ViT through
# EmbodiInfer's Triton vision path): 16 heads, head_dim 80, 64-patch windows.
# ActiveVLN runs the Transformers vision tower and uses none of these kernels.
# StreamVLN language model (Qwen2-7B): hidden 3584, MLP 18944, decode rows.

OPERATORS: tuple[Operator, ...] = (
    Operator(
        name="ada_rms_norm",
        summary="Adaptive RMSNorm with per-batch scale/shift and a materialized residual gate.",
        models=("pi05",),
        op_type="norm",
        source="embodiinfer/backend/triton/norm.py",
        entry="ada_rms_norm",
        call="ada_rms_norm(x, modulation, eps)",
        axes={"B": None, "T": None, "H": None, "H3": None, "ONE": 1},
        inputs={
            "x": Tensor(("B", "T", "H"), BF16),
            "modulation": Tensor(("B", "H3"), BF16),
            "eps": Tensor(None, "float32"),
        },
        outputs={"output": Tensor(("B", "T", "H"), BF16), "gate": Tensor(("B", "ONE", "H"), BF16)},
        constraints=("H3 == 3 * H",),
        reference="""import torch


def run(x, modulation, eps):
    values = x.float()
    variance = torch.mean(torch.square(values), dim=-1, keepdim=True)
    scale, shift, gate = modulation.unsqueeze(1).chunk(3, dim=-1)
    output = values * torch.rsqrt(variance + eps)
    output = output * (1.0 + scale.float()) + shift.float()
    return output.to(x.dtype), gate.to(x.dtype)
""",
        workloads=tuple(
            Workload(
                f"pi05-expert-b{b}",
                {"B": b, "T": PI05_TOKENS, "H": 1024, "H3": 3072},
                scalars={"eps": 1e-6},
                note="every action-expert layer, twice per layer",
            )
            for b in PI05_BATCHES
        ),
        precision=_tolerance(
            "The production kernel reduces the FP32 variance in a different order than torch.mean."
        ),
        notes="`modulation` is the AdaRMS projection in scale/shift/gate order, in the checkpoint "
        "dtype (BF16 in pi05_libero_finetuned_v044, as captured). The gate output is cast to the "
        "hidden dtype, one row per batch item.",
    ),
    Operator(
        name="gated_residual",
        summary="Residual plus gated update with BF16 rounding of the product before addition.",
        models=("pi05",),
        op_type="elementwise",
        source="embodiinfer/backend/triton/norm.py",
        entry="gated_residual",
        call="gated_residual(residual, update, gate)",
        axes={"B": None, "T": None, "H": None, "G": None},
        inputs={
            "residual": Tensor(("B", "T", "H"), BF16),
            "update": Tensor(("B", "T", "H"), BF16),
            "gate": Tensor(("B", "G", "H"), BF16),
        },
        outputs={"output": Tensor(("B", "T", "H"), BF16)},
        constraints=("G == 1 or G == T",),
        reference="""import torch


def run(residual, update, gate):
    return residual + update * gate
""",
        workloads=tuple(
            Workload(
                f"pi05-expert-b{b}",
                {"B": b, "T": PI05_TOKENS, "H": 1024, "G": 1},
                note="after attention and MLP in every action-expert layer",
            )
            for b in PI05_BATCHES
        ),
        notes="The gate broadcasts over tokens. Contracting multiply-add into an FMA "
        "violates the bit-exact contract; the production kernel disables FP fusion.",
    ),
    Operator(
        name="gated_gelu",
        summary="Tanh-approximate GELU of the gate projection times the up projection.",
        models=("pi05",),
        op_type="activation",
        source="embodiinfer/backend/triton/activation.py",
        entry="gated_gelu",
        call="gated_gelu(gate, up)",
        axes={"B": None, "T": None, "I": None},
        inputs={"gate": Tensor(("B", "T", "I"), BF16), "up": Tensor(("B", "T", "I"), BF16)},
        outputs={"output": Tensor(("B", "T", "I"), BF16)},
        reference="""import torch.nn.functional as F


def run(gate, up):
    return F.gelu(gate, approximate="tanh") * up
""",
        workloads=tuple(
            Workload(f"pi05-expert-b{b}", {"B": b, "T": PI05_TOKENS, "I": 4096}, note="every expert MLP")
            for b in PI05_BATCHES
        ),
        precision=_tolerance(
            "The production kernel evaluates tanh-GELU through its sigmoid identity in FP32, "
            "which differs from Torch's tanh formulation; both round GELU(gate) to BF16.",
            steps=4,
        ),
        notes="Rounding GELU(gate) to BF16 before multiplying by `up` matches the eager path.",
    ),
    Operator(
        name="rotate_qk",
        summary="Rotate-half RoPE on BHSD query and key with BF16 rounding of each product.",
        models=("pi05",),
        op_type="rope",
        source="embodiinfer/backend/triton/rotary.py",
        entry="rotate_qk",
        call="rotate_qk(query, key, cos, sin)",
        axes={"B": None, "HQ": None, "HK": None, "S": None, "D": None},
        inputs={
            "query": Tensor(("B", "HQ", "S", "D"), BF16),
            "key": Tensor(("B", "HK", "S", "D"), BF16),
            "cos": Tensor(("B", "S", "D"), BF16),
            "sin": Tensor(("B", "S", "D"), BF16),
        },
        outputs={
            "query_out": Tensor(("B", "HQ", "S", "D"), BF16),
            "key_out": Tensor(("B", "HK", "S", "D"), BF16),
        },
        constraints=("D == 64 or D == 128 or D == 256",),
        reference="""import torch


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def run(query, key, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return query * cos + _rotate_half(query) * sin, key * cos + _rotate_half(key) * sin
""",
        workloads=tuple(
            Workload(
                f"pi05-expert-b{b}",
                {"B": b, "HQ": 8, "HK": 1, "S": PI05_TOKENS, "D": 256},
                note="every action-expert attention layer",
            )
            for b in PI05_BATCHES
        ),
        notes="In the model, query/key are transposed views; this task uses contiguous BHSD tensors.",
    ),
    Operator(
        name="split_kv_attention",
        summary="Non-causal GQA attention over a cached prefix K/V block and the current K/V block.",
        models=("pi05",),
        op_type="attention",
        source="embodiinfer/backend/triton/split_kv_attention.py",
        entry="split_kv_attention",
        call="split_kv_attention(query, prefix_key, prefix_value, suffix_key, suffix_value, None, scaling)",
        axes={"B": None, "HQ": None, "HK": None, "P": None, "S": None, "D": None},
        inputs={
            "query": Tensor(("B", "HQ", "S", "D"), BF16),
            "prefix_key": Tensor(("B", "HK", "P", "D"), BF16),
            "prefix_value": Tensor(("B", "HK", "P", "D"), BF16),
            "suffix_key": Tensor(("B", "HK", "S", "D"), BF16),
            "suffix_value": Tensor(("B", "HK", "S", "D"), BF16),
            "scaling": Tensor(None, "float32"),
        },
        outputs={"output": Tensor(("B", "HQ", "S", "D"), BF16)},
        constraints=("HQ % HK == 0", "D == 64 or D == 128 or D == 256"),
        reference="""import torch


def run(query, prefix_key, prefix_value, suffix_key, suffix_value, scaling):
    repeats = query.shape[1] // prefix_key.shape[1]
    key = torch.cat((prefix_key, suffix_key), dim=2).repeat_interleave(repeats, dim=1)
    value = torch.cat((prefix_value, suffix_value), dim=2).repeat_interleave(repeats, dim=1)
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scaling
    return torch.matmul(torch.softmax(scores, dim=-1), value.float()).to(query.dtype)
""",
        workloads=(
            *(
                Workload(
                    f"pi05-denoise-b{b}-prefix{PI05_PREFIX}",
                    {"B": b, "HQ": 8, "HK": 1, "P": PI05_PREFIX, "S": PI05_TOKENS, "D": 256},
                    scalars={"scaling": 0.0625},
                    note="every denoise step, every expert layer; three cameras",
                )
                for b in PI05_BATCHES
            ),
            Workload(
                "pi05-denoise-b1-prefix712",
                {"B": 1, "HQ": 8, "HK": 1, "P": 712, "S": PI05_TOKENS, "D": 256},
                scalars={"scaling": 0.0625},
                note="two cameras",
            ),
        ),
        precision=_tolerance(
            "The production kernel uses split-KV online softmax with exp2; FP32 accumulation "
            "order differs from the materialized Torch softmax."
        ),
        notes="The padding mask is omitted (all keys valid). The production call also "
        "accepts a shared [B,1,1,P+S] padding mask; masked tuning needs a custom benchmark.py.",
    ),
    Operator(
        name="rotate_half_rope",
        summary="Rotate-half RoPE for packed vision Q/K, computed in FP32.",
        models=("qwen25-vln",),
        op_type="rope",
        source="embodiinfer/backend/triton/packed_rope.py",
        entry="rotate_half_rope",
        call="rotate_half_rope(q, k, cos, sin)",
        axes={"T": None, "HQ": None, "HK": None, "D": None},
        inputs={
            "q": Tensor(("T", "HQ", "D"), BF16),
            "k": Tensor(("T", "HK", "D"), BF16),
            "cos": Tensor(("T", "D"), BF16),
            "sin": Tensor(("T", "D"), BF16),
        },
        outputs={"q_out": Tensor(("T", "HQ", "D"), BF16), "k_out": Tensor(("T", "HK", "D"), BF16)},
        constraints=("D % 2 == 0", "D <= 256"),
        reference="""import torch


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def run(q, k, cos, sin):
    cosine = cos.unsqueeze(-2).float()
    sine = sin.unsqueeze(-2).float()
    q_float, k_float = q.float(), k.float()
    q_out = (q_float * cosine + _rotate_half(q_float) * sine).to(q.dtype)
    k_out = (k_float * cosine + _rotate_half(k_float) * sine).to(k.dtype)
    return q_out, k_out
""",
        workloads=(
            Workload("vit-1frame", {"T": 1024, "HQ": 16, "HK": 16, "D": 80}, note="one 448x448 frame"),
            Workload("vit-4frames", {"T": 4096, "HQ": 16, "HK": 16, "D": 80}, note="four frames"),
        ),
        notes="Every vision block applies this before attention. head_dim 80 is not a power of two. "
        "Products of BF16 values are exact in FP32, so the production kernel is bit-exact.",
    ),
    Operator(
        name="segmented_attention",
        summary="Non-causal attention independently inside packed variable-length segments.",
        models=("qwen25-vln",),
        op_type="attention",
        source="embodiinfer/backend/triton/segmented_attention.py",
        entry="segmented_attention",
        call="segmented_attention(q, k, v, segment_offsets, max_query_length=max_length, "
        "max_key_length=max_length)",
        axes={"T": None, "HQ": None, "HK": None, "D": None, "N": None},
        inputs={
            "q": Tensor(("T", "HQ", "D"), BF16),
            "k": Tensor(("T", "HK", "D"), BF16),
            "v": Tensor(("T", "HK", "D"), BF16),
            "segment_offsets": Tensor(("N",), "int32"),
            "max_length": Tensor(None, "int32"),
        },
        outputs={"output": Tensor(("T", "HQ", "D"), BF16)},
        constraints=("HQ % HK == 0", "D <= 256", "N >= 2"),
        reference="""import torch


def run(q, k, v, segment_offsets, max_length):
    del max_length  # A bound for graph-safe kernels; the reference uses the offsets.
    output = torch.empty_like(q)
    repeats = q.shape[1] // k.shape[1]
    scale = q.shape[-1] ** -0.5
    bounds = segment_offsets.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        query = q[start:end].float().transpose(0, 1)
        key = k[start:end].float().repeat_interleave(repeats, dim=1).transpose(0, 1)
        value = v[start:end].float().repeat_interleave(repeats, dim=1).transpose(0, 1)
        probs = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) * scale, dim=-1)
        output[start:end] = torch.matmul(probs, value).transpose(0, 1).to(q.dtype)
    return output
""",
        workloads=(
            Workload(
                "vit-window-1frame",
                {"T": 1024, "HQ": 16, "HK": 16, "D": 80, "N": 17},
                scalars={"max_length": 64},
                fixed={"segment_offsets": tuple(range(0, 1025, 64))},
                note="window-attention blocks, 16 windows of 64 patches",
            ),
            Workload(
                "vit-window-ragged",
                {"T": 1156, "HQ": 16, "HK": 16, "D": 80, "N": 21},
                scalars={"max_length": 64},
                fixed={"segment_offsets": (*range(0, 1025, 64), 1057, 1089, 1121, 1156)},
                note="partial edge windows of a non-multiple-of-window frame",
            ),
            Workload(
                "vit-full-1frame",
                {"T": 1024, "HQ": 16, "HK": 16, "D": 80, "N": 2},
                scalars={"max_length": 1024},
                fixed={"segment_offsets": (0, 1024)},
                note="the full-attention blocks, one segment per frame",
            ),
        ),
        precision=_tolerance(
            "The production kernel uses blockwise online softmax with exp2; FP32 accumulation "
            "order differs from the materialized Torch softmax."
        ),
        notes="`segment_offsets` holds cumulative token positions; queries and keys share them. "
        "`max_length` bounds every segment so the kernel can launch without host synchronization.",
    ),
    Operator(
        name="rms_norm",
        summary="RMSNorm of one decode row with FP32 variance.",
        models=("streamvln",),
        op_type="norm",
        source="embodiinfer/backend/triton/norm.py",
        entry="rms_norm",
        call="rms_norm(x, weight, eps)",
        axes={"M": None, "H": None},
        inputs={
            "x": Tensor(("M", "H"), BF16),
            "weight": Tensor(("H",), BF16),
            "eps": Tensor(None, "float32"),
        },
        outputs={"output": Tensor(("M", "H"), BF16)},
        constraints=("M == 1",),
        reference="""import torch


def run(x, weight, eps):
    values = x.float()
    normalized = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * weight
""",
        workloads=(Workload("qwen2-7b-decode", {"M": 1, "H": 3584}, scalars={"eps": 1e-6}),),
        precision=_tolerance(
            "The production kernel multiplies by the weight in FP32 and rounds once; the eager "
            "path rounds the normalized row to BF16 first.",
            steps=3,
        ),
        notes="The production dispatcher only routes single-row (decode) inputs to this kernel.",
    ),
    Operator(
        name="add_rms_norm",
        summary="Residual addition fused with RMSNorm of the sum for one decode row.",
        models=("streamvln",),
        op_type="norm",
        source="embodiinfer/backend/triton/norm.py",
        entry="add_rms_norm",
        call="add_rms_norm(residual, update, weight, eps)",
        axes={"M": None, "H": None},
        inputs={
            "residual": Tensor(("M", "H"), BF16),
            "update": Tensor(("M", "H"), BF16),
            "weight": Tensor(("H",), BF16),
            "eps": Tensor(None, "float32"),
        },
        outputs={"summed": Tensor(("M", "H"), BF16), "normalized": Tensor(("M", "H"), BF16)},
        constraints=("M == 1",),
        reference="""import torch


def run(residual, update, weight, eps):
    summed = residual + update
    values = summed.float()
    normalized = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + eps)
    return summed, normalized.to(summed.dtype) * weight
""",
        workloads=(Workload("qwen2-7b-decode", {"M": 1, "H": 3584}, scalars={"eps": 1e-6}),),
        precision=_tolerance(
            "The production kernel normalizes the unrounded FP32 sum and applies the weight in "
            "FP32; the eager path rounds the sum and the normalized row to BF16 first.",
            steps=4,
        ),
        notes="Both outputs are compared. The production dispatcher only routes single-row inputs.",
    ),
    Operator(
        name="swiglu",
        summary="SiLU(gate) times up over a packed [gate | up] projection.",
        models=("streamvln",),
        op_type="activation",
        source="embodiinfer/backend/triton/activation.py",
        entry="swiglu",
        # [M, 2, I] is the contiguous [M, 2 * I] projection output, viewed so that
        # every axis is inferable from the inputs.
        call="swiglu(packed.view(packed.shape[0], -1))",
        axes={"M": None, "TWO": 2, "I": None},
        inputs={"packed": Tensor(("M", "TWO", "I"), BF16)},
        outputs={"output": Tensor(("M", "I"), BF16)},
        reference="""import torch.nn.functional as F


def run(packed):
    gate, up = packed.unbind(1)
    return F.silu(gate) * up
""",
        workloads=(
            Workload("qwen2-7b-decode", {"M": 1, "I": 18944}, weight=3.0, note="every decode step"),
            Workload("qwen2-7b-chunk256", {"M": 256, "I": 18944}, note="prefill chunk"),
        ),
        precision=_tolerance(
            "The production kernel keeps SiLU(gate) in FP32 and rounds once; the eager path "
            "rounds SiLU(gate) to BF16 before the multiplication.",
            steps=3,
        ),
        notes="In the model the input is the contiguous [M, 2 * I] gate_up projection: the first "
        "half of each row is the gate, the second half is up. `packed[:, 0]` is the gate.",
    ),
)


def catalog() -> dict[str, Operator]:
    """Operators by task name, in declaration order."""
    return {operator.name: operator for operator in OPERATORS}


def select(names: list[str] | None = None, models: list[str] | None = None) -> list[Operator]:
    """Choose operators by name and/or model; no filter selects every core operator."""
    known = catalog()
    if unknown := set(names or ()) - known.keys():
        raise ValueError(f"Unknown operators: {sorted(unknown)}; known: {sorted(known)}")
    known_models = {model for operator in OPERATORS for model in operator.models}
    if unknown := set(models or ()) - known_models:
        raise ValueError(f"Unknown models: {sorted(unknown)}; known: {sorted(known_models)}")
    return [
        operator
        for operator in OPERATORS
        if (not names or operator.name in names) and (not models or set(models) & set(operator.models))
    ]
