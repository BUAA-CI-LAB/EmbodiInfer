# 0008 — Pi05 fused operators migrated from ccinfer

- Status: Draft (local implementation for review)
- Author: EmbodiInfer contributors
- Date: 2026-10-09

## 1. Summary

Migrate ccinfer's reusable CUDA/Triton operators and opt-in Pi05 execution plans
into EmbodiInfer's existing backend and policy layers. Keep the public engine,
checkpoint parameters, existing default routes, and training/RL computation intact.

## 2. Motivation and current gap

`policies/pi05/modeling_pi05.py` already supports compact cameras, cached timestep
modulations and full-loop graphs. It still materializes paired MLP projections and
concatenates prefix K/V at every denoise step. ccinfer implements paired BF16
GEMMs with a GELU lookup epilogue, strict residual/RMSNorm fusion, calibrated
FP8/NVFP4 activation encoders, query-major attention, and reusable K/V storage.
Its measured launch profiles cover Thor SM110 and Spark SM121, B1/horizon10/10
steps. Those measurements describe ccinfer, not this migrated adapter.

## 3. Goals and non-goals

Expose migrated operators through an explicit Pi05 optimization configuration.
Preserve LeRobot's tanh GELU, existing RoPE, FP32 precision islands, Euler schedule,
parameter objects and checkpoint keys. Keep native compilation out of imports.
Do not move ccinfer's Runner architecture into this engine, add a runtime dependency
on ccinfer, or copy benchmark calibration files for a different forward contract.

## 4. Design

Operator contracts and lazy registries live in `layers/normalization.py`,
`activation.py`, and `quantization.py`, alongside the existing attention/linear
registries. `OperatorRequest` and `OperatorCapabilities` declare device, dtype,
shape, encoding, arithmetic and warmed-graph support. Unsupported selections fail
explicitly. `OperatorBackends` names implementations and is reusable by other
policies; importing it does not import Triton or compile CUDA sources.

Concrete implementations live under `backend/cuda/`, `backend/triton/`, and
`backend/torch/`. CUDA owns packaged `csrc/` and an atomic, locked,
source/toolchain/device-keyed cache. Common calibrated projections live in
`backend/torch/projection.py`. Paired GEMM tile/tail choices live in their Triton
implementation; Pi05 retains model dimensions, schedule and calibrated per-layer
choices. The initial uncommitted `backend/fused/` import paths are replaced by
these implementation paths; this is a documented migration, not a second ops layer.

A Pi05-local controller owns execution-scoped plans through these contracts;
graphs own independent workspaces. Projection plans expose scope release, so the
policy does not inspect a concrete encoder's cache. The native runtime discards
graphs before operator storage. Composite residual/norm/encoding explicitly
requires compatible implementations; selecting None uses separate operators.
Torch reference implementations exercise the same rounding and buffer contracts.
Existing attention registration routes the query-major and folded-Flash variants.

An alternative was to vendor ccinfer's entire model/Runner. That duplicates the
engine and changes the checkpoint-facing model contract; migrating operators and
adapting them to the existing towers keeps ownership explicit.

## 5. Model-agnosticism verdict

No model-specific engine branches. Backend operators depend only on tensors;
Pi05 profiles, fixed action schedules and tower dimensions remain policy-local.

## 6. Losslessness and precision criterion

Reference: existing Pi05 eager/SDPA route with identical weights, observations,
noise, matmul settings and steps. Strict pointwise RMSNorm keeps Torch's FP32 mean
reduction and both BF16 residual rounding points. A device-local BF16 tanh GELU
table prevents replacing LeRobot's activation with ccinfer's exact GELU.
Paired GEMMs and alternative attention change accumulation and require explicit
opt-in plus reported full-action drift; they are not advertised as bit-exact.
Mixed precision requires a checkpoint hash and a recipe identifying this adapter's
activation contract. ccinfer's existing calibration JSONs are incompatible.

## 7. Implementation plan

Add fused backend operators, Pi05 configuration and an instance-local execution
controller. Wire them into the existing prefix/tower/native-decode entry points.
Defaults keep the previous implementation. Preserve train/refit/device-migration
invalidation and reject unsupported compiler/TP/quantization combinations.

## 8. Test plan

CPU: import safety, configuration/recipe validation, activation semantics, workspace
refresh and lifetime, unchanged parameters/keys, invalid combinations, refit/training
cleanup. CUDA: strict norm, lookup GELU/product, quantized encoders, paired GEMMs,
registered attention, repeated graphs with changed inputs and independent outputs.
Run checkpoint parity in a matching isolated LeRobot environment when available.

Local validation used RTX 4090, Python 3.11.15, Torch 2.13.0+cu130, Triton 3.7.1,
and nvcc 12.6. Relevant CPU/CUDA tests: 140 passed, 18 skipped, including 40
migration tests passed, two native NVFP4 tests skipped on Ada, and 15 operator
registry/reference tests passed. Strict norm and
BF16/FP8 activation encoding matched separate Torch operations byte-for-byte;
K/V-only full-loop output matched the existing native path byte-for-byte. Tiny
model tests verified changed inputs, complete 10-step BF16/FP8 graphs, independent
outputs, eviction/failure cleanup and refit invalidation. With the same tiny-model
weights and noise, replacing CUDA norm with a registered Torch implementation
preserved complete 10-step output bytes; Torch and CUDA FP8 reference plans also
produced identical full-loop output bytes. Paired GEMM/attention
checks bound component error; they do not establish full-model parity.

The complete CPU suite had 496 passed, 93 skipped and one Qwen persistent-cache
failure (`dataclasses.asdict` on Torch cache metadata in Python 3.11), also
reproduced using the unchanged HEAD source. Ruff and wheel inclusion of all seven
CUDA sources passed; the wheel contains no obsolete `backend/fused/` package.
No real LeRobot checkpoint or migrated Thor/Spark latency was validated;
both remote SSH connections timed out. These remain explicit review/deployment
gates rather than inferred results from ccinfer.

## 9. Benchmark plan

Use the existing recorded Pi05 benchmark interface. Compare switches independently
with identical hardware, checkpoint, dtype, masks/noise, B1, horizon10 and all10steps.
Measure warm request and complete model latency without profiler instrumentation.
Before tuning new tiles, attribute the remaining stage cost. Existing ccinfer
latencies provide provenance, not performance claims for EmbodiInfer.

## 10. Risks and limitations

Native encoders require a CUDA toolkit and the corresponding GPU instructions;
calibrated scaled_mm uses the PyTorch API tested in 2.13. Thor/Spark launch choices
are not 4090 profiles. Different activation semantics prohibit blindly reusing
ccinfer calibration. Full task quality is a separate validation requirement.
