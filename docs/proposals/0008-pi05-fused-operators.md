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
Also expose an explicit `numerics="rlinf"` compatibility profile for the complete
ccinfer inference contract: exact GELU, BF16 vision encoder/projector with FP32
patch/position embedding, FP32 time and rotary factors, BF16 query prescaling,
separate Q/K/V, contiguous BF16 down-projection matrices, and the original
arithmetic action / lookup prefix paired-GEMM split. This is a distinct opt-in
numerical contract; the existing LeRobot default is unchanged.
Do not move ccinfer's Runner architecture into this engine, add a runtime dependency
on ccinfer, or copy benchmark calibration files for a different forward contract.

## 4. Design

Operator contracts and lazy registries live in `layers/normalization.py`,
`activation.py`, `rotary.py`, and `quantization.py`, alongside the existing attention/linear
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

The complete profile additionally batches active cameras, compacts all globally
invalid prefix columns, stops the last prefix block after K/V, and hoists action
mask/position/rotary preparation out of the denoising loop. CUDA rotary and the
Thor FP4 lookup epilogue are selected through reusable operator registries.
`from_ccinfer_json` imports the frozen checkpoint-bound production precision
recipe together with this profile. Validation compares directly against ccinfer
with unchanged recipe scales, inputs and noise, separately from LeRobot numerical
gates. Derived vision casts and projection packs belong to the policy controller
and are invalidated with graphs on refit; original parameters and keys remain intact.

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
activation contract. Original ccinfer calibration is incompatible with the default
LeRobot contract; the explicit RLinf profile imports it without changing scales.

`prefix_layers` and `action_layers` independently select per-layer gate/up and
down precision. Both use `MlpLayerPrecision`; `ActionLayerPrecision` remains a
compatibility alias. The initial migration exposed only action precision, which
omitted ccinfer's experimental NVFP4-prefix/BF16-action configuration. Prefix
plans now share the same operator contracts, hash binding, graph ownership and
refit invalidation. Defaults remain BF16. ccinfer's deployed FP8 configurations
protect eight projection groups in BF16; uniform all-action quantization is a
different experiment and cannot determine whether those configurations migrated
successfully. LeRobot calibration uses its own activation ranges. The complete
RLinf compatibility profile retains both the original protection choices and
ranges and compares against the corresponding ccinfer configuration directly.

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
Subsequent validation on 2026-10-10 used identical real RLinf-Pi05-LIBERO-SFT
weights, recorded LIBERO inputs, and diffusion noise on AGX Thor, DGX Spark and
AGX Orin. This validates this adapter's LeRobot tanh-GELU/RoPE forward contract;
it does not establish RLinf/JAX alignment or task success. Six observations were
held out from four calibration observations. Error gates were frozen from the
native BF16 versus FP32-tensor reference before evaluating candidates. Complete
32-dimensional actions were retained; gates apply to the seven LIBERO dimensions.
K/V-only output matched all 32 dimensions byte-for-byte on all three devices.
Opt-in BF16 fusion/attention paths passed the held-out numerical gates. Uniform
action-MLP FP8/NVFP4 recipes did not pass and are not deployment recommendations.

The updated attention/operator regression passed 73 tests with one skip on each
Thor/Spark runtime, 70 with four skips on Orin's host Torch 2.8, and 68 with six
skips in Orin's Python 3.12/Torch 2.6 container. The latter is below LeRobot's
declared Torch minimum and is a legacy compatibility experiment, not a supported
environment certification. Query-major attention now rejects Torch versions
without the required FP32-output BMM API before kernel construction. Orin's
local FP32 oracle exhausted CUDA allocation capacity; remaining runs reuse the
matching Thor FP32 oracle with hashes and experiment conditions checked. Existing
Orin/Thor FP32 horizon-10 actions differed by at most 1.5e-6 across all dimensions.
The benchmark documentation records measured configurations and limitations.

The prefix precision follow-up passed 80 operator/attention tests with one expected
architecture skip on each Thor/Spark runtime, including independent tower choices,
combined NVFP4 prefix/FP8 action graphs and stale-calibration rejection after refit.
Real-weight guarded action FP8 passed five of six frozen numerical gates on both
devices; prefix NVFP4/BF16 action passed none. Full 32-dimensional actions were
byte-identical between CUDA fusion and separate pointwise references on both
devices. The FP8 reference uses Torch encoding; the NVFP4 reference retains the
CUDA encoder/GEMM and validates pointwise fusion only. Native BF16 outputs matched
the earlier same-device baseline byte-for-byte. Legacy global quantization was
disabled throughout and is rejected when combined with migrated plans. These
findings correct the initial experiment scope; they do not establish low precision
task quality or justify removing unrelated global quantization backends.

## 9. Benchmark plan

Use `benchmarks/pi05-benchmark/validate_optimizations.py` for independent operator
comparisons and the existing recorded interface for full request measurements.
Compare identical hardware, checkpoint, dtype, masks/noise, B1 and all ten steps;
report checkpoint horizon 50 and the explicit inference horizon 10 separately.
Keep `highest` and `high` FP32 GEMM precision groups separate. Compare against
the existing Inductor/Triton-attention graph route as well as native SDPA graphs.
Record prefix/diffusion timing, action errors and graph-cache stability after
warmup. GPU-ready model execution excludes CPU preprocessing and postprocessing.
Before tuning new tiles, attribute the remaining stage cost. Existing ccinfer
latencies provide provenance, not performance claims for EmbodiInfer. Measured
results and reproduction commands live in the
[canonical benchmark documentation](../../benchmarks/pi05-benchmark/README.md).

## 10. Risks and limitations

Native encoders require a CUDA toolkit and the corresponding GPU instructions;
calibrated scaled_mm uses the PyTorch API tested in 2.13. Thor/Spark launch choices
are not 4090 profiles. Different activation semantics prohibit blindly reusing
ccinfer calibration across numerical profiles. Full task quality is a separate
validation requirement. The RLinf profile explicitly requires inference-only
CUDA eval/no-grad execution and rejects hidden-state/training fallbacks.

Complete-migration validation used the frozen ccinfer production package as an
external benchmark reference, with its original action recipes and separately
frozen experimental prefix ranges. Thor and Spark each passed byte comparisons
for four formats (BF16, protected action FP8, prefix NVFP4, combined), six held-out
observations, every retained prefix K/V, all ten velocities and complete actions.
Graph replay also matched after changing noise. Runtime sources do not import
ccinfer; only `benchmarks/pi05-benchmark/compare_ccinfer.py` uses that checkout.
Portable recipes and reproduction conditions are in the canonical model/API and
benchmark documents. The execution infrastructures remain distinct: ccinfer
captures a full request; EmbodiInfer retains prefix and diffusion graph ownership.
