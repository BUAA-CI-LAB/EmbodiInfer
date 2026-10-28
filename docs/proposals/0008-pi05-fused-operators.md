# 0008 — Pi05 fused inference operators

- Status: Draft
- Author: EmbodiInfer contributors
- Date: 2026-10-09

## 1. Summary

Add opt-in CUDA/Triton operators and Pi05 execution plans to EmbodiInfer's
existing backend and policy layers. Fuse pointwise operations, reuse attention
storage, and support checkpoint-bound FP8/NVFP4 recipes with CUDA Graphs.

## 2. Motivation and current gap

`policies/pi05/modeling_pi05.py` supports compact cameras, cached timestep
modulations and full-loop graphs. Its remaining costs include separate MLP
projections and pointwise launches, and repeated prefix K/V concatenation in the
denoising loop. The policy needs explicit numerical and precision profiles so
these optimizations can be measured independently of its existing defaults.

## 3. Goals and non-goals

Expose optimization choices through `Pi05OptimizationConfig`. Preserve checkpoint
keys, parameter identity and the default LeRobot forward contract. Keep model
dimensions and schedules in the policy and reusable tensor operators in backends.
Provide an explicit RLinf inference profile with its own numerical contract.
Training, task evaluation and environment orchestration remain outside this work.

## 4. Design

Operator protocols and lazy registries live under `layers/`, alongside existing
attention and linear registries. `OperatorRequest` and `OperatorCapabilities`
declare device, dtype, shape, encoding, arithmetic and warmed-graph support.
`OperatorBackends` selects implementations without importing Triton or compiling
CUDA at package import time. Unsupported combinations fail explicitly.

Concrete implementations live under `backend/cuda/`, `backend/triton/`, and
`backend/torch/`. CUDA owns packaged `csrc/` sources and an atomic, locked cache
keyed by source, toolchain and device. Common calibrated projections live in
`backend/torch/projection.py`. Paired-GEMM tile and tail choices belong to Triton;
model dimensions and per-layer precision choices belong to Pi05.

Optional FlashInfer implementations live together under `backend/flashinfer/`.
They register lazily as attention `flashinfer_fa2`, normalization `flashinfer`,
and projection `flashinfer_cublaslt`. They require Linux/CUDA SM80+ and the
`flashinfer` extra pinned to `flashinfer-python==0.7.2.post1`; importing the
registries or selecting existing defaults never imports this dependency.
The public entry point is `make_policy("pi05", checkpoint=...,
kernel_profile="flashinfer")`. This is an experimental built-in backend profile,
not a verified hardware tuning profile. `KernelSelection.resolve_backend`
maps the three generic operator families to existing lazy registry names without
importing FlashInfer. Pi05 binds those names to its prepared execution plans,
enables native inference/norm execution, disables paired-MLP fusion, and bypasses
the separate Triton attention shortcuts so the profile reaches all three operators.
The existing `none`, `torch`, `auto` and named tuned-profile behavior is preserved;
uncovered operators keep their original implementation preferences.

The profile takes precedence over the three corresponding explicit backend
selections and Gemma attention switches. Other optimization settings and
numerical conventions remain intact; preset horizon/step defaults are resolved
before the profile changes operator placement. Quantized recipes are rejected,
not converted. CUDA eval/no-grad execution remains mandatory. Individual operator
ablations can still use `kernel_profile="none"` with `Pi05OptimizationConfig`.

FA2 uses `BatchPrefillWithRaggedKVCacheWrapper(backend="fa2")` for both prefix
and multi-token action attention. BHSD inputs become NHD without repeating KV
heads. Plans own staging/output tensors and 32 MiB float scratch, plus the
wrapper's integer/host scratch. Boolean masks are repacked on every execution,
with byte alignment per request and zero outputs for fully masked query rows.
Shapes, scale, mask presence and graph/stream ownership key each plan. Widths
64/128/256 are admitted; SigLIP's width 72, additive bias and dropout are rejected.
`ScopedAttentionBackend` extends the existing registry with optional scope binding,
release and clear operations. The Pi05 controller uses that protocol without
importing a concrete FlashInfer implementation.

Gemma RMSNorm accepts BF16 activations and checkpoint FP32 scales only when the
scale can be represented exactly in BF16, as required by FlashInfer's weight
dtype contract. Scale conversion is validated outside capture and tied to the
parameter identity/version. All weights are immutable until graph invalidation.
The normalization protocol advertises adaptive support; Pi05 keeps AdaRMS on its
reference path when the selected backend only implements ordinary Gemma RMSNorm.
Residual addition is rounded before the norm call. FlashInfer's fused-add API
was rejected because it changes this rounding boundary.

Selecting a nondefault projection backend now routes the unfused MLP through
`ProjectionPlan` without requiring a paired-GEMM kernel. FlashInfer handles
unquantized BF16 gate/up/down weights, preserving the native or Torch GELU/product
selection. Attention projections, vision and FP32 action/time projections keep
their existing routes. The adapter calls the pinned cuBLASLt runner with private
32 MiB scratch per device/execution scope, shared by sequential MLP projections.
Each projection owns aligned input/output storage and a warmed algorithm runner
per row count/scope. The public `mm_bf16` API was rejected here because its
process-global workspace can alias independent graphs. These copies and scratch
allocations are explicit costs to measure before any later Orin tuning.

CUDA operator instances bind to their construction device. Plans reject inputs
from another device, and each native launch selects that device and its current
Torch stream, restoring the caller's device afterwards.

A policy-local controller owns execution-scoped plans through these contracts.
Each graph owns independent workspaces. Projection plans expose scope release;
the runtime releases graphs before operator storage. Refit, training transitions
and device changes invalidate derived vision casts, projection packs and graphs.
Residual/norm/encoding fusion requires compatible operator implementations;
`norm_quant: null` selects separate execution for reference comparisons.

For non-quantized `lerobot` native execution, cached AdaRMS and gated residuals
retain the original Triton arithmetic. The strict CUDA fusion follows Torch's
mean reduction and cannot promise byte equality against Triton's reduction tree.
Prefix and other Torch-reference normalization can still use that fusion.
Alternative numerical and calibrated recipes retain their existing contracts.
The real-weight zero-tolerance gate and independent unchanged-main comparison
are recorded in the [Pi05 benchmark](../../benchmarks/pi05-benchmark/README.md).

The `openpi_rlinf` numerical contract follows RLinf's inference implementation:
exact GELU, BF16 vision encoder/projector with FP32
patch and position embeddings, FP32 time/RoPE factors, BF16 query prescaling,
separate Q/K/V, and contiguous BF16 down-projection matrices. Execution can
independently batch active cameras, compact globally invalid prefix columns,
stop the final prefix block after K/V, and prepare action masks/positions/rotary
factors outside denoising.
CUDA rotary and the Thor FP4 lookup epilogue use the reusable registries.

Deployment calls `make_policy("pi05", checkpoint=..., device_type="thor",
preset="nvfp4-fp8")`. A preset is a complete inference recipe. `strict` preserves
LeRobot numerics with pointwise fusion and K/V reuse on all four supported
devices. `bf16` and explicit prefix-action format pairs select the measured
Thor/Spark B1/H10/ten-step plan with `openpi_rlinf` numerics. The factory supplies
the required native flags, default prefix capture and fixed horizon/steps. Engine
graph configuration remains model-neutral. A caller can explicitly disable
prefix capture. Numerical semantics are declared by the preset rather than
guessed from a checkpoint filename. Hardware and calibration are validated.

This keeps the deployment interface small. Research configurations still select
numerical semantics, individual operators and per-projection precision through
`Pi05OptimizationConfig` or resolved JSON. Presets expand to the same immutable
configuration, using `from_preset` for inspection. The previous independent
deployment selectors exposed internal composition work to callers; they are
removed from the unmerged API. No `mixed` alias conflates distinct format pairs.
New calibration/format combinations require their own accuracy/performance
evaluation.

Checkpoint calibration is separate packaged data under `inference/calibration/`.
Schema 2 identifies numerical semantics and checkpoint SHA256, and maps
device → tower → requested format → per-layer formats/ranges. Presets load it
only for low precision; scales are retained exactly. Bundled data covers prefix
NVFP4 and protected action FP8 on Thor/Spark for RLinf-Pi05-LIBERO-SFT with
`openpi_rlinf` numerics. Other tower/format pairs require separately calibrated
data; missing pairs fail explicitly. A custom path supports these pairs and other
checkpoints or numerical contracts. `from_json` and
`to_json` retain the resolved configuration interface for experiments and ablations.

Configuration, presets, operator plans, MLP/vision plans and graph runtime are
cohesive modules under `policies/pi05/inference/`. Reusable backends stay outside
the policy. Public configuration imports remain under `policies.pi05`; the
intermediate action-only precision alias and compact runtime-recipe loader are
removed from the unmerged API in favor of `MlpLayerPrecision` and `from_preset`.

A separate model-specific runner was considered. It would duplicate graph and
execution ownership; policy-local plans reuse the existing model-neutral engine.

### Orin weight-only execution

The opt-in `orin`/`w4a16` recipe loads externally calibrated group64 GPTQ MLP
weights. It keeps BF16 activations and the original FP32 QK/softmax arithmetic;
it does not route INT4 through the FP8/NVFP4 activation encoder. A separate
weight-only protocol/registry accepts natural-column codes and BF16 scales.
Marlin packs these codes once and owns synchronization locks per eager stream
or graph scope. Prefix GeGLU is fused into GEMM; the small action projection
retains its rounded pointwise epilogue. One shared native source retains the
upstream license and the measured arithmetic for each kernel specialization.
The fused prefix kernel processes M128 with two M64 warp groups sharing each
compressed weight tile. Each accumulator keeps its original K order; M64 and
smaller kernels handle the remaining rows. Tile selection and occupancy remain
inside the CUDA backend, without adding deployment configuration.

A rotary-attention preparation contract owns fixed BTNH buffers and cache
conversion. RoPE stores head-major queries; strict softmax stores token-major
probabilities. The following FP32 QK and BF16 probability/value GEMMs therefore
avoid layout clones. Cached prefix conversion occurs once per request. The
policy supplies masks, factors, cache identity and scope lifetimes.

The optional cuBLASLt projection backend owns descriptors and scoped outputs.
It selects an explicit library configuration identity, checks the library
version and fails when that identity is unavailable. All native launches use
the guarded tensor device and its current stream. No modules are monkeypatched,
no benchmark imports occur in production, and no simulator or calibration loop
is added to the runtime. The user cache includes C++/CUDA/header sources,
compiler options and linked libraries; builds remain atomic and locked.

The archive validates checkpoint identity, complete tower coverage, shard hashes,
shapes, decoder layout and scales. Runtime loading never falls back to RTN.
The only permitted unused final-prefix shard is skipped by KV-only execution.
The factory loads on CPU with meta initialization, validates original MLP
dimensions, replaces their storage with archive-backed inference modules and
then places the remaining parameters. No BF16 MLP copy is kept on CUDA.
These opt-in modules require the calibrated artifact for restoration and do not
provide a training or independent BF16 state-dictionary fallback.
Quantized task quality must be evaluated separately from exact migration parity.

### Orin native integer prefix

The additional opt-in `w8a8-w4a16` preset composes a rotated INT8 prefix archive
with the existing GPTQ action archive. A small deployment bundle binds both
manifests and the checkpoint by SHA256. Calibration and archive preparation stay
offline. The integer projection protocol lives in `layers/integer.py`; its Triton
backend owns native INT8/INT32 GEMM, weight scaling and exact rounded GeGLU.
The CUDA encoder preserves the calibrated channel divisor, H128 butterfly,
FP32 division, BF16 rounding and per-token INT8 rounding. The measured down
encoder uses 512 threads; the smaller gate encoder uses 256. All activation/output
storage follows the existing stream/graph scope contract.

Prefix QK uses a private cuBLAS handle with TF32 math mode. It retains the original
FP32 score/softmax storage and BF16 probability/PV path; action QK remains strict.
Every attention execution owns its cuBLAS scratch until its graph is destroyed;
handle dispatch is serialized and binds that scratch after selecting the stream.
This avoids changing Torch global flags or monkeypatching attention dispatch.
The existing strict attention backend and `w4a16` preset retain their arithmetic.
The two numerical approximations require separate task-quality evidence.
Selective W4A4 was screened separately and rejected; its kernels and CUTLASS
dependency are not part of the runtime.

## 5. Model-agnosticism verdict

The engine gains no model-name branches. Backends operate on tensors and public
operator contracts. Pi05 owns its fixed schedule, tower layout, numerical profile
and calibrated precision choices. Other policies can use the same registries.

## 6. Losslessness and precision criterion

The default reference is the existing Pi05 eager/SDPA path with identical weights,
observations, noise, matmul precision, horizon and steps. Strict RMSNorm preserves
Torch's FP32 reduction and both BF16 residual rounding points; lookup GELU
preserves the selected activation's BF16 values. Strict pointwise and K/V-storage
checks require byte equality.

Paired GEMMs and alternative attention change accumulation order. Full-model
comparisons retain all 32 action dimensions and report drift. The LeRobot-profile
held-out gates compare max-abs/RMSE against twice the native BF16 versus
FP32-tensor difference, with floors of `1e-4`/`1e-5`, over the seven LIBERO
dimensions. Gates are frozen before candidate evaluation.

The RLinf profile changes numerical boundaries relative to LeRobot defaults.
Its frozen, matching-format reference checks cover prefix K/V, every velocity
and final actions. FP8/NVFP4 are compared with their corresponding reference
formats; this does not imply equivalence to BF16. Exact-GELU scales must not be
reused for the default tanh-GELU profile. Protected FP8 recipes and uniform
quantization are separate configurations with different error behavior.

FlashInfer is numerically equivalent in operation, not bit-exact in accumulation
order. Component acceptance in `tests/test_flashinfer_backend.py` uses BF16
inputs with fixed seeds and no dropout. FA2 is compared with FP32 Torch SDPA
using identical GQA heads, noncausal Boolean masks and scale (`rtol=2e-2`,
`atol=2e-3`); norm is compared with FP32 Torch statistics and offset affine
(`rtol=1e-2`, `atol=1e-2`); GEMM is compared with BF16 `F.linear`
(`rtol=1e-2`, `atol=2e-2`). These are fixed component limits for the included
random workloads, allowing BF16 rounding and different FP32 reduction trees;
they do not establish a whole-policy action error budget. The tiny-policy graph
test requires bit equality to eager execution of the same FlashInfer recipe,
with identical noise and ten denoise steps. Real-checkpoint velocity/action
comparisons against the selected unmodified numerical profile are still required.

## 7. Implementation plan

Add operator contracts, concrete backends, configuration and an instance-local
Pi05 controller. Wire the controller into prefix/tower/native-decode entry points.
Defaults retain existing execution. Reject unsupported Inductor, TP and global
quantization combinations before model execution. Native mixed GEMM requires the
tested Torch `scaled_mm` API; dependencies remain optional.

## 8. Test plan

CPU tests under `tests/pi05/` cover preset composition, calibration validation,
normal policy construction, import safety, configuration serialization, activation
semantics, workspace ownership, unchanged parameter keys/identity, unsupported
combinations and invalidation. CUDA tests cover strict norm/GELU, FP8/NVFP4
encoding, paired GEMMs, registered attention and full ten-step graph replay with
changed inputs. Eviction, capture failure and refit tests check storage lifetime.
Model-independent fusion checks live in `tests/test_fused_operators.py`.
Small graph fixtures use the public policy constructor and clean up after each test.
Torch reference operators exercise the same buffer and rounding contracts.

`tests/test_flashinfer_backend.py` contains CUDA kernel checks at Gemma shapes,
including B2, head width 256, P576/P672, and T10/T50. It checks dynamic mask
replay and separate graph workspaces. `tests/pi05/test_flashinfer.py` covers
one-click profile selection, complete ten-step graph replay and refit/eviction.
These GPU tests are retained for execution on the target device; they skip when
CUDA or the pinned optional
dependency is absent. No new CPU tests are included for this integration.

Real-weight validation uses recorded RLinf-Pi05-LIBERO-SFT observations and fixed
noise in isolated environments. Thor/Spark checks cover BF16, protected action
FP8, prefix NVFP4 and their combination. Orin checks cover supported BF16 paths;
unsupported low-precision capabilities are explicitly skipped. A separate
NVFP4 pointwise reference retains its CUDA encoder/GEMM and validates only fusion.
The benchmark documentation records historical failures and environment limits.

## 9. Benchmark plan

`benchmarks/pi05-benchmark/compare_optimizations.py` measures an independent
pre-PR checkout and candidate recipes in separate processes. The baseline is
revision `05d603714dd3dcb48c5210810847395da13272aa`, with its existing Inductor,
Triton attention, caching and prefix/full-loop CUDA Graphs enabled. It must not
import newly added optimization implementations.

For B1/horizon10/full ten-step measurements, warm each of six held-out observations
three times and collect two rounds of 30 requests per configuration. Run `high`
and `highest` FP32 GEMM precision separately. Report synchronized wall P50/P95,
prefix/diffusion timings, complete action drift and graph-cache stability.
The timed scope starts at GPU-ready inputs and ends at GPU actions; CPU processing,
transfers, loading and cold compilation/capture are excluded. Two consecutive
rounds in one process do not establish independent-run confidence intervals.

`validate_optimizations.py` measures individual operator choices and H50 separately.
Measured conditions, results and commands belong in the
[canonical benchmark documentation](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md).

## 10. Risks and limitations

Native operators require a compatible CUDA toolkit and device instructions.
NVFP4 requires Blackwell; Thor/Spark launch profiles cover B1/horizon10/ten steps
and must not be generalized to other shapes or devices. The RLinf profile requires
inference-only CUDA eval/no-grad execution and rejects training fallbacks.
The native Pi05 decoder exposes only deterministic `ActionDecoder` capabilities;
it does not advertise stochastic RL sampling or differentiable log-probability
recomputation.
Numerical drift and task quality require separate evaluation. Measured gains
include vision precision and execution-profile changes as well as kernel fusion;
the original-numerics comparison reports the narrower operator-only benefit.

FlashInfer SM87 admission is a capability policy, not an AGX Orin validation
claim. Installation, JIT compilation with the target toolkit, component parity,
real-checkpoint action drift and performance still require testing on the device.
The repository's CUDA 13 lock must not replace a working JetPack/CUDA 12 stack.
The pinned upstream [prefill wrapper](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2.post1/flashinfer/prefill.py)
and [cuBLASLt runner](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2.post1/flashinfer/gemm/gemm_base.py)
are the integration contracts; upgrading that pin requires rechecking mask
packing, workspace ownership and graph-safe algorithm preparation.
