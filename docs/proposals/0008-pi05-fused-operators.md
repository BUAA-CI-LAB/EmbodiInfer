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

A policy-local controller owns execution-scoped plans through these contracts.
Each graph owns independent workspaces. Projection plans expose scope release;
the runtime releases graphs before operator storage. Refit, training transitions
and device changes invalidate derived vision casts, projection packs and graphs.
Residual/norm/encoding fusion requires compatible operator implementations;
`norm_quant: null` selects separate execution for reference comparisons.

The RLinf profile selects exact GELU, BF16 vision encoder/projector with FP32
patch and position embeddings, FP32 time/RoPE factors, BF16 query prescaling,
separate Q/K/V, and contiguous BF16 down-projection matrices. It batches active
cameras, compacts globally invalid prefix columns, stops the final prefix block
after K/V, and prepares action masks/positions/rotary factors outside denoising.
CUDA rotary and the Thor FP4 lookup epilogue use the reusable registries.

`from_json(path)` loads a complete deployment recipe.
`from_runtime_json(path)` loads a compact action-precision recipe and selects the
RLinf inference profile. Activation ranges are checkpoint-bound and profile-bound;
the loader preserves supplied scales. `prefix_layers` and `action_layers` select
gate/up and down precision independently. An empty tower configuration stays BF16.

A separate model-specific runner was considered. It would duplicate graph and
execution ownership; policy-local plans reuse the existing model-neutral engine.

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

## 7. Implementation plan

Add operator contracts, concrete backends, configuration and an instance-local
Pi05 controller. Wire the controller into prefix/tower/native-decode entry points.
Defaults retain existing execution. Reject unsupported Inductor, TP and global
quantization combinations before model execution. Native mixed GEMM requires the
tested Torch `scaled_mm` API; dependencies remain optional.

## 8. Test plan

CPU tests cover import safety, configuration and recipe validation, activation
semantics, workspace ownership, unchanged parameter keys/identity, unsupported
combinations and invalidation. CUDA tests cover strict norm/GELU, FP8/NVFP4
encoding, paired GEMMs, registered attention and full ten-step graph replay with
changed inputs. Eviction, capture failure and refit tests check storage lifetime.
Torch reference operators exercise the same buffer and rounding contracts.

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
Numerical drift and task quality require separate evaluation. Measured gains
include vision precision and execution-profile changes as well as kernel fusion;
the original-numerics comparison reports the narrower operator-only benefit.
