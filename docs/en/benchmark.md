# Benchmarks

Find benchmark results and commands for offline inference, engine optimizations,
and RL integration. Each section describes the workload and timing method.

The latest π0.5 animations and engine comparison are in the
[README](https://github.com/BUAA-CI-LAB/EmbodiInfer#benchmarks) and
[π0.5 benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md).
Find detailed model reports and reproduction commands by workload below.

## Entry points

| Directory / workflow | Model | Workload and reports |
|---|---|---|
| [pi05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) | π0.5 | FlashRT harness, Thor / RTX 4090 presets and LIBERO closed loop; historical replay retained |
| [gr00t-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) | GR00T N1.7 | LIBERO-10, 1,600 observations; RTX 4090 native optimization / batching and historical Thor baseline |
| [openvla-oft-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/openvla-oft-benchmark/README.md) | OpenVLA-OFT | LIBERO-10, 1,600 observations; categorical action-token checkpoint |
| [lingbot-vla-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/lingbot-vla-benchmark/README.md) | LingBot-VLA | RoboTwin2.0, 1,600 observations; three cameras and dual-arm joints |
| [dm05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/dm05-benchmark/README.md) | DM0.5 | LIBERO-10, 1,600 observations; separately prepared OpenDM environment |
| [cosmos-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/cosmos-benchmark/README.md) | Cosmos Policy | LIBERO-10, 1,600 observations; action generation / candidate planning |
| [streamvln-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/README.md) | StreamVLN | R2R / RxR replay, fixed 128 first frames and retained closed-loop reports |
| [activevln-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/activevln-benchmark/README.md) | ActiveVLN | R2R / RxR replay, native engine comparisons, first frames and closed-loop reports |
| [qwenvl-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/qwenvl-benchmark/README.md) | Qwen R2R low / panoramic · NaViDA | First 48 R2R / RxR episodes, 4 recorded frames per episode |
| [3B-navigation](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/3B-navigation/README.md) | Qwen R2R low / panoramic · NaViDA | Real-image full-policy / kernel / compile-cache experiments; individual correctness gates |
| [pi05-quant-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-quant-benchmark/README.md) | π0.5 | Historical B=1 LIBERO replay by device and precision; distinct from the newer FlashRT harness |
| [streamvln-quant-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-quant-benchmark/README.md) | StreamVLN | B=1 full R2R / RxR replay by device and precision |
| [kernel tuning](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/docs/proposals/0008-kernel-tuning.md) | Registry operators / π0.5 | Operator capture, numerical calibration, tuning and profiling; developer workflow |

PI0.5 on Jetson AGX Thor and RTX 4090 against FlashRT, APXinf, Embodied.cpp and OpenPI, with a
LIBERO closed-loop film, is on [PI0.5 on Thor and RTX 4090](pi05-performance.md).

## Workloads and timing

Sample counts are configurable. The shared timing covers preprocessing, inference,
and CPU output, and excludes loading, disk decode, and warmup. These are offline
replay measurements. Panoramic uses a shape-compatible RGB-derived input;
Cosmos measures action generation.

The original benchmark directories ship `setup_env.py` for isolated environments;
ActiveVLN uses the existing `activevln` dependency group. In each directory,
`benchmark.py` handles data sampling, timing, and inference. Qwen Low/Panoramic use SDPA + Inductor
+ CUDA Graph. Each README gives the exact run command.

## Recorded offline results

**Historical aggregate:** the table below preserves the fixed September 7, 2026 report.
It is not a current leaderboard. Later model baselines, native optimizations and
quantization results are linked in the complete index above.

<!-- offline-results:start -->
The following measurements were taken on September 7, 2026, using a Jetson AGX
Thor and an RTX 4090. Models ran one at a time at B=1 in BF16, with the
optimizations specified in each benchmark configuration. Latency excludes
loading, compilation, and warmup.

| Model / data | Calls per card | Thor mean ms | 4090 mean ms | Thor calls/s | 4090 calls/s | 4090 / Thor throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PI0.5 / LIBERO-10 | 1,600 | 165.10 | 78.72 | 6.0569 | 12.7037 | 2.10× |
| Cosmos Policy / LIBERO-10 | 1,600 | 1,333.35 | 468.58 | 0.7500 | 2.1341 | 2.85× |
| StreamVLN / R2R | 2,997 | 491.75 | 168.94 | 2.0335 | 5.9193 | 2.91× |
| StreamVLN / RxR | 3,879 | 502.57 | 169.86 | 1.9898 | 5.8872 | 2.96× |
| Qwen Low / R2R | 192 | 250.67 | 135.04 | 3.9892 | 7.4053 | 1.86× |
| Qwen Low / RxR | 192 | 266.48 | 144.15 | 3.7527 | 6.9371 | 1.85× |
| Qwen Panoramic / R2R | 192 | 834.41 | 449.62 | 1.1984 | 2.2241 | 1.86× |
| Qwen Panoramic / RxR | 192 | 833.35 | 447.97 | 1.2000 | 2.2323 | 1.86× |
| NaViDA / R2R | 192 | 1,145.17 | 414.42 | 0.8732 | 2.4130 | 2.76× |
| NaViDA / RxR | 192 | 1,140.57 | 413.53 | 0.8768 | 2.4182 | 2.76× |

Per-directory READMEs include latency percentiles, output throughput, peak memory,
load and warmup time, and graph-replay status. The table compares the two platforms
at B=1 with their respective Torch CUDA builds and drivers.
<!-- offline-results:end -->

<a id="coverage-gaps"></a>

## Coverage gaps

These experiments remain unmeasured or incomplete in the published reports.
The README matrix and report index inventory available evidence; they do not
mean these gaps were filled.

| Area | Available evidence | Remaining work |
|---|---|---|
| GR00T / Thor | [RTX 4090 native optimization and historical Thor baseline](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) | Retest the native optimization on Thor; device-specific precision experiments. |
| OpenVLA-OFT / LingBot-VLA / Cosmos / DM0.5 | Historical Thor / RTX 4090 BF16 baselines in the model reports above | Optimized paths, applicable quantization profiles and matched reference-engine comparisons. |
| VLA models other than π0.5 | Mainly Thor / RTX 4090, with some H200 mechanism / RL validation | Model-level measurements on AGX Orin, Orin NX and DGX Spark. |
| StreamVLN | Recorded subsets on [Thor](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/thor/README.md), [RTX 4090](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/rtx-4090/README.md) and [NX](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/orin-nx/README.md) | Thor RxR and more closed-loop batch sizes; NX first frames; remaining quantization-suite profiles. |
| ActiveVLN | [Benchmark index](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/activevln-benchmark/README.md) and RLinf experiments below | NX first frames / RxR closed loop; Thor RxR / batched closed loops; RxR B=4 memory limits; full training, convergence and the pending logprob consistency check. |
| Qwen R2R low / panoramic · NaViDA | Thor / RTX 4090 replay, 3B experiments and A800 DP/TP | Orin platforms, quantization and larger batches; the NaViDA graph path that fails its correctness gate is not an admitted optimization. |

New models and deployment work are tracked in the [model roadmap](models.md#roadmap)
and [EmbodiRun coverage and roadmap](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/en/support-matrix.md).
Orin FP8 / FP4 and RTX 4090 FP4 are outside the current hardware profiles.

## Reproducing a result

The October 14 non-quantized Pi05 prefix GELU/product experiment is separate from
the offline table above. On one RTX 4090, B1/H50/ten steps with the unchanged
RoboTwin checkpoint, default-off pointwise fusion improves paired complete-request
P50 from 70.989 to 70.064 ms and 71.127 to 70.193 ms (about 1.3%). The reference is
untouched main with its best tested byte-equivalent KV/prefix/context settings;
their existing gains are part of the baseline. Four cases pass all 280 KV/AE/action
byte comparisons at zero tolerance. GPU replay has its own timing scope, and the
first high-variance run, including a long-tail sample, is retained and disclosed.
See the [Pi05 benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md)
for exact conditions, rejection evidence and limits. This result does not establish
other models, batches, hardware or closed-loop task quality.

Start with the selected benchmark directory's README. Use its environment,
checkpoint, dataset preparation, and measurement command to set up your run.

Record the code revision, GPU, driver, dependency versions, checkpoint, input
selection, precision, batch size, warmup, and measured iteration count alongside
your result. Run the corresponding numerical comparison before interpreting a
speedup.

## Generic mock benchmark

```bash
python benchmarks/benchmark.py --preset small --sweep   # hf vs embodiinfer-eager vs embodiinfer-graph
```

This compares a synthetic policy under batch-one eager, batched eager, and
graph execution where CUDA is available. CPU fallback exercises the software
path. For model throughput, use one of the model-specific benchmarks above.

For Qwen2.5-VL-3B and NaViDA CUDA-graph comparisons, see
[`benchmarks/3B-navigation/`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/3B-navigation/README.md).

The two-GPU data-parallel and tensor-parallel comparison is on the
[parallelism](parallelism.md) page.

<a id="activevln-rlinf-integration"></a>

## ActiveVLN / RLinf integration

**Sum of 8 batched inference calls per training step (seconds)**

| Hardware | Batch | SGLang (s) | EmbodiInfer (s) | Speedup |
|---|---:|---:|---:|---:|
| 2× RTX 4090 | 2 | 2.255 | 0.872 | 2.59× |
| 2× RTX 4090 | 4 | 3.148 | 1.259 | 2.50× |
| 2× RTX 4090 | 8 | 4.789 | 2.057 | 2.33× |
| 2× H20 | 2 | 1.654 | 0.639 | 2.59× |
| 2× H20 | 4 | 2.358 | 1.043 | 2.26× |
| 2× H20 | 8 | 4.125 | 1.916 | 2.15× |

**Full rollout (seconds per training step)**

| Hardware | Batch | SGLang (s) | EmbodiInfer (s) | Speedup |
|---|---:|---:|---:|---:|
| 2× RTX 4090 | 2 | 5.001 | 3.484 | 1.44× |
| 2× RTX 4090 | 4 | 6.690 | 4.944 | 1.35× |
| 2× RTX 4090 | 8 | 10.878 | 7.987 | 1.36× |
| 2× H20 | 2 | 3.392 | 2.322 | 1.46× |
| 2× H20 | 4 | 5.292 | 3.707 | 1.43× |
| 2× H20 | 8 | 9.404 | 7.576 | 1.24× |

Source: [RLinf #1789](https://github.com/RLinf/RLinf/pull/1789), downstream snapshot
[`d909549`](https://github.com/RLinf/RLinf/commit/d9095497b4f1eb2b5419a9ee63e725ca03df53f2).
`Arvil/Qwen2.5-VL-3B_rl_r2r_4000`, BF16, repeated R2R episode 1;
one GPU handles actor / rollout, the other Habitat-Sim rendering.
Means over 5 measured training steps after warmup. The first table sums 8 calls
per step; **it is not single-request latency**. Inference includes preprocessing,
generation, output packing and pending weight application; full rollout also
includes environment interaction and trajectory transfer. Gains include both
runtime and downstream integration optimizations.

The downstream PR remains unmerged. Full HabitatLab training and convergence
are unvalidated; the BF16 incremental / full-history logprob check has maximum
difference 0.119415, exceeding its 0.02 tolerance.

### A800 · Separate GRPO integration validation

[EmbodiInfer #16](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/16) records another workload:
2×A800-SXM4-80GB, one GPU for actor / rollout and the other for HabitatSim, BF16,
the same R2R checkpoint, episode 1, 2 environments, group size 2, 8 turns,
global batch 16 and microbatch 1. Six iterations per backend, the first excluded,
with statistics over the remaining five.

| HF actor / rollout | Training iteration mean ± sample SD | Rollout phase | Actor phase |
|---|---:|---:|---:|
| HF / SGLang | 20.64 ± 0.15 s | 15.27 s | 4.63 s |
| HF / EmbodiInfer | 19.73 ± 0.41 s | 14.77 s | 4.43 s |

This uses an external HabitatSim HTTP validation bridge, rather than the complete
official HabitatLab/Ray service. Trajectories differ; these stage times do not
establish SR/SPL or convergence parity. Measured runtime: `f20e92e`.
It is separate from the RTX 4090 / H20 experiment above.
[Full configuration and validation scope](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/16).

## Engine mechanism and RL integration results

These isolate one mechanism at a time, under the same model, weights, attention
implementation, and precision. Unless noted, measurements are on one H200; RLinf
end-to-end runs use four.

### pi0.5 (4.14B, LeRobot `pi05_base` weights)

- **Correctness.** The self-hosted forward is bit-identical to the LeRobot reference in
  fp32 (`max|Δaction| = 0`), and the CUDA-graph path is bit-identical to eager at the
  same noise.
- **Same-condition comparison.** Against openpi `PI0Pytorch` — the generator RLinf
  actually uses — with the same weights, eager, TF32, and `K = 10`, EmbodiInfer eager is
  1.01–1.09× over `B = 1..64`, slightly faster at the small end and converging as the
  batch becomes compute-bound.
- **CUDA-graph increment** (eager → graph, TF32). The denoising loop is 2.29× at `B = 1`,
  1.32× at `B = 4`, and 1.07× at `B = 8`, converging as the batch grows; end to end the
  engine is 2.01× at `B = 1`.
- **RLinf end to end** (LIBERO PPO, 4×H200, 4 env). Rollout generation is 11.3–12.6 s/it
  against 13.5–14.3 s/it for the native openpi backend, with PPO metrics (ratio,
  approx_kl, clip_fraction, success) at the same level. The ratio-at-`θ₀` quantiles fall
  inside the native backend's own noise floor, which comes from the precision-path
  difference between native rollout and actor recomputation. Captured graphs stay valid
  after weight synchronisation.

### GR00T N1.7 (vendored DiT action head + Qwen3-VL backbone, bf16, `N = 4`)

- **Correctness.** Parity against Isaac-GR00T passes. Ratio-at-`θ₀` for the RLinf
  integration is **exactly 1.0 at every quantile** (bit-identical), the value path is
  bit-identical, and the eval path has zero deterministic deviation.
- **Denoising loop, GPU segment** (graph vs eager, synchronised timing). 2.83× / 2.53× /
  1.89× at `B = 2/4/8` (15.2/17.2/23.5 ms against 43.0/43.6/44.5 ms), reproduced across
  machines.
- **Full predict path** (against the native eager backend, with the CPU preprocessing
  thread count fixed explicitly). 1.549× / 1.288× / 1.115× at `B = 4/8/16`.
- **RLinf end to end** (LIBERO PPO, 4×H200, 8 env). The graph and eager backends land in
  the same noise band (median 14.5 vs 14.7 s/it). In this configuration the rollout wall
  clock is dominated by simulation stepping and per-step CPU work, and a single model
  call is roughly a tenth of the epoch, so the GPU-segment gain is absorbed by
  environment-side cost and noise. Worker-side probes confirm the model-side timing
  matches the isolated measurement. Graph invalidation under offload is handled by
  pointer-change detection plus automatic re-capture, verified in the training loop.

### OpenVLA-OFT (Llama-2 7B + vendored timm dual-tower vision, single forward + categorical head, bf16)

OpenVLA-OFT predicts action tokens in a single forward pass through the
`ActionDecoder` interface.

- **Correctness.** The self-hosted Llama-2 forward is bit-identical to HF
  (`max|Δ| = 0`); vision plus projector is bit-identical to the official implementation
  with an identical action-token argmax; the full chain reaches `max|Δ| ≈ 1.6e-4` against
  the official action logits, which is cross-implementation fp32 noise.
- **RLinf ratio-at-`θ₀`.** In bf16 a categorical log-probability is sensitive to the
  logits, unlike the robustness of a flow Gaussian log-probability. The
  prefill-plus-split-decode path in `encode_prefix` introduced bf16 rounding and a ratio
  floor of `[0.67, 1.57]`. Switching to an op-aligned single full forward against the
  native implementation made the ratio **exactly 1.0**, with a log-probability
  `Δ = 0` and a key set equal on both sides.
- **CUDA graph.** `ForwardGraph` captures the decode segment, where 56 action queries
  attend to the cached KV. Graph against eager is `max|Δ| = 0`, and with 56 tokens over a
  7B model compute-bound, the decode segment gains 1.09×.
- **RLinf end to end** (LIBERO GRPO, with EGL rendering and model CUDA placed
  separately). The full-forward rollout is exactly on-policy with the actor per token
  (`ratio = 1`, `ratio_abs = 0`, `approx_kl = 0`, `clip_fraction = 0`). The predict
  segment times about 2× the native one, which is hidden by the observation-bound
  rollout wall clock, so end to end the two are even.

### LingBot-VLA (Qwen2.5-VL-3B + narrow Qwen2 MoT expert, pi0-style flow-SDE, bf16, `N = 10`)

- **Correctness.** The self-hosted MoT forward (the VL prefill collects KV and the expert
  attends to the frozen prefix KV) has `mean|Δ| = 1.4e-3` against the native reference —
  a velocity residual, not bit-exact.
- **Attention implementation.** The native MoT shared attention is a
  custom `our_eager_attention_forward` with an fp32 core, and the flash branch is not
  implemented in the vendored model, so EmbodiInfer must match it with the `eager`
  backend. With `sdpa` the velocity residual is amplified by the narrow standard
  deviation, pushing ratio-at-`θ₀` to `±16%`. With `eager` it tightens to a q01–q99 range
  of `[0.987, 1.015]` (`±1.5%`) with a median of exactly 1.0.
- **RLinf weight sync.** The self-hosted modules register against the actor's key layout
  by mirroring its `nn.Module` tree: all 1,555 keys match, with
  zero remapping.
- **RLinf end to end** (RoboTwin click_bell GRPO, separate placement). The native and
  EmbodiInfer backends report `ratio = 1.0`;
  `success_once` moves 0.88→0.97 natively against 0.92→0.98 with EmbodiInfer; `approx_kl`
  is 5.5–9e-4 natively against 1.3–6.2e-3, a little higher and traceable to the forward
  residual, consistent with the offline `±1.5%` and still inside the GRPO clip; rollout
  and predict are level at 55 ms against 53–58 ms.

### Cosmos Policy (Cosmos-Predict2 2B video-diffusion DiT + Wan2.1 VAE, bf16, `N = 5` denoising steps)

Cosmos uses a diffusion decoder for action generation and best-of-N planning.
The comparisons below use cosmos-policy's inference implementation as the
reference.

- **Correctness** (one H200, bf16, fixed initial noise, aligned with cosmos-policy's
  native `generate_samples_from_batch`, LIBERO Predict2-2B). Self-hosted DiT forward
  `max||Δ||_inf = 0.032` (mean 3.4e-3, with the net output mean and standard deviation
  matching the native one bit for bit); Wan2.1 VAE encode `max||Δ||_inf = 0.024`;
  end-to-end latent `max||Δ||_inf = 0.024`; action frames (normalised) 1.9e-3; value
  frames 7e-4.
- **Best-of-N planning.** `encode_prefix` (VAE plus text) runs
  **once** and `expand(N)` broadcasts it; the `N` diffusion trajectories denoise action
  and value frames jointly, and the best value wins. Prefix encode is 68.9 ms and a
  single candidate (5 DiT steps) is 159.6 ms, so planning latency is 229/350/580/1032 ms
  at `N = 1/2/4/8`.
- **Prefix reuse increment** (same section as the engine, `N = 4`). Reuse costs 579 ms
  (VAE plus text encoded once) against 909 ms for re-encoding per candidate, saving
  330 ms or about 36%. The DiT attends bidirectionally over all
  frames and there is no clean-frame KV cache, so reuse is limited to the encode stage.
- **CUDA graph.** Cosmos declares `supports_cuda_graph=True` with
  `cuda_graph_kind="diffusion_step"`, and `GraphManager` captures the per-step `denoise`
  (preconditioning, DiT, frame replace) as `CosmosDenoiseGraph` in the same
  `engine/graph.py` dispatch as flow and OFT. The static prefix is copied in once and
  replayed each step. Graph against eager gives a latent `max||Δ||_inf = 0`
  (bit-identical), and `sample_latent` over 5 steps goes from 157.6 to 138.2 ms, or
  1.14×; the 2B DiT is compute-bound so the gain is in per-step launch, diluted because
  the 2ab multistep host-side float64 step stays out of the graph, while a single
  DiT-forward microbenchmark is 1.36×.
- **Engine action path.** `EmbodiInfer("cosmos").act(obs)` runs end to end: `collate`
  (`Observation` to `CosmosBatch`, mapping two cameras from `[0,1]` to `[-1,1]`,
  rescaling proprioception, and taking the precomputed-instruction fast path to T5
  cross-attention), then `encode_prefix`, then diffusion denoising, then an
  `ActionChunk (16,7)` at dataset scale. Batched execution works too. Instructions
  outside the precomputed file are handled by the vendored `T5TextEncoder` leaf
  (`google-t5/t5-11b`).

### Engineering conclusions

- The CUDA-graph gain is concentrated in the small-batch, launch-bound region and
  converges to 1× as the batch grows. How much of it reaches end-to-end rollout is
  roughly the GPU share of the model call — high for pi0.5, low for GR00T — which is
  what the structure predicts.
- A benchmark of CPU preprocessing must fix the thread count explicitly and match the
  deployment environment, where an RL worker is usually single-threaded. Otherwise
  thread oversubscription distorts the CPU segment by more than an order of magnitude
  and inverts the attribution of the bottleneck.
