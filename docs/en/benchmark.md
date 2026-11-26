# Benchmarks

Find benchmark results and commands for offline inference, closed-loop navigation
and RL integration. Each section describes the workload and timing method.

The latest π0.5 animations and engine comparison are in the
[README](https://github.com/BUAA-CI-LAB/EmbodiInfer#benchmarks) and
[π0.5 benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md).
Find detailed model reports and reproduction commands by workload below.

## Latest results

The most recent batch-one result for each model and device. Workloads, weights and
timing scopes differ between rows, so compare values only within a row; the linked
report holds the full conditions and is authoritative. Dates are measurement dates
where the report records them, otherwise the month the result was published.

| Model | Device | Workload · precision | EmbodiInfer | Reference | Date | Report |
|---|---|---|---:|---|---|---|
| π0.5 | Jetson AGX Thor | FlashRT v0.2.0 harness · NVFP4 / FP8 / BF16 | 20.04 / 33.60 / 61.63 ms | FlashRT 21.07 / 40.70 / 69.24 (FP16) ms | 2026-11 | [π0.5 on Thor and RTX 4090](pi05-performance.md) |
| π0.5 | RTX 4090 | FlashRT v0.2.0 harness · FP8 / BF16 | 17.06 / 30.08 ms | FlashRT 22.06 / 31.93 (FP16) ms | 2026-11 | [π0.5 on Thor and RTX 4090](pi05-performance.md) |
| π0.5 | AGX Orin 32GB | 200 LIBERO-10 frames · BF16 | 302.0 ms | Existing Triton kernels 396.6 ms | 2026-11-02 | [pi05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) |
| π0.5 | Orin NX 16GB | `RLinf-Pi05-LIBERO-SFT` · W8A8 prefix + W4A16 action | 321.9–338.3 ms | — | 2026-10-12 | [pi05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) |
| π0.5 | DGX Spark | `RLinf-Pi05-LIBERO-SFT` · NVFP4 prefix + FP8 action | 65.76 ms | Original EmbodiInfer BF16 109.33 ms | 2026-10-10 | [pi05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) |
| GR00T N1.7 | RTX 4090 | LIBERO-10, 1,600 observations · BF16 | 31.30 ms | PhyAI 33.97 ms | 2026-09-09 | [gr00t-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) |
| GR00T N1.7 | Jetson AGX Thor | LIBERO-10, 1,600 observations · BF16, before native optimization | 92.47 ms | — | 2026-09-08 | [gr00t-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) |
| OpenVLA-OFT | Thor / RTX 4090 | LIBERO-10, 1,600 observations · BF16 | 149.36 / 63.90 ms | — | 2026-09-08 | [openvla-oft-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/openvla-oft-benchmark/README.md) |
| LingBot-VLA | Thor / RTX 4090 | RoboTwin2.0, 1,600 observations · BF16 | 212.17 / 88.36 ms | — | 2026-09-08 | [lingbot-vla-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/lingbot-vla-benchmark/README.md) |
| DM0.5 | Thor / RTX 4090 | LIBERO-10, 1,600 observations · BF16 | 260.93 / 140.37 ms | — | 2026-09-08 | [dm05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/dm05-benchmark/README.md) |
| Cosmos Policy | Thor / RTX 4090 | LIBERO-10, 1,600 observations · BF16, N=1 | 1,323.97 / 444.62 ms | — | 2026-09-08 | [cosmos-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/cosmos-benchmark/README.md) |
| ActiveVLN | RTX 4090 | 128 R2R first frames · BF16 | 55.39 ms | SGLang 177.22 ms, vLLM 117.26 ms | 2026-11-02 | [RTX 4090](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/activevln-benchmark/rtx-4090/README.md) |
| ActiveVLN | Jetson AGX Thor | 128 R2R first frames · BF16 / INT8 | 116.57 / 114.74 ms | — | 2026-10 | [Thor](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/activevln-benchmark/thor/README.md) |
| ActiveVLN | Orin NX 16GB | 128-episode R2R closed loop · BF16 / INT8 | SR 43.75% / 49.22%; model 691.34 / 537.55 ms | — | 2026-10 | [Orin NX](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/activevln-benchmark/orin-nx/README.md) |
| StreamVLN | RTX 4090 | 128 R2R first frames · BF16 | 91.25 ms | Official implementation 276.84 ms | 2026-11-02 | [RTX 4090](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/rtx-4090/README.md) |
| StreamVLN | Jetson AGX Thor | 128 R2R first frames · BF16 / INT8 | 173.19 / 180.10 ms | — | 2026-10 | [Thor](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/thor/README.md) |
| StreamVLN | Orin NX 16GB | 128-episode R2R closed loop · INT8 | SR 54.69%; model 1,359.43 ms | — | 2026-10 | [Orin NX](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/orin-nx/README.md) |
| StreamVLN | AGX Orin 32GB | 48-episode R2R replay · BF16 / INT8 | 970.95 / 2,970.90 ms | — | 2026-09 | [streamvln-quant-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-quant-benchmark/README.md) |
| Qwen R2R low | Thor / RTX 4090 | 192 R2R calls · BF16 | 212.51 / 118.75 ms | — | 2026-09-08 | [qwenvl-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/qwenvl-benchmark/README.md) |
| Qwen R2R panoramic | Thor / RTX 4090 | 192 R2R calls · BF16 | 773.61 / 398.82 ms | — | 2026-09-08 | [qwenvl-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/qwenvl-benchmark/README.md) |
| NaViDA | Thor / RTX 4090 | 192 R2R calls · BF16 | 1,145.61 / 377.54 ms | — | 2026-09-08 | [qwenvl-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/qwenvl-benchmark/README.md) |

Two-GPU data and tensor parallelism are on the [parallelism](parallelism.md) page,
and ActiveVLN rollout inside RLinf is [below](#activevln-rlinf-integration).

## Entry points

| Directory / workflow | Model | Workload and reports |
|---|---|---|
| [pi05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md) | π0.5 | FlashRT harness, Thor / RTX 4090 presets, LIBERO closed loop and other-platform results |
| [gr00t-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) | GR00T N1.7 | LIBERO-10, 1,600 observations; RTX 4090 native optimization, reference engines and batching; Thor baseline |
| [openvla-oft-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/openvla-oft-benchmark/README.md) | OpenVLA-OFT | LIBERO-10, 1,600 observations; categorical action-token checkpoint |
| [lingbot-vla-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/lingbot-vla-benchmark/README.md) | LingBot-VLA | RoboTwin2.0, 1,600 observations; three cameras and dual-arm joints |
| [dm05-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/dm05-benchmark/README.md) | DM0.5 | LIBERO-10, 1,600 observations; separately prepared OpenDM environment |
| [cosmos-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/cosmos-benchmark/README.md) | Cosmos Policy | LIBERO-10, 1,600 observations; action generation |
| [streamvln-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/README.md) | StreamVLN | R2R / RxR replay, 128 first frames, official-implementation comparison and closed-loop navigation |
| [activevln-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/activevln-benchmark/README.md) | ActiveVLN | R2R / RxR replay, SGLang / vLLM comparisons, first frames and closed-loop navigation |
| [qwenvl-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/qwenvl-benchmark/README.md) | Qwen R2R low / panoramic · NaViDA | First 48 R2R / RxR episodes, 4 recorded frames per episode |
| [3B-navigation](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/3B-navigation/README.md) | Qwen R2R low / panoramic · NaViDA | Real-image eager / CUDA Graph parity, kernel and compile-cache tools |
| [pi05-quant-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-quant-benchmark/README.md) | π0.5 | AGX Orin BF16 / INT8 B=1 LIBERO replay |
| [streamvln-quant-benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-quant-benchmark/README.md) | StreamVLN | B=1 full R2R / RxR replay by device and precision, measured before the October optimizations |
| [kernel tuning](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/docs/proposals/0008-kernel-tuning.md) | Registry operators / π0.5 | Operator capture, numerical calibration, tuning and profiling; developer workflow |

PI0.5 on Jetson AGX Thor and RTX 4090 against FlashRT, APXinf, Embodied.cpp and OpenPI, with a
LIBERO closed-loop film, is on [PI0.5 on Thor and RTX 4090](pi05-performance.md).

## Workloads and timing

Sample counts are configurable. Unless a report says otherwise, timing starts from
decoded CPU observations and ends with CPU actions, covering preprocessing,
inference and output, and excludes loading, disk decode and warmup. Most results
replay recorded observations offline; closed-loop navigation results come from
EmbodiRun's Habitat runner and report SR/SPL separately. Panoramic uses a
shape-compatible RGB-derived input; Cosmos measures action generation.

The original benchmark directories ship `setup_env.py` for isolated environments;
ActiveVLN uses the existing `activevln` dependency group. In each directory,
`benchmark.py` handles data sampling, timing, and inference. Qwen Low/Panoramic use SDPA + Inductor
+ CUDA Graph. Each README gives the exact run command.

<a id="coverage-gaps"></a>

## Coverage gaps

These experiments remain unmeasured or incomplete in the published reports.

| Area | Available evidence | Remaining work |
|---|---|---|
| GR00T / Thor | [RTX 4090 native optimization and Thor baseline](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md) | Retest the native optimization on Thor; device-specific precision experiments. |
| OpenVLA-OFT / LingBot-VLA / Cosmos / DM0.5 | Thor / RTX 4090 BF16 baselines in the model reports above | Model-specific optimized paths, applicable quantization profiles and matched reference-engine comparisons. |
| VLA models other than π0.5 | Thor / RTX 4090 only | Model-level measurements on AGX Orin, Orin NX and DGX Spark; multi-GPU execution. |
| StreamVLN | First frames on [Thor](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/thor/README.md) and [RTX 4090](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/rtx-4090/README.md); closed loop on Thor, RTX 4090 and [NX](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/orin-nx/README.md); quantized replay measured before the October optimizations | Thor RxR and BF16 B=8 closed loops; NX first frames; AGX Orin and FP8 / NVFP4 measurements on current code; DGX Spark. |
| ActiveVLN | [Benchmark index](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/activevln-benchmark/README.md) and RLinf experiments below | NX first frames / RxR closed loop; Thor RxR / batched closed loops; RxR B=4 memory limits; AGX Orin / DGX Spark; full training, convergence and the pending logprob consistency check. |
| Qwen R2R low / panoramic · NaViDA | Thor / RTX 4090 replay and A800 DP/TP | Orin platforms, quantization and larger batches. |

New models and deployment work are tracked in the [model roadmap](models.md#roadmap)
and [EmbodiRun coverage and roadmap](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/en/support-matrix.md).
Orin FP8 / FP4 and RTX 4090 FP4 are outside the current hardware profiles.

## Reproducing a result

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

For per-sample eager / CUDA Graph parity checks of Qwen2.5-VL-3B and NaViDA, see
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
