<p align="center">
  <img src="assets/embodiinfer-logo.svg" alt="EmbodiInfer" width="440">
</p>

<h3 align="center">One engine for embodied inference and RL rollout.</h3>
<p align="center">
  <a href="https://embodiinfer.readthedocs.io/">Documentation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#supported-models">Models</a> ·
  <a href="#vla-and-vln-demos">Demos</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#deployment-and-recipes">Deployment / Recipes</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://embodiinfer.readthedocs.io/"><img src="https://readthedocs.org/projects/embodiinfer/badge/?version=latest" alt="Documentation"></a>
</p>

**EmbodiInfer is an inference and RL-rollout engine for embodied models.** Run
manipulation policies, world-action models, and recurrent navigation policies
through a common Python interface, with model-specific execution optimizations
and shared batching, session, and multi-GPU machinery.

**Start with EmbodiInfer** for models, measurements and Python / serving interfaces.
Use [EmbodiRun](https://github.com/BUAA-CI-LAB/EmbodiRun) for one-command deployment
examples, Recipes and the deployment framework that connects robots or simulators.

| Entry point | What it provides |
|---|---|
| **EmbodiInfer · Model inference** | Checkpoints, policy adapters, operator optimization, batching, multiple GPUs and RL rollout. |
| [**EmbodiRun · Deployment and execution**](https://github.com/BUAA-CI-LAB/EmbodiRun) | Robot / simulator integration, service orchestration, action execution, recording and task Recipes. |

![Applications, RL trainers, and EmbodiRun connect to a shared engine with model-specific policy adapters](docs/assets/engine-overview.svg)

## VLA and VLN demos

### VLA · π0.5 / LIBERO

[![LIBERO closed-loop comparison of OpenPI, FlashRT and EmbodiInfer on RTX 4090](docs/assets/pi05/libero-4090-comparison.webp)](benchmarks/pi05-benchmark/README.md)

RTX 4090, task 6 / initial state 2, 10 denoising steps, each engine using the GPU
exclusively. **1× playback**; all three engines complete this recorded task.
The film shows one closed loop; aggregate timings and success rates follow below
and in the [report](docs/en/pi05-performance.md).

<details>
<summary>Jetson AGX Thor · OpenPI / FlashRT / EmbodiInfer (2× playback)</summary>

![LIBERO closed-loop comparison on Thor](docs/assets/pi05/libero-thor-comparison.webp)

Same task initial state, with a new decision each control step. On-screen timers
show real elapsed time; all three engines complete this recorded task.
[Conditions, success rates and original video](benchmarks/pi05-benchmark/README.md).

</details>

### VLN · ActiveVLN / Habitat

[![ActiveVLN original-speed comparison of EmbodiInfer, SGLang and vLLM inside EmbodiRun](https://raw.githubusercontent.com/BUAA-CI-LAB/EmbodiRun/aea124bf8f7eda613daa50098181d532fccd1903/docs/assets/demos/activevln-engines.webp)](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/en/demos/activevln-engines.md)

RTX 4090, BF16, B=1, the same R2R episode at **1× playback**.
[EmbodiRun](https://github.com/BUAA-CI-LAB/EmbodiRun) owns observations, HTTP requests
and action execution. This episode takes **1.979 s** with EmbodiInfer, 2.736 s with
SGLang and 2.682 s with vLLM; all finish 0.673 m from the goal.
[Recording conditions and integration status](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/en/demos/activevln-engines.md).

## Performance

[Model inference](#vla--π05-across-inference-engines) · [RL rollout](#rl-rollout--activevln--rlinf) · [Model × hardware coverage](#model--hardware-benchmark-coverage)

### VLA · π0.5 across inference engines

| Device | EmbodiInfer preset | EmbodiInfer | FlashRT | APXinf | Embodied.cpp | OpenPI |
|---|---|---:|---:|---:|---:|---:|
| Jetson AGX Thor | `nvfp4-nvfp4` | **20.04 ms** | 21.07 ms (FP4) | — | — | — |
| Jetson AGX Thor | `fp8-fp8` | **33.60 ms** | 40.70 ms | 40.88 ms | — | — |
| Jetson AGX Thor | `bf16` | **61.63 ms** | 69.24 ms (FP16) | 67.12 ms | 223.97 ms | 261.18 ms |
| RTX 4090 | `fp8-fp8` | **17.06 ms** | 22.06 ms | 40.62 ms (INT8) | — | — |
| RTX 4090 | `bf16` | **30.08 ms** | 31.93 ms (FP16) | 32.06 ms | 57.48 ms | — |

FlashRT v0.2.0 `examples/thor/pi05.py`, with each engine's loading and `run()`
swapped in: the same `RLinf-Pi05-LIBERO-SFT` weights, 8 interleaved LIBERO observations,
a 13-token prompt, B=1, two 224×224 cameras, 10 denoising steps and action horizon 10.
**Mean P50 of 3 interleaved rounds**, each with 200 warmup and 1,000 timed calls;
Thor FP4 uses a separate interleaved run with the same settings.
Thor runs in MAXN with locked clocks; RTX 4090 uses default clocks.
Precision differences are shown in the table; `—` means unreported.
[Full configuration, numerical checks and reproduction](docs/en/pi05-performance.md) ·
[Benchmark harness and reports](benchmarks/pi05-benchmark/README.md).

**500 LIBERO-spatial trials:** with OpenPI's official weights and evaluation entrypoint,
EmbodiInfer `fp8-fp8`, 2 denoising steps / ρ=3 achieves **495/500 (99.0%),
13.34 ms / 75.0 Hz**; OpenPI BF16 with 10 steps achieves 494/500 (98.8%).
This is a separate reduced-step configuration from the 10-step latency table.
[Evaluation and calibration](docs/en/pi05-performance.md).

### VLN · ActiveVLN across native engines

| Batch | EmbodiInfer mean / P95 | SGLang SRT mean / P95 | vLLM mean / P95 |
|---:|---:|---:|---:|
| 1 | **55.39 / 57.88 ms** | 177.22 / 218.77 ms | 117.26 / 131.11 ms |
| 2 | **73.60 / 77.25 ms** | 214.48 / 252.51 ms | 152.81 / 163.19 ms |
| 4 | **112.02 / 113.83 ms** | 280.95 / 319.75 ms | 223.69 / 232.43 ms |
| 8 | **220.13 / 290.67 ms** | 364.19 / 407.99 ms | 315.69 / 323.11 ms |

RTX 4090 24 GB, BF16, `Arvil/Qwen2.5-VL-3B_rl_r2r_4000@1609873`,
the first frame of each of 128 fixed R2R val-unseen episodes, one full warmup pass
and one measured pass. Timing is **CPU RGB → complete CPU actions for the whole batch**,
including preprocessing; it excludes loading, warmup, HTTP and simulation.
SGLang includes local SRT process communication. History, content caches and learned
speculative candidates are reset per batch. Engines use separate compatible
environments and their own optimizations; action outputs do not all match.
This is a performance comparison. The B=8 first-batch spike remains in the statistics.
[Configuration, output checks and first/subsequent batches](benchmarks/activevln-benchmark/rtx-4090/README.md).

| More results | Start here |
|---|---|
| StreamVLN vs. official implementation, B=1/2/4/8 | [RTX 4090 report](benchmarks/streamvln-benchmark/rtx-4090/README.md) |
| ActiveVLN first frames, long-history replay and historical closed-loop SR/SPL | [Benchmark index](benchmarks/activevln-benchmark/README.md) · [Thor](benchmarks/activevln-benchmark/thor/README.md) · [Orin NX](benchmarks/activevln-benchmark/orin-nx/README.md) |
| Historical BF16 optimizations and batch sweeps for π0.5 / GR00T | [π0.5](benchmarks/pi05-benchmark/README.md#其他平台与早期结果) · [GR00T](benchmarks/gr00t-benchmark/README.md) |
| Quantization, world-action models and multiple GPUs | [π0.5 quantization](benchmarks/pi05-quant-benchmark/README.md) · [StreamVLN quantization](benchmarks/streamvln-quant-benchmark/README.md) · [All benchmarks](docs/en/benchmark.md) · [Parallelism](docs/en/parallelism.md) |


### RL rollout · ActiveVLN / RLinf

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
[Full conditions and separate A800 measurements](docs/en/benchmark.md#activevln-rlinf-integration).

### Model × hardware benchmark coverage

**✓ Recorded results · ◐ Partial / historical or experimental integration · — No published result**.
This inventory includes frozen historical measurements; it does not claim that
current main was retested on every platform. Each cell links to its report.
Workloads and timing scopes differ across models and devices.

| Model | Jetson AGX Thor | RTX 4090 | AGX Orin 32GB | Orin NX 16GB | DGX Spark | H200 | A800 ×2 | H20 ×2 |
|---|---|---|---|---|---|---|---|---|
| π0.5 | [✓ FP4 / FP8 / BF16 + LIBERO](benchmarks/pi05-benchmark/README.md) | [✓ BF16 / FP8 + LIBERO](benchmarks/pi05-benchmark/README.md) | [✓ H10 / H50](benchmarks/pi05-benchmark/README.md) | [✓ W8A8 / W4A16](benchmarks/pi05-benchmark/README.md) | [✓ 1.66× preset](benchmarks/pi05-benchmark/README.md) | [✓ engine / RLinf](docs/en/benchmark.md) | [✓ DP / TP](docs/en/parallelism.md) | — |
| GR00T N1.7 | [◐ BF16 baseline](benchmarks/gr00t-benchmark/README.md) | [✓ native / batch](benchmarks/gr00t-benchmark/README.md) | — | — | — | [✓ engine / RLinf](docs/en/benchmark.md) | — | — |
| OpenVLA-OFT | [◐ BF16 baseline](benchmarks/openvla-oft-benchmark/README.md) | [◐ BF16 baseline](benchmarks/openvla-oft-benchmark/README.md) | — | — | — | [✓ engine / GRPO](docs/en/benchmark.md) | — | — |
| LingBot-VLA | [◐ BF16 eager](benchmarks/lingbot-vla-benchmark/README.md) | [◐ BF16 eager](benchmarks/lingbot-vla-benchmark/README.md) | — | — | — | [✓ engine / GRPO](docs/en/benchmark.md) | — | — |
| Cosmos Policy | [◐ BF16 baseline](benchmarks/cosmos-benchmark/README.md) | [◐ BF16 baseline](benchmarks/cosmos-benchmark/README.md) | — | — | — | [✓ best-of-N / graph](docs/en/benchmark.md) | — | — |
| DM0.5 | [◐ BF16 baseline](benchmarks/dm05-benchmark/README.md) | [◐ BF16 baseline](benchmarks/dm05-benchmark/README.md) | — | — | — | — | — | — |
| StreamVLN | [✓ first frame / loop / quant.](benchmarks/streamvln-benchmark/thor/README.md) | [✓ native / official / loop](benchmarks/streamvln-benchmark/rtx-4090/README.md) | [◐ BF16 / INT8 offline](benchmarks/streamvln-quant-benchmark/README.md) | [◐ R2R INT8 loop](benchmarks/streamvln-benchmark/orin-nx/README.md) | — | — | — | — |
| Qwen R2R low | [◐ BF16 offline](benchmarks/qwenvl-benchmark/README.md) | [✓ offline / graphs](benchmarks/3B-navigation/R2R_VLNCE_THROUGHPUT_REPORT.md) | — | — | — | — | [✓ DP / TP](docs/en/parallelism.md) | — |
| Qwen R2R panoramic | [◐ BF16 offline](benchmarks/qwenvl-benchmark/README.md) | [✓ offline / graphs](benchmarks/3B-navigation/R2R_VLNCE_THROUGHPUT_REPORT.md) | — | — | — | — | [✓ DP / TP](docs/en/parallelism.md) | — |
| NaViDA | [◐ BF16 offline](benchmarks/qwenvl-benchmark/README.md) | [✓ offline; graph parity failed](benchmarks/3B-navigation/R2R_VLNCE_THROUGHPUT_REPORT.md) | — | — | — | — | [✓ DP / TP](docs/en/parallelism.md) | — |
| ActiveVLN | [✓ first frame / R2R loop](benchmarks/activevln-benchmark/thor/README.md) | [✓ first frame / engines / loop](benchmarks/activevln-benchmark/rtx-4090/README.md) | — | [◐ R2R BF16 / INT8 loop](benchmarks/activevln-benchmark/orin-nx/README.md) | — | — | [◐ RLinf validation](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/16) | [◐ RLinf rollout](https://github.com/RLinf/RLinf/pull/1789) |

Updated from the November 2, 2026 11×7 inventory with the latest π0.5 results
in [#22](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/22), VLN engine comparisons
in [#18](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/18), and experimental
A800 / H20 RLinf integration. H200 mainly covers engine mechanisms and RL;
A800 DP/TP and RLinf validation are separate workloads. H20 results come from
an unmerged downstream PR. Orin FP8 / FP4 and RTX 4090 FP4 are outside the
measured hardware profiles, rather than coverage gaps.
[Methods and benchmark index](docs/en/benchmark.md).

## Supported models

✓ Implemented · ◐ Experimental · ○ Planned.
See the [model reference](docs/en/models.md) for checkpoints and validation scope.

| Family | Model | Python inference | HTTP / WirelessComm | Capabilities and limits |
|---|---|---|---|---|
| VLA | π0.5 | ✓ | ✓ | Cross-session batching; BF16 / hardware-specific quantized presets; RL decoder. |
| VLA | GR00T N1.7 | ✓ | — | Batching and RL decoder; benchmark-specific environment. |
| VLA | OpenVLA-OFT | ✓ | — | Single-forward action decoding and RL decoder. |
| VLA | LingBot-VLA | ✓ | — | Flow matching and RL decoder. |
| VLA | DM0.5 | ✓ | ✓ | Separate OpenDM environment; network serving B=1. |
| VLN | StreamVLN | ✓ | ✓ | Recurrent history; explicit native tensor batches; generic network serving B=1. |
| VLN | Qwen R2R low / panoramic · NaViDA | ✓ | — | Explicit recurrent sessions and policy-native decoding. |
| VLN | ActiveVLN | ◐ | — | Native recurrent / tensor-batch API; generic serving adaptation pending merge, real-weight GPU parity pending. |
| World-action | Cosmos Policy | ✓ | — | Diffusion action generation and candidate planning; no generic RL decoder. |
| Planned | SmolVLA · OpenVLA base model | ○ | ○ | Model adapter, input processing and serving integration pending. |

`—` means no generic network adapter; a deployment-specific service is a separate
integration. [Full capability and installation matrix](docs/en/models.md#capabilities-and-installation) ·
[Model roadmap](docs/en/models.md#roadmap).

## Quick start

### Run without a checkpoint

For a first run without a checkpoint, install the core package with Python
3.10+ and [uv](https://docs.astral.sh/uv/) 0.12.x:

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiInfer.git
cd EmbodiInfer
uv sync --frozen
uv run python examples/quickstart.py
```

This CPU-friendly example uses a synthetic policy to walk through single and
batched requests and print their action-chunk shapes.
The Linux lock includes CUDA-enabled Torch wheels, so installation
can be large even on a CPU-only machine.

### Load π0.5

Use the isolated π0.5 profile and a CUDA GPU:

```bash
uv sync --python 3.12 --frozen --no-dev --group pi05
uv run --no-sync python examples/pi05_inference.py \
  --ckpt lerobot/pi05_base --envs 1
```

This example loads real weights but uses synthetic observations to demonstrate
the engine API. Follow [Quick start](docs/en/quickstart.md) for the input contract
and [Serving](docs/en/serving.md) to connect real observations and configure their
mapping to a checkpoint.

### Choose an integration

| Workflow | Interface | Start here |
|---|---|---|
| Inference inside an application | `EmbodiInfer.act(observation)` or `EmbodiInfer.act(observations)` | [Python API](docs/en/api.md) |
| Network inference for a robot runtime | HTTP / WirelessComm sessions | [Serve and send your first observation](docs/en/serving.md) |
| RL rollout and weight updates | Rollout / refit interfaces | [RL integration](docs/en/api.md#generating-rl-rollouts) |
| Multi-GPU execution | Data parallelism / tensor parallelism | [Parallelism](docs/en/parallelism.md) |

To share a π0.5 service across clients, add `--max-batch 3 --max-wait-ms 5`
to the HTTP or WirelessComm launcher. Requests from independent sessions share
batches of up to three; a partially filled batch runs when the 5 ms collection
window expires. This window excludes queueing and inference time. See the
[serving guide](docs/en/serving.md#share-one-service-across-clients) for setup.

The distribution is named `embodiinfer`; Python imports, the `embodiinfer-*` command-line
entry points, and the `embodiinfer.policy.*` wire schemas all use the same name.

## Deployment and Recipes

[EmbodiRun](https://github.com/BUAA-CI-LAB/EmbodiRun) provides one-command example
entrypoints and the deployment framework. Prepare the checkpoint, assets or
hardware described by each Recipe, then use the shared `embodirun example` CLI
to configure, check, launch and stop the task.

| Scenario | Deployment entry | Current status |
|---|---|---|
| VLA · SO-101 / π0.5 | [Grasping Recipe](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/so101_grasping.md) · [Backend comparison](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/engine_comparison/README.md) | Physical footage available; configure the checkpoint and hardware. |
| VLN · MicroDuck / MuJoCo | [Navigation Recipe](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/microduck_vln/README.md) | Dedicated simulation Recipe; scene assets, model and GPU required. |
| VLN · Habitat / three engines | [Recording and configuration](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/en/demos/activevln-engines.md) | Experimental; GPU validation of the reusable vLLM entrypoint is pending. |
| VLN · Go2 / ActiveVLN | [Physical demo](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/en/demos/go2-activevln.md) | Footage available; adaptation remains in [Run #13](https://github.com/BUAA-CI-LAB/EmbodiRun/pull/13). |
| Agent · XLeRobot | [Mobile manipulation Recipe](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/xlerobot_snack_delivery/README.md) | Software rehearsal and supervised hardware workflow. |

[Full coverage matrix and roadmap](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/docs/en/support-matrix.md) ·
[All Recipes](https://github.com/BUAA-CI-LAB/EmbodiRun/blob/main/examples/README.md).

## Why EmbodiInfer?

Optimize the model path. Reuse the engine from robot inference to RL rollout.

<table>
<tr>
<td width="50%" valign="top">
<h3>⚡ Capture, compile, replay</h3>
<p><b>CUDA Graphs · torch.compile / Inductor</b></p>
<p>Replay static prefix and decode paths and compile model computation. Policy-specific profiles target the repeated work in denoising and token generation.</p>
<a href="docs/en/architecture.md#cuda-graph-capture">Execution paths →</a>
</td>
<td width="50%" valign="top">
<h3>🎚️ Precision to fit your GPU</h3>
<p><b>BF16 · FP8 · INT8 · NVFP4</b></p>
<p>Choose hardware-specific π0.5 and StreamVLN profiles: INT8 on AGX Orin, FP8 on RTX 4090 and Thor, and NVFP4 on Thor.</p>
<a href="docs/en/models.md#optimization-profiles">Precision and backend guide →</a>
</td>
</tr>
<tr>
<td valign="top">
<h3>🔥 Optimized down to the operators</h3>
<p><b>Triton attention · fused model operations</b></p>
<p>Specialized attention kernels and fused normalization, rotary embeddings, and gated activations accelerate supported native model paths.</p>
<a href="docs/en/models.md#optimization-profiles">Native optimizations →</a>
</td>
<td valign="top">
<h3>🧠 Reuse context, retain history</h3>
<p><b>Prefix KV reuse · recurrent sessions</b></p>
<p>Reuse fixed observation context across denoising steps. Navigation policies retain episode history, with transactional updates and explicit reset and cancellation.</p>
<a href="docs/en/architecture.md#sessionstore">Context and session lifecycle →</a>
</td>
</tr>
<tr>
<td valign="top">
<h3>🚀 From one request to many environments</h3>
<p><b>Request batching · multi-GPU execution</b></p>
<p>Batch observations, aggregate asynchronous requests, and distribute work across replicas. Supported policies also offer tensor parallelism.</p>
<a href="docs/en/parallelism.md">Parallel execution →</a>
</td>
<td valign="top">
<h3>🔄 Inference that fits the RL loop</h3>
<p><b>Action sampling · log probabilities · weight refit</b></p>
<p>Use RL-capable decoders to collect rollouts and refresh policy weights. Keep your trainer and learning objective; reuse model execution.</p>
<a href="docs/en/api.md#generating-rl-rollouts">RL integration →</a>
</td>
</tr>
</table>

Choose features by policy and hardware in the [capability reference](docs/en/models.md#capabilities-and-installation).
Adapters own model-specific optimizations; the shared scheduler stays model-independent.

**Agent-tuned kernels:** coding agents tune operator kernels for a specific GPU and
software stack. On Jetson AGX Orin, with all else equal, the `orin_pi05_r2` profile reduces
π0.5 latency from 396.6 ms with hand-written kernels to **302.0 ms** (BF16, batch size 1;
[conditions](docs/proposals/0008-kernel-tuning.md#93-orin_pi05_r2-in-uniform-bf16),
[example](examples/pi05_inference.py)). The [workflow](docs/proposals/0008-kernel-tuning.md)
is a developer tool for π0.5 today; user-facing tuning for your own hardware and
environment is planned. It draws on
[NVIDIA Kernel Design Agents (KDA)](https://github.com/NVlabs/kda)
([acknowledgments](THIRD_PARTY_NOTICES.md#kernel-tuning-workflow-references)).

## Documentation

| Task | Guide |
|---|---|
| Install a model profile | [Installation](docs/en/installation.md) |
| Run inference | [Quick start](docs/en/quickstart.md) · [Examples](examples/) |
| Deploy a service | [Serving](docs/en/serving.md) |
| Use multiple GPUs | [Parallelism](docs/en/parallelism.md) |
| Integrate or extend the engine | [Architecture](docs/en/architecture.md) · [Python API](docs/en/api.md) |
| Inspect model behavior | [Models](docs/en/models.md) |
| Evaluate performance | [Benchmarks](docs/en/benchmark.md) |
| Profile and tune kernels with an agent | [Kernel tuning workflow](docs/proposals/0008-kernel-tuning.md) |

### Roadmap

| Area | Status and next work |
|---|---|
| Native inference and hardware profiles | π0.5 Thor / 4090 and LIBERO evaluation merged; extend other policies and platforms with measured results. |
| Generic ActiveVLN serving | [#19](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/19) / [#20](https://github.com/BUAA-CI-LAB/EmbodiInfer/pull/20) remain Draft; generic serving is pending integration, with native APIs and recorded demos available. |
| New models | SmolVLA and base OpenVLA adapters with numerical validation. |
| Agent kernel tuning | π0.5 developer workflow available; user-facing tuning for hardware / environment pending. |
| Deployment examples | Complete Go2 and other EmbodiRun Recipes, independent deployment checks and scaling measurements. |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, model integration,
and numerical validation requirements. Use
[GitHub issues](https://github.com/BUAA-CI-LAB/EmbodiInfer/issues) for bugs and
feature requests, and [SECURITY.md](SECURITY.md) for private vulnerability
reports. Community participation follows our [Code of Conduct](CODE_OF_CONDUCT.md).

## Acknowledgments and references

- **Quantization:** we thank Guangxuan Xiao et al. for
  [SmoothQuant (ICML 2023)](https://proceedings.mlr.press/v202/xiao23c.html) and its
  [implementation](https://github.com/mit-han-lab/smoothquant). Our Thor NVFP4 calibration
  adapts its channel-scaling method alongside EmbodiInfer's NVFP4 packing and BF16
  outlier-channel bypass; see the [implementation and measurements](docs/en/pi05-performance.md).
- **Benchmarks and reference engines:** we thank
  [FlashRT](https://github.com/flashrt-project/FlashRT) for its open VLA runtime and
  PI0.5 benchmark; our engine comparison uses its v0.2.0 harness. We also thank
  [APXinf](https://github.com/infinigence/ApxInf),
  [Embodied.cpp](https://github.com/SEU-PAISys/Embodied.cpp),
  [OpenPI](https://github.com/Physical-Intelligence/openpi),
  [LeRobot](https://github.com/huggingface/lerobot),
  [SGLang](https://github.com/sgl-project/sglang) and [vLLM](https://github.com/vllm-project/vllm)
  for model references and comparison implementations.
- **Models, data and kernels:** we thank the upstream model authors,
  [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
  [Habitat](https://github.com/facebookresearch/habitat-lab),
  [RLinf](https://github.com/RLinf/RLinf),
  [PyTorch](https://github.com/pytorch/pytorch), [Triton](https://github.com/triton-lang/triton),
  [CUTLASS](https://github.com/NVIDIA/cutlass),
  [FlashAttention](https://github.com/Dao-AILab/flash-attention),
  [FlashInfer](https://github.com/flashinfer-ai/flashinfer) and
  [NVIDIA KDA](https://github.com/NVlabs/kda).

See [third-party notices](THIRD_PARTY_NOTICES.md) for specific uses, derived code
and license attribution. Comparisons refer to the fixed versions and settings
listed in their reports.

## License

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[third-party notices](THIRD_PARTY_NOTICES.md). Model weights and datasets retain
their upstream licenses.
