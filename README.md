<p align="center">
  <img src="assets/embodiinfer-logo.svg" alt="EmbodiInfer" width="440">
</p>

<h3 align="center">One engine for embodied inference and RL rollout.</h3>
<p align="center">
  <a href="https://embodiinfer.readthedocs.io/">Documentation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#supported-models">Models</a> ·
  <a href="#demos">Demos</a> ·
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

## Demos

### π0.5 · Jetson AGX Thor

[![LIBERO closed-loop comparison of OpenPI, FlashRT and EmbodiInfer on Thor](docs/assets/pi05/libero-thor-comparison.webp)](benchmarks/pi05-benchmark/README.md)

OpenPI BF16, FlashRT NVFP4 and EmbodiInfer `nvfp4-nvfp4`, LIBERO-spatial task 0.
**2× playback**, with real elapsed time on screen; all three engines complete
this recorded task. [Original video and conditions](benchmarks/pi05-benchmark/README.md).

### π0.5 · RTX 4090

[![LIBERO closed-loop comparison of OpenPI, FlashRT and EmbodiInfer on RTX 4090](docs/assets/pi05/libero-4090-comparison.webp)](benchmarks/pi05-benchmark/README.md)

OpenPI BF16, FlashRT FP8 and EmbodiInfer `fp8-fp8`, task 6 / initial state 2,
10 denoising steps, each engine using the GPU exclusively. **1× playback**;
all three complete this recorded task.
[Original video and conditions](benchmarks/pi05-benchmark/README.md).

## Performance

### π0.5 · Thor / RTX 4090 engine comparison

| Device | Precision | EmbodiInfer preset | EmbodiInfer | FlashRT | APXinf | Embodied.cpp | OpenPI |
|---|---|---|---:|---:|---:|---:|---:|
| Jetson AGX Thor | FP4 | `nvfp4-nvfp4` | **20.04** (49.9 Hz) | 21.07 | — | — | — |
| Jetson AGX Thor | FP8 | `fp8-fp8` | **33.60** (29.8 Hz) | 40.70 | 40.88 | — | — |
| Jetson AGX Thor | 16-bit | `bf16` | **61.63** (16.2 Hz) | 69.24 (FP16) | 67.12 | 223.97 | 261.18 |
| RTX 4090 | 8-bit | `fp8-fp8` | **17.06** (58.6 Hz) | 22.06 | 40.62 (INT8) | — | — |
| RTX 4090 | 16-bit | `bf16` | **30.08** (33.2 Hz) | 31.93 (FP16) | 32.06 | 57.48 | — |

Latency in **ms**, with EmbodiInfer's inference rate in parentheses.
FlashRT v0.2.0 `examples/thor/pi05.py`, the same `RLinf-Pi05-LIBERO-SFT` weights,
B=1, two 224×224 cameras, 10 denoising steps and action horizon 10.
**Mean P50 of 3 interleaved rounds**, each with 200 warmup and 1,000 timed calls;
Thor FP4 uses a separate interleaved run with the same settings.
Thor uses MAXN and locked clocks; RTX 4090 uses default clocks.
[Full configuration and reproduction](benchmarks/pi05-benchmark/README.md).

### π0.5 · RTX 4090 official LIBERO-spatial evaluation

| Configuration | Denoising steps | Success rate | Latency P50 | Rate |
|---|---|---:|---:|---:|
| OpenPI · PyTorch BF16 | 10 | 494/500 · 98.8% | — | — |
| **EmbodiInfer · `fp8-fp8`** | **2, ρ = 3** | **495/500 · 99.0%** | **13.34 ms** | **75.0 Hz** |
| EmbodiInfer · `fp8-fp8` | 3, ρ = 1 | 495/500 · 99.0% | 13.79 ms | 72.5 Hz |
| EmbodiInfer · `fp8-fp8` | 2, ρ = 1 | 491/500 · 98.2% | 13.33 ms | 75.0 Hz |
| EmbodiInfer · `fp8-fp8` | 3, ρ = 2 | 490/500 · 98.0% | 13.80 ms | 72.5 Hz |
| EmbodiInfer · `fp8-fp8` | 4, ρ = 1 | 489/500 · 97.8% | 14.21 ms | 70.4 Hz |

OpenPI's official `pi05_libero` checkpoint and unmodified evaluation entrypoint:
**10 tasks × 50 trials, 500 total**, executing the first 5 actions of each chunk.
ρ controls the flow time grid; reduced-step configurations use the same trained
weights. This table uses that checkpoint's own FP8 calibration, separate from the
RLinf checkpoint above. Latency is measured by `libero/latency.py` on an idle GPU;
the films above use 10 denoising steps.
[Evaluation, calibration and complete results](benchmarks/pi05-benchmark/README.md).

Thor NVFP4 channel scaling draws on
[SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html).
It succeeds in 25/30 task-0 trials, compared with 26/30 for FlashRT NVFP4.
[Thor calibration and evaluation](docs/en/pi05-performance.md).

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

![Applications, RL trainers, and EmbodiRun connect to a shared engine with model-specific policy adapters](docs/assets/engine-overview.svg)

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

**Agent-tuned kernels:** the [π0.5 developer workflow](docs/proposals/0008-kernel-tuning.md) supports operator capture, calibration and tuning, drawing on [NVIDIA KDA](https://github.com/NVlabs/kda). User-facing hardware / environment tuning is planned.

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
