# Supported models

`policy` is the identifier passed to the Python engine or the server's
`--policy` option. Choose a checkpoint and the matching dependency environment
before loading a model.

## Model table

| policy | weights | notes |
|---|---|---|
| `pi05` | `lerobot/pi05_base` | flow-matching; EmbodiInfer-owned Gemma forward; fp32 bit-identical to LeRobot |
| `dm05` | user-provided DM0.5 checkpoint and normalization statistics | requires a separately prepared OpenDM environment; see the input and serving notes below |
| `gr00t` | NVIDIA GR00T N1.7 | flow-matching; vendored DiT head + stock Qwen3-VL backbone; parity vs Isaac-GR00T |
| `openvla_oft` | `Haozhan72/Openvla-oft-SFT-*` | non-flow: single forward + categorical action-token head; EmbodiInfer-owned Llama-2 (vendored timm vision) + KV-split decode graph |
| `lingbot_vla` | `robbyant/lingbot-vla-4b` | pi0-style flow (Qwen2.5-VL + narrow Qwen2 MoT expert); EmbodiInfer-owned MoT forward |
| `cosmos` | `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` | **WAM**: video-diffusion DiT denoises action / future-state / value latent frames; EmbodiInfer-owned DiT + EDM sampler, Wan2.1 VAE leaf; model-specific best-of-N planning |
| `activevln` | `Arvil/Qwen2.5-VL-3B_rl_r2r_4000` | **experimental R2R adapter**: episode-scoped growing KV + eager autoregressive action-token decode; CPU/session tests and the pinned parity harness are present, real-checkpoint GPU parity is pending |
| `qwen2.5-vl-3b-r2r-low-level` | `Vebbern/Qwen2.5-VL-3B-R2R-low-level` | recurrent egocentric R2R policy; `Left`/`Right`/`Move`/`Stop`; native graph-backed forward |
| `qwen2.5-vl-3b-r2r-panoramic` | `Vebbern/Qwen2.5-VL-3B-R2R-panoramic` | recurrent panorama + candidate-view selector; candidate metadata lives in `Observation.metadata` |
| `navida` | `waynechu/NaVIDA` | Qwen2.5-VL-3B; official v2 JPEG/history/generation contract; up to six atomic actions; eager or StaticCache manual-CUDA-Graph decode |
| `streamvln` | published local checkpoint layout | recurrent, episode-affine; SlowFast frame selection over a 32-step fast KV window and an eight-feature slow-memory prefix |

## Capabilities and installation

The HTTP and WirelessComm launchers use the same serving adapters. π0.5 supports
cross-session batching with `--max-batch`; DM0.5 and StreamVLN currently serve
one request at a time. Policies without a network adapter use the Python API.

| Policy | Environment | HTTP | Python batching / recurrence | CUDA graphs | Generic RL decoder |
|---|---|---|---|---|---|
| π0.5 | `pi05` | Yes | Stateless batches | Decode; optional native prefix | Yes |
| DM0.5 | External OpenDM environment | Yes | Model-specific; HTTP retains history | Decode | Yes |
| GR00T | Benchmark-specific environment; no dedicated uv group | No | Stateless batches | Decode; optional native prefix | Yes |
| OpenVLA-OFT | `openvla-oft` | No | Stateless batches | Single-forward decode | Yes |
| LingBot-VLA | `lingbot-vla` | No | Stateless batches | Decode | Yes |
| Cosmos | `cosmos` | No | Model-specific candidate planning | Diffusion step | No |
| ActiveVLN | `activevln` | No | Explicit recurrent session, B=1 | Eager path | No |
| Qwen R2R low / panoramic, NaViDA | `qwen25-vln` | No | Explicit recurrent session, B=1 | Policy-native decode | No |
| StreamVLN | `streamvln` | Yes | Explicit recurrent session, B=1 | Policy-native decode | No |

For GR00T, follow the environment and asset preparation in its
[benchmark guide](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md).
For other groups, see [Installation](installation.md). Each model's input
layout and preprocessing are described below. Use the camera, state, and token
layout expected by your checkpoint. CUDA graphs require compatible hardware
and input shapes.

## Optimization profiles

EmbodiInfer combines model-specific execution paths with reusable operator
backends. Select a profile for the policy and GPU you are using:

- **CUDA Graphs** replay static computation. Depending on the policy, capture
  covers prefix encoding, individual decode steps, or a full denoising loop.
- **torch.compile / Inductor** compiles selected model paths, including native
  π0.5 and GR00T computation and StreamVLN language prefill.
- **Triton kernels** provide specialized attention and fused normalization,
  rotary embeddings, and gated activations. Native π0.5 uses these operators
  alongside prefix KV reuse across denoising steps.
- **Low-precision linear layers** offer FP8, INT8, and NVFP4 backends. The
  π0.5 and StreamVLN benchmark profiles use the following combinations:

| Hardware | Baseline precision | Quantized profiles |
|---|---|---|
| Jetson AGX Orin | BF16 | INT8 |
| RTX 4090 | BF16 | FP8 |
| Jetson AGX Thor | BF16 | FP8, NVFP4 |

These profiles run at batch size one, with CUDA Graphs enabled and Inductor
disabled for quantized execution. On RTX 4090, π0.5 uses Triton weight-only FP8;
StreamVLN uses native W8A8 FP8. Thor uses native W8A8 FP8 and native NVFP4.
Quantization changes model numerics: choose a profile using both its latency
measurements and output comparisons with BF16.

See the [π0.5 precision benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-quant-benchmark/README.md)
and [StreamVLN precision benchmark](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-quant-benchmark/README.md)
for checkpoint, GPU software, backend, and numerical-comparison details.
The [architecture guide](architecture.md#cuda-graph-capture) describes graph
capture; [parallelism](parallelism.md) covers batching across devices and tensor
parallel execution.

## Roadmap

**SmolVLA** and the base **OpenVLA** model are planned additions. OpenVLA-OFT
is already implemented and remains a separate policy.

- [ ] Checkpoint loading and observation preprocessing
- [ ] Inference adapters and reference-output checks
- [ ] HTTP / WirelessComm serving integration
- [ ] Deployment bindings in [EmbodiRun](https://github.com/BUAA-CI-LAB/EmbodiRun)

## Synthetic CPU fixture

A synthetic `mock_flow_vla` policy exercises engine, batching, and rollout
interfaces through the Python API without a checkpoint. It runs on CPU or CUDA.

## Per-model serving notes

### Pi0.5

π0.5 supports Python batches and cross-session HTTP/WirelessComm batching.
Configure camera names and state fields using the
[adapter JSON](serving.md#adapter-json), then set `--max-batch` for a shared service.

For PI0.5, `attention="eager"` also selects the reference projection layout: camera
views are encoded separately, and Q/K/V and gate/up projections remain separate.
Fused backends retain the batched and fused routes. This distinction matters for
rollout/actor log-probability comparisons on selectively cast checkpoints: changing
GEMM shapes can change rounding even with the same attention formula.

### DM0.5

Provide a DM0.5 checkpoint and its `norm_stats.json`; see
[`examples/dm05_arx5_inference.py`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/examples/dm05_arx5_inference.py). The standard
`embodiinfer-http-serve --policy dm05` launcher registers the model; HTTP serving
also requires `policy_kwargs.is_history: true` in its adapter JSON.

DM0.5 forwards `action_horizon` and `default_num_steps` to the OpenDM loader, so the
LIBERO experiment can use its native 10-action chunk. `liger_kernel=True` enables
OpenDM's fused inference modules; it defaults to `False`. The
LIBERO benchmark checks that requested fused modules were actually installed.

DM05 supports per-step and full-loop CUDA Graph decoding. The graph stores a
32-column internal state and caches separate layouts for differing prefix lengths;
public 7/14-column slicing and output conversion happen after replay.

DM0.5 uses the shared `FlowDecoder` and `engine/rollout/logprob.py`. Its policy
supplies the internal state shape, public-action conversion, and likelihood mask;
padded columns remain in the trajectory but are excluded from logprob.

DM0.5 accepts the shared `embodiinfer.Observation`: RGB float tensors in `images`, a state
tensor, raw text in `instruction`, and an empty long tensor for `instruction_tokens`
(OpenDM performs tokenization). Optional `metadata` keys are `robot_type` (defaults
to the policy setting), `speed` (defaults to `"0.5"`), `control_mode`, `state_desc`,
`history_images`, and `history_placeholder_text`. Camera order follows the
checkpoint's `image_prompts`. For different camera sizes, `processor_dm05.pack_images`
returns a padded tensor and `(height, width)` pairs for `metadata["image_sizes"]`; the
processor restores each original view before OpenDM preprocessing. HTTP serving
manages history per session. Only the processed batch layout, `DM05Batch`, is
model-specific.

### Navigation profiles

The three 3B navigation profiles use the same explicit-session API and can opt into
explicit manual CUDA Graph replay on CUDA. Low-level and panoramic graph their native
next-token forward. NaViDA v2 uses eager multimodal prefill, a fixed-address
StaticCache decode graph, and official temperature/top-k sampling outside the graph.
For the panoramic profile, `images[0]` is the panorama and candidate images can be
supplied as
`metadata={"candidate_images": [...], "candidates": [{"relative_angle": ..., "distance": ...}]}`.
The Python engine keeps recurrent episodes in separate sessions at batch size one.
For raw-model throughput experiments, `benchmarks/3B-navigation/benchmark.py`
batches fresh, aligned runner inputs.

Reference producers live under `scripts/activevln/`. Recurrent policies currently
use synchronous execution; pipeline/async execution, group sampling, and
best-of-N are unavailable.

### StreamVLN

StreamVLN is a recurrent policy with one episode assigned to each replica.

`embodiinfer/models/streamvln` loads the sharded weights directly and runs SigLIP vision, the
projector, and Qwen2 attention/RoPE/SwiGLU with the multimodal splice and KV cache.
`embodiinfer/policies/streamvln` holds the deterministic Habitat prompt, SlowFast frame
selection, episode state, autoregressive generation, and symbolic action parsing.

Each replica owns its recurrent sessions. A session retains sparse, pinned-host vision
features and a Qwen KV cache for a 32-step fast window. At a window boundary,
EmbodiInfer selects eight features uniformly from the full episode history, rebuilds the
slow-memory prefix, and starts a new fast cache. Both the feature history and the KV
update are part of the recurrent transaction and are committed only after decoding
succeeds.

```python
from embodiinfer import Vvla

model = Vvla(
    "streamvln",
    checkpoint="/models/streamvln",
    dtype="bfloat16",
    decode_block_size=4,
    cache_history_features=True,
    fast_action_decode=True,
)
```

Install the isolated dependencies with `uv sync --frozen --no-dev --group streamvln`.
This profile pins PyTorch 2.10.0, Torchvision 0.25.0, and Transformers 4.51.3; the Linux
lock uses CUDA 13.0 wheels and matching Triton 3.6.0. Do not combine it with model groups
that require a different Torch or Transformers runtime. The separately measured Torch
2.13 Thor and 4090 environments are created with
[`benchmarks/streamvln-benchmark/setup_env.py`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/setup_env.py)
and that directory's platform-specific requirements files.

StreamVLN currently runs one recurrent session per replica. Its optimised path captures
up to four dependent autoregressive tokens per CUDA Graph replay and uses compiled vision
and prefill runners. Q/K/V and MLP gate/up weights share contiguous storage but retain
independent GEMM boundaries, so BF16 accumulation matches the published projection
semantics; `cuda_graph=False` keeps the eager reference path. The optional fixed-response
fast path evaluates the published assistant prefix in the causal prefill, verifies every
prefix token with the full vocabulary and the normal repetition penalty, and accepts the
four-action result only when all action tokens and the terminal EOS match the response
grammar. A mismatch rolls the logical KV length back and runs the ordinary
autoregressive decoder. Set `fast_action_decode=False` for direct reference
measurements. Tensor parallelism is not supported. The upstream code and weights are
non-commercial; downstream use must comply with their license.

For separated model timing, `StreamVLNPolicy.prepare_prefix` performs observation
preprocessing and the device transfer of the current input, and `encode_prepared_prefix`
performs the vision and text encoding. `StreamVLNDecoder.generate_tokens` returns the
full generation result, and `finalize_generation` then does text decoding, action
parsing, and the assembly of the next memory/trace. The usual call still goes through
`encode_prefix` and `decoder.decode`, which invoke those stages in order and preserve the
original behaviour and session semantics. The benchmark uses the same stage boundaries to
time the model separately, while the model-internal cache management of history features
still belongs to prefill.
