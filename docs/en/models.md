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
| StreamVLN | `streamvln` | Yes | Generic serving B=1; explicit native tensor batches | Vision/prefill/decode/tree | No |

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

#### Opt-in fused operators

Deployment selects a device and a complete preset:

```python
policy = make_policy(
    "pi05", checkpoint="/models/RLinf-Pi05-LIBERO-SFT",
    device_type="thor", preset="nvfp4-fp8",
)
```

| Preset | Devices | Prefix / action MLP | Numerical contract |
|---|---|---|---|
| `strict` | Thor, Spark, Orin, RTX 4090 | BF16 / BF16 | LeRobot |
| `w4a16` | Orin SM87 | GPTQ INT4 weights / BF16 activations in both MLP towers | `openpi_rlinf` |
| `w8a8-w4a16` | Orin SM87 | Rotated native INT8 prefix / GPTQ W4A16 action | `openpi_rlinf` |
| `bf16` | Thor, Spark | BF16 / BF16 | `openpi_rlinf` |
| `bf16-fp8` | Thor, Spark | BF16 / protected FP8 | `openpi_rlinf` |
| `nvfp4-bf16` | Thor, Spark | NVFP4 / BF16 | `openpi_rlinf` |
| `nvfp4-fp8` | Thor, Spark | NVFP4 / protected FP8 | `openpi_rlinf` |

These presets include execution choices, numerical semantics and calibration
selection. The factory supplies required native policy options; deployment does
not assemble individual operator switches. CUDA Graph decoding remains an engine
setting. `prefix_cuda_graph=False` explicitly disables policy prefix capture.

The strict preset enables RMSNorm/residual fusion and prefix/suffix K/V storage
reuse while retaining reference attention and separate BF16 gate/up computation.
It has no fixed B1/H10 requirement. Passing no preset or optimization
configuration retains the original route.
Device selection validates GPU capability at execution; unsupported combinations
fail explicitly. Low precision is an independent, explicitly approximate choice.

The optimized presets select paired BF16 gate/up GEMMs, camera batching, prefix
compaction, last-prefix-block K/V-only execution and action context reuse.
They are available on Thor/Spark and require standard 18-layer towers, B1, inference horizon 10
and ten denoise steps. Thor uses folded FlashAttention and a lookup GELU
encoder with the `openpi_rlinf` contract; Spark uses query-major attention and
its CUDA prefix softmax launch.
Query-major attention requires `torch.bmm(out_dtype=torch.float32)`.

The Orin `w4a16` preset requires an external group64 GPTQ weight archive:

```python
policy = make_policy(
    "pi05", checkpoint="/models/pi05", device_type="orin", preset="w4a16",
    calibration="/models/pi05-gptq/manifest.json",
)
```

It retains all ten denoising steps, BF16 attention/vision and FP32 QK/softmax.
Marlin decodes calibrated INT4 weights during GEMM; no activation quantization,
online weight quantization or runtime column permutation is used. Prefix
GEMM/GeGLU, strict rotary/cache/softmax layout preparation and BF16 cuBLASLt Q/O
plans are selected together. The manifest must bind all MLP shards to the exact
checkpoint SHA256 and preserve natural-column codes and BF16 group64 scales;
corrupt or incompatible artifacts fail before use. The factory stages checkpoint
loading on CPU and releases original BF16 MLPs before CUDA placement; the
calibrated archive becomes their weight source. These inference-only MLPs have
no standalone BF16 state-dictionary or training fallback. Refit invalidates the
recipe and requires a newly calibrated artifact.
This is approximate weight compression, distinct from Blackwell NVFP4, and needs
checkpoint/task-specific quality evaluation. It is validated for B1/H10/ten steps
on SM87 with CUDA 12.6 and cuBLASLt 12.6.4. The native toolchain must be available before
capture. The default presets on other devices are unchanged.

The opt-in Orin `w8a8-w4a16` preset uses native integer Tensor Cores in the
prefix MLPs and retains W4A16 in the action expert. Its independently calibrated
prefix archive stores signed INT8 codes, FP32 per-output scales and channel
divisors. The runtime fuses channel scaling, H128 rotation, BF16 rounding and
per-token activation quantization; GEMM fuses scaling and rounded exact GeGLU.
Each stream or captured execution owns private activation buffers. Attention
uses a private cuBLAS TF32 handle for prefix QK, with FP32 softmax and the original
BF16 probabilities/PV. Action QK stays strict FP32 and vision stays unchanged.
No global Torch precision settings are modified. TF32 accumulation and integer
quantization are approximate choices requiring task-specific evaluation.

Bundle the existing independently calibrated archives without converting weights:

```bash
python scripts/prepare_pi05_mlp_bundle.py \
  --prefix /models/pi05-int8/manifest.json \
  --action /models/pi05-gptq/manifest.json \
  --output /models/pi05-integer.json
```

Pass `preset="w8a8-w4a16", calibration="/models/pi05-integer.json"` to the same
factory. The bundle binds both manifests by SHA256 and requires all used prefix
projections to have passed their frozen calibration gates. Keep both referenced
archive directories with the bundle. There is no runtime calibration or fallback
to an uncalibrated quantizer. This preset requires Triton, SM87, B1/H10/ten steps
and the same checkpoint as its artifacts; the existing `w4a16` preset is unchanged.

There is no ambiguous `mixed` precision option. Pair names explicitly order
prefix then action: for example, `bf16-nvfp4` and `fp8-nvfp4` express other
combinations and require separately calibrated data.
These formats apply to MLP projections; attention, vision and FP32 boundaries
follow the numerical contract. Calibration retains any per-layer BF16 protection.
The packaged `rlinf_libero` data covers only prefix NVFP4 and protected action FP8
on Thor/Spark for RLinf-Pi05-LIBERO-SFT under `openpi_rlinf`. It can supply either
selection or both; other tower/format pairs require their own calibration.
The name identifies the checkpoint, not an optimization preset.

Calibration is keyed by device, tower and format, and is loaded only when low
precision is requested. Missing ranges fail explicitly; the checkpoint SHA256
is checked before weight packing. Use `calibration="/path/to/calibration.json"`
for independently calibrated weights or formats.
RTX 4090 supports BF16/FP8 hardware, and NVFP4 requires Blackwell. The optimized
FP8/NVFP4 presets cover Thor/Spark; Orin also offers calibrated `w4a16` and
`w8a8-w4a16`. Other devices use `strict` or explicitly
constructed operator configurations. FP8/NVFP4 GEMMs require the Torch API
tested in 2.13; the W4A16 path uses standalone CUDA and cuBLASLt.

The `openpi_rlinf` numerical contract follows the inference code in RLinf's
`openpi_rlinf` package: BF16 vision encoder/projector, FP32
patch/position/time/rotary math, BF16 query prescaling, separate Q/K/V,
and contiguous BF16 down matrices. GELU uses the exact rather than tanh formula.
The factory supplies `native_embeddings=True`, `native_inference=True`,
`action_horizon=10` and `default_num_steps=10` for optimized presets.
The checkpoint retains its nominal horizon metadata.
CUDA Graph configuration belongs to `EngineConfig`; prefix capture remains a
policy option. Inductor, global quantization and TP are rejected in combination
with these operator plans. `openpi_rlinf` requires CUDA eval/no-grad inference.
`request_cuda_graph=True` is an additional, default-off policy option for native
CUDA execution with `EngineConfig(use_cuda_graph=True, capture_full_loop=True)`.
It captures prefix encoding and the complete denoising schedule together, keeps
the intermediate K/V private, and returns owned outputs. Staged, recurrent and
RTC-guided requests retain their existing paths. This option preserves the
checkpoint's dtype and the caller's action horizon/step count; it has no preset
that shortens them. Performance must be checked for the target workload.
The native Pi05 decoder exposes deterministic generation only; it does not
implement the `RLDecoder` sampling or differentiable log-probability contract.

For ablations, construct `Pi05OptimizationConfig` directly or use
`dataclasses.replace` on a resolved preset. `MlpLayerPrecision` selects gate/up
and down precision independently in `prefix_layers` and `action_layers`.
`OperatorBackends` selects registered normalization, quantization, GELU/product,
rotary and projection implementations; `norm_quant=None` chooses separate
normalization and encoding. Torch references support FP8; separate NVFP4
comparisons retain the CUDA encoder/native GEMM. `from_json`/`to_json` import
and export resolved configurations for reproducible experiments.

Policy-specific configuration, presets, plans and graph runtime live together
under `policies/pi05/inference/`; reusable operators remain under `layers/` and
`backend/`. Preset resolution imports no optional kernels and does not probe CUDA.

Strict pointwise parity does not establish task quality. Paired GEMMs and
alternative attention can change accumulation; `openpi_rlinf` also changes numerical
boundaries relative to LeRobot. Existing Thor/Spark measurements cover the four
optimized presets listed above; they do not establish performance for separately
calibrated combinations. Validation covers prefix K/V, ten
velocities, full actions and changed-noise graph replay against frozen,
matching-format references. Low precision changes outputs relative to BF16.
See the [measurement conditions and results](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md),
[API example](api.md#pi05-fused-operator-configuration) and
[toolkit requirements](installation.md#pi05-fused-cuda-operators).

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

Closed-loop callers that execute action chunks must distinguish primitive steps
from request IDs. Set `Observation.metadata["stream_step_id"]` to the current
primitive frame index and supply all new frames since the last model call in
`Observation.images`, ordered oldest to newest. Intermediate frames update sparse
visual history without additional language turns. The fast window resets after
crossing a 32-primitive boundary. The serialized B1 policy supports this contract;
the explicit native batching runtime rejects strides above one. Without the
metadata, the recorded-input contract remains one new frame per request.
The HTTP adapter accepts the same `stream_step_id` metadata with one to four
images named `observation.images.rgb.0` through `.3` in chronological order
(or the configured camera field with those suffixes). Frames must share a
resolution. Single-frame requests may retain the original camera field name;
request `step_id` still counts model calls. Session retries do not advance the
primitive history twice.

INT8 loading converts CPU projections individually before transferring their
packed payloads to the serving device, avoiding a full BF16 allocation first.
Vision, embeddings, LM head and excluded projections keep the requested dtype.
W8A8 output quality still requires a separate comparison against BF16.

The eager attention path retains at most 64 MiB of reusable causal masks per
backbone, evicting the least recently used shapes. Larger masks are created for
the current call without being cached. This bounds retained mask storage across
episodes without changing attention values or the recurrent KV cache; it is not
a limit on total inference memory.

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

StreamVLN generic serving runs one recurrent session per replica. Its original optimised path captures
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

Three independent, disabled-by-default options extend this path:

- `gpu_preprocess=True` uploads uint8 pixels and performs Pillow-compatible
  integer bicubic resize and SigLIP normalization on CUDA. Host validation and
  float-to-uint8 conversion remain on the CPU. It requires the published slow
  SigLIP processor configuration and preserves its integer-valued image behavior.
- `gpu_history_features=True` keeps owned copies of the existing sparse visual
  history on the GPU. This avoids host offload and reload at window boundaries,
  uses additional device memory, and requires `cache_history_features=True`.
- `speculative_tokens=2` through `8` enables experimental repeated-action
  proposals (`0` disables them). Verification uses full-vocabulary greedy
  selection and the original repetition penalty, commits only the matching
  prefix, and decodes non-action tokens serially. It requires CUDA, Triton,
  BF16 and `cuda_graph=True`. The verifier retains serial GEMM shapes and
  attention/residual-normalization arithmetic, while batching independent row
  operators and capturing candidate graphs during startup. It bypasses the
  existing response-envelope prefill shortcut even if `fast_action_decode=True`.
  Validate token/action parity and performance on the target checkpoint/device;
  extra rejected projections can make speculation slower than serial decoding.

The additional ActiveVLN-style profile is opt-in and changes floating-point
execution. `tree_decode=True` batches a budgeted grammar tree (default
`tree_budget=31`, range 1–127), including the assistant envelope, action branches
and EOS. Root partitioning and online counts use only previous model responses;
all selections remain full-vocabulary greedy. Ancestor masks and depth-based
RoPE isolate branches; only the selected path is committed. This is an
alternative to `speculative_tokens`, with genuinely batched projections.

Independent switches are `fused_ops` (rounded RMSNorm/RoPE/SwiGLU),
`split_attention` with `attention_kernel="gqa"` or `"split"`, `decode_linear`
(weight-streaming GEMM plus fused QKV/gated projections), and
`vision_kernel="hf"|"sdpa"|"fused"`. The two manual SigLIP paths stop at the
selected hidden layer. `tree_fp32_projection=True` is an alternative to
`decode_linear`, requires a tree and floating-point weights, and uses Torch
CUDA FP32-output GEMM. These switches require BF16 and CUDA graphs. Configure
before any session; changing profiles invalidates graphs and requires callers
to discard existing episode memories. Unknown graph shapes use counted eager
execution; capture is restricted to startup. Numerical equivalence of individual
operators does not establish action parity or navigation success.

Use `make_streamvln_observation(rgb_chw_uint8, instruction, uint8=True)` from
`policies.streamvln.processing` to retain byte inputs. Generic `Observation`
keeps its existing float contract. The HTTP adapter selects byte transport
when GPU preprocessing is configured.

`policy.create_batched_runtime(batch_size=2, workspace_tokens=8192)` provides
explicit multi-episode execution in `eval()` and `torch.inference_mode()`. It
requires optimized GQA attention, cached history, and either tree decoding or
serial decoding with `fast_action_decode=False`. Call
`runtime.infer(observations, request_ids, memories)`; results preserve caller
order. Warm representative calls inside `runtime.startup_capture()` before
measurement. The runtime batches vision, packs live prefill queries, skips
padded attention queries, and batches independent trees. It owns a pooled KV
scratch arena and commits private episode memories incrementally. Optional
`kv_pool_tokens` bounds the total scratch arena; rows grow and relocate within
that shared budget instead of each reserving the maximum context. Cancellation,
reordering and forks preserve input prefixes. Reconfiguration/refit, a different
CUDA stream, or concurrent calls require another runtime; capacity failures do
not truncate history. `packed_prefill=False` keeps the padded comparison path.
This API does not enable generic cross-session HTTP batching or tensor parallelism.

The paired checker in `benchmarks/streamvln-benchmark/validate_optimizations.py`
compares all options with serial decoding and the existing fast response path.
It records raw tokens, actions, effective cache lengths, log-probability
differences, graph counters and synchronized stage timings. These options do
not reduce checkpoint storage enough to fit the full BF16 model on Orin NX 16 GB.

For separated model timing, `StreamVLNPolicy.prepare_prefix` performs observation
preprocessing and the device transfer of the current input, and `encode_prepared_prefix`
performs the vision and text encoding. `StreamVLNDecoder.generate_tokens` returns the
full generation result, and `finalize_generation` then does text decoding, action
parsing, and the assembly of the next memory/trace. The usual call still goes through
`encode_prefix` and `decoder.decode`, which invoke those stages in order and preserve the
original behaviour and session semantics. The benchmark uses the same stage boundaries to
time the model separately, while the model-internal cache management of history features
still belongs to prefill.
