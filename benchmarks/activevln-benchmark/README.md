# ActiveVLN benchmark

Replay the same R2R and RxR recorded observations as `../streamvln-benchmark`:
first 48 numeric episode IDs, first instruction, every RGB frame (2,997 R2R and
3,879 RxR calls). The next frame always comes from the recording, even after a
predicted stop; generated response history is retained until the episode ends.
This is an offline latency workload, without Habitat or SR/SPL measurement.
Both datasets use `Arvil/Qwen2.5-VL-3B_rl_r2r_4000` at revision
`160987313e3e869705f42400d1b8f28177044518` and its R2R action grammar.

## Results by device

- [RTX 4090](rtx-4090/README.md): 128 R2R first frames against SGLang and vLLM
  (2026-11-02); BF16/INT8 R2R/RxR first frames at batch sizes 1/2/4/8
  (2026-11-01); batch-one closed-loop R2R/RxR navigation.
- [Jetson AGX Thor](thor/README.md): BF16/INT8 R2R/RxR first frames at batch
  sizes 1/2/4/8; batch-one closed-loop R2R navigation.
- [Orin NX](orin-nx/README.md): batch-one BF16 / calibrated INT8 closed-loop R2R
  navigation.

Closed-loop runs use the first 128 episodes of a fixed 256-episode selection and
EmbodiRun's Habitat runner; they are distinct from the 48-episode recorded replay
below.

## Recorded replay on RTX 4090

These replays were measured with the archived `vvla` package layout, before the
move to the `embodiinfer` namespace. The timings and navigation scores are evidence
for those sources, not new measurements of the namespace port.

### Selected batch-one profiles

RTX 4090, Torch 2.10.0+cu128, Transformers 4.51.3 and Triton 3.6.0. BF16, B=1,
SDPA, greedy decoding, repetition penalty 1.05, max_new_tokens=512,
max_context=128000, seed 42 and 33 continuous warmup calls; no image-resolution,
response-length or history truncation. The baseline enables none of the
optimization switches below.

| Split | Profile | Calls | E2E mean ms | Forward mean ms | E2E P50 / P95 / P99 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| R2R | Baseline | 2,997 | 724.58 | 713.20 | 810.69 / 936.96 / 1165.43 |
| R2R | `candidate-v11-tree-bf16` | 2,997 | **70.23** | **59.24** | 69.89 / 85.99 / 92.19 |
| RxR | Baseline | 3,879 | 1216.70 | 1204.85 | 917.94 / 2646.64 / 3617.81 |
| RxR | `candidate-v11-rxr-128k` | 3,879 | **104.45** | **93.45** | 89.72 / 193.01 / 225.61 |

The selected profiles combine CUDA graphs, exact-rounding fusion, split-KV
attention and root-partitioned phrase-tree verification with
`query_bucket_size=1` and BF16 tree projection; all graph fallback counters are
zero. R2R and RxR use separate prewarm/workspace configurations; the RxR profile
keeps a 128,000-token workspace with 437 text graphs. Mean prefill (including
vision) and complete decode are 28.58 / 30.63 ms for R2R and 54.20 / 39.22 ms for
RxR. Baseline responses average 17.28 (R2R) and 19.00 (RxR) tokens, with a maximum
RxR cache length of 52,989 tokens.

Tokens and actions can differ from the baseline, so the profiles are accepted on
closed-loop task success instead of output parity. Each split runs the same 48
episodes for baseline and candidate, with identical start poses, goals,
instructions, sensors, action semantics and episode limits:

| Split | Baseline successes · SPL | Selected profile successes · SPL |
| --- | ---: | ---: |
| R2R | 35/48 · 0.6828 | 35/48 · 0.6877 |
| RxR | 15/48 · 0.2442 | 17/48 · 0.2591 |

Enabling the optional FP32 tree projection lowers R2R to 31/48 (SPL 0.6155), so it
is not selected.

### Batched replay against vLLM 0.30.0 (2026-09-25)

Each configuration runs sequentially on one RTX 4090 (24,564 MiB), with four CPU
threads and 33 warmup batch calls. Both engines use the pinned checkpoint, BF16,
seed 42, greedy decoding, repetition penalty 1.05, max_new_tokens=512,
max_context=128000 and the original PIL processor with min_pixels=1024 and
max_pixels=76800. Each episode retains its own complete generated history;
numeric episode selection and remove-then-refill slot ordering match. No
quantization, vision token pruning or history truncation is used. EmbodiInfer
runs Torch 2.10.0+cu128, Transformers 4.51.3 and Triton 3.6.0; the isolated vLLM
environment runs Torch 2.13.0, Transformers 5.17.0 and Triton 3.7.1. Input audits
verify matching token IDs, pixels and image grids.

E2E starts with decoded CPU RGB and ends with parsed CPU actions, including
preprocessing, engine admission and complete generation. Complete forward covers
vision, text prefill and every decode step, including host dispatch, stopping
checks and synchronization; EmbodiInfer also snapshots private output KV inside
this interval. Neither includes disk decoding, HTTP, simulator execution,
loading, compilation, capture or warmup.

Means in milliseconds. Whole-batch columns measure one batch call. Amortized
columns divide total time by actual observations, including partially filled
tail batches; they are throughput, not individual request latency.

| Split | Batch | EmbodiInfer E2E whole batch | EmbodiInfer E2E amortized | EmbodiInfer forward whole batch | EmbodiInfer forward amortized | vLLM E2E whole batch | vLLM E2E amortized | vLLM forward whole batch | vLLM forward amortized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | 1 | 70.01 | 70.01 | 59.34 | 59.34 | 101.31 | 101.31 | 77.91 | 77.91 |
| R2R | 2 | 112.25 | 56.63 | 91.76 | 46.29 | 165.28 | 83.38 | 119.65 | 60.36 |
| R2R | 4 | 195.18 | 50.93 | 155.49 | 40.57 | 257.25 | 67.12 | 170.26 | 44.43 |
| RxR | 1 | 103.77 | 103.77 | 92.86 | 92.86 | 160.24 | 160.24 | 128.79 | 128.79 |
| RxR | 2 | 181.32 | 90.73 | 159.37 | 79.75 | 258.21 | 129.20 | 196.52 | 98.34 |
| RxR | 4 | OOM | — | — | — | 427.84 | 108.42 | 305.21 | 77.34 |

EmbodiInfer is faster in all five completed paired conditions. Actual batch counts
are 1,512/782 for R2R B=2/4 and 1,941/983 for RxR B=2/4. EmbodiInfer RxR B=4 runs
out of memory on batch 134, after 532 observations, while cloning an independent
output KV snapshot (its old input histories, scratch and new snapshots coexist to
preserve ownership); no partial latency is reported.

| Split | Batch | EmbodiInfer allocated / reserved GiB | vLLM allocated / reserved GiB |
| --- | ---: | ---: | ---: |
| R2R | 1 | 13.24 / 14.47 | 19.51 / 19.83 |
| R2R | 2 | 14.11 / 15.18 | 19.21 / 19.73 |
| R2R | 4 | 16.13 / 17.39 | 18.97 / 19.64 |
| RxR | 1 | 15.51 / 17.76 | 19.52 / 19.83 |
| RxR | 2 | 17.91 / 20.49 | 19.33 / 19.77 |
| RxR | 4 | 21.33 / 22.18 (OOM) | 19.07 / 19.76 |

vLLM reserves paged KV under `gpu_memory_utilization=0.85`, so its allocation is
not the minimum required memory.

**EmbodiInfer (B=1/2/4):** CUDA graphs for vision, text prefill, decode and
phrase-tree verification; rounded Triton fusion for RMSNorm, RoPE and SwiGLU;
split-KV attention; root-partitioned phrase-tree verification with independent
per-row acceptance/STOP decisions; packed shared KV scratch, resident-prefix
reuse, shared graph buffers and single-partition context graph executable reuse;
true tensor batching for vision, prefill and generation with independent
histories (`query_bucket_size=1`, `tree_repeat_actions=1`,
`tree_fp32_projection=false`, matching the selected B=1 profile).

**vLLM 0.30.0 (B=1/2/4):** native vision compilation and vision CUDA graphs;
FULL_AND_PIECEWISE language graphs (full graphs for uniform decode, piecewise for
prefill on this SM89/FlashAttention 2 path); optimization level 3 with
Torch/Inductor fusion and FlashAttention 2; GPU n-gram speculation with 16 draft
tokens and synchronous scheduling; paged KV, prefix caching, chunked prefill, a
4 GiB multimodal processor cache, immutable per-observation image UUIDs and GPU
image normalization; concurrent native requests with batch-scaled vision graph
shapes and admission budgets of 1,037/2,074/4,148 tokens for B=1/2/4. The
synchronous 16-draft configuration was the fastest of five B=1 configurations
tuned on a separate 523-observation R2R subset, which is excluded from the table.

The five complete EmbodiInfer runs have no invalid actions, graph fallbacks or
workspace growth, and both B=1 runs reproduce every output of the selected
profiles. The four complete vLLM B=2/4 runs pass coverage/order, history,
private-cache, input and graph checks without OOM, KV preemption, vision misses,
eager language fallback or invalid actions.

Fixed replay does not establish navigation SR/SPL. Separate downstream R2R
closed-loop runs of the batched profile complete 48 episodes at B=1/2/4 with
35/33/32 successes and SPL 0.6877/0.6465/0.6268; B=1 reproduces the selected
trajectories. RxR multi-batch and native vLLM have no validated navigation-quality
result, and the batch API remains opt-in.

## Running the benchmarks

Use the EmbodiInfer `activevln` dependency group (Transformers 4.51.3). Set the
checkpoint/data paths in `config.yaml`; `--config` accepts a separate YAML or JSON
configuration. Prepare recorded images with
`../streamvln-benchmark/prepare_data.py`.

### Batch-one replay

```bash
# From the EmbodiInfer root. Physical GPU index 1 is the second GPU.
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python benchmarks/activevln-benchmark/benchmark.py --validate-data
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python benchmarks/activevln-benchmark/benchmark.py
```

Each call records two timing scopes:

- `latency_ms`: decoded CPU RGB through observation construction, image/prompt
  processing, H2D, vision/text inference, action parsing and CPU action chunk.
- `model_timing_ms.pure_inference_ms`: synchronized wall time from device-ready
  inputs through vision, prefill and the complete autoregressive loop, including
  model-side KV management, sampling and stop control. CUDA event intervals are
  also reported for prefill, decode and their total; they include stream idle
  time.

Loading, file reading/JPEG decoding, warmup, report checks and hashing are outside
the timed calls. Final JSON reports are written atomically after a whole dataset
finishes and include every call, token/action evidence, selection/configuration,
source hashes, versions, memory peaks and percentile/throughput metrics.

### Tensor batches

`benchmark_batch.py` measures true tensor batches with independent episode
histories. Each call packs all images into one vision invocation, runs padded
text prefill and greedy decoding across the batch, then returns a separate
`ActiveVLNMemory` per observation. Different KV lengths, rotary positions,
repetition histories and EOS/STOP decisions remain independent. Finished episode
slots are refilled in numeric order; the report records both the configured batch
size and actual observations per batch.

The policy API is `policy.create_batched_runtime(...)`, followed by
`runtime.prepare(observations, memories)`, `runtime.prefill(prepared)` and
`runtime.generate(prefix)` inside `torch.inference_mode()`. Parse each returned
generation with `policy.decoder.finalize_generation(generation)`. Inputs retain
ownership of committed histories; replace them with returned memories only after
the complete batch succeeds. A prefix belongs to one runtime and is invalidated
by its next prefill or generation attempt. Generic engine/HTTP session batching,
sampling and training keep their existing contracts and are not exposed by this
policy-local inference API.

For batch one, optional `resident_kv: true` commits into a fixed `kv_pool_tokens`
arena without copying history. Continue only the latest returned memory and call
`runtime.reset_resident_kv()` between sessions; both harnesses reset automatically.
Memories borrow the arena until reset: use `memory.fork()` for private snapshots.
Insufficient capacity rejects the turn before writes. The option defaults to false.

After editing checkpoint/data paths in `config.yaml`, generate the feature-matched
B=1/2/4 configurations and run each in its own process:

```bash
python - <<'PY'
import copy, json
from pathlib import Path
import yaml
root = Path("benchmarks/activevln-benchmark")
base = yaml.safe_load((root / "config.yaml").read_text())
out = root / "runs/batch-tree-shared-context"
out.mkdir(parents=True, exist_ok=True)
for split in ("R2R", "RxR"):
    for batch in (1, 2, 4):
        config = copy.deepcopy(base)
        config["datasets"] = [s for s in config["datasets"] if s["name"] == split]
        config["output_dir"] = str(out.resolve())
        config.update(batch_size=batch, cuda_graph=True, fused_ops=True,
                      split_attention=True, tree_decode=True, query_bucket_size=1,
                      tree_fp32_projection=False, tree_repeat_actions=1,
                      kv_pool_tokens=153600 if split == "RxR" and batch == 4 else 128000,
                      graph_workspace_tokens=65536 if split == "R2R" else 128000)
        config["prewarm_context_buckets"] = [2**n for n in range(9, 17)]
        if split == "RxR":
            config["prewarm_context_buckets"].append(128000)
        (out / f"{split.lower()}-b{batch}-config.json").write_text(json.dumps(config, indent=2))
PY

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7 \
python benchmarks/activevln-benchmark/benchmark_batch.py \
  --config benchmarks/activevln-benchmark/runs/batch-tree-shared-context/r2r-b2-config.json \
  --output benchmarks/activevln-benchmark/runs/batch-tree-shared-context/r2r-b2.json
```

`kv_pool_tokens` sets the initial total scratch budget across all rows and requires
split-KV on CUDA. Independently addressed segments grow with the histories, and
graph input/output buffers are shared across context buckets of the same query
shape. RxR B=4 reserves 153,600 tokens because its aggregate histories need
143,360 tokens. The pool changes scratch allocation, not the per-row
context/response limits or attention semantics. If a history exceeds the
workspace, the runtime grows it and drops text graphs that point to the old
storage, and later calls use counted eager execution. Reports retain startup and
measured memory peaks, graph coverage and, on OOM, the failure phase and in-flight
histories; a partial or OOM run never becomes a complete-run result.

### vLLM

`benchmark_vllm_modern.py` (B=1) and `benchmark_vllm_batch.py` (B=1/2/4) require
vLLM 0.30.0 in an isolated interpreter; the validated stack uses Torch 2.13.0,
Transformers 5.17.0, NumPy 2.3.5 and Pillow 12.3.0. Preserve the original PIL
processor with request-level `mm_processor_kwargs.use_fast=False` (the native
default overrides it when set only in engine configuration), and keep
`mm_device_do_normalize=True`.

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD" \
  /path/to/vllm-0.30.0/bin/python benchmarks/activevln-benchmark/benchmark_vllm_modern.py \
  --config benchmarks/activevln-benchmark/runs/batch-tree-shared-context/r2r-b1-config.json \
  --output benchmarks/activevln-benchmark/runs/vllm-0.30.0/r2r-full.json \
  --speculation ngram_gpu --speculative-tokens 16 --no-async-scheduling

CUDA_VISIBLE_DEVICES=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD" \
  /path/to/vllm-0.30.0/bin/python benchmarks/activevln-benchmark/benchmark_vllm_batch.py \
  --config benchmarks/activevln-benchmark/runs/batch-tree-shared-context/r2r-b2-config.json \
  --output benchmarks/activevln-benchmark/runs/vllm-batch-0.30.0/r2r-b2-full.json
```

The modern entrypoint requests optimization level 3, vision compilation and CUDA
graphs, full/piecewise language graphs, native FlashAttention, prefix caching,
UUID-based multimodal caching and selectable synchronous/asynchronous scheduling,
and records the resolved settings, vision graph hits/misses, language graph modes
and verified/accepted draft tokens. `--speculation none`, `ngram` and `suffix`
select other native configurations; suffix requires `arctic-inference`. The batch
entrypoint submits the current observations as concurrent native requests with
batch-scaled vision graph budgets and admission capacity; per-episode salts
prevent cross-episode prefix reuse, and the slot order matches
`benchmark_batch.py`. Run each split/batch in a fresh, sequential process.
`--max-steps-per-episode` produces an explicitly partial smoke test whose latency
must not replace the complete workload.

`benchmark_vllm.py` runs the same workload through the official ActiveVLN
environment's vLLM 0.8.5.post1 (Torch 2.6.0/CUDA 12.4); its timing and report
helpers are shared by the modern entrypoints. That version disables native n-gram
proposals for non-default repetition penalties; keep the required 1.05 penalty.

### Comparing reports

```bash
python benchmarks/activevln-benchmark/compare.py runs/baseline/activevln-r2r.json runs/optimized/activevln-r2r.json --output runs/r2r-comparison.json
python benchmarks/activevln-benchmark/compare.py runs/baseline/activevln-rxr.json runs/optimized/activevln-rxr.json --output runs/rxr-comparison.json
```

The comparison rejects incomplete workloads or changed generation conditions. By
default it requires identical token IDs, action chunk hashes, masks, stop reasons
and cache lengths. `--accuracy-contract task-success` reports output differences
without rejecting the candidate; because replay cannot establish task accuracy,
it reports `admission_status: pending_navigation_evaluation` and exits 2 until a
separate closed-loop evaluation is assessed.

### Closed-loop service

`serve_batch.py` exposes the tensor runtime through the existing versioned HTTP API
for EmbodiRun's closed-loop runner. It reuses the generic bounded batch scheduler,
keeps one committed memory per session, sorts requests by the controller's stable
slot IDs and commits only after all outputs are finalized; a model failure
invalidates the process. Pass the measured source as `--source`, the unchanged
tensor replay `--config` and a `--ready-file`. The default 1,000 ms coalescing
bound lets a synchronous controller round arrive over HTTP; queueing, PNG decode
and simulator time are excluded from model E2E. `tensor-batches.jsonl` records
actual occupancy, memory, graph coverage and exceptions.

## Optimization switches

All switches default to off; select a combination only after validating it under
the accuracy contract of the task.

- `cuda_graph`, `fused_ops`, `split_attention`, `tree_decode`: the latter three
  require `cuda_graph`. Graph startup captures the shapes visited during warmup,
  freezes capture before measurement and reports replay/fallback counts. Training
  and differentiable recompute continue through the eager implementation.
- `query_bucket_size`: text padding; 1 keeps the exact query length.
- `prewarm_context_buckets` (for example `[512, 1024, …, 65536]`): also captures
  initial/subsequent prompt shapes from the first two selected frames of each
  episode and single-token/tree decoding across the requested context buckets.
  This adds startup time and memory, outside timed calls; unseen shapes use the
  counted fallback.
- `graph_workspace_tokens`: bounds the separately allocated graph scratch cache
  (default: the model context limit) without changing `max_context` or the public
  recurrent memory. Requested prewarm buckets must fit it.
- `tree_decode`: greedy phrase-tree verification from the public prompt. Each
  accepted token must match the full-vocabulary model choice with the original
  repetition penalty; uncovered continuations use normal decoding. Tree query
  shapes can change floating-point accumulation. `tree_repeat_actions` (default
  1, up to 3) proposes repeated phrases in one verification pass.
- `split_attention`: partitioned KV attention with compact GQA storage, in-kernel
  causal/tree visibility and FP32 accumulation of three BF16 probability
  components; the changed reduction order requires validation.
- `tree_fp32_projection`: keeps tree projection outputs and bias addition in FP32
  before casting back to BF16. It requires `tree_decode` and Torch CUDA
  `mm(out_dtype=torch.float32)`.

Even graph-only execution can change BF16 results through SDPA key padding (for
example, 521 actual keys versus padding to 1024), so operator tests or a short
smoke run do not establish full-workload action parity. Fused RMSNorm keeps the
reference Torch mean reduction and fused SwiGLU uses libdevice arithmetic; their
GPU operator tests require exact output equality.
