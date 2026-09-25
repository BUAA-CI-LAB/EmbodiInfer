# ActiveVLN benchmark

Replay the same R2R and RxR recorded observations as `../streamvln-benchmark`:
first 48 numeric episode IDs, first instruction, every RGB frame (2,997 R2R and
3,879 RxR calls). The next frame always comes from the recording, even after a
predicted stop; generated response history is retained until the episode ends.
This is an offline latency workload, without Habitat or SR/SPL measurement.
Both datasets use `Arvil/Qwen2.5-VL-3B_rl_r2r_4000` at revision
`160987313e3e869705f42400d1b8f28177044518` and its R2R action grammar.

## Tensor batches

`benchmark_batch.py` measures true tensor batches with independent episode
histories. Each call packs all images into one vision invocation, runs padded
text prefill and greedy decoding across the batch, then returns a separate
`ActiveVLNMemory` per observation. Different KV lengths, rotary positions,
repetition histories and EOS/STOP decisions remain independent. Finished episode
slots are refilled in numeric order. The last batches can have lower occupancy;
the report records both configured batch size and actual observations per batch.

The policy API is `policy.create_batched_runtime(...)`, followed by
`runtime.prepare(observations, memories)`, `runtime.prefill(prepared)` and
`runtime.generate(prefix)` inside `torch.inference_mode()`. Parse each returned
generation with `policy.decoder.finalize_generation(generation)`. Inputs retain
ownership of committed histories; replace them with returned memories only after
the complete batch succeeds. A prefix belongs to one runtime and is invalidated
by its next prefill or generation attempt. Generic engine/HTTP session batching,
sampling and training keep their existing contracts and are not exposed by this
policy-local inference API.

CUDA Graph, rounded fusion, split-KV attention and root-partitioned action trees
are supported. Each row can verify a different tree or take a singleton fallback
in the same forward. Only accepted nodes enter that row's history. Whole-batch latency is the wait for all rows,
while amortized milliseconds per observation describe throughput and are not
individual request latency. The complete forward interval includes vision,
prefill, all decode steps and private KV snapshots. E2E also includes CPU
preprocessing and action parsing, excluding disk decoding and HTTP/simulator time.

After editing checkpoint/data paths in `config.yaml`, generate explicit configs
for a feature-matched B=1/2/4 comparison:

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

# Run each configuration sequentially in its own process on physical GPU 1.
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7 \
python benchmarks/activevln-benchmark/benchmark_batch.py \
  --config benchmarks/activevln-benchmark/runs/batch-tree-shared-context/r2r-b2-config.json \
  --output benchmarks/activevln-benchmark/runs/batch-tree-shared-context/r2r-b2.json
```

The optional `kv_pool_tokens` sets the initial total scratch budget across all
rows. Independently addressed segments grow as histories grow; graph input/output
buffers are shared across context buckets of the same query shape. Single-
partition split-KV contexts also share one graph executable: all sufficient key
bounds execute the same dynamic extent and reduction order. Multi-partition
graphs remain distinct. Counters report both logical shape coverage and actual
executable count, and GPU tests require bit-exact forward/KV parity. A
128,000-token pool applies except RxR B=4, which reserves 153,600 tokens for
its aggregate histories. Historical frame-length preflight needs 143,360 tokens
for that segment plan; reserving headroom keeps captures valid through the full
replay. This changes scratch allocation, not the per-row context/response limits
or attention semantics. A pool requires split-KV on
CUDA. Initial workspace bounds scratch allocation, not history: `max_context=128000`
and `max_new_tokens=512` remain unchanged. If history exceeds workspace, the
runtime grows it and drops text graphs that point to the old storage; subsequent
calls use counted eager execution. Sufficient GPU memory is still required.
Reports retain startup and measured memory peaks, graph coverage, all completed
calls, and the exact failure phase/in-flight histories on OOM. A partial or OOM
run never becomes a complete-run result. Fixed replay does not establish
navigation success, and prior B=1 SR/SPL does not certify this new batch profile.

### Feature-matched EmbodiInfer and vLLM comparison

The September 25, 2026 comparison runs each configuration sequentially on
physical GPU 1 (RTX 4090, 24,564 MiB), using four CPU threads and 33 warmup
batch calls. Both engines use the pinned checkpoint above, BF16, seed 42,
greedy decoding, repetition penalty 1.05, max_new_tokens=512, max_context=128000,
and the original PIL processor with min_pixels=1024 and max_pixels=76800.
Each episode retains its own complete generated history; numeric episode
selection and remove-then-refill slot ordering match. No quantization, vision
token pruning or history truncation is used.

EmbodiInfer uses Torch 2.10.0+cu128, Transformers 4.51.3 and Triton 3.6.0.
The isolated vLLM 0.30.0 environment uses Torch 2.13.0, Transformers 5.17.0
and Triton 3.7.1. These are measured deployment stacks with different dependency
versions. Input audits verify matching token IDs, pixels and image grids.

**Timing contract.** E2E starts with decoded CPU RGB and ends with parsed CPU
actions, including preprocessing, engine admission and complete generation.
Complete forward covers vision, text prefill and every decode step, including
host dispatch, stopping checks and synchronization; EmbodiInfer also snapshots
private output KV inside this interval. It excludes preprocessing/action parsing
and is not a sum of GPU kernel durations. Both timings exclude disk decoding,
HTTP, simulator execution, loading, compilation, capture and warmup.

All values below are means in milliseconds. Whole-batch columns measure one
batch call. Amortized columns divide total time by actual observations,
including partially filled tail batches; they are not individual request
response latency. R2R completes 2,997 observations and RxR completes 3,879,
except the explicitly failed EmbodiInfer RxR B=4 run.

| Split | Batch | EmbodiInfer E2E whole batch | EmbodiInfer E2E amortized | EmbodiInfer forward whole batch | EmbodiInfer forward amortized | vLLM E2E whole batch | vLLM E2E amortized | vLLM forward whole batch | vLLM forward amortized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R | 1 | 70.01 | 70.01 | 59.34 | 59.34 | 101.31 | 101.31 | 77.91 | 77.91 |
| R2R | 2 | 112.25 | 56.63 | 91.76 | 46.29 | 165.28 | 83.38 | 119.65 | 60.36 |
| R2R | 4 | 195.18 | 50.93 | 155.49 | 40.57 | 257.25 | 67.12 | 170.26 | 44.43 |
| RxR | 1 | 103.77 | 103.77 | 92.86 | 92.86 | 160.24 | 160.24 | 128.79 | 128.79 |
| RxR | 2 | 181.32 | 90.73 | 159.37 | 79.75 | 258.21 | 129.20 | 196.52 | 98.34 |
| RxR | 4 | OOM | — | — | — | 427.84 | 108.42 | 305.21 | 77.34 |

EmbodiInfer is faster in all five completed paired conditions. At R2R B=4,
40.57/44.43 ms describes amortized forward, not 40 ms E2E response latency.
Actual batch counts are 1,512/782 for R2R B=2/4 and 1,941/983 for RxR B=2/4.
The full reports retain percentiles and per-observation outputs.

**EmbodiInfer enabled optimizations (B=1/2/4):**

- CUDA graphs for vision, text prefill, decode and phrase-tree verification.
- Rounded Triton fusion for RMSNorm, RoPE and SwiGLU, plus split-KV attention.
- Root-partitioned action phrase-tree verification with independent per-row
  acceptance/STOP decisions; only accepted tokens enter the history.
- Packed shared KV scratch, resident-prefix reuse, shared graph buffers and
  single-partition context graph executable reuse.
- True tensor batching for vision, prefill and generation, with independent
  histories. query_bucket_size=1, tree_repeat_actions=1 and
  tree_fp32_projection=false match the selected B=1 profile.

**vLLM 0.30.0 enabled optimizations (B=1/2/4):**

- Native vision compilation and vision CUDA graphs; language graphs use
  FULL_AND_PIECEWISE. On this SM89/FlashAttention 2 path, uniform decode uses
  full graphs and prefill uses piecewise graphs.
- Optimization level 3, Torch/Inductor automatic kernel fusion and FlashAttention
  2 for vision/text. This does not imply every specialized fusion flag is active.
- GPU n-gram speculation with 16 draft tokens and synchronous scheduling.
- Paged KV, prefix caching and chunked prefill; 4 GiB multimodal processor cache,
  immutable per-observation image UUIDs and GPU image normalization.
- Concurrent native requests with batch-scaled vision graph shapes and admission
  budgets: 1,037/2,074/4,148 tokens for B=1/2/4. Partial batches have matching
  graph coverage; uniform decode shapes include 17/34/51/68 as needed.

The synchronous 16-draft configuration was selected from five B=1 configurations
on 523 saved R2R observations: 89.61 ms E2E and 68.66 ms forward, versus
107.46/85.97 ms with asynchronous scheduling. This tuning subset is excluded
from the full-workload table. Its independent repeat reproduces all tokens and
actions, with mean timing changes of 0.49% E2E and 0.09% forward.

Measured PyTorch allocator peaks are below. vLLM reserves paged KV under
`gpu_memory_utilization=0.85`, so its allocation is not minimum required memory
and need not increase monotonically with batch size. EmbodiInfer's failed
RxR B=4 memory peak covers only the partial run.

| Split | Batch | EmbodiInfer allocated GiB | EmbodiInfer reserved GiB | vLLM allocated GiB | vLLM reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| R2R | 1 | 13.24 | 14.47 | 19.51 | 19.83 |
| R2R | 2 | 14.11 | 15.18 | 19.21 | 19.73 |
| R2R | 4 | 16.13 | 17.39 | 18.97 | 19.64 |
| RxR | 1 | 15.51 | 17.76 | 19.52 | 19.83 |
| RxR | 2 | 17.91 | 20.49 | 19.33 | 19.77 |
| RxR | 4 | 21.33 (OOM) | 22.18 (OOM) | 19.07 | 19.76 |

EmbodiInfer RxR B=4 fails on batch 134 after 532 observations while cloning an
independent output KV snapshot: an 860 MiB allocation exceeds 819.81 MiB free.
Its old input histories, scratch and new snapshots coexist to preserve ownership.
No partial latency is admitted. vLLM RxR B=4 completes all 3,879 observations,
including a 53,008-token individual prompt and 137,336-token aggregate history.

**Validation and accuracy limits.** The five complete EmbodiInfer runs have no
invalid actions, graph fallbacks or workspace growth. Both B=1 runs reproduce
all outputs of the original selected profiles. The batch runtime passes 95
CPU/GPU tests, including bit-exact graph-alias forward/KV checks at B=1/2/4.
The submission CPU rerun passes 108 targeted tests (30 hardware-dependent skips)
and 555 broader tests, with 54 skips, 51 deselections and five pre-existing
namespace compatibility failures reproduced on the upstream code. Benchmark
test imports are isolated so other entrypoints with the same module name do not
interfere with this suite.
The modern vLLM harness passes 22 tests; five old-version-only frontend tests
skip. Its four complete B=2/4 runs pass independent coverage/order, history,
private-cache, input and graph checks for 13,752 observations and 384 native
input audits, without OOM, KV preemption, vision misses, eager language fallback
or invalid actions. A B=1 harness probe reproduces 96/96 token sequences.

Fixed replay does not establish navigation SR/SPL. Separate downstream R2R
closed-loop runs complete 48 episodes at B=1/2/4 with 35/33/32 successes and SPL
0.6877/0.6465/0.6268. B=1 reproduces the original selected trajectories; B=4
loses three successes and fails the provisional two-episode tolerance.
RxR multi-batch and native vLLM have no complete validated navigation-quality
result. The batch API remains opt-in.

Unpacked reports, configurations, dependency freezes, test logs, assessments and
frozen sources remain ignored locally and are included in the complete test
snapshot below:

- `runs/batch-tree-shared-context/assessment.json`: six EmbodiInfer reports;
  frozen Python source SHA256
  `62f8b4dc2bb374c3dbc1610e61e319091eb7b57116a635900bb1fb0dcf51a88f`.
- `runs/batch-tree-validation/summary.json`: runtime test results and known
  upstream failures.
- `runs/publish-validation/batch-20260925/`: submission CPU and isolated vLLM
  test logs, including the benchmark import-isolation regression.
- `runs/vllm-0.30.0/`: native B=1 full reports, tuning and input audits.
- `runs/vllm-batch-0.30.0/assessment.json`: four full native batch reports;
  batch entrypoint SHA256
  `0400dae6eda41a96b8c1f8652e7e984d11c7048aed447eb3a9a3c258147cc6da`,
  shared modern runtime SHA256
  `0343775fef719ab26be66f6e5bdaff90f6d2f6e5892a2518d03f4a2fbab8293e`.

### Complete test snapshot

[20260925232656.tar.gz](../snapshots/20260925232656.tar.gz)
([SHA256](../snapshots/20260925232656.tar.gz.sha256), 69,043,566 bytes / 65.85 MiB)
archives the ActiveVLN EmbodiInfer and native vLLM comparison. It includes
EmbodiInfer commit `1f23aa70760cb0a94683c6d4baa518deb0c7e803`, exact measured
source snapshots, vLLM 0.8.5.post1/0.30.0 sources, dependency lists, configurations,
per-observation reports, logs, navigation evidence and historical snapshots.
Model weights, raw RGB/scenes, virtual environments, caches and native binaries
are excluded; their required versions and locations are recorded inside.
This designated archive and checksum are published with the branch.

The archive retains all 12 final split/batch/engine reports: 11 complete
fixed replays and the EmbodiInfer RxR B=4 OOM after 532 observations. Failed
navigation runs and known test failures are preserved; the snapshot does not
turn fixed replay into a navigation success-rate claim. The enclosed
`comparison.csv` separates E2E/complete forward and whole-batch/amortized costs.

SHA256: `00bffe7f9d30642070a37f7844f07d9a0cf27a2dfe33e188b01a4c972a25520c`.
All 9,054 file checksums and 15 report/source checks passed before publication.
Verify and extract from the repository root:

```bash
cd benchmarks/snapshots
sha256sum -c 20260925232656.tar.gz.sha256
tar -xzf 20260925232656.tar.gz
cd activevln-test-snapshot-20260925232656
sha256sum -c SHA256SUMS
```

Start with the archive's `README.md`, `manifest.json` and `verification.json`.
Use its measured sources and matching configurations for reproduction; the
source commit inside remains fixed even as this branch's documentation changes.

### Historical baselines

The earlier tree-disabled EmbodiInfer batch experiment is retained under
`runs/batch-full/`; it does not provide a feature-matched comparison to the
selected B=1 profile. Native vLLM 0.8.5.post1 uses eager vision and piecewise
language graphs. Its optional duplicate-placeholder-rule correction preserves
input/model semantics and reproduces every R2R output while reducing mean E2E
from 420.17 to 293.19 ms; complete forward remains 215.75 ms. Corrected RxR
completes at 383.46 ms E2E and 257.34 ms forward. The original RxR run is partial.
Details, original/corrected reports and matching source snapshots remain under
`runs/vllm-0.8.5/`. These older results do not bound modern vLLM performance.
The shared `benchmark_vllm.py` timing/report helpers are also dependencies of
the modern benchmark entrypoints.

## Single-row reference and phrase-tree profiles

The current optimization task permits different tokens and actions provided
closed-loop navigation success stays close to the baseline. Evaluate baseline
and candidate independently on the same 48 episodes per dataset, with identical
start poses, goals, instructions, sensors, action semantics and episode limits.
Report success counts/rates separately for R2R and RxR, including which episodes
change outcome; also retain SPL and final goal distances. A provisional tolerance
is at most two fewer successes per 48 episodes (4.17 percentage points), subject
to the user's chosen threshold. Fixed-trajectory replay, valid action parsing
and output agreement do not measure navigation success. The closed-loop runner
and task metrics belong to EmbodiRun, outside the inference package.

The baseline uses B=1, BF16, SDPA, greedy decoding, repetition penalty 1.05,
max_new_tokens=512, seed=42 and 33 continuous warmup frames. No image-resolution,
response-length or history truncation optimization is applied. `max_context`
uses the checkpoint's 128000 positional limit to accommodate complete episodes.

Two timing scopes are recorded per call:

- `latency_ms`: decoded CPU RGB through observation construction, image/prompt
  processing, H2D, vision/text inference, action parsing and CPU action chunk.
- `model_timing_ms.pure_inference_ms`: synchronized wall time from device-ready
  inputs through vision, prefill and the complete autoregressive forward loop,
  including model-side KV management, sampling and stop control. CUDA event
  intervals are also reported for prefill, decode and their total. These elapsed
  intervals include stream idle time, not just a sum of kernels.

Loading, file reading/JPEG decoding, warmup, report checks and hashing are outside
the timed calls. Final JSON reports are written atomically after a whole dataset
finishes; they include every call, token/action evidence, selection/configuration,
source hashes, versions, memory peaks and percentile/throughput metrics.

Use the EmbodiInfer `activevln` dependency group (Transformers 4.51.3). Set the
checkpoint/data paths in `config.yaml`; `--config` accepts a separate YAML or JSON
configuration. Prepare recorded images with the existing
`../streamvln-benchmark/prepare_data.py` script.

```bash
# From the EmbodiInfer root. Physical GPU index 1 is the second GPU.
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python benchmarks/activevln-benchmark/benchmark.py --validate-data
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python benchmarks/activevln-benchmark/benchmark.py
```

The complete R2R and RxR baselines were measured on September 24, 2026 on physical
GPU 1 (RTX 4090), Torch 2.10.0+cu128, Transformers 4.51.3 and Triton 3.6.0. It
contains all 48 selected R2R episodes and 2,997 frames; the selection SHA256 matches
StreamVLN (`f619f18b95276938068d2f5c3bb86a7e837d17c61024b7935f5d7eb6ccf7246b`).

These measurements used the archived `vvla` package layout. This checkout ports
the optimization and benchmark to the current `embodiinfer` namespace; its source
hashes change accordingly. The reported timings and navigation scores remain
evidence for the archived sources, not new measurements of the namespace port.
For exact source reproduction, use both the frozen inference sources and the
serving script included in the archive.

| Run | Calls | E2E mean ms | Forward mean ms | Prefill mean ms | Decode mean ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| R2R baseline | 2,997 | 724.58 | 713.20 | 110.73 | 602.43 |
| RxR baseline | 3,879 | 1216.70 | 1204.85 | 245.35 | 959.45 |

R2R E2E P50/P95/P99 are 810.69/936.96/1165.43 ms. The mean generated response is
17.28 tokens, with a maximum of 21. The raw report is
`runs/baseline/activevln-r2r.json`, SHA256 `7aea40e5c0a5d7dd6d57f4f568d5615bc76d0b1c9f9e750942c385434e43a013`.
RxR includes all 48 selected episodes. Its E2E P50/P95/P99 are
917.94/2646.64/3617.81 ms; responses average 19.00 tokens (maximum 21), with a
maximum cache length of 52,989 tokens. All 3,879 responses parse as valid actions.
The raw report is `runs/baseline/activevln-rxr.json`, SHA256
`1e24db30f73870894a6b04af884dbcdbab14776659fbd938087d0fec6b4025f4`;
its selection SHA256 is `5478777de23c0065d77d7a84d3690779a5fbf2c4996bb54607ea825c26df837f`.
The corrected profiles below preserve task success on the selected episodes;
neither meets the 40 ms latency target.

The first candidate combining graphs, rounded fusion, split-KV attention and
phrase-tree verification completed all 2,997 R2R calls at 74.82 ms mean E2E
(63.78 ms forward). There are 873 frames with different tokens/actions, and
mean latency exceeds 40 ms. The differences failed the original exact-output
contract; under the current task-success contract they remain diagnostic.
The subsequent complete paired R2R closed-loop test also rejects this version:
baseline success is 35/48 (72.92%), versus 29/48 (60.42%) for the same frozen v9
candidate, a 12.50 percentage-point decrease. SPL falls from 0.6828 to 0.5727.
Seven baseline successes become failures and one failure becomes a success.
All 48 paired initial RGB images, poses and initial goal distances match exactly.
Closed-loop model E2E averages 77.69 ms for v9; this is a different observation
workload from the 74.82 ms fixed replay and does not establish accuracy parity.
The paired runner lives in downstream EmbodiRun's `benchmarks/activevln-navigation`.
Its unfinished RxR latency replay was
stopped under the earlier contract to isolate numerical changes. Three-frame smoke parity had
passed, demonstrating why it cannot substitute for complete comparison.
The R2R report is `runs/candidate-v9-full/activevln-r2r.json`, SHA256
`0002475a795295fbdaf1dc618f7e19cbc173c55b143fbfe77b9e29028abd44bd`.

A later frozen v11 profile uses exact-rounding fusion, root-partitioned trees
and `query_bucket_size=1`. Its complete paired R2R navigation check recovers
35/48 successes (72.92%, SPL 0.6877) with BF16 tree projection, at 70.42 ms mean
closed-loop model E2E and 59.07 ms complete forward over 1,008 calls. Forward
includes vision encoding, prefill and the entire action-sequence decode with
CUDA synchronization; E2E additionally includes CPU observation preprocessing
and CPU action postprocessing, excluding HTTP transport and simulator execution.
All 48 initial observations/poses and
primitive execution traces were audited. Enabling the optional FP32 tree
projection gives 31/48 successes (64.58%, SPL 0.6155), so it is not selected.
These are independent closed-loop trajectories. V11's separate 96-call recorded
probe only establishes frozen-source/configuration provenance; it is not a
complete replay latency result. The same R2R-only prewarm/65,536-token workspace
profile subsequently exhausted memory in RxR episode 20 after 160 calls and
495 primitive steps (66,385 cached tokens). A fresh-process allocator retry
reproduced the same calls and OOM. Its RxR quality result is incomplete. A
separate `candidate-v11-full-context` configuration covers both splits' query
shapes and the unchanged 128,000-token context limit. It reproduces all 1,008
R2R calls and the 35/48 success count at 70.90 ms E2E/59.24 ms forward. Its 675
resident text graphs and larger workspace nevertheless exhaust memory in RxR
episode 20 after 127 calls/419 steps, despite zero fallbacks. The separately
recorded RxR-only prewarm profile below completes validation; the interrupted
runs remain diagnostic and are not scored as navigation failures.
Neither a lossless claim nor the 40 ms target is established.

The complete fixed R2R replay of the R2R-only `candidate-v11-tree-bf16` profile
subsequently covers all 2,997 frames at **70.23 ms mean E2E / 59.24 ms complete
forward**. E2E P50/P95/P99 are 69.89/85.99/92.19 ms; mean prefill (including
vision) and complete decode are 28.58/30.63 ms. All graph fallback counters are
zero. The inference source/configuration match the separately audited 35/48
navigation result; only recorded-frame selection and output directory change.
Report: `runs/candidate-v11-tree-bf16-full-r2r/activevln-r2r.json`, SHA256
`477f96f461b2c418cf696fe2521204baa4f2ca1f7416af4b4252c8a58b250fb9`.
This supports R2R task-success preservation on the selected episodes and is
distinct from the earlier 70.42 ms closed-loop mean. RxR is evaluated separately
below; the 40 ms target is not met. Downstream EmbodiRun's `r2r-final-assessment.json`
joins the complete replay with navigation evidence; the generic replay-only
comparator still reports that an external navigation assessment is required.

The RxR-only `candidate-v11-rxr-128k` profile retains the same frozen v11 source
and BF16 tree projection, with a 128,000-token workspace and 437 text graphs.
It completes all 48 paired native episodes with 17/48 successes (35.42%), versus
15/48 baseline (31.25%); SPL is 0.2591 versus 0.2442. All paired starts and
execution traces pass the downstream audit. Its closed-loop means are
98.74 ms E2E / 86.94 ms forward across 2,186 calls.

The complete fixed RxR replay then measures all 3,879 frames at **104.45 ms mean
E2E / 93.45 ms complete forward**. E2E P50/P95/P99 are 89.72/193.01/225.61 ms;
mean prefill (including vision) and complete decode are 54.20/39.22 ms. All graph
fallback counters are zero. The source/configuration match that profile's
navigation run, except recorded-frame selection/output location; warmup is
33 calls, with all capture outside measurement. Report:
`runs/candidate-v11-rxr-128k-full/activevln-rxr.json`, SHA256
`ea532db33c0bc73b587316469b7b7614680823a9c79f3a27dcfd79eb0659f346`.
R2R and RxR use explicitly separate prewarm/workspace configurations, not a
single 70 ms claim. Downstream `final-assessment.json` joins both complete
replays with their independent navigation evidence. Both selected-episode
success counts are at least the baseline; neither meets the 40 ms target.

The experiment archive is [20260925014239.tar.gz](../snapshots/20260925014239.tar.gz),
with [SHA256](../snapshots/20260925014239.tar.gz.sha256). It contains the current
benchmark/inference code and navigation runner, the exact baseline/v11 source
snapshots used for measurement, dependency metadata, actual configurations,
complete replay reports, and the original navigation evidence archive referenced
by `final-assessment.json`. That evidence also retains rejected candidates for
audit; the final assessment identifies the accepted profiles. Checkpoints, raw
images/scenes, virtual environments, caches and compiled libraries are excluded.
Reports and the archive remain untracked.

After extraction, `EmbodiRun/` preserves the repository layout; `frozen-sources/`
holds the two measured inference snapshots. `manifest.json` records their hashes
and the runtime conditions; run `sha256sum -c SHA256SUMS` from the archive root
to verify its files. Use the frozen source's `benchmark.py` and `PYTHONPATH` when
reproducing recorded timings. The archived `serve.py` stays outside those frozen
directories so their recorded source hashes remain unchanged. Saved configurations
retain the original host paths; update checkpoint/data/output paths for another
host. The included lockfiles describe development dependencies, while the report
environment and manifest record the measured Torch 2.10.0+cu128 environment.
For RxR v11, retain
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7`.
Native selections are included separately for the navigation runner's `--root`;
scene assets and a separate Habitat-Sim 0.2.4 / Python 3.9 environment are required.

Compare full baseline and candidate reports with:

```bash
python benchmarks/activevln-benchmark/compare.py runs/baseline/activevln-r2r.json runs/optimized/activevln-r2r.json --output runs/r2r-comparison.json
python benchmarks/activevln-benchmark/compare.py runs/baseline/activevln-rxr.json runs/optimized/activevln-rxr.json --output runs/rxr-comparison.json
```

The comparison rejects incomplete workloads or changed generation conditions. It
defaults to the original exact-output contract, requiring identical token IDs,
action chunk hashes, masks, stop reasons and cache lengths. The 40 ms target is evaluated on mean E2E latency over every selected
frame, independently for R2R and RxR; P50/P95/P99 remain available in the reports.
A nonzero exit status means behavior parity or the latency target was not met.

For the current task use `--accuracy-contract task-success`. Output differences
are reported without rejecting the candidate for those differences. Replay
cannot establish task accuracy, so `admitted` is `null`, `admission_status` is
`pending_navigation_evaluation`, and the command exits 2 until a separate
closed-loop evaluation is assessed. This mode still rejects incomplete replay
workloads and changed measurement conditions.

`serve_batch.py` exposes the tensor runtime through the existing versioned HTTP
API for downstream closed-loop evaluation. It reuses the generic bounded batch
scheduler, keeps one committed memory per session, sorts requests by the
benchmark controller's stable slot IDs, and commits only after all outputs have
been finalized. A model failure invalidates the process; it does not retry with
partially written graph workspace. Session ordering/reset remain owned by the
existing service. The simulator and success metrics belong to EmbodiRun.

Run this adapter with `PYTHONPATH` pointing to the measured immutable source and
pass that directory as `--source`, plus the unchanged tensor replay `--config`
and a `--ready-file`. The adapter may be outside that snapshot, preserving the
original runtime/benchmark hash; its own SHA256 is recorded separately. The
default 1,000 ms coalescing bound allows a synchronous controller round to arrive
over HTTP. Queueing, PNG decode and simulator time are excluded from model E2E.
`tensor-batches.jsonl` records actual occupancy, memory, graph coverage and
exceptions; repeat the full selected closed-loop episodes before admitting SR.

`benchmark_vllm.py` runs the same decoded-CPU-RGB workload through the official
ActiveVLN environment's vLLM 0.8.5.post1 and Transformers 4.51.3. Install these
in an isolated interpreter; do not replace the EmbodiInfer model environment.
Use B=1 BF16, greedy decoding, repetition penalty 1.05, the same 512-token budget,
128k context limit, prompt, image limits and STOP/EOS handling. It uses native
prefix caching, fused vLLM operators and V1 CUDA Graph execution, with in-process
engine dispatch to exclude HTTP/IPC from the comparison. Warm up 33 calls and
clear both KV and image preprocessing caches between episodes and after warmup.

From the repository root, with the isolated interpreter and a B=1 configuration:

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD" \
  /path/to/vllm-venv/bin/python benchmarks/activevln-benchmark/benchmark_vllm.py \
  --config benchmarks/activevln-benchmark/runs/batch-tree-shared-context/r2r-b1-config.json \
  --output benchmarks/activevln-benchmark/runs/vllm-0.8.5/r2r-full.json
```

Use the RxR B=1 config in a separate, sequential process for its complete replay.
The optional `--max-steps-per-episode` produces an explicitly partial smoke test;
its latency must not be substituted for the complete selected workload. Record
the vLLM environment separately: this pin uses Torch 2.6.0/CUDA 12.4, while the
selected EmbodiInfer environment uses Torch 2.10.0/CUDA 12.8. It is a comparison
of these two deployed stacks, not an isolation of engine code from dependencies.

In this vLLM version, graph execution is piecewise for the language model;
vision uses the native eager encoder and XFormers attention. The native n-gram
speculation path explicitly disables proposals for non-default repetition
penalties (`vllm/v1/spec_decode/utils.py::is_spec_decode_supported`). Keep the
required 1.05 penalty; do not turn it off to obtain a faster but different task.
EmbodiInfer's phrase-tree verification remains active in its selected baseline.

E2E includes full-history native input processing, all generation and CPU action
parsing. The model interval starts at `get_multimodal_embeddings`, after vLLM
has transferred the first scheduled vision inputs, and ends after all tokens
and stop checks. It includes vision, text prefill, all decode and host dispatch;
it is not first-token latency or a sum of kernel durations. First-token/prefill
timing includes first-token sampling. The benchmark checks current-turn input
IDs, pixel tensors and image grids against the pinned processor outside the
timer, records actual prefix hits and graph execution counts, and retains full
failure evidence. Independently generated histories can differ between engines;
fixed replay still makes no navigation-success claim.

The separate `benchmark_vllm_modern.py` entrypoint requires vLLM 0.30.0. The
validated isolated dependency stack uses Torch 2.13.0, Transformers 5.17.0,
NumPy 2.3.5 and Pillow 12.3.0; all 192 cross-environment CPU input audits match
the original deployment byte for byte. Preserve that processor's PIL backend
for this comparison with request-level `mm_processor_kwargs.use_fast=False`:
the native Qwen processing-info default overrides this option when it appears
only in engine configuration. A second 192-sample audit through the native
renderer checks exact raw pixels and exact BF16 input after the compiled native
GPU normalization operation. Keep `mm_device_do_normalize=True`; comparing its
uint8 transport buffer directly with reference normalized floats is invalid.
The installed dependency list and input evidence are in
`runs/vllm-0.30.0/`.

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD" \
  /path/to/vllm-0.30.0/bin/python benchmarks/activevln-benchmark/benchmark_vllm_modern.py \
  --config benchmarks/activevln-benchmark/runs/batch-tree-shared-context/r2r-b1-config.json \
  --output benchmarks/activevln-benchmark/runs/vllm-0.30.0/r2r-full.json \
  --speculation ngram_gpu --speculative-tokens 16 --no-async-scheduling
```

This entrypoint requests optimization level 3, vision compilation and CUDA
graphs, full/piecewise language graphs, native FlashAttention, prefix caching,
UUID-based multimodal caching and selectable native synchronous/asynchronous
scheduling. The measured B=1 command above selects synchronous scheduling. It
records resolved settings and real vision hits/misses, language graph modes,
verified draft tokens and accepted draft tokens. A successful measured row must
have actual vision graph coverage. `--speculation none`, `ngram` and `suffix`
provide alternative native configurations; suffix requires `arctic-inference`
in the isolated environment. Native version compatibility determines whether
the V1 or V2 model runner is selected and which scheduler options can coexist.
Drain outstanding asynchronous work inside the current observation's timer,
and stop at the first matching token prefix even when a speculative step emits
several tokens. Loading, compilation and capture remain outside timed latency.

`benchmark_vllm_batch.py` uses the selected 16-draft synchronous profile for
B=1/2/4, submitting the current observations as concurrent native requests:

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD" \
  /path/to/vllm-0.30.0/bin/python benchmarks/activevln-benchmark/benchmark_vllm_batch.py \
  --config benchmarks/activevln-benchmark/runs/batch-tree-shared-context/r2r-b2-config.json \
  --output benchmarks/activevln-benchmark/runs/vllm-batch-0.30.0/r2r-b2-full.json
```

Run each split/batch in a fresh, sequential process. Use the corresponding
`r2r-b4`, `rxr-b2` or `rxr-b4` config for the other conditions. Full histories,
checkpoint, generation parameters and all B=1 optimizations remain enabled.
Vision graph item/token budgets and native admission capacity scale with batch
size. Per-episode salts prevent cross-episode KV prefix reuse without resetting
the other live slots. Immutable UUIDs preserve image preprocessing caches.
The saved-frame slot order matches `benchmark_batch.py`, including continuing
after predicted STOP and replacing only episodes whose saved frames are exhausted.

The report separates whole-batch E2E and complete-forward latency from amortized
per-observation costs and throughput. Amortization divides total time by actual
observations, including partially filled tail batches. The complete forward
starts at the first native vision graph and ends after all requests finish;
it includes dispatch and per-request stopping checks. Prefill/decode diagnostics
split at the batch's first generated token and may overlap across requests.
Keep warmup/startup outside timings and report startup/measurement memory
separately. Native graph counters, vision item counts, scheduled request counts,
prefix hits, preemptions and first-two-frame input audits support each result.
`--max-steps-per-episode 2` is a partial smoke probe, not a full latency result.

Candidate inference switches are `cuda_graph`, `fused_ops`, `split_attention`
and `tree_decode` (all default to false; the latter three require `cuda_graph`). Graph startup
captures the shapes visited during warmup, freezes further
capture before measurement, and reports replay/fallback counts. `query_bucket_size`
controls text padding; 1 retains the exact query length for initial parity
checks. Optional `prewarm_context_buckets`, for example
`[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]`, also captures initial/subsequent
prompt shapes from the first two selected frames of each episode. It captures
single-token/tree decoding across those explicitly requested context buckets.
This work is included in reported startup time, outside timed calls. Longer
contexts or unseen image/input shapes still use the counted fallback; the
context limit and generated responses are unchanged. Larger shape coverage uses
more startup time and memory, and the resulting shape plan is recorded.
`graph_workspace_tokens` optionally bounds the separately allocated graph
scratch cache (for example 65536); its default is the model context limit.
It does not change `max_context=128000` or the public recurrent memory. Longer
histories use the counted eager fallback. Requested prewarm context buckets must
fit this workspace; both capacities are recorded in graph runtime statistics.
The current candidate uses separately owned KV workspace and rejects
execution on a different CUDA stream or after moving its captured parameters.
Training and differentiable recompute continue through the eager implementation.
Packed inference KV reduces copy launches while keeping every memory fork private.
The optional greedy `tree_decode` batches candidate phrases from the public prompt:
each accepted token must match the full-vocabulary model choice with the original
repetition penalty. Uncovered continuations use normal decoding; no token cap,
grammar constraint or recorded-answer lookup is introduced. Tree query shapes
can change floating-point accumulation; assess these changes under the selected
accuracy contract, using closed-loop success for the current task.
`tree_repeat_actions` (1 by default, up to 3) can propose repeated public phrases
in one verification pass. It keeps EOS/comma alternatives after each action and
falls back whenever the model chooses another continuation. Larger candidate
trees trade more parallel work for fewer serial forwards; actual accepted-token
and fallback counts are reported.
Task-success validation applies only to the two split-specific profiles reported
above. Other switch combinations still require real-weight validation; these
options do not establish bit-exact parity or the 40 ms latency target.

`split_attention` replaces graph text SDPA with partitioned KV attention. It
keeps GQA storage compact, computes causal/tree visibility within the kernel,
and sums three BF16 probability components with FP32 accumulation. Its changed
reduction order requires validation under the selected accuracy contract.

Root-partitioned phrase trees select only branches matching the already chosen
first token; other vocabulary choices still take the ordinary fallback.
A real-prefix isolation check also found that SDPA key padding alone can change
BF16 results: at RxR episode 5, step 1, decoding with 521 actual keys matches the
reference exactly, while padding to 1024 changes the output. Thus even graph-only
execution remains experimental; passing operator tests or an initial smoke does
not establish full-workload action parity.

`tree_fp32_projection` optionally retains projection outputs and bias addition
in FP32 before casting tree hidden states back to BF16. It requires
`tree_decode`, preserves checkpoint weights and leaves ordinary prefill/serial
projections unchanged. It needs Torch CUDA `mm(out_dtype=torch.float32)` and
still requires model-level accuracy validation. Fused RMSNorm now keeps the
reference Torch mean reduction; fused SwiGLU uses libdevice arithmetic. Their
GPU operator tests require exact output equality, including all finite BF16
SiLU inputs, rather than permitting a one-ULP difference.
