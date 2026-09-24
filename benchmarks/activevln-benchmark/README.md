# ActiveVLN benchmark

Replay the same R2R and RxR recorded observations as `../streamvln-benchmark`:
first 48 numeric episode IDs, first instruction, every RGB frame (2,997 R2R and
3,879 RxR calls). The next frame always comes from the recording, even after a
predicted stop; generated response history is retained until the episode ends.
This is an offline latency workload, without Habitat or SR/SPL measurement.
Both datasets use `Arvil/Qwen2.5-VL-3B_rl_r2r_4000` at revision
`160987313e3e869705f42400d1b8f28177044518` and its R2R action grammar.

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
