# Torch Compile Precision-v2 Report

## Measurement contract

- Date: 2026-08-22
- Source state: working tree before commit
- GPU: NVIDIA RTX 4090, physical GPU 0 only
- GPU isolation: GPU 1 had no compute workload during the runs
- Software: PyTorch `2.13.0+cu130`, CUDA `13.0`, Triton `3.7.1`
- Workload: B1, `limit=1`, seed `41`, warmup `3`, timed iterations `10`
- Input: real RGB files loaded through the strict Habitat manifest path
- Artifact directory: `/benchmark-artifacts/embodiinfer-benchmarks/navigation-3b/torch-compile-v2-real-image-20260822/final`

The manifest proves the real-file loader, processor, model, parser, and benchmark
paths without a synthetic fallback. Its Habitat image is not a complete R2R
trajectory evaluation, so these measurements do not establish SR, SPL, NE,
nDTW, multi-episode behavior, or simulator-level policy quality.

Raw JSON artifacts remain in the absolute directory above and are not committed
to the repository.

## Compile contract

The admitted Low and Panoramic compile configuration is fixed as follows:

| Field | Value |
| --- | --- |
| Compile ABI | `qwen25_vl_next_token_compile_v2` |
| Backend | `inductor` |
| `fullgraph` | `true` |
| `dynamic` | `false` |
| Inductor options | `{"triton.cudagraphs": false, "emulate_precision_casts": true}` |
| Shape policy | exact-shape cache, no timed cache growth |
| CUDA Graph owner | EmbodiInfer manual CUDA Graph, not Inductor cudagraphs |

`emulate_precision_casts=true` preserves eager BF16 precision-cast rounding
semantics more closely. It does not promise bitwise equality because Inductor
can still change reduction grouping.

## Pre-encoded next-token kernel

Latency is in milliseconds. Throughput is B1 samples per second.

| Profile | Compile | Path | Mean | P50 | P95 | Samples/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Low | `none` | HF top-level | 57.270 | 56.296 | 61.194 | 17.461 |
| Low | `none` | raw native | 53.870 | 53.778 | 56.504 | 18.563 |
| Low | `none` | manual graph | 36.074 | 36.018 | 36.386 | 27.721 |
| Low | `inductor` | HF top-level | 58.568 | 58.033 | 64.054 | 17.074 |
| Low | `inductor` | raw native | 54.889 | 53.629 | 62.695 | 18.219 |
| Low | `inductor` | compiled | 39.590 | 39.276 | 41.250 | 25.259 |
| Low | `inductor` | compiled manual graph | 32.755 | 32.592 | 33.315 | 30.529 |
| Panoramic | `none` | HF top-level | 158.746 | 158.437 | 160.900 | 6.299 |
| Panoramic | `none` | raw native | 139.850 | 139.923 | 143.144 | 7.151 |
| Panoramic | `none` | manual graph | 114.313 | 114.270 | 114.626 | 8.748 |
| Panoramic | `inductor` | HF top-level | 160.337 | 159.713 | 166.396 | 6.237 |
| Panoramic | `inductor` | raw native | 143.316 | 142.319 | 153.206 | 6.978 |
| Panoramic | `inductor` | compiled | 143.102 | 143.835 | 149.097 | 6.988 |
| Panoramic | `inductor` | compiled manual graph | 112.404 | 112.432 | 113.409 | 8.896 |

Kernel timing uses CUDA events around a pre-encoded GPU invocation. It excludes
image I/O, processor/tokenization, H2D, model loading, first compile/JIT,
compile warmups, manual graph capture, generation, text decode, and parsing.

## Full-policy benchmark

Latency is in milliseconds. `Peak MiB` is the in-process PyTorch allocated-memory
peak reported by the benchmark.

| Profile | Compile | Runtime mode | Mean | P50 | P95 | Samples/s | Peak MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Low | `none` | eager | 124.641 | 124.696 | 130.961 | 8.023 | 7452.18 |
| Low | `none` | manual graph | 104.914 | 104.047 | 114.866 | 9.532 | 7304.33 |
| Low | `inductor` | compiled | 116.681 | 114.671 | 128.590 | 8.570 | 7341.58 |
| Low | `inductor` | compiled manual graph | 103.626 | 103.514 | 105.024 | 9.650 | 7304.33 |
| Panoramic | `none` | eager | 211.978 | 210.546 | 221.639 | 4.717 | 7696.65 |
| Panoramic | `none` | manual graph | 161.924 | 161.801 | 164.010 | 6.176 | 7314.95 |
| Panoramic | `inductor` | compiled | 186.682 | 186.671 | 190.291 | 5.357 | 7416.57 |
| Panoramic | `inductor` | compiled manual graph | 153.510 | 152.867 | 157.717 | 6.514 | 7315.51 |

Full-policy wall timing synchronizes CUDA around `runner.infer_batch` and includes
prompt construction, processor, H2D, prefill/decode, token/text decode, and the
action parser. It excludes manifest/image I/O, model load/device transfer,
shape planning, first compile/JIT, compile warmups, policy warmups, paired
correctness validation, manual graph capture, seed reset, and JSON serialization.

## Relative to fresh `none` processes

Each comparison uses a separately launched fresh `none` process as its baseline.
Negative latency is faster; positive throughput is faster.

| Profile | Scope | Compared paths | Latency change | Samples/s change |
| --- | --- | --- | ---: | ---: |
| Low | Kernel uncaptured | compiled vs raw native | -26.51% | +36.07% |
| Low | Kernel graph | compiled graph vs manual graph | -9.20% | +10.13% |
| Panoramic | Kernel uncaptured | compiled vs raw native | +2.33% | -2.27% |
| Panoramic | Kernel graph | compiled graph vs manual graph | -1.67% | +1.70% |
| Low | Full-policy uncaptured | compiled vs eager | -6.39% | +6.82% |
| Low | Full-policy graph | compiled graph vs manual graph | -1.23% | +1.24% |
| Panoramic | Full-policy uncaptured | compiled vs eager | -11.93% | +13.55% |
| Panoramic | Full-policy graph | compiled graph vs manual graph | -5.20% | +5.48% |

## Cold start, capture, and process memory

`First call` is the compile runtime's first-call wall time. `Outer wall` includes
process startup, model loading, compilation, warmups, validation, timing, and
serialization. `External peak` is the externally observed process/GPU peak and
is distinct from the in-process PyTorch peak in the full-policy table.

| Scope | Profile | Compile | First call ms | Capture ms | Outer wall s | External peak MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Kernel | Low | `none` | N/A | 63.707 | 20.238 | 8312 |
| Kernel | Low | `inductor` | 67092.265 | 29.620 | 91.052 | 8304 |
| Kernel | Panoramic | `none` | N/A | 148.055 | 65.193 | 8968 |
| Kernel | Panoramic | `inductor` | 216595.671 | 110.702 | 251.820 | 8950 |
| Full-policy | Low | `none` | N/A | 67.562 | 21.401 | 8318 |
| Full-policy | Low | `inductor` | 69073.441 | 31.848 | 118.183 | 8142 |
| Full-policy | Panoramic | `none` | N/A | 124.469 | 22.429 | 8976 |
| Full-policy | Panoramic | `inductor` | 212481.051 | 116.573 | 291.918 | 8496 |

All timed windows observed zero new graph captures and no change in compile
attempts, failures, or cache entries. The cold-start cost is therefore excluded
from steady-state latency, but it is operationally significant: approximately
67-69 seconds for Low and 212-217 seconds for Panoramic in these runs.

## Correctness gate

All eight formal runs passed their applicable gate.

- HF top-level versus raw Torch kept `rtol=0.02, atol=0.05`.
- HF/raw Torch versus BF16 Inductor required `rtol=0.06, atol=0.30`, top-1 exact,
  and top-50 overlap of at least 45/50.
- Token, decoded text, and parsed action signatures were exact.
- Compiled uncaptured versus the same compiled callable under manual graph used
  `rtol=0.02, atol=0.05`; observed top-50 overlap was 50/50 and max error was 0.
- Timed compile counters were unchanged and timed graph capture count was zero.

Low and Panoramic are deterministic argmax policies. Top-1, token, decoded text,
and parsed action exactness are the behavioral gates; top-50 is an additional
distribution guard. The 45/50 threshold is a principled 90% boundary rather than
a fit to these observations: the formal manifest achieved at least 48/50, while
the extended changed-image check reached a worst case of 47/50. Precision-cast
emulation does not guarantee bitwise eager/compiled equivalence. Triton hybrid's
separate 48/50 gate is unchanged.

## NaViDA admission

NaViDA is not admitted to `torch.compile`. On the real 3B checkpoint, its history
stochastic case diverged in paired-seed token output, so compiled decode cannot
claim policy correctness. NaViDA remains on `compile_backend=none`; this report
does not publish compiled NaViDA performance.

## Recommendation

`compile_backend=none` remains the default. `inductor` is an explicit opt-in for
validated Low and Panoramic deployments that can amortize a large cold start.
The steady-state gains are workload dependent: Low improves materially in the
kernel path, Panoramic uncaptured kernel does not, and the manual-graph gains are
smaller than the full-policy uncaptured gains. Deployment decisions must account
for both steady-state throughput and the recorded first-call/outer-wall cost.
