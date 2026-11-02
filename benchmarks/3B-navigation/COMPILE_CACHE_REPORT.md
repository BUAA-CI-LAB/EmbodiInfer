# Persistent Compile-Cache Report

## Scope

- Date: 2026-08-22.
- Result: **12/12 passed**.
- Artifact root: `/benchmark-artifacts/embodiinfer-benchmarks/navigation-3b/torch-compile-cache-v1-final2-real-image-20260822`.
- Result JSON remains under that remote artifact root and is not committed.
- Hardware: NVIDIA RTX 4090, physical GPU 0 only. GPU 1 reported zero compute activity.
- Software: PyTorch `2.13.0+cu130`, CUDA 13, Transformers `4.57.1`, Triton `3.7.1`.
- Input: real Habitat-document image manifest, batch 1, limit 1, seed 41, warmup 3, timed iterations 10. The document image is real input data, but it is not a complete R2R trajectory.
- Text shapes: Low `503 -> 512`; Panoramic `1301 -> 1344`.

The producer started with an empty persistent cache. The same-directory consumer
reused the producer directory. The fresh-directory consumer admitted the same
content-addressed artifacts through the Mega-Cache path.

## First-call and admission

| Scope | Producer cold (s) | Same-dir consumer (s) | Same reduction | Fresh-dir consumer (s) | Fresh reduction |
|---|---:|---:|---:|---:|---:|
| Low kernel | 69.235 | 31.529 | -54.46% | 32.168 | -53.54% |
| Low full policy | 70.466 | 33.786 | -52.05% | 34.021 | -51.72% |
| Panoramic kernel | 222.392 | 58.445 | -73.72% | 58.199 | -73.83% |
| Panoramic full policy | 220.369 | 59.773 | -72.88% | 57.844 | -73.75% |

Both consumer modes reported:

- artifact loaded: `true`;
- artifact published: `false`;
- compilation skipped by cache admission: `true`;
- FX hit/miss: `1/0`;
- admission: passed.

Each producer published its artifact and reported FX hit/miss `0/1`. No AOT
artifact was present, so AOT admission was not required and its hit/miss count
was `0/0`. Residual asynchronous compiler misses were still observed outside
the admitted FX artifact. These results therefore demonstrate persistent FX
artifact admission, not zero compilation.

## Published artifacts

SHA-256 values are abbreviated because the validation record supplied only
their prefix and suffix.

| Scope | Bytes | SHA-256 |
|---|---:|---|
| Low kernel | 15,407,444 | `b6d987a8...73ad42` |
| Low full policy | 15,453,575 | `2bd5da57...1973da` |
| Panoramic kernel | 33,899,391 | `f71be54d...4b0df3` |
| Panoramic full policy | 33,979,584 | `df2d5e2e...3df4b4` |

## Kernel steady state

Values in the mode columns are `mean / p50 / p95 ms; samples/s`.

Producer-only HF and raw-native references:

| Profile | HF | Raw native |
|---|---|---|
| Low | 61.695 / 61.741 / 63.887; 16.209 | 59.632 / 58.799 / 64.641; 16.770 |
| Panoramic | 170.249 / 169.947 / 171.885; 5.874 | 153.443 / 153.309 / 156.522; 6.517 |

Compiled and captured modes:

| Profile | Phase | Compiled | Captured |
|---|---|---|---|
| Low | Producer | 43.771 / 43.397 / 47.393; 22.846 | 34.702 / 34.646 / 35.265; 28.817 |
| Low | Same-dir | 43.643 / 43.773 / 45.851; 22.913 | 34.974 / 34.833 / 35.872; 28.593 |
| Low | Fresh-dir | 43.015 / 43.142 / 44.728; 23.248 | 34.661 / 34.599 / 35.272; 28.851 |
| Panoramic | Producer | 149.990 / 148.867 / 160.522; 6.667 | 116.496 / 115.475 / 126.099; 8.584 |
| Panoramic | Same-dir | 148.232 / 147.327 / 156.638; 6.746 | 116.169 / 114.777 / 127.038; 8.608 |
| Panoramic | Fresh-dir | 147.473 / 146.430 / 157.108; 6.781 | 116.648 / 115.160 / 126.515; 8.573 |

## Full-policy steady state

Values are `mean / p50 / p95 ms; samples/s`.

| Profile | Phase | Compiled | Compiled + manual CUDA Graph |
|---|---|---|---|
| Low | Producer | 116.609 / 114.878 / 132.263; 8.576 | 112.424 / 107.407 / 154.275; 8.895 |
| Low | Same-dir | 121.639 / 118.220 / 148.724; 8.221 | 108.236 / 107.643 / 115.759; 9.239 |
| Low | Fresh-dir | 120.179 / 118.408 / 135.845; 8.321 | 109.294 / 107.384 / 122.807; 9.150 |
| Panoramic | Producer | 462.912 / 555.062 / 592.959; 2.160 | 160.610 / 158.493 / 174.620; 6.226 |
| Panoramic | Same-dir | 189.555 / 186.577 / 200.816; 5.276 | 169.074 / 160.554 / 235.757; 5.915 |
| Panoramic | Fresh-dir | 193.468 / 191.524 / 208.915; 5.169 | 158.541 / 156.265 / 168.259; 6.307 |

The Panoramic producer compiled measurement is a clear first-process steady
window outlier. It is retained rather than filtered.

Peak external GPU memory was 8,308 MiB for Low and 8,980 MiB for Panoramic.

## Cross-run no-bucket reference

This table compares the final2 fresh-directory means with the formal Commit 4
no-bucket run. It is a **cross-run reference, not a controlled A/B result**.
Positive latency deltas mean final2 was slower.

| Scope | Commit 4 no-bucket (ms) | Final2 fresh (ms) | Latency delta |
|---|---:|---:|---:|
| Low kernel compiled | 39.590 | 43.015 | +8.65% |
| Low kernel captured | 32.755 | 34.661 | +5.82% |
| Panoramic kernel compiled | 143.102 | 147.473 | +3.05% |
| Panoramic kernel captured | 112.404 | 116.648 | +3.78% |
| Low full compiled | 116.681 | 120.179 | +3.00% |
| Low full graph | 103.626 | 109.294 | +5.47% |
| Panoramic full compiled | 186.682 | 193.468 | +3.64% |
| Panoramic full graph | 153.510 | 158.541 | +3.28% |

Text buckets exist to reuse compiled programs across multiple raw text shapes.
This one-sample manifest exercised one raw shape per profile, so it does **not**
prove that buckets reduce the number of compilations across a representative
shape distribution. The modest cross-run steady-state regressions can reflect
masked padding, changed shapes, and run-to-run variance; persistent-cache
admission primarily improves cold first-call latency and does not guarantee a
steady-state speedup.

## Timing and correctness

- First-call compilation, artifact load/save, JIT work, warmup, and graph
  capture were outside the timed steady-state windows.
- Timed compile counters did not change, and timed CUDA Graph capture count was
  zero.
- All configured logits parity gates, token/text/action checks, action validity,
  and stability checks passed.
- GPU 1 was unused and recorded zero compute activity.
