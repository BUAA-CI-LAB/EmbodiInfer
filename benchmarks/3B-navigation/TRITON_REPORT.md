# Qwen2.5-VL Navigation Triton Evaluation Report

## Run identity

| Field | Value |
|---|---|
| Date | 2026-08-22 |
| Source revision | `working tree before commit` |
| GPU | NVIDIA RTX 4090, physical GPU 0 only |
| Other GPU | GPU 1 unused |
| Compute capability | sm89 |
| PyTorch | `2.13.0+cu130` |
| Triton | `3.7.1` |
| Batch | B1 |
| Warmup | 3 iterations |
| Timed iterations | 10 |

The manifest references real RGB files. The image provenance is Habitat
documentation imagery, not frames rendered along an R2R trajectory. These results
measure the policy runtime and kernel paths only. They are not R2R navigation
quality, SR, SPL, NE, or nDTW results.

All runs used physical GPU 0. GPU 1 was not used. Triton JIT compilation and CUDA
Graph capture were completed outside the timed window. JIT latency was not
measured as a separate metric.

The formal measurements used the same operator implementation as the final code.
After the runs, only the selection/cache ABI metadata was updated from v2 to v3 to
describe the actual short/long launches and FP32 precision correctly. Kernel
computation and the timed execution path did not change, so the performance runs
do not need to be repeated.

## Pre-encoded next-token kernel

Latency values are milliseconds. Throughput is samples per second. Peak memory is
the run-level allocated-memory peak and is repeated across the three paths from the
same run. `C/R/E` means graph captures, replays, and cache entries. Capture time is
reported only on the captured path.

| Model | Effective backend plan | Path | Mean | P50 | P95 | Samples/s | Peak MiB | Capture ms | C/R/E |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Low | Torch SDPA | HF top-level eager | 56.263 | 56.278 | 56.337 | 17.774 | 7453.922 | N/A | N/A |
| Low | Torch SDPA | Native uncaptured | 52.525 | 52.538 | 52.596 | 19.039 | 7453.922 | N/A | N/A |
| Low | Torch SDPA | Captured replay | 36.065 | 36.062 | 36.200 | 27.728 | 7453.922 | 65.062 | 1/14/1 |
| Low | Triton hybrid | HF top-level eager | 56.826 | 56.337 | 58.597 | 17.598 | 7453.922 | N/A | N/A |
| Low | Triton hybrid | Native uncaptured | 51.881 | 51.765 | 53.159 | 19.275 | 7453.922 | N/A | N/A |
| Low | Triton hybrid | Captured replay | 37.466 | 37.427 | 37.785 | 26.691 | 7453.922 | 58.857 | 1/14/1 |
| Panoramic | Torch SDPA | HF top-level eager | 158.307 | 158.112 | 159.420 | 6.317 | 7698.388 | N/A | N/A |
| Panoramic | Torch SDPA | Native uncaptured | 143.083 | 142.889 | 149.002 | 6.989 | 7698.388 | N/A | N/A |
| Panoramic | Torch SDPA | Captured replay | 114.101 | 113.919 | 115.055 | 8.764 | 7698.388 | 129.825 | 1/14/1 |
| Panoramic | Triton hybrid | HF top-level eager | 161.829 | 160.975 | 168.616 | 6.179 | 7698.388 | N/A | N/A |
| Panoramic | Triton hybrid | Native uncaptured | 129.043 | 128.531 | 133.304 | 7.749 | 7698.388 | N/A | N/A |
| Panoramic | Triton hybrid | Captured replay | 122.248 | 122.121 | 122.911 | 8.180 | 7698.388 | 74.757 | 1/14/1 |

### Kernel throughput change

The percentages below compare Triton hybrid with Torch SDPA for the same model and
path.

| Model | Path | Samples/s change |
|---|---|---:|
| Low | Native uncaptured | +1.241% |
| Panoramic | Native uncaptured | +10.881% |
| Low | Captured replay | -3.738% |
| Panoramic | Captured replay | -6.665% |

Triton improves the uncaptured Panoramic kernel most clearly, but the captured
replay path is slower than Torch SDPA for both models in this run.

## Full-policy benchmark

The full-policy path includes prompt construction, processor, host-to-device
transfer, prefill, decode, text decode, and action parsing. Capture and warmup are
outside the timed window.

| Model | Effective backend plan | Mode | Mean ms | P50 ms | P95 ms | Samples/s | Peak MiB | Capture ms | C/R/E |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Low | Torch SDPA | Eager | 125.043 | 123.735 | 134.680 | 7.997 | 7432.661 | N/A | N/A |
| Low | Torch SDPA | Manual graph | 98.248 | 97.780 | 101.159 | 10.178 | 7346.327 | 66.156 | 1/13/1 |
| Low | Triton hybrid | Eager | 126.642 | 125.293 | 136.529 | 7.896 | 7432.661 | N/A | N/A |
| Low | Triton hybrid | Manual graph | 102.878 | 102.819 | 104.679 | 9.720 | 7346.327 | 59.811 | 1/13/1 |
| Panoramic | Torch SDPA | Eager | 207.425 | 207.487 | 211.154 | 4.821 | 7672.890 | N/A | N/A |
| Panoramic | Torch SDPA | Manual graph | 161.496 | 161.974 | 163.344 | 6.192 | 7430.411 | 133.638 | 1/13/1 |
| Panoramic | Triton hybrid | Eager | 207.360 | 207.008 | 210.155 | 4.823 | 7672.890 | N/A | N/A |
| Panoramic | Triton hybrid | Manual graph | 168.894 | 168.738 | 171.772 | 5.921 | 7430.411 | 71.568 | 1/13/1 |

| Model | Full-policy manual-graph samples/s change, Triton vs Torch |
|---|---:|
| Low | -4.501% |
| Panoramic | -4.380% |

All four full-policy runs passed. Timed capture count was zero, token/text/action
parity was exact, outputs were stable within each mode, and every parsed action was
valid.

## Correctness contract and hybrid plan

The explicit Triton request resolves to a hybrid implementation:

| Component | Resolved implementation |
|---|---|
| Window attention | `triton_segmented` |
| Full attention | `torch_sdpa` |
| Vision RoPE | `triton` |

Torch SDPA HF-vs-native and HF-vs-captured comparisons use
`rtol=0.02, atol=0.05`. Triton hybrid accumulates reduction-order differences over
32 vision layers, so HF-vs-native and HF-vs-captured use
`rtol=0.06, atol=0.30`, require exact top-1, and require at least 48 of the top 50
token IDs to overlap. The 48/50 condition is a 96% boundary-set gate.

Native-vs-captured always uses `rtol=0.02, atol=0.05`, independent of backend.
All kernel parity gates passed. Full-policy token, decoded text, and parsed action
comparisons remained exact. No fallback was accepted for the explicit Triton runs.

## NaViDA paired-seeded rerun

NaViDA is stochastic and does not use the Low/Panoramic shared Triton hybrid plan.
The formal rerun validates the corrected stochastic benchmark contract:
`stochastic=true`, `stability_required=false`, and
`paired_seeded_parity=true`.

| Mode | Mean ms | P50 ms | P95 ms | Samples/s | Tokens/s | Peak MiB | Capture ms | C/R/E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Eager | 548.916876 | 546.492466 | 564.242433 | 1.821769 | 38.257159 | 7374.810 | N/A | N/A |
| Manual graph | 300.975636 | 300.876466 | 302.815212 | 3.322528 | 69.773090 | 7374.804 | 33.350158 | 1/300/1 |

The manual-graph samples/s improvement over eager was +82.38%. Status was `pass`:
paired eager/graph runs used the same recorded seed and produced exact token, text,
and action outputs; all actions were valid; timed capture count was zero. Random
timed samples were not incorrectly required to remain signature-stable.

## Historical one-iteration reference

The earlier `iters=1` observations are retained only as historical context and are
not strictly comparable with the formal `warmup=3, iters=10` runs.

| Model | Eager mean ms | Graph mean ms |
|---|---:|---:|
| Low | 139.575 | 99.298 |
| Panoramic | 239.239 | 161.691 |
| NaViDA | 554.483 | 302.750 |

No performance conclusion in this report is based on the historical one-iteration
numbers.

## Artifacts

Formal Low and Panoramic JSON artifacts remain on the remote host at:

```text
/benchmark-artifacts/embodiinfer-benchmarks/navigation-3b/triton-v1-real-image-20260822/final/final-run
```

The formal NaViDA rerun artifact is:

```text
/benchmark-artifacts/embodiinfer-benchmarks/navigation-3b/triton-v1-real-image-20260822/final/final-run/navida-rerun.json
```

The JSON result files are intentionally not committed to the repository.

## Conclusion

`auto` remains on Torch SDPA. The explicit `triton` backend remains experimental.
The hybrid path improves native uncaptured throughput, especially for Panoramic,
and reduces measured capture time, but it regresses captured replay and full-policy
manual-graph throughput in these formal runs. Adoption should wait for a captured
path improvement while preserving the documented parity gates.
