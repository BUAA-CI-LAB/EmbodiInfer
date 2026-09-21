# Recurrent History Image Cache Report

Date: 2026-08-22

## Scope

This report records the formal CPU-only benchmark of the Low and Panoramic
recurrent history image cache. It measures continuous preprocessing through the
formal policy prompt, image interleave, PIL preparation, official Qwen
processor, cache lookup/reconstruction, and successful current-entry commit.
It is not a model end-to-end benchmark and does not measure navigation task
success.

The result JSON and image assets remain external and are not committed. The
complete artifact root is:

```text
/benchmark-artifacts/embodiinfer-benchmarks/navigation-3b/history-image-cache-v1-final2-real-image-smoke-20260822
```

## Environment and workload

| Item | Value |
| --- | --- |
| Python | `3.12.3` |
| PyTorch | `2.13.0+cu130` |
| NumPy | `2.5.2` |
| Interpreter | `/benchmark-artifacts/venvs/embodiinfer/bin/python` |
| Device visibility | CPU-only, `CUDA_VISIBLE_DEVICES=""` |
| Steps per session | 8 |
| Schedule | 5 iterations, ABBA on even iterations and BAAB on odd iterations |
| Timed samples | 10 sessions and 80 steps per mode and profile |
| Cache capacity | 16 MiB per session |

PyTorch is a CUDA build, but CUDA was made invisible to both formal benchmark
subprocesses. No GPU work is included in the timing results below.

Each manifest reuses existing real RGB assets from the b5e Habitat smoke
directory. The source is Habitat Lab's `habitat-lab-demo.png`, SHA256
`c7e1a9e6344bee883dbb605bd00e0e0b9f5b8c57bb14ce33409c9b49205803e8`.
The profile images are deterministic PIL RGB fits/crops of that file, not
random, zero, placeholder, or generated pixels.

| Profile | Final 8-step manifest SHA256 | Source b5e manifest SHA256 |
| --- | --- | --- |
| Low | `0ee978ac5e16398ac2eb8576000f897f1cb97858cf882ee3d5cded4a5d8f6ac5` | `0dc1934348dba6af8b76d4f6f6e3f454ba5668e71cb69fd554bc4c60a0d020b3` |
| Panoramic | `caf1bfbca54b8ff75aae62fd0d0d6013adde8737f3a2514c1400bd016ef6cd2a` | `ab9a71d983af119327b22d0e14f3f4098e795fd92eac1f4d0a668455726a8a54` |

The Low session repeats one real source image for eight steps. The Panoramic
session repeats one real `960x240` panorama and the same four real `320x240`
candidate crops for eight steps. Provenance is therefore
`real_image_smoke_reuse_not_trajectory`: this is not an R2R trajectory and is
not evidence of trajectory-level performance.

## Timing boundary

Each mode receives one complete untimed warmup session. Timed sessions start
from empty policy memory and follow the alternating ABBA/BAAB schedule to reduce
order bias.

Included:

- Formal prompt construction and image interleave.
- History, current, and candidate PIL preparation.
- History cache lookup and RGB reconstruction.
- Official Qwen processor execution.
- Successful current-entry cache commit.

Excluded:

- Checkpoint processor/tokenizer load.
- Manifest parsing and image decode.
- Per-mode warmup.
- Model load and model inference.
- Result JSON serialization.

The outer wall time reported later covers the complete subprocess, so it is not
the same timing boundary as the pooled per-step values.

## Pooled step results

| Profile | Mode | Mean (ms) | P50 (ms) | P95 (ms) | Steps/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Low | `none` | 244.494 | 245.982 | 425.275 | 4.090 |
| Low | `rgb_bytes` | 70.390 | 71.914 | 84.659 | 14.207 |
| Panoramic | `none` | 84.736 | 92.447 | 132.375 | 11.801 |
| Panoramic | `rgb_bytes` | 80.553 | 80.533 | 127.921 | 12.414 |

Relative `rgb_bytes` change against the paired `none` mode:

| Profile | Mean latency | P50 latency | P95 latency | Steps/s |
| --- | ---: | ---: | ---: | ---: |
| Low | -71.21% | -70.76% | -80.09% | +247.34% |
| Panoramic | -4.94% | -12.89% | -3.36% | +5.19% |

## Cache and render accounting

The following counters are totals over the 10 timed sessions for each mode.

| Profile | Mode | Current renders | History renders | Candidate renders | Hits | Misses | Appends | Evictions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | `none` | 80 | 280 | 0 | 0 | 0 | 0 | 0 |
| Low | `rgb_bytes` | 80 | 0 | 0 | 280 | 0 | 80 | 0 |
| Panoramic | `none` | 80 | 280 | 320 | 0 | 0 | 0 | 0 |
| Panoramic | `rgb_bytes` | 80 | 0 | 320 | 280 | 0 | 80 | 0 |

Every cached session ended with `base_frame_index=0` and no eviction:

| Profile | Final entries per session | Final resident bytes per session |
| --- | ---: | ---: |
| Low | 8 | 1,843,200 |
| Panoramic | 8 | 5,529,600 |

Panoramic candidate rendering remained exactly 320 calls in both modes.
Candidates are intentionally processed on every step and never enter the
history cache.

## Correctness

Both profiles returned `status=pass`. For every warmup and timed session, all
eight steps had exact processor tensor names, dtypes, shapes, and content
SHA256 between `none` and `rgb_bytes`. No numerical tolerance is used.

The formal processor SHA256 was
`8d55622ce2c010e1991233a47c138fccca18541721fba78dd43fd03180894e36`.
The system prompt SHA256 values were:

| Profile | System prompt SHA256 |
| --- | --- |
| Low | `f5a7f75be433db6c6ead1702baf334ee0f0d81191a9a7a22bf02003fbce52559` |
| Panoramic | `5fe1a4bc58b4d124f15c4b376bfa2374681a10259619abd221926eb74c3a905e` |

## Process resources

`wall_seconds` uses `time.perf_counter()` around the complete subprocess.
Peak RSS is Linux `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`.

| Profile | Subprocess return | Outer wall (s) | Peak RSS (KiB) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Low | 0 | 50.865 | 1,364,340 | 1,332.363 |
| Panoramic | 0 | 36.833 | 1,017,228 | 993.387 |

## Validation gates

CPU-only policy and history-cache specialty gate:

```text
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$D" /benchmark-artifacts/venvs/embodiinfer/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_qwen25_vln_history_image_cache.py tests/test_qwen25_vln_policies.py
54 passed in 5.65s; outer wall 18.194s
```

Final CPU-only full suite:

```text
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$D" /benchmark-artifacts/venvs/embodiinfer/bin/python \
  -m pytest -q -p no:cacheprovider
246 passed, 36 skipped in 89.39s; outer wall 95.858s; rc=0
```

Independent real-checkpoint GPU parity used physical GPU 0 only:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  pytest -q -s -x -p no:cacheprovider tests/test_qwen25_vln_real_parity.py
11 passed, 2 skipped, 14 warnings in 686.31s; outer wall 696.956s
```

GPU 0 peak allocation was 9,032 MiB and GPU 1 had zero compute observations.
The independent GPU lifecycle artifact is:

```text
/benchmark-artifacts/validation/commit6-lifecycle-full-parity-1787402600
```

## Interpretation

Low's large preprocessing gain primarily comes from avoiding repeated resize
and history image conversion: its real source image is `1414x1872` and the
formal policy repeatedly converts it to `320x240`. Panoramic input already
matches the formal `960x240` contract, while four uncached candidates remain in
every step. Its smaller gain is therefore more representative of already
contract-sized visual input.

These results establish exact preprocessing parity and show the cost removed by
the recurrent RGB cache for this fixed smoke workload. They must not be
extrapolated to R2R trajectory diversity, full-policy model throughput, SR,
SPL, nDTW, navigation quality, or task success rate.
