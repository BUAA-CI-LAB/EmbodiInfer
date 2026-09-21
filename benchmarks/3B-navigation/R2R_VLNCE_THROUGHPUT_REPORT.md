# R2R-VLNCE aligned RGB throughput report

Date: 2026-08-23

## Decision

The admitted configuration is **Torch SDPA with the existing manual CUDA Graph runtime**. The new fused Triton operator is not part of this commit or this report.

- Qwen R2R Low: CUDA Graph improves full-policy throughput from 5.132 to 6.380 samples/s.
- Qwen R2R Panoramic: CUDA Graph improves full-policy throughput from 1.811 to 2.293 samples/s.
- Torch Inductor is not admitted: its Low benchmark changed the top token/action on 3 of 192 samples.
- NaViDA CUDA Graph is not admitted on this corpus: paired-seed eager/graph output differed on 14 of 192 samples.

These are inference-throughput measurements. They are not Habitat navigation-success, SR, SPL, or nDTW results.

## Dataset and provenance

The benchmark contains 192 observations: 48 R2R-VLNCE train episodes and four deterministically selected steps per episode. Every sample joins the R2R instruction, the existing StreamVLN-aligned episode RGB export, the current frame, four historical frames, and the action history using the `frame[i] -> action[i+1]` convention.

Important provenance limits:

- The R2R-VLNCE train annotation SHA256 is `340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea`.
- The StreamVLN annotation mirror is pinned to revision `db7602ba7e43a28ab9ffe57adfdea2ea32d91720`; SHA256 is `d0a1c255c2641c61e2d586756a3208ed58e78cf7e781359305ce8f1360d54305`.
- The existing 48-episode image export is structurally aligned with the annotation, but its bytes were not verified against the gated official 23.6 GB image tar. Formal classification is `structurally_aligned_existing_export_not_official_tar_byte_verified`.
- Low consumes the aligned RGB frames directly. Panoramic uses documented, deterministic real-RGB resize/temporal-neighbor derivatives; these are shape-compatible inputs, not official 360-degree panoramas or navigation-neighbor candidates.
- No simulator is run. The report measures inference only.

Manifest details:

| Item | Value |
|---|---|
| Samples | 192 |
| Episodes | 48 |
| Steps per episode | 4 |
| History length | 4 |
| Low manifest SHA256 | `903c5c9daf1ebd68242163c1cbae0c36b5b832bec685c042ebcc820295426972` |
| Panoramic manifest SHA256 | `706cab98167034f211046441ecbd0822adcd2971657a2ca64fc43f80bb3bcd86` |
| Provenance SHA256 | `14871f64a788ba93852908efa843f283249598f0ce72a1651300b7e9c1c7e80f` |

## Environment and configuration

| Setting | Value |
|---|---|
| GPU | One RTX 4090, physical GPU 0 |
| GPU isolation | `CUDA_VISIBLE_DEVICES=0`; GPU 1 observed zero compute activity |
| Torch | `2.13.0+cu130` |
| CUDA runtime | 13.0 |
| Batch size | 1 |
| Seed | 41 |
| Warmup / iterations | 1 / 1 per sample and path |
| Attention | Torch SDPA |
| Torch compile | Disabled for admitted results |
| CUDA Graph | EmbodiInfer manual graph, one fixed text bucket per model |
| Low text bucket | 1024 tokens |
| Panoramic text bucket | 3072 tokens |

Model loading, graph capture, warmup, and disk image loading are outside the timed window.

## Admitted single-token kernel results

This scope starts from already encoded tensors and measures the GPU next-token forward. It excludes the processor, host-to-device transfer, text/action parsing, memory commit, and simulator.

| Model / path | Mean ms | p50 ms | p95 ms | Samples/s |
|---|---:|---:|---:|---:|
| Low, Hugging Face eager | 126.697 | 126.544 | 129.879 | 7.893 |
| Low, self-authored Torch | 113.797 | 114.089 | 117.848 | 8.788 |
| Low, self-authored CUDA Graph | 87.463 | 87.306 | 88.564 | 11.433 |
| Panoramic, Hugging Face eager | 442.228 | 438.462 | 449.589 | 2.261 |
| Panoramic, self-authored Torch | 351.337 | 350.414 | 360.442 | 2.846 |
| Panoramic, self-authored CUDA Graph | 325.372 | 325.000 | 327.085 | 3.073 |

Relative to self-authored uncaptured Torch, CUDA Graph reduces mean kernel latency by 23.1% for Low and 7.4% for Panoramic. Throughput increases by 30.1% and 8.0%, respectively.

## Admitted raw-policy end-to-end results

This scope takes one manifest observation and returns the model action. It includes prompt construction, processor work, host-to-device transfer, model forward, output decoding/action parsing, and policy memory handling. It excludes disk I/O, model load, warmup, graph capture, and simulator execution.

| Model / path | Mean ms | p50 ms | p95 ms | Samples/s | External GPU0 peak MiB |
|---|---:|---:|---:|---:|---:|
| Low, eager | 194.844 | 193.766 | 200.979 | 5.132 | 8,706 |
| Low, CUDA Graph | 156.735 | 154.907 | 168.788 | 6.380 | 8,706 |
| Panoramic, eager | 552.037 | 548.419 | 562.754 | 1.811 | 10,396 |
| Panoramic, CUDA Graph | 436.160 | 435.472 | 442.281 | 2.293 | 10,396 |

CUDA Graph reduces full-policy mean latency by 19.6% for Low and 21.0% for Panoramic. Throughput increases by 24.3% and 26.6%, respectively.

For all 192 samples in both admitted models, Hugging Face/reference, self-authored uncaptured, and captured paths passed the configured logits gate and produced exact matching token, text, and action signatures. No graph capture occurred inside a timed window.

## Non-admitted diagnostics

These numbers describe observed performance only. They must not be presented as accepted backend results.

| Diagnostic | Eager/native | Optimized | Outcome |
|---|---:|---:|---|
| Low Inductor kernel | Native 113.210 ms, 8.833 samples/s | Compiled 109.033 ms, 9.172 samples/s; graph 84.178 ms, 11.880 samples/s | Rejected: 8/192 gate failures, including 3 top-token/action changes |
| NaViDA raw policy | Eager 586.471 ms, 1.705 samples/s, 33.267 tokens/s | Graph 386.094 ms, 2.590 samples/s, 50.546 tokens/s | Rejected: 14/192 paired-seed token/text/action mismatches |

The Inductor configuration used `backend=inductor`, `fullgraph=true`, `dynamic=false`, `triton.cudagraphs=false`, and `emulate_precision_casts=true`. Its compiled and captured paths agreed with each other; the rejected drift was against the Hugging Face/native reference. NaViDA actions remained valid, but paired output identity is required for admission.

## Artifacts

- Manifests and provenance: `/benchmark-artifacts/vvla-benchmarks/navigation-3b/commit7-r2r-vlnce-aligned-real-rgb-v3-20260822`
- Admitted Torch/CUDA Graph JSON and logs: `/benchmark-artifacts/vvla-benchmarks/navigation-3b/commit7-r2r-vlnce-aligned-real-rgb-v3-20260822/results-torch-only-v9-admitted`
- Low Inductor diagnostic: `/benchmark-artifacts/vvla-benchmarks/navigation-3b/commit7-r2r-vlnce-aligned-real-rgb-v3-20260822/results-torch-only-v7-final/kernel-low-inductor.json`
- NaViDA diagnostic: `/benchmark-artifacts/vvla-benchmarks/navigation-3b/commit7-r2r-vlnce-aligned-real-rgb-v3-20260822/results-torch-only-v10-navida/full-navida-none.json`

