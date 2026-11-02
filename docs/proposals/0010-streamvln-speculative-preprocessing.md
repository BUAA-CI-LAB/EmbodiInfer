# 0010 — StreamVLN tree decoding and ActiveVLN operator integration

- Status: Implemented, opt-in validation
- Date: 2026-10-27

## 1. Summary

Add independently selectable StreamVLN action speculation and GPU image
preprocessing, borrowing the candidate-verification and exact Pillow-resize
approaches from ActiveVLN. Policy-specific work stays under `policies/streamvln`;
integer RGB resizing belongs in the reusable backend. Defaults retain the
existing execution path.

## 2. Motivation and current gap

`StreamVLNDecoder._decode_cuda_blocks` records serial transformer calls in one
graph, but still reads the model weights once per token. The current fixed
response-envelope shortcut also generates four actions serially. StreamVLN's
repeated, single-token arrows allow a small causal verification batch.
`StreamVLNProcessor._preprocess` currently performs Pillow resize and SigLIP
normalization on the CPU before uploading float pixels.

## 3. Goals and non-goals

Reduce serial action forwards and CPU image-processing work while retaining
full-vocabulary greedy selection, the original repetition penalty, EOS, token
budget, cancellation and transactional episode state. New quantization algorithms, changes to
image interpolation, constrained action sampling and generic engine scheduling are
outside this change. This does not make the BF16 model fit Orin NX 16 GB.

## 4. Design

### ActiveVLN optimization integration

The complete opt-in profile retains batched projections and adds grammar-tree
verification, root partitioning, online proposal counts, ancestor-only attention,
depth-based rotary positions and accepted-path KV compaction. The old repeated
chain remains an independent numerical control, not the performance profile.
The explicit batched runtime combines vision, packed causal prefill and
independent tree queries. A pooled scratch arena imports private episode KV
only when it is not already resident; accepted paths are compacted and committed
incrementally. Older forks cannot borrow another output's unused capacity.
Cancellation invalidates residency and preserves committed prefixes. Runtime
ownership includes configuration, parameter storage/version and CUDA stream.
Each node selects from the full vocabulary. Grammar is a proposal prior, never
a sampling constraint or an output-length cap.

`tree_refresh` configures proposal-cache invalidation after a positive integer
number of completed responses, defaulting to 64. Shorter intervals let new
response counts affect proposals sooner while increasing CPU drafting work.
Single-session and batched runtimes use the same setting; changing the policy
profile invalidates its graph runtime, and an existing batch runtime rejects
configuration changes. Refresh changes do not constrain the target vocabulary.
Changed tree layouts still require real-weight token, log-probability and KV
parity checks, followed by latency and navigation validation; fewer predicted
verification calls alone do not establish a speedup.

| ActiveVLN mechanism | StreamVLN integration |
| --- | --- |
| Rounded RMSNorm/RoPE/SwiGLU | Shared backend operators, independent `fused_ops` switch |
| Split-KV and grouped-query attention | Shared kernels, `split_attention` and `attention_kernel` |
| Decode GEMM, fused QKV and gated projection | Shared `decode_linear` backend, batched candidate rows |
| FP32-output tree projections | Alternative `tree_fp32_projection` |
| Phrase/grammar trees | StreamVLN envelope and symbolic-action grammar, budgeted tree |
| Vision/prefill/decode/tree graphs | Startup-only capture, shared pools and counted fallbacks |
| Resident KV and incremental commit | Fixed-address cache plus accepted tree-path compaction |
| GPU image pipeline and byte inputs | Exact SigLIP processing, optional uint8 observations |
| Fused vision attention | SigLIP adaptation of shared segmented attention |
| Exact query shapes / no padded work | Existing B=1 exact-length prefill; ragged batching uses a separate runtime |

The ActiveVLN published optimized profile permits different output tokens and
requires closed-loop navigation validation. That contract is not silently
inherited by StreamVLN. Keep the reference and strict repeated-chain paths;
report every optimized-profile token/action mismatch. Replay timing alone does
not certify losslessness or navigation equivalence. BF16 weights stay unchanged;
existing quantization backends require their separate precision evaluation.

The explicit `sdpa` and `fused` SigLIP paths capture their selected operators
directly into CUDA Graphs in both single-session and batched runtimes. Do not
apply an additional whole-vision `torch.compile` pass: it can fuse across BF16
rounding boundaries and alter features even with identical weights and pixels.
Graph/eager equality is checked for the same attention operator; choosing the
FP32-probability fused attention profile remains a separately reported numerical
change from SDPA. The HF compatibility path retains its existing compiler path.

INT8 checkpoint loading converts one CPU projection at a time, packs the
converted QKV and gate/up tensors, and transfers only final tensors to the
serving device. This avoids first allocating the complete BF16 model on a
16 GB Jetson. Preserve floating vision, embeddings, LM head, biases and ignored
projections. Conversion uses the existing per-output-channel INT8 rule; packed
payloads and scales must match the existing full-model CPU conversion exactly.
This loading change does not make W8A8 lossless relative to BF16. Checkpoint
placement, copying and storage lifecycle remain deployment concerns. Jetson
closed-loop validation requires a complete, hash-verified checkpoint on local
NVMe before service startup; remote memory-mapped checkpoint reads are excluded
from that deployment protocol. Report loading and warmup separately from
inference timings.
Native INT8 prefill remains eager and can allocate fresh KV storage at a window
boundary. Its tree graphs therefore own separate, fixed-address KV staging:
copy the current prefix into that storage, replay the captured graph, and copy
only newly verified nodes back. Never stage into another session's borrowed
cache. Preserve the caller's prefix on graph failure, validate parameter and
buffer ownership before reuse, and count staged replays separately from direct
replays and eager fallbacks. Validate exact scores, newly written KV, and prior
session preservation across multiple windows before using this path in a full
navigation comparison.
The additional `quantized_prefill_graphs` profile captures exact-length INT8
prefills through the same optimized text operators as eager execution. Prefill
and tree graphs share one private fixed-address KV arena and allocator pool;
session storage remains separate. Copy the committed prefix into the arena and
publish only the newly computed suffix after success. Reuse requires unchanged
parameter/buffer storage, text options and CUDA stream. Unknown frozen query
shapes use a counted eager fallback. This opt-in path avoids another complete
KV allocation and must pass interleaved prefill/tree tests, failure rollback,
fresh-window replay, and real-weight memory/accuracy checks before promotion.
The shared native INT8 activation converter reduces extrema before widening and
uses one owned floating workspace for division, rounding and clipping. Large
activations are converted in row chunks with a 16 MiB workspace target, while
retaining the complete INT8 matrix and original GEMM shape. This
preserves the existing INT8 payloads/scales while reducing temporary storage at
large slow-memory prefills; it never mutates input or graph-owned activations.

The optional INT8 `native_fused` backend fuses row conversion and output scaling
around the same native integer GEMM. It preserves activation payloads/scales,
the existing 17-row padding for small matrices, and each FP16/BF16 epilogue
rounding boundary. It does not inherit ActiveVLN's FP32-rescaled INT8 epilogue,
which is a different numerical contract. Default backend selection is unchanged.
Admission requires exact activation and projection comparisons, graph replay
with changed inputs, realistic 18,944-channel rows, and real-weight navigation;
CPU routing or synthetic tests alone do not certify this backend for a device.
The `triton_native` alternative also fuses integer GEMM with that rounded epilogue,
retaining the accumulator in registers rather than allocating a complete INT32
matrix. This removes the large gate/up accumulation buffer at slow-memory resets.
Integer tiling is exact within the validated K bound; its native-rounding
epilogue is intentionally distinct from the ActiveVLN calibrated adapter.

For closed-loop action chunks, `Observation.metadata["stream_step_id"]` names
the current primitive frame. `Observation.images` contains every new frame since
the committed history, including intermediate frames between model calls. Only
potential slow-memory candidates pass through the vision encoder; intermediate
frames never create language turns. Reset fast KV after crossing a 32-primitive
boundary and use that boundary's eight-frame slow selection. Validate that frame
count and step ID agree before encoding, and leave committed memory unchanged on
failure. Omitting this metadata retains the one-frame-per-request replay contract.
The explicit batching runtime accepts independent primitive-frame strides in
each row. It encodes current frames together and retained intermediate frames
in chunks bounded by episode capacity, restoring their original frame indices
before slow-memory selection. These intermediate frames create no text turns.
Features own storage before vision graph scratch is reused; failures leave each
input history intact. Graph ownership includes INT8 payload/scale buffers as
well as floating parameters. Batch-shape changes still require their own
numerical and navigation comparison; B1 agreement is not a certificate for B8.
Its optional `packed_prefill_bucket` rounds the total projected token count
up to a fixed multiple, allowing ragged lengths in that bucket to reuse one
graph. The default remains one (exact length). Padding uses unique KV slots
after the episode pool, never a live session's region, and has zero attention
outputs without reading unwritten partial reductions. Actual query lengths,
rotary positions and attention partition extents remain unchanged. Startup
`prewarm_prefill` accepts explicit per-row projected token lengths and captures
zero-embedding probes without decoding or retaining histories. Graph coverage
and padding work are counted. Same-bucket eager/graph comparisons are required;
larger GEMM shapes, especially floating projections, additionally require
comparison against exact-length execution and real-weight navigation before
promotion. These checks cannot be replaced by synthetic graph replay alone.
Optional `shared_graph_buffers` reuses fixed-address text input and output
storage across serial graphs. Power-of-two growth retains older blocks so
captured addresses never move, while forward intermediates are released within
capture. The runtime's existing stream ownership and call lock remain required;
replay results are borrowed until the next runtime call. Counters expose the
retained buffer bytes. Test older graphs after larger and smaller captures,
including mixed episode strides, before admitting this storage optimization.
The benchmark can census a hash-pinned instruction manifest before serving.
It derives initial, ordinary and slow-window projected lengths from the real
tokenizer and the warmup's actual vision feature shape, covering mixed phases
and partial batches with zero-embedding probes. It reads no navigation outcomes
and decodes no actions. Exact-length census supports B1; larger batches require
token buckets. Insufficient KV or graph budget fails startup explicitly.
Large prefills also release each layer's MLP temporaries immediately after its
down projection. The rounded SwiGLU kernel reads gate/up projection slices with
their row strides, avoiding two full contiguous copies while preserving inputs
and every existing rounding boundary. GPU admission checks both exact output
bits and peak allocation on the production 1880-by-18944 projection shape.
The public HTTP adapter forwards the same metadata and up to four chronologically
indexed camera images through the engine transaction. Validate frame names,
count and dimensions before execution; keep legacy single-camera requests and
transport retry semantics intact.

### Strict repeated-chain control

`speculative_tokens=0` disables speculation. A positive block size proposes a
repeat of the already selected action token. One causal forward verifies the
block; each row selects from the complete vocabulary with penalties applied
only to generated tokens and that row's candidate prefix. Commit the root and
the longest matching continuation; reuse the first rejected prediction as the
next root. Ignore physical KV writes beyond the accepted logical length.
Non-action roots use serial decoding. Speculation bypasses the existing
assistant-envelope prefill shortcut: a wider recorded replay found that the
shortcut itself can differ from serial decoding. The existing default path is
unchanged. Verification keeps each projection's serial GEMM shape, uses the
decode attention kernel's FP32 probabilities, and retains the unrounded FP32
residual sum inside fused RMSNorm. Independent row operators run together;
candidate CUDA graphs capture only during startup. Unknown frozen graph shapes
use the same decode operators eagerly. Production speculation requires BF16,
CUDA, Triton and `cuda_graph=True`.

`gpu_preprocess=False` retains the original processor. The opt-in path preserves
the slow SigLIP processor's float-to-uint8 conversion, Pillow's two-pass 22-bit
integer bicubic resize (including intermediate clipping), and float32
normalization. Upload uint8 pixels and use device resize kernels and a 256-entry
normalization lookup table. Preserve the reference's integer-valued float-image
special case. Unsupported image-processor configurations fail explicitly.

`gpu_history_features=False` retains sparse pinned-host visual history. Opting
in keeps owned copies of the same sparse features on CUDA, avoiding host
offload and window-boundary uploads. Never retain a view of a vision graph's
reused output. This trades device memory for transfers; it is independent of
the recurrent KV cache and requires history feature caching.

The grammar tree uses a bounded best-first proposal, avoiding an exhaustive
Cartesian tree. Token selection and generation limits stay unchanged. Torch
bicubic interpolation is unsuitable because its rounding differs from Pillow.

## 5. Model-agnosticism verdict

Speculation uses StreamVLN's token grammar and remains policy-owned. The RGB
resize backend takes only image sizes, coefficients and uint8 tensors. Neither
feature changes engine or session-store contracts.

## 6. Losslessness and precision criterion

GPU preprocessing must be bit-exact against Transformers 4.51.3's slow
SiglipImageProcessor on identical float32 RGB inputs, including binary images,
up/downsampling, identity dimensions and invalid-input rejection. Tree verification is mathematically greedy-equivalent, but batched GEMM,
attention reduction and rounded residual paths can change floating-point
outputs. The strict repeated-chain control retains serial projection shapes
and decode arithmetic; the full profile is separately marked experimental.
Require exact generated tokens/actions in
paired real-weight checks; do not promote a profile with mismatches as lossless.
Record log-probability differences separately. Keep the same BF16 checkpoint,
seed 42, matmul settings, token budget, preprocessing and recurrent schedule.

## 7. Implementation plan

Add processor/factory switches, a policy-owned verifier, backend resize kernels,
benchmark option forwarding and per-path counters. Preserve existing factory
defaults and fallback behavior. Retain the prior bounded causal-mask fix.

## 8. Test plan

CPU tests cover proposal acceptance, mismatch, EOS, repetition history,
capacity and cancellation. CPU/GPU image tests compare exact reference pixels.
Device tests cover graph reuse and mask/storage invariants. Paired real-weight
replays cover the step-32 history transition and more than one episode.

The opt-in `kv_cache_initial_capacity` bounds each quantized eager window's
initial allocation and grows it geometrically up to the unchanged `max_context`.
Reserve the full generation budget plus tree padding before decoding. Growth
copies only the committed prefix into new storage; failure leaves the previous
storage and length intact. Private tree graph staging retains fixed addresses
and maximum capacity, accepting differently sized window caches. GQA tree
partition counts use the original maximum context plan so storage growth does
not change attention reductions, including serial off-grammar fallback. This currently requires quantized tree decoding
with grouped-query attention. Validate both allocation failure and exact GPU
scores/KV across different physical capacities before using this memory profile.

## 9. Benchmark plan

Compare reference, GPU preprocessing, speculation, and both together using the
existing recorded R2R/RxR benchmark. Record hardware/software, checkpoint,
warmup, measured calls, graph counters, accepted tokens, memory and E2E/prefill/
decode timing. Shared-device measurements are exploratory. Candidate misses
may cost more than serial decoding and must remain visible in the counters.

`benchmarks/streamvln-benchmark/serve.py` exposes a frozen B1 configuration through
the versioned HTTP API. Its readiness file records source hash, environment,
configuration, graph captures and startup time. The timer includes decoded CPU
RGB through owned CPU actions; PNG decoding and HTTP transport are excluded.
Complete token/action traces, runtime graph counters and memory peaks accompany
each response. Warmup uses consecutive recorded frames in a private session,
with a declared primitive stride (33 frames at stride four means nine model
calls, including the first 32-primitive reset).
Closed-loop R2R selection, Habitat execution, primitive actions and SR/SPL live
in EmbodiRun's `benchmarks/streamvln-navigation`, with per-episode durable reports.
Compare the same frozen 256-episode manifest and report quantized precision
separately. Sharing a GPU with simulation precludes isolated latency claims.

`serve_batch.py` exposes explicit B=1/2/4/8 profiles through the existing generic
batch scheduler. Stable simulator slots determine row order, while session IDs
own recurrent memory. Each response records the shared tensor invocation ID,
actual occupancy, complete batch elapsed time, graph counters and memory peaks.
Only after every result validates are all session memories committed. A failed
model invocation invalidates the benchmark service, preventing retries against
partially modified scratch. The downstream runner requires a single tensor ID
per synchronous round and retains partial final batches. Summaries count shared
batch elapsed time once, report occupancy and amortized throughput separately,
and omit stage timings that the batch API did not measure.

## 10. Risks and limitations

Rejected candidates add projection work and may erase the speed benefit of
batching other operators. First-time graph/kernel
preparation is excluded from timing; unknown frozen graph shapes fall back to
eager. Rejected KV tails must never be committed. GPU image processing still
validates and converts host observations before transfer. Small images and
low candidate acceptance may show no benefit.
