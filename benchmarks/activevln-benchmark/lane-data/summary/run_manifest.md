# ActiveVLN benchmark run manifest (2026-09-25)

## Final matrix

See `summary_table.md`. Raw JSON in `results/`, run logs in `results/logs/`.

## Protocol

- Dataset: recorded RGB trajectories, **not** closed-loop rollout (no SR/SPL).
- Selection (both ends identical): first 8 annotations sorted by `(int(id), video)`,
  first 12 recorded frames each → 96 requests/repeat; 3 repeats; mean ± std reported.
- A sample = one frame observation → one model request.
  `samples/s = completed requests / measured phase wall clock` (not mean of inverse latencies).
- Generation (both ends): greedy `temperature=0`, `top_p=1`, `repetition_penalty=1.05`,
  `max_new_tokens=512`, stop at EOS or length cap. EmbodiInfer `stop_phrase_early_stop=False`
  so both ends generate to EOS (no adapter stop-phrase early exit).
- Warmup and model load are excluded. One backend at a time; no overlapping GPU timing work.
- Resize (both ends): official client `smart_resize(max_pixels=76800, factor=28)`.
- BF16 weights both ends.

## Measured boundaries (disclosed)

| | native vLLM | EmbodiInfer |
|---|---|---|
| samples/s wall | JPEG read/decode + resize + PNG + base64 + HTTP + server queue + generation | JPEG read/decode + resize + tensor + model call |
| request latency | HTTP issued → response (server image preprocess + queue + prefill + decode) | CUDA-synced `encode_prefix(_batch)+decode(_batch)` (Qwen processor CPU + vision + prefill + decode) |
| batch semantics | B concurrent HTTP requests → vLLM continuous batching | B independent episode sessions in one true tensor batch, one forward per step; B=1 serial single-session |

## Exact run commands (on lenovo-ThinkStation-P8, remote root `/home/qy/lxygit/activevln-benchmark-review`)

Native vLLM server (one model at a time, port 8003):

```bash
bash scripts/vllm_serve.sh        # R2R checkpoint
bash scripts/vllm_serve_rxr.sh    # RxR checkpoint
```

Native vLLM runs (client; server startup excluded from timing):

```bash
EmbodiInfer/.venv/bin/python EmbodiInfer/benchmarks/activevln-benchmark/vllm_benchmark.py \
  --dataset r2r --episodes 8 --frames 12 --batch 1 --repeats 3 --output logs/vllm-r2r-b1.json
EmbodiInfer/.venv/bin/python EmbodiInfer/benchmarks/activevln-benchmark/vllm_benchmark.py \
  --dataset r2r --episodes 8 --frames 12 --batch 2,4,8 --repeats 3 --output logs/vllm-r2r-batches.json
EmbodiInfer/.venv/bin/python EmbodiInfer/benchmarks/activevln-benchmark/vllm_benchmark.py \
  --dataset rxr --episodes 8 --frames 12 --batch 1,2,4,8 --repeats 3 --output logs/vllm-rxr-batches.json
```

EmbodiInfer runs:

```bash
EmbodiInfer/.venv/bin/python EmbodiInfer/benchmarks/activevln-benchmark/activevln_benchmark.py \
  --dataset r2r --episodes 8 --frames 12 --batch 1 --repeats 3 --output-dir logs
EmbodiInfer/.venv/bin/python EmbodiInfer/benchmarks/activevln-benchmark/activevln_benchmark.py \
  --dataset r2r --episodes 8 --frames 12 --batch 2 --repeats 3 --output-dir logs
EmbodiInfer/.venv/bin/python EmbodiInfer/benchmarks/activevln-benchmark/activevln_benchmark.py \
  --dataset r2r --episodes 8 --frames 12 --batch 4,8 --repeats 3 --output-dir logs
EmbodiInfer/.venv/bin/python EmbodiInfer/benchmarks/activevln-benchmark/activevln_benchmark.py \
  --dataset rxr --episodes 8 --frames 12 --batch 1 --repeats 3 --output-dir logs
EmbodiInfer/.venv/bin/python EmbodiInfer/benchmarks/activevln-benchmark/activevln_benchmark.py \
  --dataset rxr --episodes 8 --frames 12 --batch 2,4,8 --repeats 3 --output-dir logs
```

## Environment (measured versions, not reinstalled)

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.95.05; CUDA 13.0.
- Native vLLM env (`vllm-env`): vLLM **0.30.0**, torch **2.13.0+cu130**.
  Disclosure: official ActiveVLN pins vLLM 0.8.5.post1, which has no sm_120 kernels;
  results are for vLLM 0.30.0, not a verbatim old-version official reproduction.
  Server flags: `--runner generate --trust-remote-code --limit-mm-per-prompt '{"image":200,"video":0}'`
  `--mm-processor-kwargs '{"max_pixels":80000}' --max-model-len 32768 --enable-prefix-caching`
  `--gpu-memory-utilization 0.85 --no-enable-log-requests --port 8003`.
- EmbodiInfer env (`EmbodiInfer/.venv`): torch **2.10.0+cu130**, transformers **4.51.3**,
  numpy 2.5.2, Pillow 12.3.0, Python 3.12.13.
- Checkpoints: R2R `/home/qy/models/activevln/Qwen2.5-VL-3B_rl_r2r_4000`;
  RxR `/home/qy/models/activevln/Qwen2.5-VL-3B_rl_rxr_4000_step350`
  (RxR action space uses the 30/60/90-degree prompt; R2R 15/30/45).

## Known divergences / limitations (do not block the perf numbers)

1. At B1, vLLM vs EmbodiInfer output texts: R2R 69/96 identical, RxR 53/96 identical;
   total output tokens R2R 1961 vs 1961, RxR 2064 vs 2066. Differences start as greedy
   bf16 near-tie flips and compound through history; the measured request set is identical,
   generated-token workloads are within 0.1%.
2. EmbodiInfer batch paths (B≥2) vs its serial path: 30-45% of frame texts differ
   (padded-batch kernels vs single-session kernels, greedy near-ties). Token totals match.
3. vLLM latency includes HTTP + client/server serialization; EmbodiInfer runs in-process
   (no HTTP, no base64). EmbodiInfer latency includes CPU Qwen preprocessing inside
   `encode_prefix`; both include resize in wall but not in latency.
4. vLLM `gpu_memory_used_mib_max` is device-level usage with `--gpu-memory-utilization 0.85`
   pre-allocation (~30.1GB); EmbodiInfer numbers are actual device allocations
   (`cuda_peak_allocated` 7.7-13.4GB). Not directly comparable memory metrics.
5. Prefix caching makes vLLM's first repeat of some cells slower (e.g. R2R B2 r0 10.69 vs
   11.21/11.60); EmbodiInfer rebuilds sessions every repeat. Stats include that effect.
6. RxR B8 EmbodiInfer is near the 32GB limit (device used 30.7GB, reserved 27.5GB):
   no OOM, but scaling stalls (7.7 → 8.6 samples/s); reported as measured.
7. The older ~3.09 samples/s EmbodiInfer number used the official sampling profile
   (temp 0.2/top_p 0.8) on 2997 calls and a different timing contract; it is not the
   baseline for the ratios here.
8. Older result `logs/vllm-b1.json` (8 requests, old script) is preserved but superseded
   by `results/vllm-r2r-b1.json`.
