# Parallelism

Data parallelism runs independent requests on full-model replicas. Tensor
parallelism splits one model's computation across devices. They address
different constraints: aggregate throughput versus the cost and memory of one
logical request.

Configure replicas and multi-GPU execution through the Python APIs below.

## Data-parallel replicas

Construct a full-model replica on each device and pass the replicas to
`DataParallelEngine`. Stateless requests are distributed across healthy
replicas. Recurrent requests must provide a `SessionKey`: the first turn selects a
replica, and later turns remain pinned there until `reset_sessions()` clears the state
and releases the affinity. Turns within one recurrent session remain B=1, while
independent episodes can execute concurrently across devices.

Recurrent state is replica-local, so failed recurrent requests cannot migrate or retry
on another replica. `cancel_sessions()` is routed to the owning replica but does not
release affinity.

```python
from embodiinfer import DataParallelEngine
from embodiinfer.types import SessionKey

dp = DataParallelEngine([core_on_cuda0, core_on_cuda1])
sessions = [SessionKey("env-0", "ep-0"), SessionKey("env-1", "ep-1")]
chunks = dp.execute([obs0, obs1], session_ids=sessions)
dp.reset_sessions(sessions)
```

Here `core_on_cuda0` and `core_on_cuda1` are already constructed engine cores,
each with its own model, and `obs0` and `obs1` belong to independent episodes.
For a complete stateless π0.5 example, see
[`examples/pi05_inference.py`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/examples/pi05_inference.py),
which constructs one replica per GPU when invoked with `--gpus 2`.

For recurrent policies, use one session at a time per replica and distribute
independent episodes across replicas. Async/pipeline execution, group sampling,
and best-of-N currently support stateless policies only.

## Two-GPU data/tensor parallel results

Measured on two NVIDIA A800 80GB PCIe GPUs (`SYS` topology, no NVLink), using BF16
eager execution, batch size one, two warmup iterations, and ten measured logical
samples. DP throughput is aggregate throughput from two full-model replicas; TP
throughput is one logical request sharded across both GPUs. CUDA Graph was disabled
for this comparison.

The three navigation policies use the same real RGB observation from the
R2R-VLNCE-aligned benchmark input. Pi0.5 uses a deterministic three-camera,
48-token input and its standard 10-step
denoising loop.

| policy | single GPU samples/s | DP=2 samples/s | DP speedup | TP=2 samples/s | TP / single |
|---|---:|---:|---:|---:|---:|
| Qwen R2R Low | 12.440 | 19.955 | 1.604x | 10.583 | 0.851x |
| Qwen R2R Panoramic | 4.639 | 8.407 | 1.812x | 3.849 | 0.830x |
| NaViDA | 1.614 | 2.248 | 1.393x | 1.304 | 0.808x |
| Pi0.5 | 3.969 | 6.817 | 1.717x | 3.370 | 0.849x |

All navigation DP/TP outputs matched the single-GPU action outputs exactly for all
ten samples. Pi0.5 DP also matched exactly; Pi0.5 TP matched both ranks exactly and
matched the single-GPU reference within `rtol=0.02`, `atol=0.05`
(`max_abs=0.015625`). On this batch-one, non-NVLink setup, DP improves aggregate
throughput by 1.39-1.81x, while TP communication overhead makes TP 15-19% slower than
a single GPU.
