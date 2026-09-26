#!/bin/bash
cd /home/qy/lxygit/activevln-benchmark-review/EmbodiInfer-opt
export PYTHONPATH=.
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7
PY=../EmbodiInfer/.venv/bin/python
echo "=== RxR smoke start $(date) ==="
$PY benchmarks/activevln-benchmark/benchmark.py --config benchmarks/activevln-benchmark/rxr-b1-smoke.yaml > ../logs/opt-rxr-smoke.log 2>&1
echo "=== RxR smoke rc=$? ==="
echo "=== RxR B1 full start $(date) ==="
$PY benchmarks/activevln-benchmark/benchmark.py --config benchmarks/activevln-benchmark/rxr-b1-full.yaml > ../logs/opt-rxr-b1.log 2>&1
echo "=== RxR B1 full rc=$? $(date) ==="
for b in 2 4 8; do
  out=benchmarks/activevln-benchmark/runs/opt-full-rxr-b$b/report.json
  mkdir -p "$(dirname "$out")"
  echo "=== RxR B$b start $(date) ==="
  $PY benchmarks/activevln-benchmark/benchmark_batch.py --config benchmarks/activevln-benchmark/rxr-b$b.json --output "$out" > ../logs/opt-rxr-b$b.log 2>&1
  echo "=== RxR B$b rc=$? $(date) ==="
done
echo "=== rxr all done ==="
