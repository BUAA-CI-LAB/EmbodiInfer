#!/bin/bash
cd /home/qy/lxygit/activevln-benchmark-review/EmbodiInfer-opt
export PYTHONPATH=.
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7
PY=../EmbodiInfer/.venv/bin/python
for b in 2 4 8; do
  out=benchmarks/activevln-benchmark/runs/opt-full-r2r-b$b/report.json
  mkdir -p "$(dirname "$out")"
  echo "=== R2R B$b start $(date) ==="
  $PY benchmarks/activevln-benchmark/benchmark_batch.py --config benchmarks/activevln-benchmark/r2r-b$b.json --output "$out" > ../logs/opt-r2r-b$b.log 2>&1
  echo "=== R2R B$b rc=$? $(date) ==="
done
echo "=== all done ==="
