#!/bin/bash
cd /home/qy/lxygit/activevln-benchmark-review/EmbodiInfer-opt
export PYTHONPATH=.
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7
PY=../EmbodiInfer/.venv/bin/python
for job in "r2r b2 r2r-b2" "r2r b4 r2r-b4" "rxr b2 rxr-b2"; do
  set -- $job
  ds=$1; b=$2; cfg=$3
  out=benchmarks/activevln-benchmark/runs/opt-full-$ds-$b-fix/report.json
  mkdir -p "$(dirname "$out")"
  echo "=== POSTFIX $ds $b start $(date) ==="
  $PY benchmarks/activevln-benchmark/benchmark_batch.py --config benchmarks/activevln-benchmark/$cfg.json --output "$out" > ../logs/opt-$ds-$b-fix.log 2>&1
  echo "=== POSTFIX $ds $b rc=$? $(date) ==="
done
echo "=== postfix all done ==="
