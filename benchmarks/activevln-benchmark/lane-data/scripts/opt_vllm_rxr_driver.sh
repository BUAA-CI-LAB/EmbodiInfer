#!/bin/bash
cd /home/qy/lxygit/activevln-benchmark-review/EmbodiInfer-opt
export PYTHONPATH=.
export PATH=/home/qy/lxygit/activevln-benchmark-review/vllm-env/bin:$PATH
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_V1_MULTIPROCESSING=0
PY=../vllm-env/bin/python
out1=benchmarks/activevln-benchmark/runs/opt-vllm-rxr-b1/report.json
mkdir -p "$(dirname "$out1")"
echo "=== vLLM RxR B1 start $(date) ==="
$PY benchmarks/activevln-benchmark/benchmark_vllm_modern.py --config benchmarks/activevln-benchmark/rxr-vllm-b1.json --output "$out1" > ../logs/opt-vllm-rxr-b1.log 2>&1
echo "=== vLLM RxR B1 rc=$? $(date) ==="
for b in 2 4 8; do
  out=benchmarks/activevln-benchmark/runs/opt-vllm-rxr-b$b/report.json
  mkdir -p "$(dirname "$out")"
  echo "=== vLLM RxR B$b start $(date) ==="
  $PY benchmarks/activevln-benchmark/benchmark_vllm_batch.py --config benchmarks/activevln-benchmark/rxr-b$b.json --output "$out" > ../logs/opt-vllm-rxr-b$b.log 2>&1
  echo "=== vLLM RxR B$b rc=$? $(date) ==="
done
echo "=== vllm rxr all done ==="
