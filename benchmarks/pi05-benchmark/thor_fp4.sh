#!/usr/bin/env bash
# The Thor nvfp4-nvfp4 latency in one command: RUNS fresh processes of FlashRT's examples/thor/pi05.py
# harness with EmbodiInfer swapped in (flashrt_harness.py), each with WARMUP warmup and ITERS timed calls,
# then every run's P50 and their mean.
#
# Required: FLASHRT_ROOT (FlashRT v0.2.0 checkout), CHECKPOINT (RLinf-Pi05-LIBERO-SFT in LeRobot layout),
# NORM_STATS (its physical-intelligence/libero/norm_stats.json), EMBODIINFER_CUTLASS_DIR, EMBODIINFER_FA4_DIR.
# Optional: PYTHON (python3), PRESET (nvfp4-nvfp4), RUNS (3), WARMUP (200), ITERS (1000),
# OUT (results/pi05-thor-<preset>).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${FLASHRT_ROOT:?set FLASHRT_ROOT to a FlashRT v0.2.0 checkout}"
: "${CHECKPOINT:?set CHECKPOINT to RLinf-Pi05-LIBERO-SFT in LeRobot layout}"
: "${NORM_STATS:?set NORM_STATS to physical-intelligence/libero/norm_stats.json of the checkpoint}"
: "${EMBODIINFER_CUTLASS_DIR:?set EMBODIINFER_CUTLASS_DIR to a CUTLASS 4.4 checkout}"
: "${EMBODIINFER_FA4_DIR:?set EMBODIINFER_FA4_DIR to the FlashAttention-4 package}"
PYTHON=${PYTHON:-python3}
PRESET=${PRESET:-nvfp4-nvfp4}
RUNS=${RUNS:-3}
OUT=${OUT:-results/pi05-thor-$PRESET}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} EMBODIINFER_CUTLASS_DIR EMBODIINFER_FA4_DIR

gpu=/sys/class/devfreq/gpu-gpc-0
if [ -r "$gpu/min_freq" ] && [ "$(cat "$gpu/min_freq")" != "$(cat "$gpu/max_freq")" ]; then
  echo "warning: GPU clocks are not locked (sudo nvpmodel -m 0 && sudo jetson_clocks); latency will vary" >&2
fi
mkdir -p "$OUT"
for i in $(seq 1 "$RUNS"); do
  "$PYTHON" "$here/flashrt_harness.py" --flashrt-root "$FLASHRT_ROOT" --checkpoint "$CHECKPOINT" \
    --norm-stats "$NORM_STATS" --preset "$PRESET" --warmup "${WARMUP:-200}" --iters "${ITERS:-1000}" \
    --output "$OUT/run_$i.npz" > "$OUT/run_$i.log" 2>&1 || { tail -n 20 "$OUT/run_$i.log" >&2; exit 1; }
  echo "run $i/$RUNS: $(tail -n 1 "$OUT/run_$i.log")"
done
"$PYTHON" -c '
import json, sys
from pathlib import Path
out, count = Path(sys.argv[1]), int(sys.argv[2])
runs = [json.loads((out / f"run_{i}.json").read_text()) for i in range(1, count + 1)]
p50 = [run["latency_ms"] for run in runs]
p95 = max(run["p95_ms"] for run in runs)
mean = sum(p50) / len(p50)
values = " / ".join(f"{value:.3f}" for value in p50)
print(f"P50 per run: {values} ms; mean {mean:.2f} ms ({1000 / mean:.1f} Hz); max P95 {p95:.2f} ms")
' "$OUT" "$RUNS"
