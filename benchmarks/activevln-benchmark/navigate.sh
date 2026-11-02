#!/usr/bin/env bash
# Run on the simulator host after copying the service metadata and episode manifest.
set -euo pipefail
: "${EMBODIRUN:?Set EMBODIRUN to the checkout containing activevln-navigation}"
: "${HABITAT_PYTHON:?Set HABITAT_PYTHON to the existing Habitat interpreter}"
: "${SCENES:?Set SCENES to scene_datasets}"
if [[ $# != 4 ]]; then
    echo "Usage: bash navigate.sh DATASET RUN_ROOT ENDPOINT OUTPUT" >&2
    exit 2
fi
dataset=$1
run_root=$2
endpoint=$3
output=$4
python - "$dataset" "$run_root" <<'CHECK'
import hashlib
import json
import sys
from pathlib import Path
dataset, root = sys.argv[1], Path(sys.argv[2])
config = json.loads((root / "config.json").read_text())
ready = json.loads((root / "service-ready.json").read_text())
if dataset not in ("r2r", "rxr") or config["evaluation_dataset"] != dataset:
    raise ValueError("Dataset differs from the model service configuration")
manifest = root / f"selected-{dataset}-navigation-256.json"
expected = config["navigation"]["selections"][dataset]["sha256"]
if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
    raise ValueError("Episode selection changed")
if ready["navigation_manifest_sha256"] != expected or ready["config"] != config:
    raise ValueError("Service configuration or shape prewarm differs")
if not (root / "service-token").read_text().strip():
    raise ValueError("Empty service token")
CHECK
export PYTHONPATH="$EMBODIRUN/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -u "$EMBODIRUN/benchmarks/activevln-navigation/benchmark.py" \
    --root "$run_root" --config "$run_root/config.json" \
    --habitat-python "$HABITAT_PYTHON" --scenes "$SCENES" --output "$output" \
    --endpoint "$endpoint" --service-ready "$run_root/service-ready.json" \
    --token-file "$run_root/service-token" --datasets "$dataset" --episode-count 256 \
    --batch-size 1 --simulator-gpu "${SIMULATOR_GPU:-0}"
