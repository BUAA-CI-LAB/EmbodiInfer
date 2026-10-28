"""Resolve a device profile for the downstream 256-episode navigation benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parent
REPOSITORY = BENCHMARK.parents[1]


def main(directory: Path) -> None:
    """Freeze one device/dataset/precision configuration and start its HTTP service."""
    parser = argparse.ArgumentParser(description=__doc__)
    precisions = ("bf16", "int8") if directory.name == "orin-nx" else ("bf16",)
    parser.add_argument("precision", choices=precisions)
    parser.add_argument("--dataset", choices=("r2r", "rxr"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, required=True, help="Recorded R2R data for startup warmup only"
    )
    parser.add_argument("--output", type=Path, required=True, help="New service output directory")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index visible to this SSH instance")
    parser.add_argument("--port", type=int, default=18767)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (args.precision == "int8") != (args.calibration is not None):
        parser.error("INT8 requires --calibration; omit it for BF16")
    if args.gpu < 0 or not 1 <= args.port <= 65535:
        parser.error("GPU must be nonnegative and port must be in [1, 65535]")
    config = json.loads((directory / "config.json").read_text())
    profile = config.pop("int8_profile", None)
    selection = config["navigation"]["selections"][args.dataset]
    manifest = args.manifest.expanduser().resolve(strict=True)
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != selection["sha256"]:
        raise ValueError("Manifest differs from the frozen scene-proportional 256-episode selection")
    episodes = json.loads(raw)["episodes"]
    if len(episodes) != 256 or len({str(row["episode_id"]) for row in episodes}) != 256:
        raise ValueError("Manifest must contain 256 distinct episodes")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    data = args.data_root.expanduser().resolve(strict=True)
    annotations = data / "annotations_v1-3.json"
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError("Checkpoint must contain config.json")
    records = json.loads(annotations.read_text())
    selected = sorted(records, key=lambda row: (int(row["id"]), row["video"]))[:48]
    identities = [{"episode_id": int(row["id"]), "frames": len(row["actions"])} for row in selected]
    digest = hashlib.sha256(json.dumps(identities, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if digest != config["datasets"][0]["selection_sha256"]:
        raise ValueError("Recorded warmup data differs from the pinned R2R selection")
    calibration = None
    if args.calibration is not None:
        calibration = args.calibration.expanduser().resolve(strict=True)
        if hashlib.sha256(calibration.read_bytes()).hexdigest() != profile["calibration_sha256"]:
            raise ValueError("Calibration differs from the pinned W8A8 profile")
        config["precision_experiment"] = profile
    output = args.output.expanduser().resolve()
    config.update(
        checkpoint=str(checkpoint),
        output_dir=str(output),
        evaluation_dataset=args.dataset,
        navigation_manifest=str(output / f"selected-{args.dataset}-navigation-256.json"),
        physical_gpu=args.gpu,
    )
    config["datasets"][0].update(root=str(data), annotations=str(annotations))
    output.mkdir(parents=True, exist_ok=False)
    Path(config["navigation_manifest"]).write_bytes(raw)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    token = secrets.token_urlsafe(32)
    with open(
        output / "service-token", "x", opener=lambda path, flags: os.open(path, flags, 0o600)
    ) as stream:
        stream.write(token + "\n")
    if args.prepare_only:
        print(f"Prepared {config_path}; no model or CUDA execution")
        return
    environment = dict(os.environ)
    environment.pop("PYTORCH_ALLOC_CONF", None)
    environment.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONPATH=os.pathsep.join(
            filter(None, (str(REPOSITORY), str(BENCHMARK), environment.get("PYTHONPATH")))
        ),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS=str(config["cpu_threads"]),
        OPENBLAS_NUM_THREADS=str(config["cpu_threads"]),
        MKL_NUM_THREADS=str(config["cpu_threads"]),
        PYTORCH_CUDA_ALLOC_CONF=config["allocator_config"],
        ACTIVEVLN_EVICT_CHECKPOINT="1",
        ACTIVEVLN_BENCHMARK_TOKEN=token,
    )
    cuda_include = Path(environment.get("CUDA_HOME", "/usr/local/cuda")) / "targets/aarch64-linux/include"
    if directory.name == "orin-nx" and cuda_include.is_dir():
        environment["CPATH"] = os.pathsep.join(filter(None, (str(cuda_include), environment.get("CPATH"))))
    command = [
        sys.executable,
        "-u",
        str(BENCHMARK / "orin-nx/serve.py"),
        "--config",
        str(config_path),
        "--source",
        str(REPOSITORY),
        "--ready-file",
        str(output / "service-ready.json"),
        "--port",
        str(args.port),
        "--max-wait-ms",
        "0",
    ]
    if config["streamed_loading"]:
        command.append("--streamed-loading")
    if calibration is not None:
        command.extend(["--calibration", str(calibration)])
    os.execve(sys.executable, command, environment)


if __name__ == "__main__":
    raise SystemExit("Use orin-nx/run.py or rtx-4090/run.py")
