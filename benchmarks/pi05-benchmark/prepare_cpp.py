"""Convert the benchmark's exact LeRobot checkpoint with pinned C++ converters."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    """Hash the file contents without loading the entire checkpoint into RAM."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_checkpoint(source: Path, target: Path) -> dict:
    """Remove only the LeRobot policy's model. namespace; retain every tensor bit."""
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / "conversion-source.json"
    weights = source / "model.safetensors"
    if manifest.exists():
        record = json.loads(manifest.read_text())
        if (
            record["source"] != str(source)
            or record["source_sha256"] != sha256(weights)
            or record["converted_sha256"] != sha256(target / "model.safetensors")
        ):
            raise ValueError("existing canonical checkpoint does not match its source manifest")
        return record
    if any(target.iterdir()):
        raise ValueError("canonical checkpoint directory contains an incomplete conversion")
    tensors = {}
    tensor_records = []
    with safe_open(str(weights), framework="pt") as handle:
        for key in handle.keys():  # noqa: SIM118 -- safetensors reader is not a dict/iterable.
            name = key.removeprefix("model.")
            if name in tensors:
                raise ValueError(f"duplicate converted tensor: {name}")
            value = handle.get_tensor(key)
            tensors[name] = value
            tensor_records.append(
                {
                    "source": key,
                    "target": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
                }
            )
    save_file(tensors, str(target / "model.safetensors"))
    for path in source.iterdir():
        if path.is_file() and path.name != "model.safetensors" and not (target / path.name).exists():
            (target / path.name).symlink_to(path.resolve())
    record = {
        "source": str(source),
        "source_sha256": sha256(weights),
        "converted_sha256": sha256(target / "model.safetensors"),
        "transform": "strip model. namespace only; no dtype/value changes",
        "tensors": tensor_records,
    }
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    """Convert the same weights and statistics, then record all artifact hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("vlacpp", "embodied"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.engine == "vlacpp":
        canonical = output / "checkpoint"
        record = canonical_checkpoint(args.checkpoint.resolve(), canonical)
    else:
        # Embodied.cpp already resolves both native and unprefixed LeRobot keys.
        canonical = args.checkpoint.resolve()
        record = {"source_sha256": sha256(canonical / "model.safetensors")}
    scripts = args.source.resolve() / "scripts"
    gguf = output / f"pi05-{args.engine}.gguf"
    if gguf.exists():
        raise FileExistsError(gguf)
    sys.path.insert(0, str(scripts))
    if args.engine == "vlacpp":
        # Upstream accepts complete dataset statistics, but hardcodes QUANTILES
        # in its GGUF metadata. Preserve this checkpoint's explicit MEAN_STD mode.
        statistics = {}
        for metadata, registry, feature in (
            ("policy_preprocessor.json", "normalizer_processor", "observation.state"),
            ("policy_postprocessor.json", "unnormalizer_processor", "action"),
        ):
            steps = json.loads((canonical / metadata).read_text())["steps"]
            step = next(step for step in steps if step["registry_name"] == registry)
            kind = "STATE" if feature == "observation.state" else "ACTION"
            if step["config"]["norm_map"][kind] != "MEAN_STD":
                raise ValueError("this conversion profile requires checkpoint MEAN_STD statistics")
            with safe_open(str(canonical / step["state_file"]), framework="pt") as handle:
                statistics[feature] = {
                    name: handle.get_tensor(f"{feature}.{name}").tolist()
                    for name in ("mean", "std", "q01", "q99")
                }
        stats_path = output / "checkpoint-statistics.json"
        stats_path.write_text(json.dumps(statistics, indent=2) + "\n")
        spec = importlib.util.spec_from_file_location(
            "upstream_pi05_converter", scripts / "convert_pi05_to_gguf.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = module.write_pi_kv

        def metadata(writer, key, config, adarms=False):
            original(writer, key, config, adarms=adarms)
            writer.add_string(key("norm_mode"), "mean_std")

        module.write_pi_kv = metadata
        sys.argv = [
            str(scripts / "convert_pi05_to_gguf.py"),
            "--ckpt",
            str(canonical),
            "--out",
            str(gguf),
            "--dataset-stats",
            str(stats_path),
        ]
        if module.main() != 0:
            raise RuntimeError("upstream converter failed")
        print("Applied checkpoint normalization: MEAN_STD (overrides upstream quantile default)", flush=True)
    else:
        subprocess.run(
            [
                sys.executable,
                str(scripts / "convert_pi05_to_gguf.py"),
                "--ckpt",
                str(canonical),
                "--out",
                str(gguf),
                "--outtype",
                "bf16",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(scripts / "convert_pi05_mmproj_to_gguf.py"),
                "--ckpt",
                str(canonical),
                "--out",
                str(output / "pi05-mmproj.gguf"),
                "--outtype",
                "bf16",
            ],
            check=True,
        )
    summary = {
        "engine": args.engine,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.source, text=True
        ).strip(),
        "checkpoint_source_sha256": record["source_sha256"],
        "normalization": "MEAN_STD",
        "artifacts": {p.name: sha256(p) for p in output.glob("*.gguf")},
    }
    (output / "conversion.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
