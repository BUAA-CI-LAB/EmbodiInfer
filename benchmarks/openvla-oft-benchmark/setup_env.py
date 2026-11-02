"""Create an isolated benchmark environment from a verified Thor or RTX 4090 runtime.

This copies packages (never symlinks/hardlinks Python packages to the source env),
then installs the profile's pinned dependencies. The source environment is read-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
RUNTIMES = {
    "thor": ("2.13.0+cu132", [11, 0]),
    "4090": ("2.13.0+cu129", [8, 9]),
}


def validate_runtime(runtime: dict[str, object], platform: str) -> None:
    """Reject a mismatched CUDA build or GPU before creating an environment."""
    version, capability = RUNTIMES[platform]
    if runtime["torch"] != version or runtime["capability"] != capability:
        raise ValueError(f"{platform} requires torch {version} on {capability}; got {runtime}")
    if runtime["device_count"] != 1:
        raise ValueError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES before preparing an environment")


def main() -> None:
    """Prepare a profile-local .venv, retaining the selected working Torch/CUDA wheels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-root", type=Path, required=True, help="existing EmbodiInfer source checkout"
    )
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--platform", choices=tuple(RUNTIMES), default="thor")
    parser.add_argument(
        "--resume", action="store_true", help="retry installation in this script's existing environment"
    )
    args = parser.parse_args()
    root = args.inference_root.resolve(strict=True)
    if not (root / "embodiinfer").is_dir() or not (root / "pyproject.toml").is_file():
        raise ValueError("--inference-root must point to the EmbodiInfer source checkout")
    source_python = args.runtime_python.resolve(strict=True)
    # Keep the executable's original venv path: resolving a python symlink loses sys.prefix.
    source_python = args.runtime_python.absolute()
    probe = subprocess.check_output(
        [
            str(source_python),
            "-c",
            "import json,sysconfig,torch; print(json.dumps(dict(site=sysconfig.get_paths()['purelib'],torch=torch.__version__,cuda=torch.version.cuda,capability=torch.cuda.get_device_capability(),device_count=torch.cuda.device_count(),device_name=torch.cuda.get_device_name())))",
        ],
        text=True,
    )
    runtime = json.loads(probe)
    validate_runtime(runtime, args.platform)
    runtime["platform"] = args.platform
    target = DIRECTORY / ".venv"
    if target.exists() and not args.resume:
        raise FileExistsError(f"environment already exists: {target}; inspect it before replacing")
    new_environment = not target.exists()
    if new_environment:
        subprocess.run([str(source_python), "-m", "venv", str(target)], check=True)
    python = target / "bin/python"
    site = Path(
        subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True
        ).strip()
    )
    source = Path(runtime["site"])
    candidates = list(source.iterdir())
    for path in candidates if new_environment else ():
        if "embodiinfer" in path.name or path.name == "__pycache__":
            continue
        destination = site / path.name
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True, symlinks=False)
        else:
            shutil.copy2(path, destination)
    requirements = DIRECTORY / f"requirements-{args.platform}.txt"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)],
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "-e", str(root)], check=True)
    baseline = subprocess.run([str(source_python), "-m", "pip", "check"], capture_output=True, text=True)
    check = subprocess.run([str(python), "-m", "pip", "check"], capture_output=True, text=True)
    baseline_issues = set(baseline.stdout.splitlines()) if baseline.returncode else set()
    current_issues = set(check.stdout.splitlines()) if check.returncode else set()
    unexpected = current_issues - baseline_issues
    (target / "dependency-check.json").write_text(
        json.dumps(
            {
                "inherited_runtime_issues": sorted(current_issues & baseline_issues),
                "new_issues": sorted(unexpected),
                "pip_check_exit": check.returncode,
                "note": f"{args.platform} dependency exceptions inherited from the supplied runtime; real model validation remains required",
            },
            indent=2,
        )
        + "\n"
    )
    if unexpected:
        raise RuntimeError(f"new dependency conflicts: {sorted(unexpected)}")
    if current_issues:
        print(
            f"Recorded {len(current_issues)} inherited runtime dependency exceptions; see dependency-check.json"
        )
    (target / "runtime-source.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(f"Ready: {python}")


if __name__ == "__main__":
    main()
