"""Copy a working model environment into this benchmark's independent .venv."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    """Preserve the machine's working CUDA wheels without modifying the source env."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    args = parser.parse_args()
    source_python = args.runtime_python.absolute()
    root = args.inference_root.resolve(strict=True)
    if not (root / "embodiinfer" / "quantization").is_dir():
        raise ValueError("--inference-root must contain the quantization branch")
    target = Path(__file__).resolve().parent / ".venv"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing environment: {target}")
    probe = (
        "import json,sys,sysconfig,importlib.metadata as m; "
        "print(json.dumps(dict(python=sys.version,site=sysconfig.get_paths()['purelib'],"
        "sites=[p for p in sys.path if p.endswith('site-packages')],"
        "packages={name:m.version(name) for name in ('torch','transformers','numpy','Pillow')},"
        "distribution_versions={d.metadata['Name']:m.version(d.metadata['Name']) "
        "for d in m.distributions() if d.metadata['Name']})))"
    )
    runtime = json.loads(subprocess.check_output([str(source_python), "-c", probe], text=True))
    subprocess.run([str(source_python), "-m", "venv", str(target)], check=True)
    python = target / "bin" / "python"
    site = Path(
        subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True
        ).strip()
    )
    for source_site in reversed(runtime["sites"]):
        for path in sorted(Path(source_site).iterdir()):
            if (
                "embodiinfer" in path.name
                or "embodiinfer" in path.name
                or path.name.startswith("__editable__")
                or path.name == "__pycache__"
            ):
                continue
            destination = site / path.name
            if path.is_dir():
                # Replace normal packages so stale extension modules cannot win imports.
                # Namespace packages (for example nvidia) may span multiple sites.
                if (path / "__init__.py").exists() and destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(path, destination, dirs_exist_ok=True, symlinks=False)
            else:
                shutil.copy2(path, destination)
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        expected = runtime["distribution_versions"].get(distribution.metadata.get("Name"))
        if expected is not None and distribution.version != expected:
            shutil.rmtree(distribution._path)
    (site / "quantization-inference-source.pth").write_text(str(root) + "\n")
    runtime.update(source_python=str(source_python), inference_root=str(root))
    (target / "runtime-source.json").write_text(json.dumps(runtime, indent=2) + "\n")
    imported = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import json,torch,transformers; print(json.dumps(dict(torch=torch.__file__,transformers=transformers.__file__)))",
            ],
            text=True,
        )
    )
    if any(not Path(path).resolve().is_relative_to(target) for path in imported.values()):
        raise RuntimeError(f"copied environment still imports packages from outside .venv: {imported}")
    check = subprocess.run([str(python), "-m", "pip", "check"], capture_output=True, text=True)
    (target / "dependency-check.json").write_text(
        json.dumps(
            {
                "exit_code": check.returncode,
                "stdout": check.stdout,
                "stderr": check.stderr,
                "note": "Copied model environment; inherited dependency conflicts require runtime validation.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Ready: {python}", flush=True)


if __name__ == "__main__":
    main()
