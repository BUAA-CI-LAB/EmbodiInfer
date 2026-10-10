"""Compile packaged CUDA sources into a toolchain-specific user cache."""

import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

_SOURCES = Path(__file__).parent / "csrc"


def _compile(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"CUDA compilation failed:\n{error.stderr}") from error


@dataclass(frozen=True)
class NativeLibrary:
    """Loaded C ABI and its actual compilation settings."""

    library: ctypes.CDLL
    compiler: str
    command: list[str]


def cache_directory() -> Path:
    """Return the writable cache, honoring EMBODIINFER_CUDA_CACHE_DIR and XDG_CACHE_HOME."""
    configured = os.environ.get("EMBODIINFER_CUDA_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return root / "embodiinfer" / "cuda"


def build_library(
    name: str, *, strict: bool = False, specific: bool = True, native_fp4: bool = False
) -> NativeLibrary:
    """Compile once per source, compiler and architecture; serialize concurrent builds.

    Args:
        name: Packaged CUDA source stem.
        strict: Disable fused multiply/add to preserve the validated rounding order.
        specific: Use an architecture-specific target when the GPU supports it.
        native_fp4: Verify native FP4 conversions in the emitted PTX.

    Returns:
        Loaded library with its compiler version and command.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Initialize CUDA operators before graph capture")
    if name not in {path.stem for path in _SOURCES.glob("*.cu")}:
        raise ValueError(f"Unknown packaged CUDA source: {name}")
    cuda_home = os.environ.get("CUDA_HOME")
    compiler_path = str(Path(cuda_home) / "bin/nvcc") if cuda_home else shutil.which("nvcc")
    compiler_path = compiler_path or "/usr/local/cuda/bin/nvcc"
    try:
        compiler = subprocess.check_output([compiler_path, "--version"], text=True).strip()
    except FileNotFoundError as error:
        raise RuntimeError("CUDA compilation requires nvcc; set CUDA_HOME to the CUDA toolkit") from error
    major, minor = torch.cuda.get_device_capability()
    if native_fp4 and major < 10:
        raise RuntimeError("Native FP4 conversions require a Blackwell CUDA device")
    # Ada/Ampere have no architecture-specific 'a' compilation target.
    architecture = f"sm_{major}{minor}{'a' if specific and major >= 9 else ''}"
    flags = ["-O3", "-std=c++17", f"-arch={architecture}"]
    if strict:
        flags.append("--fmad=false")
    digest = hashlib.sha256((compiler_path + compiler + repr(flags)).encode())
    for source in sorted(_SOURCES.glob("*.cu")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    directory = cache_directory() / f"{architecture}-{digest.hexdigest()[:20]}"
    directory.mkdir(parents=True, exist_ok=True)
    source = _SOURCES / f"{name}.cu"
    target = directory / f"{name}.so"
    command = [
        compiler_path,
        *flags,
        "--shared",
        "-Xcompiler=-fPIC",
        str(source),
        "-o",
        str(target),
    ]
    with (directory / f"{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            if not target.exists():
                built = Path(temporary) / target.name
                _compile([*command[:-1], str(built)])
                built.replace(target)
            if native_fp4:
                ptx = target.with_suffix(".ptx")
                if not ptx.exists():
                    built = Path(temporary) / ptx.name
                    _compile([compiler_path, *flags, "-ptx", str(source), "-o", str(built)])
                    built.replace(ptx)
                if "cvt.rn.satfinite.e2m1x2" not in ptx.read_text():
                    raise RuntimeError(f"{name} did not emit native FP4 conversion")
    return NativeLibrary(ctypes.CDLL(str(target)), compiler, command)
