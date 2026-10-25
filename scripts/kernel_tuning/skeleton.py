"""Build a randomly initialized copy of a Hugging Face checkpoint for shape capture.

Kernel shapes depend on the model configuration and its inputs, never on weight
values. This downloads every small file of a repository (configs, tokenizer,
processor) and, for each ``*.safetensors`` file, reads only its header with an
HTTP range request and writes a file with the same tensor names, dtypes, and
shapes filled with small random values. Model loaders then run unchanged.

The result is for measuring shapes only; its outputs are meaningless. Decode
lengths of generative models follow the random logits, so cap them in the
capture configuration.

Run in a runtime with Torch and safetensors:
``python -m scripts.kernel_tuning skeleton REPO --revision REV --output DIR``
"""

from __future__ import annotations

import json
import os
import struct
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts import ContractError, relative_path

#: Files larger than this are weights or data, never configuration.
SMALL_FILE_BYTES = 64 * 1024 * 1024
SAFETENSORS_DTYPES = {
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "F64": "float64",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _request(url: str, *, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"User-Agent": "embodiinfer-kernel-tuning/1"}
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        return response.read()


def list_files(repo: str, revision: str) -> list[dict[str, Any]]:
    """Every file of a model repository at a revision, with sizes."""
    url = f"{_endpoint()}/api/models/{repo}/tree/{revision}?recursive=true"
    return [entry for entry in json.loads(_request(url)) if entry["type"] == "file"]


def file_url(repo: str, revision: str, path: str) -> str:
    """Resolve URL of one repository file."""
    return f"{_endpoint()}/{repo}/resolve/{revision}/{path}"


def read_header(repo: str, revision: str, path: str) -> dict[str, Any]:
    """Read a safetensors header (names, dtypes, shapes) without the tensor data."""
    url = file_url(repo, revision, path)
    (size,) = struct.unpack("<Q", _request(url, start=0, end=7))
    if size > SMALL_FILE_BYTES:
        raise ContractError(f"{path}: implausible safetensors header of {size} bytes")
    return json.loads(_request(url, start=8, end=7 + size))


def _initializer(name: str, shape: list[int], dtype: str, generator: Any) -> Any:
    """Small random weights; unit norm scales, zero biases, and index buffers as ranges."""
    import torch

    torch_dtype = getattr(torch, SAFETENSORS_DTYPES[dtype])
    leaf = name.rsplit(".", 1)[-1]
    if not torch_dtype.is_floating_point:
        if "position_ids" in name and shape:
            return torch.arange(shape[-1], dtype=torch_dtype).expand(*shape).contiguous()
        return torch.zeros(shape, dtype=torch_dtype)
    if leaf == "bias":
        return torch.zeros(shape, dtype=torch_dtype)
    if "norm" in name.lower() and leaf == "weight":
        return torch.ones(shape, dtype=torch_dtype)
    values = torch.empty(shape, dtype=torch.float32)
    values.normal_(0.0, 0.02, generator=generator)
    return values.to(torch_dtype)


def synthesize(header: dict[str, Any], destination: Path, *, seed: int = 0) -> int:
    """Write a safetensors file matching ``header`` with random values; return tensor count."""
    import torch
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(seed)
    metadata = header.get("__metadata__")
    tensors = {
        name: _initializer(name, spec["shape"], spec["dtype"], generator)
        for name, spec in header.items()
        if name != "__metadata__"
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(destination), metadata=metadata)
    return len(tensors)


def build(repo: str, revision: str, output: Path) -> dict[str, Any]:
    """Create the skeleton checkpoint directory and return a summary of what was written."""
    if output.exists() and any(output.iterdir()):
        raise ContractError(f"{output} is not empty")
    files = list_files(repo, revision)
    summary: dict[str, Any] = {"repo": repo, "revision": revision, "copied": [], "synthesized": {}}
    for index, entry in enumerate(sorted(files, key=lambda e: e["path"])):
        path = relative_path(entry["path"])
        target = output / PurePosixPath(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".safetensors"):
            summary["synthesized"][path] = synthesize(read_header(repo, revision, path), target, seed=index)
        elif entry.get("size", 0) <= SMALL_FILE_BYTES:
            target.write_bytes(_request(file_url(repo, revision, path)))
            summary["copied"].append(path)
        elif path.endswith((".bin", ".pt", ".pth", ".ckpt")):
            raise ContractError(f"{path}: only safetensors weights can be synthesized from headers")
    if not summary["synthesized"]:
        raise ContractError(f"{repo}@{revision} has no safetensors weights")
    (output / "SKELETON.json").write_text(
        json.dumps({**summary, "note": "random weights for shape capture only"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
