"""Checkpoint-bound offline GPTQ MLP artifacts; calibration stays outside runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MlpWeightArchive:
    """Load original-column uint4 codes without online quantization or permutation.

    The manifest identifies the checkpoint and each projection's file hash,
    [output,input] dimensions and quantization method. Gate/up rows are ordered
    gate then up. The final prefix MLP is unused by the KV-only execution plan.
    """

    def __init__(self, manifest: str, checkpoint_sha256: str, *, layers: int) -> None:
        """Validate the complete archive contract before any kernel preparation."""
        self.path = Path(manifest).expanduser().resolve()
        data = json.loads(self.path.read_text())
        if (
            data.get("status") != "complete"
            or data.get("method") != "gptq-static-actorder"
            or data.get("checkpoint_sha256") != checkpoint_sha256
            or data.get("bits") != 4
            or data.get("group_size") != 64
            or data.get("runtime_extra_ops") != 0
            or data.get("scales_dtype") != "bfloat16"
        ):
            raise ValueError("Unsupported or checkpoint-mismatched GPTQ weight archive")
        self.rows = data.get("projections", {})
        expected = {
            f"llm.layers.{i}.mlps.{expert}.{projection}"
            for i in range(layers)
            for expert in (0, 1)
            for projection in ("gate_up", "down")
        }
        if set(self.rows) != expected:
            raise ValueError("GPTQ archive must describe every prefix/action MLP projection")
        for name, row in self.rows.items():
            method = row.get(
                "method", "gptq" if row.get("activation_rows", 0) > 0 else "rtn-unused-prefix-mlp"
            )
            shape, digest = row.get("shape"), row.get("sha256")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(n) is not int or n <= 0 for n in shape)
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or method not in ("gptq", "rtn-unused-prefix-mlp")
            ):
                raise ValueError(f"Malformed GPTQ projection metadata: {name}")
            if method == "rtn-unused-prefix-mlp" and not name.startswith(f"llm.layers.{layers - 1}.mlps.0."):
                raise ValueError("RTN weights are allowed only for the unused final prefix MLP")

    def load(
        self, layer: int, *, expert: bool, projection: str, shape: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Verify each shard's bytes and dimensions before unpacking CPU nibbles."""
        from safetensors import safe_open

        name = f"llm.layers.{layer}.mlps.{int(expert)}.{projection}"
        row = self.rows[name]
        path = self.path.parent / f"{name}.safetensors"
        if tuple(row["shape"]) != shape or _sha256(path) != row["sha256"]:
            raise ValueError(f"GPTQ shard shape or SHA256 mismatch: {name}")
        with safe_open(path, framework="pt", device="cpu") as shard:
            if set(shard.keys()) != {"codes", "scales"}:
                raise ValueError(f"Unexpected GPTQ tensors: {name}")
            codes, scales = shard.get_tensor("codes"), shard.get_tensor("scales")
        n, k = shape
        if (
            codes.dtype != torch.uint8
            or codes.shape != (n, k // 2)
            or scales.dtype != torch.bfloat16
            or scales.shape != (n, k // 64)
        ):
            raise ValueError(f"Invalid GPTQ packed tensor layout: {name}")
        codes = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2)
        return codes, scales
