"""Calibrated MLP W8A8 adapter; runtime files remain at the measured resident-KV commit."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import torch
from int8_text_forward import _text_forward
from nx_int8_kernel import int8_capability, int8_linear

from embodiinfer.backend.torch.int8 import quantize_weight
from embodiinfer.layers.linear import register_linear, resolve_linear_backend
from embodiinfer.models.linear import INT8Linear
from embodiinfer.policies.activevln import cuda_graph
from embodiinfer.policies.activevln.modeling_activevln import ActiveVLNPolicy


def install_dispatch() -> dict[str, str]:
    """Register the original W8A8 kernel and route only quantized text projections to it."""
    register_linear("int8_triton", int8_linear, int8_capability)
    path = Path(__file__).with_name("int8_text_forward.py")
    cuda_graph.ActiveVLNGraphRuntime._text_forward = _text_forward
    return {
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "text_forward_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kernel_sha256": hashlib.sha256(
            Path(__file__).with_name("nx_int8_kernel.py").read_bytes()
        ).hexdigest(),
    }


class _SmoothedLinear(INT8Linear):
    def __init__(
        self, linear: torch.nn.Linear, maximum: torch.Tensor, alpha: float, *, fused: bool = True
    ) -> None:
        weight = linear.weight.detach().float()
        scale = (
            (maximum.clamp_min(1e-5).pow(alpha) / weight.abs().amax(0).clamp_min(1e-5).pow(1 - alpha)).clamp(
                1e-5, 1e5
            )
            if alpha
            else torch.ones_like(maximum)
        )
        quantized, weight_scale = quantize_weight(weight * scale)
        super().__init__(
            quantized, weight_scale, linear.bias, compute_dtype=linear.weight.dtype, backend="triton"
        )
        self.register_buffer("input_scale", scale)
        self.alpha = alpha
        self.fused = fused

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Balance input channels in FP32, then round to BF16 before W8A8 quantization."""
        if self.alpha and not self.fused:
            inputs = (inputs.float() / self.input_scale).to(inputs.dtype)
        return super().forward(inputs)

    def _linear_metadata(self) -> dict[str, torch.Tensor]:
        return {"input_scale": self.input_scale} if self.alpha and self.fused else {}


def _selected(policy: ActiveVLNPolicy) -> Iterator[tuple[str, torch.nn.Module, str, torch.nn.Linear]]:
    for scope, root in (("text", policy._text), ("visual", policy._visual)):
        for name, layer in root.named_modules():
            if (
                isinstance(layer, torch.nn.Linear)
                and ".mlp." in name
                and (scope == "text" or name.startswith("blocks."))
            ):
                yield scope + "." + name, root, name, layer


@torch.no_grad()
def convert(policy: ActiveVLNPolicy, calibration_path: Path, config: dict) -> dict:
    """Apply the retained smooth_both calibration to the same 204 text/vision MLP layers."""
    expected = config["precision_experiment"]
    if expected["variant"] != "smooth_both" or config["tree_fp32_projection"]:
        raise ValueError("Expected smooth_both and the unchanged BF16 tree projection setting")
    if hashlib.sha256(calibration_path.read_bytes()).hexdigest() != expected["calibration_sha256"]:
        raise ValueError("Calibration artifact differs from the retained experiment")
    artifact = torch.load(calibration_path, map_location="cpu", weights_only=True)
    metadata = artifact.get("metadata", artifact)
    if metadata["checkpoint_revision"] != config["checkpoint_revision"]:
        raise ValueError("Calibration checkpoint revision differs")
    if metadata["calibration_episode_ids"] != expected["calibration_episode_ids"]:
        raise ValueError("Calibration selection differs")
    manifest = Path(config["checkpoint"]) / "conversion-manifest.json"
    if (
        metadata.get("checkpoint_manifest_sha256")
        and hashlib.sha256(manifest.read_bytes()).hexdigest() != metadata["checkpoint_manifest_sha256"]
    ):
        raise ValueError("Checkpoint conversion manifest differs")
    provenance = install_dispatch()
    names = [(name, root, local) for name, root, local, _ in _selected(policy)]
    if len(names) != 204 or any("merger" in name for name, _, _ in names):
        raise ValueError("Expected only 204 text and vision-block MLP projections")
    layers = []
    for name, root, local in names:
        parent, attribute = local.rsplit(".", 1)
        original = root.get_submodule(local)
        maximum = artifact["maxima"][name].to(original.weight.device)
        if maximum.shape != (original.in_features,) or not torch.isfinite(maximum).all():
            raise ValueError("Invalid calibration channel maxima: " + name)
        converted = _SmoothedLinear(original, maximum, 0.5, fused=True)
        setattr(root.get_submodule(parent), attribute, converted)
        probe = torch.empty(
            (1, converted.in_features), device=converted.weight.device, dtype=converted.compute_dtype
        )
        backend = resolve_linear_backend(
            "int8", converted.backend, probe, converted.weight, converted.weight_scale
        )
        if backend != "triton":
            raise RuntimeError("Expected real Triton W8A8 execution, not weight-only fallback")
        layers.append(
            {
                "name": name,
                "backend": backend,
                "weight_shape": list(converted.weight.shape),
                "weight_dtype": str(converted.weight.dtype),
            }
        )
    if provenance["adapter_sha256"] != expected["script_sha256"]:
        raise ValueError("Adapter does not match the experiment configuration")
    return {
        "enabled": True,
        "method": "int8",
        "activation_scheme": "dynamic",
        "layer_count": len(layers),
        "layers": layers,
        "accuracy_contract": "lossy",
        "calibration_sha256": expected["calibration_sha256"],
        "runtime_adapter": provenance,
    }
