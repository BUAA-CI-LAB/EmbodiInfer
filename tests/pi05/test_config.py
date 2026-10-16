"""Pi05 deployment presets and explicit numerical configuration."""

import json
import sys
from dataclasses import asdict

import pytest
import torch

from embodiinfer.layers import OperatorBackends
from embodiinfer.policies.config import VLAPolicyConfig
from embodiinfer.policies.pi05.inference.config import MlpLayerPrecision, Pi05OptimizationConfig
from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy


@pytest.mark.parametrize("device", ["thor", "spark", "orin", "4090"])
def test_strict_preset_preserves_default_numerics_without_fixed_request_shape(device):
    config = Pi05OptimizationConfig.from_preset(device)
    assert config == Pi05OptimizationConfig(hardware=device)
    assert config.numerics == "lerobot" and not config.fused_mlp


@pytest.mark.parametrize("prefix_mlp,action_mlp", [("fp8", "bf16"), ("bf16", "nvfp4"), ("fp8", "nvfp4")])
def test_custom_calibration_supports_other_tower_format_combinations(tmp_path, prefix_mlp, action_mlp):
    ranges = {
        "prefix": {"fp8": [asdict(MlpLayerPrecision("fp8", "bf16", 2.5, 8.75))] * 18},
        "action": {"nvfp4": [asdict(MlpLayerPrecision("nvfp4", "nvfp4", 4.5, 9.25))] * 18},
    }
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            dict(
                schema_version=2,
                numerics="openpi_rlinf",
                activation="gelu_pytorch_exact",
                checkpoint_sha256="a" * 64,
                devices={"thor": ranges},
            )
        )
    )
    config = Pi05OptimizationConfig.from_preset(
        "thor",
        f"{prefix_mlp}-{action_mlp}",
        calibration=path,
    )
    assert config.checkpoint_sha256 == "a" * 64
    assert config.prefix_layers == (
        (MlpLayerPrecision("fp8", "bf16", 2.5, 8.75),) * 18 if prefix_mlp == "fp8" else ()
    )
    assert config.action_layers == (
        (MlpLayerPrecision("nvfp4", "nvfp4", 4.5, 9.25),) * 18 if action_mlp == "nvfp4" else ()
    )


@pytest.mark.parametrize("preset", ["fp8-bf16", "bf16-nvfp4"])
def test_bundled_calibration_rejects_unmeasured_formats(preset):
    with pytest.raises(ValueError, match="18 .* MLP layer ranges"):
        Pi05OptimizationConfig.from_preset("thor", preset)


@pytest.mark.parametrize("device", ["thor", "spark"])
@pytest.mark.parametrize("prefix_mlp", ["bf16", "nvfp4"])
@pytest.mark.parametrize("action_mlp", ["bf16", "fp8"])
def test_optimized_presets_compose_independent_tower_precision(device, prefix_mlp, action_mlp, tmp_path):
    preset = "bf16" if prefix_mlp == action_mlp == "bf16" else f"{prefix_mlp}-{action_mlp}"
    config = Pi05OptimizationConfig.from_preset(device, preset)
    assert config.numerics == "openpi_rlinf" and config.activation == "gelu_pytorch_exact"
    assert all(
        (config.batch_cameras, config.compact_prefix, config.prefix_kv_only, config.reuse_action_context)
    )
    assert config.attention == ("folded_flash" if device == "thor" else "query_major")
    assert config.operators.gelu_mul == ("cuda_lookup" if device == "thor" else "cuda")
    assert config.operators.paired_gelu == "triton_exact"
    assert config.operators.projection == "torch_matmul" and config.operators.rotary == "cuda"
    if action_mlp == "fp8":
        assert len(config.action_layers) == 18
        formats = [value for layer in config.action_layers for value in (layer.gate_up, layer.down)]
        assert formats.count("fp8") == 28 and formats.count("bf16") == 8
    else:
        assert not config.action_layers
    if prefix_mlp == "nvfp4":
        assert len(config.prefix_layers) == 18
        assert all(layer.gate_up == layer.down == "nvfp4" for layer in config.prefix_layers)
    else:
        assert not config.prefix_layers
    assert config.has_quantized_mlp == (prefix_mlp != "bf16" or action_mlp != "bf16")
    path = tmp_path / "resolved.json"
    config.to_json(path)
    assert Pi05OptimizationConfig.from_json(path) == config


@pytest.mark.parametrize(
    "options",
    [
        dict(device="cpu"),
        dict(device="thor", preset="unknown"),
        dict(device="thor", preset="rlinf"),
        dict(device="thor", preset="mixed"),
        dict(device="thor", preset="fp8"),
        dict(device="thor", preset="optimized"),
        dict(device="4090", preset="bf16"),
        dict(device="orin", preset="fp8-fp8"),
        dict(device="4090", preset="nvfp4-bf16"),
        dict(device="thor", preset="strict", calibration="rlinf_libero"),
    ],
)
def test_presets_reject_unsupported_or_ambiguous_selectors(options):
    with pytest.raises(ValueError):
        Pi05OptimizationConfig.from_preset(**options)


def test_preset_resolution_is_cpu_safe_and_does_not_import_kernels(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("preset resolution probed CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected)
    before = set(sys.modules)
    Pi05OptimizationConfig.from_preset("thor", "nvfp4-fp8")
    assert not any(
        name.startswith(("triton", "embodiinfer.backend.cuda", "embodiinfer.backend.triton"))
        for name in set(sys.modules) - before
    )


@pytest.mark.parametrize("change", ["contract", "device", "layers", "checkpoint", "format"])
def test_custom_calibration_is_validated_before_execution(tmp_path, change):
    data = dict(
        schema_version=2,
        numerics="openpi_rlinf",
        activation="gelu_pytorch_exact",
        checkpoint_sha256="a" * 64,
        devices={"thor": {"action": {"fp8": [asdict(MlpLayerPrecision("fp8", "fp8", 2.5, 8.75))] * 18}}},
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(data))
    config = Pi05OptimizationConfig.from_preset("thor", "bf16-fp8", calibration=path)
    assert config.checkpoint_sha256 == "a" * 64
    assert config.action_layers == (MlpLayerPrecision("fp8", "fp8", 2.5, 8.75),) * 18
    if change == "contract":
        data["activation"] = "gelu_pytorch_tanh"
    elif change == "device":
        data["devices"] = {}
    elif change == "layers":
        data["devices"]["thor"]["action"]["fp8"].pop()
    elif change == "checkpoint":
        data.pop("checkpoint_sha256")
    else:
        data["devices"]["thor"]["action"]["fp8"][0]["down"] = "nvfp4"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        Pi05OptimizationConfig.from_preset("thor", "bf16-fp8", calibration=path)


def test_native_wrapped_model_uses_public_constructor_without_lerobot(make_pi05_policy, monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot.policies.pi05.modeling_pi05", None)
    policy = make_pi05_policy(device="cpu")
    assert policy.native_inference and policy.native_embeddings
    assert policy.config.action_horizon == 10
    assert policy._prefix_tower is policy._m.paligemma_with_expert.paligemma.model.language_model


def test_optimization_recipe_preserves_activation_contract(tmp_path):
    config = Pi05OptimizationConfig(
        hardware="thor",
        fused_mlp=True,
        action_layers=(MlpLayerPrecision(gate_up="fp8", gate_up_max=3.0),),
        checkpoint_sha256="1" * 64,
    )
    path = tmp_path / "recipe.json"
    config.to_json(path)
    assert Pi05OptimizationConfig.from_json(path) == config
    path.write_text('{"hardware": "thor", "action_layers": []}')
    with pytest.raises(ValueError, match="tanh GELU calibration contract"):
        Pi05OptimizationConfig.from_json(path)


@pytest.mark.parametrize(
    "values",
    [
        {"hardware": "unknown"},
        {"fused_mlp": True},
        {"attention": "flash"},
        {"activation": "exact"},
        {"norm_fusion": "yes"},
        {"action_layers": (MlpLayerPrecision(down="nvfp4"),)},
        {"checkpoint_sha256": "x" * 64},
        {"checkpoint_sha256": 42},
        {"schema_version": True},
    ],
)
def test_optimization_config_rejects_invalid_contract(values):
    with pytest.raises(ValueError):
        Pi05OptimizationConfig(**values)


def test_prefix_precision_is_checkpoint_bound_and_roundtrips(tmp_path):
    from embodiinfer.policies.pi05 import MlpLayerPrecision

    layers = (MlpLayerPrecision("nvfp4", "bf16", 3.0, 4.0),)
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        Pi05OptimizationConfig(prefix_layers=layers)
    config = Pi05OptimizationConfig(prefix_layers=layers, checkpoint_sha256="1" * 64)
    assert config.has_quantized_mlp and not config.action_layers
    path = tmp_path / "prefix.json"
    config.to_json(path)
    assert Pi05OptimizationConfig.from_json(path) == config
    with pytest.raises(ValueError, match="prefix_layers"):
        Pi05OptimizationConfig(prefix_layers=[MlpLayerPrecision()])


@pytest.mark.parametrize(
    "values",
    [
        {"native_inference": False},
        {"native_inference": True, "compile_backend": "inductor"},
        {"native_inference": True, "quantization": "fp8"},
        {"native_inference": True, "tensor_parallel_size": 2},
        {"native_inference": True, "denoise_attention": "triton"},
    ],
)
def test_unsupported_combinations_fail_before_checkpoint_loading(values):
    with pytest.raises(ValueError, match="Fused Pi05|Select optimized attention"):
        Pi05Policy(VLAPolicyConfig(), None, optimizations=Pi05OptimizationConfig(), **values)


def test_nested_backend_configuration_roundtrips(tmp_path):
    operators = OperatorBackends(
        normalization="torch", quantization="torch", gelu_mul="torch", norm_quant=None, paired_gelu="torch"
    )
    config = Pi05OptimizationConfig(fused_mlp=True, operators=operators)
    path = tmp_path / "operators.json"
    config.to_json(path)
    assert Pi05OptimizationConfig.from_json(path) == config
    with pytest.raises(ValueError, match="nonempty"):
        OperatorBackends(normalization="")
