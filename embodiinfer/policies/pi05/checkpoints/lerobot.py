"""Load LeRobot Pi0.5 checkpoints without changing their native semantics."""

from __future__ import annotations

from typing import Any


def load_lerobot_checkpoint(
    checkpoint: str, *, load_device: str | None = None, low_cpu_mem_usage: bool = False
) -> Any:
    """Load native weights, optionally avoiding full random model initialization.

    The opt-in meta path retains native parameter dtypes, key remapping and tied
    parameters. It fails on incomplete weights instead of returning an untrained
    model. Generation and checkpoint processor configuration are unchanged.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if low_cpu_mem_usage:
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if load_device is not None:
            config.device = load_device
        return _load_meta_checkpoint(PI05Policy, config, checkpoint)
    if load_device is None:
        return PI05Policy.from_pretrained(checkpoint).eval()
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = load_device
    return PI05Policy.from_pretrained(checkpoint, config=config).eval()


def _load_meta_checkpoint(policy_type: type, config: Any, checkpoint: str) -> Any:
    """Materialize native meta modules with strict, dtype-preserving assignment."""
    import torch
    from safetensors.torch import load_file
    from transformers.utils import cached_file

    target = torch.device(config.device)
    weights = cached_file(checkpoint, "model.safetensors")
    config.device = "meta"
    try:
        with torch.device("meta"):
            model = policy_type(config)
    finally:
        config.device = str(target)
    expected = model.state_dict()
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    tensors = model._fix_pytorch_state_dict_keys(load_file(weights, device="cpu"), config)
    tensors = {key if key.startswith("model.") else "model." + key: value for key, value in tensors.items()}
    missing, unexpected = set(expected) - set(tensors), set(tensors) - set(expected)
    if missing or unexpected:
        raise RuntimeError(
            f"PI0.5 checkpoint key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for name, tensor in tensors.items():
        parameter = expected[name]
        if tensor.shape != parameter.shape:
            raise RuntimeError(f"PI0.5 checkpoint shape mismatch: {name}")
        tensors[name] = tensor.to(device=target, dtype=parameter.dtype)
    model.load_state_dict(tensors, strict=True, assign=True)
    del tensors, expected
    # Native copy loading leaves aliases tied; assignment must restore that link.
    # The last alias in module traversal has the same precedence as copy loading.
    for names in aliases.values():
        if len(names) > 1:
            parameter = model.get_parameter(names[-1])
            for name in names[:-1]:
                owner, attribute = name.rsplit(".", 1)
                setattr(model.get_submodule(owner), attribute, parameter)
    _materialize_buffers(model, target)
    return model.eval()


def _materialize_buffers(model: Any, device: Any) -> None:
    """Restore checkpoint-omitted vision positions and native rotary frequencies."""
    import torch

    for name, module in model.named_modules():
        if name.endswith("rotary_emb"):
            # Reuse the checkpoint's RoPE implementation and configuration instead
            # of approximating frequencies or assuming one Transformers version.
            with torch.device("cpu"):
                reference = type(module)(module.config, device="cpu")
            for key, value in reference.named_buffers(recurse=False, remove_duplicate=False):
                original = getattr(module, key)
                setattr(module, key, value.to(device=device, dtype=original.dtype))
    for name, buffer in list(model.named_buffers(remove_duplicate=False)):
        owner, attribute = name.rsplit(".", 1)
        if buffer.is_meta:
            if attribute != "position_ids":
                raise RuntimeError(f"unhandled PI0.5 runtime buffer: {name}")
            value = torch.arange(buffer.numel(), device=device, dtype=buffer.dtype).reshape(buffer.shape)
        else:
            value = buffer.to(device=device)
        setattr(model.get_submodule(owner), attribute, value)
