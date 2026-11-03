"""Framework-agnostic in-place policy weight refitting.

Training systems own transport, sharding, and their actor-side parameter names.
EmbodiInfer only owns the live rollout policy and the lifecycle of weights installed in
it.  This module is the boundary between the two: callers may either copy a
mapping of tensors with :func:`refit_module`, or mutate tensors returned by
``refit_state_dict`` through a zero-copy transport and finish with
:func:`commit_refit`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import torch

WeightNameMap: TypeAlias = Mapping[str, str | None] | Callable[[str], str | None]
"""How a framework renames its weights onto the policy's own parameter names.

Either a mapping from source name to policy name, or a callable performing the same
lookup. A missing source name maps to itself; a ``None`` value or return ignores that
source entry, leaving the policy parameter untouched. The mapping policy belongs to
the integrating framework, not to the engine.
"""

_POLICY_VERSION_ATTR = "_embodiinfer_policy_version"


@dataclass(frozen=True)
class RefitResult:
    """Result of one completed in-place refit."""

    version: int
    updated_keys: tuple[str, ...]
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()


def policy_version(module: torch.nn.Module) -> int:
    """Return the monotonic version of weights visible to inference."""

    return int(getattr(module, _POLICY_VERSION_ATTR, 0))


def refit_state_dict(
    module: torch.nn.Module,
    *,
    keep_vars: bool = False,
) -> dict[str, torch.Tensor]:
    """Return live refittable tensors in EmbodiInfer's canonical module namespace.

    The returned mapping is shallow: its tensors share storage with the policy.
    A transport may update those tensors directly and then call
    :func:`commit_refit`.  Framework-specific aliases belong in the caller.
    """

    return dict(module.state_dict(keep_vars=keep_vars))


def _resolve_name(name: str, name_map: WeightNameMap | None) -> str | None:
    if name_map is None:
        return name
    if callable(name_map):
        return name_map(name)
    return name_map.get(name, name)


def _refit_version(module: torch.nn.Module, version: int | None) -> int:
    current = policy_version(module)
    committed = current + 1 if version is None else int(version)
    if committed < current:
        raise ValueError(f"cannot commit stale refit version {committed}; current version is {current}")
    return committed


def _storage_view_key(tensor: torch.Tensor) -> tuple | None:
    """Identify exact dense aliases without conflating disjoint storage views."""
    if tensor.layout != torch.strided or tensor.is_meta or tensor.numel() == 0:
        return None
    return (
        tensor.device,
        tensor.dtype,
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tensor.stride(),
    )


def _validate_tied_updates(
    target: Mapping[str, torch.Tensor],
    resolved: Mapping[str, tuple[str, torch.Tensor]],
) -> list[str]:
    """Reject contradictory values for exact ties before any destination copy."""
    first_by_view: dict[tuple, str] = {}
    copy_names: list[str] = []
    for target_name, (_, tensor) in resolved.items():
        destination = target[target_name]
        key = _storage_view_key(destination)
        previous_name = first_by_view.get(key) if key is not None else None
        if previous_name is None:
            if key is not None:
                first_by_view[key] = target_name
            copy_names.append(target_name)
            continue

        previous = resolved[previous_name][1]
        # Compare the values that copy_ would install, including dtype casting.
        # This avoids rejecting sources that become equal in the target dtype.
        if not torch.equal(
            previous.to(device=destination.device, dtype=destination.dtype),
            tensor.to(device=destination.device, dtype=destination.dtype),
        ):
            raise ValueError(
                f"conflicting refit weights for tied targets {previous_name!r} and {target_name!r}"
            )
    return copy_names


def commit_refit(module: torch.nn.Module, *, version: int | None = None) -> int:
    """Commit weights already installed in ``module`` and publish their version.

    This is the completion half of the zero-copy refit protocol.  ``version``
    may be supplied by an external learner; stale versions are rejected.  The
    optional ``on_refit`` hook runs before the version becomes visible, so a
    hook failure cannot publish a version for an incomplete runtime refresh.
    The zero-copy tensors remain caller-owned and are not rolled back.
    """

    committed = _refit_version(module, version)
    hook = getattr(module, "on_refit", None)
    if hook is not None:
        hook(committed)
    setattr(module, _POLICY_VERSION_ATTR, committed)
    return committed


@torch.no_grad()
def refit_module(
    module: torch.nn.Module,
    weights: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
    name_map: WeightNameMap | None = None,
    version: int | None = None,
) -> RefitResult:
    """Validate and copy new weights into a live module without reallocating it.

    Args:
        module: Live rollout policy or module to update.
        weights: Source-name to tensor mapping supplied by a learner/transport.
        strict: Require every target tensor and reject unknown source tensors.
        name_map: Optional source-name to EmbodiInfer-name mapping. Returning ``None``
            intentionally ignores a source entry. Mapping policy is owned by the
            integrating framework, not EmbodiInfer.
        version: Optional external learner version to publish after the copy.

    Shape/name, version and exact tied-weight validation complete before the
    first copy. Conflicting values for two names of the same storage view are
    rejected; consistent ties are copied once. Partial updates may supply only
    one tied name. This does not provide rollback after a device-copy or hook
    failure, or a transaction across multiple refit calls.
    """

    committed_version = _refit_version(module, version)
    target = refit_state_dict(module, keep_vars=True)
    resolved: dict[str, tuple[str, torch.Tensor]] = {}
    unexpected: list[str] = []

    for source_name, tensor in weights.items():
        target_name = _resolve_name(source_name, name_map)
        if target_name is None:
            continue
        if target_name not in target:
            unexpected.append(source_name)
            continue
        if target_name in resolved:
            previous = resolved[target_name][0]
            raise ValueError(
                f"multiple source weights map to {target_name!r}: {previous!r} and {source_name!r}"
            )
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"refit weight {source_name!r} must be a torch.Tensor, got {type(tensor).__name__}"
            )
        expected = target[target_name]
        if tensor.shape != expected.shape:
            raise RuntimeError(
                f"shape mismatch for refit weight {source_name!r} -> {target_name!r}: "
                f"expected {tuple(expected.shape)}, got {tuple(tensor.shape)}"
            )
        resolved[target_name] = (source_name, tensor)

    missing = [name for name in target if name not in resolved]
    if strict and unexpected:
        raise KeyError(f"unexpected refit weight {unexpected[0]!r}")
    if strict and missing:
        suffix = "..." if len(missing) > 4 else ""
        raise KeyError(f"missing weights for refit: {missing[:4]}{suffix}")

    copy_names = _validate_tied_updates(target, resolved)
    for target_name in copy_names:
        tensor = resolved[target_name][1]
        destination = target[target_name]
        destination.copy_(tensor.to(device=destination.device, dtype=destination.dtype))

    committed = commit_refit(module, version=committed_version)
    return RefitResult(
        version=committed,
        updated_keys=tuple(resolved),
        missing_keys=tuple(missing),
        unexpected_keys=tuple(unexpected),
    )


__all__ = [
    "RefitResult",
    "WeightNameMap",
    "commit_refit",
    "policy_version",
    "refit_module",
    "refit_state_dict",
]
