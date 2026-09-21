"""Tests for VVLA's framework-neutral in-place refit API."""

import pytest
import torch

from embodiinfer import (
    EngineConfig,
    EngineCore,
    GenerationBackend,
    commit_refit,
    make_policy,
    policy_version,
    refit_module,
    refit_state_dict,
)


class RefitModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 2)
        self.register_buffer("scale", torch.ones(2))
        self.committed_versions: list[int] = []

    def on_refit(self, version: int) -> None:
        self.committed_versions.append(version)


class FailingHookModule(RefitModule):
    def __init__(self):
        super().__init__()
        self.fail_hook = True

    def on_refit(self, version: int) -> None:
        super().on_refit(version)
        if self.fail_hook:
            raise RuntimeError(f"runtime refresh failed for version {version}")


def _replacement(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: torch.full_like(tensor, index + 1)
        for index, (name, tensor) in enumerate(module.state_dict().items())
    }


def test_refit_copies_in_place_and_commits_external_version():
    module = RefitModule()
    storage = {name: tensor.data_ptr() for name, tensor in module.state_dict().items()}
    replacement = _replacement(module)

    result = refit_module(module, replacement, version=7)

    assert result.version == policy_version(module) == 7
    assert result.updated_keys == tuple(module.state_dict())
    assert result.missing_keys == result.unexpected_keys == ()
    assert module.committed_versions == [7]
    for name, tensor in module.state_dict().items():
        assert tensor.data_ptr() == storage[name]
        assert torch.equal(tensor, replacement[name])


def test_refit_name_map_and_partial_update_are_caller_owned():
    module = RefitModule()
    before = {name: tensor.clone() for name, tensor in module.state_dict().items()}

    result = refit_module(
        module,
        {
            "actor.weight": torch.zeros_like(module.proj.weight),
            "actor.ignored": torch.tensor(1),
        },
        name_map={"actor.weight": "proj.weight", "actor.ignored": None},
        strict=False,
    )

    assert result.updated_keys == ("proj.weight",)
    assert set(result.missing_keys) == {"proj.bias", "scale"}
    assert result.unexpected_keys == ()
    assert torch.equal(module.proj.weight, torch.zeros_like(module.proj.weight))
    assert torch.equal(module.proj.bias, before["proj.bias"])


def test_refit_validates_complete_schema_before_copying():
    module = RefitModule()
    before = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    invalid = _replacement(module)
    invalid["scale"] = torch.zeros(3)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        refit_module(module, invalid)

    for name, tensor in module.state_dict().items():
        assert torch.equal(tensor, before[name])
    assert policy_version(module) == 0


def test_zero_copy_refit_view_requires_an_explicit_commit():
    module = RefitModule()
    live = refit_state_dict(module)
    live["scale"].fill_(5)

    assert torch.equal(module.scale, torch.full_like(module.scale, 5))
    assert policy_version(module) == 0
    assert commit_refit(module, version=3) == 3
    assert policy_version(module) == 3

    with pytest.raises(ValueError, match="stale refit version"):
        commit_refit(module, version=2)


def test_hook_failure_does_not_publish_new_version():
    module = FailingHookModule()
    live = refit_state_dict(module)
    live["scale"].fill_(5)

    with pytest.raises(RuntimeError, match="runtime refresh failed"):
        commit_refit(module, version=3)

    assert policy_version(module) == 0
    assert torch.equal(module.scale, torch.full_like(module.scale, 5))


def test_copy_refit_hook_failure_keeps_previous_version():
    module = FailingHookModule()
    replacement = _replacement(module)

    with pytest.raises(RuntimeError, match="runtime refresh failed"):
        refit_module(module, replacement, version=3)

    assert policy_version(module) == 0
    assert torch.equal(module.proj.weight, replacement["proj.weight"])


def test_hook_failure_after_nonzero_version_keeps_version_and_retry_succeeds():
    module = FailingHookModule()
    module.fail_hook = False
    first = refit_module(module, _replacement(module), version=4)
    assert first.version == policy_version(module) == 4

    module.fail_hook = True
    with pytest.raises(RuntimeError, match="runtime refresh failed"):
        refit_module(module, _replacement(module), version=5)
    assert policy_version(module) == 4

    module.fail_hook = False
    retry = refit_module(module, _replacement(module), version=5)
    assert retry.version == policy_version(module) == 5


def test_vla_policy_exposes_the_same_public_refit_contract():
    policy = make_policy("mock_flow_vla", preset="tiny")
    replacement = _replacement(policy)

    result = policy.refit(replacement, version=4)

    assert result.version == policy.policy_version == 4
    assert tuple(policy.refit_state_dict()) == tuple(policy.state_dict())


def test_generation_backend_forwards_framework_name_map_and_version():
    policy = make_policy("mock_flow_vla", preset="tiny")
    backend = GenerationBackend(EngineCore(policy, EngineConfig(device="cpu", use_cuda_graph=False)))
    actor_weights = {f"actor.{name}": tensor.clone() for name, tensor in policy.state_dict().items()}

    result = backend.refit(
        actor_weights,
        name_map=lambda name: name.removeprefix("actor."),
        version=12,
    )

    assert result.version == backend.policy.policy_version == 12


class TiedRefitModule(RefitModule):
    def __init__(self, *, separate_parameters: bool = False):
        super().__init__()
        self.left = torch.nn.Parameter(torch.ones(2, 3))
        self.right = torch.nn.Parameter(self.left.detach()) if separate_parameters else self.left


@pytest.mark.parametrize("separate_parameters", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_conflicting_tied_updates_fail_before_any_mutation(separate_parameters, reverse_order):
    module = TiedRefitModule(separate_parameters=separate_parameters)
    before = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    weights = {name: torch.zeros_like(tensor) for name, tensor in before.items()}
    weights["right"].fill_(3)
    if reverse_order:
        weights = dict(reversed(list(weights.items())))

    with pytest.raises(ValueError, match="conflicting refit weights for tied targets"):
        refit_module(module, weights, version=5)

    assert policy_version(module) == 0
    assert module.committed_versions == []
    for name, tensor in module.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_consistent_ties_preserve_alias_and_parameter_gradient():
    module = TiedRefitModule()
    pointer = module.left.data_ptr()
    result = refit_module(
        module,
        {"actor.left": torch.full_like(module.left, 2), "actor.right": torch.full_like(module.right, 2)},
        strict=False,
        name_map={"actor.left": "left", "actor.right": "right"},
        version=4,
    )
    assert result.updated_keys == ("left", "right")
    assert module.left is module.right
    assert module.left.data_ptr() == pointer
    (module.left.sum() + module.right.sum()).backward()
    assert torch.equal(module.left.grad, torch.full_like(module.left, 2))
    assert module.committed_versions == [4]


def test_partial_tied_update_changes_all_aliases():
    module = TiedRefitModule()
    refit_module(module, {"right": torch.zeros_like(module.right)}, strict=False)
    assert module.left is module.right
    assert torch.count_nonzero(module.left) == 0


def test_tied_sources_are_compared_in_destination_dtype():
    module = TiedRefitModule().to(dtype=torch.bfloat16)
    refit_module(
        module,
        {"left": torch.full((2, 3), 1.001), "right": torch.full((2, 3), 1.002)},
        strict=False,
    )
    assert torch.equal(module.left, torch.ones_like(module.left))


def test_disjoint_views_of_one_storage_accept_different_values():
    module = RefitModule()
    storage = torch.zeros(4)
    module.register_buffer("first_half", storage[:2])
    module.register_buffer("second_half", storage[2:])
    refit_module(module, {"first_half": torch.ones(2), "second_half": torch.full((2,), 2)}, strict=False)
    assert torch.equal(storage, torch.tensor([1.0, 1.0, 2.0, 2.0]))


def test_stale_refit_rejects_before_copy_and_preserves_committed_state():
    module = RefitModule()
    refit_module(module, _replacement(module), version=8)
    before = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    weights = {name: torch.zeros_like(tensor) for name, tensor in before.items()}
    with pytest.raises(ValueError, match="stale refit version"):
        refit_module(module, weights, version=7)
    assert policy_version(module) == 8
    assert module.committed_versions == [8]
    for name, tensor in module.state_dict().items():
        assert torch.equal(tensor, before[name])
