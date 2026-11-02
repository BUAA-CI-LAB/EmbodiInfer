"""GPU integration checks for the optional Pi05 FlashInfer recipe."""

import pytest
import torch


@pytest.mark.gpu
def test_flashinfer_full_loop_graph_refit_and_scope_lifetime(make_pi05_policy):
    from embodiinfer.backend.flashinfer.runtime import availability
    from embodiinfer.policies.pi05.inference.runtime import _LoopGraph
    from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch

    available, reason = availability()
    if not available:
        pytest.skip(reason)
    torch.manual_seed(71)
    policy = make_pi05_policy(
        kernel_profile="flashinfer",
        native_inference=False,
        prefix_attention="triton",
        denoise_attention="triton",
    )
    assert policy.native_inference and policy.kernel_profile == "flashinfer"
    assert policy.prefix_attention == policy.denoise_attention == "sdpa"
    config = policy._optimization_config
    assert config.attention == "flashinfer_fa2" and config.norm_fusion and not config.fused_mlp
    assert config.operators.normalization == "flashinfer"
    assert config.operators.projection == "flashinfer_cublaslt"
    with pytest.raises(RuntimeError, match="eval/no-grad"):
        policy._get_optimizations()
    parameter_ids = {name: id(value) for name, value in policy.named_parameters()}
    noise = torch.randn(1, 10, 8, device="cuda")
    with torch.inference_mode():
        previous = None
        for tokens, masks in (([1, 2, 3], [True, False, True]), ([3, 2, 1], [True, True, False])):
            batch = Pi05Batch(
                [torch.randn(1, 3, 8, 8, device="cuda")],
                [torch.tensor([True], device="cuda")],
                torch.tensor([tokens], device="cuda"),
                torch.tensor([masks], device="cuda"),
            )
            prefix = policy.encode_prefix(batch)
            expected = policy._runtime.decode(noise, prefix, 10, False)
            actual = policy._runtime.decode(noise, prefix, 10, True)
            torch.cuda.synchronize()
            # Same backend, tensors, seed/noise and ten-step schedule: graph
            # capture itself must introduce no additional numerical difference.
            assert torch.equal(actual, expected)
            if previous is not None:
                assert not torch.equal(previous, actual)
            previous = actual.clone()
        operators = policy._fused_ops
        assert operators.attention.name == "flashinfer_fa2"
        assert all(not plan.adaptive for plan in operators.norm_plans.values())
        assert operators.mlp_plans and operators.attention._plans and operators.projection._workspaces
        old = next(iter(policy._runtime.loop_graphs.values()))
        old_scope = ("graph", id(old.scope))
        replacement = _LoopGraph(policy._runtime, noise, prefix, 10)
        policy._runtime._remember(policy._runtime.loop_graphs, "replacement", replacement, limit=1)
        assert not any(key[-1] == old_scope for key in operators.attention._plans)
        assert not any(key[-1] == old_scope for key in operators.projection._workspaces)
        assert parameter_ids == {name: id(value) for name, value in policy.named_parameters()}
        policy.on_refit(1)
        assert not operators.attention._plans and not operators.projection._workspaces
        assert not operators.norm_plans and not operators.mlp_plans
        assert policy._fused_ops is None
