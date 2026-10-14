"""Optional whole-request graph: actual CUDA parity/lifetime checks, no weights needed."""

from types import SimpleNamespace

import pytest
import torch

from embodiinfer.policies.pi05.inference.config import Pi05OptimizationConfig
from embodiinfer.policies.pi05.processor_pi05 import Pi05Batch


def _batch(seed, device="cuda", batch_size=1):
    generator = torch.Generator(device=device).manual_seed(seed)
    return Pi05Batch(
        [torch.randn(batch_size, 3, 8, 8, device=device, generator=generator)],
        [torch.ones(batch_size, dtype=torch.bool, device=device)],
        torch.randint(1, 20, (batch_size, 4), device=device, generator=generator),
        torch.tensor([[True, False, True, True]], device=device).expand(batch_size, -1).clone(),
    )


def test_request_capability_is_default_off_and_cpu_falls_back(make_pi05_policy):
    policy = make_pi05_policy(None, device="cpu")
    assert not policy.request_cuda_graph
    policy.request_cuda_graph = True
    with torch.inference_mode():
        assert not policy.decoder.supports_request_execution(SimpleNamespace(full_loop=True))
        with pytest.raises(ValueError, match="native CUDA"):
            policy._runtime.request(_batch(1, "cpu"), torch.zeros(1, 10, 8), 3)


@pytest.mark.gpu
@pytest.mark.parametrize("operators", [False, True])
def test_whole_request_matches_two_graphs_changed_inputs_and_refit(make_pi05_policy, operators):
    policy = make_pi05_policy(Pi05OptimizationConfig() if operators else None)
    policy.request_cuda_graph = True
    runtime = policy._runtime
    noise = torch.randn(1, 10, 8, device="cuda")
    with torch.inference_mode():
        outputs = []
        for seed in (1, 2, 1):
            batch = _batch(seed)
            prefix = policy.encode_prefix(batch)
            saved_prefix = [(k.clone(), v.clone()) for k, v in prefix.kv]
            expected = runtime.decode(noise, prefix, 3, True)
            actual = runtime.request(batch, noise, 3)
            torch.cuda.synchronize()
            assert torch.equal(actual, expected)
            assert all(
                torch.equal(k, sk) and torch.equal(v, sv) for (k, v), (sk, sv) in zip(prefix.kv, saved_prefix)
            )
            outputs.append((actual, actual.clone()))
        assert len(runtime.request_graphs) == 1
        assert not torch.equal(outputs[0][0], outputs[1][0])
        assert torch.equal(outputs[0][0], outputs[2][0])
        for output, saved in outputs:
            assert torch.equal(output, saved)
        assert outputs[0][0].data_ptr() != outputs[2][0].data_ptr()
        runtime.request(_batch(1, batch_size=2), noise.expand(2, -1, -1), 3)
        runtime.request(_batch(1), noise, 4)
        assert len(runtime.request_graphs) == 3
        with torch.no_grad():
            policy._m.action_out_proj.weight.add_(0.01)
        old_graphs = list(runtime.request_graphs.values())
        policy.on_refit(1)
        assert not runtime.request_graphs
        assert all(graph.captured._closed for graph in old_graphs)
        updated = runtime.request(_batch(1), noise, 3)
        assert not torch.equal(updated, outputs[0][0])


@pytest.mark.gpu
def test_request_streams_and_policy_instances_have_independent_workspaces(make_pi05_policy):
    first, second = make_pi05_policy(None), make_pi05_policy(None)
    noise = torch.randn(1, 10, 8, device="cuda")
    batches = [_batch(3), _batch(4)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode():
        expected = [first._runtime.request(batch, noise, 3) for batch in batches]
        actual = []
        for stream, batch in zip(streams, batches):
            with torch.cuda.stream(stream):
                actual.append(first._runtime.request(batch, noise, 3))
        other = second._runtime.request(batches[0], noise, 3)
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert len(first._runtime.request_graphs) == 3
        assert len(second._runtime.request_graphs) == 1
        assert other.data_ptr() not in {value.data_ptr() for value in actual}


@pytest.mark.gpu
def test_failed_request_capture_does_not_publish_or_prevent_retry(make_pi05_policy, monkeypatch):
    policy = make_pi05_policy(None)
    original = policy._encode_prefix_impl

    def fail(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("request capture injected failure")
        return original(*args, **kwargs)

    with torch.inference_mode():
        monkeypatch.setattr(policy, "_encode_prefix_impl", fail)
        with pytest.raises(RuntimeError, match="injected failure"):
            policy._runtime.request(_batch(1), torch.zeros(1, 10, 8, device="cuda"), 3)
        assert not policy._runtime.request_graphs
        monkeypatch.setattr(policy, "_encode_prefix_impl", original)
        assert policy._runtime.request(_batch(1), torch.zeros(1, 10, 8, device="cuda"), 3).shape == (1, 10, 8)


@pytest.mark.gpu
@pytest.mark.parametrize("combined", [False, True])
def test_graph_pins_schedule_after_shared_lru_eviction(make_pi05_policy, combined):
    import weakref

    policy = make_pi05_policy(None)
    runtime = policy._runtime
    batch, noise = _batch(2), torch.randn(1, 10, 8, device="cuda")
    with torch.inference_mode():
        if combined:
            expected = runtime.request(batch, noise, 3)
            graph = next(iter(runtime.request_graphs.values()))
        else:
            prefix = policy.encode_prefix(batch)
            expected = runtime.decode(noise, prefix, 3, True)
            graph = next(iter(runtime.loop_graphs.values()))
        references = [weakref.ref(row[0]) for row in graph.schedule]
        for steps in range(10, 20):
            runtime.schedule(noise, steps)
        assert all(reference() is not None for reference in references)
        assert all(value is not graph.schedule for value in runtime.schedules.values())
        actual = runtime.request(batch, noise, 3) if combined else runtime.decode(noise, prefix, 3, True)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)


@pytest.mark.gpu
def test_combined_eviction_releases_only_evicted_operator_scope(make_pi05_policy):
    policy = make_pi05_policy(Pi05OptimizationConfig())
    runtime = policy._runtime
    batch, noise = _batch(2), torch.randn(1, 10, 8, device="cuda")
    with torch.inference_mode():
        runtime.request(batch, noise, 2)
        old = next(iter(runtime.request_graphs.values()))
        operators = policy._fused_ops
        old_scope = ("graph", id(old.scope))
        assert any(key[-1] == old_scope for key in operators.kv_buffers)
        for steps in range(3, 11):
            runtime.request(batch, noise, steps)
        assert len(runtime.request_graphs) == 8
        assert old.captured._closed
        assert not any(key[-1] == old_scope for key in operators.kv_buffers)
        assert operators.kv_buffers
        policy._clear_inference_caches()
        assert not operators.kv_buffers
