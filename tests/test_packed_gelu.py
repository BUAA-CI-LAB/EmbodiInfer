"""GPU contracts for the non-quantized packed BF16 activation prototype."""

import pytest
import torch
from torch.nn import functional as F

from embodiinfer.layers import OperatorRequest, gated_gelu_backends

pytestmark = pytest.mark.gpu


def _backend():
    return gated_gelu_backends.get(
        "cuda_packed",
        OperatorRequest(torch.device("cuda", torch.cuda.current_device()), torch.bfloat16, bits=16),
        approximate="tanh",
    )


@pytest.mark.parametrize("columns", [7, 1024])
def test_packed_gelu_retains_torch_rounding_and_reads_changed_inputs(columns):
    packed = torch.randn(3, 3 * columns, device="cuda", dtype=torch.bfloat16)
    gate, up = packed[:, :columns], packed[:, columns : 2 * columns]
    backend = _backend()
    plan = backend.plan(gate, up)
    for _ in range(3):
        expected = F.gelu(gate, approximate="tanh") * up
        actual = plan.run(gate, up)
        torch.cuda.synchronize()
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        packed.mul_(0.5)
    with pytest.raises(ValueError, match="layout"):
        plan.run(gate.contiguous(), up.contiguous())


def test_packed_gelu_exhaustive_finite_bf16_lookup():
    values = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    values = values[torch.isfinite(values)].reshape(-1, 1).contiguous()
    up = torch.ones_like(values)
    plan = _backend().plan(values, up)
    expected = F.gelu(values, approximate="tanh") * up
    actual = plan.run(values, up)
    torch.cuda.synchronize()
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))


def test_packed_gelu_capture_replay_and_owned_clone():
    packed = torch.randn(3, 2048, device="cuda", dtype=torch.bfloat16)
    gate, up = packed.chunk(2, dim=-1)
    plan = _backend().plan(gate, up)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            plan.run(gate, up)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = plan.run(gate, up)
    graph.replay()
    torch.cuda.synchronize()
    previous = result.clone()
    frozen = previous.clone()
    packed.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(result.view(torch.uint8), (F.gelu(gate, approximate="tanh") * up).view(torch.uint8))
    assert torch.equal(previous, frozen)
    graph.reset()
