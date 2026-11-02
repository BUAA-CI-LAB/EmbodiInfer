"""GPU checks for optional FlashInfer kernels and CUDA Graph replay."""

import pytest
import torch
import torch.nn.functional as F

from embodiinfer.layers import (
    OperatorRequest,
    get_attention_backend,
    normalization_backends,
    projection_backends,
)


def _require_flashinfer():
    from embodiinfer.backend.flashinfer.runtime import availability

    available, reason = availability()
    if not available:
        pytest.skip(reason)


@pytest.mark.gpu
@pytest.mark.parametrize("batch,qlen,klen", [(1, 576, 576), (1, 10, 586), (2, 50, 722)])
def test_fa2_gemma_shapes_parity_and_graph_mask_replay(batch, qlen, klen):
    _require_flashinfer()
    torch.manual_seed(47)
    q = torch.randn(batch, 8, qlen, 256, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(batch, 1, klen, 256, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    mask = torch.rand(batch, 1, qlen, klen, device="cuda") > 0.2
    mask[:, :, 0] = False
    backend = get_attention_backend("flashinfer_fa2")
    backend.bind_workspace_key(lambda _: ("graph", 1))
    with torch.inference_mode():
        backend.attend(q, k, v, mask)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = backend.attend(q, k, v, mask)
        for _ in range(2):
            q.normal_()
            mask.logical_not_()
            graph.replay()
            expected = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), attn_mask=mask, enable_gqa=True
            )
            torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-3)
        with pytest.raises(ValueError, match="width"):
            backend.attend(q[..., :72], k[..., :72], v[..., :72])
        with pytest.raises(ValueError, match="Boolean"):
            backend.attend(q, k, v, mask.float())
        with pytest.raises(ValueError, match="dropout"):
            backend.attend(q, k, v, mask, dropout_p=0.1)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1, 576, 2048), (2, 50, 1024)])
def test_gemma_norm_parity_and_graph(shape):
    _require_flashinfer()
    torch.manual_seed(53)
    inputs = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16).float()
    backend = normalization_backends.get(
        "flashinfer", OperatorRequest(inputs.device, inputs.dtype, shape, cuda_graph=True)
    )
    with torch.inference_mode():
        plan = backend.plan(inputs, adaptive=False)
        plan.normalize(inputs, scale=scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, _ = plan.normalize(inputs, scale=scale)
        inputs.normal_()
        graph.replay()
        expected = (
            inputs.float() * (inputs.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt() * (1 + scale)
        )
        torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("rows,k,n", [(10, 1024, 4096), (50, 4096, 1024), (576, 2048, 16384)])
def test_cublaslt_bf16_parity_and_independent_graphs(rows, k, n):
    _require_flashinfer()
    torch.manual_seed(59)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    backend = projection_backends.get(
        "flashinfer_cublaslt", OperatorRequest(inputs.device, inputs.dtype, cuda_graph=True)
    )
    scope = [("graph", 1)]
    with torch.inference_mode():
        plan = backend.plan(weight, 1.0, "bf16", None, lambda _: scope[0])
        graphs = []
        for index in (1, 2):
            scope[0] = ("graph", index)
            plan.apply(inputs, None)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = plan.apply(inputs, None)
            graphs.append((graph, output))
        assert graphs[0][1].data_ptr() != graphs[1][1].data_ptr()
        assert len(backend._workspaces) == 2
        for graph, output in graphs:
            inputs.normal_()
            graph.replay()
            torch.testing.assert_close(output, F.linear(inputs, weight), rtol=1e-2, atol=2e-2)
