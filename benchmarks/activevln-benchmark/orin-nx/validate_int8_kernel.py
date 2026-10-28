"""Check calibrated W8A8 rounding and CUDA Graph replay on the executing device."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
import triton
from nx_int8_kernel import _quantize_rows, int8_linear
from torch.nn import functional


@torch.inference_mode()
def validate(output: Path) -> None:
    """Require bit-exact payloads, scales and outputs against native INT32 GEMM."""
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    cases = []
    shapes = [
        (1, 2048, 11008),
        (31, 2048, 11008),
        (65, 11008, 2048),
        (257, 1280, 3420),
        (33, 3420, 1280),
        (3, 97, 129),
    ]
    for dtype in (torch.bfloat16, torch.float16):
        for index, (rows, depth, width) in enumerate(shapes):
            inputs = torch.randn((rows, depth), device="cuda", dtype=dtype)
            if rows > 1:
                inputs[0].zero_()
            weight = torch.randint(-127, 128, (depth, width * 2), device="cuda", dtype=torch.int8)[:, ::2]
            weight_scale = torch.rand(width, device="cuda", dtype=torch.float32) * 0.01
            channel_scale = torch.rand(depth, device="cuda", dtype=torch.float32) + 0.1 if index % 2 else None
            bias = torch.randn(width, device="cuda", dtype=dtype) if index % 3 else None
            smoothed = inputs if channel_scale is None else (inputs.float() / channel_scale).to(dtype)
            maxima = smoothed.float().abs().amax(1)
            scales = torch.where(maxima > 0, maxima * (1.0 / 127.0), torch.ones_like(maxima))
            payload = torch.round(smoothed.float() / scales[:, None]).to(torch.int8)
            actual_payload = torch.empty_like(payload)
            actual_scales = torch.empty_like(scales)
            _quantize_rows[(rows,)](
                inputs,
                actual_payload,
                actual_scales,
                weight_scale if channel_scale is None else channel_scale,
                depth,
                inputs.stride(0),
                channel_scale is not None,
                triton.next_power_of_2(depth),
            )
            torch.testing.assert_close(actual_payload, payload, rtol=0, atol=0)
            torch.testing.assert_close(actual_scales, scales, rtol=0, atol=0)
            # Native GEMM alignment padding contributes exact integer zeros.
            padded_m = triton.cdiv(rows, 32) * 32
            padded_k = triton.cdiv(depth, 32) * 32
            padded_n = triton.cdiv(width, 32) * 32
            q = functional.pad(payload, (0, padded_k - depth, 0, padded_m - rows))
            w = functional.pad(weight.contiguous(), (0, padded_n - width, 0, padded_k - depth))
            accumulated = torch._int_mm(q, w)[:rows, :width]
            reference = accumulated.float() * scales[:, None]
            reference = reference * weight_scale[None, :]
            if bias is not None:
                reference = reference + bias.float()
            reference = reference.to(dtype)

            forward = partial(
                int8_linear,
                inputs,
                weight,
                weight_scale,
                bias,
                input_scale=channel_scale,
            )

            actual = forward()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    forward()
            torch.cuda.current_stream().wait_stream(warmup)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = forward()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, reference, rtol=0, atol=0)
            cases.append(
                {
                    "shape": [rows, depth, width],
                    "dtype": str(dtype),
                    "smoothing": channel_scale is not None,
                    "bias": bias is not None,
                    "strided_weight": True,
                    "quantization_exact": True,
                    "eager_exact": True,
                    "graph_exact": True,
                }
            )
    record = {
        "status": "complete",
        "exact": True,
        "cases": cases,
        "reference": "INT32 torch._int_mm then ordered FP32 input-scale/weight-scale/bias, one final dtype cast",
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "kernel_sha256": hashlib.sha256(
            Path(__file__).with_name("nx_int8_kernel.py").read_bytes()
        ).hexdigest(),
    }
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps({"status": record["status"], "cases": len(cases), "exact": True}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    validate(parser.parse_args().output)
