"""CPU validation of Pi05 diagnostic capture hooks; no CUDA/model parity claim."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
spec = importlib.util.spec_from_file_location(
    "pi05_numerical_capture", ROOT / "benchmarks/pi05-benchmark/validate_nonquant.py"
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class CaptureTests(unittest.TestCase):
    """Exercise instrumentation ownership, step accounting, and method restoration."""

    def test_fp32_diagnostic_cache_is_bounded_without_changing_gemm_results(self) -> None:
        from unittest.mock import patch

        from embodiinfer.policies.pi05 import modeling_pi05

        first, second = torch.nn.Linear(3, 2, bias=False).eval(), torch.nn.Linear(3, 2, bias=False).eval()
        inputs = torch.randn(4, 3)
        original = modeling_pi05._fused_linear_pair
        with (
            torch.inference_mode(),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            expected = original(inputs, first, second)
            with adapter.ephemeral_packed_mlp_weights():
                for _ in range(3):
                    actual = modeling_pi05._fused_linear_pair(inputs, first, second)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
                    self.assertIsNone(first._vvla_fused_pair_weight)
        self.assertIs(modeling_pi05._fused_linear_pair, original)

    def test_container_provenance_rejects_source_substitution(self) -> None:
        from unittest.mock import patch

        # Git metadata is supplied by the host in the minimal GPU container.
        # Hash real runtime bytes here; the unit test needs no Git executable.
        with patch.object(
            adapter.subprocess, "check_output", side_effect=["f" * 40, " M host-fixture"]
        ) as git:
            provenance = adapter.source_provenance(ROOT)
            with tempfile.TemporaryDirectory() as directory:
                manifest = Path(directory) / "manifest.json"
                manifest.write_text(json.dumps(provenance))
                checked = adapter.source_provenance(ROOT, manifest)
                self.assertEqual(checked["source_sha256"], provenance["source_sha256"])
                self.assertEqual(checked["dirty_status"], provenance["dirty_status"])
                name = next(iter(provenance["files"]))
                provenance["files"][name] = "0" * 64
                manifest.write_text(json.dumps(provenance))
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    adapter.source_provenance(ROOT, manifest)
            self.assertEqual(git.call_count, 2)

    def make_policy(self) -> SimpleNamespace:
        """Build the narrow Pi05 interface the recorder observes with CPU operations."""
        policy = SimpleNamespace(_prefix_tower=SimpleNamespace(layers=[0, 1]))
        prefix = SimpleNamespace(
            kv=[(torch.ones(1, 2), torch.full((1, 2), 2.0)) for _ in range(2)],
            prefix_pad_masks=torch.ones(1, 2, dtype=torch.bool),
        )
        policy._encode_prefix_impl = lambda: prefix
        policy._denoise_step_impl = lambda state, time, prefix: state * 2 + time

        def integrate(state: torch.Tensor, prefix: SimpleNamespace, steps: int) -> torch.Tensor:
            for step in range(steps):
                time = torch.tensor(1 - step / steps)
                velocity = policy._denoise_step_impl(state, time, prefix)
                state = state + velocity * (-1 / steps)
            return state

        policy._runtime = SimpleNamespace(integrate=integrate)
        return policy

    def test_exact_native_trace_and_method_restoration(self) -> None:
        policy = self.make_policy()
        original = (policy._encode_prefix_impl, policy._denoise_step_impl, policy._runtime.integrate)
        active = [False]
        with adapter.CaptureTrace(policy, 2, lambda: active[0]) as recorder:
            # Constructor warmup calls must not consume any trace steps.
            prefix = policy._encode_prefix_impl()
            policy._runtime.integrate(torch.ones(1, 2), prefix, 2)
            self.assertEqual(recorder.buffers, {})
            active[0] = True
            prefix = policy._encode_prefix_impl()
            state = torch.ones(1, 2)
            output = policy._runtime.integrate(state, prefix, 2)
            trace = recorder.snapshot(output)
            self.assertEqual(recorder.step, 2)
            np.testing.assert_array_equal(trace["state/001"].values, [[-0.5, -0.5]])
            np.testing.assert_array_equal(trace["state/002"].values, [[-0.25, -0.25]])
            state.fill_(99)
            np.testing.assert_array_equal(trace["state/000"].values, [[1, 1]])
            self.assertEqual(trace["prefix/mask"].dtype, "torch.bool")
        self.assertEqual(
            (policy._encode_prefix_impl, policy._denoise_step_impl, policy._runtime.integrate), original
        )

    def test_incomplete_capture_is_not_a_pass(self) -> None:
        policy = self.make_policy()
        with (
            adapter.CaptureTrace(policy, 2, lambda: False) as recorder,
            self.assertRaisesRegex(RuntimeError, "incomplete graph trace"),
        ):
            recorder.snapshot(torch.ones(1, 2))

    def test_extra_steps_and_exceptions_restore_methods(self) -> None:
        policy = self.make_policy()
        original = policy._denoise_step_impl
        with (
            self.assertRaisesRegex(RuntimeError, "extra denoising"),
            adapter.CaptureTrace(policy, 1, lambda: True),
        ):
            prefix = policy._encode_prefix_impl()
            policy._runtime.integrate(torch.ones(1, 2), prefix, 2)
        self.assertIs(policy._denoise_step_impl, original)

    def test_bfloat16_snapshot_keeps_original_bits(self) -> None:
        tensor = torch.tensor([0.0, -0.0, 1.0], dtype=torch.bfloat16)
        snapshot = adapter.tensor_snapshot(tensor)
        self.assertEqual(snapshot.dtype, "torch.bfloat16")
        self.assertEqual(snapshot.values.dtype, np.float32)
        self.assertEqual(len(snapshot.storage), 6)
        self.assertEqual(snapshot.storage, tensor.view(torch.uint8).numpy().tobytes())
        tensor.fill_(2)
        np.testing.assert_array_equal(snapshot.values, [0, 0, 1])

    def test_fp32_diagnostic_promotes_captured_noise_without_redrawing(self) -> None:
        original = torch.tensor([[[0.0, -0.0, 0.123, -5.25]]], dtype=torch.bfloat16)
        snapshot = adapter.tensor_snapshot(original)
        promoted = adapter.fp32_initial_state(snapshot, "cpu")
        self.assertEqual(promoted.dtype, torch.float32)
        self.assertTrue(torch.equal(promoted, original.float()))
        self.assertTrue(torch.signbit(promoted[0, 0, 1]))
        promoted.fill_(99)
        np.testing.assert_array_equal(snapshot.values, original.float().numpy())

    def test_saved_trace_can_be_read_without_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            snapshot = adapter.tensor_snapshot(torch.tensor([1.0, -0.0]))
            adapter.save_trace(path, {"state/000": snapshot})
            with np.load(path, allow_pickle=False) as result:
                np.testing.assert_array_equal(result["values_0"], [1, 0])
                self.assertEqual(result["bytes_0"].tobytes(), snapshot.storage)
                self.assertIn("state/000", str(result["manifest_json"]))


if __name__ == "__main__":
    unittest.main()
