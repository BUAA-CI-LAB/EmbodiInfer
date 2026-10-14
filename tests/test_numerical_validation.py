"""CPU-only regression tests for benchmark gates; also runnable with unittest."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
from numerical_validation import (  # noqa: E402
    LOSSLESS,
    Snapshot,
    Thresholds,
    benchmark_paired_requests,
    benchmark_requests,
    compare,
    compare_traces,
    gripper_flips,
    synchronized_call,
    timing_summary,
)


def snapshot(value: object, dtype: object = np.float32) -> Snapshot:
    """Make an owned reference for a tiny synthetic test."""
    return Snapshot.from_numpy(np.asarray(value, dtype=dtype))


class NumericalGateTests(unittest.TestCase):
    """Synthetic arithmetic cannot certify model parity, but the gate must be strict."""

    def test_lossless_threshold_is_fixed_zero(self) -> None:
        self.assertEqual((LOSSLESS.max_abs, LOSSLESS.rms, LOSSLESS.relative_l2), (0, 0, 0))
        self.assertTrue(LOSSLESS.exact_bits)

    def test_identical_passes_with_exact_bits(self) -> None:
        result = compare(snapshot([1, -2, 0]), snapshot([1, -2, 0]))
        self.assertTrue(result["passed"])
        self.assertTrue(result["exact_bits"])
        self.assertEqual(result["max_abs"], 0)
        json.dumps(result, allow_nan=False)

    def test_metrics_and_strict_threshold(self) -> None:
        result = compare(snapshot([3, 4]), snapshot([3, 5]))
        self.assertFalse(result["passed"])
        self.assertEqual(result["max_abs"], 1)
        self.assertAlmostEqual(result["rms"], 1 / np.sqrt(2))
        self.assertAlmostEqual(result["relative_l2"], 0.2)

    def test_signed_zero_is_a_bit_difference(self) -> None:
        result = compare(snapshot([0.0]), snapshot([-0.0]))
        self.assertFalse(result["passed"])
        self.assertEqual(result["max_abs"], 0)
        self.assertIn("bit_mismatch", result["failures"])

    def test_equal_values_cannot_hide_changed_dtype(self) -> None:
        result = compare(snapshot([1], np.float32), snapshot([1], np.float64))
        self.assertFalse(result["passed"])
        self.assertIn("dtype_mismatch", result["failures"])

    def test_broadcastable_shape_mismatch_rejected(self) -> None:
        result = compare(snapshot([[1, 2]]), snapshot([1, 2]))
        self.assertEqual(result["failures"], ["shape_mismatch"])

    def test_empty_rejected(self) -> None:
        self.assertFalse(compare(snapshot([]), snapshot([]))["passed"])

    def test_nonfinite_fails_even_if_bytes_equal(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf")):
            result = compare(snapshot([value]), snapshot([value]))
            self.assertFalse(result["passed"])
            self.assertIn("nonfinite_tensor", result["failures"])
            json.dumps(result, allow_nan=False)

    def test_zero_reference_uses_declared_floor(self) -> None:
        result = compare(snapshot([0]), snapshot([1]))
        self.assertEqual(result["relative_l2"], 1 / LOSSLESS.relative_floor)

    def test_rms_avoids_square_overflow(self) -> None:
        result = compare(snapshot([1e200], np.float64), snapshot([1e200], np.float64))
        self.assertTrue(result["passed"])
        self.assertEqual(result["reference_rms"], 1e200)

    def test_difference_overflow_fails_with_json_safe_output(self) -> None:
        result = compare(snapshot([1e308], np.float64), snapshot([-1e308], np.float64))
        self.assertFalse(result["passed"])
        self.assertIn("nonfinite_metric", result["failures"])
        json.dumps(result, allow_nan=False)

    def test_thresholds_must_be_valid(self) -> None:
        for kwargs in ({"max_abs": -1}, {"rms": float("inf")}, {"relative_floor": 0}):
            with self.assertRaises(ValueError):
                Thresholds(**kwargs)

    def test_reference_owns_storage(self) -> None:
        data = np.array([1, 2], dtype=np.float32)
        reference = Snapshot.from_numpy(data)
        data[:] = 7
        self.assertEqual(reference.values.tolist(), [1, 2])
        self.assertFalse(reference.values.flags.writeable)

    def test_trace_keys_must_match(self) -> None:
        result = compare_traces({"prefix/0": snapshot([1])}, {"prefix/1": snapshot([1])})
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing"], ["prefix/0"])
        self.assertEqual(result["unexpected"], ["prefix/1"])
        self.assertFalse(compare_traces({}, {})["passed"])

    def test_gripper_threshold_equality_is_explicit(self) -> None:
        reference, candidate = snapshot([[0.0], [1.0]]), snapshot([[-0.1], [1.0]])
        gt = gripper_flips(reference, candidate, index=0, threshold=0, comparison="gt")
        ge = gripper_flips(reference, candidate, index=0, threshold=0, comparison="ge")
        self.assertEqual(gt["flips"], 0)
        self.assertEqual(ge["flips"], 1)
        self.assertEqual(ge["flip_indices"], [[0]])
        self.assertEqual(ge["reference_at_threshold"], 1)

    def test_gripper_rejects_undefined_cases(self) -> None:
        for kwargs in ({"index": -1}, {"index": 2}, {"threshold": float("nan")}, {"comparison": "close"}):
            values = dict(index=0, threshold=0, comparison="gt") | kwargs
            with self.assertRaises(ValueError):
                gripper_flips(snapshot([[1]]), snapshot([[1]]), **values)
        with self.assertRaises(ValueError):
            gripper_flips(snapshot([[float("nan")]]), snapshot([[1]]), index=0, threshold=0, comparison="gt")


class TimingTests(unittest.TestCase):
    """Make timing boundaries observable without sleeping or requiring CUDA."""

    def test_paired_timing_balances_order_and_preserves_all_samples(self) -> None:
        calls = []
        ticks = iter(float(i) for i in range(20))
        result = benchmark_paired_requests(
            {
                "base": lambda case: calls.append(("base", case)),
                "candidate": lambda case: calls.append(("candidate", case)),
            },
            cases=2,
            warmup=1,
            iterations=4,
            synchronize=lambda: None,
            clock=lambda: next(ticks),
        )
        self.assertEqual(calls[:4], [("base", 0), ("candidate", 0), ("base", 1), ("candidate", 1)])
        self.assertEqual(calls[4:8], [("base", 0), ("candidate", 0), ("candidate", 1), ("base", 1)])
        self.assertEqual(len(result["raw_samples"]), 8)
        self.assertEqual([row["position"] for row in result["raw_samples"]], [0, 1] * 4)
        self.assertTrue(all(row["wall_ms"] == 1000 for row in result["raw_samples"]))

    def test_percentiles_and_samples(self) -> None:
        result = timing_summary([1, 2, 3, 4, 5])
        self.assertEqual(result["p50_ms"], 3)
        self.assertEqual(result["p95_ms"], 4.8)
        self.assertEqual(result["count"], 5)

    def test_timing_rejects_invalid_samples(self) -> None:
        for values in ([], [-1], [float("nan")], [[1]]):
            with self.assertRaises(ValueError):
                timing_summary(values)

    def test_warmup_reset_synchronization_and_raw_rows(self) -> None:
        calls = []
        ticks = iter([0, 0.010, 0.1, 0.120, 1, 1.001, 2, 2.003, 3, 3.002])
        result = benchmark_requests(
            lambda case: calls.append(("request", case)),
            cases=2,
            warmup=2,
            iterations=3,
            synchronize=lambda: calls.append("sync"),
            before_request=lambda case: calls.append(("reset", case)),
            clock=lambda: next(ticks),
        )
        self.assertEqual([row["case"] for row in result["raw_samples"]], [0, 1, 0])
        self.assertEqual(
            calls[:12],
            ["sync"]
            + [("reset", 0), ("request", 0)] * 2
            + ["sync", "sync"]
            + [("reset", 1), ("request", 1)] * 2
            + ["sync"],
        )
        self.assertEqual(calls[12:16], [("reset", 0), "sync", ("request", 0), "sync"])
        self.assertAlmostEqual(result["warmup_timing"]["total_wall_ms"], 30)
        self.assertEqual(len(result["warmup_timing"]["raw_samples"]), 2)
        self.assertAlmostEqual(result["summary"]["p50_ms"], 2)
        self.assertEqual(len(result["raw_samples"]), 3)

    def test_cold_setup_measurement_is_synchronized(self) -> None:
        calls = []
        ticks = iter([3, 3.25])
        result, elapsed = synchronized_call(
            lambda: calls.append("operation") or "output",
            lambda: calls.append("sync"),
            lambda: next(ticks),
        )
        self.assertEqual(calls, ["sync", "operation", "sync"])
        self.assertEqual(result, "output")
        self.assertEqual(elapsed, 250)

    def test_timing_requires_warmup(self) -> None:
        with self.assertRaises(ValueError):
            benchmark_requests(lambda _: None, cases=1, warmup=0, iterations=1, synchronize=lambda: None)


class AdapterImportTests(unittest.TestCase):
    """The CLI's configuration and helpers remain inspectable without GPU/model dependencies."""

    def test_cli_import_and_explicit_gripper_semantics(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "validate_nonquant", BENCHMARKS / "pi05-benchmark/validate_nonquant.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parser = module.make_parser()
        args = parser.parse_args(
            [
                "--checkpoint",
                "weights",
                "--inputs",
                "inputs.pt",
                "--out",
                "results",
                "--physical-action-dim",
                "7",
                "--gripper-index",
                "6",
                "--gripper-threshold",
                "0.5",
                "--gripper-comparison",
                "ge",
            ]
        )
        self.assertIsNone(args.steps)
        self.assertIsNone(args.horizon)
        self.assertEqual((args.threads, args.interop_threads), (1, 1))
        self.assertEqual(args.gripper_comparison, "ge")
        restored = module.physical_actions(np.array([[[1, 2, 99]]]), np.array([4, 5]), np.array([2, 3]), 2)
        np.testing.assert_array_equal(restored, np.array([[[6, 11]]]))

    def test_sampling_defaults_follow_checkpoint_and_overrides_are_explicit(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "validate_nonquant_sampling", BENCHMARKS / "pi05-benchmark/validate_nonquant.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        native = SimpleNamespace(max_action_dim=32, chunk_size=50, num_inference_steps=12)
        result = module.resolve_sampling(native, None, None)
        self.assertEqual(result["resolved"], dict(action_dim=32, horizon=50, steps=12))
        self.assertTrue(result["matches_checkpoint"])
        self.assertEqual(result["explicit_overrides"], dict(horizon=None, steps=None))
        shortened = module.resolve_sampling(native, 10, 8)
        self.assertEqual(shortened["resolved"], dict(action_dim=32, horizon=10, steps=8))
        self.assertFalse(shortened["matches_checkpoint"])
        self.assertEqual(shortened["checkpoint"]["horizon"], 50)
        self.assertEqual(shortened["explicit_overrides"], dict(horizon=10, steps=8))
        for horizon, steps in ((51, None), (0, None), (None, 0)):
            with self.assertRaises(ValueError):
                module.resolve_sampling(native, horizon, steps)
        with self.assertRaises(ValueError):
            module.resolve_sampling(SimpleNamespace(), None, None)

    def test_thread_configuration_records_exact_values_and_blocker(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "validate_nonquant_threads", BENCHMARKS / "pi05-benchmark/validate_nonquant.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        settings = {"intraop": 8, "interop": 4}

        def fixed_interop(value: int) -> None:
            raise RuntimeError("inter-op pool already initialized")

        api = SimpleNamespace(
            set_num_threads=lambda value: settings.update(intraop=value),
            get_num_threads=lambda: settings["intraop"],
            set_num_interop_threads=fixed_interop,
            get_num_interop_threads=lambda: settings["interop"],
        )
        result = module.configure_threads(api, 1, 1)
        self.assertEqual(result["actual_intraop"], 1)
        self.assertEqual(result["actual_interop"], 4)
        self.assertEqual(result["requested_interop"], 1)
        self.assertIn("already initialized", result["interop_configuration_error"])
        api.set_num_interop_threads = lambda value: settings.update(interop=value)
        result = module.configure_threads(api, 2, 3)
        self.assertEqual(result["actual_intraop"], 2)
        self.assertEqual(result["actual_interop"], 3)
        self.assertIsNone(result["interop_configuration_error"])
        with self.assertRaises(ValueError):
            module.configure_threads(api, 0, 1)


if __name__ == "__main__":
    unittest.main()
