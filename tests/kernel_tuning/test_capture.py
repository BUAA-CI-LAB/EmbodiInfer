"""Shape capture with duck-typed tensors and fake kernel modules; no Torch required."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from scripts.kernel_tuning import capture
from scripts.kernel_tuning.__main__ import main
from scripts.kernel_tuning.contracts import ContractError, TaskPackage
from scripts.kernel_tuning.generate import render
from scripts.kernel_tuning.operators import Operator, Tensor, catalog


class FakeTensor:
    """Just the metadata the mappings read: shape, dtype, numel, and values."""

    def __init__(self, *shape: int, dtype: str = "torch.bfloat16", values: list[int] | None = None) -> None:
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = False
        self._values = values

    def numel(self) -> int:
        count = 1
        for size in self.shape:
            count *= size
        return count

    def tolist(self) -> list[int]:
        assert self._values is not None
        return self._values


def record(name: str, **arguments: object) -> list[dict]:
    recorder = capture.Recorder()
    recorder.record(name, arguments)
    return recorder.rows()


def test_every_catalog_operator_has_a_mapping() -> None:
    assert capture.MAPPINGS.keys() == catalog().keys()


def test_production_calls_become_task_cases() -> None:
    (row,) = record(
        "gated_residual",
        residual=FakeTensor(8, 50, 1024),
        update=FakeTensor(8, 50, 1024),
        gate=FakeTensor(8, 1, 1024),
    )
    assert row == {"operator": "gated_residual", "axes": {"B": 8, "T": 50, "H": 1024, "G": 1}, "count": 1}
    (row,) = record("swiglu", packed=FakeTensor(1, 1, 37888))
    assert row["axes"] == {"M": 1, "I": 18944}
    (row,) = record(
        "split_kv_attention",
        query=FakeTensor(1, 8, 50, 256),
        prefix_key=FakeTensor(1, 1, 560, 256),
        prefix_value=FakeTensor(1, 1, 560, 256),
        suffix_key=FakeTensor(1, 1, 50, 256),
        suffix_value=FakeTensor(1, 1, 50, 256),
        mask=FakeTensor(1, 1, 1, 610, dtype="torch.float32"),
        scaling=0.0625,
    )
    assert row["axes"] == {"B": 1, "HQ": 8, "HK": 1, "P": 560, "S": 50, "D": 256}
    assert row["scalars"] == {"scaling": 0.0625}


def test_segment_offsets_are_captured_as_fixed_inputs_and_unsupported_calls_skipped() -> None:
    q = FakeTensor(96, 16, 80)
    offsets = FakeTensor(3, dtype="torch.int32", values=[0, 64, 96])
    common = {"q": q, "k": q, "v": q, "q_segment_offsets": offsets, "kv_segment_offsets": None}
    (row,) = record("segmented_attention", **common, scaling=None, max_query_length=64, max_key_length=64)
    assert row["axes"] == {"T": 96, "HQ": 16, "HK": 16, "D": 80, "N": 3}
    assert row["scalars"] == {"max_length": 64} and row["fixed"] == {"segment_offsets": [0, 64, 96]}
    (row,) = record("segmented_attention", **common, scaling=0.5, max_query_length=64, max_key_length=64)
    assert row == {"operator": "segmented_attention", "skipped": "non-default scaling", "count": 1}
    (row,) = record(
        "gated_gelu", gate=FakeTensor(1, 50, 4096, dtype="torch.float16"), up=FakeTensor(1, 50, 4096)
    )
    assert row["skipped"] == "dtype torch.float16"
    (row,) = record("gated_gelu", gate=FakeTensor(1, 50, 4096))
    assert row["skipped"].startswith("unmapped call: KeyError")


@pytest.fixture
def fake_kernel(monkeypatch: pytest.MonkeyPatch) -> Operator:
    """A kernel module plus a second module that imported the function by name."""
    module = types.ModuleType("fake_kernels.norm")

    def scale(x, factor=2.0):
        return [factor * v for v in x.values]

    module.scale = scale
    importer = types.ModuleType("fake_model")
    importer.scale = scale
    monkeypatch.setitem(sys.modules, "fake_kernels", types.ModuleType("fake_kernels"))
    monkeypatch.setitem(sys.modules, "fake_kernels.norm", module)
    monkeypatch.setitem(sys.modules, "fake_model", importer)
    monkeypatch.setitem(
        capture.MAPPINGS, "fake", lambda a: {"axes": {"N": len(a["x"].values)}, "scalars": {"f": a["factor"]}}
    )
    return Operator(
        name="fake",
        summary="",
        models=(),
        op_type="elementwise",
        source="fake_kernels/norm.py",
        entry="scale",
        call="scale(x)",
        axes={"N": None},
        inputs={"x": Tensor(("N",), "bfloat16")},
        outputs={"y": Tensor(("N",), "bfloat16")},
        reference="def run(x):\n    return x\n",
        workloads=(),
        notes="",
    )


def test_run_wraps_every_holder_records_calls_and_restores(fake_kernel: Operator, tmp_path: Path) -> None:
    script = tmp_path / "infer.py"
    script.write_text(
        "import sys, types, fake_model\n"
        "from fake_kernels.norm import scale\n"
        "x = types.SimpleNamespace(values=[1, 2, 3])\n"
        "assert fake_model.scale(x) == [2.0, 4.0, 6.0]\n"
        "scale(x, factor=3.0); scale(x, factor=3.0)\n"
        "assert sys.argv[1:] == ['--steps', '2']\n",
        encoding="utf-8",
    )
    rows = capture.run([str(script), "--steps", "2"], tmp_path / "calls.jsonl", [fake_kernel])
    assert rows == [
        {"operator": "fake", "axes": {"N": 3}, "scalars": {"f": 3.0}, "count": 2},
        {"operator": "fake", "axes": {"N": 3}, "scalars": {"f": 2.0}, "count": 1},
    ]
    assert capture.load(tmp_path / "calls.jsonl") == rows
    original = sys.modules["fake_kernels.norm"].scale
    assert not hasattr(original, "__wrapped_kernel__") and sys.modules["fake_model"].scale is original


def test_captured_cases_replace_estimated_workloads(tmp_path: Path) -> None:
    rows = [
        {"operator": "split_kv_attention", "axes": axes, "scalars": {"scaling": 0.0625}, "count": count}
        for axes, count in (
            ({"B": 1, "HQ": 8, "HK": 1, "P": 560, "S": 50, "D": 256}, 1800),
            ({"B": 1, "HQ": 8, "HK": 1, "P": 592, "S": 50, "D": 256}, 180),
        )
    ]
    rows.append({"operator": "split_kv_attention", "skipped": "dtype torch.float16", "count": 3})
    operator, summary = capture.apply(catalog()["split_kv_attention"], rows)
    assert [w.uuid for w in operator.workloads] == [
        "captured1-B1-HQ8-HK1-P560-S50-D256",
        "captured2-B1-HQ8-HK1-P592-S50-D256",
    ]
    assert [w.weight for w in operator.workloads] == [1800.0, 180.0]
    assert summary.endswith("2 captured shapes covering 100% of 1980 calls; 3 calls outside the task skipped")
    task = render(operator, tmp_path / "task")
    assert task.settings.weights == {w.uuid: w.weight for w in operator.workloads}
    assert "captured production calls" in (task.root / "README.md").read_text(encoding="utf-8")
    kept, note = capture.apply(catalog()["swiglu"], rows)
    assert kept == catalog()["swiglu"] and "keeping estimated shapes" in note
    with pytest.raises(ContractError, match="binds"):
        capture.apply(catalog()["swiglu"], [{"operator": "swiglu", "axes": {"M": 1}, "count": 1}])


def test_captured_segment_offsets_render_as_task_data(tmp_path: Path) -> None:
    rows = [
        {
            "operator": "segmented_attention",
            "axes": {"T": 96, "HQ": 16, "HK": 16, "D": 80, "N": 3},
            "scalars": {"max_length": 64},
            "fixed": {"segment_offsets": [0, 64, 96]},
            "count": 5,
        }
    ]
    operator, _ = capture.apply(catalog()["segmented_attention"], rows)
    task = render(operator, tmp_path / "task")
    assert "data/captured1-T96-HQ16-HK16-D80-N3.safetensors" in task.hashes


def test_cli_capture_and_generate_from_capture(
    fake_kernel: Operator, tmp_path: Path, monkeypatch, capsys
) -> None:
    script = tmp_path / "infer.py"
    script.write_text(
        "import types\nfrom fake_kernels.norm import scale\nscale(types.SimpleNamespace(values=[1]))\n"
    )
    monkeypatch.setattr(
        "scripts.kernel_tuning.operators.select", lambda names=None, models=None: [fake_kernel]
    )
    output = tmp_path / "calls.jsonl"
    assert main(["capture", "--output", str(output), "--", str(script)]) == 0
    assert "Captured 1 calls in 1 cases" in capsys.readouterr().out
    monkeypatch.undo()
    rows = [{"operator": "rms_norm", "axes": {"M": 1, "H": 3584}, "scalars": {"eps": 1e-6}, "count": 7}]
    captured = tmp_path / "rms.jsonl"
    captured.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    argv = ["generate", "rms_norm", "--output", str(tmp_path / "tasks"), "--captured", str(captured)]
    assert main(argv) == 0
    assert "rms_norm: 1 captured shapes covering 100% of 7 calls" in capsys.readouterr().out
    assert TaskPackage.load(tmp_path / "tasks/rms_norm").workload_ids == ("captured1-M1-H3584",)
