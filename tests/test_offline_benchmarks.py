"""Sampling and metric regressions for simulator-free performance benchmarks."""

from __future__ import annotations

import importlib.util
import io
import json
import runpy
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def load_benchmark(profile: str):
    path = Path(__file__).resolve().parents[1] / "benchmarks" / f"{profile}-benchmark" / "benchmark.py"
    spec = importlib.util.spec_from_file_location(f"offline_{profile}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stream = load_benchmark("streamvln")
measurement = libero_run = load_benchmark("pi05")
cosmos = load_benchmark("cosmos")
NavigationEpisode = stream.NavigationEpisode
load_navigation = stream.load_navigation
axis_angle_to_quaternion = cosmos.axis_angle_to_quaternion
load_libero, read_libero = libero_run.load_libero, libero_run.read_libero
uniform_indices = libero_run.uniform_indices
output_record, summarize, timed_call = (
    measurement.output_record,
    measurement.summarize,
    measurement.timed_call,
)


def navigation_fixture(root: Path) -> dict:
    """Write shuffled annotations and tiny recorded-frame stand-ins."""
    records = []
    for number in (9, 1, 3):
        video = f"images/scene_r2r_{number:06}"
        folder = root / video / "rgb"
        folder.mkdir(parents=True)
        for index in range(1, 5):
            Image.new("RGB", (4, 3), (number, index, 0)).save(folder / f"{index:03}.jpg")
        records.append(
            {
                "id": number,
                "video": video,
                "instructions": [f"instruction {number}"],
                "actions": [-1, 1, 2, 3],
            }
        )
    path = root / "annotations.json"
    path.write_text(json.dumps(records))
    return {"name": "R2R", "root": str(root), "annotations": str(path), "episode_limit": 2}


def test_navigation_selects_numeric_ids_and_aligns_initial_dummy(tmp_path: Path) -> None:
    episodes = load_navigation(navigation_fixture(tmp_path))
    assert [episode.episode_id for episode in episodes] == [1, 3]
    assert episodes[0].actions == (1, 2, 3, 0)
    assert episodes[0].instruction == "instruction 1"


def test_missing_selected_episode_never_substitutes_later_available_data(tmp_path: Path) -> None:
    config = navigation_fixture(tmp_path)
    (tmp_path / "images/scene_r2r_000001/rgb/002.jpg").unlink()
    with pytest.raises(ValueError, match="consecutive"):
        load_navigation(config)


def test_navigation_count_is_configurable_and_short_release_is_an_error(tmp_path: Path) -> None:
    config = navigation_fixture(tmp_path)
    config["episode_limit"] = 1
    assert len(load_navigation(config)) == 1
    config["episode_limit"] = 4
    with pytest.raises(ValueError, match="release has 3"):
        load_navigation(config)


def test_stream_warmup_uses_a_long_enough_selected_episode_without_changing_selection() -> None:
    script = Path(__file__).resolve().parents[1] / "benchmarks/streamvln-benchmark/benchmark.py"
    select = runpy.run_path(str(script))["select_warmup_episode"]
    episodes = tuple(
        NavigationEpisode(
            "RxR",
            episode_id,
            f"images/{episode_id}",
            "Walk forward",
            tuple(Path(f"{index:03}.jpg") for index in range(1, length + 1)),
            (1,) * length,
        )
        for episode_id, length in ((1, 22), (5, 40))
    )
    datasets = [({"name": "RxR"}, episodes)]
    assert select(datasets, 33) is episodes[1]
    assert select(datasets, 10) is episodes[0]
    assert [episode.episode_id for episode in datasets[0][1]] == [1, 5]
    with pytest.raises(ValueError, match="no selected trajectory"):
        select([({"name": "RxR"}, episodes[:1])], 33)


def test_split_public_archive_extracts_only_selected_complete_episodes(tmp_path: Path) -> None:
    config = navigation_fixture(tmp_path)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted((tmp_path / "images").rglob("*.jpg")):
            archive.add(path, arcname=str(path.relative_to(tmp_path)))
    payload = buffer.getvalue()
    middle = len(payload) // 2
    paths = [tmp_path / "images.tar.gz.part0", tmp_path / "images.tar.gz.part1"]
    for path, value in zip(paths, (payload[:middle], payload[middle:])):
        path.write_bytes(value)
    output = tmp_path / "extracted"
    script = Path(__file__).resolve().parents[1] / "benchmarks/streamvln-benchmark/prepare_data.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--annotations",
            config["annotations"],
            "--archives",
            *map(str, paths),
            "--output-root",
            str(output),
            "--episodes",
            "2",
        ],
        check=True,
    )
    config["root"] = str(output)
    assert [episode.episode_id for episode in load_navigation(config)] == [1, 3]
    assert not (output / "images/scene_r2r_000009").exists()


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_sampling_rejects_invalid_counts(count) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        uniform_indices(16, count)


def test_uniform_sampling_never_duplicates_or_exceeds_available_frames() -> None:
    assert uniform_indices(10, 4) == (0, 3, 6, 9)
    assert uniform_indices(5, 1) == (0,)
    with pytest.raises(ValueError, match="distinct frames"):
        uniform_indices(3, 4)


def libero_fixture(root: Path) -> dict:
    """Create two small HDF5 tasks with deliberately nonlexical demo IDs."""
    h5py = pytest.importorskip("h5py")
    for task in ("b", "a"):
        with h5py.File(root / f"{task}.hdf5", "w") as handle:
            data = handle.create_group("data")
            data.attrs["problem_info"] = json.dumps({"language_instruction": f"task {task}"})
            data.attrs["macros_image_convention"] = "opengl"
            for demo in ("demo_10", "demo_2", "demo_0"):
                obs = data.create_group(f"{demo}/obs")
                for name in ("agentview_rgb", "eye_in_hand_rgb"):
                    obs.create_dataset(name, data=np.zeros((9, 3, 4, 3), dtype=np.uint8))
                for name, width in (("ee_pos", 3), ("ee_ori", 3), ("gripper_states", 2)):
                    obs.create_dataset(name, data=np.zeros((9, width)))
    return {
        "root": str(root),
        "task_limit": 2,
        "demos_per_task": 2,
        "frames_per_demo": 3,
        "sample_limit": None,
    }


def test_libero_default_rules_and_configurable_limit_share_frame_identity(tmp_path: Path) -> None:
    config = libero_fixture(tmp_path)
    samples = load_libero(config)
    assert len(samples) == 12
    assert [sample.sample_id for sample in samples[:4]] == [
        "libero_10/a.hdf5/demo_0/0",
        "libero_10/a.hdf5/demo_0/4",
        "libero_10/a.hdf5/demo_0/8",
        "libero_10/a.hdf5/demo_2/0",
    ]
    assert read_libero(samples[0])["agentview_rgb"].size == (4, 3)
    config["sample_limit"] = 5
    assert load_libero(config) == samples[:5]
    config["sample_limit"] = 13
    with pytest.raises(ValueError, match="exceeds selected"):
        load_libero(config)


def test_libero_missing_proprio_cannot_be_filled_with_synthetic_zeros(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    config = libero_fixture(tmp_path)
    with h5py.File(tmp_path / "a.hdf5", "a") as handle:
        del handle["data/demo_0/obs/ee_ori"]
    with pytest.raises(KeyError):
        load_libero(config)


def test_cosmos_state_rotation_reconstruction_uses_xyzw() -> None:
    np.testing.assert_allclose(axis_angle_to_quaternion(np.zeros(3)), [0, 0, 0, 1])
    np.testing.assert_allclose(axis_angle_to_quaternion(np.array([0, 0, np.pi])), [0, 0, 1, 0], atol=1e-7)


def test_throughput_uses_total_time_and_reports_distinct_output_units() -> None:
    report = summarize(
        [
            {"latency_ms": 10, "action_slots": 16, "generated_tokens": 0},
            {"latency_ms": 90, "action_slots": 16, "generated_tokens": 0},
        ]
    )
    assert report["calls_per_second"] == 20
    assert report["action_slots_per_second"] == 320
    assert report["generated_tokens_per_second"] == 0
    assert report["latency_ms"]["p50"] == 50
    assert report["latency_ms"]["p95"] == 86
    with pytest.raises(ValueError, match="without measured calls"):
        summarize([])


def test_latency_includes_device_completion_but_excludes_report_hashing(monkeypatch) -> None:
    clock = {"ns": 0}

    def synchronize(device) -> None:
        clock["ns"] += 2_000_000

    def infer() -> dict:
        clock["ns"] += 5_000_000
        return output_record(torch.ones(2, 7))

    original_hash = measurement.hashlib.sha256

    def slow_report_hash(value):
        clock["ns"] += 100_000_000
        return original_hash(value)

    monkeypatch.setattr(measurement.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(measurement.time, "perf_counter_ns", lambda: clock["ns"])
    monkeypatch.setattr(measurement.hashlib, "sha256", slow_report_hash)
    row = timed_call(infer, torch.device("cuda"))
    assert row["latency_ms"] == 7
    assert clock["ns"] == 109_000_000
    assert row["action_slots"] == 2
    assert len(row["output_sha256"]) == 64
    assert "_cpu_output" not in row


def test_nonfinite_model_outputs_cannot_be_reported_as_successful_calls(monkeypatch) -> None:
    monkeypatch.setattr(measurement.torch.cuda, "synchronize", lambda device: None)
    with pytest.raises(ValueError, match="finite output"):
        timed_call(lambda: output_record(torch.full((2, 7), float("nan"))), torch.device("cuda"))


@pytest.mark.parametrize("captures,capture_during_measurement", [(0, False), (1, False), (1, True)])
def test_libero_graph_report_requires_warm_capture_without_timed_growth(
    tmp_path: Path, monkeypatch, captures: int, capture_during_measurement: bool
) -> None:
    config = {
        "dataset": libero_fixture(tmp_path),
        "model": "fixture",
        "validate_data_only": False,
        "warmup_calls": 2,
        "cuda_graph": True,
        "seed": 42,
        "repeats": 1,
        "output": str(tmp_path / "report.json"),
    }
    calls = []
    reports = []

    def build(config, device):
        def infer(sample, raw):
            calls.append(sample.sample_id)
            return output_record(torch.ones(2, 7))

        def stats():
            count = captures + int(capture_during_measurement and len(calls) > 2)
            return {"graphs": {"capture_count": count}}

        return infer, {}, stats

    monkeypatch.setattr(libero_run, "load_config", lambda path: (config, path))
    monkeypatch.setattr(libero_run, "cuda_device", lambda config: torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(libero_run, "write_report", lambda *args: reports.append(args))
    if captures == 0 or capture_during_measurement:
        with pytest.raises(RuntimeError, match="CUDA Graph"):
            libero_run.run(tmp_path / "config.yaml", build)
        assert not reports
    else:
        libero_run.run(tmp_path / "config.yaml", build)
        assert len(reports[0][2]) == 12
        assert len(calls) == 14
    assert "/a.hdf5/" in calls[0]
    assert "/b.hdf5/" in calls[1]


@pytest.mark.parametrize("profile", ["dm05", "gr00t", "openvla-oft"])
def test_new_libero_profiles_preserve_existing_sample_identity(tmp_path: Path, profile: str) -> None:
    """New model timing must use the exact PI0.5 task/demo/frame selection."""
    config = libero_fixture(tmp_path)
    benchmark = load_benchmark(profile)
    expected = [sample.sample_id for sample in load_libero(config)]
    selected = [sample.sample_id for sample in benchmark.load_libero(config)]
    assert selected == expected
    assert benchmark.digest_json(selected) == measurement.digest_json(expected)


def test_lingbot_native_joint_mapping_round_trips_grippers_without_using_padding(tmp_path: Path) -> None:
    """The checkpoint places two grippers at 14:16, after a two-column arm pad."""
    benchmark = load_benchmark("lingbot-vla")
    stats = {}
    for prefix in ("observation.state", "action"):
        for key, width in (("arm.position", 12), ("effector.position", 2)):
            stats[f"{prefix}.{key}"] = {"q01": [-2.0] * width, "q99": [2.0] * width}
    path = tmp_path / "norm.json"
    path.write_text(json.dumps({"norm_stats": stats}))
    processor = benchmark.RobotwinProcessor(path)
    sample = benchmark.RobotwinSample(tmp_path / "episode0.hdf5", "task", 0, 0, "move")
    raw = {
        "state": np.arange(14, dtype=np.float32) / 10,
        "images": [Image.new("RGB", (8, 6), (1, 2, 3)) for _ in range(3)],
    }
    obs = processor.prepare(sample, raw)
    assert obs.state.shape == (75,)
    assert obs.images.shape == (3, 3, 224, 224)
    assert torch.count_nonzero(obs.state[12:14]) == 0
    actions = obs.state.expand(50, -1).clone()
    actions[:, 12:14] = 999  # Structural padding must never leak into physical actions.
    restored = processor.restore(actions)
    torch.testing.assert_close(restored[0], torch.from_numpy(raw["state"]), atol=3e-7, rtol=0)
    assert restored.shape == (50, 14)


@pytest.mark.parametrize(
    "profile", ["dm05", "gr00t", "lingbot-vla", "openvla-oft", "pi05", "cosmos", "streamvln", "qwenvl"]
)
def test_model_timing_covers_both_stages_and_excludes_output_transform(monkeypatch, profile):
    module = load_benchmark(profile)
    ticks = [0.0]
    records = []

    class Event:
        def __init__(self, **kwargs):
            self.value = None

        def record(self, stream):
            self.value = ticks[0]
            records.append(self.value)

        def synchronize(self):
            records.append("done")

        def elapsed_time(self, other):
            return other.value - self.value

    monkeypatch.setattr(module.torch.cuda, "Event", Event)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda device: records.append("ready"))
    monkeypatch.setattr(module.torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: int(ticks[0] * 1e6))
    ticks[0] += 100  # Already-finished preprocessing must not be charged to model time.

    def prefill():
        ticks[0] += 7
        return "prefix"

    def decode(prefix):
        assert prefix == "prefix"
        for _ in range(10):
            ticks[0] += 2
        return "model actions"

    prefix, actions, timing = module.timed_model(prefill, decode, torch.device("cuda"))
    ticks[0] += 300  # Action restoration is after the measured interval.
    assert (prefix, actions) == ("prefix", "model actions")
    assert timing == {
        "prefill_ms": 7.0,
        "decode_ms": 20.0,
        "gpu_inference_ms": 27.0,
        "pure_inference_ms": 27.0,
    }
    assert records == ["ready", 100, 107, 127, "done"]
    rows = [
        {"latency_ms": 500, "action_slots": 10, "generated_tokens": 0, "model_timing_ms": timing},
        {
            "latency_ms": 600,
            "action_slots": 10,
            "generated_tokens": 0,
            "model_timing_ms": {k: v * 3 for k, v in timing.items()},
        },
    ]
    metrics = module.summarize(rows)
    assert metrics["latency_ms"]["mean"] == 550
    assert metrics["model_timing_ms"]["pure_inference_ms"]["mean"] == 54
    assert metrics["model_calls_per_second"] == pytest.approx(2 / 0.108)
    assert metrics["calls_per_second"] == pytest.approx(2 / 1.1)
