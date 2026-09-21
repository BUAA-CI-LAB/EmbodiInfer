from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "3B-navigation"
sys.path.insert(0, str(BENCHMARK_DIR))
import benchmark as full_policy_benchmark  # noqa: E402
import dataset as benchmark_dataset  # noqa: E402
import prepare_r2r_vlnce_manifest as prepare  # noqa: E402

LOW_PROFILE = "qwen2.5-vl-3b-r2r-low-level"


def _fixture(tmp_path: Path, size: tuple[int, int] = (640, 480)):
    stream_root = tmp_path / "StreamVLN" / "R2R"
    images_root = stream_root / "images"
    train_episodes, annotations = [], []
    actions = [-1, 1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3]
    for episode_id in range(2):
        scene = f"scene{episode_id}"
        instruction = f"Go to target {episode_id}."
        train_episodes.append(
            {
                "episode_id": episode_id,
                "scene_id": f"data/scene_datasets/mp3d/{scene}/{scene}.glb",
                "instruction": {"instruction_text": instruction},
            }
        )
        video = f"{scene}_r2r_{episode_id:06d}"
        rgb = images_root / video / "rgb"
        rgb.mkdir(parents=True)
        for frame in range(1, len(actions) + 1):
            Image.new("RGB", size, (episode_id * 50, frame * 10, 255 - frame * 10)).save(
                rgb / f"{frame:03d}.jpg", format="JPEG", quality=95
            )
        annotations.append(
            {
                "id": episode_id,
                "video": f"images/{video}",
                "instructions": [instruction],
                "actions": actions,
            }
        )
    train = tmp_path / "train.json.gz"
    with gzip.open(train, "wt", encoding="utf-8") as output:
        json.dump({"episodes": train_episodes}, output)
    (stream_root / "annotations_v1-3.json").write_text(json.dumps(annotations), encoding="utf-8")
    return train, images_root


def test_prepare_r2r_vlnce_manifest_small_real_rgb_fixture(tmp_path: Path) -> None:
    train, images_root = _fixture(tmp_path)
    output = tmp_path / "prepared"
    result = prepare.build_manifests(
        train,
        images_root,
        output,
        episode_count=2,
        steps_per_episode=4,
        history_depth=4,
    )
    low = json.loads((output / "low.json").read_text())["samples"]
    navida = json.loads((output / "navida.json").read_text())["samples"]
    panoramic = json.loads((output / "panoramic.json").read_text())["samples"]
    assert len(low) == len(navida) == len(panoramic) == 8
    assert all(len(sample["history_images"]) == 4 for sample in low)
    assert all(len(sample["history_responses"]) == 4 for sample in low)
    assert all(len(sample["history_images"]) == 4 for sample in navida)
    assert all(len(sample["history_panoramas"]) == 4 for sample in panoramic)
    assert all(len(sample["candidates"]) == 4 for sample in panoramic)
    assert low[0]["instruction"] == "Go to target 0."
    assert low[0]["history_responses"] == ["Move", "Left", "Right", "Move"]
    assert low[0]["distance_traveled"] == 0.5
    assert low[0]["current_image"].endswith("/005.jpg")
    assert not low[0]["current_image"].startswith("derived/")
    for sample in navida:
        for relative in [sample["current_image"], *sample["history_images"]]:
            with Image.open(output / relative) as image:
                assert image.size == (320, 240)
    for sample in panoramic:
        with Image.open(output / sample["panorama_image"]) as image:
            assert image.size == (960, 240)
        for candidate in sample["candidates"]:
            with Image.open(output / candidate["image"]) as image:
                assert image.size == (320, 240)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["classification"] == (
        "structurally_aligned_existing_export_not_official_tar_byte_verified"
    )
    assert provenance["official_tar_byte_verified"] is False
    assert provenance["synthetic_pixels"] is False
    assert provenance["trajectory_success_claim"] is False
    assert provenance["selection"]["episode_count"] == 2
    assert len(provenance["selection"]["selection_sha256"]) == 64
    assert provenance["contracts"]["history_per_sample"] == 4
    assert provenance["contracts"]["panoramic_candidates_per_sample"] == 4
    assert provenance["contracts"]["action_alignment"] == ("frame[i] corresponds to actions[i+1]")
    assert provenance["selection"]["samples"][0]["frame_index"] == 4
    assert len(provenance["selection"]["samples"][0]["history_responses"]) == 4
    assert provenance["derivatives"]
    for record in provenance["derivatives"]:
        assert len(record["source_sha256"]) == 64
        assert len(record["output_sha256"]) == 64
    for profile, record in provenance["manifests"].items():
        payload = Path(record["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record["sha256"], profile
    assert result["selection"]["selection_sha256"] == provenance["selection"]["selection_sha256"]
    unverified = benchmark_dataset.load_manifest(output / "low.json", images_root, LOW_PROFILE, 8)
    assert unverified.provenance["status"] == "unverified"
    verified = benchmark_dataset.load_manifest(
        output / "low.json",
        images_root,
        LOW_PROFILE,
        8,
        output / "provenance.json",
        require_formal=False,
    )
    assert verified.provenance["status"] == "verified"
    assert verified.provenance["formal"] is False
    with pytest.raises(ValueError, match="48 episodes x 4 steps"):
        benchmark_dataset.load_manifest(
            output / "low.json",
            images_root,
            LOW_PROFILE,
            8,
            output / "provenance.json",
            require_formal=True,
        )


def test_full_policy_reference_gate_cannot_pass_compiled_self_comparison() -> None:
    reference = {
        "text": "Move",
        "token_ids": [1],
        "actions": [1],
        "action_error": None,
    }
    compiled = {
        "text": "Left",
        "token_ids": [2],
        "actions": [2],
        "action_error": None,
    }
    assert full_policy_benchmark._paired_signature_exact(reference, reference, reference)
    assert not full_policy_benchmark._paired_signature_exact(reference, compiled, compiled)


def _command(
    script: str,
    manifest: Path,
    root: Path,
    attention: str,
    compile_backend: str,
    profile: str = LOW_PROFILE,
    provenance: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = str(REPO_ROOT)
    command = [
        sys.executable,
        str(BENCHMARK_DIR / script),
        "--manifest",
        str(manifest),
        "--data-root",
        str(root),
        "--checkpoint",
        "must-not-load",
        "--profile",
        profile,
        "--attention-backend",
        attention,
        "--compile-backend",
        compile_backend,
        "--limit",
        "1",
        "--seed",
        "41",
        "--warmup",
        "1",
        "--iters",
        "1",
        "--output",
        str(root / "unused.json"),
    ]
    if provenance is not None:
        command.extend(["--provenance", str(provenance)])
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("script", "cuda_error"),
    (
        ("benchmark.py", "requires an available CUDA device"),
        ("benchmark_cuda_graph.py", "requires CUDA"),
    ),
)
def test_cli_admits_inductor_before_model_load(tmp_path: Path, script: str, cuda_error: str) -> None:
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 24), (1, 2, 3)).save(image, format="JPEG")
    manifest = tmp_path / "low.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "one",
                        "instruction": "Go forward.",
                        "current_image": image.name,
                    }
                ]
            }
        )
    )
    result = _command(script, manifest, tmp_path, "torch_sdpa", "inductor")
    assert result.returncode != 0
    assert cuda_error in result.stderr
    assert "compile never falls back" not in result.stderr


@pytest.mark.parametrize("attention", ("triton", "auto"))
@pytest.mark.parametrize("script", ("benchmark.py", "benchmark_cuda_graph.py"))
def test_cli_rejects_legacy_or_auto_inductor_before_manifest_load(
    tmp_path: Path, script: str, attention: str
) -> None:
    result = _command(script, tmp_path / "missing.json", tmp_path, attention, "inductor")
    assert result.returncode != 0
    assert "--attention-backend=torch_sdpa; compile never falls back" in result.stderr


def test_cli_rejects_nonformal_provenance_before_model_load(tmp_path: Path) -> None:
    train, images_root = _fixture(tmp_path)
    output = tmp_path / "prepared"
    prepare.build_manifests(
        train,
        images_root,
        output,
        episode_count=2,
        steps_per_episode=4,
        history_depth=4,
    )
    result = _command(
        "benchmark.py",
        output / "low.json",
        images_root,
        "torch_sdpa",
        "inductor",
        provenance=output / "provenance.json",
    )
    assert result.returncode != 0
    assert "48 episodes x 4 steps = 192 samples" in result.stderr
    assert "must-not-load" not in result.stdout
