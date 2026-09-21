import copy

import pytest
from scripts.activevln.compare_reference import required_output_key_failures
from scripts.activevln.schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    load_lock,
    validate_manifest,
)


def _manifest():
    lock = load_lock()
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "producer": "hf_incremental",
        "source": copy.deepcopy(lock["source"]),
        "checkpoint": copy.deepcopy(lock["checkpoint"]),
        "software": {"python": "3.12", "torch": "2.6.0"},
        "runner_profile": "official_eval_r2r",
        "sampling": {
            "requested": {"temperature": 0.2},
            "generation_config": {"repetition_penalty": 1.05},
            "effective": {"temperature": 0.2, "repetition_penalty": 1.05},
            "seed": 0,
        },
        "cases": [{"case_id": "synthetic", "turns": [{"step_idx": 0}]}],
        "tensor_index": {},
        "files": {"tensors.safetensors": "0" * 64},
        "created_at": "2026-07-15T00:00:00+00:00",
    }


def test_activevln_reference_manifest_accepts_pinned_schema():
    validate_manifest(_manifest())


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("producer", "native", "unknown producer"),
        ("runner_profile", "latest", "unknown runner profile"),
        ("schema_version", 99, "unsupported schema_version"),
    ],
)
def test_activevln_reference_manifest_rejects_unpinned_values(field, value, match):
    manifest = _manifest()
    manifest[field] = value
    with pytest.raises(ValueError, match=match):
        validate_manifest(manifest)


def test_official_reference_requires_effective_sampling():
    manifest = _manifest()
    manifest["producer"] = "official_vllm"
    manifest["sampling"]["effective"] = {}
    with pytest.raises(ValueError, match="effective sampling"):
        validate_manifest(manifest)


def test_reference_rejects_checkpoint_revision_drift():
    manifest = _manifest()
    manifest["checkpoint"]["revision"] = "f" * 40
    with pytest.raises(ValueError, match="checkpoint revision"):
        validate_manifest(manifest)


def test_checkpoint_lock_covers_weights_and_tokenizer_assets():
    files = load_lock()["checkpoint"]["files"]
    assert "model-00001-of-00004.safetensors" in files
    assert "model-00004-of-00004.safetensors" in files
    assert "model.safetensors.index.json" in files
    assert "tokenizer.json" in files
    assert "chat_template.json" in files


def test_reference_comparator_rejects_missing_model_outputs():
    prefix = "cases/0/turns/0"
    required = {
        f"{prefix}/response_token_ids": object(),
        f"{prefix}/response_action_mask": object(),
        f"{prefix}/response_token_logprobs": object(),
        f"{prefix}/topk_token_ids": object(),
        f"{prefix}/topk_logprobs": object(),
        f"{prefix}/parsed_action_tensor": object(),
        f"{prefix}/parsed_action_mask": object(),
    }
    missing = dict(required)
    missing.pop(f"{prefix}/response_token_logprobs")
    failures = required_output_key_failures(required, missing)
    assert any("response_token_logprobs" in failure for failure in failures)
