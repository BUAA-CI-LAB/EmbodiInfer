from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from embodiinfer.models.qwen25_vl import compile_cache as cache_module
from embodiinfer.models.qwen25_vl.compile_cache import (
    QWEN25_VL_CACHE_BOOTSTRAP_ENV,
    QWEN25_VL_CACHE_BOOTSTRAP_TOKEN,
    QWEN25_VL_PERSISTENT_CACHE_SCHEMA,
    Qwen25VLPersistentCompileCache,
)


@dataclass
class _CacheInfo:
    artifacts: dict[str, tuple[str, ...]]


def _zero_counters() -> dict[str, dict[str, int]]:
    return {
        group: {"hit": 0, "miss": 0}
        for group in (
            "dynamo_fx",
            "aot_autograd",
            "async_compile",
            "triton_bundle",
        )
    }


def _fx_hit_counters() -> dict[str, dict[str, int]]:
    result = _zero_counters()
    result["dynamo_fx"]["hit"] = 1
    return result


@contextmanager
def _fresh_scope():
    yield


def _configure_cache_env(monkeypatch, root: Path) -> Path:
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    libdevice = root.parent / f"{root.name}.libdevice.10.bc"
    libdevice.write_bytes(b"unit-test-libdevice")
    monkeypatch.setenv(QWEN25_VL_CACHE_BOOTSTRAP_ENV, QWEN25_VL_CACHE_BOOTSTRAP_TOKEN)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(root))
    monkeypatch.setenv("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    monkeypatch.setenv("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    monkeypatch.setenv("TRITON_LIBDEVICE_PATH", str(libdevice))
    monkeypatch.delenv("TORCHINDUCTOR_FORCE_DISABLE_CACHES", raising=False)
    cache_module._PROCESS_CACHE_ROOT = None
    monkeypatch.setattr(
        Qwen25VLPersistentCompileCache,
        "_fresh_artifact_scope",
        staticmethod(_fresh_scope),
    )
    return libdevice


def _set_counter_sequence(monkeypatch, *values) -> None:
    snapshots = iter(values)
    monkeypatch.setattr(
        Qwen25VLPersistentCompileCache,
        "_counter_snapshot",
        staticmethod(lambda: next(snapshots)),
    )


def _set_compiler_artifacts(monkeypatch, *, load, save) -> None:
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", load, raising=False)
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", save, raising=False)


def _publish(
    cache: Qwen25VLPersistentCompileCache,
    execution_key: tuple[object, ...],
    inputs: tuple[torch.Tensor, ...],
):
    with cache.compilation_lifecycle(
        execution_key,
        inputs,
        resolved_attention_backend="torch_sdpa",
    ) as entry:
        cache.publish(entry)
    return entry


def test_cache_requires_pre_import_bootstrap_assertion(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    monkeypatch.delenv(QWEN25_VL_CACHE_BOOTSTRAP_ENV)
    with pytest.raises(RuntimeError, match=QWEN25_VL_CACHE_BOOTSTRAP_ENV):
        Qwen25VLPersistentCompileCache(root, {"case": "bootstrap"})


def test_execution_fingerprint_is_pure_and_does_not_load(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    calls = 0

    def load(_payload):
        nonlocal calls
        calls += 1
        return _CacheInfo({})

    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", load, raising=False)
    cache = Qwen25VLPersistentCompileCache(root, {"case": "pure"})
    inputs = (torch.zeros((1, 7)),)
    first = cache.execution_fingerprint(("shape", 7), inputs, resolved_attention_backend="torch_sdpa")
    second = cache.execution_fingerprint(("shape", 7), inputs, resolved_attention_backend="torch_sdpa")
    assert first == second
    assert calls == 0
    assert cache.stats()["entries"] == 0


def test_per_execution_manifest_blob_hash_atomic_and_cache_info(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    payload = b"per-execution-mega-cache"
    info = _CacheInfo({"dynamo_fx": ("fx-key",), "aot_autograd": ("aot-key",)})
    _set_counter_sequence(monkeypatch, _zero_counters(), _zero_counters())
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: pytest.fail("cold execution must not load"),
        save=lambda: (payload, info),
    )
    cache = Qwen25VLPersistentCompileCache(root, {"case": "publish"})
    entry = _publish(cache, ("shape", 8), (torch.zeros((1, 8)),))

    manifests = list((root / "vvla-execution-entries").glob("*.json"))
    blobs = list((root / "vvla-content-blobs").glob("*.bin"))
    assert len(manifests) == 1
    assert len(blobs) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    digest = hashlib.sha256(payload).hexdigest()
    assert manifest["schema"] == "qwen25_vl_inductor_execution_cache_v3"
    assert manifest["fingerprint"] == entry.fingerprint
    assert manifest["blob_sha256"] == digest
    assert manifest["blob_bytes"] == len(payload)
    assert blobs[0].name == f"{digest}.bin"
    assert blobs[0].read_bytes() == payload
    assert not list(root.rglob(".tmp-*"))

    stats = cache.stats()
    assert stats["schema"] == QWEN25_VL_PERSISTENT_CACHE_SCHEMA
    assert stats["entries"] == 1
    assert stats["entry_stats"][entry.fingerprint]["artifact_published"] is True
    assert stats["artifact_publish_skipped"] is False
    assert stats["save_cache_info"]["artifacts"]["dynamo_fx"] == ["fx-key"]
    assert stats["cache_info_artifacts"]["saved"]["aot_autograd"]["count"] == 1


def test_admitted_hit_without_aot_artifact_skips_publish(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    payload = b"shared-payload"
    _set_counter_sequence(monkeypatch, _zero_counters(), _zero_counters())
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: _CacheInfo({}),
        save=lambda: (
            payload,
            _CacheInfo({"inductor": ("producer",), "autotune": ("producer-tune",)}),
        ),
    )
    producer = Qwen25VLPersistentCompileCache(root, {"case": "hit"})
    _publish(producer, ("shape", 8), (torch.zeros((1, 8)),))
    manifest_path = next((root / "vvla-execution-entries").glob("*.json"))
    blob_path = next((root / "vvla-content-blobs").glob("*.bin"))
    original_manifest = manifest_path.read_bytes()
    original_blob = blob_path.read_bytes()

    _set_counter_sequence(monkeypatch, _zero_counters(), _fx_hit_counters())
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda actual: _CacheInfo(
            {
                "inductor": (hashlib.sha256(actual).hexdigest(),),
                "autotune": ("loaded-tune",),
            }
        ),
        save=lambda: pytest.fail("an admitted hit must not be republished"),
    )
    consumer = Qwen25VLPersistentCompileCache(root, {"case": "hit"})
    entry = _publish(consumer, ("shape", 8), (torch.zeros((1, 8)),))
    stats = consumer.stats()["entry_stats"][entry.fingerprint]
    assert stats["artifact_loaded"] is True
    assert stats["artifact_published"] is False
    assert stats["artifact_publish_skipped"] is True
    assert stats["persistent_hit_admission"]["admitted"] is True
    assert stats["persistent_hit_admission"]["dynamo_fx"]["hit"] == 1
    assert stats["persistent_hit_admission"]["loaded_artifact_counts"] == {
        "inductor": 1,
        "aot_autograd": 0,
    }
    assert stats["persistent_hit_admission"]["aot_autograd"] == {
        "artifact_present": False,
        "required_hit_min": 0,
        "required_miss": 0,
        "hit": 0,
        "miss": 0,
    }
    assert manifest_path.read_bytes() == original_manifest
    assert blob_path.read_bytes() == original_blob


def test_loaded_aot_artifact_without_aot_hit_is_not_admitted(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    payload = b"aot-payload"
    artifacts = {
        "inductor": ("producer-fx",),
        "aot_autograd": ("producer-aot",),
    }
    _set_counter_sequence(monkeypatch, _zero_counters(), _zero_counters())
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: _CacheInfo({}),
        save=lambda: (payload, _CacheInfo(artifacts)),
    )
    producer = Qwen25VLPersistentCompileCache(root, {"case": "aot-required"})
    _publish(producer, ("shape", 8), (torch.zeros((1, 8)),))

    _set_counter_sequence(monkeypatch, _zero_counters(), _fx_hit_counters())
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: _CacheInfo(artifacts),
        save=lambda: None,
    )
    consumer = Qwen25VLPersistentCompileCache(root, {"case": "aot-required"})
    with pytest.raises(RuntimeError, match="unadmitted load.*no per-execution"):
        _publish(consumer, ("shape", 8), (torch.zeros((1, 8)),))
    admission = consumer.stats()["persistent_hit_admission"]
    assert admission["admitted"] is False
    assert admission["aot_autograd"]["artifact_present"] is True
    assert admission["aot_autograd"]["required_hit_min"] == 1
    assert admission["aot_autograd"]["hit"] == 0


@pytest.mark.parametrize(
    ("loaded_artifacts", "fx_hit", "fx_miss"),
    [
        pytest.param({"inductor": ("fx",)}, 0, 0, id="fx-hit-required"),
        pytest.param({"inductor": ("fx",)}, 1, 1, id="fx-miss-forbidden"),
        pytest.param({"autotune": ("tune",)}, 1, 0, id="inductor-required"),
    ],
)
def test_inductor_artifact_and_fx_counters_are_mandatory_for_admission(
    monkeypatch, tmp_path, loaded_artifacts, fx_hit, fx_miss
) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    payload = b"fx-requirements"
    _set_counter_sequence(monkeypatch, _zero_counters(), _zero_counters())
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: _CacheInfo({}),
        save=lambda: (payload, _CacheInfo(loaded_artifacts)),
    )
    producer = Qwen25VLPersistentCompileCache(root, {"case": "fx-required"})
    _publish(producer, ("shape", 8), (torch.zeros((1, 8)),))

    after = _zero_counters()
    after["dynamo_fx"] = {"hit": fx_hit, "miss": fx_miss}
    _set_counter_sequence(monkeypatch, _zero_counters(), after)
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: _CacheInfo(loaded_artifacts),
        save=lambda: None,
    )
    consumer = Qwen25VLPersistentCompileCache(root, {"case": "fx-required"})
    with pytest.raises(RuntimeError, match="unadmitted load.*no per-execution"):
        _publish(consumer, ("shape", 8), (torch.zeros((1, 8)),))
    admission = consumer.stats()["persistent_hit_admission"]
    assert admission["admitted"] is False
    assert admission["loaded_artifact_counts"]["inductor"] == (1 if "inductor" in loaded_artifacts else 0)
    assert admission["dynamo_fx"]["hit"] == fx_hit
    assert admission["dynamo_fx"]["miss"] == fx_miss


def test_cold_compile_save_none_fails_loud(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    _set_counter_sequence(monkeypatch, _zero_counters(), _zero_counters())
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: pytest.fail("cold execution must not load"),
        save=lambda: None,
    )
    cache = Qwen25VLPersistentCompileCache(root, {"case": "cold-none"})
    with pytest.raises(RuntimeError, match="cold compile.*no per-execution"):
        _publish(cache, ("shape", 9), (torch.zeros((1, 9)),))
    stats = cache.stats()
    assert stats["cold_compile_reason"] == "execution_manifest_missing"
    assert stats["artifact_loaded"] is False


def test_execution_shapes_have_isolated_manifests_and_shared_content_blob(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    payload = b"deduplicated-content"
    _set_counter_sequence(
        monkeypatch,
        _zero_counters(),
        _zero_counters(),
        _zero_counters(),
        _zero_counters(),
    )
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: pytest.fail("new shape must not load another manifest"),
        save=lambda: (payload, _CacheInfo({})),
    )
    cache = Qwen25VLPersistentCompileCache(root, {"case": "shapes"})
    first = _publish(cache, ("bucket", 8), (torch.zeros((1, 8)),))
    second = _publish(cache, ("bucket", 16), (torch.zeros((1, 16)),))
    assert first.fingerprint != second.fingerprint
    assert len(list((root / "vvla-execution-entries").glob("*.json"))) == 2
    assert len(list((root / "vvla-content-blobs").glob("*.bin"))) == 1
    stats = cache.stats()
    assert stats["entries"] == 2
    assert set(stats["entry_stats"]) == {first.fingerprint, second.fingerprint}


def test_corrupt_blob_is_quarantined_then_cold_published(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    _set_counter_sequence(monkeypatch, _zero_counters(), _zero_counters())
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: _CacheInfo({}),
        save=lambda: (b"valid", _CacheInfo({})),
    )
    producer = Qwen25VLPersistentCompileCache(root, {"case": "quarantine"})
    _publish(producer, ("shape", 4), (torch.zeros((1, 4)),))
    blob = next((root / "vvla-content-blobs").glob("*.bin"))
    blob.write_bytes(b"corrupt")

    _set_counter_sequence(monkeypatch, _zero_counters(), _zero_counters())

    def unexpected_load(_payload):
        pytest.fail("hash validation must precede PyTorch artifact loading")

    _set_compiler_artifacts(
        monkeypatch,
        load=unexpected_load,
        save=lambda: (b"replacement", _CacheInfo({})),
    )
    consumer = Qwen25VLPersistentCompileCache(root, {"case": "quarantine"})
    entry = _publish(consumer, ("shape", 4), (torch.zeros((1, 4)),))
    stats = consumer.stats()["entry_stats"][entry.fingerprint]
    assert stats["cold_compile_reason"] == "corrupt_artifact_quarantined"
    assert "SHA256 mismatch" in stats["quarantine_reason"]
    assert stats["artifact_published"] is True
    assert len(list((root / "vvla-quarantine").iterdir())) == 2


def test_blob_leaf_symlink_is_fail_closed(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    _set_counter_sequence(monkeypatch, _zero_counters(), _zero_counters())
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: _CacheInfo({}),
        save=lambda: (b"valid", _CacheInfo({})),
    )
    producer = Qwen25VLPersistentCompileCache(root, {"case": "blob-symlink"})
    _publish(producer, ("shape", 4), (torch.zeros((1, 4)),))
    blob = next((root / "vvla-content-blobs").glob("*.bin"))
    blob.unlink()
    victim = tmp_path / "outside-victim.bin"
    victim_bytes = b"must-not-be-modified"
    victim.write_bytes(victim_bytes)
    blob.symlink_to(victim)

    _set_counter_sequence(monkeypatch, _zero_counters(), _zero_counters())

    def unexpected_load(_payload):
        pytest.fail("blob symlinks must be rejected before PyTorch artifact loading")

    _set_compiler_artifacts(
        monkeypatch,
        load=unexpected_load,
        save=lambda: (b"replacement", _CacheInfo({})),
    )
    consumer = Qwen25VLPersistentCompileCache(root, {"case": "blob-symlink"})
    try:
        entry = _publish(consumer, ("shape", 4), (torch.zeros((1, 4)),))
    except (OSError, RuntimeError) as exc:
        rejection = str(exc).lower()
        assert "symlink" in rejection or "symbolic link" in rejection
    else:
        stats = consumer.stats()["entry_stats"][entry.fingerprint]
        assert stats["cold_compile_reason"] == "corrupt_artifact_quarantined"
        rejection = str(stats["quarantine_reason"]).lower()
        assert "symlink" in rejection or "symbolic link" in rejection
    assert victim.read_bytes() == victim_bytes


def test_compilation_lifecycle_is_process_serialized(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    monkeypatch.setattr(
        Qwen25VLPersistentCompileCache,
        "_counter_snapshot",
        staticmethod(_zero_counters),
    )
    first_cache = Qwen25VLPersistentCompileCache(root, {"case": "first"})
    second_cache = Qwen25VLPersistentCompileCache(root, {"case": "second"})
    first_entered = threading.Event()
    second_ready = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    failures: list[BaseException] = []

    def worker(cache, key, entered, ready=None, hold=False):
        try:
            if ready is not None:
                ready.set()
            with cache.compilation_lifecycle(
                key,
                (torch.zeros((1, 2)),),
                resolved_attention_backend="torch_sdpa",
            ):
                entered.set()
                if hold and not release_first.wait(2):
                    raise TimeoutError("test did not release first lifecycle")
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(
        target=worker,
        args=(first_cache, ("first",), first_entered),
        kwargs={"hold": True},
        daemon=True,
    )
    second = threading.Thread(
        target=worker,
        args=(second_cache, ("second",), second_entered),
        kwargs={"ready": second_ready},
        daemon=True,
    )
    first.start()
    assert first_entered.wait(1)
    second.start()
    assert second_ready.wait(1)
    assert not second_entered.wait(0.05)
    release_first.set()
    first.join(2)
    second.join(2)
    assert second_entered.is_set()
    assert not first.is_alive() and not second.is_alive()
    assert failures == []


def test_compile_cache_launcher_has_no_torch_import() -> None:
    launcher = Path(__file__).parents[1] / "benchmarks" / "3B-navigation" / "benchmark_compile_cache.py"
    tree = ast.parse(launcher.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imports


def _directory_snapshot(root: Path) -> list[tuple[str, bool, int]]:
    return [
        (
            str(path.relative_to(root)),
            path.is_dir(),
            0 if path.is_dir() else path.stat().st_size,
        )
        for path in sorted(root.rglob("*"))
    ]


def test_fingerprint_isolated_dimensions_and_no_io_side_effects(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    monkeypatch.setattr(
        Qwen25VLPersistentCompileCache,
        "_counter_snapshot",
        staticmethod(lambda: pytest.fail("fingerprint must not read counters")),
    )
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: pytest.fail("fingerprint must not load"),
        save=lambda: pytest.fail("fingerprint must not save"),
    )
    first = Qwen25VLPersistentCompileCache(root, {"identity": "first"})
    second = Qwen25VLPersistentCompileCache(root, {"identity": "second"})
    contiguous = torch.zeros((2, 3), dtype=torch.float32)
    strided = torch.zeros((3, 2), dtype=torch.float32).t()
    before = _directory_snapshot(root)
    values = {
        "base": first.execution_fingerprint(
            ("key", 1), (contiguous,), resolved_attention_backend="torch_sdpa"
        ),
        "key": first.execution_fingerprint(
            ("key", 2), (contiguous,), resolved_attention_backend="torch_sdpa"
        ),
        "shape": first.execution_fingerprint(
            ("key", 1),
            (torch.zeros((2, 4)),),
            resolved_attention_backend="torch_sdpa",
        ),
        "dtype": first.execution_fingerprint(
            ("key", 1),
            (contiguous.to(torch.float64),),
            resolved_attention_backend="torch_sdpa",
        ),
        "stride": first.execution_fingerprint(
            ("key", 1), (strided,), resolved_attention_backend="torch_sdpa"
        ),
        "backend": first.execution_fingerprint(("key", 1), (contiguous,), resolved_attention_backend="other"),
        "identity": second.execution_fingerprint(
            ("key", 1), (contiguous,), resolved_attention_backend="torch_sdpa"
        ),
    }
    assert len(set(values.values())) == len(values)
    assert _directory_snapshot(root) == before
    assert first.stats()["entries"] == 0
    assert second.stats()["entries"] == 0


def test_manifest_publish_is_atomic_when_replace_fails(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    _set_counter_sequence(monkeypatch, _zero_counters(), _zero_counters())
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: _CacheInfo({}),
        save=lambda: (b"version-one", _CacheInfo({})),
    )
    producer = Qwen25VLPersistentCompileCache(root, {"case": "atomic"})
    entry = _publish(producer, ("shape", 4), (torch.zeros((1, 4)),))
    manifest = root / "vvla-execution-entries" / f"{entry.fingerprint}.json"
    original_manifest = manifest.read_bytes()
    original_blob = next((root / "vvla-content-blobs").glob("*.bin"))
    original_blob_value = original_blob.read_bytes()

    _set_counter_sequence(monkeypatch, _zero_counters(), _zero_counters())
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: _CacheInfo({}),
        save=lambda: (b"version-two", _CacheInfo({})),
    )
    real_replace = cache_module.os.replace

    def fail_manifest_replace(source, destination):
        destination_path = Path(destination)
        if destination_path.parent.name == "vvla-execution-entries":
            raise OSError("injected manifest replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(cache_module.os, "replace", fail_manifest_replace)
    consumer = Qwen25VLPersistentCompileCache(root, {"case": "atomic"})
    with pytest.raises(OSError, match="injected manifest"):
        _publish(consumer, ("shape", 4), (torch.zeros((1, 4)),))
    assert manifest.read_bytes() == original_manifest
    assert original_blob.read_bytes() == original_blob_value
    assert not list(root.rglob(".tmp-*"))


def test_manifest_blob_sha_cannot_escape_blob_directory(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    _set_counter_sequence(monkeypatch, _zero_counters(), _zero_counters())
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: _CacheInfo({}),
        save=lambda: (b"safe", _CacheInfo({})),
    )
    producer = Qwen25VLPersistentCompileCache(root, {"case": "path"})
    entry = _publish(producer, ("shape", 4), (torch.zeros((1, 4)),))
    manifest = root / "vvla-execution-entries" / f"{entry.fingerprint}.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["blob_sha256"] = "../../outside"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"must-not-move")

    _set_counter_sequence(monkeypatch, _zero_counters(), _zero_counters())
    _set_compiler_artifacts(
        monkeypatch,
        load=lambda _payload: pytest.fail("invalid SHA must not reach loader"),
        save=lambda: (b"replacement", _CacheInfo({})),
    )
    consumer = Qwen25VLPersistentCompileCache(root, {"case": "path"})
    _publish(consumer, ("shape", 4), (torch.zeros((1, 4)),))
    assert outside.read_bytes() == b"must-not-move"
    stats = consumer.stats()
    assert stats["cold_compile_reason"] == "corrupt_artifact_quarantined"
    assert "blob SHA256" in str(stats["quarantine_reason"])


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="requires symlink support")
def test_cache_lock_rejects_symlink_without_touching_target(monkeypatch, tmp_path) -> None:
    root = tmp_path / "cache"
    _configure_cache_env(monkeypatch, root)
    victim = tmp_path / "victim"
    victim.write_bytes(b"victim")
    victim.chmod(0o644)
    (root / ".vvla-compile-cache.lock").symlink_to(victim)
    monkeypatch.setattr(
        Qwen25VLPersistentCompileCache,
        "_counter_snapshot",
        staticmethod(_zero_counters),
    )
    cache = Qwen25VLPersistentCompileCache(root, {"case": "lock-symlink"})
    with (
        pytest.raises((RuntimeError, OSError)),
        cache.compilation_lifecycle(
            ("shape", 2),
            (torch.zeros((1, 2)),),
            resolved_attention_backend="torch_sdpa",
        ),
    ):
        pass
    assert victim.read_bytes() == b"victim"
    assert victim.stat().st_mode & 0o777 == 0o644


_SUBPROCESS_HELPER = textwrap.dedent(
    """
    import json
    import os
    from pathlib import Path
    import sys

    import torch

    if not hasattr(torch.compiler, "load_cache_artifacts") or not hasattr(
        torch.compiler, "save_cache_artifacts"
    ):
        raise SystemExit(77)
    try:
        from torch.compiler._cache import CacheArtifactManager
    except Exception:
        raise SystemExit(77)
    if not hasattr(CacheArtifactManager, "with_fresh_cache"):
        raise SystemExit(77)

    from embodiinfer.models.qwen25_vl.compile import Qwen25VLCompileRuntime

    class Tiny(torch.nn.Module):
        def forward(self, value):
            return torch.sin(value) * 1.25 + value.square()

    root = Path(sys.argv[1])
    output = Path(sys.argv[2])
    runtime = Qwen25VLCompileRuntime(
        "inductor",
        compile_cache_dir=root,
        cache_identity={"case": "real-subprocess-v1"},
        target="cpu_minimal_forward",
    )
    value = torch.arange(8, dtype=torch.float32).view(2, 4)
    execution_key = ("tiny", (2, 4), "float32")
    execute = runtime.get_callable(
        Tiny(),
        execution_key,
        (value,),
        resolved_attention_backend="torch_sdpa",
        persistent_execution_key=execution_key,
    )
    actual = execute(value)
    expected = torch.sin(value) * 1.25 + value.square()
    torch.testing.assert_close(actual, expected)
    output.write_text(
        json.dumps(runtime.stats(), sort_keys=True), encoding="utf-8"
    )
    """
)


def _run_real_cache_child(
    cache_root: Path,
    output: Path,
    *,
    libdevice: Path,
    repo: Path,
) -> dict[str, object]:
    environment = dict(os.environ)
    environment[QWEN25_VL_CACHE_BOOTSTRAP_ENV] = QWEN25_VL_CACHE_BOOTSTRAP_TOKEN
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root)
    environment["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    environment["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    environment["TRITON_LIBDEVICE_PATH"] = str(libdevice)
    environment.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(repo) if not inherited else str(repo) + os.pathsep + inherited
    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_HELPER, str(cache_root), str(output)],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode == 77:
        pytest.skip("installed PyTorch lacks Mega-Cache artifact APIs")
    assert completed.returncode == 0, (
        f"child failed with {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.mark.skipif(sys.platform != "linux", reason="persistent cache uses fcntl")
def test_real_three_process_persistent_cache_admission(tmp_path) -> None:
    if not hasattr(torch.compiler, "load_cache_artifacts") or not hasattr(
        torch.compiler, "save_cache_artifacts"
    ):
        pytest.skip("installed PyTorch lacks Mega-Cache artifact APIs")
    repo = Path(__file__).resolve().parents[1]
    libdevice = tmp_path / "libdevice.10.bc"
    libdevice.write_bytes(b"cpu-test-libdevice")
    producer_root = tmp_path / "producer"
    fresh_root = tmp_path / "fresh"
    producer_root.mkdir(mode=0o700)
    fresh_root.mkdir(mode=0o700)

    producer = _run_real_cache_child(
        producer_root,
        tmp_path / "producer.json",
        libdevice=libdevice,
        repo=repo,
    )
    for name in ("vvla-execution-entries", "vvla-content-blobs"):
        shutil.copytree(producer_root / name, fresh_root / name)
    same_consumer = _run_real_cache_child(
        producer_root,
        tmp_path / "same.json",
        libdevice=libdevice,
        repo=repo,
    )
    fresh_consumer = _run_real_cache_child(
        fresh_root,
        tmp_path / "fresh.json",
        libdevice=libdevice,
        repo=repo,
    )

    producer_cache = producer["persistent_cache"]
    same_cache = same_consumer["persistent_cache"]
    fresh_cache = fresh_consumer["persistent_cache"]
    assert producer_cache["artifact_published"] is True
    assert producer_cache["artifact_loaded"] is False
    assert producer_cache["fingerprint"] == same_cache["fingerprint"] == fresh_cache["fingerprint"]
    for cache_stats in (same_cache, fresh_cache):
        assert cache_stats["artifact_loaded"] is True
        admission = cache_stats["persistent_hit_admission"]
        assert admission["admitted"] is True
        assert cache_stats["artifact_publish_skipped"] is True
        loaded = admission["loaded_artifact_counts"]
        assert loaded["inductor"] >= 1
        assert admission["dynamo_fx"]["hit"] >= 1
        assert admission["dynamo_fx"]["miss"] == 0
        assert admission["aot_autograd"]["required_hit_min"] == (1 if loaded["aot_autograd"] else 0)
        if loaded["aot_autograd"]:
            assert admission["aot_autograd"]["hit"] >= 1
        assert admission["aot_autograd"]["miss"] == 0
        assert cache_stats["quarantine_reason"] is None
