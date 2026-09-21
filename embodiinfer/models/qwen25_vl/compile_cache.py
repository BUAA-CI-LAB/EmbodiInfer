"""Execution-scoped persistent cache ownership for Qwen2.5-VL Inductor."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from threading import Lock, RLock

import torch

QWEN25_VL_PERSISTENT_CACHE_SCHEMA = "qwen25_vl_inductor_execution_cache_v3"
QWEN25_VL_CACHE_BOOTSTRAP_ENV = "EMBODIINFER_COMPILE_CACHE_BOOTSTRAP_ASSERTION"
QWEN25_VL_CACHE_BOOTSTRAP_TOKEN = "qwen25_vl_compile_cache_preimport_v1"

_PROCESS_ROOT_LOCK = Lock()
_PROCESS_COMPILE_LOCK = RLock()
_PROCESS_CACHE_ROOT: Path | None = None


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=repr)
    return str(value)


def _canonical_json(value: object) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validated_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} must be exactly 64 lowercase hexadecimal digits")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qwen25_vl_model_config_sha256(config: object) -> str:
    if hasattr(config, "to_dict"):
        value = config.to_dict()
    elif hasattr(config, "__dict__"):
        value = vars(config)
    else:
        value = repr(config)
    return _sha256(_canonical_json(value))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cache_info_value(value: object) -> object:
    if value is None:
        return None
    if is_dataclass(value):
        return _json_value(asdict(value))
    if hasattr(value, "_asdict"):
        return _json_value(value._asdict())
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return repr(value)


def _cache_info_artifacts(value: object | None) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    artifacts = value.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return {}
    return {
        str(name): {
            "count": len(keys) if isinstance(keys, (tuple, list)) else 0,
            "keys": _json_value(keys),
        }
        for name, keys in artifacts.items()
    }


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _pre_import_launcher_hint(root: Path, libdevice: str | None) -> str:
    libdevice_value = libdevice or "/absolute/path/to/libdevice.10.bc"
    return (
        "Set this contract in a pure-stdlib launcher before importing torch:\n"
        f"export {QWEN25_VL_CACHE_BOOTSTRAP_ENV}="
        f"{QWEN25_VL_CACHE_BOOTSTRAP_TOKEN}\n"
        f"export TORCHINDUCTOR_CACHE_DIR={_shell_quote(str(root))}\n"
        "export TORCHINDUCTOR_FX_GRAPH_CACHE=1\n"
        "export TORCHINDUCTOR_AUTOGRAD_CACHE=1\n"
        f"export TRITON_LIBDEVICE_PATH={_shell_quote(libdevice_value)}\n"
        "python ..."
    )


def _assert_launcher_contract(value: str | Path) -> tuple[Path, dict[str, object]]:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise ValueError(
            "compile_cache_dir must be a trusted absolute path.\n"
            + _pre_import_launcher_hint(requested.absolute(), None)
        )
    if requested.is_symlink():
        raise ValueError(
            "compile_cache_dir must not be a symbolic link.\n" + _pre_import_launcher_hint(requested, None)
        )
    try:
        root = requested.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            "compile_cache_dir must already exist before this process starts.\n"
            + _pre_import_launcher_hint(requested, None)
        ) from exc
    if root != requested or not root.is_dir():
        raise ValueError(
            "compile_cache_dir must be an existing canonical directory without "
            "symbolic-link path components.\n" + _pre_import_launcher_hint(root, None)
        )
    root_stat = root.stat()
    if hasattr(os, "geteuid") and root_stat.st_uid != os.geteuid():
        raise PermissionError("compile_cache_dir must be owned by the current user")
    if stat.S_IMODE(root_stat.st_mode) != 0o700:
        raise PermissionError("compile_cache_dir must have mode 0700")

    libdevice_env = os.environ.get("TRITON_LIBDEVICE_PATH")
    expected = {
        QWEN25_VL_CACHE_BOOTSTRAP_ENV: QWEN25_VL_CACHE_BOOTSTRAP_TOKEN,
        "TORCHINDUCTOR_CACHE_DIR": str(root),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
    }
    problems = [
        f"{name}={os.environ.get(name)!r}, expected {wanted!r}"
        for name, wanted in expected.items()
        if os.environ.get(name) != wanted
    ]
    libdevice_path: Path | None = None
    if libdevice_env is None:
        problems.append("TRITON_LIBDEVICE_PATH is missing")
    else:
        candidate = Path(libdevice_env)
        if not candidate.is_absolute():
            problems.append("TRITON_LIBDEVICE_PATH must be absolute")
        elif not candidate.is_file():
            problems.append(f"TRITON_LIBDEVICE_PATH does not name an existing file: {candidate}")
        else:
            libdevice_path = candidate.resolve(strict=True)
    if os.environ.get("TORCHINDUCTOR_FORCE_DISABLE_CACHES") == "1":
        problems.append("TORCHINDUCTOR_FORCE_DISABLE_CACHES must not be 1")
    if problems:
        raise RuntimeError(
            "persistent Inductor cache launcher contract is not asserted:\n- "
            + "\n- ".join(problems)
            + "\n"
            + _pre_import_launcher_hint(root, libdevice_env)
        )
    assert libdevice_path is not None

    global _PROCESS_CACHE_ROOT
    with _PROCESS_ROOT_LOCK:
        if _PROCESS_CACHE_ROOT is not None and root != _PROCESS_CACHE_ROOT:
            raise RuntimeError(
                "PyTorch Inductor cache is process-global and is already asserted "
                f"for {_PROCESS_CACHE_ROOT}; cannot switch to {root}"
            )
        _PROCESS_CACHE_ROOT = root
    libdevice_stat = libdevice_path.stat()
    return root, {
        "path": str(libdevice_path),
        "bytes": int(libdevice_stat.st_size),
        "sha256": _sha256_file(libdevice_path),
    }


@dataclass
class Qwen25VLPersistentExecutionEntry:
    fingerprint: str
    identity: object
    artifact_loaded: bool = False
    artifact_published: bool = False
    artifact_publish_skipped: bool = False
    artifact_sha256: str | None = None
    artifact_bytes: int = 0
    load_cache_info: object | None = None
    save_cache_info: object | None = None
    counters_before: dict[str, dict[str, int]] | None = None
    counters_after: dict[str, dict[str, int]] | None = None
    cold_compile_reason: str | None = None
    quarantine_reason: str | None = None


class Qwen25VLPersistentCompileCache:
    """Own per-execution Mega-Cache artifacts under one process compile lock."""

    def __init__(self, root: str | Path, identity: Mapping[str, object]) -> None:
        self.root, self._libdevice_identity = _assert_launcher_contract(root)
        self._identity = _json_value(dict(identity))
        self._entries_dir = self.root / "embodiinfer-execution-entries"
        self._blobs_dir = self.root / "embodiinfer-content-blobs"
        self._quarantine_dir = self.root / "embodiinfer-quarantine"
        for directory in (
            self._entries_dir,
            self._blobs_dir,
            self._quarantine_dir,
        ):
            directory.mkdir(exist_ok=True, mode=0o700)
            if directory.is_symlink():
                raise RuntimeError(f"persistent cache directory is a symlink: {directory}")
            os.chmod(directory, 0o700)
        self._lock = RLock()
        self._entries: dict[str, Qwen25VLPersistentExecutionEntry] = {}
        self._last_fingerprint: str | None = None
        self._active_entry: Qwen25VLPersistentExecutionEntry | None = None

    @staticmethod
    def unconfigured_stats() -> dict[str, object]:
        return {
            "configured": False,
            "schema": QWEN25_VL_PERSISTENT_CACHE_SCHEMA,
            "root": None,
            "launcher_contract_asserted": False,
            "libdevice": None,
            "fingerprint": None,
            "manifest_key": None,
            "artifact_loaded": False,
            "artifact_published": False,
            "artifact_publish_skipped": False,
            "artifact_sha256": None,
            "artifact_bytes": 0,
            "load_cache_info": None,
            "save_cache_info": None,
            "cache_info_artifacts": {"loaded": {}, "saved": {}},
            "cache_counters_before": {
                group: {"hit": 0, "miss": 0}
                for group in (
                    "dynamo_fx",
                    "aot_autograd",
                    "async_compile",
                    "triton_bundle",
                )
            },
            "cache_counters_after": {
                group: {"hit": 0, "miss": 0}
                for group in (
                    "dynamo_fx",
                    "aot_autograd",
                    "async_compile",
                    "triton_bundle",
                )
            },
            "cache_counters_delta": {
                group: {"hit": 0, "miss": 0}
                for group in (
                    "dynamo_fx",
                    "aot_autograd",
                    "async_compile",
                    "triton_bundle",
                )
            },
            "persistent_hit_admission": {
                "admitted": False,
                "requires_artifact_loaded": True,
                "loaded_artifact_counts": {
                    "inductor": 0,
                    "aot_autograd": 0,
                },
                "dynamo_fx": {
                    "requires_loaded_artifact": "inductor",
                    "required_hit_min": 1,
                    "required_miss": 0,
                    "hit": 0,
                    "miss": 0,
                },
                "aot_autograd": {
                    "artifact_present": False,
                    "required_hit_min": 0,
                    "required_miss": 0,
                    "hit": 0,
                    "miss": 0,
                },
                "async_compile_misses": 0,
                "triton_bundle": {"hit": 0, "miss": 0},
            },
            "cold_compile_reason": None,
            "quarantine_reason": None,
            "corruption_policy": "quarantine_then_cold_compile",
            "entries": 0,
            "entry_stats": {},
            "counter_sources": {
                "dynamo_fx": {
                    "hit": "inductor.fxgraph_cache_hit",
                    "miss": "inductor.fxgraph_cache_miss",
                },
                "aot_autograd": {
                    "hit": "aot_autograd.autograd_cache_hit",
                    "miss": "aot_autograd.autograd_cache_miss",
                },
                "async_compile": {
                    "hit": "inductor.async_compile_cache_hit",
                    "miss": "inductor.async_compile_cache_miss",
                },
                "triton_bundle": {
                    "hit": "inductor.triton_bundler_read_and_emit_kernel",
                    "miss": "inductor.triton_bundler_save_kernel",
                },
            },
        }

    @staticmethod
    def _counter_snapshot() -> dict[str, dict[str, int]]:
        from torch._dynamo.utils import counters

        inductor = counters["inductor"]
        aot = counters["aot_autograd"]
        return {
            "dynamo_fx": {
                "hit": int(inductor["fxgraph_cache_hit"]),
                "miss": int(inductor["fxgraph_cache_miss"]),
            },
            "aot_autograd": {
                "hit": int(aot["autograd_cache_hit"]),
                "miss": int(aot["autograd_cache_miss"]),
            },
            "async_compile": {
                "hit": int(inductor["async_compile_cache_hit"]),
                "miss": int(inductor["async_compile_cache_miss"]),
            },
            "triton_bundle": {
                "hit": int(inductor["triton_bundler_read_and_emit_kernel"]),
                "miss": int(inductor["triton_bundler_save_kernel"]),
            },
        }

    @staticmethod
    def _counter_delta(
        before: Mapping[str, Mapping[str, int]],
        after: Mapping[str, Mapping[str, int]],
    ) -> dict[str, dict[str, int]]:
        return {
            group: {
                kind: int(after.get(group, {}).get(kind, 0)) - int(before.get(group, {}).get(kind, 0))
                for kind in ("hit", "miss")
            }
            for group in (
                "dynamo_fx",
                "aot_autograd",
                "async_compile",
                "triton_bundle",
            )
        }

    def _runtime_identity(
        self,
        execution_key: tuple[object, ...],
        example_inputs: tuple[torch.Tensor, ...],
        resolved_attention_backend: str | None,
    ) -> object:
        cuda_devices = sorted({str(tensor.device) for tensor in example_inputs if tensor.is_cuda})
        if len(cuda_devices) > 1:
            raise RuntimeError("one compiled callable cannot span multiple CUDA devices")
        if cuda_devices:
            device = torch.device(cuda_devices[0])
            capability = torch.cuda.get_device_capability(device)
            sm = f"sm{capability[0]}{capability[1]}"
            gpu_name = torch.cuda.get_device_name(device)
        else:
            sm = None
            gpu_name = None
        tensor_contract = [
            {
                "shape": tuple(tensor.shape),
                "stride": tuple(tensor.stride()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "requires_grad": bool(tensor.requires_grad),
                "layout": str(tensor.layout),
            }
            for tensor in example_inputs
        ]
        return _json_value(
            {
                "schema": QWEN25_VL_PERSISTENT_CACHE_SCHEMA,
                "compile_identity": self._identity,
                "execution_key": execution_key,
                "execution_tensors": tensor_contract,
                "resolved_attention_backend": resolved_attention_backend,
                "software": {
                    "torch": str(torch.__version__),
                    "cuda": torch.version.cuda,
                    "triton": _package_version("triton"),
                    "python": platform.python_version(),
                },
                "libdevice": self._libdevice_identity,
                "device": {
                    "cuda_devices": cuda_devices,
                    "sm": sm,
                    "name": gpu_name,
                },
            }
        )

    def execution_fingerprint(
        self,
        execution_key: tuple[object, ...],
        example_inputs: tuple[torch.Tensor, ...],
        *,
        resolved_attention_backend: str | None,
    ) -> str:
        identity = self._runtime_identity(execution_key, example_inputs, resolved_attention_backend)
        return _sha256(_canonical_json(identity))

    def _entry(
        self,
        execution_key: tuple[object, ...],
        example_inputs: tuple[torch.Tensor, ...],
        resolved_attention_backend: str | None,
    ) -> Qwen25VLPersistentExecutionEntry:
        identity = self._runtime_identity(execution_key, example_inputs, resolved_attention_backend)
        fingerprint = _sha256(_canonical_json(identity))
        entry = self._entries.get(fingerprint)
        if entry is None:
            entry = Qwen25VLPersistentExecutionEntry(fingerprint, identity)
            self._entries[fingerprint] = entry
        elif entry.identity != identity:
            raise RuntimeError("persistent execution fingerprint collision")
        self._last_fingerprint = fingerprint
        return entry

    def _manifest_path(self, fingerprint: str) -> Path:
        digest = _validated_sha256(fingerprint, label="execution fingerprint")
        return self._bounded_child(self._entries_dir, f"{digest}.json", label="execution manifest")

    def _blob_path(self, sha256: str) -> Path:
        digest = _validated_sha256(sha256, label="artifact blob SHA256")
        return self._bounded_child(self._blobs_dir, f"{digest}.bin", label="artifact blob")

    @staticmethod
    def _require_resolved_parent(path: Path, directory: Path, *, label: str) -> None:
        resolved_directory = directory.resolve(strict=True)
        if path.parent.resolve(strict=True) != resolved_directory:
            raise RuntimeError(f"{label} escapes its configured cache directory")

    @classmethod
    def _bounded_child(cls, directory: Path, name: str, *, label: str) -> Path:
        path = directory / name
        cls._require_resolved_parent(path, directory, label=label)
        if path.is_symlink():
            raise RuntimeError(f"{label} must not be a symbolic link: {path}")
        return path

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        import fcntl

        lock_path = self._bounded_child(
            self.root, ".embodiinfer-compile-cache.lock", label="compile cache lock"
        )
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(lock_path, flags, 0o600)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("compile cache lock must be a regular file")
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _atomic_temp(directory: Path, value: bytes) -> Path:
        fd, name = tempfile.mkstemp(prefix=".tmp-", dir=directory)
        path = Path(name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            with suppress(OSError):
                os.close(fd)
            path.unlink(missing_ok=True)
            raise
        return path

    def _quarantine_locked(
        self,
        entry: Qwen25VLPersistentExecutionEntry,
        manifest_path: Path | None,
        blob_path: Path | None,
        reason: str,
    ) -> None:
        suffix = f"{time.time_ns()}-{os.getpid()}"
        moved = False
        if manifest_path is not None:
            self._require_resolved_parent(manifest_path, self._entries_dir, label="execution manifest")
        if manifest_path is not None and (manifest_path.exists() or manifest_path.is_symlink()):
            destination = self._bounded_child(
                self._quarantine_dir,
                f"{entry.fingerprint}-{suffix}.manifest.json",
                label="quarantined manifest",
            )
            os.replace(manifest_path, destination)
            moved = True
        if blob_path is not None:
            self._require_resolved_parent(blob_path, self._blobs_dir, label="artifact blob")
        if blob_path is not None and (blob_path.exists() or blob_path.is_symlink()):
            destination = self._bounded_child(
                self._quarantine_dir,
                f"{blob_path.stem}-{suffix}.blob.bin",
                label="quarantined blob",
            )
            os.replace(blob_path, destination)
            moved = True
        if moved:
            self._fsync_directory(self._quarantine_dir)
            self._fsync_directory(self._entries_dir)
            self._fsync_directory(self._blobs_dir)
        entry.quarantine_reason = reason
        entry.cold_compile_reason = "corrupt_artifact_quarantined"

    def _load_exact(self, entry: Qwen25VLPersistentExecutionEntry) -> None:
        manifest_path = self._manifest_path(entry.fingerprint)
        entry.artifact_loaded = False
        entry.artifact_publish_skipped = False
        entry.load_cache_info = None
        entry.cold_compile_reason = None
        entry.quarantine_reason = None
        payload: bytes | None = None
        blob_path: Path | None = None
        expected_sha: str | None = None
        try:
            with self._file_lock():
                if not manifest_path.is_file():
                    entry.cold_compile_reason = "execution_manifest_missing"
                    return
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("schema") != QWEN25_VL_PERSISTENT_CACHE_SCHEMA:
                    raise RuntimeError("persistent cache schema mismatch")
                if manifest.get("fingerprint") != entry.fingerprint:
                    raise RuntimeError("persistent cache fingerprint mismatch")
                if manifest.get("identity") != entry.identity:
                    raise RuntimeError("persistent cache execution identity mismatch")
                expected_sha = _validated_sha256(manifest.get("blob_sha256"), label="manifest blob SHA256")
                blob_path = self._blob_path(expected_sha)
                if not blob_path.is_file():
                    raise RuntimeError("persistent cache content blob is missing")
                payload = blob_path.read_bytes()
                if _sha256(payload) != expected_sha:
                    raise RuntimeError("persistent cache content blob SHA256 mismatch")
                if len(payload) != int(manifest.get("blob_bytes", -1)):
                    raise RuntimeError("persistent cache content blob byte mismatch")
        except Exception as exc:
            with self._file_lock():
                self._quarantine_locked(entry, manifest_path, blob_path, f"{type(exc).__name__}: {exc}")
            return
        assert payload is not None and expected_sha is not None
        loader = getattr(torch.compiler, "load_cache_artifacts", None)
        if loader is None:
            raise RuntimeError("installed PyTorch does not provide load_cache_artifacts")
        try:
            loaded_info = loader(payload)
            if loaded_info is None:
                raise RuntimeError("load_cache_artifacts returned None")
        except Exception as exc:
            with self._file_lock():
                self._quarantine_locked(entry, manifest_path, blob_path, f"{type(exc).__name__}: {exc}")
            return
        entry.artifact_loaded = True
        entry.artifact_sha256 = expected_sha
        entry.artifact_bytes = len(payload)
        entry.load_cache_info = _cache_info_value(loaded_info)

    @staticmethod
    @contextmanager
    def _fresh_artifact_scope() -> Iterator[None]:
        try:
            from torch.compiler._cache import CacheArtifactManager
        except Exception as exc:
            raise RuntimeError(
                "persistent per-execution artifacts require torch.compiler._cache.CacheArtifactManager"
            ) from exc
        scope = getattr(CacheArtifactManager, "with_fresh_cache", None)
        if scope is None:
            raise RuntimeError("installed PyTorch lacks CacheArtifactManager.with_fresh_cache")
        with scope():
            yield

    @contextmanager
    def compilation_lifecycle(
        self,
        execution_key: tuple[object, ...],
        example_inputs: tuple[torch.Tensor, ...],
        *,
        resolved_attention_backend: str | None,
        expected_fingerprint: str | None = None,
    ) -> Iterator[Qwen25VLPersistentExecutionEntry]:
        with _PROCESS_COMPILE_LOCK, self._lock:
            entry = self._entry(execution_key, example_inputs, resolved_attention_backend)
            if expected_fingerprint is not None and entry.fingerprint != expected_fingerprint:
                raise RuntimeError("graph-key persistent fingerprint does not match compile execution")
            if self._active_entry is not None:
                raise RuntimeError("nested persistent compilation lifecycle")
            self._active_entry = entry
            entry.counters_before = self._counter_snapshot()
            try:
                self._load_exact(entry)
                with self._fresh_artifact_scope():
                    yield entry
            finally:
                self._active_entry = None

    @staticmethod
    def _admission(
        entry: Qwen25VLPersistentExecutionEntry,
    ) -> tuple[bool, dict[str, dict[str, int]], dict[str, int]]:
        before = entry.counters_before or {}
        after = entry.counters_after or before
        delta = Qwen25VLPersistentCompileCache._counter_delta(before, after)
        loaded_artifacts = _cache_info_artifacts(entry.load_cache_info)

        def artifact_count(name: str) -> int:
            value = loaded_artifacts.get(name)
            if not isinstance(value, Mapping):
                return 0
            count = value.get("count", 0)
            return int(count) if isinstance(count, int) else 0

        artifact_counts = {
            "inductor": artifact_count("inductor"),
            "aot_autograd": artifact_count("aot_autograd"),
        }
        fx = delta["dynamo_fx"]
        aot = delta["aot_autograd"]
        required_aot_hit_min = 1 if artifact_counts["aot_autograd"] else 0
        admitted = bool(
            entry.artifact_loaded
            and artifact_counts["inductor"] >= 1
            and fx["hit"] >= 1
            and fx["miss"] == 0
            and aot["hit"] >= required_aot_hit_min
            and aot["miss"] == 0
        )
        return admitted, delta, artifact_counts

    def publish(self, entry: Qwen25VLPersistentExecutionEntry) -> None:
        if self._active_entry is not entry:
            raise RuntimeError("publish must run inside its persistent compile lifecycle")
        entry.counters_after = self._counter_snapshot()
        admitted, _, _ = self._admission(entry)
        if admitted:
            entry.artifact_publish_skipped = True
            return
        saver = getattr(torch.compiler, "save_cache_artifacts", None)
        if saver is None:
            raise RuntimeError("installed PyTorch does not provide save_cache_artifacts")
        result = saver()
        if result is None:
            state = "cold compile" if not entry.artifact_loaded else "unadmitted load"
            raise RuntimeError(f"{state} produced no per-execution cache artifacts to publish")
        payload_value, cache_info = result
        payload = bytes(payload_value)
        blob_sha = _sha256(payload)
        blob_path = self._blob_path(blob_sha)
        manifest_path = self._manifest_path(entry.fingerprint)
        manifest = {
            "schema": QWEN25_VL_PERSISTENT_CACHE_SCHEMA,
            "fingerprint": entry.fingerprint,
            "identity": entry.identity,
            "blob_sha256": blob_sha,
            "blob_bytes": len(payload),
            "cache_info": _cache_info_value(cache_info),
        }
        with self._file_lock():
            if blob_path.exists() and _sha256_file(blob_path) != blob_sha:
                self._quarantine_locked(
                    entry,
                    None,
                    blob_path,
                    "content-addressed blob filename collision",
                )
            if not blob_path.exists():
                blob_tmp = self._atomic_temp(self._blobs_dir, payload)
                try:
                    os.replace(blob_tmp, blob_path)
                    self._fsync_directory(self._blobs_dir)
                finally:
                    blob_tmp.unlink(missing_ok=True)
            manifest_tmp = self._atomic_temp(self._entries_dir, _canonical_json(manifest) + b"\n")
            try:
                os.replace(manifest_tmp, manifest_path)
                self._fsync_directory(self._entries_dir)
            finally:
                manifest_tmp.unlink(missing_ok=True)
        entry.artifact_published = True
        entry.artifact_sha256 = blob_sha
        entry.artifact_bytes = len(payload)
        entry.save_cache_info = _cache_info_value(cache_info)

    @staticmethod
    def _entry_stats(entry: Qwen25VLPersistentExecutionEntry) -> dict[str, object]:
        before = {group: dict(values) for group, values in (entry.counters_before or {}).items()}
        after = {
            group: dict(values)
            for group, values in (entry.counters_after or entry.counters_before or {}).items()
        }
        admitted, delta, artifact_counts = Qwen25VLPersistentCompileCache._admission(entry)
        required_aot_hit_min = 1 if artifact_counts["aot_autograd"] else 0
        return {
            "fingerprint": entry.fingerprint,
            "manifest_key": entry.fingerprint,
            "artifact_loaded": entry.artifact_loaded,
            "artifact_published": entry.artifact_published,
            "artifact_publish_skipped": entry.artifact_publish_skipped,
            "artifact_sha256": entry.artifact_sha256,
            "artifact_bytes": entry.artifact_bytes,
            "load_cache_info": entry.load_cache_info,
            "save_cache_info": entry.save_cache_info,
            "cache_info_artifacts": {
                "loaded": _cache_info_artifacts(entry.load_cache_info),
                "saved": _cache_info_artifacts(entry.save_cache_info),
            },
            "cache_counters_before": before,
            "cache_counters_after": after,
            "cache_counters_delta": delta,
            "persistent_hit_admission": {
                "admitted": admitted,
                "requires_artifact_loaded": True,
                "loaded_artifact_counts": artifact_counts,
                "dynamo_fx": {
                    "requires_loaded_artifact": "inductor",
                    "required_hit_min": 1,
                    "required_miss": 0,
                    **delta["dynamo_fx"],
                },
                "aot_autograd": {
                    "artifact_present": artifact_counts["aot_autograd"] > 0,
                    "required_hit_min": required_aot_hit_min,
                    "required_miss": 0,
                    **delta["aot_autograd"],
                },
                "async_compile_misses": delta["async_compile"]["miss"],
                "triton_bundle": delta["triton_bundle"],
            },
            "cold_compile_reason": entry.cold_compile_reason,
            "quarantine_reason": entry.quarantine_reason,
        }

    def stats(self) -> dict[str, object]:
        with self._lock:
            result = self.unconfigured_stats()
            result.update(
                {
                    "configured": True,
                    "root": str(self.root),
                    "launcher_contract_asserted": True,
                    "libdevice": dict(self._libdevice_identity),
                    "entries": len(self._entries),
                    "entry_stats": {
                        fingerprint: self._entry_stats(entry) for fingerprint, entry in self._entries.items()
                    },
                    "counter_sources": {
                        "dynamo_fx": {
                            "hit": "inductor.fxgraph_cache_hit",
                            "miss": "inductor.fxgraph_cache_miss",
                        },
                        "aot_autograd": {
                            "hit": "aot_autograd.autograd_cache_hit",
                            "miss": "aot_autograd.autograd_cache_miss",
                        },
                        "async_compile": {
                            "hit": "inductor.async_compile_cache_hit",
                            "miss": "inductor.async_compile_cache_miss",
                        },
                        "triton_bundle": {
                            "hit": "inductor.triton_bundler_read_and_emit_kernel",
                            "miss": "inductor.triton_bundler_save_kernel",
                        },
                    },
                }
            )
            if self._last_fingerprint is not None:
                latest = result["entry_stats"][self._last_fingerprint]
                for name in (
                    "fingerprint",
                    "manifest_key",
                    "artifact_loaded",
                    "artifact_published",
                    "artifact_publish_skipped",
                    "artifact_sha256",
                    "artifact_bytes",
                    "load_cache_info",
                    "save_cache_info",
                    "cache_info_artifacts",
                    "cache_counters_before",
                    "cache_counters_after",
                    "cache_counters_delta",
                    "persistent_hit_admission",
                    "cold_compile_reason",
                    "quarantine_reason",
                ):
                    result[name] = latest[name]
            return result


__all__ = [
    "QWEN25_VL_CACHE_BOOTSTRAP_ENV",
    "QWEN25_VL_CACHE_BOOTSTRAP_TOKEN",
    "QWEN25_VL_PERSISTENT_CACHE_SCHEMA",
    "Qwen25VLPersistentCompileCache",
    "Qwen25VLPersistentExecutionEntry",
    "qwen25_vl_model_config_sha256",
]
