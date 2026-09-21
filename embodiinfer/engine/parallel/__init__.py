"""Data- and tensor-parallel execution primitives."""

from importlib import import_module
from typing import TYPE_CHECKING

from .pi05 import parallelize_pi05_towers
from .qwen25_vl import FusedQKVColumnParallelLinear, parallelize_qwen25_vl
from .tensor_parallel import ColumnParallelLinear, RowParallelLinear, TensorParallelContext

if TYPE_CHECKING:
    # The runtime __getattr__ below is invisible to static analysis, so declare the
    # lazy re-exports for type checkers and the API reference.
    from .data_parallel import (
        DataParallelEngine,
        Dispatcher,
        InProcessReplica,
        LeastLoadedDispatcher,
        ProcessExecutor,
        Replica,
        ReplicaExecutor,
        RoundRobinDispatcher,
        ThreadedExecutor,
    )

_DATA_PARALLEL_EXPORTS = {
    "DataParallelEngine",
    "Dispatcher",
    "InProcessReplica",
    "LeastLoadedDispatcher",
    "ProcessExecutor",
    "Replica",
    "ReplicaExecutor",
    "RoundRobinDispatcher",
    "ThreadedExecutor",
}


def __getattr__(name: str):
    if name not in _DATA_PARALLEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".data_parallel", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "ColumnParallelLinear",
    "DataParallelEngine",
    "Dispatcher",
    "FusedQKVColumnParallelLinear",
    "InProcessReplica",
    "LeastLoadedDispatcher",
    "ProcessExecutor",
    "Replica",
    "ReplicaExecutor",
    "RowParallelLinear",
    "RoundRobinDispatcher",
    "TensorParallelContext",
    "ThreadedExecutor",
    "parallelize_pi05_towers",
    "parallelize_qwen25_vl",
]
