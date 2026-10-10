"""Shared test configuration.

Auto-skip tests marked ``gpu`` or ``pi05`` when no CUDA device is present, so the
CPU/CI run collects the whole suite and skips (rather than errors on) GPU-only
tests. GPU tests still run when a CUDA device is available; ``pi05`` additionally
needs ``VVLA_PI05_CKPT`` (guarded in the test itself).
"""

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_no_cuda = pytest.mark.skip(reason="requires CUDA (run on a GPU host)")
    for item in items:
        if item.get_closest_marker("gpu") is not None or item.get_closest_marker("pi05") is not None:
            item.add_marker(skip_no_cuda)
