from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def test_policy_catalog_does_not_import_optional_adapters_or_compile() -> None:
    script = """
import sys
import torch

def forbidden_compile(*args, **kwargs):
    raise AssertionError("torch.compile ran during module import")

torch.compile = forbidden_compile
import embodiinfer.policies as policies

assert "pi05" in policies.available_policies()
assert "streamvln" in policies.available_policies()
assert "dm05" in policies.available_policies()
assert "embodiinfer.policies.pi05.modeling_pi05" not in sys.modules
assert "embodiinfer.policies.streamvln.policy" not in sys.modules
assert "embodiinfer.policies.dm05.modeling_dm05" not in sys.modules

stream_policy = policies.StreamVLNPolicy
assert stream_policy.__name__ == "StreamVLNPolicy"
assert "embodiinfer.policies.streamvln.policy" in sys.modules
assert "embodiinfer.policies.pi05.modeling_pi05" not in sys.modules

try:
    policies.make_policy("dm05")
except ValueError as error:
    assert "both checkpoint and norm_stats" in str(error)
else:
    raise AssertionError("DM0.5 must validate its checkpoint before loading OpenDM")
assert policies.DM05Policy.__name__ == "DM05Policy"
assert policies.DM05Batch.__name__ == "DM05Batch"
assert "opendm" not in sys.modules
assert "embodiinfer.policies.pi05.modeling_pi05" not in sys.modules

import embodiinfer.policies.pi05.modeling_pi05 as pi05
assert pi05._COMPILED_INFERENCE_HELPERS == {}
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
