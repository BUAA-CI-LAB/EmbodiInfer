"""Exercise the installed EmbodiInfer command-line entry points."""

import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "command",
    [
        "embodiinfer-serve",
        "embodiinfer-http-serve",
        "embodiinfer-wireless-serve",
    ],
)
def test_installed_cli_commands(command: str) -> None:
    entries = {
        entry.name: entry
        for entry in distribution("embodiinfer").entry_points
        if entry.group == "console_scripts"
    }
    assert command in entries
    result = subprocess.run(
        [str(Path(sys.executable).parent / command), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
