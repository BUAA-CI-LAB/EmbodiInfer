"""Exercise the installed EmbodiInfer commands and their compatibility aliases."""

import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("current", "legacy"),
    [
        ("embodiinfer-serve", "vvla-serve"),
        ("embodiinfer-http-serve", "vvla-http-serve"),
        ("embodiinfer-wireless-serve", "vvla-wireless-serve"),
    ],
)
def test_installed_cli_aliases(current: str, legacy: str) -> None:
    entries = {
        entry.name: entry
        for entry in distribution("embodiinfer").entry_points
        if entry.group == "console_scripts"
    }
    assert entries[current].load() is entries[legacy].load()
    for name in (current, legacy):
        result = subprocess.run(
            [str(Path(sys.executable).parent / name), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
