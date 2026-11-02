"""Hardware guidance separates queried facts from measurements and hypotheses."""

from __future__ import annotations

import sys

import pytest
from scripts.kernel_tuning import hardware
from scripts.kernel_tuning.contracts import ContractError


@pytest.mark.parametrize("capability", [[8, 7], [11, 0], [8, 9]])
def test_notes_do_not_invent_device_bandwidth_or_winning_tiles(capability):
    notes = hardware.render_notes({"identity": {"capability": capability, "multi_processor_count": 8}})
    assert "no performance calibration" in notes
    assert '"multi_processor_count": 8' in notes
    assert "hypotheses, not measured winners" in notes
    assert "Unknown unless supplied" in notes


def test_driver_fallback_does_not_claim_torch_readiness(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(
        hardware, "_driver_properties", lambda device: {"capability": [11, 0], "torch": "not probed"}
    )
    monkeypatch.setattr(hardware.shutil, "which", lambda _: None)
    report = hardware.probe()
    assert report["identity"]["torch"] == "not probed"
    assert "no model-runtime readiness claim" in report["source"]


def test_invalid_driver_device_is_explicit():
    with pytest.raises(ContractError, match="cuda:INDEX"):
        hardware._driver_properties("cpu")
