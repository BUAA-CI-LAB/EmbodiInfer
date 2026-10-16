"""Deployment checks reject the known EmbodiRun schema/tag split without changing it."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compatibility", ROOT / "scripts/check_serving_compatibility.py"
)
compatibility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compatibility)


def test_current_wire_snapshot_and_matching_client(tmp_path):
    serve = ROOT / "embodiinfer/engine/serve"
    server = compatibility.contracts(
        [serve / name for name in ("contracts.py", "service.py", "wireless_server.py")]
    )
    assert server["schema:step"] == ["vvla.policy.step.v1"]
    assert server["schema:session"] == ["vvla.policy.session.v1"]
    assert server["schema:rpc"] == ["vvla.policy.rpc.v1"]
    assert server["REQUEST_TAG"] == [0x56564C41]
    assert server["RESPONSE_TAG"] == [0x56564C42]
    client = {
        key: value for key, value in server.items() if key not in {"schema:step.result", "schema:reset"}
    }
    assert compatibility.compare(server, client)["compatible_literals"]
    old = {
        key: [value.replace("vvla.", "embodiinfer.") if isinstance(value, str) else value for value in values]
        for key, values in client.items()
    }
    old.update(REQUEST_TAG=[0x454D4249], RESPONSE_TAG=[0x454D424A])
    result = compatibility.compare(server, old)
    assert not result["compatible_literals"]
    assert {"schema:step", "schema:session", "schema:rpc", "REQUEST_TAG", "RESPONSE_TAG"} <= result[
        "differences"
    ].keys()


def test_unknown_or_partial_clients_fail_closed():
    assert not compatibility.compare({}, {})["compatible_literals"]
    assert not compatibility.compare(
        {"schema:step": ["vvla.policy.step.v1"]}, {"schema:step": ["vvla.policy.step.v1"]}
    )["compatible_literals"]
