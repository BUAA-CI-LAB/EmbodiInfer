"""Both spellings of the serving wire schema are accepted.

The serving API was renamed from ``vvla`` to ``embodiinfer``, and the two
first-party deployments ended up on opposite sides of the rename: this engine's
own examples and tests send ``vvla.policy.*``, while the EmbodiRun deployment
runtime sends ``embodiinfer.policy.*``. Before this alias was added, a real
EmbodiRun process could not open a session against this server at all — the
mismatch is invisible to either repository's tests, because each one only ever
talks to itself. It was found by running the two processes against each other.

These tests pin both directions of the compatibility decision:

* both spellings are *accepted* on input, so neither client has to change;
* responses still carry the **legacy** spelling, because a client asserting on it
  exists in-tree. Flipping emission is a deliberate migration, not a side effect
  of accepting an alias.
"""

from __future__ import annotations

import pytest

from embodiinfer.engine.serve.contracts import (
    SESSION_SCHEMA,
    SESSION_SCHEMA_ALIASES,
    STEP_SCHEMA,
    STEP_SCHEMA_ALIASES,
    ModelAction,
    ModelResult,
    RawImage,
    ServeError,
    parse_structured_step,
)
from embodiinfer.engine.serve.service import PolicyService

NEW_SESSION_SCHEMA = "embodiinfer.policy.session.v1"
NEW_STEP_SCHEMA = "embodiinfer.policy.step.v1"


class _StubAdapter:
    """The smallest adapter the service will drive."""

    action_space = "stub.action_chunk.v1"

    def capabilities(self) -> dict:
        return {"model": "stub"}

    def infer(self, request) -> ModelResult:
        return ModelResult(
            action_space=self.action_space,
            actions=(ModelAction(kind="action_chunk", values={"data": [[0.0]]}),),
            timing={"policy_ms": 1.0},
        )

    def reset(self, session_id: str) -> None:
        return None


def _structured_step(schema: str) -> dict:
    return {
        "schema": schema,
        "session_id": "session-a",
        "request_id": "req-1",
        "step_id": 0,
        "instruction": "pick up the bowl",
        "state": {"observation.state": [0.0]},
        "images": [{"name": "image", "mime_type": "image/png", "data": b"png"}],
    }


def _open(service: PolicyService, schema: str) -> dict:
    return service.open_session({"schema": schema, "robot_id": "robot", "action_space": service.action_space})


def test_schema_aliases_cover_both_spellings():
    assert SESSION_SCHEMA in SESSION_SCHEMA_ALIASES
    assert NEW_SESSION_SCHEMA in SESSION_SCHEMA_ALIASES
    assert STEP_SCHEMA in STEP_SCHEMA_ALIASES
    assert NEW_STEP_SCHEMA in STEP_SCHEMA_ALIASES


@pytest.mark.parametrize("schema", [SESSION_SCHEMA, NEW_SESSION_SCHEMA])
def test_open_session_accepts_either_spelling(schema):
    service = PolicyService(_StubAdapter())
    opened = _open(service, schema)
    assert opened["session_id"].startswith("session-")


def test_open_session_rejects_an_unknown_spelling():
    service = PolicyService(_StubAdapter())
    with pytest.raises(ServeError) as excinfo:
        _open(service, "vvla.policy.session.v2")
    assert excinfo.value.code == "unsupported_schema"


@pytest.mark.parametrize("schema", [STEP_SCHEMA, NEW_STEP_SCHEMA])
def test_structured_step_accepts_either_spelling(schema):
    request = parse_structured_step(_structured_step(schema))
    assert request.session_id == "session-a"
    assert request.step_id == 0


def test_structured_step_rejects_an_unknown_spelling():
    with pytest.raises(ServeError) as excinfo:
        parse_structured_step(_structured_step("vvla.policy.step.v2"))
    assert excinfo.value.code == "unsupported_schema"


def test_responses_keep_emitting_the_legacy_spelling():
    """Pins the no-emission-change decision so a future rename is deliberate."""
    service = PolicyService(_StubAdapter())
    opened = _open(service, NEW_SESSION_SCHEMA)
    assert opened["schema"] == SESSION_SCHEMA
    response = service.step(
        parse_structured_step(
            {
                **_structured_step(NEW_STEP_SCHEMA),
                "session_id": opened["session_id"],
            }
        )
    )
    assert response["schema"].startswith("vvla.")


def test_a_new_spelling_client_round_trips_a_step():
    """The whole point: an ``embodiinfer.policy.*`` client can drive a step."""
    service = PolicyService(_StubAdapter())
    opened = _open(service, NEW_SESSION_SCHEMA)
    request = parse_structured_step(
        {
            **_structured_step(NEW_STEP_SCHEMA),
            "session_id": opened["session_id"],
            "images": [{"name": "image", "mime_type": "image/png", "data": b"png"}],
        }
    )
    response = service.step(request)
    assert response["request_id"] == "req-1"
    assert response["actions"][0]["values"]["data"] == [[0.0]]


def test_raw_policy_request_carries_images_unchanged():
    """Guards the multipart-free path the wireless transport uses."""
    request = parse_structured_step(_structured_step(NEW_STEP_SCHEMA))
    assert isinstance(request.images, tuple)
    assert isinstance(request.images[0], RawImage)
    assert request.images[0].data == b"png"
