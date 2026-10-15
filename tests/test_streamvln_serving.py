from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from embodiinfer.engine.serve.contracts import RawImage, RawPolicyRequest
from embodiinfer.engine.serve.service import PolicyService
from embodiinfer.policies.streamvln.serving import (
    STREAMVLN_ACTION_SPACE,
    StreamVLNServingAdapter,
)
from embodiinfer.types import ActionChunk, DecodeTrace, SessionKey


class _Policy:
    is_recurrent = True

    def __init__(self) -> None:
        self.observations = []

    def collate(self, observations, request_ids):
        self.observations.extend(observations)
        return SimpleNamespace(observations=observations, request_ids=request_ids)


class _Core:
    def __init__(self) -> None:
        self.policy = _Policy()
        self.sessions = []
        self.resets = []

    def execute(self, batch, *, session_ids):
        self.sessions.append((batch, tuple(session_ids)))
        return [
            ActionChunk(
                request_id=batch.request_ids[0],
                actions=torch.tensor(
                    [
                        [1.0, 0.25],
                        [2.0, 15.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ]
                ),
                latency_ms=4.5,
                trace=DecodeTrace(
                    token_ids=torch.tensor([1, 2, 3]),
                    action_mask=torch.tensor([True, True, True, False]),
                ),
                policy_version=7,
            )
        ]

    def reset_sessions(self, sessions):
        self.resets.append(tuple(sessions))


def _png() -> bytes:
    image = Image.new("RGB", (3, 2), color=(255, 64, 0))
    payload = BytesIO()
    image.save(payload, format="PNG")
    return payload.getvalue()


def _request(*, session_id: str = "session-1", images=None) -> RawPolicyRequest:
    return RawPolicyRequest(
        session_id=session_id,
        request_id="request-1",
        step_id=3,
        instruction="walk to the door",
        state={},
        images=images if images is not None else (RawImage("observation.images.rgb", "image/png", _png()),),
        metadata={"episode": "demo"},
    )


def test_streamvln_factory_accepts_http_load_device(monkeypatch):
    from embodiinfer.policies.streamvln import policy

    load_options = []
    backbone = object()
    loaded = SimpleNamespace(
        backbone=backbone, tokenizer=object(), image_processor=object(), eos_token_ids=(1,)
    )

    def load_checkpoint(*args, **kwargs):
        load_options.append(kwargs)
        return loaded

    monkeypatch.setattr(policy, "load_streamvln_checkpoint", load_checkpoint)
    monkeypatch.setattr(policy, "StreamVLNProcessor", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy, "StreamVLNPolicy", lambda *args, **kwargs: args[0])
    assert policy.build_streamvln("/checkpoint", load_device="cuda:2") is backbone
    assert load_options == [{"dtype": "bfloat16", "max_context": 32768, "load_device": "cuda:2"}]


def test_streamvln_serving_maps_image_session_and_valid_actions():
    core = _Core()
    adapter = StreamVLNServingAdapter(
        core=core,
        config={"image_fields": ["observation.images.rgb"], "return_steps": 4},
    )

    result = adapter.infer(_request())

    observation = core.policy.observations[0]
    assert observation.images.shape == (1, 3, 2, 3)
    assert observation.instruction == "walk to the door"
    assert observation.metadata == {"step_id": 3, "episode": "demo"}
    assert core.sessions[0][1] == (SessionKey("session-1", "http"),)
    assert result.action_space == STREAMVLN_ACTION_SPACE
    assert result.actions[0].values == {
        "data": [[1.0, 0.25], [2.0, 15.0], [0.0, 0.0]],
    }
    assert result.policy_revision == "7"


def test_streamvln_serving_reset_uses_the_inference_session_key():
    core = _Core()
    adapter = StreamVLNServingAdapter(core=core)

    adapter.reset("session-2")

    assert core.resets == [(SessionKey("session-2", "http"),)]


def test_streamvln_shared_service_keeps_retries_and_resets_transactional():
    core = _Core()
    service = PolicyService(StreamVLNServingAdapter(core=core, config={"return_steps": 4}))
    opened = service.open_session(
        {
            "schema": "embodiinfer.policy.session.v1",
            "robot_id": "recorded-navigation",
            "action_space": STREAMVLN_ACTION_SPACE,
        }
    )
    session_id = opened["session_id"]
    request = replace(_request(session_id=session_id), step_id=0)

    first = service.step(request)
    assert service.step(request) == first
    assert len(core.sessions) == 1
    assert first["action_space"] == STREAMVLN_ACTION_SPACE
    assert first["actions"] == [
        {"type": "action_chunk", "values": {"data": [[1.0, 0.25], [2.0, 15.0], [0.0, 0.0]]}}
    ]

    reset = service.reset(session_id, {"request_id": "reset-1"})
    assert reset["session_revision"] == 2
    assert core.resets == [(SessionKey(session_id, "http"),)]
    restarted = service.step(replace(request, request_id="request-2"))
    assert restarted["step_id"] == 0
    assert restarted["session_revision"] == 3
    assert len(core.sessions) == 2
    assert core.sessions[-1][1] == (SessionKey(session_id, "http"),)


def test_streamvln_serving_rejects_missing_or_duplicate_rgb_fields():
    adapter = StreamVLNServingAdapter(core=_Core())

    with pytest.raises(ValueError, match="missing requested image field"):
        adapter.infer(_request(images=()))
    duplicate = RawImage("observation.images.rgb.png", "_it_does_not_matter", b"")
    with pytest.raises(ValueError, match="duplicate image field"):
        adapter.infer(
            _request(
                images=(
                    RawImage("observation.images.rgb", "image/png", _png()),
                    duplicate,
                )
            )
        )


def test_streamvln_serving_rejects_action_chunks_without_a_mask():
    core = _Core()

    def execute(batch, *, session_ids):
        del session_ids
        return [ActionChunk(batch.request_ids[0], torch.tensor([[0.0, 0.0]]))]

    core.execute = execute
    adapter = StreamVLNServingAdapter(core=core)

    with pytest.raises(RuntimeError, match="missing its action mask"):
        adapter.infer(_request())
