"""Model-neutral request capability and capture ownership contracts."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from embodiinfer.backend.torch.request_graph import CapturedRequest
from embodiinfer.engine.config import EngineConfig
from embodiinfer.engine.core import EngineCore
from embodiinfer.policies.decoder import DecodeResult, FlowActionDecoder
from embodiinfer.policies.factory import make_policy
from embodiinfer.types import Observation, collate


class _RequestDecoder(FlowActionDecoder):
    calls = 0

    def supports_request_execution(self, graphs):
        return True

    def decode_request(self, state, batch, num_steps, bucket, graphs):
        self.calls += 1
        prefix = self.policy.encode_prefix(batch)
        return DecodeResult(self.produce_chunk(state, prefix, num_steps, bucket, graphs))


def test_generic_request_route_preserves_noise_steps_ids_and_actions():
    torch.manual_seed(8)
    policy = make_policy("mock_flow_vla", preset="tiny")
    core = EngineCore(policy, EngineConfig(device="cpu", use_cuda_graph=False))
    cfg = policy.config
    batch = collate(
        [
            Observation(
                torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
                torch.rand(cfg.state_dim),
                torch.randint(cfg.vocab_size, (cfg.max_lang_len,)),
            )
        ],
        ["request"],
    )
    baseline = core.execute(batch, num_steps=3, generator=torch.Generator().manual_seed(12))
    policy._decoder = _RequestDecoder(policy)
    actual = core.execute(batch, num_steps=3, generator=torch.Generator().manual_seed(12))
    assert torch.equal(actual[0].actions, baseline[0].actions)
    assert actual[0].request_id == "request"
    assert actual[0].meta["num_steps"] == 3
    assert actual[0].policy_version == baseline[0].policy_version
    assert set(actual[0].timing) == {"request_ms", "e2e_ms"}
    assert policy.decoder.calls == 1
    # Explicit prefill/decode remains available for pipeline and RL callers.
    staged = core._prefill(batch, 3, torch.Generator().manual_seed(12))
    assert torch.equal(core._integrate(staged.x, staged.prefix, 3, 1)[0], actual[0].actions)
    assert policy.decoder.calls == 1


def test_request_errors_are_not_retried_or_silently_fallen_back(monkeypatch):
    policy = make_policy("mock_flow_vla", preset="tiny")
    core = EngineCore(policy, EngineConfig(device="cpu", use_cuda_graph=False))
    policy._decoder = _RequestDecoder(policy)
    batch = SimpleNamespace(batch_size=1, request_ids=["x"], to=lambda *_: None)
    monkeypatch.setattr(policy, "pad", lambda value, bucket: value)
    monkeypatch.setattr(
        policy.decoder, "decode_request", lambda *args: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(ValueError, match="bad"):
        core.execute(batch)


def test_rtc_guidance_keeps_the_existing_two_stage_route(monkeypatch):
    policy = make_policy("mock_flow_vla", preset="tiny")
    core = EngineCore(policy, EngineConfig(device="cpu", use_cuda_graph=False))
    policy._decoder = _RequestDecoder(policy)
    guidance = object()
    staged = SimpleNamespace(x=None, prefix=None, num_steps=3, bucket=1, rtc_guidance=guidance)
    received = []
    monkeypatch.setattr(core, "_prefill", lambda batch, steps, generator, rtc: received.append(rtc) or staged)
    monkeypatch.setattr(core, "_integrate", lambda state, prefix, steps, bucket, rtc: received.append(rtc))
    monkeypatch.setattr(core, "_pack", lambda *args: ["existing RTC result"])
    assert core.execute(SimpleNamespace(batch_size=1), rtc_guidance=guidance) == ["existing RTC result"]
    assert received == [guidance, guidance]
    assert policy.decoder.calls == 0


class _FakeStream:
    def __init__(self, identifier=1):
        self.cuda_stream = identifier

    def wait_stream(self, other):
        pass


def _fake_cuda(monkeypatch):
    current = [_FakeStream()]
    resets = []
    synchronizations = []

    class Graph:
        def replay(self):
            pass

        def reset(self):
            resets.append(self)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: current[0])
    monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: current.__setitem__(0, stream))
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_: _FakeStream(2))
    monkeypatch.setattr(torch.cuda, "stream", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "graph", lambda *_, **__: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronizations.append(device))
    return current, resets, synchronizations


def test_capture_owns_output_enforces_stream_and_closes_once(monkeypatch):
    current, resets, syncs = _fake_cuda(monkeypatch)
    release = []
    source = torch.ones(2)
    captured = CapturedRequest(lambda: source, torch.device("cuda:0"), release=lambda: release.append(1))
    result = captured.run(lambda: None)
    source.zero_()
    assert torch.equal(result, torch.ones(2))
    current[0] = _FakeStream(3)
    with pytest.raises(RuntimeError, match="owning stream"):
        captured.run(lambda: None)
    captured.close()
    captured.close()
    assert len(resets) == len(syncs) == len(release) == 1
    with pytest.raises(RuntimeError, match="closed"):
        captured.run(lambda: None)


def test_capture_failure_resets_and_releases_workspace(monkeypatch):
    _, resets, syncs = _fake_cuda(monkeypatch)
    release = []
    calls = []

    def fail_capture():
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError("capture failed")
        return torch.ones(1)

    with pytest.raises(RuntimeError, match="capture failed"):
        CapturedRequest(fail_capture, torch.device("cuda:0"), release=lambda: release.append(1))
    assert len(resets) == len(syncs) == len(release) == 1


@pytest.mark.gpu
def test_real_capture_changed_inputs_own_storage_and_stream_guard():
    source = torch.ones(4, device="cuda")
    graph = CapturedRequest(lambda: source.square() + 2, source.device)
    try:
        first = graph.run(lambda: source.fill_(2))
        second = graph.run(lambda: source.fill_(3))
        torch.cuda.synchronize()
        assert torch.equal(first, torch.full_like(first, 6))
        assert torch.equal(second, torch.full_like(second, 11))
        assert first.data_ptr() != second.data_ptr()
        with torch.cuda.stream(torch.cuda.Stream()), pytest.raises(RuntimeError, match="owning stream"):
            graph.run(lambda: None)
    finally:
        graph.close()


def test_capture_end_failure_restores_callers_stream(monkeypatch):
    from contextlib import contextmanager

    current, resets, syncs = _fake_cuda(monkeypatch)
    original = current[0]

    @contextmanager
    def broken_capture(*args, **kwargs):
        current[0] = _FakeStream(99)
        yield
        raise RuntimeError("capture_end failed")

    monkeypatch.setattr(torch.cuda, "graph", broken_capture)
    with pytest.raises(RuntimeError, match="capture_end failed"):
        CapturedRequest(lambda: torch.ones(1), torch.device("cuda:0"))
    assert current[0] is original
    assert len(resets) == len(syncs) == 1


def test_capture_rejects_changed_precision_context(monkeypatch):
    _fake_cuda(monkeypatch)
    graph = CapturedRequest(lambda: torch.ones(1), torch.device("cuda:0"))
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium" if previous != "medium" else "highest")
        with pytest.raises(RuntimeError, match="recapture required"):
            graph.run(lambda: None)
    finally:
        torch.set_float32_matmul_precision(previous)
        graph.close()


def test_capture_rejects_autocast_cached_weight_lifetime(monkeypatch):
    _fake_cuda(monkeypatch)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *args: True)
    monkeypatch.setattr(torch, "is_autocast_cache_enabled", lambda: True)
    with pytest.raises(ValueError, match="cache_enabled=False"):
        CapturedRequest(lambda: torch.ones(1), torch.device("cuda:0"))


def test_capture_rejects_changed_attention_backend_context(monkeypatch):
    _fake_cuda(monkeypatch)
    graph = CapturedRequest(lambda: torch.ones(1), torch.device("cuda:0"))
    original = torch.backends.cuda.flash_sdp_enabled()
    try:
        torch.backends.cuda.enable_flash_sdp(not original)
        with pytest.raises(RuntimeError, match="recapture required"):
            graph.run(lambda: None)
    finally:
        torch.backends.cuda.enable_flash_sdp(original)
        graph.close()


def test_close_drops_release_callback_and_retained_owner(monkeypatch):
    import weakref

    _fake_cuda(monkeypatch)

    class Owner:
        pass

    owner = Owner()
    reference = weakref.ref(owner)
    graph = CapturedRequest(lambda: torch.ones(1), torch.device("cuda:0"), release=lambda value=owner: None)
    del owner
    assert reference() is not None
    graph.close()
    assert reference() is None
    assert graph._release is None
