"""Real tiny-Qwen batching, graph lifecycle and episode-ownership regressions."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from embodiinfer.exceptions import SessionCancelledError
from embodiinfer.policies.streamvln.batching import StreamVLNBatchedRuntime, _GraphInputArena
from embodiinfer.policies.streamvln.policy import StreamVLNPolicy
from embodiinfer.policies.streamvln.processing import StreamVLNProcessor, make_streamvln_observation


def _policy(monkeypatch, tree=False):
    from test_streamvln_tree import _small_backbone

    class Tokenizer:
        def __call__(self, text, *, add_special_tokens):
            return {"input_ids": [len(text) % 29 + 1]}

        def decode(self, ids, **kwargs):
            return "↑" * min(len(ids), 4)

    backbone = _small_backbone()
    backbone.vision_tower = nn.Linear(1, 1).to(device="cuda", dtype=torch.bfloat16)
    batches = []

    def vision(pixels):
        batches.append(pixels.shape[0])
        return pixels.mean(dim=(1, 2, 3))[:, None, None].expand(-1, 4, backbone.hidden_size).contiguous()

    monkeypatch.setattr(backbone, "_encode_frames_eager", vision)
    processor = StreamVLNProcessor(
        Tokenizer(),
        SimpleNamespace(preprocess=lambda frame, **kwargs: {"pixel_values": frame[None]}),
        window_size=4,
        num_history=2,
    )
    policy = StreamVLNPolicy(
        backbone,
        processor,
        eos_token_ids=(63,),
        max_new_tokens=5,
        fast_action_decode=False,
        gpu_history_features=True,
        tree_decode=tree,
        fused_ops=True,
        split_attention=True,
        decode_linear=True,
        vision_kernel="sdpa",
    ).eval()
    return policy, batches


def _observations(count=2):
    return [
        make_streamvln_observation(torch.full((3, 8, 8), 50 + i, dtype=torch.uint8), f"go {i}")
        for i in range(count)
    ]


@pytest.mark.gpu
def test_tree_refresh_reaches_both_runtimes_and_batch_ownership(monkeypatch):
    from embodiinfer.policies.streamvln.tree import StreamVLNTreeRuntime

    with torch.inference_mode():
        policy, _ = _policy(monkeypatch, tree=True)
        policy.tree_refresh = 4
        single = StreamVLNTreeRuntime(policy.decoder)
        runtime = policy.create_batched_runtime(batch_size=1, workspace_tokens=256)
        assert single.drafter.refresh == runtime.drafter.refresh == 4
        assert runtime.stats()["tree_refresh"] == 4
        policy.tree_refresh = 16
        with pytest.raises(RuntimeError, match="configuration changed"):
            runtime.infer(_observations(1), ["a"])


def test_graph_inputs_keep_prior_addresses_on_growth_and_isolate_names_and_dtypes():
    arena = _GraphInputArena()
    old = arena.bind({"hidden": torch.arange(15.0).reshape(1, 5, 3)})["hidden"]
    original = old.clone()
    larger = arena.bind({"hidden": torch.full((2, 8, 3), 7.0)})["hidden"]
    assert larger.data_ptr() != old.data_ptr()
    assert torch.equal(old, original)
    assert arena.allocated_bytes == (16 + 64) * 4
    small = arena.bind({"hidden": torch.full((1, 4, 3), 9.0)})["hidden"]
    assert small.data_ptr() == old.data_ptr()
    assert torch.equal(old.view(-1)[:12], torch.full((12,), 9.0))
    separate = arena.bind({"position": torch.ones(4), "hidden": torch.ones(4, dtype=torch.long)})
    assert len({small.data_ptr(), separate["position"].data_ptr(), separate["hidden"].data_ptr()}) == 3
    # Reusing input storage never lets an arbitrary graph input mutate its source.
    source = torch.arange(12.0).reshape(3, 4).t()
    expected = source.clone()
    view = arena.bind({"hidden": source})["hidden"]
    assert view.is_contiguous() and torch.equal(view, expected)
    view.zero_()
    assert torch.equal(source, expected)


@pytest.mark.gpu
@torch.inference_mode()
def test_graph_input_arena_replays_older_graphs_after_larger_shape_capture():
    from embodiinfer.policies.streamvln.batching import _Call

    arena = _GraphInputArena()
    pool = torch.cuda.graph_pool_handle()
    calls = []
    for size in (15, 48, 12, 40):
        inputs = {"hidden": torch.randn(size, device="cuda"), "offset": torch.ones(size, device="cuda")}
        call = _Call(
            lambda values: values["hidden"].square() + values["offset"],
            inputs,
            capture=True,
            pool=pool,
            arena=arena,
        )
        calls.append((size, call))
    for size, call in calls[::-1] + calls:
        inputs = {"hidden": torch.randn(size, device="cuda"), "offset": torch.randn(size, device="cuda")}
        expected = inputs["hidden"].square() + inputs["offset"]
        assert torch.equal(call.replay(inputs), expected)
    assert arena.allocated_bytes == 3 * (16 + 64) * 4


def test_primitive_frames_keep_private_history_when_vision_scratch_is_reused():
    runtime = object.__new__(StreamVLNBatchedRuntime)
    runtime.batch_size = 2
    runtime.policy = SimpleNamespace(_retain_history_feature=lambda feature: feature.clone())
    scratch = torch.empty(2, 4, 3)
    calls = []

    def vision(pixels):
        calls.append(len(pixels))
        scratch[: len(pixels)].copy_(pixels[:, 0, 0, 0, None, None])
        return scratch[: len(pixels)]

    runtime._vision = vision

    def pixels(value):
        return torch.full((1, 3, 2, 2), float(value))

    committed = (torch.full((4, 3), 10.0), None, None, None)
    prepared = [
        SimpleNamespace(
            current_pixels=pixels(14),
            committed=SimpleNamespace(frame_features=committed),
            intermediate_pixels=((2, pixels(12)),),
        ),
        SimpleNamespace(
            current_pixels=pixels(25),
            committed=SimpleNamespace(frame_features=(None,) * 5),
            intermediate_pixels=((0, pixels(20)), (2, pixels(22)), (4, pixels(24))),
        ),
    ]
    current, histories = runtime._encode_primitive_frames(prepared)
    scratch.fill_(-99)
    assert calls == [2, 2, 2]
    assert [feature.unique().item() for feature in current] == [14, 25]
    assert [None if feature is None else feature.unique().item() for feature in histories[0]] == [
        10,
        None,
        12,
        None,
    ]
    assert [None if feature is None else feature.unique().item() for feature in histories[1]] == [
        20,
        None,
        22,
        None,
        24,
    ]
    assert committed[2] is None and prepared[1].committed.frame_features == (None,) * 5


def test_batch_owner_detects_int8_payload_and_scale_changes():
    runtime = object.__new__(StreamVLNBatchedRuntime)
    runtime.policy = nn.Module()
    runtime.policy.register_buffer("weight_int8", torch.ones(8, 8, dtype=torch.int8))
    runtime.policy.register_buffer("weight_scale", torch.ones(8))
    original = runtime._parameter_signature()
    runtime.policy.weight_scale.mul_(2)
    assert runtime._parameter_signature() != original
    changed = runtime._parameter_signature()
    runtime.policy.weight_int8 = runtime.policy.weight_int8.clone()
    assert runtime._parameter_signature() != changed


def test_prefill_buckets_share_shapes_and_keep_padding_outside_episode_slots():
    runtime = object.__new__(StreamVLNBatchedRuntime)
    runtime.device = torch.device("cpu")
    runtime.packed_prefill = True
    runtime.packed_prefill_bucket = 64
    runtime.pool_tokens = 512
    runtime._row_starts = [0, 256]
    runtime.policy = SimpleNamespace(backbone=SimpleNamespace(num_heads=16, num_kv_heads=2))
    runtime.counters = dict(prefill_live_tokens=0, prefill_padding_tokens=0, prefill_rows=0)
    recorded = []

    def text(inputs, *, packed, tree):
        assert packed and not tree
        recorded.append(inputs)
        return inputs["hidden"]

    runtime._text = text
    for sizes in ([51, 93], [49, 82]):
        embeddings = [torch.full((1, size, 3), float(row + 1)) for row, size in enumerate(sizes)]
        assert torch.equal(runtime._prefill(embeddings, [3, 7]), torch.tensor([[1.0] * 3, [2.0] * 3]))
        values = recorded[-1]
        total = sum(sizes)
        assert values["live"].tolist() == [total]
        assert values["slots"][total:].tolist() == list(range(512, 512 + 192 - total))
        assert len(set(values["slots"].tolist())) == 192
        assert not values["hidden"][:, total:].any() and not values["rotary"][:, total:].any()
        assert values["lengths"].tolist() == sizes
    assert {k: v.shape for k, v in recorded[0].items()} == {k: v.shape for k, v in recorded[1].items()}
    assert runtime.counters["prefill_padding_tokens"] == 48 + 61


@pytest.mark.gpu
@torch.inference_mode()
def test_bucketed_prefill_reuses_graph_for_new_lengths_and_preserves_private_kv(monkeypatch):
    policy, _ = _policy(monkeypatch)
    runtime = policy.create_batched_runtime(batch_size=2, workspace_tokens=256, packed_prefill_bucket=64)
    assert runtime.cache.capacity == 512 + 63
    torch.manual_seed(84)
    first = torch.randn(1, 53, 512, device="cuda", dtype=torch.bfloat16)
    second = torch.randn(1, 93, 512, device="cuda", dtype=torch.bfloat16)
    with runtime.startup_capture():
        runtime.prewarm_prefill([[51, 93]])
    assert runtime.stats()["text_graphs"] == 1
    private_prefixes = [runtime._row(i).key_storage[..., :past, :].clone() for i, past in enumerate([3, 7])]
    # A new ragged shape still has 192 physical rows, so it must replay the
    # existing graph. Compare against eager with exactly the same padding.
    graphs = runtime.graphs
    runtime.graphs = {}
    expected = runtime._prefill([first, second[:, :82]], [3, 7]).clone()
    expected_cache = runtime.cache.key_storage.clone()
    runtime.graphs = graphs
    before = runtime.stats()["text_replays"]
    actual = runtime._prefill([first, second[:, :82]], [3, 7])
    assert torch.equal(actual, expected)
    assert torch.equal(runtime.cache.key_storage, expected_cache)
    assert runtime.stats()["text_replays"] == before + 1
    for i, past in enumerate([3, 7]):
        assert torch.equal(runtime._row(i).key_storage[..., :past, :], private_prefixes[i])


@pytest.mark.gpu
@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("shared_inputs", [False, True])
def test_primitive_batch_graphs_preserve_mixed_strides_and_window_histories(monkeypatch, tree, shared_inputs):
    with torch.inference_mode():
        policy, _ = _policy(monkeypatch, tree)
        runtime = policy.create_batched_runtime(
            batch_size=2, workspace_tokens=128, shared_graph_buffers=shared_inputs
        )
        schedules = [(0, 0), (3, 2), (6, 5), (8, 9)]

        def replay():
            memories, records = [None, None], []
            for steps in schedules:
                observations = []
                for row, step in enumerate(steps):
                    start = 0 if memories[row] is None else memories[row].step_id
                    frames = torch.stack(
                        [
                            torch.full((3, 8, 8), 30 * (row + 1) + frame, dtype=torch.uint8)
                            for frame in range(start, step + 1)
                        ]
                    )
                    observation = make_streamvln_observation(
                        frames[-1], f"go {row}", uint8=True, metadata={"stream_step_id": step}
                    )
                    observation.images = frames
                    observations.append(observation)
                original = [
                    None if memory is None else memory.cache.key_storage[..., : memory.seq_len, :].clone()
                    for memory in memories
                ]
                results = runtime.infer(observations, ["a", "b"], memories)
                for row, result in enumerate(results):
                    memory = result.next_memory
                    assert memory.step_id == steps[row] + 1
                    assert len(memory.frame_features) == memory.step_id
                    for frame, feature in enumerate(memory.frame_features):
                        if frame % 2:
                            assert feature is None
                        else:
                            expected = torch.tensor(
                                (30 * (row + 1) + frame) / 255, device="cuda", dtype=torch.bfloat16
                            )
                            assert torch.equal(feature, expected.expand_as(feature))
                    if memories[row] is not None:
                        assert torch.equal(
                            memories[row].cache.key_storage[..., : memories[row].seq_len, :], original[row]
                        )
                    records.append(
                        (
                            result.traces[0].token_ids.clone(),
                            memory.cache.key_storage[..., : memory.seq_len, :].clone(),
                            memory.slow_frame_indices,
                        )
                    )
                memories = [result.next_memory for result in results]
            assert memories[0].slow_frame_indices == (0, 4)
            assert memories[1].slow_frame_indices == (0, 4)
            return records

        eager = replay()
        with runtime.startup_capture():
            replay()
        captured = replay()
        for (tokens, keys, slow), (other_tokens, other_keys, other_slow) in zip(eager, captured, strict=True):
            assert torch.equal(tokens, other_tokens)
            assert torch.equal(keys, other_keys)
            assert slow == other_slow
        assert runtime.stats()["vision_replays"] and runtime.stats()["text_replays"]


@pytest.mark.gpu
@pytest.mark.parametrize("tree", [False, True])
def test_batch_graphs_match_eager_and_failures_preserve_private_memories(monkeypatch, tree):
    with torch.inference_mode():
        policy, batches = _policy(monkeypatch, tree)
        runtime = policy.create_batched_runtime(batch_size=2, workspace_tokens=128)
        observations, names = _observations(), ["a", "b"]
        eager = runtime.infer(observations, names)
        assert batches[-1] == 2
        with runtime.startup_capture():
            runtime.infer(observations, names)
        captured = runtime.infer(observations, names)
        for a, b in zip(eager, captured):
            assert torch.equal(a.traces[0].token_ids, b.traces[0].token_ids)
            assert torch.equal(
                a.next_memory.cache.key_storage[..., : a.next_memory.seq_len, :],
                b.next_memory.cache.key_storage[..., : b.next_memory.seq_len, :],
            )
        assert runtime.stats()["text_replays"] > 0
        assert runtime.stats()["capture_frozen"]
        memories = [r.next_memory for r in captured]
        snapshots = [m.cache.key_storage[..., : m.seq_len, :].clone() for m in memories]
        calls = 0

        def cancel_after_prefill():
            nonlocal calls
            calls += 1
            return calls == 3

        with pytest.raises(SessionCancelledError):
            runtime.infer(observations, names, memories, cancelled=cancel_after_prefill)
        assert all(r is None for r in runtime.resident)
        for memory, snapshot in zip(memories, snapshots):
            assert torch.equal(memory.cache.key_storage[..., : memory.seq_len, :], snapshot)
        retry = runtime.infer(observations, names, memories)
        assert all(r.next_memory.step_id == 2 for r in retry)
        # Reordering episodes and replaying an older fork cannot overwrite a
        # newer output. Only the latest unique owner may reuse unused capacity.
        latest = [r.next_memory for r in retry]
        saved = [m.cache.key_storage[..., : m.seq_len, :].clone() for m in latest]
        runtime.infer(observations[::-1], names[::-1], memories[::-1])
        for memory, snapshot in zip(latest, saved):
            assert torch.equal(memory.cache.key_storage[..., : memory.seq_len, :], snapshot)
        with pytest.raises(RuntimeError, match="another CUDA stream"), torch.cuda.stream(torch.cuda.Stream()):
            runtime.infer(observations, names)
        with pytest.raises(ValueError, match="unique"):
            runtime.infer(observations, ["same", "same"])


@pytest.mark.gpu
def test_packed_prefill_isolates_episodes_and_uses_only_live_tokens(monkeypatch):
    with torch.inference_mode():
        policy, _ = _policy(monkeypatch)
        runtime = policy.create_batched_runtime(batch_size=2, workspace_tokens=256)
        torch.manual_seed(42)
        first = torch.randn(1, 51, 512, device="cuda", dtype=torch.bfloat16)
        second = torch.randn(1, 93, 512, device="cuda", dtype=torch.bfloat16)
        expected = runtime._prefill([first, second], [3, 7]).clone()
        original_keys = runtime._row(0).key_storage[..., :54, :].clone()
        actual = runtime._prefill([first, -second], [3, 7])
        assert torch.equal(actual[0], expected[0])
        assert not torch.equal(actual[1], expected[1])
        assert torch.equal(runtime._row(0).key_storage[..., :54, :], original_keys)
        assert runtime.stats()["prefill_live_tokens"] == 2 * (51 + 93)
        with runtime.startup_capture():
            runtime._prefill([first, second], [3, 7])
        assert torch.equal(runtime._prefill([first, second], [3, 7]), expected)
        # Profiles and weights are fixed for the graph's entire lifetime.
        policy.backbone.vision_kernel = "fused"
        with pytest.raises(RuntimeError, match="configuration changed"):
            runtime.infer(_observations(), ["a", "b"])


@pytest.mark.gpu
def test_batch_rejects_capacity_before_writing_history(monkeypatch):
    with torch.inference_mode():
        policy, _ = _policy(monkeypatch)
        runtime = policy.create_batched_runtime(batch_size=2, workspace_tokens=4)
        original = runtime.cache.key_storage.clone()
        with pytest.raises(ValueError, match="exceed workspace"):
            runtime.infer(_observations(), ["a", "b"])
        assert torch.equal(runtime.cache.key_storage, original)


@pytest.mark.gpu
def test_shared_pool_relocates_rows_and_preserves_episode_history(monkeypatch):
    with torch.inference_mode():
        policy, _ = _policy(monkeypatch, tree=True)
        runtime = policy.create_batched_runtime(batch_size=2, workspace_tokens=128, kv_pool_tokens=128)
        assert runtime.cache.capacity == 128  # Half the independent-row reservation.
        observations, names = _observations(), ["a", "b"]
        memories = [None, None]
        with runtime.startup_capture():
            for _ in range(5):  # Includes the toy processor's step-4 window reset.
                snapshots = [
                    None if m is None else m.cache.key_storage[..., : m.seq_len, :].clone() for m in memories
                ]
                results = runtime.infer(observations, names, memories)
                for memory, snapshot in zip(memories, snapshots):
                    if memory is not None:
                        assert torch.equal(memory.cache.key_storage[..., : memory.seq_len, :], snapshot)
                memories = [result.next_memory for result in results]
        assert all(memory.step_id == 5 for memory in memories)
        assert runtime.stats()["pool_relocations"] > 1
        saved = runtime.cache.key_storage.clone()
        with pytest.raises(ValueError, match="pooled KV cannot fit"):
            runtime._plan_rows([100, 100])
        assert torch.equal(runtime.cache.key_storage, saved)
