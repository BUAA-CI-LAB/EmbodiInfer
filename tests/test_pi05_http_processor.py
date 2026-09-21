import io
import json
import sys
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from embodiinfer.engine.serve.contracts import RawImage, RawPolicyRequest
from embodiinfer.policies.pi05.checkpoints import checkpoint_format
from embodiinfer.policies.pi05.checkpoints.openpi import OpenPiConfig
from embodiinfer.policies.pi05.checkpoints.orbax import convert_orbax_params
from embodiinfer.policies.pi05.processor_pi05 import (
    LeRobotProcessor,
    OpenPiProcessor,
    _ensure_pi05_processor_compatibility,
    _local_tokenizer_override,
    make_processor,
)
from embodiinfer.policies.pi05.serving import (
    Pi05ServingAdapter,
    Pi05ServingConfig,
    _feature_width,
    _image_name,
    _load_raw_image,
)


def test_pi05_image_name_preserves_policy_feature_keys():
    assert _image_name("observation.images.image") == "observation.images.image"
    assert _image_name("camera/front.jpg") == "front"
    assert _image_name("camera/WRIST.JPEG") == "WRIST"


def test_pi05_external_feature_width_is_distinct_from_model_padding():
    features = {"observation.state": SimpleNamespace(shape=(8,))}
    assert _feature_width(features, "observation.state", "state") == 8


def test_local_pi05_tokenizer_is_loaded_without_network(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {"tokenizer_name": "example/local-tokenizer"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    sentinel = SimpleNamespace(get_vocab=lambda: {"<unk>": 0, "text": 1}, all_special_ids=[0])
    calls = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append((name, kwargs))
            return sentinel

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))

    override = _local_tokenizer_override(str(checkpoint))

    assert override == {"tokenizer_processor": {"tokenizer": sentinel}}
    assert calls == [("example/local-tokenizer", {"local_files_only": True})]


def test_hub_pi05_checkpoint_keeps_online_resolution_available():
    assert _local_tokenizer_override("organization/pi05-checkpoint") == {}


def test_pi05_image_keys_map_deploy_fields_to_checkpoint_features():
    config = Pi05ServingConfig.from_mapping(
        {
            "state_fields": ["joints"],
            "image_fields": ["observation.images.image", "observation.images.wrist_image"],
            "image_keys": {
                "observation.images.image": "observation.images.base_0_rgb",
                "observation.images.wrist_image": "observation.images.left_wrist_0_rgb",
            },
        }
    )
    adapter = object.__new__(Pi05ServingAdapter)
    adapter._config = config
    adapter._native_image_keys = (
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    )
    adapter._width = 2
    adapter._height = 2
    adapter._processor = object.__new__(LeRobotProcessor)

    image = Image.new("RGB", (2, 2), color=(255, 0, 0))
    payload = BytesIO()
    image.save(payload, format="PNG")
    images = (
        RawImage("observation.images.image", "image/png", payload.getvalue()),
        RawImage("observation.images.wrist_image", "image/png", payload.getvalue()),
    )

    tensors, native_names = adapter._image_tensor_stack(images)

    assert tensors.shape == (2, 3, 2, 2)
    assert native_names == (
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
    )


def test_pi05_image_keys_reject_unknown_source_fields():
    with pytest.raises(ValueError, match="not listed in image_fields"):
        Pi05ServingConfig.from_mapping(
            {
                "state_fields": ["joints"],
                "image_fields": ["observation.images.image"],
                "image_keys": {
                    "observation.images.other": "observation.images.base_0_rgb",
                },
            }
        )


def test_pi05_checkpoint_processor_names_are_registered():
    pytest.importorskip("lerobot")
    from lerobot.processor import ProcessorStepRegistry, RelativeActionsProcessorStep

    _ensure_pi05_processor_compatibility()
    relative = ProcessorStepRegistry.get("relative_actions_processor")

    assert issubclass(relative, RelativeActionsProcessorStep)
    assert ProcessorStepRegistry.get("pi05_prepare_state_tokenizer_processor_step") is not None


def _observation_adapter(**config):
    adapter = object.__new__(Pi05ServingAdapter)
    adapter._config = Pi05ServingConfig.from_mapping(
        {"state_fields": ["observation.state"], "image_fields": ["image"], **config}
    )
    adapter._state_dim = 8
    adapter._processor = object.__new__(LeRobotProcessor)
    adapter._processor._state_dim = 8
    adapter._native_image_keys = ("observation.images.image", "observation.images.image2")
    adapter._width = adapter._height = 4
    return adapter


@pytest.mark.parametrize(
    "state",
    [
        {"observation.state": list(range(8))},
        {"observation": {"state": list(range(8))}},
        {"observation.state": list(range(8)), "observation": {"state": [-1] * 8}},
    ],
)
def test_pi05_state_accepts_flat_and_nested_keys_with_flat_precedence(state):
    result = _observation_adapter()._state_vector(state)
    torch.testing.assert_close(result, torch.arange(8, dtype=torch.float32), rtol=0, atol=0)


def test_pi05_state_preserves_configured_field_order_and_padding():
    adapter = _observation_adapter(state_fields=["gripper", "arm.position"])
    result = adapter._state_vector({"arm.position": [1, 2], "gripper": [3]})
    assert result.tolist() == [3, 1, 2, 0, 0, 0, 0, 0]


def test_openpi_state_does_not_invent_missing_physical_features(openpi_recipe):
    adapter = _observation_adapter()
    adapter._state_dim = openpi_recipe.state_dim
    adapter._processor = OpenPiProcessor(openpi_recipe)
    with pytest.raises(ValueError, match="every checkpoint state feature"):
        adapter._state_vector({"observation.state": [1, 2]})


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({}, "missing required field"),
        ({"observation": []}, "missing required field"),
        ({"observation.state": [0] * 9}, "state_dim too large"),
        ({"observation.state": [float("nan")]}, "non-finite"),
        ({"observation.state": [float("inf")]}, "non-finite"),
        ({"observation.state": [True]}, "numeric"),
        ({"observation.state": None, "observation": {"state": [1]}}, "numeric"),
    ],
)
def test_pi05_state_rejects_invalid_values_without_nested_fallback(state, message):
    with pytest.raises(ValueError, match=message):
        _observation_adapter()._state_vector(state)


def _raw_image(name, color):
    data = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(data, format="PNG")
    return RawImage(name=name, mime_type="image/png", data=data.getvalue())


def test_pi05_camera_mapping_preserves_input_order_and_pixels():
    adapter = _observation_adapter(
        image_fields=["observation.images.image", "observation.images.wrist_image"],
        image_keys={"observation.images.wrist_image": "observation.images.image2"},
    )
    images, names = adapter._image_tensor_stack(
        (
            _raw_image("observation.images.wrist_image", "blue"),
            _raw_image("observation.images.image", "red"),
        )
    )
    assert names == adapter._native_image_keys
    assert images.shape == (2, 3, 4, 4)
    assert images[0, :, 0, 0].tolist() == [1, 0, 0]
    assert images[1, :, 0, 0].tolist() == [0, 0, 1]


def test_pi05_camera_mapping_defaults_to_identity():
    adapter = _observation_adapter(image_fields=["observation.images.image"])
    _, names = adapter._image_tensor_stack((_raw_image("observation.images.image", "red"),))
    assert names == ("observation.images.image",)


@pytest.mark.parametrize("image_keys", [[], {"missing": "image2"}, {"image": ""}])
def test_pi05_camera_mapping_rejects_invalid_config(image_keys):
    with pytest.raises((TypeError, ValueError), match="image_keys"):
        _observation_adapter(image_keys=image_keys)


def test_pi05_camera_mapping_rejects_unsupported_target():
    adapter = _observation_adapter(image_keys={"image": "not-in-checkpoint"})
    with pytest.raises(ValueError, match="unsupported pi05 feature"):
        adapter._image_tensor_stack((_raw_image("image", "red"),))


def test_pi05_camera_mapping_rejects_missing_duplicate_and_colliding_images():
    adapter = _observation_adapter(image_keys={"image": "observation.images.image"})
    image = _raw_image("image", "red")
    with pytest.raises(ValueError, match="missing requested image"):
        adapter._image_tensor_stack(())
    with pytest.raises(ValueError, match="duplicate image"):
        adapter._image_tensor_stack((image, image))
    adapter = _observation_adapter(
        image_fields=["image", "wrist"],
        image_keys={"image": "observation.images.image", "wrist": "observation.images.image"},
    )
    with pytest.raises(ValueError, match="same pi05 feature"):
        adapter._image_tensor_stack((image, _raw_image("wrist", "blue")))


@pytest.fixture
def openpi_recipe(tmp_path, monkeypatch):
    stats = {
        name: {"q01": [-2.0] * 4, "q99": [2.0] * 4, "mean": [0.5] * 4, "std": [2.0] * 4}
        for name in ("state", "actions")
    }
    (tmp_path / "norm_stats.json").write_text(json.dumps({"norm_stats": stats}))
    (tmp_path / "tokenizer.model").write_bytes(b"test-tokenizer")

    class Tokenizer:
        calls = []

        def __init__(self, **kwargs):
            pass

        def encode(self, text, add_bos=False):
            self.calls.append((text, add_bos))
            return ([2] if add_bos else []) + [ord(c) for c in text]

    monkeypatch.setitem(sys.modules, "sentencepiece", SimpleNamespace(SentencePieceProcessor=Tokenizer))
    return OpenPiConfig.from_mapping(
        dict(
            action_horizon=10,
            state_dim=3,
            output_action_dim=3,
            action_dim=4,
            image_keys=["base", "wrist"],
            tokenizer="tokenizer.model",
            norm_stats="norm_stats.json",
            use_quantile_norm=True,
            discrete_state_input=False,
            delta_action_mask=[True, True, False],
        ),
        tmp_path,
    )


def test_openpi_formats_reject_missing_and_ambiguous_weights(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        checkpoint_format(tmp_path)
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "_METADATA").write_text("{}")
    assert checkpoint_format(tmp_path) == "openpi-orbax"
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match="both Orbax and PyTorch"):
        checkpoint_format(tmp_path)


def test_openpi_formats_distinguish_pytorch_dialects(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}")
    assert checkpoint_format(tmp_path) == "lerobot"
    (tmp_path / "openpi_config.json").write_text("{}")
    assert checkpoint_format(tmp_path) == "openpi-pytorch"


def test_openpi_prompt_without_state_uses_separate_newline(openpi_recipe):
    processor = OpenPiProcessor(openpi_recipe)
    tokens, masks = processor.tokenize("  pick_up\nfruit  ", np.zeros(4))
    assert processor.tokenizer.calls == [("pick up fruit", True), ("\n", False)]
    assert tokens[0, :14].tolist() == [2] + [ord(c) for c in "pick up fruit"]
    assert tokens[0, 14].item() == ord("\n")
    assert masks[0].sum().item() == 15
    assert not masks[0, 15:].any()


def test_openpi_discrete_state_is_opt_in(openpi_recipe):
    processor = OpenPiProcessor(replace(openpi_recipe, discrete_state_input=True))
    processor.tokenize("pick_fruit", np.array([-1, 0, 1, 0]))
    assert processor.tokenizer.calls == [("Task: pick fruit, State: 0 128 255 128;\nAction: ", True)]


@pytest.mark.parametrize("quantile", [True, False])
def test_openpi_actions_restore_observation_relative_chunk_not_cumulative(openpi_recipe, quantile):
    processor = OpenPiProcessor(replace(openpi_recipe, use_quantile_norm=quantile))
    normalized = torch.tensor([[0.0, 0.0, -1.0, 0.0], [1.0, -1.0, 1.0, 0.0]])
    state = torch.tensor([10.0, 20.0, 0.0])
    expected = (
        (normalized.numpy() + 1) / 2 * (4 + 1e-6) - 2 if quantile else normalized.numpy() * (2 + 1e-6) + 0.5
    )[:, :3]
    expected[:, :2] += np.array([10.0, 20.0])
    actual = processor.restore_actions(normalized, state)
    np.testing.assert_array_equal(actual.numpy(), expected)
    # A different session cannot alter this request's reference state.
    processor.restore_actions(normalized, torch.tensor([100.0, 200.0, 1.0]))
    torch.testing.assert_close(processor.restore_actions(normalized, state), actual, rtol=0, atol=0)


def test_openpi_camera_order_and_missing_camera(openpi_recipe):
    processor = OpenPiProcessor(openpi_recipe)
    images = {"wrist": torch.ones(3, 224, 224), "base": torch.zeros(3, 224, 224)}
    batch = processor.prepare(torch.zeros(3), images, "pick")
    assert batch.images[0].eq(-1).all()
    assert batch.images[1].eq(1).all()
    assert all(mask.tolist() == [True] for mask in batch.img_masks)
    with pytest.raises(ValueError, match="missing OpenPI camera"):
        processor.prepare(torch.zeros(3), {"base": images["base"]}, "pick")


def test_openpi_statistics_reject_nan_and_bad_dimensions(openpi_recipe):
    path = Path(openpi_recipe.norm_stats)
    payload = json.loads(path.read_text())
    payload["norm_stats"]["actions"]["q01"] = [float("nan")] * 4
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="finite action_dim"):
        OpenPiProcessor(openpi_recipe)


def test_openpi_resize_preserves_aspect_ratio_and_black_padding(openpi_recipe):
    data = BytesIO()
    Image.new("RGB", (8, 4), "red").save(data, format="PNG")
    image = _load_raw_image(
        RawImage("image", "image/png", data.getvalue()), 4, 4, processor=OpenPiProcessor(openpi_recipe)
    )
    assert image.shape == (3, 4, 4)
    assert not image[:, 0].any() and not image[:, 3].any()
    assert (image[0, 1:3] == 1).all() and not image[1:].any()


def test_openpi_conversion_rejects_incomplete_trees():
    with pytest.raises(ValueError, match="missing tensor"):
        convert_orbax_params({"action_in_proj": {"kernel": np.zeros((2, 2))}})


def test_pi05_factory_takes_chunk_dimensions_from_loaded_checkpoint(monkeypatch):
    from embodiinfer.policies.pi05 import modeling_pi05

    class LoadedPolicy:
        def __init__(self, config, **kwargs):
            self.config = config
            self._lerobot = SimpleNamespace(
                config=SimpleNamespace(max_action_dim=32, chunk_size=10, num_inference_steps=10)
            )

    monkeypatch.setattr(modeling_pi05, "Pi05Policy", LoadedPolicy)
    policy = modeling_pi05._build_pi05(checkpoint="test")
    assert policy.config.action_horizon == 10
    assert policy.config.action_dim == 32
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        modeling_pi05._build_pi05(checkpoint="test", action_horizon=50)


@pytest.mark.parametrize("valid_weights", [True, False])
@pytest.mark.parametrize("storage", ["pytorch", "orbax"])
def test_openpi_formats_share_strict_loading(openpi_recipe, tmp_path, monkeypatch, valid_weights, storage):
    pytest.importorskip("lerobot")
    from lerobot.policies.pi05 import modeling_pi05 as native
    from safetensors.torch import save_file

    from embodiinfer.policies.pi05 import checkpoints
    from embodiinfer.policies.pi05.checkpoints import orbax

    class NativePolicy:
        def __init__(self, config):
            self.model = torch.nn.Linear(2, 2, bias=False)
            self.model.paligemma_with_expert = SimpleNamespace(
                paligemma=SimpleNamespace(lm_head=None), gemma_expert=SimpleNamespace(lm_head=None)
            )

        def eval(self):
            return self

        def to(self, device):
            return self

    calls = []

    def converted_weights(root):
        assert storage == "orbax", "PyTorch checkpoints must not import/restore Orbax"
        calls.append(root)
        return tmp_path / "converted.safetensors"

    monkeypatch.setattr(native, "PI05Policy", NativePolicy)
    monkeypatch.setattr(orbax, "cached_orbax_weights", converted_weights)
    (tmp_path / "openpi_config.json").write_text(json.dumps(asdict(openpi_recipe)))
    weights = {"weight" if valid_weights else "unknown": torch.ones(2, 2)}
    if storage == "orbax":
        (tmp_path / "params").mkdir()
        (tmp_path / "params" / "_METADATA").write_text("{}")
    name = "converted.safetensors" if storage == "orbax" else "model.safetensors"
    save_file(weights, str(tmp_path / name))
    if valid_weights:
        loaded, recipe = checkpoints.load_checkpoint(str(tmp_path))
        torch.testing.assert_close(loaded.model.weight, weights["weight"], rtol=0, atol=0)
        assert recipe.action_horizon == 10
    else:
        with pytest.raises(RuntimeError, match="Missing key"):
            checkpoints.load_checkpoint(str(tmp_path))
    assert calls == ([tmp_path] if storage == "orbax" else [])


@pytest.mark.parametrize("device", [None, "cpu"])
@pytest.mark.parametrize("local", [False, True])
def test_lerobot_loading_preserves_hub_and_device_behavior(tmp_path, monkeypatch, device, local):
    pytest.importorskip("lerobot")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    from embodiinfer.policies.pi05.checkpoints import load_checkpoint

    if local:
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"weights")
    source = str(tmp_path) if local else "organization/pi05-checkpoint"
    config = SimpleNamespace(device="cuda")
    calls = []
    model = SimpleNamespace(eval=lambda: model)

    def native_load(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return model

    monkeypatch.setattr(PI05Policy, "from_pretrained", native_load)
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda path: config)
    loaded, recipe = load_checkpoint(source, load_device=device)
    assert loaded is model and recipe is None
    assert calls == [(source, {} if device is None else {"config": config})]
    assert config.device == ("cuda" if device is None else device)


def test_processor_selection_uses_recipe_not_checkpoint_path(openpi_recipe, monkeypatch):
    from embodiinfer.policies.pi05 import processor_pi05

    calls = []
    sentinel = object()

    def lerobot_processor(policy, checkpoint):
        calls.append((policy, checkpoint))
        return sentinel

    monkeypatch.setattr(processor_pi05, "LeRobotProcessor", lerobot_processor)
    native = object()
    policy = SimpleNamespace(_lerobot=native, openpi_config=openpi_recipe)
    assert isinstance(make_processor(policy, "any/path"), OpenPiProcessor)
    assert calls == []
    policy.openpi_config = None
    assert make_processor(policy, "any/path") is sentinel
    assert calls == [(native, "any/path")]


@pytest.mark.parametrize("dialect", ["lerobot", "openpi"])
def test_serving_uses_common_processor_and_request_local_action_state(openpi_recipe, monkeypatch, dialect):
    pytest.importorskip("lerobot")
    from lerobot.policies import factory
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    from embodiinfer.policies.pi05 import processor_pi05

    native = SimpleNamespace(config=openpi_recipe.native_config())
    native._preprocess_images = lambda batch: (
        [batch[name].unsqueeze(0) for name in openpi_recipe.image_keys],
        [torch.ones(1, dtype=torch.bool) for _ in openpi_recipe.image_keys],
    )

    def preprocessor(payload):
        assert payload["task"] == "pick"
        assert payload["observation.state"].shape == (3,)
        return {
            **payload,
            OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3]]),
            OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True]]),
        }

    monkeypatch.setattr(processor_pi05, "_ensure_pi05_processor_compatibility", lambda: None)
    monkeypatch.setattr(factory, "make_pre_post_processors", lambda *a, **kw: (preprocessor, lambda x: x + 2))
    raw_actions = torch.tensor([[0.0, 0.0, -1.0, 9.0], [1.0, -1.0, 1.0, 9.0], [0.5, 0.5, 0.5, 9.0]])
    batches = []

    def execute(batch):
        batches.append(batch)
        return [SimpleNamespace(actions=raw_actions, latency_ms=1.0)]

    core = SimpleNamespace(
        policy=SimpleNamespace(_lerobot=native, openpi_config=openpi_recipe if dialect == "openpi" else None),
        execute=execute,
        policy_version="revision",
    )
    adapter = Pi05ServingAdapter(
        core=core,
        checkpoint="organization/pi05-checkpoint",
        config={
            "state_fields": ["observation.state"],
            "image_fields": ["front", "hand"],
            "image_keys": {"front": "base", "hand": "wrist"},
            "return_steps": 2,
        },
    )
    assert isinstance(adapter._processor, OpenPiProcessor if dialect == "openpi" else LeRobotProcessor)
    for index, state in enumerate(([10.0, 20.0, 0.0], [100.0, 200.0, 1.0])):
        request = RawPolicyRequest(
            session_id=f"session-{index}",
            request_id=f"request-{index}",
            step_id=0,
            instruction="pick",
            state={"observation.state": state},
            images=(_raw_image("hand", "blue"), _raw_image("front", "red")),
            metadata={},
        )
        result = adapter.infer(request)
        expected = raw_actions[:2, :3].numpy() + 2
        if dialect == "openpi":
            expected = (raw_actions[:2, :3].numpy() + 1) / 2 * (4 + 1e-6) - 2
            expected[:, :2] += np.array(state[:2], dtype=np.float32)
        np.testing.assert_array_equal(result.actions[0].values["data"], expected)
        assert result.action_space == "pi05.action_chunk.v1"
        assert result.policy_revision == "revision" and result.timing == {"policy_ms": 1.0}
        batch = batches[-1]
        assert batch.request_ids == [request.request_id]
        assert len(batch.images) == 2 and batch.images[0].shape == (1, 3, 224, 224)
        assert batch.images[0][0, 0, 0, 0].item() == 1.0
        assert batch.images[1][0, 2, 0, 0].item() == 1.0


@pytest.mark.parametrize("vocab", [{}, {"<bos>": 0, "<unk>": 1}])
def test_local_pi05_rejects_tokenizer_with_no_text_vocabulary(tmp_path, monkeypatch, vocab):
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {"steps": [{"registry_name": "tokenizer_processor", "config": {"tokenizer_name": "test/tokens"}}]}
        )
    )
    tokenizer = SimpleNamespace(get_vocab=lambda: vocab, all_special_ids=[0, 1])
    loader = SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=loader))
    with pytest.raises(ValueError, match="contains no text tokens.*HF_HOME"):
        _local_tokenizer_override(str(tmp_path))


@pytest.mark.parametrize(
    "size,black_rows,black_columns",
    [
        ((640, 480), 56, 0),
        ((480, 640), 0, 56),
        ((224, 224), 0, 0),
        ((4, 3), 56, 0),
    ],
)
def test_lerobot_wire_resize_preserves_geometry(size, black_rows, black_columns):
    pytest.importorskip("lerobot")
    processor = object.__new__(LeRobotProcessor)
    pixels = processor.resize_image(Image.new("RGB", size, "white"), 224, 224)
    assert pixels.shape == (3, 224, 224)
    assert int((pixels == 0).all(axis=0).all(axis=1).sum()) == black_rows
    assert int((pixels == 0).all(axis=0).all(axis=0).sum()) == black_columns
    assert pixels[:, 112, 112].tolist() == [1.0, 1.0, 1.0]


def test_lerobot_resize_passes_original_geometry_to_native_processor(monkeypatch):
    seen = []

    def native_resize(pixels, height, width):
        seen.append((pixels.shape, height, width, pixels[0, 0, 0].tolist()))
        return torch.full((1, height, width, 3), 0.25)

    monkeypatch.setitem(
        sys.modules,
        "lerobot.policies.pi05.modeling_pi05",
        SimpleNamespace(resize_with_pad_torch=native_resize),
    )
    actual = object.__new__(LeRobotProcessor).resize_image(Image.new("RGB", (8, 4), "red"), 6, 6)
    assert seen == [(torch.Size([1, 4, 8, 3]), 6, 6, [1.0, 0.0, 0.0])]
    assert actual.shape == (3, 6, 6)
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, np.full((3, 6, 6), 0.25, dtype=np.float32))
