from types import SimpleNamespace

import pytest
import torch

from embodiinfer.models.qwen25_vl.next_token import Qwen25VLNextTokenForward
from embodiinfer.models.qwen25_vl.shape_buckets import (
    QWEN25_VL_TEXT_BUCKET_SCHEMA,
    apply_qwen25_vl_text_bucket,
    normalize_qwen25_vl_text_buckets,
    qwen25_vl_bucket_schema,
    qwen25_vl_text_bucket_key,
)


def _encoded(sequence_length: int = 3) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.arange(1, sequence_length + 1).view(1, -1),
        "attention_mask": torch.ones((1, sequence_length), dtype=torch.long),
        "pixel_values": torch.arange(24, dtype=torch.float32).view(6, 4),
        "image_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.long),
    }


def test_text_bucket_normalization_is_strict() -> None:
    assert normalize_qwen25_vl_text_buckets(()) == ()
    assert normalize_qwen25_vl_text_buckets((4, 8, 16)) == (4, 8, 16)
    for invalid in ((8, 4), (4, 4), (0,), (-1,), (True,)):
        with pytest.raises(ValueError, match="positive|strictly increasing"):
            normalize_qwen25_vl_text_buckets(invalid)


def test_text_bucket_left_pads_only_text_tensors() -> None:
    encoded = _encoded()
    pixels = encoded["pixel_values"]
    grid = encoded["image_grid_thw"]
    bucketed, bucket = apply_qwen25_vl_text_bucket(
        encoded,
        pad_token_id=99,
        buckets=(5, 8),
    )

    assert bucket.as_dict() == {
        "schema": "qwen25_vl_left_masked_text_v3",
        "source_length": 3,
        "target_length": 5,
        "left_padding": 2,
    }
    assert bucketed["input_ids"].tolist() == [[99, 99, 1, 2, 3]]
    assert bucketed["attention_mask"].tolist() == [[0, 0, 1, 1, 1]]
    assert bucketed["pixel_values"] is pixels
    assert bucketed["image_grid_thw"] is grid
    torch.testing.assert_close(bucketed["pixel_values"], pixels, rtol=0, atol=0)
    torch.testing.assert_close(bucketed["image_grid_thw"], grid, rtol=0, atol=0)
    assert bucket.bucket_id == (QWEN25_VL_TEXT_BUCKET_SCHEMA, 5)
    assert qwen25_vl_text_bucket_key(bucketed, (5, 8)) == bucket.bucket_id


def test_text_bucket_never_truncates() -> None:
    with pytest.raises(ValueError, match="truncation is forbidden"):
        apply_qwen25_vl_text_bucket(
            _encoded(9),
            pad_token_id=0,
            buckets=(4, 8),
        )


def test_no_bucket_preserves_original_mapping_and_exact_shape_key() -> None:
    encoded = _encoded()
    result, bucket = apply_qwen25_vl_text_bucket(
        encoded,
        pad_token_id=None,
        buckets=(),
    )
    assert result is encoded
    assert bucket.left_padding == 0
    assert qwen25_vl_text_bucket_key(result, ()) == (
        "qwen25_vl_exact_text_shape_v1",
        3,
    )


def test_bucket_schema_declares_visual_shapes_exact() -> None:
    assert qwen25_vl_bucket_schema("panoramic", (512, 768)) == {
        "schema": "qwen25_vl_left_masked_text_v3",
        "profile": "panoramic",
        "text_buckets": [512, 768],
        "text_padding": "left_attention_mask_zero",
        "truncation": False,
        "pixel_values": "exact",
        "image_grid_thw": "exact",
        "history_count": "exact",
        "candidate_count": "exact",
        "batch": "exact",
    }


class _LanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(128, 4)
        self.last_attention_mask: torch.Tensor | None | object = object()
        self.last_position_ids: torch.Tensor | None = None

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        del use_cache, return_dict
        self.last_attention_mask = attention_mask
        self.last_position_ids = position_ids
        return (inputs_embeds,)


class _Visual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = torch.nn.Identity()
        self.merger = torch.nn.Identity()
        self.blocks = torch.nn.ModuleList()
        self.fullatt_block_indexes: tuple[int, ...] = ()
        self.spatial_merge_unit = 1


def _forward_fixture(*, use_text_attention_mask: bool | None):
    language_model = _LanguageModel()
    model = SimpleNamespace(
        model=SimpleNamespace(visual=_Visual(), language_model=language_model),
        lm_head=torch.nn.Identity(),
        config=SimpleNamespace(image_token_id=99),
    )
    kwargs = {} if use_text_attention_mask is None else {"use_text_attention_mask": use_text_attention_mask}
    forward = Qwen25VLNextTokenForward(model, **kwargs)
    forward._attention_selection = object()
    return forward, language_model


@pytest.mark.parametrize("use_text_attention_mask", [None, False, True])
def test_next_token_forward_mask_is_bucket_opt_in_only(
    use_text_attention_mask: bool | None,
) -> None:
    forward, language_model = _forward_fixture(use_text_attention_mask=use_text_attention_mask)
    attention_mask = torch.tensor([[0, 1, 1]], dtype=torch.long)
    logits = forward(
        torch.tensor([[99, 2, 3]], dtype=torch.long),
        torch.ones((1, 4)),
        attention_mask,
        torch.arange(3).view(1, 1, 3).expand(3, -1, -1),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.empty((1, 1)),
        torch.empty((1, 1)),
    )
    assert logits.shape == (1, 4)
    if use_text_attention_mask is True:
        expected = torch.tensor([[[[False, False, False], [False, True, False], [False, True, True]]]])
        assert isinstance(language_model.last_attention_mask, dict)
        torch.testing.assert_close(language_model.last_attention_mask["full_attention"], expected)
    else:
        assert language_model.last_attention_mask is None


def _run_masked_dummy(
    forward: Qwen25VLNextTokenForward,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> None:
    forward(
        input_ids,
        torch.ones((1, 4)),
        attention_mask,
        position_ids,
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.empty((1, 1)),
        torch.empty((1, 1)),
    )


def test_left_padding_preserves_three_axis_mrope_suffix() -> None:
    exact_forward, exact_language = _forward_fixture(use_text_attention_mask=True)
    padded_forward, padded_language = _forward_fixture(use_text_attention_mask=True)
    exact_positions = torch.stack(
        (
            torch.arange(3),
            torch.arange(3) + 10,
            torch.arange(3) + 20,
        )
    ).unsqueeze(1)
    padded_positions = torch.cat((torch.zeros((3, 1, 2), dtype=torch.long), exact_positions), dim=-1)
    _run_masked_dummy(
        exact_forward,
        input_ids=torch.tensor([[99, 2, 3]]),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        position_ids=exact_positions,
    )
    padded_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)
    _run_masked_dummy(
        padded_forward,
        input_ids=torch.tensor([[0, 0, 99, 2, 3]]),
        attention_mask=padded_mask,
        position_ids=padded_positions,
    )
    assert exact_language.last_position_ids is exact_positions
    assert padded_language.last_position_ids is padded_positions
    expected_mask = (
        torch.ones((5, 5), dtype=torch.bool).tril()[None, None] & padded_mask[:, None, None, :].bool()
    )
    assert isinstance(padded_language.last_attention_mask, dict)
    torch.testing.assert_close(padded_language.last_attention_mask["full_attention"], expected_mask)
    assert not bool(padded_mask[:, :2].any().item())
    assert torch.equal(
        padded_language.last_position_ids[..., -3:],
        exact_language.last_position_ids,
    )
