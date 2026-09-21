"""Compare two validated ActiveVLN parity reference bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.activevln.schema import validate_bundle

_EXACT_SUFFIXES = (
    "/context_input_ids",
    "/context_attention_mask",
    "/delta_input_ids",
    "/delta_attention_mask",
    "/position_ids",
    "/response_position_ids",
    "/response_token_ids",
    "/response_action_mask",
    "/topk_token_ids",
    "/parsed_action_tensor",
    "/parsed_action_mask",
)
_REQUIRED_OUTPUT_SUFFIXES = (
    "/response_token_ids",
    "/response_action_mask",
    "/response_token_logprobs",
    "/topk_token_ids",
    "/topk_logprobs",
    "/parsed_action_tensor",
    "/parsed_action_mask",
)


def _load_tensors(root: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(root / "tensors.safetensors", device="cpu")


def required_output_key_failures(left: dict, right: dict) -> list[str]:
    failures = []
    for suffix in _REQUIRED_OUTPUT_SUFFIXES:
        left_keys = {key for key in left if key.endswith(suffix)}
        right_keys = {key for key in right if key.endswith(suffix)}
        if left_keys != right_keys:
            failures.append(
                f"required {suffix} keys differ: missing-left={sorted(right_keys - left_keys)}, "
                f"missing-right={sorted(left_keys - right_keys)}"
            )
    return failures


def _position_ids_match_full_incremental(
    full: torch.Tensor, incremental: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align a delta mRoPE tensor with the matching full-history suffix.

    The full-history producer records positions for the whole prompt; the
    incremental producer records positions only for the newly appended turn.
    They are intentionally different shapes after turn 1.  The invariant is
    that the incremental positions equal the suffix of the full prompt.
    """
    if full.ndim != 3 or incremental.ndim != 3:
        raise ValueError("ActiveVLN position_ids must have shape [3, B, T]")
    if full.shape[:2] != incremental.shape[:2] or full.shape[-1] < incremental.shape[-1]:
        raise ValueError(
            "cannot align full/incremental position_ids: "
            f"full={tuple(full.shape)} incremental={tuple(incremental.shape)}"
        )
    return full[:, :, -incremental.shape[-1] :], incremental


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    left_manifest = validate_bundle(args.left)
    right_manifest = validate_bundle(args.right)
    left = _load_tensors(args.left)
    right = _load_tensors(args.right)
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError("reference bundles have no common tensors")

    failures = required_output_key_failures(left, right)
    if failures:
        raise AssertionError("ActiveVLN reference schema mismatch:\n" + "\n".join(failures))

    compared = 0
    logprob_errors: list[torch.Tensor] = []
    for key in common:
        a, b = left[key], right[key]
        if key.endswith("/position_ids") and {left_manifest["producer"], right_manifest["producer"]} == {
            "hf_full",
            "hf_incremental",
        }:
            full, incremental = (a, b) if left_manifest["producer"] == "hf_full" else (b, a)
            try:
                a, b = _position_ids_match_full_incremental(full, incremental)
            except ValueError as exc:
                failures.append(f"{key}: {exc}")
                continue
        if a.shape != b.shape:
            failures.append(f"{key}: shape {tuple(a.shape)} != {tuple(b.shape)}")
            continue
        try:
            if key.endswith(_EXACT_SUFFIXES) or not (a.is_floating_point() or b.is_floating_point()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            else:
                torch.testing.assert_close(a.float(), b.float(), rtol=args.rtol, atol=args.atol)
        except AssertionError as exc:
            failures.append(f"{key}: {str(exc).splitlines()[0]}")
        if key.endswith("/response_token_logprobs"):
            logprob_errors.append((a.float() - b.float()).abs().flatten())
        compared += 1

    if failures:
        raise AssertionError("ActiveVLN reference mismatch:\n" + "\n".join(failures[:20]))
    print(
        f"ActiveVLN references match: {left_manifest['producer']} vs "
        f"{right_manifest['producer']}, tensors={compared}"
    )
    if logprob_errors:
        errors = torch.cat(logprob_errors)
        print(
            f"selected-token logprob abs error: max={errors.max().item():.8g} mean={errors.mean().item():.8g}"
        )


if __name__ == "__main__":
    main()
