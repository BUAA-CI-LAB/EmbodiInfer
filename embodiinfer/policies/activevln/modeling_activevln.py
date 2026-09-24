"""ActiveVLN R2R policy over a self-hosted Qwen2.5-VL text forward."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ...exceptions import SessionCancelledError, UnsupportedRecurrentModeError
from ...layers import get_attention_backend
from ...types import DecodeTrace, Observation
from ..base import PrefixState, VLAPolicy
from ..config import VLAPolicyConfig
from ..decoder import AutoregressiveDecoder, DecodeResult
from ..factory import register_policy
from .cache_activevln import ActiveVLNMemory, BranchedActiveVLNMemory
from .processor_activevln import ActiveVLNBatch, ActiveVLNProcessor, ProcessedTurn
from .prompt_activevln import actions_to_tensor, parse_r2r_actions

if TYPE_CHECKING:
    from .cuda_graph import ActiveVLNGraphRuntime


ACTIVEVLN_REPO = "https://github.com/arvillion/ActiveVLN"
ACTIVEVLN_COMMIT = "3a0c63b00e4f42c828cc74c3554afce17641da60"
ACTIVEVLN_CHECKPOINT = "Arvil/Qwen2.5-VL-3B_rl_r2r_4000"
ACTIVEVLN_REVISION = "160987313e3e869705f42400d1b8f28177044518"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def mrope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen2.5-VL rotary tables, matching transformers 4.51.3."""
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32) / head_dim)
    )
    inv = inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
    pos = position_ids[:, :, None, :].float()
    freqs = (inv @ pos).transpose(2, 3)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sections: tuple[int, int, int] | list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    split = list(sections) * 2
    cos = torch.cat([part[i % 3] for i, part in enumerate(cos.split(split, dim=-1))], dim=-1)
    sin = torch.cat([part[i % 3] for i, part in enumerate(sin.split(split, dim=-1))], dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def rectangular_causal_mask(
    prefix_len: int,
    query_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive ``[1,1,Q,prefix+Q]`` mask without a full-history square allocation."""
    min_value = torch.finfo(dtype).min
    prefix = torch.zeros(1, 1, query_len, prefix_len, device=device, dtype=dtype)
    current = torch.triu(
        torch.full((query_len, query_len), min_value, device=device, dtype=dtype),
        diagonal=1,
    )[None, None]
    return torch.cat([prefix, current], dim=-1)


def build_mrope_position_ids(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    *,
    vision_start_token_id: int,
    image_token_id: int,
    spatial_merge_size: int,
    offset: int = 0,
) -> tuple[torch.Tensor, int]:
    """Image+text position construction from Qwen2.5-VL 4.51.3, specialized to B=1."""
    if input_ids.shape[0] != 1:
        raise ValueError("ActiveVLN mRoPE currently supports batch size 1")
    tokens = input_ids[0].tolist()
    if image_grid_thw is None or image_token_id not in tokens:
        pos = torch.arange(len(tokens), device=input_ids.device, dtype=input_ids.dtype) + offset
        return pos[None, None].expand(3, 1, -1), int(pos[-1].item()) + 1 if len(tokens) else offset

    vision_starts = torch.where(input_ids[0] == vision_start_token_id)[0]
    vision_types = input_ids[0, vision_starts + 1]
    image_count = int((vision_types == image_token_id).sum().item())
    if image_count != image_grid_thw.shape[0]:
        raise ValueError(
            f"image grid count mismatch: {image_count} image markers vs {image_grid_thw.shape[0]} grids"
        )

    chunks: list[torch.Tensor] = []
    start = 0
    for grid_index in range(image_count):
        image_start = tokens.index(image_token_id, start)
        text_len = image_start - start
        chunk_start = int(chunks[-1].max().item()) + 1 if chunks else offset
        if text_len:
            chunks.append(
                torch.arange(text_len, device=input_ids.device, dtype=input_ids.dtype)[None].expand(3, -1)
                + chunk_start
            )
        t, h, w = (int(x) for x in image_grid_thw[grid_index].tolist())
        h //= spatial_merge_size
        w //= spatial_merge_size
        grid_start = (int(chunks[-1].max().item()) + 1) if chunks else offset
        t_index = torch.arange(t, device=input_ids.device).view(-1, 1).expand(-1, h * w).flatten()
        h_index = torch.arange(h, device=input_ids.device).view(1, -1, 1).expand(t, -1, w).flatten()
        w_index = torch.arange(w, device=input_ids.device).view(1, 1, -1).expand(t, h, -1).flatten()
        chunks.append(torch.stack([t_index, h_index, w_index]).to(input_ids.dtype) + grid_start)
        start = image_start + t * h * w

    if start < len(tokens):
        text_start = int(chunks[-1].max().item()) + 1 if chunks else offset
        text_len = len(tokens) - start
        chunks.append(
            torch.arange(text_len, device=input_ids.device, dtype=input_ids.dtype)[None].expand(3, -1)
            + text_start
        )
    positions = torch.cat(chunks, dim=1)
    if positions.shape[1] != input_ids.shape[1]:
        raise ValueError(
            f"mRoPE positions do not align with tokens: {positions.shape[1]} vs {input_ids.shape[1]}"
        )
    return positions[:, None], int(positions.max().item()) + 1


def _rmsnorm(norm, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return norm.weight * xf.to(dtype)


def _mlp(mlp, x: torch.Tensor) -> torch.Tensor:
    return mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))


def _apply_repetition_penalty(logits: torch.Tensor, token_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or token_ids.numel() == 0:
        return logits
    scores = logits.gather(1, token_ids)
    scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    return logits.scatter(1, token_ids, scores)


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    remove = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1) > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)


@dataclass
class PreparedActiveVLNTurn:
    """Device-ready turn, with CPU image/token processing completed."""

    turn: ProcessedTurn
    positions: torch.Tensor
    memory: ActiveVLNMemory | None
    next_position: int


@dataclass
class ActiveVLNGeneration:
    """Generated tokens before final text/action conversion and output transfer."""

    memory: ActiveVLNMemory
    token_ids: torch.Tensor
    token_logprobs: torch.Tensor
    stop_reason: str


@dataclass
class ActiveVLNPrefix:
    memory: ActiveVLNMemory
    next_logits: torch.Tensor
    batch_size: int = 1

    def to(self, device: torch.device | str) -> ActiveVLNPrefix:
        return ActiveVLNPrefix(self.memory.to(device), self.next_logits.to(device), self.batch_size)

    def expand(self, num_samples: int) -> PrefixState:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if num_samples == 1:
            return self
        return ActiveVLNBranchPrefix(
            tuple(ActiveVLNPrefix(self.memory.fork(), self.next_logits.clone()) for _ in range(num_samples))
        )


@dataclass(frozen=True)
class ActiveVLNBranchPrefix:
    """Independent L1-shared candidates decoded by the serial-ragged anchor."""

    branches: tuple[ActiveVLNPrefix, ...]

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("branch prefix cannot be empty")

    @property
    def batch_size(self) -> int:
        return len(self.branches)

    @property
    def memory(self) -> BranchedActiveVLNMemory:
        return BranchedActiveVLNMemory(tuple(branch.memory for branch in self.branches))

    def to(self, device: torch.device | str) -> ActiveVLNBranchPrefix:
        return ActiveVLNBranchPrefix(tuple(branch.to(device) for branch in self.branches))

    def expand(self, num_samples: int) -> ActiveVLNBranchPrefix:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        return ActiveVLNBranchPrefix(
            tuple(
                ActiveVLNPrefix(branch.memory.fork(), branch.next_logits.clone())
                for branch in self.branches
                for _ in range(num_samples)
            )
        )

    def compact(
        self, keep: list[int] | tuple[int, ...] | torch.Tensor
    ) -> ActiveVLNPrefix | ActiveVLNBranchPrefix:
        indices = keep.tolist() if isinstance(keep, torch.Tensor) else list(keep)
        selected = tuple(self.branches[int(index)] for index in indices)
        if not selected:
            raise ValueError("cannot compact all ActiveVLN prefix branches")
        return selected[0] if len(selected) == 1 else ActiveVLNBranchPrefix(selected)


@dataclass
class ARRecomputeState:
    token_ids: torch.Tensor
    action_mask: torch.Tensor

    def to(self, device: torch.device | str) -> ARRecomputeState:
        return ARRecomputeState(self.token_ids.to(device), self.action_mask.to(device))


class _ActiveVLNDecoder(AutoregressiveDecoder):
    def __init__(self, policy: ActiveVLNPolicy) -> None:
        self.policy = policy

    def _distribution(self, logits: torch.Tensor, memory: ActiveVLNMemory) -> torch.Tensor:
        logits = _apply_repetition_penalty(
            logits,
            # ``append_token`` grows the backing buffer in-place. Autograd's
            # gather/scatter saves these indices, so use an immutable snapshot
            # during differentiable GRPO recompute.
            memory.token_ids.clone(),
            self.policy.repetition_penalty,
        )
        # Hugging Face generation applies temperature/top-p only to sampled
        # decoding.  Greedy generation still uses repetition penalty, but its
        # categorical score must be derived from the unwarped logits.
        if not self.policy.do_sample:
            return logits
        logits = logits / self.policy.temperature
        return _top_p_filter(logits, self.policy.top_p)

    def _select(
        self,
        logits: torch.Tensor,
        memory: ActiveVLNMemory,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        effective = self._distribution(logits, memory)
        log_probs = torch.log_softmax(effective, dim=-1)
        if self.policy.do_sample:
            token = torch.multinomial(torch.softmax(effective, dim=-1), 1, generator=generator)
        else:
            token = effective.argmax(dim=-1, keepdim=True)
        return token, log_probs.gather(1, token).squeeze(1)

    def _decode_one(
        self,
        state: torch.Tensor | None,
        prefix: ActiveVLNPrefix,
        num_steps: int,
        bucket: int,
        graphs,
        *,
        generator: torch.Generator | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> DecodeResult:
        del state, num_steps, bucket, graphs
        generation = self.generate_tokens(prefix, generator=generator, cancelled=cancelled)
        return self.finalize_generation(generation)

    def generate_tokens(
        self,
        prefix: ActiveVLNPrefix,
        *,
        generator: torch.Generator | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> ActiveVLNGeneration:
        """Run the complete AR forward loop, retaining its original stop semantics."""
        runtime = getattr(self.policy, "_inference_graphs", None)
        if getattr(runtime, "action_trees", ()):
            from .speculation_activevln import generate_tree_tokens

            return generate_tree_tokens(self, prefix, runtime, cancelled=cancelled)
        memory = prefix.memory
        logits = prefix.next_logits
        tokens: list[torch.Tensor] = []
        logprobs: list[torch.Tensor] = []
        stop_reason = "max_tokens"
        for _ in range(self.policy.max_new_tokens):
            if cancelled is not None and cancelled():
                raise SessionCancelledError("ActiveVLN generation was cancelled")
            token, logprob = self._select(logits, memory, generator)
            tokens.append(token)
            logprobs.append(logprob)
            memory, logits = self.policy.append_token(memory, token)
            if int(token.item()) in self.policy.eos_token_ids:
                stop_reason = "eos"
                break
            # ``stop`` is an environment terminal action, not merely text.  End
            # the token loop as soon as the complete phrase is visible instead
            # of spending the remaining budget generating an irrelevant suffix.
            partial_text = self.policy.tokenizer.decode(
                torch.cat(tokens, dim=1)[0].tolist(), skip_special_tokens=True
            ).strip()
            partial = parse_r2r_actions(partial_text)
            if partial.valid and partial.actions[-1].name == "stop":
                stop_reason = "stop"
                break

        token_ids = torch.cat(tokens, dim=1)
        token_logprobs = torch.stack(logprobs, dim=1)
        return ActiveVLNGeneration(memory, token_ids, token_logprobs, stop_reason)

    def finalize_generation(self, generation: ActiveVLNGeneration) -> DecodeResult:
        """Parse a finished response into the public action chunk and trace."""
        memory = generation.memory
        token_ids = generation.token_ids
        token_logprobs = generation.token_logprobs
        action_mask = torch.ones_like(token_ids, dtype=torch.bool)
        text = self.policy.tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True).strip()
        parsed = parse_r2r_actions(text)
        actions, parsed_mask = actions_to_tensor(parsed)
        trace = DecodeTrace(
            token_ids=token_ids[0],
            token_logprobs=token_logprobs[0],
            action_mask=action_mask[0],
            text=text,
            parsed_actions=parsed,
            stop_reason=generation.stop_reason,
            meta={"parsed_action_mask": parsed_mask, "runner_profile": "official_eval_r2r"},
        )
        return DecodeResult(
            actions=actions[None].to(token_ids.device),
            behavior_logprob=token_logprobs,
            recompute_state=ARRecomputeState(token_ids, action_mask),
            next_memory=memory,
            traces=[trace],
        )

    def decode(
        self,
        state: torch.Tensor | None,
        prefix: ActiveVLNPrefix | ActiveVLNBranchPrefix,
        num_steps: int,
        bucket: int,
        graphs,
        *,
        generator: torch.Generator | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> DecodeResult:
        if isinstance(prefix, ActiveVLNPrefix):
            return self._decode_one(
                state,
                prefix,
                num_steps,
                bucket,
                graphs,
                generator=generator,
                cancelled=cancelled,
            )

        # Correctness-first ragged fallback.  It intentionally runs each branch
        # serially, then pads only the returned token records.  No padded-KV or
        # continuous-batching performance claim is made by this path.
        results = [
            self._decode_one(
                None,
                branch,
                num_steps,
                1,
                None,
                generator=generator,
                cancelled=cancelled,
            )
            for branch in prefix.branches
        ]
        max_tokens = max(result.recompute_state.token_ids.shape[1] for result in results)
        device = results[0].actions.device
        token_ids = torch.zeros(len(results), max_tokens, dtype=torch.long, device=device)
        action_mask = torch.zeros(len(results), max_tokens, dtype=torch.bool, device=device)
        logprobs = torch.zeros(
            len(results),
            max_tokens,
            dtype=results[0].behavior_logprob.dtype,
            device=device,
        )
        memories = []
        traces = []
        for row, result in enumerate(results):
            state_row = result.recompute_state
            length = state_row.token_ids.shape[1]
            token_ids[row, :length] = state_row.token_ids[0]
            action_mask[row, :length] = state_row.action_mask[0]
            logprobs[row, :length] = result.behavior_logprob[0].to(logprobs.dtype)
            memories.append(result.next_memory)
            traces.append(result.traces[0])
        return DecodeResult(
            actions=torch.cat([result.actions for result in results], dim=0),
            behavior_logprob=logprobs,
            recompute_state=ARRecomputeState(token_ids, action_mask),
            next_memory=BranchedActiveVLNMemory(tuple(memories)),
            traces=traces,
        )

    def sample_with_logprob(self, prefix, num_steps, sigma, generator=None):
        del sigma
        result = self.decode(None, prefix, num_steps, prefix.batch_size, None, generator=generator)
        return result.actions, result.behavior_logprob, result.recompute_state

    def recompute_logprob(self, prefix, recompute_state, num_steps, sigma):
        state = recompute_state
        if not isinstance(state, ARRecomputeState):
            raise TypeError("ActiveVLN recompute_state must be ARRecomputeState")
        if isinstance(prefix, ActiveVLNBranchPrefix):
            rows = []
            for index, branch in enumerate(prefix.branches):
                row_state = ARRecomputeState(
                    state.token_ids[index : index + 1],
                    state.action_mask[index : index + 1],
                )
                rows.append(self.recompute_logprob(branch, row_state, num_steps, sigma))
            return torch.cat(rows, dim=0)

        del num_steps, sigma

        memory = prefix.memory.fork()
        logits = prefix.next_logits
        logprobs = []
        for i in range(state.token_ids.shape[1]):
            if not bool(state.action_mask[0, i]):
                logprobs.append(torch.zeros(1, device=logits.device, dtype=logits.dtype))
                continue
            token = state.token_ids[:, i : i + 1]
            effective = self._distribution(logits, memory)
            selected = torch.log_softmax(effective, dim=-1).gather(1, token).squeeze(1)
            logprobs.append(selected)
            memory, logits = self.policy.append_token(memory, token)
        out = torch.stack(logprobs, dim=1)
        return out * state.action_mask.to(out.dtype)


class ActiveVLNPolicy(VLAPolicy):
    def __init__(
        self,
        qwen,
        processor: ActiveVLNProcessor,
        *,
        attention: str = "eager",
        max_new_tokens: int = 512,
        max_context: int = 32768,
        temperature: float = 0.2,
        top_p: float = 0.8,
        repetition_penalty: float = 1.05,
        do_sample: bool = True,
    ) -> None:
        config_dtype = str(getattr(qwen.config, "torch_dtype", "float32")).removeprefix("torch.")
        super().__init__(
            VLAPolicyConfig(
                name="activevln",
                action_dim=2,
                action_horizon=3,
                default_num_steps=1,
                dtype=config_dtype,
            )
        )
        self.qwen = qwen
        self._processor = processor
        self._attn = get_attention_backend(attention)
        self.max_new_tokens = max_new_tokens
        self.max_context = max_context
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.eos_token_ids = tuple(
            int(x)
            for x in (
                qwen.generation_config.eos_token_id
                if isinstance(qwen.generation_config.eos_token_id, list)
                else [qwen.generation_config.eos_token_id]
            )
        )
        self._decoder = _ActiveVLNDecoder(self)
        self._inference_runtime: ActiveVLNGraphRuntime | None = None

    @property
    def _inference_graphs(self) -> ActiveVLNGraphRuntime | None:
        # Differentiable rollout recompute retains the original eager path.
        return None if self.training or torch.is_grad_enabled() else self._inference_runtime

    def clear_cuda_graphs(self) -> None:
        """Release optional captures before moving/replacing model parameters."""
        self._inference_runtime = None

    @contextmanager
    def startup_cuda_graph_capture(
        self,
        *,
        query_bucket_size: int = 32,
        fused_ops: bool = False,
        split_attention: bool = False,
        tree_decode: bool = False,
        tree_fp32_projection: bool = False,
        tree_repeat_actions: int = 1,
        workspace_tokens: int | None = None,
    ) -> Iterator[None]:
        """Enable experimental inference graphs; capture is restricted to startup.

        Padding, fused arithmetic and tree shapes can alter BF16 behavior. These
        opt-in candidates require workload token/action parity validation.
        """
        from .cuda_graph import ActiveVLNGraphRuntime

        if self._inference_runtime is not None:
            raise RuntimeError("ActiveVLN graph startup has already completed")
        runtime = ActiveVLNGraphRuntime(
            self,
            query_bucket_size=query_bucket_size,
            fused_ops=fused_ops,
            split_attention=split_attention,
            tree_decode=tree_decode,
            tree_fp32_projection=tree_fp32_projection,
            tree_repeat_actions=tree_repeat_actions,
            workspace_tokens=workspace_tokens,
        )
        self._inference_runtime = runtime
        runtime.capture_enabled = True
        try:
            yield
        except BaseException:
            self._inference_runtime = None
            raise
        finally:
            runtime.capture_enabled = False

    def cuda_graph_stats(self) -> dict[str, object]:
        """Report actual optional graph coverage without claiming engine capture support."""
        if self._inference_runtime is None:
            return {"enabled": False}
        return {"enabled": True, **self._inference_runtime.stats()}

    def prewarm_cuda_graphs(
        self, observations: Iterable[Observation], *, context_buckets: Sequence[int]
    ) -> dict[str, object]:
        """Prewarm observed input shapes and explicit context buckets during startup.

        Each observation supplies initial/subsequent prompt lengths and a vision
        layout. Only shapes are retained; no response or recurrent state is
        generated. Unseen shapes after startup keep the counted eager fallback.
        """
        runtime = self._inference_graphs
        if runtime is None or not runtime.capture_enabled:
            raise RuntimeError("ActiveVLN shape prewarming requires active inference startup capture")
        initial, recurring = set(), set()
        count = 0
        with runtime.lock:
            for observation in observations:
                count += 1
                for first, lengths in ((True, initial), (False, recurring)):
                    turn = self._processor.process_turn(observation, initial=first)
                    lengths.add(int(turn.input_ids.shape[1]))
                    runtime.prewarm_vision(turn)
            shapes = runtime.prewarm_text(sorted(initial), sorted(recurring), context_buckets)
        return {
            "observations": count,
            "initial_query_lengths": sorted(initial),
            "recurring_query_lengths": sorted(recurring),
            "context_buckets": sorted(set(context_buckets)),
            "text_shapes": [list(shape) for shape in shapes],
        }

    def reset_cuda_graph_runtime_stats(self) -> None:
        """Exclude startup replays and workspace restores from measured counters."""
        if self._inference_runtime is not None:
            self._inference_runtime.reset_stats()

    @property
    def _text(self):
        model = self.qwen.model
        return model.language_model if hasattr(model, "language_model") else model

    @property
    def _visual(self):
        model = self.qwen.model
        return model.visual if hasattr(model, "visual") else self.qwen.visual

    @property
    def _lm_head(self):
        return self.qwen.lm_head

    @property
    def _text_config(self):
        return getattr(self.qwen.config, "text_config", self.qwen.config)

    @property
    def tokenizer(self):
        return self._processor.tokenizer

    @property
    def is_recurrent(self) -> bool:
        return True

    @property
    def decoder(self) -> _ActiveVLNDecoder:
        return self._decoder

    def collate(self, observations: list[Observation], request_ids: list[str]) -> ActiveVLNBatch:
        return self._processor.collate(observations, request_ids)

    def pad(self, batch: ActiveVLNBatch, target_batch_size: int) -> ActiveVLNBatch:
        if target_batch_size != 1:
            raise UnsupportedRecurrentModeError("ActiveVLN does not support padded batching")
        return batch

    def _embed_turn(self, turn: ProcessedTurn) -> torch.Tensor:
        embeds = self._text.embed_tokens(turn.input_ids)
        visual_dtype = next(self._visual.parameters()).dtype
        image_embeds = None
        runtime = self._inference_graphs
        if runtime is not None:
            image_embeds = runtime.vision(turn)
        if image_embeds is None:
            image_embeds = self._visual(turn.pixel_values.to(visual_dtype), grid_thw=turn.image_grid_thw)
        image_embeds = getattr(image_embeds, "pooler_output", image_embeds)
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = image_embeds[0]
        mask = turn.input_ids == self.qwen.config.image_token_id
        if int(mask.sum()) != image_embeds.shape[0]:
            raise ValueError(
                f"image features and image tokens differ: {image_embeds.shape[0]} vs {int(mask.sum())}"
            )
        return embeds.masked_scatter(
            mask.unsqueeze(-1).expand_as(embeds), image_embeds.to(embeds.device, embeds.dtype)
        )

    def _forward_chunk(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_kv: list[tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        text = self._text
        head_dim = text.layers[0].self_attn.head_dim
        sections = tuple(self._text_config.rope_scaling["mrope_section"])
        cos, sin = mrope_cos_sin(
            position_ids,
            head_dim,
            float(self._text_config.rope_theta),
            hidden.dtype,
        )
        prefix_len = 0 if prefix_kv is None else prefix_kv[0][0].shape[2]
        mask = rectangular_causal_mask(
            prefix_len,
            hidden.shape[1],
            device=hidden.device,
            dtype=hidden.dtype,
        )
        collected = []
        for i, layer in enumerate(text.layers):
            residual = hidden
            h = _rmsnorm(layer.input_layernorm, hidden)
            attn = layer.self_attn
            batch, query_len = h.shape[:2]
            q = attn.q_proj(h).view(batch, query_len, -1, head_dim).transpose(1, 2)
            k = attn.k_proj(h).view(batch, query_len, -1, head_dim).transpose(1, 2)
            v = attn.v_proj(h).view(batch, query_len, -1, head_dim).transpose(1, 2)
            q, k = apply_mrope(q, k, cos, sin, sections)
            collected.append((k, v))
            if prefix_kv is not None:
                pk, pv = prefix_kv[i]
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            out = self._attn.attend(q, k, v, attn_mask=mask, scaling=head_dim**-0.5)
            hidden = residual + attn.o_proj(out.transpose(1, 2).reshape(batch, query_len, -1))
            residual = hidden
            hidden = residual + _mlp(layer.mlp, _rmsnorm(layer.post_attention_layernorm, hidden))
        return _rmsnorm(text.norm, hidden), collected

    def encode_prefix(
        self,
        batch: ActiveVLNBatch,
        memory: ActiveVLNMemory | None = None,
    ) -> ActiveVLNPrefix:
        return self.encode_prepared_prefix(self.prepare_prefix(batch, memory))

    def prepare_prefix(
        self,
        batch: ActiveVLNBatch,
        memory: ActiveVLNMemory | None = None,
    ) -> PreparedActiveVLNTurn:
        """Process the observation and transfer inputs before the model-only interval."""
        observation = batch.observations[0]
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        turn = self._processor.process_turn(observation, initial=memory is None)
        offset = 0 if memory is None else memory.next_position
        positions, next_position = build_mrope_position_ids(
            turn.input_ids,
            turn.image_grid_thw,
            vision_start_token_id=int(self.qwen.config.vision_start_token_id),
            image_token_id=int(self.qwen.config.image_token_id),
            spatial_merge_size=int(self.qwen.config.vision_config.spatial_merge_size),
            offset=offset,
        )
        return PreparedActiveVLNTurn(turn.to(device, dtype), positions.to(device), memory, next_position)

    def encode_prepared_prefix(self, prepared: PreparedActiveVLNTurn) -> ActiveVLNPrefix:
        """Execute vision and text prefill without CPU observation preprocessing."""
        runtime = self._inference_graphs
        with runtime.lock if runtime is not None else nullcontext():
            return self._encode_prepared_prefix(prepared)

    def _encode_prepared_prefix(self, prepared: PreparedActiveVLNTurn) -> ActiveVLNPrefix:
        turn, positions, memory = prepared.turn, prepared.positions, prepared.memory
        device = turn.input_ids.device
        embeds = self._embed_turn(turn)
        working = (
            None
            if memory is None
            else memory.to(device).fork(extra_capacity=turn.input_ids.shape[-1] + self.max_new_tokens)
        )
        old_kv = None if working is None else working.visible_kv()
        result = None
        runtime = self._inference_graphs
        if runtime is not None:
            result = runtime.forward(embeds, positions, working)
        hidden, new_kv = self._forward_chunk(embeds, positions, old_kv) if result is None else result
        if working is None:
            working = ActiveVLNMemory.from_chunk(
                new_kv,
                turn.input_ids,
                turn.attention_mask,
                positions,
                max_length=self.max_context,
                prompt_hash=turn.prompt_sha256,
                next_position=prepared.next_position,
            )
        else:
            working.append_chunk(
                new_kv,
                turn.input_ids,
                turn.attention_mask,
                positions,
                prompt_hash=turn.prompt_sha256,
                next_position=prepared.next_position,
            )
        if result is not None:
            runtime.bind_memory(working)
        return ActiveVLNPrefix(working, self._lm_head(hidden[:, -1]))

    def append_token(
        self, memory: ActiveVLNMemory, token: torch.Tensor
    ) -> tuple[ActiveVLNMemory, torch.Tensor]:
        runtime = self._inference_graphs
        with runtime.lock if runtime is not None else nullcontext():
            return self._append_token(memory, token)

    def _append_token(
        self, memory: ActiveVLNMemory, token: torch.Tensor
    ) -> tuple[ActiveVLNMemory, torch.Tensor]:
        token = token.to(memory.token_ids_buffer.device)
        next_position = memory.next_position
        position = torch.full(
            (3, 1, 1),
            next_position,
            device=token.device,
            dtype=torch.long,
        )
        hidden = self._text.embed_tokens(token)
        result = None
        runtime = self._inference_graphs
        if runtime is not None:
            result = runtime.forward(hidden, position, memory)
        hidden, kv = self._forward_chunk(hidden, position, memory.visible_kv()) if result is None else result
        memory.append_chunk(kv, token, torch.ones_like(token), position, next_position=next_position + 1)
        if result is not None:
            runtime.bind_memory(memory)
        return memory, self._lm_head(hidden[:, -1])

    def full_logits(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Reference self-hosted full-sequence logits used by the parity harness."""
        del input_ids
        hidden, _ = self._forward_chunk(inputs_embeds, position_ids, None)
        return self._lm_head(hidden)


def _validate_qwen(qwen) -> None:
    cfg = qwen.config
    text = getattr(cfg, "text_config", cfg)
    if cfg.model_type != "qwen2_5_vl":
        raise ValueError(f"ActiveVLN needs qwen2_5_vl, got {cfg.model_type!r}")
    architectures = getattr(cfg, "architectures", [])
    if "Qwen2_5_VLForConditionalGeneration" not in architectures:
        raise ValueError(f"unexpected ActiveVLN architecture: {architectures}")
    sections = list(text.rope_scaling["mrope_section"])
    if sections != [16, 24, 24]:
        raise ValueError(f"unexpected ActiveVLN mRoPE sections: {sections}")


@register_policy("activevln")
def _build_activevln(
    checkpoint: str | None = None,
    *,
    revision: str = ACTIVEVLN_REVISION,
    allow_download: bool = False,
    attention: str = "eager",
    max_new_tokens: int = 512,
    max_context: int = 32768,
    temperature: float = 0.2,
    top_p: float = 0.8,
    repetition_penalty: float = 1.05,
    do_sample: bool = True,
    **overrides,
) -> VLAPolicy:
    if checkpoint is None:
        raise ValueError(
            "activevln needs a checkpoint, e.g. make_policy('activevln', checkpoint='/models/activevln')"
        )
    if overrides:
        raise ValueError(f"unknown ActiveVLN overrides: {sorted(overrides)}")
    path = Path(checkpoint)
    if not path.exists() and not allow_download:
        raise ValueError("activevln checkpoint must be a local snapshot unless allow_download=True")
    if not path.exists() and revision != ACTIVEVLN_REVISION:
        raise ValueError("remote ActiveVLN checkpoints require the pinned immutable revision")

    from transformers import Qwen2_5_VLForConditionalGeneration

    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        revision=revision,
        trust_remote_code=False,
        local_files_only=not allow_download,
        torch_dtype="auto",
    )
    _validate_qwen(qwen)
    processor = ActiveVLNProcessor(checkpoint, revision, allow_download=allow_download)
    return ActiveVLNPolicy(
        qwen,
        processor,
        attention=attention,
        max_new_tokens=max_new_tokens,
        max_context=max_context,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
    )
