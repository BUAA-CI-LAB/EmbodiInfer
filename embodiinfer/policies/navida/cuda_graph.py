"""Self-hosted StaticCache CUDA Graph decode runtime for NaViDA."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class CapturedNaViDADecodeGraph:
    graph: torch.cuda.CUDAGraph
    cache: object
    static_token: torch.Tensor
    static_attention_mask: torch.Tensor
    static_cache_position: torch.Tensor
    static_position_ids: torch.Tensor
    static_logits: torch.Tensor
    max_cache_len: int
    capture_ms: float
    replay_count: int = 0


class NaViDAGraphRuntime:
    def configure_cuda_graph(self, enabled: bool) -> bool:
        if enabled and self.tensor_parallel.enabled:
            raise ValueError("NaViDA tensor parallelism does not yet support CUDA Graph capture")
        self.cuda_graph_requested = enabled
        device = next(self.model.parameters()).device
        effective = enabled and device.type == "cuda"
        self.cuda_graph_enabled = effective
        return effective

    def manual_graph_stats(self) -> dict[str, object]:
        capture_ms = [entry.capture_ms for entry in self._navida_graph_cache.values()]
        entry_replays = [entry.replay_count for entry in self._navida_graph_cache.values()]
        return {
            "capture_count": self._navida_graph_captures,
            "replay_count": self._navida_graph_replays,
            "cache_entries": len(self._navida_graph_cache),
            "capture_ms": capture_ms,
            "entry_replays": entry_replays,
            "next_token": {
                "capture_count": 0,
                "replay_count": 0,
                "cache_entries": 0,
            },
            "navida_decode": {
                "capture_count": self._navida_graph_captures,
                "replay_count": self._navida_graph_replays,
                "cache_entries": len(self._navida_graph_cache),
                "capture_ms": capture_ms,
                "entry_replays": entry_replays,
            },
        }

    @staticmethod
    def _navida_cache_bucket(required: int, context_limit: int) -> int:
        bucket = 512
        while bucket < required:
            bucket *= 2
        if bucket > context_limit:
            raise ValueError(f"NaViDA generation needs {required} cache positions, above {context_limit}")
        return bucket

    def _capture_navida_decode_graph(
        self,
        *,
        batch_size: int,
        max_cache_len: int,
        position_axes: int,
        attention_dtype: torch.dtype,
    ) -> CapturedNaViDADecodeGraph:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        text_config = self.model.config.get_text_config(decoder=True)
        cache = self._static_cache_type(self.model.config, max_cache_len=max_cache_len)
        cache.early_initialization(
            batch_size=batch_size,
            num_heads=int(text_config.num_key_value_heads),
            head_dim=int(
                getattr(
                    text_config,
                    "head_dim",
                    text_config.hidden_size // text_config.num_attention_heads,
                )
            ),
            dtype=dtype,
            device=device,
        )
        pad_token_id = int(self.model.generation_config.pad_token_id)
        static_token = torch.full((batch_size, 1), pad_token_id, dtype=torch.long, device=device)
        static_attention_mask = torch.zeros(batch_size, max_cache_len, dtype=attention_dtype, device=device)
        static_attention_mask[:, 0] = 1
        static_cache_position = torch.zeros(1, dtype=torch.long, device=device)
        static_position_ids = torch.zeros(position_axes, batch_size, 1, dtype=torch.long, device=device)

        def decode_once() -> torch.Tensor:
            return self.model(
                input_ids=static_token,
                attention_mask=static_attention_mask,
                position_ids=static_position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=static_cache_position,
                pixel_values=None,
                image_grid_thw=None,
                logits_to_keep=1,
                return_dict=True,
            ).logits[:, -1]

        current_stream = torch.cuda.current_stream(device)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(3):
                cache.reset()
                decode_once()
        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize(device)
        cache.reset()
        graph = torch.cuda.CUDAGraph()
        started = time.perf_counter_ns()
        with torch.inference_mode(), torch.cuda.graph(graph):
            static_logits = decode_once()
        torch.cuda.synchronize(device)
        cache.reset()
        return CapturedNaViDADecodeGraph(
            graph=graph,
            cache=cache,
            static_token=static_token,
            static_attention_mask=static_attention_mask,
            static_cache_position=static_cache_position,
            static_position_ids=static_position_ids,
            static_logits=static_logits,
            max_cache_len=max_cache_len,
            capture_ms=(time.perf_counter_ns() - started) / 1e6,
        )

    def _navida_graph_entry(
        self,
        encoded: dict[str, torch.Tensor],
        position_axes: int,
    ) -> CapturedNaViDADecodeGraph:
        input_ids = encoded["input_ids"]
        batch_size, prompt_len = input_ids.shape
        text_config = self.model.config.get_text_config(decoder=True)
        context_limit = int(text_config.max_position_embeddings)
        max_cache_len = self._navida_cache_bucket(prompt_len + self.max_new_tokens, context_limit)
        device = input_ids.device
        key = (
            id(self.model),
            str(device),
            str(next(self.model.parameters()).dtype),
            str(getattr(self.model.config, "_attn_implementation", None)),
            batch_size,
            max_cache_len,
            position_axes,
        )
        entry = self._navida_graph_cache.get(key)
        if entry is None:
            if len(self._navida_graph_cache) >= 8:
                raise RuntimeError("NaViDA CUDA Graph cache is full (maximum 8 entries)")
            entry = self._capture_navida_decode_graph(
                batch_size=batch_size,
                max_cache_len=max_cache_len,
                position_axes=position_axes,
                attention_dtype=encoded["attention_mask"].dtype,
            )
            self._navida_graph_cache[key] = entry
            self._navida_graph_captures += 1
        return entry

    def _process_navida_scores(self, sequences: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        scores = self._navida_repetition(sequences, logits)
        scores = self._navida_temperature(sequences, scores)
        return self._navida_top_k(sequences, scores)

    def _navida_graph_generate(
        self,
        encoded: dict[str, torch.Tensor],
        *,
        generator: torch.Generator | None = None,
        return_scores: bool = False,
        prefill_complete: Callable[[], None] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate a full response; optionally mark the end of prefill without synchronizing."""
        if not encoded["input_ids"].is_cuda:
            raise RuntimeError("NaViDA manual CUDA Graph generation requires CUDA")
        attention_mask = encoded["attention_mask"]
        if not bool(attention_mask.detach().all().item()):
            raise ValueError("NaViDA CUDA Graph requires a homogeneous batch without padding")
        input_ids = encoded["input_ids"]
        batch_size, prompt_len = input_ids.shape
        qwen = self.model.model
        position_ids, rope_deltas = qwen.get_rope_index(
            input_ids,
            encoded["image_grid_thw"],
            encoded.get("video_grid_thw"),
            second_per_grid_ts=encoded.get("second_per_grid_ts"),
            attention_mask=attention_mask,
        )
        entry = self._navida_graph_entry(encoded, int(position_ids.shape[0]))
        eos_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos = torch.tensor(eos_ids, dtype=torch.long, device=input_ids.device)
        pad_token_id = int(self.model.generation_config.pad_token_id)
        if generator is not None and str(generator.device) != str(input_ids.device):
            generator = None

        with self._navida_graph_lock, torch.inference_mode():
            entry.cache.reset()
            entry.static_attention_mask.zero_()
            entry.static_attention_mask[:, :prompt_len].copy_(attention_mask)
            cache_position = torch.arange(prompt_len, dtype=torch.long, device=input_ids.device)
            prefill = self.model(
                **encoded,
                position_ids=position_ids,
                past_key_values=entry.cache,
                use_cache=True,
                cache_position=cache_position,
                logits_to_keep=1,
                return_dict=True,
            )
            if prefill_complete is not None:
                prefill_complete()
            logits = prefill.logits[:, -1]
            sequences = input_ids.clone()
            generated: list[torch.Tensor] = []
            saved_scores: list[torch.Tensor] = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
            for step in range(self.max_new_tokens):
                if return_scores:
                    saved_scores.append(logits.detach().float().clone())
                next_token = self._sample_navida_token(sequences, logits, generator=generator)
                next_token = torch.where(
                    finished[:, None],
                    torch.full_like(next_token, pad_token_id),
                    next_token,
                )
                generated.append(next_token)
                sequences = torch.cat((sequences, next_token), dim=1)
                finished |= torch.isin(next_token[:, 0], eos)
                if bool(finished.all().item()) or step + 1 == self.max_new_tokens:
                    break

                token_position = prompt_len + step
                entry.static_token.copy_(next_token)
                entry.static_cache_position.fill_(token_position)
                entry.static_attention_mask[:, token_position] = 1
                text_position = entry.static_cache_position.view(1, 1, 1).expand(1, batch_size, 1)
                mrope_position = entry.static_cache_position.view(1, 1, 1) + rope_deltas.reshape(
                    1, batch_size, 1
                )
                if entry.static_position_ids.shape[0] == 4:
                    position = torch.cat((text_position, mrope_position.expand(3, -1, -1)), dim=0)
                else:
                    position = mrope_position.expand(entry.static_position_ids.shape[0], -1, -1)
                entry.static_position_ids.copy_(position)
                entry.graph.replay()
                entry.replay_count += 1
                self._navida_graph_replays += 1
                logits = entry.static_logits.clone()
            return torch.cat(generated, dim=1), saved_scores
