"""Stock Hugging Face eager generation and runner dispatch for NaViDA."""

from __future__ import annotations

import torch
import torch.distributed as dist

from ...types import Observation
from .contract import NaViDAMemory


def navida_generation_kwargs(max_new_tokens: int) -> dict[str, object]:
    return {
        "do_sample": True,
        "temperature": 0.2,
        "top_k": 50,
        "top_p": 1.0,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.05,
        "num_return_sequences": 1,
        "use_cache": True,
    }


class _TensorParallelSamplingProcessor:
    """Sample once on TP rank zero and force every rank to emit that token."""

    def __init__(self, runner: NaViDAGenerationRuntime) -> None:
        self.runner = runner

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        token = self.runner._sample_navida_token(input_ids, scores)
        forced = torch.full_like(scores, torch.finfo(scores.dtype).min)
        return forced.scatter(1, token, 0)


class NaViDAGenerationRuntime:
    def _sample_navida_token(
        self,
        sequences: torch.Tensor,
        logits: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        scores = self._process_navida_scores(sequences, logits)
        probabilities = torch.nn.functional.softmax(scores, dim=-1)
        context = self.tensor_parallel
        if not context.enabled or context.rank == 0:
            token = torch.multinomial(probabilities, 1, generator=generator)
        else:
            token = torch.empty(probabilities.shape[0], 1, dtype=torch.long, device=probabilities.device)
        if context.enabled:
            source = 0 if context.process_group is None else dist.get_global_rank(context.process_group, 0)
            dist.broadcast(token, src=source, group=context.process_group)
        return token

    def infer_batch(
        self, observations: list[Observation], memories: list[NaViDAMemory]
    ) -> list[tuple[str, torch.Tensor, list[float]]]:
        if len(observations) != len(memories) or not observations:
            raise ValueError("observations and memories must be non-empty and aligned")
        encoded = self._encode_batch(observations, memories)
        with torch.inference_mode():
            if self.cuda_graph_enabled:
                tokens, _ = self._navida_graph_generate(encoded)
            elif self.tensor_parallel.enabled:
                from transformers import LogitsProcessorList

                sequences = self.model.generate(
                    **encoded,
                    # The custom processor performs the one stochastic draw on
                    # TP rank zero; generate then selects its forced token.
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=1,
                    use_cache=True,
                    logits_processor=LogitsProcessorList([_TensorParallelSamplingProcessor(self)]),
                    use_model_defaults=True,
                )
                tokens = sequences[:, encoded["input_ids"].shape[1] :]
            else:
                sequences = self.model.generate(
                    **encoded,
                    **navida_generation_kwargs(self.max_new_tokens),
                    use_model_defaults=True,
                )
                tokens = sequences[:, encoded["input_ids"].shape[1] :]
        decoded = self.processor.batch_decode(tokens, skip_special_tokens=True)
        return [(text.strip(), tokens[row], []) for row, text in enumerate(decoded)]

    def infer(self, observation: Observation, memory: NaViDAMemory):
        return self.infer_batch([observation], [memory])[0]
