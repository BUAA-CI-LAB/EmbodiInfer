"""The discrete action-token head for OpenVLA-OFT (RLinf variant).

Each of the ``action_dim * num_action_chunks`` action positions is an independent
categorical over the 256 action bins that overwrite the top of the Llama vocab
(``[vocab_size - n_action_bins, vocab_size)``). This replicates, op-for-op, RLinf's
``OpenVLAOFTForRLActionPrediction`` sampling / log-prob / detokenisation so an embodiinfer
rollout is bit-exact against RLinf's generator:

  * sample: mask non-bin logits -> ``/temperature`` -> optional top-k -> softmax ->
    ``multinomial``; greedy = argmax on the masked logits (no temperature).
  * behavior/recompute log-prob: ``-cross_entropy`` over the (temp-scaled, top-k,
    bin-masked) logits at the sampled token ids, token-level ``[B, n_tokens]``.
  * detokenise: ``vocab - id -> clip(.-1, 0, n_bins-2) -> bin_center -> q01/q99``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class CategoricalActionHead(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_action_bins: int,
        action_dim: int,
        num_action_chunks: int,
        q01: np.ndarray,
        q99: np.ndarray,
        mask: np.ndarray,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_action_bins = int(n_action_bins)
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.n_tokens = self.action_dim * self.num_action_chunks
        # bins/centres exactly as RLinf: np.linspace(-1, 1, n_action_bins), centres = midpoints
        bins = np.linspace(-1.0, 1.0, self.n_action_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0  # [n_action_bins - 1]
        self.register_buffer("bin_centers", torch.tensor(bin_centers, dtype=torch.float32), persistent=False)
        self.register_buffer("q01", torch.tensor(np.asarray(q01), dtype=torch.float32), persistent=False)
        self.register_buffer("q99", torch.tensor(np.asarray(q99), dtype=torch.float32), persistent=False)
        self.register_buffer("norm_mask", torch.tensor(np.asarray(mask), dtype=torch.bool), persistent=False)

    # ---- logit masking (only the 256 action bins are valid) -----------------
    def _mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        lo = self.vocab_size - self.n_action_bins
        logits[..., :lo] = -torch.inf
        logits[..., self.vocab_size :] = -torch.inf
        return logits

    @staticmethod
    def _top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k is None or top_k <= 0:
            return logits
        k = min(top_k, logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < kth, -torch.inf)

    # ---- sampling -----------------------------------------------------------
    def sample(
        self,
        logits: torch.Tensor,  # [B, n_tokens, vocab]
        do_sample: bool,
        temperature: float = 1.0,
        top_k: int = -1,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(idxs [B, n_tokens], behavior_logprob [B, n_tokens])`` (absolute token ids)."""
        logits = self._mask_logits(logits)
        if do_sample:
            proc = logits / temperature
            proc = self._top_k(proc, top_k)
            probs = F.softmax(proc, dim=-1)
            flat = probs.reshape(-1, probs.shape[-1])
            idxs = torch.multinomial(flat, num_samples=1, generator=generator).reshape(probs.shape[:-1])
        else:
            proc = logits
            idxs = proc.argmax(dim=-1)
        logprob = self._logprob_from_processed(proc, idxs)
        return idxs, logprob

    def recompute_logprob(
        self,
        logits: torch.Tensor,  # [B, n_tokens, vocab]
        idxs: torch.Tensor,  # [B, n_tokens]
        temperature: float = 1.0,
        top_k: int = -1,
    ) -> torch.Tensor:
        """Differentiable re-score of sampled tokens (matches RLinf ``default_forward``)."""
        proc = self._mask_logits(logits) / temperature
        proc = self._top_k(proc, top_k)
        return self._logprob_from_processed(proc, idxs)

    def _logprob_from_processed(self, proc: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        # RLinf re-masks the processed logits before cross-entropy; -inf/T is still -inf so
        # this is idempotent, but replicate it for exactness.
        action_logits = self._mask_logits(proc)
        v = action_logits.shape[-1]
        logprob = -F.cross_entropy(action_logits.reshape(-1, v), idxs.reshape(-1), reduction="none")
        return logprob.view(*idxs.shape).float()

    # ---- detokenise ---------------------------------------------------------
    @torch.no_grad()
    def tokens_to_actions(self, idxs: torch.Tensor) -> torch.Tensor:
        """Absolute token ids ``[B, n_tokens]`` -> unnormalised actions ``[B, num_chunks, action_dim]``."""
        discretized = self.vocab_size - idxs  # in (0, n_bins]
        discretized = torch.clamp(discretized - 1, 0, self.bin_centers.shape[0] - 1)
        normalized = self.bin_centers[discretized]  # [B, n_tokens], in [-1, 1]
        normalized = normalized.reshape(idxs.shape[0], self.num_action_chunks, self.action_dim)
        low, high = self.q01, self.q99  # [action_dim]
        unnorm = 0.5 * (normalized + 1.0) * (high - low + 1e-8) + low
        actions = torch.where(self.norm_mask, unnorm, normalized)
        return actions
