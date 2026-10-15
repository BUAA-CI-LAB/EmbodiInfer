"""A thin GRPO trainer over embodiinfer's rollout surface.

GRPO (Group Relative Policy Optimization) drops the value critic: for each
observation it samples a group of ``G`` candidates, scores them, and uses the
*group-relative* reward as the advantage —

    A_i = (r_i - mean_g r) / (std_g r + eps)

then takes a PPO-style clipped policy-gradient step (optionally regularized by a
KL to a frozen reference policy). This maps cleanly onto embodiinfer's rollout surface:
group sampling is :meth:`GenerationBackend.sample_group` (one prefix encode
broadcast to all candidates), and the policy gradient needs the differentiable
log-prob recompute (:func:`~embodiinfer.engine.rollout.logprob.flow_logprob_recompute`).

This trainer is intentionally a *thin demo layer*: embodiinfer is the rollout /
generation backend, and the RL algorithm sits on top. It exists to (a) prove the
closed loop learns and (b) let the generation path's throughput be measured
inside a real update loop. It is not a production RL framework.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

import torch

from ....types import SessionKey
from ..generation_backend import GenerationBackend, RolloutSamples
from .reward import Reward


@dataclass
class GRPOConfig:
    group_size: int = 8  # G candidates sampled per observation
    lr: float = 1e-4
    clip_eps: float = 0.2  # PPO ratio clip
    kl_coef: float = 0.0  # KL-to-reference penalty; 0 disables (no reference forward)
    # Optimizer steps reusing one rollout batch. Keep at 1: the flow surrogate
    # log-prob has tiny per-step variance (sigma^2 * dt), so one update moves the
    # recomputed log-prob far enough that exp(new - behavior) underflows to 0
    # (and its gradient with it). >1 needs ratio/log-prob clamping to be stable.
    num_inner_epochs: int = 1
    sigma: float = 0.1  # SDE sampling temperature
    num_steps: int | None = None  # denoise steps; None -> policy/engine default
    adv_eps: float = 1e-6  # group-std floor for advantage normalization
    max_grad_norm: float | None = 1.0


class GRPOTrainer:
    """Group Relative Policy Optimization on a :class:`GenerationBackend`.

    The backend's policy is trained in place; after each update the fresh weights
    are pushed back through the backend's weight-sync surface (an identity copy
    in single-process, the real broadcast in a split-worker deployment).
    """

    def __init__(
        self,
        backend: GenerationBackend,
        reward: Reward,
        config: GRPOConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        self.backend = backend
        self.policy = backend.policy
        self.reward = reward
        self.cfg = config or GRPOConfig()
        self.device = backend.device
        self.dtype = backend.dtype
        self.opt = optimizer or torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self._rollout_ids = itertools.count()
        # A frozen reference for the optional KL penalty (GRPO's regularizer).
        self._ref = None
        if self.cfg.kl_coef > 0:
            self._ref = copy.deepcopy(self.policy).eval()
            for p in self._ref.parameters():
                p.requires_grad_(False)

    # ---- one GRPO iteration -------------------------------------------------
    def step(self, observations: list) -> dict:
        """Run one rollout + update and return scalar stats for logging."""
        cfg = self.cfg
        num_steps = cfg.num_steps or self.backend.core.config.num_steps or self.backend.pcfg.default_num_steps

        # 1) rollout: group sampling under no_grad — embodiinfer's throughput surface.
        session_ids = self._session_ids(observations, cfg.group_size)
        try:
            samples = self.backend.sample_group(
                observations,
                cfg.group_size,
                sigma=cfg.sigma,
                num_steps=num_steps,
                session_ids=session_ids,
            )
        finally:
            if session_ids is not None:
                self.backend.reset_sessions(session_ids)
        B, G = samples.batch_size, cfg.group_size

        # 2) reward -> group-relative advantage (detached; the policy gradient
        #    flows only through the recomputed log-prob).
        reward = self.reward(observations, samples.actions).to(torch.float32)  # [B, G] (cpu)
        advantage = self._advantage(reward).to(self.device)  # [B, G]
        behavior_logp = self._candidate_logprob(samples.behavior_logprob.to(self.device), B, G)
        trajectory = samples.recompute_state.to(self.device)  # flow: [B*G, N+1, H, A]

        ref_logp = None
        if self._ref is not None:
            ref_logp = self._candidate_logprob(
                self._logprob(self._ref, observations, trajectory, G, num_steps, cfg.sigma),
                B,
                G,
            )

        # 3) clipped policy-gradient update(s) on this rollout batch.
        was_training = self.policy.training
        self.policy.train()
        stats: dict[str, float] = {}
        for _ in range(cfg.num_inner_epochs):
            new_logp = self._candidate_logprob(
                self._logprob(self.policy, observations, trajectory, G, num_steps, cfg.sigma),
                B,
                G,
            )
            ratio = torch.exp(new_logp - behavior_logp)  # [B, G]; == 1 at the first inner step
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantage
            pg_loss = -torch.minimum(unclipped, clipped).mean()
            loss = pg_loss
            if ref_logp is not None:
                # k3 KL estimator (unbiased, non-negative): E[exp(r) - r - 1], r = ref - new.
                r = ref_logp - new_logp
                kl = (torch.exp(r) - r - 1.0).mean()
                loss = loss + cfg.kl_coef * kl
                stats["kl"] = float(kl.detach())
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.max_grad_norm is not None:
                gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                stats["grad_norm"] = float(gn)
            self.opt.step()
        if not was_training:
            self.policy.eval()

        # 4) push updated weights back into the rollout replica (the RL surface).
        self.backend.update_weights(self.policy.state_dict())

        stats.update(
            reward_mean=float(reward.mean()),
            reward_max=float(reward.max()),
            advantage_std=float(advantage.std()),
            ratio_mean=float(ratio.mean().detach()),
            pg_loss=float(pg_loss.detach()),
            loss=float(loss.detach()),
        )
        return stats

    # ---- helpers ------------------------------------------------------------
    def _advantage(self, reward: torch.Tensor) -> torch.Tensor:
        mean = reward.mean(dim=1, keepdim=True)
        std = reward.std(dim=1, keepdim=True)
        return (reward - mean) / (std + self.cfg.adv_eps)

    @staticmethod
    def _candidate_logprob(logprob: torch.Tensor, batch_size: int, group_size: int) -> torch.Tensor:
        """Reduce decoder-native scalar or action-token logprob per candidate."""
        if logprob.shape[:2] == (batch_size, group_size):
            shaped = logprob
        elif logprob.shape[0] == batch_size * group_size:
            shaped = logprob.reshape(batch_size, group_size, *logprob.shape[1:])
        else:
            raise ValueError(
                "decoder logprob does not align with the rollout group: "
                f"shape={tuple(logprob.shape)} B={batch_size} G={group_size}"
            )
        if shaped.ndim == 2:
            return shaped
        return shaped.flatten(start_dim=2).sum(dim=-1)

    def _session_ids(self, observations: list, group_size: int) -> list[SessionKey] | None:
        if not self.policy.is_recurrent:
            return None
        if len(observations) != 1:
            raise ValueError("recurrent GRPO currently supports one common episode start")
        rollout = next(self._rollout_ids)
        env_id = observations[0].env_id if observations[0].env_id is not None else "grpo"
        return [SessionKey(env_id, f"grpo-{rollout}", branch) for branch in range(group_size)]

    def _logprob(self, policy, observations, trajectory, group_size, num_steps, sigma) -> torch.Tensor:
        """Recompute the flow log-prob of ``trajectory`` under ``policy``.

        Re-encodes the prefix from the observations (so the gradient can reach
        the backbone when ``policy`` is trainable) and expands it to match the
        per-candidate trajectory layout. Runs under no_grad only for the frozen
        reference policy.
        """
        grad = policy is self.policy
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            batch = policy.collate(observations, [f"grpo{i}" for i in range(len(observations))])
            batch = batch.to(self.device, self.dtype)
            prefix = policy.encode_prefix(batch).expand(group_size)
            return policy.decoder.recompute_logprob(prefix, trajectory, num_steps, sigma)

    def sample_group(self, observations: list) -> RolloutSamples:
        """Expose the backend's group sampler (used by the throughput panel)."""
        cfg = self.cfg
        session_ids = self._session_ids(observations, cfg.group_size)
        try:
            return self.backend.sample_group(
                observations,
                cfg.group_size,
                sigma=cfg.sigma,
                num_steps=cfg.num_steps,
                session_ids=session_ids,
            )
        finally:
            if session_ids is not None:
                self.backend.reset_sessions(session_ids)
