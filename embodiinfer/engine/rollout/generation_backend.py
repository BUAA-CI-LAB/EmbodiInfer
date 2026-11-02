"""Generation backend: the drop-in rollout surface for RL trainers.

A split-worker learner owns the environments, the RL algorithm, and the
optimizer; it delegates *action generation* to this object.
The contract is intentionally small:

    generate(obs)                 -> deterministic action chunks (fast serving path,
                                     CUDA-graph + prefix reuse via EngineCore)
    generate_with_logprob(obs)    -> stochastic action chunks + surrogate log-prob
                                     (the policy-gradient rollout path)
    best_of_n(obs, N, scorer)     -> planning: N candidates from one prefix, pick best
    refit(state_dict)             -> in-place weights after an optimizer step

Everything the trainer needs from the inference layer lives here; the trainer
itself is out of scope for embodiinfer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from ...exceptions import SessionCancelledError, SessionRequiredError, UnsupportedRecurrentModeError
from ...policies.decoder import RLDecoder
from ...types import ActionChunk, Observation, SessionKey
from ..core import EngineCore
from .refit import RefitResult, WeightNameMap
from .weight_sync import LocalWeightSync


@dataclass
class RolloutSamples:
    """A group-sampled rollout batch, carrying what an on-policy update needs.

    Produced by :meth:`GenerationBackend.sample_group`. ``recompute_state`` is the
    decoder-specific record a trainer re-scores under the current policy to get the
    differentiable ``log pi_theta`` (the visited-state trajectory for a flow decoder;
    the sampled action-token ids for OpenVLA-OFT's categorical decoder). Pass it back
    to :meth:`GRPOTrainer._logprob` / ``decoder.recompute_logprob``.

    ``behavior_logprob`` keeps the decoder's native granularity: ``[B, group_size]``
    for the flow surrogate (one scalar per candidate), ``[B, group_size, n_tokens]``
    for OFT's token-level categorical. Candidate ordering is contiguous per
    observation (row ``b * group_size + g`` = observation ``b``'s candidate ``g``).
    """

    observations: list[Observation]
    actions: torch.Tensor  # [B, group_size, H, A]
    behavior_logprob: torch.Tensor  # [B, group_size] (flow) or [B, group_size, n_tokens] (OFT)
    recompute_state: object  # decoder-specific trajectory or sampled-token record
    num_steps: int
    sigma: float
    group_size: int

    @property
    def batch_size(self) -> int:
        return self.actions.shape[0]


class GenerationBackend:
    """The rollout surface an RL trainer delegates action generation to.

    A split-worker learner owns the environments, the algorithm, and the optimizer; it
    calls this object only to generate actions. The contract is deliberately small:
    ``generate`` for the deterministic serving path (CUDA graph plus prefix reuse via
    :class:`~embodiinfer.engine.core.EngineCore`), ``generate_with_logprob`` and ``sample_group``
    for the stochastic policy-gradient path, ``best_of_n`` for planning, and ``refit`` for
    in-place weight synchronisation that keeps captured graphs valid.

    The stateless methods require the policy decoder to satisfy the generic
    policy-gradient contract; a policy that cannot support it is rejected explicitly rather
    than silently sampled from.
    """

    # Public capability disclosure.  Recurrent calls accept ragged environment
    # histories, but the correctness path schedules B=1 executions serially;
    # it is not padded-KV or continuous batching.
    recurrent_batching_mode = "serial_ragged"
    supports_true_ragged_batching = False

    def __init__(self, core: EngineCore, weight_sync: LocalWeightSync | None = None):
        self.core = core
        self.policy = core.policy
        self.device = core.device
        self.dtype = core.dtype
        self.pcfg = core.pcfg
        self.weight_sync = weight_sync or LocalWeightSync(self.policy)

    def _rl_decoder(self) -> RLDecoder:
        """Require the generic policy-gradient capability for stateless paths."""
        decoder = self.policy.decoder
        if not isinstance(decoder, RLDecoder):
            raise TypeError(
                f"{type(self.policy).__name__}'s decoder ({type(decoder).__name__}) is not an "
                "RLDecoder: it has no generic policy-gradient rollout surface"
            )
        return decoder

    # ---- deterministic serving path ----------------------------------------
    def generate(
        self,
        observations: list[Observation],
        num_steps: int | None = None,
        *,
        session_ids: Sequence[SessionKey] | None = None,
    ) -> list[ActionChunk]:
        if self.policy.is_recurrent:
            if session_ids is None or len(session_ids) != len(observations):
                raise SessionRequiredError(
                    "recurrent generation requires one explicit SessionKey per observation"
                )
            # Exact serial anchor for ragged recurrent histories.  Each call has
            # its own transaction and cancellation epoch, so one early stop or
            # cancellation cannot perturb another environment's token stream.
            outputs: list[ActionChunk] = []
            for index, (observation, session_id) in enumerate(zip(observations, session_ids)):
                batch = self.policy.collate([observation], [f"g{index}"])
                outputs.extend(self.core.execute(batch, num_steps, session_ids=[session_id]))
            return outputs
        batch = self.policy.collate(observations, [f"g{i}" for i in range(len(observations))])
        return self.core.execute(batch, num_steps, session_ids=session_ids)

    def reset_sessions(self, session_ids: Sequence[SessionKey]) -> None:
        self.core.reset_sessions(session_ids)

    def cancel_sessions(self, session_ids: Sequence[SessionKey]) -> None:
        self.core.cancel_sessions(session_ids)

    def _prep(self, observations: list[Observation]):
        batch = self.policy.collate(observations, [f"r{i}" for i in range(len(observations))])
        return batch.to(self.device, self.dtype)

    # ---- policy-gradient rollout path --------------------------------------
    @torch.no_grad()
    def generate_with_logprob(
        self,
        observations: list[Observation],
        num_steps: int | None = None,
        sigma: float = 0.1,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        *,
        session_ids: Sequence[SessionKey] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return actions [B,N,H,A] and decoder-native behavior log-probabilities.

        Prefix is encoded once for the batch and broadcast across the N
        best-of-N candidates — the compute-bound VLM backbone runs once, not N
        times. Scalar flow scores are [B,N]; token-level autoregressive scores
        remain [B,N,T].
        """
        if self.policy.is_recurrent:
            samples = self.sample_group(
                observations,
                num_samples,
                sigma=sigma,
                num_steps=num_steps,
                generator=generator,
                session_ids=session_ids,
            )
            return samples.actions, samples.behavior_logprob
        decoder = self._rl_decoder()
        num_steps = num_steps or self.core.config.num_steps or self.pcfg.default_num_steps
        batch = self._prep(observations)
        B = batch.batch_size
        prefix = self.policy.encode_prefix(batch).expand(num_samples)
        actions, logprob, _ = decoder.sample_with_logprob(prefix, num_steps, sigma, generator)
        H, A = self.pcfg.action_horizon, self.pcfg.action_dim
        return (
            actions.view(B, num_samples, H, A),
            logprob.reshape(B, num_samples, *logprob.shape[1:]),
        )

    @staticmethod
    def _candidate_scores(logprob: torch.Tensor) -> torch.Tensor:
        """Reduce decoder-native log-probs to one score per candidate.

        Scalar flow scores already have shape ``[B, N]``. Categorical decoders
        preserve one or more trailing token dimensions, which represent the
        joint candidate likelihood and are therefore summed before ranking.
        """
        if logprob.ndim < 2:
            raise ValueError(
                f"candidate log-probabilities must have shape [B, N, ...]; got {tuple(logprob.shape)}"
            )
        if logprob.ndim == 2:
            return logprob
        return logprob.reshape(*logprob.shape[:2], -1).sum(dim=-1)

    # ---- group sampling for policy-gradient RL (GRPO/PPO) -------------------
    @torch.no_grad()
    def sample_group(
        self,
        observations: list[Observation],
        group_size: int,
        sigma: float = 0.1,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
        *,
        session_ids: Sequence[SessionKey] | None = None,
    ) -> RolloutSamples:
        """Sample ``group_size`` candidates per observation, keeping the trajectory.

        This is the rollout surface a group-relative RL algorithm (GRPO) drives:
        one compute-bound prefix encode per observation is broadcast across all
        candidates (``prefix.expand``), and every observation's group runs in one
        batched denoise pass — the throughput lever embodiinfer exists for. The returned
        :class:`RolloutSamples` also carries the visited-state trajectory so the
        trainer can recompute a differentiable log-prob for the update.
        """
        if self.policy.is_recurrent:
            return self._sample_recurrent_group(
                observations,
                group_size,
                sigma,
                num_steps,
                generator,
                session_ids,
            )
        decoder = self._rl_decoder()
        num_steps = num_steps or self.core.config.num_steps or self.pcfg.default_num_steps
        batch = self._prep(observations)
        B = batch.batch_size
        prefix = self.policy.encode_prefix(batch).expand(group_size)
        actions, logprob, recompute_state = decoder.sample_with_logprob(prefix, num_steps, sigma, generator)
        H, A = self.pcfg.action_horizon, self.pcfg.action_dim
        # keep the decoder's log-prob granularity: [B*G] -> [B, G] (flow) or
        # [B*G, n_tokens] -> [B, G, n_tokens] (OFT categorical).
        return RolloutSamples(
            observations=list(observations),
            actions=actions.view(B, group_size, H, A),
            behavior_logprob=logprob.reshape(B, group_size, *logprob.shape[1:]),
            recompute_state=recompute_state,
            num_steps=num_steps,
            sigma=sigma,
            group_size=group_size,
        )

    def _sample_recurrent_group(
        self,
        observations: list[Observation],
        group_size: int,
        sigma: float,
        num_steps: int | None,
        generator: torch.Generator | None,
        session_ids: Sequence[SessionKey] | None,
    ) -> RolloutSamples:
        """Branch an episode start after one prefill, then decode isolated rows.

        L1 sharing is valid only at the common start.  Once actions/observations
        diverge, callers use :meth:`generate` with one observation and session
        key per branch (the serial-ragged path).
        """
        if len(observations) != 1:
            raise UnsupportedRecurrentModeError(
                "recurrent prefix sharing currently accepts one common start observation"
            )
        if group_size < 1:
            raise ValueError("group_size must be positive")
        if session_ids is None or len(session_ids) != group_size:
            raise SessionRequiredError(
                "recurrent sample_group requires one distinct SessionKey per rollout branch"
            )
        if len(set(session_ids)) != group_size:
            raise ValueError("recurrent rollout branches require distinct SessionKeys")

        leases = []
        try:
            leases = self.core.checkout_sessions(session_ids)
            if any(lease.memory is not None for lease in leases):
                raise UnsupportedRecurrentModeError(
                    "prefix sharing is only valid before rollout branches diverge"
                )
            steps = num_steps or self.core.config.num_steps or self.pcfg.default_num_steps
            batch = self.policy.collate(observations, ["rg0"]).to(self.device, self.dtype)
            shared = self.policy.encode_prefix(batch, None)
            expanded = shared.expand(group_size)
            branches = getattr(expanded, "branches", None)
            if group_size == 1 and branches is None:
                branches = (expanded,)
            if branches is None or len(branches) != group_size:
                raise UnsupportedRecurrentModeError(
                    "recurrent policy did not provide branch-safe prefix expansion"
                )

            results = []
            for branch, lease in zip(branches, leases):
                if lease.cancelled():
                    raise SessionCancelledError(f"rollout branch was cancelled: {lease.key!r}")
                result = self.policy.decoder.decode(
                    self.policy.decoder.init_state(1, generator),
                    branch,
                    steps,
                    1,
                    None,
                    generator=generator,
                    cancelled=lease.cancelled,
                )
                if result.next_memory is None:
                    raise RuntimeError("recurrent group branch did not return next_memory")
                results.append(result)

            max_tokens = max(result.behavior_logprob.shape[1] for result in results)
            logprob = torch.zeros(
                group_size,
                max_tokens,
                device=results[0].behavior_logprob.device,
                dtype=results[0].behavior_logprob.dtype,
            )
            first_state = results[0].recompute_state
            token_ids = torch.zeros(
                group_size,
                max_tokens,
                device=first_state.token_ids.device,
                dtype=first_state.token_ids.dtype,
            )
            action_mask = torch.zeros(
                group_size,
                max_tokens,
                device=first_state.action_mask.device,
                dtype=torch.bool,
            )
            for row, result in enumerate(results):
                length = result.behavior_logprob.shape[1]
                logprob[row, :length] = result.behavior_logprob[0]
                token_ids[row, :length] = result.recompute_state.token_ids[0]
                action_mask[row, :length] = result.recompute_state.action_mask[0]

            recompute_state = type(first_state)(token_ids, action_mask)
            self.core.commit_sessions(
                leases,
                [result.next_memory for result in results],
            )

            actions = torch.cat([result.actions for result in results], dim=0)
            return RolloutSamples(
                observations=list(observations),
                actions=actions.view(1, group_size, *actions.shape[1:]),
                behavior_logprob=logprob.view(1, group_size, max_tokens),
                recompute_state=recompute_state,
                num_steps=steps,
                sigma=sigma,
                group_size=group_size,
            )
        except BaseException:
            for lease in leases:
                lease.rollback()
            raise

    # ---- best-of-N planning -------------------------------------------------
    @torch.no_grad()
    def best_of_n(
        self,
        observations: list[Observation],
        num_samples: int,
        scorer: Callable[[list[Observation], torch.Tensor], torch.Tensor] | None = None,
        num_steps: int | None = None,
        sigma: float = 0.1,
    ) -> list[ActionChunk]:
        """Sample N candidates per obs from one prefix, keep the highest-scoring.

        ``scorer(observations, actions[B,N,H,A]) -> values[B,N]``. If ``None``,
        decoder-native token log-probs are summed to one score per candidate
        before choosing the maximum. In a real WAM (Cosmos), ``scorer`` is the
        learned future-state/value head.
        """
        if self.policy.is_recurrent:
            raise UnsupportedRecurrentModeError(
                "recurrent best_of_n requires committing the selected candidate memory"
            )
        actions, logprob = self.generate_with_logprob(
            observations, num_steps=num_steps, sigma=sigma, num_samples=num_samples
        )
        values = (
            scorer(observations, actions) if scorer is not None else self._candidate_scores(logprob)
        )  # [B, N]
        expected_shape = (len(observations), num_samples)
        if values.shape != expected_shape:
            raise ValueError(
                f"best-of-N scorer must return shape {expected_shape}; got {tuple(values.shape)}"
            )
        best = values.argmax(dim=1)  # [B]
        out = []
        for i in range(len(observations)):
            j = int(best[i])
            out.append(
                ActionChunk(
                    request_id=f"bon{i}",
                    actions=actions[i, j].float().cpu(),
                    logprob=logprob[i, j].detach().float().cpu(),
                    value=float(values[i, j]),
                    meta={"num_samples": num_samples},
                )
            )
        return out

    # ---- weight sync --------------------------------------------------------
    def refit(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        *,
        name_map: WeightNameMap | None = None,
        version: int | None = None,
    ) -> RefitResult:
        """Install learner weights through EmbodiInfer's framework-neutral refit API."""
        if self.policy.is_recurrent and self.core.has_session_state():
            raise UnsupportedRecurrentModeError(
                "reset recurrent sessions before updating weights; cached KV belongs to the old policy version"
            )
        return self.weight_sync.update(
            state_dict,
            strict=strict,
            name_map=name_map,
            version=version,
        )

    def update_weights(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        *,
        name_map: WeightNameMap | None = None,
        version: int | None = None,
    ) -> RefitResult:
        """Backward-compatible alias for :meth:`refit`."""
        return self.refit(
            state_dict,
            strict=strict,
            name_map=name_map,
            version=version,
        )
