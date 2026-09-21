"""High-level entry point, analogous to ``vllm.LLM``.

    from embodiinfer import EmbodiInfer
    engine = EmbodiInfer(policy="mock_flow_vla", preset="small")
    chunk = engine.act(observation)          # single
    chunks = engine.act([obs1, obs2, ...])   # batched in one forward

For RL rollout, ``engine.backend`` exposes the GenerationBackend (logprob,
best-of-N, weight sync).
"""

from __future__ import annotations

from ...policies.base import VLAPolicy
from ...policies.factory import make_policy
from ...types import ActionChunk, Observation, validate_observation
from ..config import EngineConfig
from ..core import EngineCore
from ..rollout.generation_backend import GenerationBackend


class EmbodiInfer:
    """High-level entry point pairing one policy with the execution engine.

    Construction builds the policy from its factory name, or accepts an already-built
    one, and wraps it in :class:`~embodiinfer.engine.core.EngineCore`; the RL rollout surface is
    available as ``backend``. ``act`` runs a single observation or a whole list of them
    and returns the action chunk or chunks.

    This is the object most callers need. Serving frontends and notebooks drive ``act``;
    an RL trainer reaches through ``backend`` for log-probability, best-of-N, and weight
    synchronisation.

        from embodiinfer import EmbodiInfer

        engine = EmbodiInfer(policy="mock_flow_vla", preset="small")
        chunk = engine.act(observation)
        chunks = engine.act([obs1, obs2])
    """

    def __init__(
        self,
        policy: str | VLAPolicy = "mock_flow_vla",
        preset: str | None = None,
        engine_config: EngineConfig | None = None,
        **policy_kwargs,
    ):
        if isinstance(policy, str):
            # ``preset`` is a mock-scale knob, not universal: only forward it when
            # given, so checkpoint-backed policies (``EmbodiInfer("pi05", checkpoint=...)``)
            # are not handed an argument their builder does not accept.
            if preset is not None:
                policy_kwargs["preset"] = preset
            policy = make_policy(policy, **policy_kwargs)
        self.policy = policy
        self.core = EngineCore(policy, engine_config)
        self.backend = GenerationBackend(self.core)

    @property
    def config(self):
        return self.policy.config

    def act(
        self,
        observation: Observation | list[Observation],
        num_steps: int | None = None,
    ) -> ActionChunk | list[ActionChunk]:
        single = isinstance(observation, Observation)
        obs = [observation] if single else list(observation)
        for o in obs:
            validate_observation(o, self.policy.config)
        results = self.backend.generate(obs, num_steps=num_steps)
        return results[0] if single else results
