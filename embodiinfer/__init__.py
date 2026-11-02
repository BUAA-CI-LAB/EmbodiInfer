"""EmbodiInfer — a vLLM-class inference & RL-rollout engine for flow VLA models.

Quick start:
    from embodiinfer import EmbodiInfer
    engine = EmbodiInfer(policy="mock_flow_vla", preset="small")
    chunk = engine.act(observation)
"""

from .engine import (
    AsyncEngine,
    DataParallelEngine,
    EngineCore,
    InProcessReplica,
    LeastLoadedDispatcher,
    RoundRobinDispatcher,
    ThreadedExecutor,
)
from .engine.config import EngineConfig
from .engine.rollout import GenerationBackend
from .engine.rollout.demo import (
    RolloutEngine,
    ToyReachEnv,
)  # demo/toy scaffolding, re-exported for convenience
from .engine.rollout.refit import (
    RefitResult,
    WeightNameMap,
    commit_refit,
    policy_version,
    refit_module,
    refit_state_dict,
)
from .engine.serve import EmbodiInfer
from .exceptions import (
    EmbodiInferError,
    ObservationError,
    PolicyNotFoundError,
    ReplicaExecutionError,
)
from .policies import (
    MockFlowVLA,
    VLAPolicy,
    available_policies,
    make_policy,
    register_policy,
)
from .policies.config import VLAPolicyConfig
from .policies.mock import preset_config
from .types import ActionChunk, Observation, SampleParams, TrajectoryRecord

__version__ = "0.1.0"
__all__ = [
    "EmbodiInfer",
    "EngineCore",
    "AsyncEngine",
    "DataParallelEngine",
    "InProcessReplica",
    "RoundRobinDispatcher",
    "LeastLoadedDispatcher",
    "ThreadedExecutor",
    "EmbodiInferError",
    "ReplicaExecutionError",
    "PolicyNotFoundError",
    "ObservationError",
    "EngineConfig",
    "VLAPolicyConfig",
    "preset_config",
    "VLAPolicy",
    "MockFlowVLA",
    "make_policy",
    "register_policy",
    "available_policies",
    "RefitResult",
    "WeightNameMap",
    "refit_module",
    "refit_state_dict",
    "commit_refit",
    "policy_version",
    "GenerationBackend",
    "RolloutEngine",
    "ToyReachEnv",
    "Observation",
    "ActionChunk",
    "TrajectoryRecord",
    "SampleParams",
]
