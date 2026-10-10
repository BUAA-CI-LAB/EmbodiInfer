# Python API

This reference covers the main exports of `embodiinfer`, followed by integration
patterns for checkpoint adapters, RL rollout, weight updates, and sessions.

The engine's primary interfaces are its command-line entry points and its HTTP and
WirelessComm serving contract, covered in [Serving](serving.md). This page is for callers
that embed the engine in Python.

## Engine entry points

::: embodiinfer
    options:
      members:
        - EmbodiInfer
        - EngineCore
        - AsyncEngine
        - EngineConfig
        - VLAPolicyConfig
        - preset_config

## Policies

::: embodiinfer
    options:
      members:
        - VLAPolicy
        - MockFlowVLA
        - make_policy
        - register_policy
        - available_policies

## Parallelism

::: embodiinfer
    options:
      members:
        - DataParallelEngine
        - InProcessReplica
        - RoundRobinDispatcher
        - LeastLoadedDispatcher
        - ThreadedExecutor

## RL rollout

::: embodiinfer
    options:
      members:
        - GenerationBackend
        - RolloutEngine
        - ToyReachEnv
        - RefitResult
        - WeightNameMap
        - refit_module
        - refit_state_dict
        - commit_refit
        - policy_version

## Data types

::: embodiinfer
    options:
      members:
        - Observation
        - ActionChunk
        - TrajectoryRecord
        - SampleParams

## Errors

::: embodiinfer
    options:
      members:
        - VvlaError
        - ReplicaExecutionError
        - PolicyNotFoundError
        - ObservationError

## Integration patterns

The following snippets illustrate integration contracts. Variables such as
`loaded_lerobot_policy`, `observation`, and learner weights come from the
calling application.

## LeRobot pi0.5 adapter

For an existing LeRobot Pi0.5 application, use the checkpoint's preprocessing and
postprocessing around the optional EmbodiInfer adapter:

```python
from embodiinfer.policies.pi05.lerobot_adapter import LeRobotPi05Adapter

policy = LeRobotPi05Adapter(loaded_lerobot_policy, attention="eager", cuda_graph=True)
normalized_actions = policy.predict_action_chunk(preprocessor(observation))
actions = postprocessor(normalized_actions)
```

This B=1 adapter enables EmbodiInfer-owned per-camera SigLIP and positional math,
Gemma eager attention and the complete denoising CUDA graph. Direct policy
construction can opt into the same embeddings with `native_embeddings=True`; the
existing embedding path remains the default. CUDA `EngineConfig(dtype="auto")`
preserves mixed checkpoint tensor dtypes and uses the policy's `execution_dtype` for
inputs and decoder state. Explicit dtypes still cast uniformly; CPU remains FP32.
See the [AGX comparison](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md#agx-orin-mixed-precision-2026-09-17)
for measured latency, output parity and the CPU-offload memory tradeoff.

## Pi05 fused operator configuration

```python
from embodiinfer import EngineConfig, make_policy
from embodiinfer.engine import EngineCore
from embodiinfer.policies.pi05 import Pi05OptimizationConfig

policy = make_policy(
    "pi05",
    checkpoint="/models/pi05",
    native_inference=True,
    prefix_cuda_graph=True,
    optimizations=Pi05OptimizationConfig(),
)
engine = EngineCore(
    policy,
    EngineConfig(device="cuda", dtype="auto", capture_full_loop=True),
)
```

This enables strict normalization/residual fusion and K/V workspace reuse with
the existing attention and MLP implementation. Select alternative attention,
paired GEMMs or calibrated per-layer precision through the configuration; see
[supported profiles and precision contracts](models.md#opt-in-fused-operators).
`Pi05OptimizationConfig.from_json(path)` loads a standalone recipe;
`config.to_json(path)` exports one. JSON must explicitly identify its
matching GELU/numerics contract. The existing demo accepts
`--optimizations /path/to/recipe.json`; use `--envs 1` for B1 profiles.
Full-checkpoint parity and task quality remain deployment gates, including for
the strict operator route; component parity alone does not establish them.

For the RLinf inference profile on Thor:

```python
import torch

torch.set_float32_matmul_precision("highest")
options = Pi05OptimizationConfig.from_json("configs/pi05/rlinf_libero_thor_fp8.json")
policy = make_policy(
    "pi05", checkpoint="/models/RLinf-Pi05-LIBERO-SFT",
    native_embeddings=True, native_inference=True, prefix_cuda_graph=True,
    low_cpu_mem_usage=True, action_horizon=10, default_num_steps=10,
    optimizations=options,
)
engine = EngineCore(
    policy, EngineConfig(device="cuda", dtype="auto", use_cuda_graph=True, capture_full_loop=True),
)
```

Use the Spark recipe on SM121. The `nvfp4_prefix` and
`nvfp4_prefix_fp8_action` recipe variants reproduce the experimental mixed formats.
`from_runtime_json(path)` loads a compact action-precision recipe and selects the
complete inference profile without changing scales. These are explicitly RLinf
inference semantics; they do not replace LeRobot's default or its rollout contract.

Select or extend an implementation through the shared operator layer:

```python
from embodiinfer.layers import OperatorBackends, normalization_backends

# MyRMSNorm implements NormalizationBackend and declares OperatorCapabilities.
normalization_backends.register("my_rmsnorm", MyRMSNorm)
config = Pi05OptimizationConfig(
    operators=OperatorBackends(normalization="my_rmsnorm", norm_quant=None),
)
```

The registries also expose `register_lazy(name, module, class_name)` and
`available()`. Direct reuse by another model uses
`normalization_backends.get(name, OperatorRequest(device, dtype))`, then creates
fixed-layout plans. The nested JSON `operators` mapping has the same field names
as `OperatorBackends`. See [prepared operator lifetimes](architecture.md#prepared-operator-interfaces).

## Generating RL rollouts

The rollout surface is available only for policies whose decoder implements
`RLDecoder`:

```python
backend = engine.backend
actions, logprob = backend.generate_with_logprob(obs_list, num_samples=1)
picked = backend.best_of_n(obs_list, num_samples=4, scorer=None)
result = backend.refit(new_state_dict, strict=True)
print(result.version)
```

## Zero-copy refit

Frameworks with their own zero-copy transport can update the live views and commit
the learner's version explicitly. Name aliases stay in the framework adapter:

```python
live_weights = engine.policy.refit_state_dict()
# Framework transport writes into live_weights[...] in place.
engine.policy.commit_refit(version=learner_step)
```

The zero-copy commit runs the policy's `on_refit` refresh hook before publishing the
new version. A hook failure leaves the previous version visible, but the
already-mutated live tensors are not rolled back; the caller must repair or discard
that policy before retrying.

## Copy-based refit

```python
# Copy-based integrations may map source names without teaching embodiinfer about the framework.
engine.policy.refit(actor_weights, name_map=actor_to_vvla_name, version=learner_step)
```

Copy-based refit validates names, shapes, the requested version, and exact tied
storage views before writing any weights. Two names for the same view must supply
equal values after conversion to the destination dtype; conflicting values raise
`ValueError` without changing weights or the policy version. Consistent ties are
copied once, while `strict=False` may update a tie through only one of its names.
This preserves existing ties; it does not create ties when loading a new model.

The preflight checks do not provide rollback after device-copy or `on_refit`
failures, detect arbitrary overlapping views, or join several partial calls into one
transaction. A framework using bucketed or zero-copy transport remains responsible
for validating the whole update and publishing its version only after all buckets
succeed.

## ActiveVLN sessions

ActiveVLN currently uses explicit single-session backend execution:

```python
from embodiinfer import EngineConfig, EmbodiInfer
from embodiinfer.types import SessionKey

engine = Vvla(
    "activevln",
    checkpoint="/models/activevln",
    engine_config=EngineConfig(
        max_batch_size=1,
        use_cuda_graph=False,
        capture_full_loop=False,
    ),
)
key = SessionKey(env_id="env-0", episode_id="episode-0")
chunk = engine.backend.generate([obs], session_ids=[key])[0]
engine.backend.reset_sessions([key])
```

The three 3B navigation profiles use the same explicit-session API; see
[supported models](models.md) for their serving notes.
