# Python API

本页列出 `embodiinfer` 的主要公开接口，并给出 LeRobot 适配、RL rollout、权重更新和会话管理示例，
供 Python 应用直接集成引擎。通过网络调用模型时，请参阅[部署推理服务](serving.md)。

## 引擎与配置

::: embodiinfer
    options:
      members:
        - EmbodiInfer
        - EngineCore
        - AsyncEngine
        - EngineConfig
        - VLAPolicyConfig
        - preset_config

## 策略

::: embodiinfer
    options:
      members:
        - VLAPolicy
        - MockFlowVLA
        - make_policy
        - register_policy
        - available_policies

## 并行

::: embodiinfer
    options:
      members:
        - DataParallelEngine
        - InProcessReplica
        - RoundRobinDispatcher
        - LeastLoadedDispatcher
        - ThreadedExecutor

## RL rollout {#rl-rollout}

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

## 数据类型

::: embodiinfer
    options:
      members:
        - Observation
        - ActionChunk
        - TrajectoryRecord
        - SampleParams

## 异常类型

::: embodiinfer
    options:
      members:
        - VvlaError
        - ReplicaExecutionError
        - PolicyNotFoundError
        - ObservationError

## 接口用法

以下示例中的 `loaded_lerobot_policy`、`observation` 和训练端权重等变量由应用提供。

## LeRobot pi0.5 适配器 {#lerobot-pi05-adapter}

已有 LeRobot Pi0.5 应用可使用此适配器接入 EmbodiInfer，并沿用检查点的预处理和后处理：

```python
from embodiinfer.policies.pi05.lerobot_adapter import LeRobotPi05Adapter

policy = LeRobotPi05Adapter(loaded_lerobot_policy, attention="eager", cuda_graph=True)
normalized_actions = policy.predict_action_chunk(preprocessor(observation))
actions = postprocessor(normalized_actions)
```

该适配器以 B=1 运行，启用 EmbodiInfer 的逐相机 SigLIP、位置运算、Gemma eager attention
和完整去噪循环 CUDA graph。直接创建策略时，可用 `native_embeddings=True` 选择相同的嵌入实现，
默认仍使用原有实现。CUDA 上的 `EngineConfig(dtype="auto")` 保留检查点中各张量原有的数据类型，
输入和解码器状态使用策略的 `execution_dtype`。显式指定 dtype 时统一转换，CPU 使用 FP32。
参见 [AGX 对比](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/README.md#agx-orin-mixed-precision-2026-09-17)
了解延迟、输出一致性，以及将权重卸载到 CPU 后的内存与性能取舍。

## 生成 RL rollout {#generating-rl-rollouts}

rollout 接口仅适用于解码器实现了
`RLDecoder` 的策略：

```python
backend = engine.backend
actions, logprob = backend.generate_with_logprob(obs_list, num_samples=1)
picked = backend.best_of_n(obs_list, num_samples=4, scorer=None)
result = backend.refit(new_state_dict, strict=True)
print(result.version)
```

## 零拷贝权重更新 {#zero-copy-refit}

训练框架若已有零拷贝传输机制，可直接写入当前权重视图，再提交训练端的版本号。
参数名映射由框架适配器维护：

```python
live_weights = engine.policy.refit_state_dict()
# 框架传输会就地写入 live_weights[...]。
engine.policy.commit_refit(version=learner_step)
```

提交时先调用策略的 `on_refit` 刷新运行状态，成功后才发布新版本。
若刷新失败，版本号不变，但已写入的张量不会回滚；需修复或重新创建策略后再重试。

## 复制权重更新

```python
# 基于拷贝的集成可以映射源名称，而无需让 embodiinfer 了解该框架。
engine.policy.refit(actor_weights, name_map=actor_to_vvla_name, version=learner_step)
```

复制前会检查参数名、形状、目标版本，以及完全相同的共享存储视图。
多个参数名若指向同一视图，转换为目标 dtype 后的值必须一致；冲突时抛出 `ValueError`，不修改权重或版本。
共享视图只复制一次。使用 `strict=False` 时，可通过其中一个参数名更新共享权重。
此操作保留已有的权重共享关系，不为新模型建立共享关系。

预检查不检测任意部分重叠的视图，也不将多次局部更新合并为一个事务。
设备复制或 `on_refit` 失败后，已写入内容不会自动回滚。
使用分桶或零拷贝传输的框架需校验完整更新，等全部分桶成功后再发布版本。

## ActiveVLN 会话 {#activevln-sessions}

ActiveVLN 通过后端逐个执行会话，调用方需显式传入会话标识：

```python
from embodiinfer import EngineConfig, Vvla
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

三个 3B 导航模型使用相同的会话 API，具体要求见
[支持的模型](models.md)。
