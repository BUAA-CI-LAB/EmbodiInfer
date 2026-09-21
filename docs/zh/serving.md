# 部署推理服务

EmbodiInfer 为不同模型提供统一的 HTTP 推理接口，也支持可选的 WirelessComm 传输。
HTTP 与 WirelessComm 策略服务采用相同的会话管理、请求顺序和幂等机制。

## 启动 HTTP 服务

先安装对应的[模型依赖](installation.md)，准备检查点。使用 π0.5 时，还需创建适配器 JSON，
将请求中的状态和相机字段映射到检查点输入，见[适配器配置](#adapter-json)。

在该环境中启动 `embodiinfer-http-serve`（或 `embodiinfer-serve`）：

```bash
uv run --no-sync embodiinfer-http-serve \
  --policy pi05 \
  --checkpoint /models/pi05-checkpoint \
  --adapter-config ./adapter-config.json \
  --host 127.0.0.1 --port 8000
```

将路径替换为实际的检查点和适配器文件，然后在另一终端检查服务：

```bash
curl --fail http://127.0.0.1:8000/healthz
curl --fail http://127.0.0.1:8000/v1/capabilities
```

`healthz` 返回服务状态，`capabilities` 描述已加载适配器支持的功能。
确认服务就绪后，发送一次观测检查相机和状态映射。
上述命令只监听本机；Control 位于其他节点时，需改为该节点可访问的监听地址，并配置网络访问权限。

## 发送观测并获取动作 {#send-your-first-observation}

保持服务运行，准备 JPEG/PNG 相机图像和 `state.json`，字段名与适配器配置一致。
下方[双相机示例](#adapter-json)使用 `{"observation.state": [...]}` 格式；
数组填写完整状态向量，顺序和单位需与检查点一致。

```bash
uv run --no-sync python examples/http_client.py \
  --endpoint http://127.0.0.1:8000 \
  --state ./state.json \
  --image observation.images.front=./front.jpg \
  --image observation.images.wrist=./wrist.jpg \
  --instruction "Pick up the cube and place it in the bowl."
```

[客户端示例](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/examples/http_client.py)
会读取 capabilities、打开会话，以 multipart 格式发送 `step_id: 0` 的请求并携带幂等头，
打印 JSON 响应并关闭会话。响应字段包括 `schema: embodiinfer.policy.step.result.v1`、`step_id: 0`、
`session_id`、`action_space`、`actions` 和 `timing`；动作维度取决于检查点。

对于需要认证的服务，先在本地设置变量再添加 `--token-env EMBODIINFER_TOKEN`，
服务端使用同一个 token。如果 HTTP 请求失败，请查看服务日志和响应状态码：
401 表示认证问题，400 表示请求格式错误，409 表示会话顺序或幂等冲突，422 表示 adapter 输入错误。
该示例不会自动重试请求。

这个示例只发送一次观测。有状态应用应复用同一会话，
每完成一步推理后递增 `step_id`，再发送下一次观测。

## 启动参数

两种传输共用的模型与引擎参数：

| 参数 | 含义 |
|---|---|
| `--policy` | 策略名，默认 `pi05`。 |
| `--checkpoint` | 检查点路径或 hub id。 |
| `--adapter-config` | 适配器配置 JSON 文件。 |
| `--device` | 执行设备，默认 `cuda`。 |
| `--dtype` | 计算数据类型，默认 `auto`。 |
| `--max-batch` | 每批最大请求数，默认 `1`。π0.5 支持跨会话批处理。 |
| `--max-wait-ms` | 请求收集窗口，单位毫秒，默认 `5`。 |
| `--num-steps` | 覆盖解码步数。 |
| `--no-cuda-graph` | 关闭 CUDA graph 捕获。 |
| `--capture-full-loop` | 捕获整个解码循环，而不是逐步捕获。 |

HTTP 专用参数包括 `--host`（默认 `0.0.0.0`）、`--port`（默认 `8000`）、
`--token`（可选 Bearer token）、`--max-body-bytes`、`--max-image-bytes`、
`--max-images`、`--max-sessions` 和 `--idempotency-cache-size`。

## 多客户端批处理 {#share-one-service-across-clients}

对于三个独立的 π0.5 会话，在任一服务命令上添加 `--max-batch 3 --max-wait-ms 5`。
在收集窗口内就绪的请求会合并为一个张量批次执行；达到批次上限或窗口到期后开始执行，
无需等齐所有客户端。该窗口限制请求收集时间，不包含等待模型空闲的排队时间。

批处理不改变各会话的步序和幂等缓存。结果分别返回原会话，动作还原使用各请求自己的状态。
某条观测格式错误只影响对应请求，同批次中的有效请求仍会执行。
待处理队列最多容纳 1,024 个请求，满时返回
`429 queue_full`。

默认的 `--max-batch 1` 直接执行单个请求。适配器支持批处理时，才能调大该值；否则启动时会报错。
有状态适配器仍按单会话执行。CUDA graph 按批次大小分桶缓存，最大不超过配置值；
例如 `--max-batch 3` 在批次满时使用 B=3 的 graph。

## 启动 WirelessComm 服务

要通过静态配置的 WirelessComm 节点提供服务：

```bash
embodiinfer-wireless-serve \
  --policy pi05 \
  --checkpoint <pi05-checkpoint> \
  --adapter-config <adapter-config.json> \
  --comm-config configs/wireless.example.yaml \
  --token <shared-token>
```

节点 YAML 使用 WirelessComm 的 `local`、`peers`、`comm` 配置结构。
发送速率控制（pacing）需按网络情况手动配置，默认不开启。
此传输用于可信隔离网络；应用 token 只负责授权，不提供链路加密或对端身份认证。

WirelessComm 专用参数包括 `--comm-config`（必填）、`--token` 和 `--max-in-flight`；
`--max-image-bytes`、`--max-images`、`--max-sessions` 和 `--idempotency-cache-size`
是共用的限制。

## 会话管理与请求去重

HTTP 与 WirelessComm 遵循相同的会话和推理步骤规则。

- 会话通过 schema `embodiinfer.policy.session.v1` 打开；服务端分配 `session_id`，
  `session_revision` 从 0 开始。超过 `--max-sessions` 返回 `429 too_many_sessions`。
- 每个 step 带一个 `request_id` 和一个 `step_id`。同一会话内的 step 串行执行。
- `step_id` 必须等于下一个期望值。小于期望值返回 `409 step_id_too_old`；
  跳号或超前返回 `409 out_of_order_step`。
- 使用相同 `request_id` 和内容重试，会返回缓存响应。缓存有容量限制，
  已完成请求的响应被移出缓存后，再次请求会被拒绝，不重新计算。
  同一 ID 携带不同内容时返回 `409 idempotency_conflict`；
  使用旧 `session_revision` 的请求 ID 时返回 `409 stale_idempotency_key`。
- `reset` 要求 `Idempotency-Key` 头等于 `request_id`。它会清空缓存的 step 响应，
  把 `next_step_id` 重置为 0，并递增 `session_revision`。
- 未知或已关闭的 `session_id` 返回 `404 session_not_found`。
- step 响应使用 schema `embodiinfer.policy.step.result.v1`，报告 `session_revision`、
  `action_space`、`actions`、`timing` 和 `policy_revision`。

## 适配器 JSON 配置 {#adapter-json}

`--adapter-config` 指向一个 JSON 对象。下面是一个 π0.5 示例，包含一个 state 向量和两个相机：

```json
{
  "state_fields": ["observation.state"],
  "image_fields": ["observation.images.front", "observation.images.wrist"],
  "image_keys": {
    "observation.images.front": "observation.images.image",
    "observation.images.wrist": "observation.images.image2"
  }
}
```

按检查点调整特征名和状态向量顺序。通过 EmbodiRun 部署时，通常会根据策略绑定和部署配置自动生成该文件。

- `state_fields` —— 请求 `state` 对象中的字段名。例如 `"observation.state"` 读取
  `state["observation.state"]`。
- `image_fields` —— 按期望的输入顺序列出传入的相机名。
- `image_keys` —— 可选对象，把这些相机名映射到检查点的图像特征；例如
  `"observation.images.wrist_image": "observation.images.image2"`。未映射的名称保留原名。
  映射来源必须出现在 `image_fields` 中，目标必须存在于检查点中。重复的传入相机名，
  以及多个相机映射到同一特征，都会被拒绝。
- `policy_kwargs` —— 创建模型时传给策略工厂函数的参数。DM0.5 的 HTTP 服务需要
  `policy_kwargs.is_history: true`。

## 接入 RLinf 训练

[BUAA-CI-LAB/RLinf](https://github.com/BUAA-CI-LAB/RLinf/tree/embodiinfer-rollout-backend)
的 `embodiinfer-rollout-backend` 分支已将 EmbodiInfer 注册为 rollout 后端
（`rollout.model.model_type` = `embodiinfer` / `embodiinfer_gr00t` / `embodiinfer_openvla_oft` /
`embodiinfer_lingbotvla`）：EmbodiInfer 在 PPO/GRPO 内提供 rollout，actor 使用原生模型。
适配器通过 EmbodiInfer 的策略工厂函数创建模型，通过 refit API 更新权重。
这些集成覆盖 LIBERO 上的 π0.5、GR00T 和 OpenVLA-OFT，以及 RoboTwin 上的 LingBot-VLA。

权重更新流程见 [refit 接口用法](api.md#zero-copy-refit)。
