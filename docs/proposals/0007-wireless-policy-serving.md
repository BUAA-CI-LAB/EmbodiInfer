# Proposal 0007: WirelessComm policy serving

Status: Accepted

## 1. 问题与范围

在多机器人共享 Wi-Fi 的部署中，为版本化 policy API 增加 WirelessComm 数据面，复用其持久
TCP 连接、结构化多 segment payload、backpressure 和 pacing。改动限于
`embodiinfer.engine.serve`，不改变 policy、模型执行、action 数值或 Deploy 的机器人控制语义。

## 2. 动机与现状差距

原 HTTP 入口把 multipart 解析、session 状态机和模型调用放在同一个
`PolicyHttpService` 中，因此第二种 transport 无法在不复制 session/idempotency 逻辑的情况下
接入。HTTP 客户端还需要先把 JSON metadata 和所有图片拼成完整 multipart body。

WirelessComm 已提供静态 peer directory、持久全双工 TCP、bytes segment、每 peer 有界队列及
可选 byte-quantum/pacing，但只提供 `send`/`recv`，不提供应用级 RPC correlation。

## 3. 目标与非目标

目标：

- HTTP 与 WirelessComm 共享同一个 session、step、reset、close 状态机；
- Wireless step 直接传输结构化 metadata 与编码后的 JPEG/PNG bytes segment；
- 支持并发请求 correlation、超时、结构化远端错误和原有 step 幂等；
- WirelessComm 为可选依赖，未安装时 HTTP 和核心 inference import 不受影响。

非目标：

- 不修改 WirelessComm framing 或 scheduler；
- 不加入 peer discovery、跨 inference replica 重试或 session 迁移；
- 不宣称 TLS、节点认证、自动 rate tuning 或相对 HTTP 的性能收益；
- 不改变模型输入、输出或数值计算。

## 4. 设计

`PolicyService` 接收已验证的 `RawPolicyRequest`，唯一拥有 session ordering、request fingerprint、
幂等缓存及 adapter 调用。`PolicyHttpService` 仅把 HTTP route/multipart 转成该类型；
`WirelessPolicyServer` 把 WirelessComm structured payload 转成同一类型。

Wireless RPC 使用固定 request/response tag 和 piggypayload envelope：

```text
schema: vvla.policy.rpc.v1
kind: request | response
rpc_id: 每次网络请求的 correlation ID
method: health | capabilities | open_session | step | reset | close
status/code: response 状态
```

`rpc_id` 只负责匹配网络响应。step 的 `request_id` 仍是业务幂等键，因此同一请求可以使用新的
`rpc_id` 向同一个 session 重试。每个 peer 有一个接收循环；请求交给有界并发任务，响应可以乱序，
由客户端 correlation table 恢复对应关系。模型 adapter 是同步接口，通过 worker thread 调用，避免
阻塞 WirelessComm event loop。

备选方案一是传输完整 HTTP method/path/header/body。它改动更小，但保留 multipart 拼接与解析，
并令新的数据面永久依赖 HTTP 表示，因此不采用。备选方案二是 sidecar bridge，适合独立 A/B，
但增加进程、复制和运维面，不作为正式接口。

## 5. 模型无关性判定

本功能属于模型无关的 serve capability。它只依赖 `ServingAdapter` 的 `capabilities`、`infer`、
`reset` 公开协议和通用 `RawPolicyRequest`/`ModelResult`，没有 policy 名称分支或 checkpoint 假设。

## 6. 无损性与精度判据

网络入口不改变 adapter 输入或模型执行。CPU fake-adapter 测试要求相同 structured step 的首次调用
和幂等重试返回完全相同的 response，且 adapter 只执行一次。真实模型 action parity 不需要新增
数值容差；HTTP 与 Wireless 传入相同编码图片、state、instruction 时，目标为 adapter 输入逐字段
一致以及 action response 逐字段一致。

## 7. 实现与兼容

- `service.py`：transport-neutral `PolicyService`；
- `http_server.py`：保留原 HTTP route、CLI 与 `PolicyHttpService` 导入路径；
- `wireless_server.py`：Wireless ingress 及 `embodiinfer-wireless-serve` CLI；
- `factory.py`：两个 CLI 共用的 model/engine construction；
- `wireless-comm` 放入 `wireless` optional extra。

HTTP 是默认行为，不配置 Wireless 时没有行为变化。Wireless session 固定在接受
`open_session` 的 server，客户端不得把失败后的 recurrent session 迁移到其他 peer。

## 8. 测试计划

CPU 测试覆盖 HTTP 状态机回归、Wireless open/step、结构化图片、幂等重试、认证错误和 RPC
correlation。测试使用 fake adapter 和本机 TCP，不需要 CUDA、checkpoint 或物理机器人。

## 9. 基准计划

性能结论需要在相同机器人数量、图片大小、step 周期、模型延迟和 Wi-Fi 拓扑下，对比现有 HTTP
与 WirelessComm。记录 encode、local queue、wire、service queue、policy 和完整 RTT 的 p50/p95/p99，
并报告 quantum/rate/burst。现有 WirelessComm 对 RLinf Channel 的结果不能替代该 A/B。

## 10. 风险与局限

- WirelessComm 静态 peer identity 不是加密认证；token 也不提供链路保密性。当前入口只适用于可信
  隔离网络，生产部署需要 transport 级 TLS/mTLS 或等价机制。
- 客户端超时不等于服务端取消。step 只能以相同 `request_id` 在相同 server/session 上重试。
- pacing 最优值依赖 AP、节点数量、信号和负载，不提供通用默认速率。
- Recurrent adapters keep their declared execution limits; stateless adapters may
  opt into the cross-session batching extension below.

## 11. Cross-session serving batches

Status: Accepted. Scope: the shared serving layer and the π0.5 adapter.

The core already executes tensor batches, but the CLI rejects `max_batch > 1`
and the π0.5 adapter serializes single requests. Add an optional
`BatchServingAdapter.infer_batch` protocol and a bounded, single-worker serving
queue. `--max-batch` remains 1 by default; `--max-wait-ms` bounds the collection
window once the worker takes the first request. HTTP and WirelessComm use the
same wrapper. A model adapter must explicitly implement the batch protocol;
unsupported adapters fail at startup rather than pretending to batch.

Session ordering, reset/close serialization, and idempotent replay remain in
`PolicyService`. The batch worker receives only admitted requests from separate
session calls. A batch method returns one result or exception per request in
input order; malformed observations must not poison valid peers. Shutdown
rejects queued work and drains the in-flight call. Queues are bounded and report
429 under overload. Client disconnect is not model-call cancellation.

π0.5 retains its checkpoint processor for each row, concatenates the prepared
camera/token/mask tensors, calls `EngineCore.execute` once, and restores actions
with each row's own state. No model names, tensors, or checkpoint assumptions
enter the queue. Recurrent batching and async-engine pipeline changes are out
of scope. Reusing `AsyncEngine` was considered, but it accepts observations and
recollates them, losing the serving processor's checkpoint-specific masks and
normalization. A transport-local batch queue would duplicate HTTP/Wireless logic.

The graph bucket cannot exceed the configured batch ceiling: with max batch 3,
three requests use bucket 3, not the default next power-of-two bucket 4.

Validation requires exact row routing and processor parity in deterministic CPU
tests, including B=1/2/3, partial windows, malformed peers, overload, shutdown,
idempotent retries, and reset ordering. Same-batch eager/graph parity uses fixed
noise, dtype, attention, schedule and decode steps; it must not compare different
batch shapes or claim cross-shape bit-exactness. CUDA/checkpoint tests remain
opt-in. No new throughput claim is made without matched real-weight measurements
of batch size, wait window, client concurrency, warmup and measured iterations.
