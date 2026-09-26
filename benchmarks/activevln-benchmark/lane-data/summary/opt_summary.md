# 优化分支 `feat/activevln-inference-optimization` 实测汇总（RTX 5090 32GB）

- 分支: BUAA-CI-LAB/EmbodiInfer `feat/activevln-inference-optimization` @ `8b39867`
  （在 `05d6037` 上新增 3 个 commit：graph/fused 推理、优化 tensor batching、文档快照）
- 本 lane 追加（未 push 前为本地 worktree `EmbodiInfer-opt`）:
  - `feat`: RxR action space（官方 RxR checkpoint + 30/60/90 提示词）接入与 benchmark 接线，batch 白名单放宽到 8
  - `fix`: 每行持久 committed-KV 缓冲，替代每轮 `clone()`，修复 RxR B8 OOM（见第 3 节）
- 硬件/环境: RTX 5090 32GB (sm_120)，驱动 580.95.05
  - EmbodiInfer: torch 2.10.0+cu130, transformers 4.51.3, triton 3.6.0
  - vLLM modern: vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.17.0
- 协议: 48 episodes 全量记录帧重放（R2R 2997 calls / RxR 3879 calls），greedy，
  repetition_penalty=1.05，max_new_tokens=512，max_context=128000；
  优化开关：cuda_graph/fused_ops/split_attention/tree_decode=true, query_bucket_size=1；
  vLLM modern：ngram_gpu 16 draft、vision compile + FULL_AND_PIECEWISE CUDA graph、paged KV/prefix cache。
- 单位：ms/obs = 摊薄到每次观测的 E2E（batch 行为整批 E2E/实际观测数）；sps = 请求数/实测时间。
- B1 走单会话路径（`cuda_graph.py`，未被 OOM fix 触及）；B2/4/8 为修复后重跑（`*-fix.json`）。

## 1. 最终矩阵（16 格，15 完成 + RxR B8 OOM）

| dataset | batch | EmbodiInfer sps | EmbodiInfer ms/obs | EmbodiInfer GiB | vLLM sps | vLLM ms/obs | vLLM GiB | 快者 |
|---|---|---|---|---|---|---|---|---|
| R2R | 1 | 17.01 | 58.8 | 11.4 | 11.55 | 86.6 | 26.2 | EmbodiInfer |
| R2R | 2 | 22.89 | 43.7 | 13.9 | 17.25 | 58.0 | 26.0 | EmbodiInfer |
| R2R | 4 | 26.60 | 37.6 | 15.4 | 22.33 | 44.8 | 25.7 | EmbodiInfer |
| R2R | 8 | 27.62 | 36.2 | 22.6 | 27.07 | 36.9 | 24.9 | EmbodiInfer |
| RxR | 1 | 10.81 | 92.5 | 16.2 | 7.19 | 139.1 | 25.9 | EmbodiInfer |
| RxR | 2 | 12.92 | 77.4 | 16.9 | 10.96 | 91.2 | 25.9 | EmbodiInfer |
| RxR | 4 | 13.11 | 76.2 | 20.8 | 13.44 | 74.4 | 25.6 | vLLM |
| RxR | 8 | **OOM** | — | — | 16.08 | 62.2 | 24.9 | vLLM |

补充：R2R B1 p50/p95 59.6/72.1ms（vLLM 81.0/141.0）；RxR B1 p50/p95 85.8/156.2ms（vLLM 116.8/291.6）；
全矩阵 graph fallback=0。RxR B8 EmbodiInfer 见第 3 节 OOM 说明。

## 2. 同一硬件前后对比（main 快照 eager 逐 token → 优化分支）

| dataset | batch | 旧 sps | 新 sps | 加速 | 旧 ms/请求 | 新 ms/请求 |
|---|---|---|---|---|---|---|
| R2R | 1 | 2.86 | 17.01 | 5.9x | 345.9 | 58.8 |
| R2R | 2 | 5.40 | 22.89 | 4.2x | 364.3 | 43.7 |
| R2R | 4 | 9.19 | 26.60 | 2.9x | 422.6 | 37.6 |
| R2R | 8 | 13.31 | 27.62 | 2.1x | 575.7 | 36.2 |
| RxR | 1 | 2.70 | 10.81 | 4.0x | 366.0 | 92.5 |
| RxR | 2 | 4.95 | 12.92 | 2.6x | 397.8 | 77.4 |
| RxR | 4 | 7.69 | 13.11 | 1.7x | 507.5 | 76.2 |
| RxR | 8 | 8.62 | OOM | — | 902.2 | — |

（旧数据为 8 ep × 12 帧、96 请求/重复的小协议；新数据为 48 ep 全量重放。口径差异见第 5 节。）

## 3. OOM fix（本 lane 新增）

**问题**：`batching_activevln.py::_finish_generation` 每轮对每行做
`self._row_kv(row, 0, lengths[row]).clone().contiguous()`——整段历史重新分配 + 拷贝
（36.9KB/token；RxR 单行 2-4 万 token ≈ 1-1.5GiB）。B8 时 pool scratch（256k token ≈ 9.4GiB）
与 8 行 owned 拷贝并存，加上图的私有池，峰值 26.6-28.5GiB → OOM。

**修复**：新增每行持久 committed-KV 缓冲 `_owned_rows`，按 4096 token 对齐增长；
同一 memory 链上原地补齐（append-only，前缀不变），跨 episode/行重排时按 weakref 判定，
旧 memory 仍被持有时分配新缓冲，保持“旧 memory 不可变”。计数：R2R B8 337 次分配 / 2660 次复用，
RxR B4 353 / 3526（~90% 复用）。

**验证**：
- CPU 单测 54 passed（batching / batch_serving / graph_warmup）
- 逐 token 回归零差异：R2R B8 2997/2997、RxR B4 3879/3879（token 与 stop_reason 全等）
- 性能/显存：R2R B8 27.38→27.62 sps，22.9→22.6GiB；RxR B4 13.05→13.11 sps，22.8→20.8GiB
- RxR B8：OOM 点从 600 obs 推迟到 840 obs（尝试 pool 196.6k / workspace 64k 均未通过），**仍标记 OOM**

**剩余根因（未修，属架构级）**：pool scratch 与每行 owned KV 双份并存，8 行长历史时
KV 需 ~2× 显存。彻底修复需让 attention 直接读每行自有缓冲（multi-base / 冻结段回收分配器），
超出本次 bug fix 范围；32GB 上 RxR B4 可完整跑（20.8GiB），B8 需更大显存或该改造。

## 4. 为什么分支 README 的数字与第一轮差距大

1. **代码代差（主因）**：第一轮 main 快照是 eager 逐 token、每 token 逐层 KV concat、padded dense mask；
   分支新增策略内 CUDA Graph（vision/prefill/decode/tree）、Triton 融合算子（RMSNorm/RoPE/SwiGLU）、
   split-KV attention、packed KV、短语树验证。同机 R2R B1：345.9→58.8ms（5.9×），decode 305→28ms。
2. **对照 vLLM 也不同**：第一轮 plain API server；本轮换成分支 modern 栈（ngram_gpu、vision compile+graph）。
3. **协议/边界**：分支 E2E 从已解码 CPU RGB 起算、48ep 全量；第一轮小协议且 wall 含磁盘解码。
4. **硬件**：分支 README 是 4090 24GB；5090 上同代码更快（R2R B1 58.8 vs 70.0ms）。

## 5. 口径与限制

- 吞吐 = 完成请求数 / 实测时间；batch 行为整批 E2E 摊到实际观测数，含尾部未满批（R2R B8 occupancy 7.57/8）。
- EmbodiInfer E2E 含 CPU 预处理+动作解析，不含磁盘解码/HTTP；vLLM 边界对齐（解码后 RGB→解析后动作，
  排除磁盘、HTTP、启动、编译、capture、warmup）。
- vLLM 显存为 `gpu_memory_utilization=0.85` 预分配（~25-26GiB），非最小占用；EmbodiInfer 为实际分配峰值。
- RxR B8 EmbodiInfer OOM 三次（原名 600 obs / pool196.6k 696 obs / fix 840 obs / fix+ws64k 840 obs），
  均如实记录，未使用 partial latency。
- 记录帧重放吞吐，不是闭环 SR/SPL；BF16 padding 可能改变个别贪心输出（不影响吞吐口径）。
- 上游优化实现未改；本 lane 仅接线（RxR action space/batch-8）+ OOM fix。
