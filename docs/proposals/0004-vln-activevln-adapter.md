# 0004 — ActiveVLN（Qwen2.5-VL）作为第一个 VLN / 自回归 policy，经 verl 对齐；引入 `MemoryState` + `AutoregressiveDecoder` 两个扩展点

- 状态：Draft → **Rev E：以可重复的 native-vs-VVLA 自造 observation-stream A/B 作为首版合入门；正式 val-unseen/FSDP 与 padded-ragged 转为后续扩展**
- 日期：2026-07-14；Rev A/Rev B 2026-07-15；Rev C 2026-07-16

> 实现状态（2026-08-28）：EmbodiInfer 只保留通用 policy、generation、logprob 与
> refit 接口；仓内 trainer-specific integration 已移除。下文关于具体训练框架
> 接缝的内容仅作为早期方案背景，不代表当前公开 API。

## 1. 摘要

将 **ActiveVLN**（arXiv 2509.12618，`Qwen2.5-VL-3B` 微调 + verl 原生 GRPO）接为 EmbodiInfer 的第一个 **VLN（Vision-and-Language Navigation）** policy，也是第一个 **自回归（autoregressive, AR）文本-动作生成** policy。ActiveVLN 与现有三类 policy 结构均不同：pi0.5/GR00T 是 flow 去噪循环、OpenVLA-OFT 是单次并行 categorical；ActiveVLN 是 **在一条跨导航步不断增长的多模态序列上，逐 token 自回归生成自然语言动作短语**（"move forward 25cm, turn left 15 degrees"，逗号分隔、每步 ≤3 个低层动作）。

VLN 与 manipulation VLA 范式差异大，正好检验 EmbodiInfer 抽象的可扩展性。本提案识别出 ActiveVLN 引入的两处差异——**(a) 跨 env step 的 history/memory（growing KV cache）** 与 **(b) 回合内自回归动作生成**——并指出它们在实现层面是**同一个机制**：一条**可增量扩展的 KV cache**（cross-step memory 是其在 episode 尺度的复用，within-step AR 生成是其在 token 尺度的追加）。据此把差异收敛到两个清晰扩展点，而非为 VLN 开一堆特例：

1. **`MemoryState`（引擎层，模型无关）**：一个 per-env、跨 `EngineCore.execute` 调用持久、episode 边界复位的不透明记忆槽，穿过 `encode_prefix`。是对现有 `PrefixState` 生命周期的**泛化**（"一次预测内复用" → "整条 episode 增量复用"）。flow/OFT 不声明 recurrent，行为逐位不变。
2. **`AutoregressiveDecoder`（policy 层，复用 `ActionDecoder` serving contract）**：增量 KV decode + 逐 token categorical trace。ActiveVLN 已实现 branch-safe prefix expansion 与 recurrent group rollout，但 selected-memory best-of-N 仍未实现，因此它继续走专用 recurrent capability 路径而不冒充通用 `RLDecoder`。

**框架定位（对齐 verl）**：EmbodiInfer 的 rollout-backend 故事从 RLinf（接入结果见 architecture.md 与 benchmark.md）扩到 **verl**。ActiveVLN 是 verl 原生的 turnkey 基线（代码 + 权重 + RL loop + Habitat env server 齐备，Apache-2.0），提供可对齐的参照生成器与 E2E 回路——正是 OFT 当初经 RLinf `openvla_oft` 生成器获得的东西，现经 verl 获得。EmbodiInfer 的增量价值 = 用优化推理（decode CUDA-graph、KV 复用、**growing multimodal KV 的增量 prefill**）替换 verl 现有 rollout 路径，且逐位保真。**verl 自带的 bit-exact rollout `VeXact`（arXiv 2605.14220）只覆盖纯文本 dense/MoE 单轮，不支持 VLM、不支持 multi-turn**——ActiveVLN 的负载（VLM + 跨步增长的多模态 KV + 小 batch launch-bound decode）恰是 VeXact 未覆盖、而 EmbodiInfer 擅长的区间。

## 2. 动机与现状差距

- EmbodiInfer 现有三个真权重 policy 均为「Markov 观测 → 定长动作」结构：pi0.5/GR00T（flow 去噪，`FlowVLAPolicy`）、OFT（单次 categorical，`ParallelDecoder`）。三者的 `encode_prefix` 都是**每次 `execute` 无状态**：`embodiinfer/engine/core.py:86` 每步重算 prefix、用完即弃；`PrefixState`（`embodiinfer/policies/base.py:48`）只在**一次预测内**跨 denoise step / best-of-N 复用，不跨 env step。
- VLN video-VLM 的记忆活在**跨 env step 增长的 token 序列/KV** 里。ActiveVLN 的官方实现（`verl/workers/agent/parallel_env_vlnce.py`）逐回合 concat `running_states = [prompt] + [obs tokens+image] + [生成的 action tokens] + [下一步 obs+image] + ...`，`max_response_length≈25480`、`max_turn_budget=40`、`max_vllm_images=200`——**一条不断增长的单序列**，对应论文 Eq.6 的 `H_{<t}={V1,A1,...,V_{t-1},A_{t-1},V_t}`。现有 `encode_prefix(batch) -> PrefixState` 无处承载这个跨步状态。
- OFT 的 `ParallelDecoder` 是单次前向；ActiveVLN 的动作是 Qwen2.5-VL **逐 token 自回归生成的自然语言短语**（`vlnce_server/prompt.py`：`"move forward {25,50,75}cm"`、`"turn left {15,30,45} degrees"`、`"stop"`；每回合最多 3 个、逗号分隔）。要与 ActiveVLN 生成器 bit-exact 对齐，EmbodiInfer 必须做**增量 KV-cache AR decode**——这正是 OFT proposal 0003 §3 明确 defer 的一类（"vanilla 自回归 OpenVLA，需 KV-cache 增量 decode 引擎，另议"）。
- 若为 VLN 硬塞 if-else，会破坏引擎模型无关性。维护侧的一贯硬约束是「能统一就统一抽象」。本提案的核心工作即找到 memory 与 AR-decode 的统一上位机制（§4）。

## 3. 目标与非目标

**目标**
- 引入 **`MemoryState`** 引擎抽象 + per-env 记忆槽 + episode 复位（`EngineCore`/`GenerationBackend`），behavior-preserving：flow/OFT 逐位不变。
- 引入 **`AutoregressiveDecoder(ActionDecoder)`**：增量 KV decode + structured `DecodeResult`；token-level logprob/recompute 与 branch-aware recurrent rollout 已接通，同时保留与通用 `RLDecoder` 的能力边界。
- 新增 `embodiinfer/policies/activevln/`：`OpenVLAOFTPolicy` 同款「模块加载、forward 自持」——vision = Qwen2.5-VL 视觉编码器（vendored/leaf），LLM = Qwen2.5-VL decoder 自持前向（增量 KV），动作短语 tokenize/detokenize 对齐 ActiveVLN `vlnce_server/prompt.py`。
- **训练框架边界**：EmbodiInfer 提供通用 rollout/logprob/refit 接口；具体训练框架的注册、配置和数据转换由框架侧维护。
- 注册 `@register_policy("activevln")`、`[activevln]` extra、pytest mark、parity 测试。

**非目标（后续增量，如实分离）**
- **online Habitat E2E 提速的完整闭环**：先做**免仿真的离线 parity**（§6 Stage 1）；online E2E（接 ActiveVLN 的 Habitat HTTP env server，Habitat-Sim 0.1.7 + MP3D）作 Stage 3 增量（需 EGL×CUDA 分离 placement，同 LIBERO 的处理）。
- **VeXact 级双端 bit-exact**（rollout 与 FSDP actor 同用 batch-invariant kernel）：VeXact 当前不支持 VLM，属探索线（§10 Open），不作本提案保证。
- **StreamVLN**（真·滑窗 KV cache + 剪枝慢记忆，CC-BY-NC-SA、SFT-only）作为「growing-KV 抽象最纯载体」的 showcase：`MemoryState` 抽象须设计得能容纳它，但其接入列为后续（§10）。
- **AR decode 的 CUDA-graph 捕获**：变长生成 + 变长 KV 的图捕获比 OFT 定长单-pass 难（§10 风险），先 eager AR decode 保证正确性，graph 捕获作增量。
- backbone（Qwen2.5-VL 视觉塔）前向自持：作黑盒 leaf（同 GR00T/OFT vision 取舍），不重写。

## 4. 设计

### 4.1 统一上位机制：一条可增量扩展的 KV cache

ActiveVLN 的两处「新」——跨步 history memory 与回合内 AR 生成——在实现层是**同一件事**：往一条持久 KV cache 上**追加** token 的 K/V，后续 attention 只读已冻结的前缀 KV（causal）。区别只是作用域：

| 作用域 | 追加什么 | 现状对应 | 本提案 |
|---|---|---|---|
| 回合内（token 尺度） | AR 生成的每个 action token 的 KV | OFT 单次前向（无增量） | `AutoregressiveDecoder` 增量 decode |
| 跨回合（episode 尺度） | 新观测（图像 patch + 文本）+ 上一步动作的 KV | 无（prefix 用完即弃） | `MemoryState` 引擎持久槽 |

因此**一个引擎能力（持有并增量扩展 KV cache）同时覆盖两者**。这是本提案「统一、不过度细分」的落点：不是为 VLN 加两套机制，而是把「静态一次性 prefix」泛化为「可增量扩展的持久 KV」，AR-decode 与 memory 都是它的用例。

### 4.2 `MemoryState`（引擎层，模型无关）

```
MemoryState (Protocol)                     # 不透明，同 PrefixState 只暴露最小接口
  seq_len: int                             # 当前已缓存的序列长度（图像+文本+动作）
  def to(device) -> MemoryState
  def reset() -> MemoryState | None         # episode 复位（或引擎直接丢弃换 None）

VLAPolicy (base，新增，默认关闭)
  is_recurrent: bool = False               # 能力探测；recurrent policy 置 True
  def encode_prefix(batch, memory: MemoryState | None = None) -> PrefixState
      # 非 recurrent：忽略 memory（签名新增可选参数，pi0.5/OFT 行为不变）
      # recurrent：memory=None 时全量 prefill 新 episode；否则把 batch 的新观测
      #   增量追加到 memory 的 KV，返回指向扩展后 KV 的 PrefixState，并把更新后的
      #   memory 挂在 PrefixState.next_memory 上供引擎回存。
```

**引擎改动**（`EngineCore` / `GenerationBackend`）：持 `dict[env_id -> MemoryState]`。每次 `execute`：取该 env 的 memory（新 episode 为 None）→ `encode_prefix(batch, memory)` → 回存 `PrefixState.next_memory`；env `done` 时 `reset_memory(env_ids)`（丢弃对应槽）。`env_id` 已在 `Observation.env_id`（`embodiinfer/types.py:40`）/`BatchedObservation.env_ids` 中现成。

**关键性质（模型无关性）**：memory 不透明（引擎不读其内部，同 `PrefixState`）；能力经 `is_recurrent` 探测，**不按模型名分支**（见 `CONTRIBUTING.md` 的 Engineering principles）。非 recurrent policy：`is_recurrent=False`、`encode_prefix` 忽略新参数 → 引擎不建槽 → 行为逐位不变（bit-exact 判据见 §6 Stage 0）。

### 4.3 `AutoregressiveDecoder`（policy 层，复用 `ActionDecoder`）

新框架将能力分为两层：`ActionDecoder` 是所有 decoder 的 serving contract（`init_state` + `produce_chunk`，以及 behavior-preserving 的 structured `decode` 包装）；`RLDecoder(ActionDecoder)` 才表示可由通用 `GenerationBackend` 驱动的 policy-gradient rollout（candidate expansion + behavior/recompute logprob）。AR decode 作为 plain serving implementation 嵌入：

```
AutoregressiveDecoder(ActionDecoder)
  init_state -> None
  decode(prefix, ...):
      # 增量 AR 生成动作 token，追加 KV，解析为固定形状环境动作
      # 返回 DecodeResult(actions, traces, recompute_state, next_memory)
  produce_chunk(...):
      # 默认取 decode(...).actions

_ActiveVLNDecoder(AutoregressiveDecoder)
  sample_with_logprob(...)  # direct parity/future helper
  recompute_logprob(...)    # teacher-forced token path helper
```

ActiveVLN 只对模型生成的 response token 计 logprob（`action_mask=1`）。`ActiveVLNPrefix.expand(n)` 已提供 branch-safe L1 sharing，`GenerationBackend` 用显式、互异的 `SessionKey` 完成逐分支 decode 与原子 group commit；generic recurrent best-of-N 仍因 selected candidate memory 缺少提交语义而拒绝。`_ActiveVLNDecoder` **不是 `RLDecoder`**：stateless 通用路径继续按 nominal capability 门禁，ActiveVLN 使用独立 recurrent group contract。

### 4.4 `activevln` policy（policy 层）

参照 `embodiinfer/policies/openvla_oft/`（「模块加载、forward 自持、vision 作 leaf」）：

- **vision**：Qwen2.5-VL 视觉编码器（transformers `Qwen2_5_VLForConditionalGeneration` 的 visual 塔）作 leaf，跑一次编码新观测图像 → patch embedding。多图（`max_vllm_images` 逐步累积）经 memory 追加。
- **LLM 自持前向**：保留 `Qwen2_5_VLForConditionalGeneration` 仅作权重持有模块，transformer forward（RMSNorm / RoPE(含 mRoPE) / attention 经 `AttentionBackend` / SwiGLU）EmbodiInfer 自持（同 OFT 自持 Llama、pi0.5 自持 Gemma），**无模型层黑盒**（设计要求，便于 cudagraph/增量 KV 优化）。
- **动作短语 tokenize/detokenize**：对齐 `vlnce_server/prompt.py` 的短语表 + system prompt（"up to 3 actions, separated by ','"）；detokenize 短语 → 低层动作（含 25/50/75cm、15/30/45°、RxR 用 30/60/90°；连续同动作合并）。
- **checkpoint**：`Arvil/Qwen2.5-VL-3B_rl_r2r_4000` 等（标准 `Qwen2_5_VLForConditionalGeneration` safetensors），权重映射 vision/projector/language_model。

### 4.5 训练框架边界

EmbodiInfer 不内置训练框架注册或 vLLM 兼容 shim。框架侧负责把 observation、token、
action mask 和权重同步映射到 EmbodiInfer 的通用 generation/logprob/refit 接口；EmbodiInfer
本身不导入训练框架，也不拥有其配置命名。

### 4.6 备选与取舍

- **Design A（memory 放 observation、引擎无状态、每步重喂整条序列）**：改动最小、无引擎状态；但 ActiveVLN 是单条增长序列，重喂 = 每步 O(T) 重编码历史（`max_response_length≈25480` 下代价大），丢掉增量 KV 这一 EmbodiInfer 存在的意义。**作 fallback 基线**（正确但慢），不作目标。
- **Design B（引擎持增量 KV，本提案）**：匹配 ActiveVLN 官方 `running_states` 增长序列，复用 KV。目标。
- **选 CompassNav（sim-free、Qwen2.5-VL、EasyR1）而非 ActiveVLN**：sim-free 更便宜、离线 A* reward 利于确定性对齐；但任务是 ObjectNav（非指令式 R2R/RxR，代表性弱）、无 license、EasyR1 非 verl 原生。Long 定"对齐 verl" → ActiveVLN（指令 VLN-CE + Apache + verl 原生 + 4-phrase categorical）。CompassNav 可作 sim-free 对照后补。
- **选 StreamVLN（真滑窗 KV，最纯 memory 抽象）**：SFT-only（RL greenfield）、CC-BY-NC-SA。作 `MemoryState` 抽象的 showcase 后续增量，不作首个（无 RL 参照）。
- **wrap ActiveVLN/vLLM 的 generate（不自持前向）**：会 delegate 计算、`AttentionBackend` 变假、无法 cudagraph/增量 KV 优化 → 否决（同 pi0.5/GR00T/OFT 硬约束）。

## 5. 模型无关性判定

- `MemoryState` + per-env 槽 + episode 复位属**引擎抽象层**扩展：memory 不透明、经 `is_recurrent` 能力探测、不按模型名分支；非 recurrent policy behavior-preserving。
- `AutoregressiveDecoder` 属**引擎/policy 边界的解码策略层**（`ActionDecoder` 第三实现），经 `layers/attention.py` 既有 `AttentionBackend`；引擎只依赖 `encode_prefix + decoder`，不知晓「AR 生成 vs 单-pass vs 去噪」。
- `activevln` 新增代码落 **policy 层** `embodiinfer/policies/activevln/`；Qwen2.5-VL vision 作 leaf（同 GR00T/OFT），不构成引擎对模型的隐含依赖。
- trainer integration 位于训练框架一侧，不进入 EmbodiInfer 主仓。

## 6. 无损性与精度判据

- **Stage 0（现有模型保真）**：新增 `encode_prefix` 可选 `memory` 参数 + `is_recurrent` 默认 False + 引擎记忆槽（非 recurrent 不建槽）。判据：pi0.5/GR00T/OFT 现有 CPU 全套 + box 真权重 parity + rollout logprob 逐位不变（`max|Δ|=0`）。
- **Stage 1（ActiveVLN 离线 parity，免仿真）**：固定一批多模态观测序列（图像 + instruction，可用 `data/r2r_val_tiny.parquet` 的 instruction + 固定图像注入），EmbodiInfer 自持 Qwen2.5-VL AR decode vs native（HF `Qwen2_5_VLForConditionalGeneration` 或 `vllm serve`）——action token argmax 逐位一致、逐 token logprob `≤ε`（交叉实现 fp32/bf16 噪声底，同 GR00T/OFT 跨环境 reference 方法学）。**不需 Habitat**。含 memory：喂固定 episode 观测流，验每步随 KV 累积的 parity。
- **Stage 2（RL 消费量）**：ratio-at-θ0——EmbodiInfer rollout 的 tokens/action_mask/logprob 交 verl actor `compute_log_prob` 重算，`exp(Δ)` 分位数对照 native 噪声底（PPO 实际消费量），沿用既有 RL 集成的五层精度方法学。
- 判据分级同 `CONTRIBUTING.md` 的 Numerical discipline：引擎变换（memory 槽、增量 KV vs 全量重算、cudagraph）目标 bit-exact；跨实现（EmbodiInfer 自持 Qwen2.5-VL vs HF/vLLM）目标数值等价 `≤ε`，来源标注（RoPE/mRoPE dtype、attention kernel）。

## 7. 实现计划（分阶段，file-by-file）

**Stage 0 — MemoryState 引擎扩展（behavior-preserving）**
- `embodiinfer/policies/base.py`：`VLAPolicy` 加 `is_recurrent: bool = False`；`encode_prefix(batch, memory=None)`（现有 policy 忽略 memory）；`PrefixState` 文档补 optional `next_memory`。新增 `MemoryState` Protocol。
- `embodiinfer/engine/core.py`：`_prefill` 传 memory、回存 `next_memory`；`EngineCore` 持 `dict[env_id->MemoryState]` + `reset_memory`。非 recurrent 路径零行为差异（开关：`is_recurrent` False 即旧路径）。
- `embodiinfer/engine/rollout/generation_backend.py`：`sample_group`/`generate_with_logprob` 支持 recurrent 的 per-env memory + episode 复位钩子。
- 验证：CPU 全套 + box pi0.5/GR00T/OFT parity `max|Δ|=0`。

**Stage 1 — AutoregressiveDecoder + activevln policy**
- `embodiinfer/policies/decoder.py`：`AutoregressiveDecoder`（增量 KV decode + token-level categorical logprob，复用 OFT head 面）。
- `embodiinfer/policies/activevln/{__init__,modeling_activevln,processor_activevln,vision_qwen}.py`：`ActiveVLNPolicy(VLAPolicy, is_recurrent=True)`（Qwen2.5-VL 自持 LLM forward + 增量 KV，vision leaf，动作短语 tokenize/detokenize 对齐 `vlnce_server/prompt.py`）。
- 注册 `@register_policy("activevln")`；`pyproject.toml` `[activevln]` extra + mark；`policies/__init__` 导入。
- 验证：CPU `test_activevln_registered`；box 离线 parity（§6 Stage 1）。

**Stage 2 — 外部 trainer adapter + RL logprob seam + E2E parity**
- adapter 在训练框架仓库维护，并只消费 EmbodiInfer 的通用接口。
- 验证：ratio-at-θ0（§6 Stage 2）；短跑 GRPO 健康（ratio≈1 / approx_kl 小 / clip 小 / success 与 native 同水平）。

**Stage 3 — online Habitat E2E + 计时（增量）**
- 接 ActiveVLN `vlnce_server` Habitat HTTP env（Habitat-Sim 0.1.7 + MP3D，32 并行 / 2 GPU，`r2r_gpu_plan`）；EGL×CUDA 分离 placement（记忆 libero-egl-cuda-placement）。先 mock env 返回固定 obs 做 decode microbench，再真 env E2E s/it 对照 native。

## 8. 测试计划

- **CI（CPU）**：`test_memory_state`（recurrent 槽 + episode 复位 + 非 recurrent 零差异，mock）；`test_autoregressive_decoder`（增量 decode == 全量前向，mock/小 Qwen）；`test_activevln_registered`（注册 + 缺 checkpoint 抛错，不依赖权重）；现有 CPU 全套回归绿。
- **box（mark）**：pi0.5/GR00T/OFT parity 回归（Stage 0）；`test_activevln_parity`（Stage 1，门控 `VVLA_ACTIVEVLN_CKPT`）；verl E2E（Stage 2/3）。
- **跨环境 reference**：native ActiveVLN 生成器（其 vLLM/verl env）预跑存 reference（tokens/logprob/actions + 固定 RNG + 固定 obs 序列）；EmbodiInfer env 侧读取比对（同 GR00T/OFT 方法学，注入 obs 隔离图像变换差异）。

## 9. 基准计划

- **离线 decode microbench**：EmbodiInfer（增量 KV + cudagraph 若就绪）vs native HF/vLLM，Qwen2.5-VL-3B，固定 growing-KV 长度分档（模拟 turn=1..40 的 KV 累积），小 batch（ActiveVLN `train_batch_size=8`、`rollout.n=4`）。报告条件：dtype、KV 长度、batch、attention 后端。
- **E2E s/it**：EmbodiInfer vs native rollout（同 verl config，仅 `rollout.name` 不同），Habitat R2R/RxR。诚实标注收益区间与消失点（小 batch launch-bound 受益；env/render 主导时 e2e 收益 = 模型调用 GPU 占比，同 GR00T LIBERO 经验）。
- 增量 prefill 的价值：EmbodiInfer（每步只编码新观测、复用历史 KV）vs 重喂整条序列（Design A），随 turn 增长的每步 latency 对照——这是 growing-KV 抽象的直接收益度量。

## 10. 风险与局限

- **AR decode 的 CUDA-graph**：变长生成 + 变长 KV，图捕获比 OFT 定长单-pass 难。缓解：单个 decode step 静态形状可捕获（同 `DenoiseGraph` 捕一步），KV 长度按桶 pad；先 eager AR decode 保正确性，graph 作增量。列风险，不判死。
- **与 vLLM 定位重叠**：AR LLM decode 是 vLLM 主场。差异化：EmbodiInfer 专供 VLA/VLN rollout（小 batch、growing 多模态 KV、cudagraph、RL bit-exact parity），且 **VeXact 不覆盖 VLM+multi-turn**——填的是 verl 现有路径（HFRollout 无优化 / VeXact 文本单轮）的真实空白。如实陈述边界。
- **VeXact 级双端 bit-exact 需 FSDP actor 也切 batch-invariant kernel**，且 VeXact 当前不支持 VLM → 只改 rollout 一侧无法单方面 bit-exact；ratio=1 的强判据依赖此，列 Open（§6 Stage 2 先用「rollout vs native」的 argmax+logprob≤ε，不强求双端 bit-exact）。
- **ActiveVLN vendored 旧 SPMD verl，未 pin commit**：接入面（旧 `agent_rollout_loop` vs verl main 的 `AgentLoop`）需先在 box 核实其 verl 版本再定 external-注册 vs in-worker 替换。
- **online-only rollout**：ActiveVLN 真 rollout 强依赖 Habitat env server（无离线轨迹）；故分层验证——parity 免仿真（Stage 1），E2E 提速需 sim/mock（Stage 3）。
- **env 冲突**：ActiveVLN 栈（vLLM + Habitat 0.1.7 + Qwen2.5-VL）与 EmbodiInfer env 可能冲突 → reference 分环境产出（同 OFT/GR00T）。
- **动作短语 detokenize 的边界**：ActiveVLN 短语表 + 合并规则 + RxR 角度差异，需逐条对齐 `vlnce_server/prompt.py`，parity 时若不 0 需查。

## 11. Open（未判死）
1. ActiveVLN 精确 pin 的 verl commit → 决定 external-注册 vs in-worker 替换接缝。
2. VeXact 能否扩到 Qwen2.5-VL（README 无 VLM 证据）→ 决定能否走双端 bit-exact 的强 ratio 判据；作一条探索线，不预设。
3. AR decode 变长 + growing-KV 的 cudagraph 捕获策略（桶 pad KV 长度 / 分段捕获）。
4. `MemoryState` 抽象能否同时干净容纳 StreamVLN 的「滑窗 + 周期剪枝」有界 recurrent（而非无界 append-only）→ 决定 memory 接口是否需暴露「窗口/压缩」钩子。
5. Qwen2.5-VL mRoPE（多模态 3D RoPE）自持前向的 parity 细节（同 OFT Llama RoPE 的跨实现噪声底）。
6. 使用哪台 GPU 主机（候选为 4×H200 与 8×H200）——验证前确认。

---

# Rev A（2026-07-15）：验收标准映射与实现/测试门槛细化

> Rev A 不改 §1–§11 的选型与抽象设计，只补三件事：(a) 异构适配的现状盘点——哪些解耦已存在、哪些缺口由本提案补；(b) 八条验收标准逐条映射到机制与判据；(c) PR 门槛的四类测试清单。实现即按本节验收。

## 12. 异构适配现状盘点：解耦已有什么、缺什么

**问题**：引擎如何"屏蔽模型差异"？现状是三层解耦已存在并被四类模型验证（pi0.5/GR00T flow、OFT categorical、LingBot flow-MoT、Cosmos WAM）：

| 已有解耦 | 位置 | 屏蔽了什么差异 |
|---|---|---|
| `VLAPolicy` 契约（`encode_prefix` + `decoder` 属性） | `policies/base.py` | 引擎不知道动作怎么产生，只知道"prefill 一次 + 交给解码策略" |
| decoder capability（`ActionDecoder` / `RLDecoder`） | `policies/decoder.py` | serving 解码范式与 policy-gradient rollout 能力分离；仅 Flow/Parallel 声明通用 RL 面，Cosmos/Autoregressive 保持诚实的 serving capability |
| `PrefixState` Protocol（`batch_size/to/expand`） | `policies/base.py` | KV 布局差异（dense 列表 vs HF cache vs GR00T 特征） |
| 能力探测属性（`supports_cuda_graph` 等） | 各处 | 优化可达性差异，不按模型名分支 |

**ActiveVLN 暴露的缺口**（前四条即 §4 的两个扩展点，后四条由验收标准新增）：

| 缺口 | 现状 | 归属 |
|---|---|---|
| 跨 `execute` 的持久记忆 | `encode_prefix` 无状态、prefix 用完即弃 | `MemoryState` + 引擎 per-env 槽（§4.2） |
| 增量 AR 解码 | 无（OFT 是单次前向） | `AutoregressiveDecoder`（§4.3） |
| 变长 KV 的 ragged batching | bucket 只有 batch 一维 | §13.4 |
| 多 rollout 前缀共享（episode 尺度） | `expand` 只覆盖"一次预测内"广播 | §13.5 |
| 逐序列取消（stop/done/timeout） | 无 per-seq 早停 | §13.6 |
| 规范轨迹记录 | `ActionChunk.meta` 是自由 dict，无 schema、无 policy_version | §13.3（附带补齐权重版本语义） |
| episode 级吞吐口径 | bench 只报 obs/s | §13.8 |

结论：**解耦框架存在且已四次验证；ActiveVLN 适配 = 在既有扩展点上新增两个实现 + 六项引擎级能力，全部经 Protocol/能力探测接入，不动契约、不按模型名分支。**

## 13. 八条验收标准 → 机制与判据

### 13.1 官方 runner 语义一致（同 checkpoint/episode/seed）

- **机制**：动作短语 tokenize/detokenize/合并规则逐条对齐 `vlnce_server/prompt.py`（§4.4）；观测注入隔离图像变换差异（§8 跨环境 reference 方法学）。
- **判据（分两档，诚实边界）**：greedy 模式——action token argmax 与官方 runner 逐位一致、parsed action 序列逐条一致（这是 trajectory 语义等价的锚）；sampled 模式——跨实现 RNG 序列不可对齐，改验 (a) 固定注入 token 序列下逐 token logprob `≤ε`，(b) 同 seed 多 episode 的 SR 统计等价。**不宣称 sampled 轨迹逐位一致。**

### 13.2 full-prefill 与增量 session/KV 对齐

- **机制**：causal attention 下"逐步追加 KV"与"整条重算"数学同构；增量路径是 Design B，全量重喂是 Design A（§4.6），A 永远保留为对照与 fallback。
- **判据**：固定 episode 观测流，每步对比两模式的 logits 与生成 token：fp32 `max|Δ|=0`（bit-exact，引擎变换级）；bf16 `≤ε` 且来源可归因（attention kernel 归约顺序）。该判据同时是 §9"增量 prefill 收益 bench"的正确性前提。

### 13.3 轨迹记录 schema（TrajectoryRecord）

新增规范 dataclass（引擎填自有字段，verl adapter 补 env 侧字段）：

```
TrajectoryRecord:
  env_id / episode_id / step_idx
  policy_version        # 见下：权重同步层新增单调版本号
  seed                  # per-episode generator seed
  raw_tokens            # 生成的 action token ids
  token_logprobs        # behavior logprob（或 recompute_state 引用）
  parsed_action         # detokenize 后的低层动作序列
  executed_action       # env 回显（adapter 填）
  reward / done         # adapter 填
  timing                # prefill_ms / decode_ms / e2e_ms
```

- **配套改动**：`LocalWeightSync`/NCCL 同步路径增加单调 `policy_version` 计数器，`update` 时 +1，engine 侧只读——补齐"哪个版本的权重生成了这条轨迹"的审计链（RL 正确性所需，此前缺失）。
- **判据**：单测验证每字段来源与必填性；e2e 短跑后抽样核对 version 与权重更新时间线一致。

### 13.4 变长环境的 ragged batching

- **机制**：不同 env 的 episode 长度（=KV 长度）不同。批内按 **KV 长度桶** pad + attention mask：bucket 键从 `(batch,)` 扩为 `(batch_bucket, kv_len_bucket)`（kv 桶如 2k/4k/8k/16k/26k，上限对齐 `max_response_length≈25480`）。decode 单步形状 = [bucket_B, 1] query × [bucket_B, kv_bucket] KV，为后续图捕获保留静态形状（§10 风险 1 的铺垫）。
- **分阶段**：Stage 1 逐 env 串行（正确性锚）→ Stage 2 padded batch + mask（本验收项）→ 图捕获后置增量。
- **判据**：ragged batch 输出与逐 env 串行逐位一致（fp32）；批内长度方差最大化的构造用例过测。
- **Rev C 当前状态**：逐 env `serial_ragged` 正确性锚已实现并测试；padded KV + mask 尚未实现，因此 13.4 的最终性能验收仍开放。

### 13.5 一个 prompt 的多 rollout prefix sharing

- **机制（两级）**：GRPO `rollout.n` 条轨迹共享同一 episode 起点。**L1 计算共享（必做）**：起点段 [system prompt + instruction + 初始 obs] prefill 一次，`MemoryState.expand(n)` 复制 KV 到 n 个 rollout 槽——backbone 只跑一次，显存 n 份；**L2 存储共享（优化项，不入本次验收）**：只读共享段 + per-rollout 私有增长段，attention 读 [shared | private] 拼接视图。
- **诚实边界**：第一步动作分叉后各 rollout 观测不同，共享仅限公共起点段。
- **判据**：L1 路径与"n 次独立 prefill"逐位一致；prefill FLOPs 计数减少 n 倍（microbench）。

### 13.6 stop/done/timeout 取消剩余生成

三层取消语义：

| 层 | 触发 | 行为 |
|---|---|---|
| token 级 | 生成出 eos / "stop" 短语 / 达 max_new_tokens | 批内 alive mask 置 0，该序列停止扩展（写 pad），全灭则提前退出 decode 循环 |
| step 级 | parse 出 stop 动作 | 本 env 的 episode 进入终止流程，记录 done |
| env 级 | env done / timeout / 外部 cancel | `GenerationBackend.cancel(env_ids)`：丢弃记忆槽 + 丢弃在飞生成，幂等 |

- **判据**：单测覆盖三层与组合（批内部分取消、全取消、取消后复用槽位）；取消不影响批内其余序列的数值（与无取消对照逐位一致）。

### 13.7 RL 曲线与 SR/SPL 无系统性退化

- **机制**：ratio-at-θ0 先行（§6 Stage 2，PPO/GRPO 实际消费量对齐）；再短跑对照。
- **判据**：同 config 仅换 `rollout.name` 的短跑 GRPO，reward/SR 曲线在 native run-to-run 方差带内；同 checkpoint 的 val eval SR/SPL 差异 ≤ native 自身 seed 方差。参照 rlinf-vvla-integration 的三级验收先例（动作 → logprob → 训练曲线）。

### 13.8 端到端 episode 吞吐口径

- **指标集**（bench 输出 schema，禁止只报 token/s）：`episodes/hour`、`env-steps/s`、每 turn 延迟 p50/p99（随 turn 1→40 的曲线）、`gen_frac`（生成时间 / rollout 总时间）、增量 prefill vs 全量重喂（Design B vs A）的每步延迟对照。
- **报告条件**：dtype、KV 桶、batch、并行 env 数、attention 后端、Habitat 配置——按 `CONTRIBUTING.md` change flow 的阶段 5 给全条件与收益消失点。

## 14. 实现计划 Rev A（在 §7 Stage 0–3 上的增补）

| Stage | §7 原有 | Rev A 增补 |
|---|---|---|
| 0 | MemoryState 引擎扩展（behavior-preserving） | `MemoryState` 接口带 `expand(n)`（为 13.5）与**compact 钩子占位**（`compact(policy)` 默认 no-op——防 append-only 假设焊死，对应 §11 Open 4；CI 用 mock 滑窗 policy 测接口）；`weight_sync` 加 `policy_version` |
| 1 | AutoregressiveDecoder + activevln policy | alive-mask/三层取消与 `TrajectoryRecord` 已实现；ragged 当前为串行正确性锚，KV 桶 pad + mask 仍开放（13.4） |
| 2 | verl adapter + ratio@θ0 | adapter 补 env 侧字段（executed/reward/done）；prefix sharing L1（13.5）接 GRPO `rollout.n` |
| 3 | online Habitat E2E | episode 吞吐 bench（13.8 全指标集）+ 短跑退化检验（13.7） |

## 15. PR 门槛：四类测试清单

**① 单元测试（CI，CPU，mock/小模型）**
- `test_memory_state`：槽生命周期、episode 复位、`expand(n)`、compact 钩子（mock 滑窗 policy）、非 recurrent 零行为差异；
- `test_autoregressive_decoder`：增量 decode ≡ 全量前向（小 Qwen/mock）、alive-mask 早停、三层取消组合；
- `test_ragged_batching`：变长批 ≡ 逐 env 串行；
- `test_trajectory_record`：字段完备性、policy_version 单调性；
- 现有全套 CPU 回归绿（Stage 0 behavior-preserving 判据）。

**② 模型推理精度测试（box，真权重，免仿真）**
- 离线 parity（§6 Stage 1）：EmbodiInfer 自持 Qwen2.5-VL vs HF/vLLM native——greedy argmax 逐位、logprob `≤ε`（分 fp32/bf16 口径）；
- full-prefill vs 增量（13.2）：逐步 `max|Δ|=0`（fp32）；
- prefix sharing L1 ≡ 独立 prefill（13.5）。

**③ 端到端集成精度测试（box，verl + Habitat 或注入 obs 流）**
- ratio-at-θ0 分位数对照 native 噪声底（§6 Stage 2）；
- 官方 runner 轨迹语义一致（13.1 greedy 档）；
- 短跑 GRPO 健康指标 + SR/SPL 无退化（13.7）。

**④ 端到端性能测试（box）**
- 13.8 全指标集，EmbodiInfer vs native rollout（仅 `rollout.name` 不同）；
- decode microbench（§9：KV 长度分档、小 batch）；
- 增量 prefill 收益曲线（Design B vs A，随 turn 增长）。

**交付形态：单 PR**（遵循 `CONTRIBUTING.md` change flow 的阶段 7：开发期在本分支细粒度提交，最终 squash 为一个干净提交合入）。Stage 0–3 是**分支内的里程碑门**而非独立 PR——每个 Stage 的判据（§13/§15 对应项）在分支内达标后才进入下一 Stage；全部四类测试绿 + 八条验收齐备后，整体作为一个 PR 提交评审。PR 描述按本提案模板给出：改动清单、四类测试结果、八条验收逐条的证据链接（脚本/日志归档 `dev/scripts`、`dev/logs`）。

## 16. Rev B：ActiveVLN 优先适配的当前实施边界

Rev B 记录 2026-07-15 的实施决策：proposal 0007 所讨论的统一 `SessionState → StepContext → ActionGenerator` 重构后置；当前以本提案的 `MemoryState + AutoregressiveDecoder` 为基础完成 ActiveVLN 适配，但修正 session 提交时机和首版支持边界。适配产生的公共接口属于阶段性契约，真实模型 parity 完成后允许在统一重构中调整。

### 16.1 固定基线

- 官方 source：`https://github.com/arvillion/ActiveVLN`，commit `3a0c63b00e4f42c828cc74c3554afce17641da60`；
- R2R checkpoint：`Arvil/Qwen2.5-VL-3B_rl_r2r_4000`，revision `160987313e3e869705f42400d1b8f28177044518`；
- 首版只支持 R2R；RxR 后置；
- Qwen 参考版本：`transformers==4.51.3`；官方环境记录 `torch==2.6.0`、`vllm==0.8.5.post1`、`flash-attn==2.7.4.post1`；
- baseline profile 使用 standalone R2R evaluation：`temperature=0.2`、`top_p=0.8`、`max_new_tokens=512`、每 turn 最多 3 个动作、逗号分隔；checkpoint `generation_config.json` 中未被请求覆盖的参数必须计入 effective sampling 配置。

### 16.2 Session identity 与事务提交

首版 session key 显式使用 `(env_id, episode_id, rollout_id)`，由 engine/backend 控制面传递，不从 `PolicyBatch` 或 request id 隐式推导。

`MemoryState` 的正确生命周期是：

```text
checkout committed memory
  → 增量 prefill 当前 observation
  → AR decode 并追加本轮 action tokens（包括 terminal token）
  → parse/pack 成功
  → commit final memory
```

不得在 `encode_prefix` 后立即回存 memory：该时点尚未包含本轮 action-token KV。异常、timeout、external cancel 或 pack 失败均 rollback，上一份 committed memory 保持不变。reset/cancel 递增 per-session epoch，使旧 in-flight lease 即使晚到也不能复活或覆盖新 episode。

### 16.3 Additive decoder contract

基于 `origin/main` 的最终能力树，`ActionDecoder` 只定义 serving，`RLDecoder(ActionDecoder)` 才定义通用 `sample_with_logprob` / `recompute_logprob`；`FlowDecoder`、`ParallelDecoder` 属 `RLDecoder`，`CosmosDiffusionDecoder` 与 `AutoregressiveDecoder` 属 plain `ActionDecoder`。`ActionDecoder` 增加默认 `decode(...) -> DecodeResult`，默认实现仅包装现有 `produce_chunk`，因此 flow、parallel categorical 和 Cosmos diffusion 的计算顺序不变。`AutoregressiveDecoder` override `decode`，返回：

- 固定形状环境动作 tensor；
- token ids、selected-token logprobs、action mask、文本、parser 结果和 stop reason；
- teacher-forced recompute state；
- decode 后完整 `next_memory`。

`_ActiveVLNDecoder` 保留 direct `sample_with_logprob` / `recompute_logprob` helper，但不通过 nominal inheritance 宣称通用 `RLDecoder`。`GenerationBackend` 的 stateless 路径仍严格执行 `RLDecoder` capability gate；ActiveVLN recurrent group 则走显式 session ownership、branch-safe `MemoryState.expand`、逐分支 decode 和原子 group commit 的专用路径。普通 AR decoder 不会仅因存在同名 helper 就获得通用 RL 能力。

首版 R2R 动作 tensor 为 `[3,2]`：第一列 action id（pad=-1、stop=0、forward=1、left=2、right=3），第二列为距离厘米或角度；raw/typed 信息保留在 trace。无效输出不静默转换为 stop。

### 16.4 首版正确性边界

当前已实现：

- eager AR decode；
- transactional session memory；
- R2R prompt/parser；
- 自持 Qwen2.5-VL text forward、mRoPE 和 growing KV；
- 单环境共同起点的一次 prefill、多 rollout branch 隔离和原子提交；
- 多环境 `serial_ragged` 正确性 fallback；
- token-level behavior logprob、teacher-forced recompute、ratio-at-theta0 和 local GRPO 健康更新；
- `TrajectoryRecord`、policy version、三层取消与 prefill/decode/e2e timing；
- verl external-module/vLLM-shaped shim；
- Habitat adapter、SR/SPL evaluator 与 observation-stream E2E runner 已迁出 inference 仓，由下游 deploy 维护；
- Stage 1 reference schema、producer、validator、comparator 和真权重 GPU 门禁。

以下模式仍明确拒绝，不能静默以无状态或错误 session 语义运行：recurrent CUDA graph/full-loop capture、pipeline overlap、AsyncEngine continuous batching、DataParallelEngine、真 padded-KV ragged 并行和需要提交 selected-candidate memory 的 recurrent best-of-N。当前多环境模式明确标记为 `serial_ragged`，不冒充真 ragged batching。

### 16.5 当前停止点与 GPU 要求

截至 2026-07-16，Stage 0 CPU contract、Stage 1 真权重 parity、Stage 2 ratio/local GRPO/verl shim、Stage 3 注入 observation-stream 与公开 MP3D 场景 `17DRP5sb8fy` 的真实 Habitat renderer→model→action→simulator→native metric 闭环均已执行。真 checkpoint GPU suite 为 4 项全绿；注入 E2E 为 2/2 episode、SR/SPL=1.0；该场景 6 条 `val_seen` 为 5/6 success、SR=0.8333、SPL=0.6821，75 条 `train` 全部完成并得到 SR=0.8000、SPL=0.7411。16-turn HF full-history shadow 与 EmbodiInfer 的 action/token 语义为 100% 一致（terminal 唯一差异是 EmbodiInfer 在 `stop` 后早停而 HF 多生成 EOS），selected-token logprob 最大误差 `4.1544e-05`。

仍不得把公开单场景结果表述为正式 Habitat val-unseen：后者需要授权的完整 MP3D scene 数据，且该公开场景不出现在 `val_unseen`。真 padded-KV ragged、pinned verl FSDP actor 对照和不同 CUDA query shape 下 raw FP32 logits 逐位一致仍是开放扩展。环境端的证据、容差与任务指标由下游 deploy 维护。

## 17. Rev E：首版合入采用自造数据 A/B 实现验收

本 PR 的目标是判断 EmbodiInfer 替换原 ActiveVLN rollout 计算后是否引入实现层的
精度或效率问题，而不是复现论文的完整 MP3D 泛化分数。因此首版合入不依赖
授权的完整 MP3D scene：允许使用确定性生成或固定记录的 observation stream，
但 native 与 EmbodiInfer 必须使用同一 checkpoint、输入、历史、采样参数和计时边界。

首版合入门调整为：

1. 单元/回归测试全绿，未支持的 recurrent 并行模式必须显式拒绝；
2. native full-history 与 EmbodiInfer incremental 的 greedy token/action 语义一致，
   selected-token logprob 在实测容差内；不同 CUDA query shape 的 raw FP32
   logits 使用 `atol=5e-4, rtol=0`，不要求不现实的跨形状 bit-exact；
3. 自造 E2E observation stream 上，两端动作轨迹一致，因而相对 native 的
   SR/SPL 退化为零；ratio-at-theta0 与本地真权重 GRPO health 必须通过；
4. 性能使用 native vLLM full-history 与 EmbodiInfer incremental 的相同输入 A/B，
   报告模型调用和 episode schema，不把模型层加速冒充官方 Habitat E2E 加速。

以下内容不再阻塞首版 B=1/`serial_ragged` 能力合入，但保留为独立后续门：完整
MP3D `val_unseen` 泛化分数、pinned verl FSDP 分布式训练曲线、真 padded-KV
ragged 并行，以及跨 CUDA query shape 的 raw-logit 逐位一致。首版必须通过能力
标志和异常拒绝这些未支持的高阶执行模式，不能静默降级后宣称具备其性能。

## 18. Recorded-trajectory performance and inference optimization

`benchmarks/activevln-benchmark` replays the same first 48 numeric R2R/RxR
trajectories and every RGB frame as the StreamVLN benchmark. It keeps generated
response history within each episode and resets at episode boundaries. Both
splits use the pinned R2R checkpoint and R2R action grammar: RxR is an input
workload, not a claim of an RxR-trained model or navigation accuracy. No simulator
or task metrics enter this benchmark.

The B=1 BF16 greedy performance profile fixes seed 42, repetition penalty 1.05,
512 response tokens, and SDPA attention. The context limit is 128000 (the
checkpoint's positional limit), so long recorded episodes are not silently
truncated or reset. The checkpoint's image processor is unchanged. Measurement
starts at decoded CPU RGB and ends at the CPU action chunk; loading, image file
I/O, warmup and report serialization are excluded. `prepare_prefix` and
`encode_prepared_prefix`, plus `generate_tokens` and `finalize_generation`, expose
the same operations as the existing convenience methods. The model-only interval
covers vision, incremental text prefill and the entire AR loop, including KV
management and host stop control, while excluding input processing/H2D and final
text/action conversion/D2H.

The optimization work targets policy-local CUDA Graph execution and reuse of
existing Qwen vision and StreamVLN fused kernel mechanisms. Admission requires
real-checkpoint comparisons with the same episode selection, precision and
decoding settings. The task's updated accuracy contract permits different
tokens/actions if closed-loop navigation success remains close to the baseline;
exact-output comparison remains the default for other callers. The current
provisional tolerance is two fewer successes out of 48 episodes per split
(4.17 percentage points), pending the user's preferred bound. Each policy must
execute its own actions and resulting observations from the same starting poses,
goals and instructions. EmbodiRun owns that simulator evaluation and its SR/SPL
metrics. Replay alone cannot certify this contract: `compare.py
--accuracy-contract task-success` reports parity differences diagnostically and
leaves admission unknown. Retain all generated token IDs, action masks, chunks,
stop reasons and cache lengths for diagnosis. Capture and compilation must
finish before measured calls, with graph replay and fallback counts reported.
Optimized inference must preserve cancellation and committed-memory isolation;
training and unsupported execution modes retain their existing behavior.

The benchmark-specific `serve.py` exposes a frozen replay policy/configuration
through the existing versioned policy HTTP API for this paired evaluation.
It preserves the replay's preprocessing, timed forward and postprocessing calls,
warms graph shapes before accepting requests, and owns one private recurrent
memory per session. It contains no simulator loop or navigation metric. The
downstream EmbodiRun `benchmarks/activevln-navigation` controller owns execution,
STOP/distance success evidence and SPL, and verifies the source fingerprint
against the original latency report before running episodes.

Implementation constraints for this performance profile:

- Preserve the pinned 4.51.3 vision attention semantics. Its reference path uses
  a full block-diagonal SDPA mask; switching to independently segmented SDPA is
  a separate numerical change, even if the allowed attention edges are equal.
  Precompute shape-only layout tensors before capture and retain a reference
  path for real-weight comparison.
- Text mRoPE position and cache write position are distinct: images compress
  rotary positions, while the KV cache retains every image token. A captured
  decoder must receive both positions and may not use cache length as mRoPE.
- Respect BF16 rounding boundaries when fusing RMSNorm, SwiGLU and rotary
  operations. Existing StreamVLN fused kernels are useful mechanisms but are
  not automatically numerically identical to the ActiveVLN Torch reference.
- Graphs use stable tensor storage and explicit device length/position inputs;
  no `.item()` or CPU decoding belongs in captured execution. Growing episodes,
  early EOS/stop and graph replay after a different episode require validation.
- Keep committed memories and expanded rollout branches isolated. Reducing KV
  allocation/copy overhead must not permit a failed or cancelled decode to
  mutate a previously committed prefix. Training/recompute must remain on a
  differentiable path.

The first optional graph candidate uses a shared graph memory pool, fixed-address
KV execution workspace and explicit dynamic cache insertion offsets. The
checkpoint's dense-mask vision blocks run with precomputed rotary/window layout.
Text uses the existing SDPA backend and explicit padding masks; query bucketing
is configurable, including exact query lengths. Unknown shapes after startup
fall back to eager and are counted. Rounded fused operators live under
`backend/triton/rounded_ops.py`; their unit contract is exact rotary,
normalization and activation outputs. RMSNorm retains the reference Torch
FP32 mean reduction, and SwiGLU uses CUDA libdevice exponentiation and
round-to-nearest division. Full-model accuracy admission remains separate.

Inference memories with uniform layer layouts use a single packed KV allocation.
Fork/reserve/graph-workspace restore and captured-output append can copy all
layers at once, without changing the visible key/value layout or arithmetic.
Fork preallocates room for the incoming turn and the unchanged response budget,
bounded by the existing context limit. Each fork and rollout branch still owns
separate storage; graph workspace never aliases committed memory. Gradient-enabled
appends retain independent layer buffers to avoid shared autograd version counters.

An additional opt-in greedy candidate verifies one public action-phrase token
trie per forward. Each node attends only to the actual prefix and its ancestors;
its rotary coordinate is the original next coordinate plus its depth. Full-vocabulary
argmax and repetition penalty are evaluated independently on every path. Only
the path that matches those model choices is accepted and copied into the private
linear memory. Candidates come from the ten phrases in the unchanged public prompt,
never recorded benchmark answers. Unexpected tokens use ordinary AR decoding;
the grammar does not constrain model choices. EOS/stop, the response budget,
cancellation and sampling/training fallback retain their original contracts.
Different query shapes can change BF16 accumulation, so this path remains subject
to real-checkpoint validation under the selected accuracy contract before admission.
The candidate trie can optionally include each public phrase repeated up to
three times, retaining EOS/comma alternatives at every action boundary. This
amortizes weight reads when model choices repeat; mismatching continuations
still take the ordinary fallback. Tree size and accepted-token counts are
reported, and the response budget remains unchanged.

Startup shape prewarming can inspect initial/subsequent prompt lengths and image
layouts from supplied observations. It captures those input lengths across
explicit context buckets, plus scalar and tree decoding. Initial prompts only
need their empty-history bucket. Synthetic KV is confined to the disposable
workspace, whose resident marker is invalidated afterward, including on failure.
No responses are assumed and no public memory is changed. Capture remains
forbidden during measured calls; unknown shapes and contexts retain counted
eager fallback. The benchmark records its first-two-frames-per-episode probe
plan, context buckets and the complete startup time separately from E2E latency.

The graph workspace can have a smaller token capacity than the model's public
memory limit. This saves resident scratch storage without changing history or
response limits. Every graph call checks its entire padded query against the
physical workspace before writing; requests that exceed it use eager execution
against the complete public memory. Shape prewarming validates against the
physical capacity, and reports expose both workspace and model context limits.

A further optional text-attention candidate partitions the KV axis and merges
FP32 online-softmax partials, without materializing repeated GQA heads or the
query-by-context mask. Causal and tree-ancestor edges use the same dynamic cache
position. Probability/value multiplication decomposes FP32 probabilities into
BF16 components before tensor-core products, reducing the rounding difference
from the reference SDPA math backend. Reduction order remains different: operator
error is recorded against BF16 SDPA with FP32 intermediates. Complete replays
measure latency; closed-loop task success supplies the current accuracy gate.

Public phrase tries can be partitioned by their first token. Since the actual
first token is selected by the unchanged full-vocabulary greedy distribution
before verification, all other roots are already rejected. Capturing the smaller
root-specific trees reduces work without removing a potentially accepted path;
changed GEMM shapes remain subject to full model accuracy admission.

Real-prefix isolation has confirmed that SDPA math key padding can change BF16
results even with identical query lengths, weights, KV contents and positions.
The first R2R candidate with graphs/fusion/split attention/tree verification was
rejected by the original full token/action comparison. That output difference
does not alone reject it under the updated task-success contract. However, the
complete paired R2R closed-loop check subsequently drops from 35/48 baseline
successes to 29/48 for the frozen v9 candidate (−12.50 percentage points), with
SPL 0.6828 versus 0.5727. Initial images, poses and goal distances match for all
48 pairs. This candidate therefore also fails the task-success gate; the result
does not certify later candidates. All graph optimization switches
therefore remain opt-in experimental candidates; no lossless or sub-40-ms claim
is established by their CPU or operator-level checks.

An exact-rounding refinement keeps Torch's FP32 mean reduction for RMSNorm,
fusing only the input square/cast and the post-reduction normalization/weight
steps. SwiGLU uses CUDA libdevice exponentiation and round-to-nearest division
with the original BF16 intermediate cast. The operator gate is tightened to
bitwise equality before model-level validation; it must not inherit the earlier
one-ULP allowance after changing the implementation.

Tree verification can optionally request FP32 outputs from BF16 projection
matmuls, add bias in FP32 and cast once to BF16. This preserves checkpoint
weights/storage and avoids changing global BLAS settings. The option applies
only to the batched candidate tree, including its LM head; ordinary prefill and
serial fallback retain their reference projections. It addresses measured
shape-dependent projection rounding but does not assert exact full-model parity.

The subsequent paired R2R test of frozen v11 combines exact-rounding fusion,
root-partitioned trees, split-KV attention and actual query lengths. Keeping
BF16 tree projection yields 35/48 successes, matching the baseline count, with
SPL 0.6877 versus 0.6828, 70.42 ms mean closed-loop model E2E and 59.07 ms
complete forward over 1,008 calls. Forward includes vision encoding, prefill
and complete decoding with CUDA synchronization; E2E additionally includes
CPU observation preprocessing and CPU action postprocessing. Neither timer
includes HTTP transport or simulator execution.
Enabling FP32 tree projection gives 31/48 successes and SPL 0.6155, so the
operator-level improvement does not justify selecting that profile. Initial
RGB/poses/goal distances and primitive execution traces match the paired
protocol for all 48 episodes. RxR execution of this R2R-only prewarm profile
twice exhausted memory after 160 calls/495 primitive steps in episode 20, with
66,385 cached tokens exceeding its 65,536-token graph workspace. The allocator
retry reproduced every recorded call. The incomplete runs are preserved without
scoring infrastructure errors as task failures. A separately named full-context
configuration covers both splits' query shapes and the unchanged 128,000-token
context limit. It reproduces all 1,008 R2R calls at 70.90 ms E2E/59.24 ms forward,
but its 675 resident text graphs and larger workspace exhaust memory in RxR
episode 20 after 127 calls/419 steps, despite zero inference fallbacks. A
separate RxR-only prewarm configuration is evaluated below; per-workload
configurations and reports must remain explicit. The v11 recorded
probe contains only the first two frames per episode for source/configuration
provenance; these closed-loop timings must not be presented as complete replay
latency or a sub-40-ms result.

The subsequent complete 2,997-frame fixed R2R replay of the same R2R-only BF16
profile measures 70.23 ms mean E2E and 59.24 ms complete forward (prefill including
vision 28.58 ms; entire decode 30.63 ms). All graph fallback counters are zero.
Its source/configuration match the audited 35/48 closed-loop result, with only
recorded-frame selection and output location changed. Report SHA256:
`477f96f461b2c418cf696fe2521204baa4f2ca1f7416af4b4252c8a58b250fb9`.
This validates the R2R latency and selected-episode success count; the 40 ms
target is not met.

The separate RxR-only prewarm/128,000-token workspace profile retains the same
frozen source and BF16 tree projection while reducing resident text graphs from
675 to 437. All 48 native RxR episodes complete: 17/48 successes versus baseline
15/48, SPL 0.2591 versus 0.2442. Its first 127 episode-20 calls match the previous
interrupted combined-prewarm run, and it completes the full 500-step protocol.
All initial-state and primitive-trace audits pass, with zero graph fallbacks.
The matching complete 3,879-frame fixed replay measures 104.45 ms mean E2E and
93.45 ms complete forward, again with zero graph fallbacks. Report SHA256:
`ea532db33c0bc73b587316469b7b7614680823a9c79f3a27dcfd79eb0659f346`.
Both workload-specific configurations preserve success count on the selected
episodes. This is not an exact-output or unseen-split accuracy claim; neither
workload meets the original 40 ms E2E target.
