# 0003 — OpenVLA-OFT 作为第一个非-flow 一等 policy，与 ActionDecoder 抽象统一

- 状态：Draft（approach A；Stage 0 抽象 + Stage 1 OpenVLA-OFT 自持前向 + end-to-end parity + KV-split cudagraph + RL token-level logprob 均已落地并 box 验证；RLinf E2E adapter 待做）
- 实测（box，2026-07-13）：EmbodiInfer 自持 OFT vs 官方 canonical 模型 —— vision+projector `max|Δ|=0`、action-token argmax 56/56 一致、action-logits `max|Δ|=1.08e-4`（交叉实现 fp32 噪声）；自持 Llama vs HF Llama `max|Δ action-logits|=0`。**LLM 前向 EmbodiInfer 自持（无模型层黑盒），vision=vendored timm 0.9.x leaf。**
- 日期：2026-07-13

## 1. 摘要

将 **OpenVLA-OFT** 接为 EmbodiInfer 的第一个**非-flow** 一等 `VLAPolicy`。OFT 与现有两个 flow policy（pi0.5 / GR00T N1.7）结构相反：**单次前向、无 N 步去噪循环、动作分布是离散 256-bin categorical（非 flow-SDE）**。为在加入它的同时保持架构清晰，本提案把现有引擎里**隐式摊开**的「prefix → action chunk」这一步提升为显式策略 `ActionDecoder`，使「flow 去噪循环」与「单次前向」成为同一契约的两种实现（**approach A**，已批）。

选型依据（OpenVLA 家族是 RL-for-VLA 事实标准 baseline；RLinf 同时集成 OpenVLA/OpenVLA-OFT 且发布 RL checkpoint，提供可对齐参照生成器）。**对齐目标经源码核实为离散-token 变体**（RLinf `openvla_oft`，`implement_version="rlinf"`），故本提案只需 `CategoricalHead`，不引入连续/高斯头。

## 2. 动机与现状差距

- EmbodiInfer 现有「模型无关」只做到 `encode_prefix`；「产 action chunk」这一步 flow-baked 地摊在 `VLAPolicy.denoise_step`（`embodiinfer/policies/base.py:110`）+ `flow_schedule`（`base.py:135`）+ `sample_actions` 的 Euler 循环（`base.py:202-206`）+ `EngineCore._integrate`（`embodiinfer/engine/core.py:66-91`）+ `engine/graph.py` 三个 graph 类里。`_prefill` 还无条件抽噪声 `x=randn(...)`（`core.py:105-112`）。
- RL 侧 `rollout/logprob.py` 整体是 flow-SDE 逐步转移密度；`rollout/generation_backend.py` 的 `generate_with_logprob`/`sample_group` 写死 `flow_sample_with_logprob`；`RolloutSamples`（backend.py:31）带 flow 专有的 `trajectory/sigma/num_steps`。
- OFT 单次前向、categorical logprob 会与以上全部冲突。若以 if-else 并列塞入，会破坏引擎的模型无关性；设计要求是「能统一就统一抽象」。

## 3. 目标与非目标

**目标**
- 引入 `ActionDecoder` 策略抽象 + `FlowVLAPolicy`/`FlowDecoder` 下沉（approach A），**behavior-preserving**：pi0.5 / GR00T 的 parity 与 rollout logprob 逐位不变。
- 新增 `embodiinfer/policies/openvla_oft/`：`encode_prefix`（Prismatic 视觉+prompt 前缀）+ `ParallelDecoder`（EmbodiInfer 自持单次 causal 前向 → `CategoricalHead`）。
- 对齐 RLinf `openvla_oft` 生成器：ground-truth action + token-level logprob parity；接入 RLinf E2E（对齐 ratio-at-θ0）。
- 注册 `@register_policy("openvla_oft")`、`[openvla_oft]` extra、pytest mark、parity 测试。

**非目标（后续增量）**
- 连续 L1 / 高斯 / diffusion 头（RLinf RL 路径不走，官方 L1 SFT 权重不可复用 → 不在范围）。
- vanilla 自回归 OpenVLA（需 KV-cache 增量 decode 引擎，另议）。
- `official` 变体（RoboTwin，256-宽/相对 id/film/proprio）——先做 `rlinf` 变体（LIBERO/ManiSkill）。
- Prismatic backbone 前向自持（backbone 作黑盒 prefill，同 GR00T 取舍）。

## 4. 设计

### 4.1 统一抽象：`ActionDecoder` 策略（approach A）

把「prefix → action chunk」收成显式策略对象，挂在 policy 上。引擎与 graph 层只依赖 `encode_prefix + decoder`，无模型分支。

```
VLAPolicy (瘦身，模型无关)
  encode_prefix(batch) -> PrefixState        # 保留，共享
  collate / pad / config
  supports_cuda_graph / allocate_static_prefix / copy_prefix_into   # cudagraph prefix 管线
  decoder -> ActionDecoder                    # 新增：generation 策略

ActionDecoder (Protocol)
  produce_chunk(prefix, params) -> actions               # 确定性生成
  sample_with_logprob(prefix, params) -> RolloutSample   # actions + behavior_logprob + recompute_state
  recompute_logprob(prefix, recompute_state) -> logprob  # differentiable，PPO/GRPO ratio 用
  # cudagraph hook：暴露可捕获的静态前向（供 GraphManager）

FlowVLAPolicy(VLAPolicy)          # pi0.5 / GR00T 基类（近乎不改）
  denoise_step / flow_schedule    # flow 专有，留在此层
  decoder = FlowDecoder(self)

FlowDecoder(ActionDecoder)        # 平移今天的循环 / SDE-logprob
  produce_chunk  = 今天的 _integrate（Euler 循环，含 randn 噪声种子）
  sample_with_logprob = flow_sample_with_logprob
  recompute_logprob   = flow_logprob_recompute

OpenVLAOFTPolicy(VLAPolicy)       # 新增，非-flow
  encode_prefix -> OFTPrefix（Prismatic 视觉 patch + prompt 的 LLM KV）
  decoder = ParallelDecoder(self, head=CategoricalHead)
```

**关键性质**：OFT 走 `VLAPolicy` 直系，永不继承 `denoise_step`/`flow_schedule`（下沉进 `FlowVLAPolicy`/`FlowDecoder`）。这不是并列 if-else，而是把现有 flow loop 已有的模型无关性质推广到「单次前向」。

**引擎改动**（`core.py`）：`_integrate` → `decoder.produce_chunk`；`_prefill` 里的 `randn` 噪声下沉进 `FlowDecoder`（OFT 不抽噪声，`_Staged.x` 变 decoder-internal）；`execute` = `_prefill`(仅 prefix+簿记) → `decoder.produce_chunk` → `_pack`（`_pack` 原样复用）。`execute_pipelined`/`_pipeline_step` 的 overlap 管线保留（opt-in，flow 相关）；OFT 的 produce_chunk 极轻、prefill 即全部 compute，overlap 价值不同——如实记录、非本提案目标。

**graph 改动**（`graph.py`）：把 `DenoiseGraph`（捕获一次 `denoise_step`）泛化为 `ForwardGraph`（捕获任意 `decoder` 的静态前向），复用 `allocate_static_prefix`/`copy_prefix_into`/`set_prefix` + thread_local capture + private-stream warmup。`FlowDecoder.produce_chunk` 内部仍用 `LoopGraph`/`DenoiseGraph`（不变）；`ParallelDecoder.produce_chunk` 用 `ForwardGraph`（去噪循环是 N 步特例、OFT 是 N=1 特例）。`GraphManager` 对 OFT 按 `(bucket,)` 键（无 `num_steps`）。

**RL 改动**（`rollout/`, `rl/`）：`RolloutSamples.trajectory/sigma/num_steps` 泛化为 opaque `recompute_state`（flow 存 trajectory+sigma，OFT 存 sampled action tokens）；`GenerationBackend.generate_with_logprob`/`sample_group` 改调 `decoder.sample_with_logprob`；`GRPO._logprob`（`embodiinfer/rl/grpo.py`）改调 `decoder.recompute_logprob`。GRPO 的 PPO-ratio 机制本身 logprob-agnostic，不动。

#### 带 padding 的共享 flow 状态

共享 flow 接口同时覆盖内部状态带 padding 的模型（DM0.5）。
`FlowVLAPolicy.flow_state_shape(batch_size)` 声明积分状态形状，
`finalize_actions(state, prefix)` 在积分或 Graph replay 完成后执行一次输出转换，
`flow_logprob_mask(state)` 提供可广播的计分维度 mask。默认实现分别使用配置中的
动作形状、恒等转换和全维度计分，因此普通 flow policy 的默认行为保持不变。
DM0.5 只实现这些接口及模型前向，不另建 decoder 或 rollout 循环。

`FlowDecoder` 统一调用 policy 的 `new_noise`，向 GraphManager 传递当前 prefix，
并将 mask 交给通用 flow-SDE helper。采样与重算保留完整内部轨迹，仅在转移密度
中排除 mask 未选中的维度；mask 在每个 batch row 中必须选择相同的正数个元素。
带 mask 的采样按实际保存的转移残差计算 logprob，以保留 DM0.5 的低精度计分
语义；未传 mask 的 helper 保持原有噪声计分及计算顺序。

显式传给 `FlowDecoder.sample_with_logprob` 的 generator 同时控制初始噪声和每步
噪声，修正旧通用 decoder 仅把它用于初始噪声的问题。未传 generator 时保持原有
全局 RNG 路径。重构验证固定 dtype、步数、sigma 和种子，对照原实现的动作、完整
轨迹、behavior/recompute logprob 与梯度；Graph 验证同时覆盖逐步和整段回放。

### 4.2 OFT 两阶段映射（对照 RLinf 生成器）

参照 `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py`（`OpenVLAOFTForRLActionPrediction`）。RLinf 是**一次** LLM forward 出全部 action-token logits；EmbodiInfer 拆成 `encode_prefix`（prefill 视觉+prompt 的 KV）+ `ParallelDecoder`（56 个 action-token 读 KV），因 attention 为 **causal**，前缀 KV 对 action-token 只读可复用，拆分数学等价（bit-exact 由 box parity 判定，同 pi0.5 自持 KV）。

- **`encode_prefix`（每观测一次）**：
  1. Prismatic backbone 作黑盒：fused DINOv2+SigLIP（224px、每图 256 patch、6-通道 stack）→ projector → 256 vision patch embedding；prompt `"In: What action should the robot take to {task}?\nOut: "`（tokenizer **左 padding**，首 token BOS=1、末 token 空格=29871）。
  2. 视觉 patch 前置 + prompt token，跑 Llama-2 7B **prefill**，缓存 per-layer KV（`OFTPrefix{kv, prompt_len, n_patches, ...}`）。`position_ids = cumsum(attention_mask)-1`。
- **`ParallelDecoder.produce_chunk`（一次前向）**：追加 `action_dim*num_chunks = 7*8 = 56` 个 placeholder token 位置，**embedding 置零**（empty action-query），单次 causal forward 读 `encode_prefix` 的 KV → `lm_head` → 取这 56 个位置的 logits `[B,56,32064]` → `CategoricalHead`（见 4.3）。
- **`CategoricalHead`**：屏蔽非 action-bin（有效 token id ∈ `[vocab-256, vocab)=[31744,32000)`，vocab=32000）；`produce_chunk`(greedy)=`argmax`；token→action：`discretized = vocab - token_id; clip(discretized-1, 0, 254); a_norm = bin_centers[idx]`（`bins=linspace(-1,1,256)`，255 个中点）+ q01/q99 反归一（`BOUNDS_Q99`，`dataset_statistics.json`）→ `[B, num_chunks, action_dim]`。

### 4.3 RL logprob seam（categorical，bit-exact 规格）

`ParallelDecoder.sample_with_logprob`（rollout）与 `recompute_logprob`（train）逐 op 对齐 RLinf：

- **采样**：`logits' = logits/temperature`，若 `top_k>0` 过 `TopKLogitsWarper`，`log_softmax` → `multinomial(exp(logprob),1)` → 绝对 token idxs `[B,56]`；greedy=`argmax`（**无 temperature**）。LIBERO GRPO：`temperature=1.6, top_k=-1`（不裁剪）。
- **behavior logprob**：在 **temperature-scaled + top_k + `-inf` 屏蔽后**的 logits 上，`logprob = -F.cross_entropy(logits', target=idxs, reduction="none")` → reshape **token-level `[B,56]`**（56 = num_chunks 8 × action_dim 7）。
- **recompute（differentiable）**：同构（同样恒 `/temperature` + top_k + 屏蔽），`target = sampled tokens`（`recompute_state`）→ 逐位可与 actor 的 `default_forward` 对齐。
- `recompute_state` = sampled 绝对 token idxs `[B,num_chunks,action_dim]`（`rlinf` 变体约定：全 32064 宽 + 绝对 id + 固定 slice 偏移）。
- value head（PPO 用；LIBERO GRPO `add_value_head=False`）：取 hidden index `-(56+1)`（首个 action-token 前一位）→ `ValueHead(4096→512→128→out)`，`action_level→out=8`/`chunk_level→out=1`。

### 4.4 备选与取舍

- **approach B（不拆策略，`VLAPolicy` 加 `produce_chunk` 默认=flow 循环，OFT override）**：改动小，但 base class 仍带 flow 味、OFT 要 stub 掉用不上的 `denoise_step`/`flow_schedule`。**否决**（Long 批 A）。
- **连续 L1 / 高斯头**：RLinf RL 路径不走、官方 L1 SFT 权重不可复用；**排除**（若将来接 RIPT-VLA 再议）。
- **backbone 自持（重写 Prismatic 前向）**：backbone 是「运行一次」的 prefill，重写收益低、风险高；**黑盒 backbone**（同 GR00T 备选 B）。
- **wrap RLinf/官方 forward**：会 delegate 计算、AttentionBackend 变假、无法 cudagraph；**否决**（同 pi0.5/GR00T 硬约束）。

## 5. 模型无关性判定

- `ActionDecoder`/`FlowVLAPolicy`/`FlowDecoder` 属**引擎抽象层**重构，behavior-preserving；引擎与 graph 只依赖 `encode_prefix + decoder`，不按模型名分支。
- OFT 新增代码落在 **policy 层** `embodiinfer/policies/openvla_oft/`（+ 通用 `ParallelDecoder`/`CategoricalHead`）；`ParallelDecoder` 的前向经 `layers/attention.py` 既有 `AttentionBackend`。
- backbone 作黑盒 prefill（内部 attention 后端不在自持范围），同 GR00T，不构成引擎对模型的隐含依赖。

## 6. 无损性与精度判据

- **Stage 0（flow 保真）**：`FlowDecoder` = 现有代码平移，构造上 bit-exact。判据：pi0.5 / GR00T 现有 parity 测试 + rollout logprob（`flow_sample_with_logprob`/`flow_logprob_recompute`）逐位不变（CPU 全套 + box 复验 `max|Δ|=0`）。
- **Stage 1（OFT ground-truth）**：对照 RLinf `openvla_oft` 生成器（同权重、同 obs、同 temperature/top_k、`do_sample=True` 固定 RNG）——action token idxs 逐位一致、token-level logprob `[B,56]` 达 bit/1e-6 级、token→action 反归一逐位一致。因 RLinf env（tf4.53 + openpi 栈）与 EmbodiInfer env 冲突，采跨环境 reference（同 GR00T 方法学）。
- **Stage 2（RL 消费量）**：ratio-at-θ0——EmbodiInfer rollout 的 tokens/prev_logprobs 喂 actor `default_forward` 重算，`exp(Δ)` 分位数对照 native 噪声底（PPO 实际消费的量），照 `rlinf-vvla-integration` 的五层精度方法学。

## 7. 实现计划（分三阶段，file-by-file）

**Stage 0 — ActionDecoder 重构（behavior-preserving）**
- 新增 `embodiinfer/policies/decoder.py`：`ActionDecoder` Protocol + `FlowDecoder`（平移 `core._integrate` 的三路径 + `flow_sample_with_logprob`/`flow_logprob_recompute` 调用）。
- 改 `embodiinfer/policies/base.py`：`VLAPolicy` 瘦身（去 `denoise_step`/`flow_schedule`/`sample_actions`/`new_noise` 抽象，加 `decoder` 抽象属性）；新增 `FlowVLAPolicy(VLAPolicy)`（保留 `denoise_step`/`flow_schedule`，`decoder=FlowDecoder(self)`）。
- 改 `embodiinfer/policies/{pi05,gr00t,mock}/modeling_*.py`：基类 `VLAPolicy`→`FlowVLAPolicy`（其余不动）。
- 改 `embodiinfer/engine/core.py`：`_integrate`→`decoder.produce_chunk`；`_prefill` 去 randn（`decoder.init_state` 下沉 FlowDecoder）。
- 改 `embodiinfer/engine/rollout/generation_backend.py` + `embodiinfer/rl/grpo.py`：backend `generate_with_logprob`/`sample_group` 与 GRPO `_logprob` 改调 `decoder.*`（flow 实现经 FlowDecoder 委托原 helper，逐字不变；`RolloutSamples.trajectory` 暂留原名，Stage 1 泛化为 opaque `recompute_state`）。
- `graph.py` 的 `DenoiseGraph→ForwardGraph` 泛化**移到 Stage 1**（OFT 真正需要时再动；Stage 0 不碰 graph 层，`FlowDecoder.produce_chunk` 原样用 `GraphManager`），降低对两个 working 模型的风险。
- **状态（2026-07-13）：Stage 0 代码完成，本地 CPU 全套 59 passed / 6 skipped、ruff 干净（flow 数值路径 CPU 级逐位不变）；box pi0.5/GR00T 真权重 parity + graph 路径 `max|Δ|=0` 复验待做。**

**Stage 1 — OFT policy + ParallelDecoder**
- 新增 `embodiinfer/policies/openvla_oft/{__init__,modeling_openvla_oft,processor_openvla_oft}.py`：`OpenVLAOFTPolicy(VLAPolicy)`（`encode_prefix` = Prismatic 黑盒 + Llama prefill KV）；`ParallelDecoder` + `CategoricalHead`（4.2/4.3）。
- 新增 `embodiinfer/policies/decoder.py` 里 `ParallelDecoder`/`CategoricalHead`（通用，可被别的 single-pass 模型复用）。
- 注册 `@register_policy("openvla_oft")`；`pyproject.toml` 加 `[openvla_oft]` extra + mark；`policies/__init__.py` 导入。
- `supports_cuda_graph=True` + `allocate_static_prefix`/`copy_prefix_into`（`OFTPrefix` 静态 KV）→ 继承 `ForwardGraph`。
- 验证：CPU `test_openvla_oft_registered`；box ground-truth parity（§6 Stage 1）。

**Stage 2 — RL logprob seam + RLinf E2E**
- RLinf fork：`rlinf/models/embodiment/vvla/vvla_openvla_oft_action_model.py`（adapter，格式胶水 + weight-sync 键映射）；`SupportedModel` 注册 + `libero_*_grpo_vvla_openvlaoft.yaml`。
- 验证：ratio-at-θ0（§6 Stage 2）+ E2E rollout→update PPO 健康 + 计时对照 native。

## 8. 测试计划

- **CI（CPU）**：`test_decoder_refactor`（FlowDecoder 数值 == 重构前，mock）；`test_openvla_oft_registered`（注册 + 缺 checkpoint 抛 `ValueError`，不依赖权重）；现有 39+ CPU 测试全绿（回归）。
- **box（mark）**：pi0.5/GR00T parity 回归（Stage 0）；`test_openvla_oft_parity`（Stage 1，门控 `VVLA_OPENVLA_OFT_CKPT` + `VVLA_OPENVLA_OFT_REF`）；RLinf E2E（Stage 2）。
- **跨环境 reference**：RLinf `openvla_oft` 生成器（其 env）预跑一次存 reference（tokens/logprob/actions + 固定 RNG）；EmbodiInfer env 侧读取比对（同 GR00T reference 方法学）。

## 9. 风险与局限

- **Stage 0 回归风险**：重构触及 `base.py`/`core.py`/`graph.py`/`logprob.py` 与两个 working 模型 → 必须 bit-exact 保真，先 CPU 后 box 双验，任一 `max|Δ|≠0` 即回退。
- **KV 拆分 parity**：EmbodiInfer 拆 prefill/decode KV vs RLinf 单次 forward，causal 下数学等价但数值需 box 确认（同 pi0.5 自持 KV 的既有经验）。
- **causal vs bidirectional**：OFT paper 摘要未描述 bidirectional，HF/RLinf 实现实测为 causal single-pass；若正文另有 bidirectional 变体，需按实际 checkpoint 校正（Open）。
- **权重基座**：须用离散-token OFT SFT（`Haozhan72/Openvla-oft-SFT-*`）+ RLinf RL ckpt；官方 moojink L1 ckpt 不可用。box 先确认可下载。
- **env 冲突**：RLinf openvla_oft 栈（tf4.53 + openpi/prismatic）与 EmbodiInfer env 冲突 → reference 分环境产出。
- **eval determinism**：RLinf config `temperature_eval=1.6` 使 eval 也采样；真 deterministic 需 `temperature_eval<=0`（argmax，无 temperature）——parity 用 `do_sample=True` 固定 RNG。

## 10. Open（未判死）
1. OFT 是否有 bidirectional 变体（只读摘要；实现为 causal）。
2. `RLinf/*` ckpt 精确 state_dict key（按 Prismatic 模块结构推得，未逐 key dump）。
3. `official` 变体（RoboTwin，256-宽/相对 id/film/proprio）后续是否需要。
4. OFT single-pass 下 `execute_pipelined` overlap 的价值边界（prefill 即全部 compute，无明显 denoise 段可 overlap）。
5. vanilla 自回归 OpenVLA（KV-cache 增量 decode）是否作为后续第二个非-flow 覆盖。
