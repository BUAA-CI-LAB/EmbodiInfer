# 0005 — LingBot-VLA 作为第三个 flow 一等 policy（Qwen2.5-VL + MoT flow-matching expert）

- 状态：Draft（选型 + 设计。实现未开始；架构/参照/parity 目标已 scope，见 §4/§6）
- 日期：2026-07-14
- 更新：**模型结构全自持**——Qwen2.5-VL backbone 前向也 EmbodiInfer 自持（同 OFT 自持 Llama、pi0.5 自持 Gemma），vision encoder 作 leaf；不走 GR00T 的 backbone 黑盒 prefill。§3/§4.4 相应调整。

## 1. 摘要

将 **LingBot-VLA**（Robbyant / 蚂蚁集团，arXiv 2601.18692；v2 2607.06403）接为 EmbodiInfer 的第三个 flow 一等 `VLAPolicy`。LingBot-VLA 是 **pi0 式 flow-matching VLA**：`Qwen2.5-VL-3B` backbone + **Mixture-of-Transformers（MoT）action expert**，~10 步 flow-SDE 去噪，action chunk 50、`action_dim=14`（双臂 7+7）、state 14 维。其结构与 pi0.5 同族——MoT 使 VL token 与 action token 经**共享 self-attention** 耦合，即「VLM prefix KV + action expert attend prefix」，正对 EmbodiInfer 两阶段（`encode_prefix` 一次 / `denoise_step` N 次）。本提案属 **policy 层**（新增 `embodiinfer/policies/lingbot_vla/`），复用现有 `FlowVLAPolicy`/`FlowDecoder`/flow-SDE logprob/`LoopGraph`，不改引擎。

**选型依据**（scope 已核实）：LingBot-VLA 是 EmbodiInfer 的 sweet spot（flow 去噪头，`native rollout = eager HF、无 CUDA-graph`），且 **RLinf 已集成**（`rlinf/models/embodiment/lingbotvla/lingbotvla_action_model.py`，`LingbotvlaActionModel(BasePolicy)`，config `robotwin_click_bell_grpo_lingbotvla.yaml`，GRPO + RoboTwin 2.0，**RL checkpoint 已放** `RLinf/RLinf-lingbotvla-{click-bell,place-shoe}-grpo`）——提供可对齐的参照生成器与 RL E2E 路径，方法学与 pi0.5/GR00T 完全一致。开源 **Apache-2.0**（`robbyant/lingbot-vla-4b`）。

预期收益方向：(1) 验证引擎模型无关性——第三个 flow policy 复用既有 `LoopGraph`/`execute_pipelined`/数据并行，不改一行引擎；(2) LingBot-VLA 的 flow-SDE 去噪循环（N=10）在小 batch launch-bound 区经 `LoopGraph` 捕获的增量，同 GR00T（去噪循环 graph 2.5–2.8×@小 batch）——**量级由 microbench 定**（§9，见 §3 边界说明：RLinf 已复用 prefix KV，故增量杠杆是 CUDA-graph 而非 KV 复用）。

## 2. 动机与现状差距

- EmbodiInfer 现有三个真权重 policy：pi0.5、GR00T N1.7（flow，`FlowVLAPolicy`）、OpenVLA-OFT（单-pass categorical，`ParallelDecoder`）。引擎模型无关性主张（DEVELOPMENT §2.4）已由前两者+OFT 实证，但第三个 flow policy 能进一步验证「新 flow 模型零引擎改动继承 cudagraph/pipeline」。
- LingBot-VLA 的 native rollout（RLinf `LingbotvlaActionModel.sample_actions`）是 **eager PyTorch、无 `torch.compile`/CUDA-graph** 的 N=10 flow-SDE 去噪循环——正是 EmbodiInfer `LoopGraph`（proposal 0001 静态循环捕获）的适用场景，与 pi0.5/GR00T 的 launch-bound 收益结构同构。
- LingBot-VLA 的 RL logprob = **flow-SDE 逐步 Gaussian、按去噪链聚合**（`get_logprob_norm`，`noise_method=flow_sde`、`num_steps=10`、`noise_level=0.5`）——与 EmbodiInfer 已实现并能对齐的 pi0.5 flow-SDE logprob 同族（`embodiinfer/engine/rollout/logprob.py`）。

## 3. 目标与非目标

**目标**
- 新增 `embodiinfer/policies/lingbot_vla/`：`LingBotVLAPolicy(FlowVLAPolicy)`——`encode_prefix`（Qwen2.5-VL prefill 收 VL 前缀 KV + state features）、`denoise_step`（**EmbodiInfer 自持** MoT action-expert 前向，attention 走 `AttentionBackend`）、`flow_schedule`、`collate`/`pad`。
- attention 先支持 **sdpa**（parity 锚点，匹配 RLinf native 默认）；`supports_cuda_graph=True` + `allocate_static_prefix`/`copy_prefix_into` → 继承 `LoopGraph`。
- flow-SDE rollout logprob：复用 `FlowDecoder.sample_with_logprob`/`recompute_logprob`，对齐 RLinf `LingbotvlaActionModel` 的 `flow_sde` 逐步 Gaussian（num_steps=10, noise_level=0.5）。
- 注册 `@register_policy("lingbot_vla")`、`[lingbot_vla]` extra、`lingbot_vla` pytest mark、parity 测试。
- RLinf fork adapter（`rlinf/models/embodiment/vvla/vvla_lingbotvla_action_model.py`）+ ratio-at-θ0 + RoboTwin E2E（照 pi0.5/GR00T/OFT 模板）。

**模型结构全自持（硬要求）**
- Qwen2.5-VL backbone 的 **transformer 前向自持**（RMSNorm / mRoPE / attention 走 `AttentionBackend` / SwiGLU），保留官方 module 仅作权重持有（同 OFT 自持 Llama、pi0.5 自持 Gemma）；**vision encoder（Qwen2.5-VL ViT）作 vendored leaf**（跑一次编码图像，同 OFT Prismatic / GR00T vision）。**MoT action expert 前向亦自持**（`LoopGraph` 捕获的内循环）。全链无模型层黑盒——便于后续 cudagraph / KV 优化。

**非目标（后续增量，如实分离）**
- **depth 变体 / v2.0（`lingbot-vla-4b-depth`、arXiv 2607.06403 的 head/waist/mobile/dexterous DoF + predictive-dynamics）**：先做 base `lingbot-vla-4b` + RoboTwin post-train ckpt；变体后续。
- **LingBot-VA（arXiv 2607.08639，causal DiT + MoE video-action 世界模型，~15.3B，无 RLinf 参照）**：另一架构（world-action / predict-then-act），落 EmbodiInfer 现只有模板（Cosmos）的 WAM / best-of-N planning 路径，**单独 scope**，不在本提案范围。
- **PaliGemma-3B backbone 变体**（repo 支持）：先做默认 Qwen2.5-VL。

## 4. 设计

`make_policy("lingbot_vla", checkpoint=..., backbone_path=...)` 可显式使用固定的本地 Qwen2.5-VL 配置和 processor 资产；显式路径无效时直接失败。该选项用于离线复现，不改变 checkpoint 权重或推理数学；未指定时保留原有 checkpoint/canonical backbone 解析行为。

### 4.1 两阶段映射（对照 RLinf native 路径）

RLinf 参考路径：`LingbotvlaActionModel.predict_action_batch` → 底层 `lingbotvla.models.vla.pi0.modeling_lingbot_vla.LingbotVlaPolicy`；`sample_actions()` = `model.embed_prefix(...)` + `qwenvl_with_expert.forward(use_cache=True, fill_kv_cache=True)`（prefill 一次收 KV）→ `for idx in range(num_steps): x_t_mean, x_t_std, value_t = sample_mean_var_val(...)`（每步 `fill_kv_cache=False` 复用 `past_key_values`）→ `x_t = x_t_mean + sample_noise*x_t_std`。

- **`encode_prefix`（每观测一次）**：跑 Qwen2.5-VL backbone 的 prefill（图像 patch + 语言 token + 可选 state），在 MoT 共享 transformer 里收 **VL 前缀 per-layer KV**（`fill_kv_cache=True` 等价）+ state features。产物 `LingBotPrefix{kv, state_features, prefix_pad_mask, ...}`（静态形状，供 `LoopGraph`）。
- **`denoise_step`（N=10 次）**：`action_encoder(x_t, t)` → action tokens 经 **EmbodiInfer 自持 MoT action-expert 前向**（attention `[VL 前缀 KV ++ action tokens]`，走 `AttentionBackend`）→ 输出速度场 $v(x_t,t\mid\text{prefix})$。结构同 pi0.5（action expert attend 冻结的 VLM prefix KV），非 GR00T 的独立 DiT cross-attn。

### 4.2 flow schedule

LingBot-VLA 线性 timestep `linspace(1, 1/N, N)`、速度场积分 `x0_pred = x_t - v_t * t`——**方向 $t:1\to0$、$dt=-1/N$，与 pi0.5 同**（pi0.5 亦 override 为 $1\to0$）。故 `flow_schedule` 照 pi0.5 override（降序），非基类默认（GR00T 的 $0\to1$）。具体符号/离散化以 §6 的 box parity 对齐 RLinf native。

### 4.3 attention 子层（自持，走 backend）

MoT action-expert 的 attention 复刻 RLinf `qwenvl_with_expert` 的 expert 路径：`q/k/v` proj → reshape `[B,heads,S,head_dim]` → `self._attn.attend(q,k,v,mask,scaling)` → out proj。action tokens attend `[VL 前缀 KV(冻结) ++ action tokens(causal/full 按 native)]`；mask 语义（action 段是否 bidirectional、是否 attend 全 VL 前缀）逐 op 照 RLinf native 定，box parity 判定。RMSNorm/RoPE(Qwen2.5-VL mRoPE)/SwiGLU 自持（同 pi0.5/OFT），Qwen2.5-VL 的 mRoPE 2D 位置对齐同 GR00T proposal 0002 §4.1 的 `get_rope_index` 处理。

### 4.4 备选与取舍

- **备选 A：wrap RLinf `sample_actions`/官方 policy 前向。** 会 delegate 去噪计算、`AttentionBackend` 变假、循环无法 cudagraph → 违背引擎价值主张。**否决**（同 pi0.5/GR00T/OFT 硬约束）。
- **备选 B：backbone 作黑盒 prefill（GR00T 式，仅自持 expert）。** 改动小；但硬要求**模型结构全自持**（无模型层黑盒，便于后续 cudagraph/KV 优化，同 OFT/pi0.5）→ **否决**，采全自持（backbone transformer + MoT expert 均自持，vision ViT 作 leaf）。
- **备选 C：目标 depth / v2.0 变体。** 先做 base + RoboTwin post-train（有 RLinf RL ckpt 可对齐），变体后续。

## 5. 模型无关性判定

- 新增代码全部落 **policy 层** `embodiinfer/policies/lingbot_vla/`，实现基类协议，未改引擎。引擎仅经公共协议（`encode_prefix`/`denoise_step`/`flow_schedule`/`pad`/`supports_cuda_graph`/`allocate_static_prefix`/`copy_prefix_into`）驱动，未按模型名分支。
- `denoise_step` attention 经 `layers/attention.py` 既有 `AttentionBackend`。
- backbone 作黑盒 prefill：内部 attention 后端（FA2/SDPA）不在自持范围，属「运行一次」预填，不构成引擎对模型的隐含依赖（同 GR00T）。

## 6. 无损性与精度判据

- **对照对象**：RLinf `LingbotvlaActionModel` 的 native eager rollout（`num_steps=10`、`noise_method=flow_sde`、`noise_level=0.5`），注入固定初始噪声 $x_0$ + 固定采样噪声以消其内部 `sample_noise` 随机。因 lingbotvla 栈（LeRobot v3.0 + VeOmni + torch 2.8）与 EmbodiInfer env 可能冲突，采**跨环境 reference**（同 GR00T/OFT 方法学）：独立 env 的 box 脚本预跑存 reference（inputs + $x_0$ + 逐步噪声 + `ref_actions` + 可选 `vl_embeds`/`state_features`/logprob），EmbodiInfer env 侧读取比对。
- **两级判据**（口径：同 dtype、固定 $x_0$/噪声、byte-identical obs）：
  1. **action-expert 隔离**：把 native 的 prefix（VL KV + state features）注入 EmbodiInfer 去噪循环 → $\max\lVert\Delta a\rVert_\infty$ 达 bit-exact 或 bf16 底噪（同 GR00T 判据 1 的 ~1.6e-2 量级）。
  2. **端到端**：EmbodiInfer 全链路（Qwen2.5-VL backbone + 自持 MoT expert）vs native → $\max\lVert\Delta a\rVert_\infty$；backbone 跨版本对齐（mRoPE 2D、pre/post-norm 若适用）后收敛到判据 1 的 expert 底噪。
- **RL logprob parity**：behavior logprob 与 $\theta=\theta_{\text{behavior}}$ 处 recompute 一致、ratio-at-θ0 分位对照 native 噪声底（flow-SDE 高斯 logprob 对 velocity 小差异鲁棒，同 pi0.5/GR00T）。
- 复现：`tests/test_lingbot_vla_parity.py`（`lingbot_vla` mark，门控 `VVLA_LINGBOT_VLA_CKPT` + `VVLA_LINGBOT_VLA_REF`）+ box 脚本 `dev/scripts/{lingbot_ref_run,lingbot_compare,lingbot_inject}.py`。

## 7. 实现计划

- 新增 `embodiinfer/policies/lingbot_vla/{__init__,modeling_lingbot_vla,processor_lingbot_vla}.py`；`pyproject.toml` 加 `[lingbot_vla]` extra + `lingbot_vla` mark；`embodiinfer/policies/__init__.py` 导入注册。
- `supports_cuda_graph=True` + `allocate_static_prefix`/`copy_prefix_into`（`LingBotPrefix` 静态 KV）→ 免费继承 `LoopGraph`/`DenoiseGraph`。
- RLinf fork：`rlinf/models/embodiment/vvla/vvla_lingbotvla_action_model.py`（adapter，格式胶水 + weight-sync 键映射）；`SupportedModel` 注册 + `robotwin_*_grpo_vvla_lingbotvla.yaml`。
- **Stage 0 实现前置**：box 直读 `rlinf/models/embodiment/lingbotvla/lingbotvla_action_model.py`（44KB）+ `lingbotvla.models.vla.pi0.modeling_lingbot_vla` 定死 prefill/decode 张量形状、MoT attention mask 语义、flow schedule 符号、logprob 聚合（`joint_logprob`）——scope agent 是 raw fetch 摘要，实现前须逐 op 核实。
- 向后兼容：纯新增，不改公共 API。

## 8. 测试计划

- **CI（CPU）**：`test_lingbot_vla_registered`——注册 + 缺 `checkpoint` 抛 `ValueError`（不依赖权重）；现有 CPU 全套回归绿。
- **box（`lingbot_vla` mark）**：`test_lingbot_vla_parity`——读预存 reference，跑 EmbodiInfer policy（sdpa），断言 action-expert 注入 < 3e-2、端到端 < 6e-2（阈值随 box 实测口径定）；flow-SDE logprob ratio-at-θ0。
- **跨环境 reference**：lingbotvla 栈（LeRobot v3.0 + torch 2.8）与 EmbodiInfer env 若冲突，分环境产 reference（同 GR00T/OFT）。

## 9. 基准计划

- **LoopGraph vs eager（核心收益度量）**：bf16、$N=10$、$B\in\{1,2,4,8\}$、box(H200)。测去噪循环 graph 捕获相对 eager 的加速——**给全条件 + 消失点**（小 batch launch-bound 受益，大 batch compute-bound 收敛 ~1×，同 GR00T）。
- **KV-复用已存在的边界说明（此前评审关注点）**：RLinf native 已 `use_cache` 复用 prefix KV，故 EmbodiInfer 的**增量收益主要来自 CUDA-graph 捕获 action-expert 去噪循环，非 KV 复用**。microbench 显式拆分：(a) LoopGraph vs eager（graph 增量），(b) 若 native 未复用 KV 的假想基线对照（量化 KV 复用贡献，说明其已被 native 吃掉）。诚实标注 EmbodiInfer 相对 native 的净增量 = graph 段。
- **predict 全路径 / E2E**：EmbodiInfer vs native eager rollout（同 RoboTwin config、仅 rollout 后端不同），报告 predict s 与 E2E s/it（预期同 GR00T：graph GPU 段收益按模型调用 GPU 占比兑现，env-bound 时被仿真掩盖）。

## 10. 风险与局限

- **收益量级不确定**：native 已复用 KV，EmbodiInfer 净增量 = LoopGraph 段，量级由 microbench 定，可能温和（若 action expert 相对 prefill 占比低）。如实报告，不预判。
- **MoT attention mask 语义**：action 段是否 bidirectional / 是否 attend 全 VL 前缀，需逐 op 照 native；parity 若不达标先查此。
- **跨版本/跨栈对齐**：Qwen2.5-VL mRoPE 2D 位置、pre/post-norm 特征、torch 2.8 vs EmbodiInfer env 的算术差异（bf16 底噪，非 bit-exact），同 GR00T 经验处理。
- **env 冲突**：lingbotvla（LeRobot v3.0 + VeOmni）与 EmbodiInfer env 冲突 → reference 分环境产出。
- **RoboTwin 2.0 E2E**：RoboTwin 仿真的 GPU/EGL 占用与 EGL×CUDA 争用（同 LIBERO 的 EGL×CUDA 争用处理），E2E 需分离 placement；8×H200 GPU 服务器。

## 11. Open（未判死）
1. `lingbotvla_action_model.py` + `modeling_lingbot_vla` 精确 prefill/decode 张量形状、MoT mask、flow schedule 符号、logprob 聚合（`joint_logprob` True/False）——实现前 box 直读核实。
2. LoopGraph 相对 native（已 KV-复用）的净增量量级——microbench 定。
3. backbone 是否需自持（若 prefill 占比高到值得优化）——先黑盒，实测再定。
4. depth / v2.0 变体、PaliGemma backbone 变体是否后续接。
5. LingBot-VA（world-action 模型）作为 WAM-track 独立 scope 的取舍（此前讨论提出"能不能也接"→ 建议 VLA 落地后单独一轮）。
6. Robbyant/LingBot 的确切中文品牌名（scope 未逐字核实，无实现影响）。
