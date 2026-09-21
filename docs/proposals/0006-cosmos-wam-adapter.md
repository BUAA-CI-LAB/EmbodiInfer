# 0006 — Cosmos Policy 作为 WAM（world-action model）一等 policy

- 状态：Implemented（自持 DiT + VAE leaf + 扩散采样器落地；对齐 cosmos-policy 自身推理的 parity 通过；best-of-N planning 演示完成，见 §6/§9）
- 日期：2026-07-15

## 1. 摘要

将 **Cosmos Policy**（NVIDIA，`NVlabs/cosmos-policy`，arXiv 2601.16163，Apache-2.0）接为 EmbodiInfer 的第 5 个一等 `VLAPolicy`、一个 **WAM（world-action model）**。与前 4 个 action-head 模型（pi0.5 / GR00T N1.7 / OpenVLA-OFT / LingBot-VLA）本质不同：Cosmos Policy 是一个**生成式 video-diffusion 世界-动作模型**（predict-then-act）——当前图像、腕部图像、proprio、待预测的 **action chunk / 未来状态 / 标量 value** 全部编码为一段 latent 视频 `[B,16,T',H',W']` 里的 latent *帧*，由一个 **Cosmos-Predict2 2B DiT** 在 **frame-replace** 条件下做扩散去噪（EDM-sigma Karras 采样器 + rectified-flow 网络预条件）。同一个 DiT + 不同的 frame-replace mask 给出三种预测：policy（去噪 action 帧）、world-model（去噪未来状态帧）、**value（去噪 value 帧 → 标量）**——因此 **best-of-N planning 的 scorer 就是 value 帧本身**，与动作在同一次扩散中联合产生。

本提案属**引擎可扩展性验证 + policy 层**：落地 `embodiinfer/policies/cosmos/`（自持 DiT、vendored VAE leaf、扩散采样器、CosmosPolicy），并引入第 4 类 `ActionDecoder`（`CosmosDiffusionDecoder`）；引擎（批处理、图捕获、rollout 面）零改动。**方法学与前 4 个一致**（模块加载、forward 自持、vendored leaf、逐 op parity），但**非 RL rollout track**——RLinf 无 cosmos 生成器，故交付是 **planning 演示 + 对齐 cosmos-policy 自身推理的 parity**，不是 RL 端到端（若后续要 RL，另 scope）。

## 2. 动机与现状差距

- DESIGN §7.2 已 scope WAM / best-of-N planning 的接入点（`best_of_n(observations, num_samples, scorer)` + scorer = value/future-state head），但此前 `cosmos/` 仅为接口模板。本提案兑现该接口的**首个真实实例**。
- Cosmos Policy 的推理结构与 EmbodiInfer 两阶段（`encode_prefix` 一次 + 去噪循环 N 次）天然契合：VAE-encode 图像 + 注入 proprio + build 条件 + T5 cross-attn context 是**跨去噪步、跨 best-of-N 候选恒定的 prefix**（§4.3 复用红利），去噪循环是 DiT 的 5 步 EDM 扩散。
- 引入第一个 diffusion（非 flow、非 single-pass、非自回归）解码范式，进一步验证 `ActionDecoder` 抽象对新范式的覆盖（第 4 类，继 Flow / Parallel / Autoregressive）。

## 3. 目标与非目标

**目标**
- 自持 Cosmos-Predict2 2B DiT 前向（`models/video_dit/cosmos_predict2.py`，`CosmosPredict2DiT`，`video_dit` 类别的一个 family）：patchify、per-frame AdaLN-LoRA modulation、joint 3D self-attention（QK-norm + 3D RoPE）、text cross-attention、GELU MLP、final unpatchify，全部经 EmbodiInfer `AttentionBackend`；无 TransformerEngine / megatron 依赖。
- vendored Wan2.1 VAE leaf（`models/video_vae/wan.py`，`WanVAE`）：causal-3D-conv AE，encode 图像 → latent 帧（decode 仅用于未来图像可视化）。同 pi0.5 SigLIP / OFT timm / GR00T Qwen3-VL vision 的 leaf 定位。
- 自持扩散采样器（`schedulers/diffusion.py`）：rectified-flow 预条件 + Karras `get_rev_ts` + 2ab multistep + sample_clean。
- `CosmosPolicy(VLAPolicy)`（`modeling_cosmos.py`）：`encode_prefix`（VAE + 注入 + mask + text）、`denoise`（frame-replace + 预条件 + DiT）、`read_action` / `read_value`、`best_of_n_plan`（WAM planning）。
- 注册 `@register_policy("cosmos")`、`[cosmos]` extra、`cosmos` pytest mark、parity 测试。

**非目标（如实分离）**
- **RL 端到端**：RLinf 无 cosmos 生成器 → 无 ratio-at-θ0 / GRPO 对齐；planning + parity 是交付边界。
- **完整 planning 机制**：world-model rollout（未来态推进）、V/Q 分步、tree search、value ensemble（lcb / gamma-weighted）——cosmos-policy 有但 LIBERO 默认不启用；先做 base 联合去噪 value 的 best-of-N（shipped LIBERO 路径），完整 planning 后续。
- **RoboCasa / ALOHA 变体、planning-model checkpoint**：先做 LIBERO Predict2-2B base。
- **VAE decode 的正确性 parity**：仅 encode 走 parity（action/value 从 latent 帧直读、不需 decode）；decode 作可视化 leaf。

## 4. 设计

### 4.1 两阶段映射
- **`encode_prefix`（每观测一次）**：Wan2.1 VAE encode 33-帧像素序列 → 9 个 latent 帧；把 proprio tile-fill 注入 frame 1；build `condition_video_input_mask`（前 `num_conditional_frames=4` 帧为条件）；携带 T5 crossattn（`[B,512,1024]`）。产物 `CosmosPrefix{gt_frames, mask, crossattn, padding}`，**跨去噪步 / 跨 N 候选恒定**。
- **`CosmosDiffusionDecoder`（第 4 类 ActionDecoder）**：`init_state` 播种 `x_sigma_max = noise·sigma_max`；`produce_chunk` 跑 `sample_2ab`（Karras + 2ab + sample_clean，5 步），每步调 `policy.denoise`（rectified-flow 预条件 + frame-replace + 自持 DiT `x0`-预测）；读 action 帧。
- **读出**：action 帧(4) 对 112 个 tile 取平均 → `[B,16,7]`；value 帧(8) 整帧取平均 → 标量。

### 4.2 为何是新 decoder 而非 FlowDecoder
Cosmos 是 sigma-空间 EDM 扩散 + 2ab multistep（**带前步记忆**）+ sample_clean + 网络预条件，非 pi0.5 式 `x + v·dt` 线性 Euler。强行套 FlowDecoder 无法逐位复刻 2ab。架构（DESIGN §7）明确支持多 `ActionDecoder`；扩散采样器是自然的第 4 类，引擎零改动、模型无关性不破。

### 4.3 best-of-N planning（WAM 卖点）
`best_of_n_plan(batch, num_samples)`：`encode_prefix` **一次** → `expand(N)` 广播 → N 条扩散轨迹（各一 seed）联合去噪 action + value 帧 → 读 N 个 value → argmax 选最优。对应 DESIGN §7.2 的 `best_of_n(scorer=value head)`。**prefix 复用红利** = compute-bound 的 VAE + text encode 只跑一次（§4.3）；**诚实边界**：DiT 每步仍全帧过（frame-replace 非 KV-cache），复用限于 encode 层、非 per-step KV。

## 5. 模型无关性判定
- 新增代码全部落 policy 层 `embodiinfer/policies/cosmos/`；引擎仅经公共协议（`encode_prefix` + `ActionDecoder`）驱动，未按模型名分支。
- DiT self-attn / cross-attn 经 `layers/attention.py` 既有 `AttentionBackend`（sdpa）。
- VAE 作 vendored leaf（跑一次 encode），不在优化环内 op-level 复刻——同 GR00T/OFT/pi0.5 的 vision leaf。

## 6. 无损性与精度判据（对齐 cosmos-policy 自身推理，非 RLinf）
口径：同权重、固定初始 noise `x_sigma_max`、byte-identical obs、bf16（权重即 bf16）。参照 = box `.venv`（cosmos-policy 原生 env）跑 `CosmosPolicyVideo2WorldModel.generate_samples_from_batch`（LIBERO Predict2-2B）预存 reference，embodiinfer_env 侧比对（跨环境 reference，同 GR00T/OFT/LingBot）。**实测（单 H200，bf16，num_steps=5）**：

| 通路 | max\|Δ\| | 说明 |
|---|---|---|
| 自持 DiT 前向（注入 net 输入） | 0.032（mean 3.4e-3，rel 0.8%） | bf16 交叉-kernel 噪声底；net 输出 mean/std 与原生逐位吻合 |
| Wan2.1 VAE encode（vs 原生 latent） | 0.024（mean 1.7e-3） | vendored VAE 复现原生 encode |
| 端到端 latent（VAE+采样器+frame-replace） | 0.024（mean 2.1e-3） | 全自持链路 |
| action 帧（normalized） | 1.9e-3 | 主信号精确匹配 |
| value 帧（raw mean） | 7e-4（0.4184 vs 0.4176） | value 通路精确 |

bf16 权重下无 bit-exact；残差在 bf16 交叉实现噪声底内（同 GR00T/LingBot 的 ~1.6e-2 量级、OFT 的 categorical bf16 底宽）。

## 7. 实现计划（已完成）
- 可复用组件（引擎无关、不 import policies/engine）上提为独立层：`embodiinfer/models/video_dit/{base,cosmos_predict2}.py`（`VideoDiT` 类别 + `CosmosPredict2DiT` family + `register_video_dit` 注册表）、`embodiinfer/models/video_vae/{base,wan}.py`（`VideoVAE` + `WanVAE`）、`embodiinfer/schedulers/diffusion.py`（扩散采样数学）。policy 层薄组合：`embodiinfer/policies/cosmos/{modeling_cosmos,processor_cosmos}.py`（`CosmosPolicy` 选/建 DiT+VAE+scheduler + frame-replace/planning）。`pyproject.toml` 加 `[cosmos]` extra（einops）+ `cosmos` mark；`embodiinfer/policies/__init__` 已导入注册。
- 分层依据（参照 Diffusers `models/schedulers/pipelines` 三分 + vLLM/vllm-omni：sampler 自成子系统非 layer、model 引擎无关、policy=pipeline 组合）：`models/`=纯网络（按 family 组织，不 import policies/engine）、`schedulers/`=无权重采样数学（`ActionDecoder` 驱动）、`policies/`=引擎契约 + 组合。前 4 个 action-head policy 为单消费者自持前向、不动（LeRobot 式"共享才上提"）。
- T5 text encoder（`models/text_encoders/t5.py`，`T5TextEncoder` leaf，默认 `google-t5/t5-11b`）：lazy-load，逐 op 复刻 cosmos `encode_prompts`（pad max_length=512 + 超长位置置 0）→ `[B,512,1024]`。cosmos 侧 `CosmosTextEmbedder`：预计算 pkl 快路径（40 个 LIBERO 任务，与 T5 输出一致）+ pkl 外指令 lazy T5 fallback + cache。online-vs-pkl GPU 对拍暂跳（t5-11b 45GB；pkl 已验、代码逐 op 复刻，对拍只再确认同一 T5）。
- 引擎集成：`CosmosPolicy.collate`（`Observation`→`CosmosBatch`：images[0]=wrist/[1]=primary [0,1]→[-1,1]、blank→-1；proprio rescale；instruction→T5 crossattn）+ `pad`（bucket 复用）；`produce_chunk` 返回 dataset-scale 动作（`unnormalize_actions`）；`_build_cosmos` 加载 `dataset_statistics.json` + `t5_embeddings.pkl`。**box 验证 `EmbodiInfer("cosmos").act(obs)` 端到端跑通**（collate→encode_prefix→扩散去噪→`ActionChunk (16,7)`，batched 亦 OK；文本走 pkl 快路径）。LIBERO 精确图像预处理（flip/JPEG-q95/center-crop）为 serving 前端步骤、假定在 Observation 上游完成（同 pi0.5 tokenize 注记）。
- CUDA-graph：`CosmosDiffusionDecoder` 经 `_CosmosDenoiseGraph` 捕获 per-step `denoise`（预条件+DiT+frame-replace），static prefix 拷入一次、每步 replay（`allocate_static_prefix`/`copy_prefix_into`），opt-in `policy.use_cuda_graph`。**box：graph vs eager latent `max|Δ|=0`（逐位一致）、action `max|Δ|=0`；sample_latent(5 步) 157.6→138.2ms=1.14×**（DiT 2B compute-bound，收益在 per-step launch；2ab host 侧 float64 步不入图故稀释，单 DiT-forward microbench 1.36×）。
- checkpoint：`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`（DiT `.pt`，`net.*` bf16）+ `Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth`（Wan2.1 VAE）+ `libero_t5_embeddings.pkl`（预计算 T5，推理无需 text encoder）+ `libero_dataset_statistics.json`（归一化）。
- 蓝图：按源码逐 op 核实。

## 8. 测试计划
- **CI（CPU）**：注册 + 缺 checkpoint 抛 `ValueError`；latent inject/readout round-trip；rectified-flow scaling / Karras schedule 纯函数（无权重）。
- **box（`cosmos` mark，门控 `EMBODIINFER_COSMOS_CKPT` / `EMBODIINFER_COSMOS_VAE` / `EMBODIINFER_COSMOS_REF`）**：读预存 reference，跑 EmbodiInfer 自持 encode_prefix + 采样器，断言 action/value < 5e-2（bf16 底）。

## 9. 基准（planning 演示，单 H200，bf16，N 候选）
- prefix encode（VAE+text，一次）**68.9 ms**；单候选采样（5 步 DiT）**159.6 ms**。
- planning 延迟：N=1/2/4/8 = **229 / 350 / 580 / 1032 ms**（每候选一条 5 步 DiT 轨迹 + 固定 prefix）。
- **prefix 复用增量（N=4）**：复用（encode 1×）**579 ms** vs 逐候选重编码（encode 4×）**909 ms** → **省 330 ms（~36%）**（§4.3；量级随 N 增大而增）。
- **CUDA-graph 增量（DiT 前向、静态形状、小 batch）**：eager 37.3 ms vs graph 27.5 ms = **1.36×**（launch-bound 区收益；随 batch / compute 增大收敛于 1×——去噪循环全捕获进 `ForwardGraph`/`LoopGraph` 的引擎级接入为后续）。

## 10. 风险与局限
- **bf16 底噪**：权重 bf16 → 无 bit-exact；action/value 残差在噪声底内，planning 的 value 排序对小残差鲁棒（观测：best-of-4 value 排序稳定）。
- **prefix 复用边界**：DiT 全帧双向注意力、无 clean-帧 KV-cache → 复用限 encode 层（诚实标注）。
- **cudagraph 引擎级接入**：当前 microbench 证 DiT 前向 1.36×；去噪循环（含 2ab 的 host 侧 float64 算术）全捕获进引擎为后续。
- **VAE 流式 causal-conv**：33 帧分块 encode 的 feat-cache 逻辑 vendored 自 reference；encode parity 已验，decode 仅可视化未逐 op parity。
- **serving 前端边界**：Cosmos 吃 raw 图像 + T5 embedding（非 tokenized `Observation`）→ `collate` 抛显式错误，需 `CosmosProcessor` 或前端提供 T5 embedding（同 pi0.5 的 state-injection 前端注记）。

## 11. Open（未判死）
1. 去噪循环全捕获进 `ForwardGraph`/`LoopGraph`（引擎级 cudagraph）的收益与消失点。
2. 完整 planning（world-model rollout / V,Q 分步 / tree search / value ensemble）的接入与任务指标。
3. RoboCasa / ALOHA 变体、planning-model checkpoint。
4. bf16 残差是否可经 fp32 solver 算术收紧（当前 solver float64、net bf16；net 是残差主源）。
5. RL track（若 RLinf 后续出 cosmos 生成器）：ratio-at-θ0 + GRPO 对齐，另 scope。
