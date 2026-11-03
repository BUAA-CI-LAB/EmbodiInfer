# 0002 — GR00T N1.7 作为第二个一等 policy

- 状态：Draft（sdpa 路径 + 跨环境 ground-truth parity 已达成，见 §6/§11）
- 日期：2026-07-06

## 1. 摘要

将 NVIDIA Isaac-GR00T（当前 `main` 为 **N1.7**）接为 EmbodiInfer 的第二个一等 `VLAPolicy`。GR00T 是「Qwen3-VL（Cosmos-Reason2-2B）VLM backbone + flow-matching **DiT** action head」结构，与 EmbodiInfer 两阶段（`encode_prefix` 一次 / `denoise_step` N 次）天然对应。本提案属 **policy 层**（新增 `embodiinfer/policies/gr00t/`），并触发一处 **layers 层** 的后续扩展点（新增 `flash` attention 后端）。预期收益方向有二：(1) 验证引擎的模型无关性——第二个 flow policy 复用既有 cudagraph / `execute_pipelined` / async 调度；(2) 提供一个「标准 flash-兼容 attention、无自定义 mask」的对照，用以量化「以 sdpa 为基线、更专门的 fused kernel 能否再快」，据此反推 pi0.5 上 FlexAttention 的价值边界。

本提案范围先限定为 **sdpa 路径跑通 + parity**；flash 后端、cudagraph capability、引擎全链路 bench 为后续增量。

**内集成（zero gr00t 运行时依赖）**：不 import `gr00t` 包运行。action head 建模 vendored 进 `embodiinfer/policies/gr00t/modules_gr00t.py`（逐行照 gr00t `dit.py` + `embodiment_conditioned_mlp.py`，Apache-2.0）；backbone 用 transformers 官方 `Qwen3VLForConditionalGeneration`（gr00t 内部也是它），从 Cosmos-Reason2-2B config build、从 GR00T safetensors strict-load `backbone.model.*`；权重、DiT 去噪循环均 EmbodiInfer 自持。运行时依赖收敛到 torch + transformers + diffusers + safetensors，GR00T 复用 pi0.5 所在的单一 EmbodiInfer env（transformers>=5），不为其单独建生产 env。

## 2. 动机与现状差距

- EmbodiInfer 目前唯一的真权重一等 policy 是 pi0.5（`embodiinfer/policies/pi05/modeling_pi05.py`）。引擎的模型无关性主张（见 `CONTRIBUTING.md` 的 Engineering principles）尚缺第二个独立模型的实证。
- pi0.5 的 attention 是 big_vision prefix-LM 的自定义 2D block mask，`sdpa` 在该 mask 下未必命中 flash 快路径；是否值得为其实现 FlexAttention（`embodiinfer/layers/attention.py` 的 `flex` stub）尚无对照数据（开放问题）。
- GR00T 的 DiT 使用 diffusers 的 `Attention`（`AttnProcessor2_0` → `F.scaled_dot_product_attention`），主 `DiT.forward` 不传任何 mask（`Isaac-GR00T/gr00t/model/modules/dit.py:292-336`）；`AlternateVLDiT` 仅在 cross-attention 传一个 image/text 的 **key-padding** 布尔 mask（`dit.py:349-414`）。二者都非 pi0.5 式的 query×key 2D block mask，故 GR00T 是一个干净的 sdpa 原生对照。

## 3. 目标与非目标

**目标**
- 新增 `embodiinfer/policies/gr00t/`：`encode_prefix` / `denoise_step` / `flow_schedule` / `pad`，`denoise_step` 为 **EmbodiInfer 自持**（DiT 循环重写，attention 走 `AttentionBackend`）。
- attention 先支持 **sdpa**；parity 锚点为 sdpa（vs 官方 `get_action`）。
- 注册 `@register_policy("gr00t")`、`[gr00t]` extra、`gr00t` pytest mark、parity 测试。

**非目标（后续增量）**
- flash / eager 的性能对比与 `flash` 后端实现（§9 记录框架，代码分离提交）。
- cudagraph capability（`supports_cuda_graph` 先为 `False`；DiT 循环形状静态、具备可捕获性，但需先在 box 上验证 sdpa 正确性 + 处理 Qwen3-VL 打包视觉输入的 bucket padding）。
- cross-attn KV 缓存（`to_k/to_v(vl_embeds)` 跨步恒定，可缓存；MVP 先每步重算以与官方逐位对齐，缓存作无损优化后加）。
- GR00T 原生 `collate`（从裸 `Observation` 走官方 Qwen3-VL processor）与 RTC（real-time chunking）inpainting。
- backbone 前向自持（见 §5，backbone 作黑盒 prefill）。

## 4. 设计

### 4.1 两阶段映射（对照官方推理路径）

官方参考路径：`Gr00tN1d7.get_action` → `backbone(...)`（VLM，一次）→ `action_head.get_action` → `_encode_features`（一次）+ `get_action_with_features`（去噪循环 N 次），见 `Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py:325-475`。

- **`encode_prefix`（每观测一次）**：
  1. 官方 Qwen3-VL backbone 前向 → `backbone_features [B,S,2048]`、`backbone_attention_mask`、`image_mask`（`qwen3_backbone.py:278-293`）。gr00t 取 `hidden_states[-1]`，即 pop 到 `select_layer` 层后的 **pre-final-norm** 残差流；transformers 4.57（gr00t 的 pin）在该位置返回 pre-norm 张量，而 transformers>=5 返回 **post-norm** 张量。因 backbone 只取 hidden state、不用 logits，`Gr00tPolicy.__init__` 在 strict-load 之后把已成 vestigial 的末层 text norm 换成 `nn.Identity()`，恢复 gr00t 训练所用的 pre-norm 特征，使 `encode_prefix` 对 transformers 版本无关（该 pre/post-norm 差异由 box ground-truth 对比查出，见 §6/§11）。**另一处跨版本对齐 = 多模态 RoPE 位置**：tf>=5 的 Qwen3-VL 需 `mm_token_type_ids` 标记图像 token 才会给它们 2D（temporal/height/width）网格位置，GR00T 以最小输入集手动驱动 backbone 时缺该标记 → 退化成 1D 顺序位置（vision token RoPE 旋转错误）；`encode_prefix` 显式构造 `mm_token_type_ids`（image→1）调 `get_rope_index` 算出正确 2D 位置传入（与 gr00t tf4.57 位置 $\max\lvert\Delta\rvert=0$）；
  2. `vlln`（LayerNorm 2048）+ `vl_self_attention`（默认 `Identity`）→ `vl_embeds`；
  3. `state_encoder(state, embodiment_id)` → `state_features [B,1,1536]`（per-embodiment 权重）；
  4. 预计算 `AlternateVLDiT` 的两个 cross-attn key-padding mask（image / text，跨步恒定）。
  产物 `Gr00tPrefix{vl_embeds, image_key_mask, text_key_mask, state_features, embodiment_id}`。
- **`denoise_step`（N 次）**：`action_encoder(x_t, t_disc, emb) [+ pos_embed]` → `sa_embs = cat([state_features, action_features])` → **EmbodiInfer 自持 DiT 循环**（16 层：偶数层 cross-attn 到 `vl_embeds`、奇数层 self-attn；`interleave_self_attention=True`）→ AdaLN 输出头 → `action_decoder` → 取 `[:, -action_horizon:]` 为速度场 $v(x_t,t\mid\text{prefix})$。

### 4.2 flow schedule

GR00T 为标准 rectified-flow：$x(t)=(1-t)\,\epsilon + t\,a$，网络学 $v=a-\epsilon$，推理正向积分 $t:0\to1$，$dt=+1/N$，$N=4$（`num_inference_timesteps`）。这与 `VLAPolicy` 基类默认 `flow_schedule`（升序 $t=i/N$，$dt=+1/N$）**完全一致**，故 GR00T **不 override** `flow_schedule`（与 pi0.5 的 $1\to0$ 相反）。网络实际接收离散 bucket $t_{\text{disc}}=\lfloor t\cdot 1000\rfloor$，该转换在 `denoise_step` 内以 fp32 完成（$N=4$ 时 $t\in\{0,.25,.5,.75\}$ → bucket $\{0,250,500,750\}$ 精确）。

### 4.3 attention 子层（自持，走 backend）

复刻 diffusers `AttnProcessor2_0`：`to_q/to_k/to_v` → reshape `[B,heads,S,head_dim]` → `self._attn.attend(q,k,v,mask,scaling=None)`（scale $=head\_dim^{-1/2}$，GR00T 无 qk-norm、`upcast_attention=False`）→ `to_out[0]`(Linear)+`to_out[1]`(Dropout, eval 为恒等)。self-attn 无 mask；cross-attn 传 `[B,1,1,S]` 布尔 key-padding mask（`AlternateVLDiT` 按 block idx 在 image/text 间交替，`idx % (2·attend_text_every_n_blocks)==0` 取 text，否则 image）。norm/FeedForward 直接复用加载的官方 module（非 attention，无 delegation 顾虑，同 pi0.5 复用 mlp/rmsnorm）。

### 4.4 备选方案与取舍

- **备选 A：wrap 官方 `action_head.get_action`。** 取舍：会 delegate 去噪计算，`AttentionBackend` 变假、循环无法 cudagraph——违背引擎价值主张（与 pi0.5 的硬约束一致）。**否决**。
- **备选 B：连 backbone 也自持（重写 Qwen3-VL 前向）。** 取舍：GR00T 的 DiT 只 cross-attend backbone 的**输出特征**（非其内部 KV），backbone 是「运行一次」的 prefill；重写 Qwen3-VL 2B 前向收益极低（不在被优化的内循环里）、风险与工作量极高。**采用黑盒 backbone**：与 pi0.5 引擎 prefill 只跑一次同构，边界清晰。
- **备选 C：目标 N1.5（Eagle backbone）。** 取舍：官方 `main` 仅含 N1.7 建模代码（`AutoModel.register(Gr00tN1d7Config, ...)`），N1.5/N1.6 为 Eagle backbone、需各自旧代码，无法被本仓 `AutoModel.from_pretrained` 加载。**取 N1.7**（最新稳定 + 有公开 PyTorch 权重 `nvidia/GR00T-N1.7-3B`，safetensors ≈6.9GB）。

## 5. 模型无关性判定

- 新增代码全部落在 **policy 层** `embodiinfer/policies/gr00t/`，实现基类协议，未改引擎。引擎（`engine/core.py`）仅通过公共协议（`encode_prefix`/`denoise_step`/`flow_schedule`/`pad`/`supports_cuda_graph`）驱动 GR00T，未按模型名分支。
- `denoise_step` 的 attention 经 `layers/attention.py` 的既有 `AttentionBackend` registry 获取，符合 §2.2 的 Protocol + registry 扩展模式。
- backbone 作黑盒 prefill：其内部 attention 后端（FA2/SDPA，由 gr00t 的 `use_flash_attention` 决定）不在 EmbodiInfer 自持范围内，属「运行一次」的模型特异预填；不构成引擎对模型的隐含依赖。

## 6. 无损性与精度判据

- **对照对象**：官方 `Gr00tN1d7ActionHead.get_action_with_features` 的逐步计算（注入固定初始噪声 $x_0$ 以消除其内部 `torch.randn`）。因 `gr00t` 包 pin transformers 4.57 且拉训练栈、与 EmbodiInfer env（transformers>=5）冲突，无法在 EmbodiInfer env 内 in-process import：由独立 `gr00t_env`（tf4.57）的 box 脚本 `gr00t_ref_run.py` 预跑一次、`torch.save` reference（inputs + $x_0$ + `ref_actions`，可选中间量 `vl_embeds`/`state_features`），EmbodiInfer env 侧读取比对。
- **两级判据**（口径：同 dtype bf16、固定 $x_0$、byte-identical obs inputs）：
  1. **DiT/action-head 隔离**：把 gr00t-native 自己的 prefix（`vl_embeds` + `state_features`）直接注入 EmbodiInfer 去噪循环 → $\max\lVert\Delta a\rVert_\infty \approx 1.6\times10^{-2}$（rel 0.4%），即给定相同 prefix 时 EmbodiInfer 自持 DiT 在 bf16 级忠实复现 gr00t-native（同 env 内 embodiinfer-sdpa vs vendored diffusers 路径为 $\max\lvert\Delta\rvert=0$，bit-exact）。
  2. **端到端**：EmbodiInfer 全链路（tf>=5 backbone + 自持 DiT）vs gr00t-native（tf4.57）→ $\max\lVert\Delta a\rVert_\infty \approx 1.6\times10^{-2}$（rel 0.4%）。修好 §4.1 的两处 tf 跨版本对齐（pre-norm 特征 + mrope 2D 位置）后，backbone 复现 gr00t select-layer 特征（`vl_embeds` cos ~0.99），端到端 delta **收敛到与判据 1 相等的 DiT 底噪**（backbone 贡献已可忽略），残余即 torch/diffusers 跨版本的 bf16 算术差异（非 bit-exact）。
- **复现**：`tests/test_gr00t_parity.py`（`gr00t` mark，门控 `EMBODIINFER_GR00T_CKPT` + `EMBODIINFER_COSMOS_PATH` + `EMBODIINFER_GR00T_REF`）+ box 脚本 `dev/scripts/{gr00t_ref_run,gr00t_compare,gr00t_inject}.py`。

## 7. 实现计划

- 新增：`embodiinfer/policies/gr00t/{__init__,modeling_gr00t,processor_gr00t}.py`；`pyproject.toml` 加 `[gr00t]` extra 与 `gr00t` mark；`embodiinfer/policies/__init__.py` 导入注册。
- `supports_cuda_graph=True` + `allocate_static_prefix`/`copy_prefix_into` 已实现（`Gr00tPrefix` 静态张量）→ 免费继承引擎 cudagraph，实测见 §9.1。
- 向后兼容：纯新增，不改公共 API。

## 8. 测试计划

- **CI（CPU）**：`test_gr00t_registered`——`gr00t` 已注册、缺 `checkpoint` 或缺 `cosmos_path` 均抛 `ValueError`（不依赖 gr00t 包 / 权重）。
- **box（gr00t mark）**：`test_gr00t_matches_native_reference`——读预存 reference，跑 EmbodiInfer 内集成 policy（sdpa），断言端到端 $\max|\Delta a| < 6\times10^{-2}$ 与注入-prefix DiT $< 3\times10^{-2}$。门控 `EMBODIINFER_GR00T_CKPT` + `EMBODIINFER_COSMOS_PATH` + `EMBODIINFER_GR00T_REF`。
- **env（分环境产 reference，统一 env 跑 EmbodiInfer）**：`gr00t` 包 pin `transformers==4.57` 且拉训练栈，与 EmbodiInfer env（transformers>=5）冲突，故不在同进程比对。
  - reference 侧：独立 `gr00t_env`（tf4.57，`gr00t` `pip install --no-deps`）跑 `gr00t_ref_run.py`。其 import 链拉训练 pipeline（`gr00t/model/__init__.py` → `setup` → `gr00t.data.*` → pandas/albumentations…），故 patch site-packages 的两处推理无关 import（注释 `model/__init__` 的 `Gr00tN1d7Pipeline`、把 `gr00t_n1d7.__init__` 里的数据侧 `Gr00tN1d7DataCollator` 构造改为 `self.collator = None`），只留 `Gr00tN1d7` model + config 可 import；backbone 的 `model_name`（gated `nvidia/Cosmos-Reason2-2B`）指向本地 Cosmos 目录的符号链接以离线加载。
  - EmbodiInfer 侧：统一 `embodiinfer_env`（editable-install，transformers>=5），读 reference 跑 `Gr00tPolicy`。

## 9. 基准计划

**修正**：不以 eager 作性能对比（对 GR00T「用更慢的比更快的」无意义，且 GR00T 参考本身即 sdpa）。以 **sdpa 为基线**，测更专门的 fused kernel 能否再快。

| backend | 角色 |
|---|---|
| `eager` / `eager_bc` | 仅 correctness 参考（fp32 softmax 可读基线） |
| `sdpa` | **基线 + parity 锚点** |
| `flash`（后续新增，调 `flash_attn` 库） | sdpa 之上的性能候选 |
| `flex` | 留给 pi0.5（带 2D block mask 编译 fused），GR00T 无 mask 用不上 |

- 条件：bf16、$N=4$、$B\in\{1,8,16,32\}$、box(H200)。分别测 DiT **self-attn（无 mask，纯 flash 场景）** 与 **cross-attn（image/text key-padding mask，masked 场景）**。
- 反推逻辑：若 no-mask 场景 `sdpa≈flash`（增量小），说明 SDPA 已充分利用 flash，attention kernel 非该场景杠杆；则 pi0.5 的机会精确定位为「2D block mask 使 sdpa 掉出快路径」，FlexAttention 把带 mask 的 attention 编译成 fused 才是其特有价值。若 `flash` 显著优于 `sdpa`，则通用 flash 后端对 GR00T/pi0.5 均值得。
- 另测引擎收益：`execute` / `execute_pipelined` / cudagraph 在 GR00T 上免费继承（验证模型无关性）。

### 9.1 引擎 cudagraph 继承实测（box H200，sdpa/bf16/$N=4$）

`Gr00tPolicy` 实现 `supports_cuda_graph=True` + `allocate_static_prefix` + `copy_prefix_into`（`Gr00tPrefix` 静态张量），引擎 `GraphManager`/`DenoiseGraph`/`LoopGraph`（原封未改）即捕获 GR00T 去噪循环 —— **第二个一等 policy 未改一行引擎继承 cudagraph**，模型无关优化路径的实证。DiT 的正弦编码含 `torch.tensor(10000.0)` 但捕获通过、无需 graph-safe 改写。

**去噪循环 eager vs graph**（`prefix.expand(B)` 隔离去噪计时；`bench_gr00t_cudagraph.py`）：全 batch **bit-exact $\max\lvert\Delta a\rvert=0$**（per-step 与 full-loop 均是）；加速比 —— B=1 **3.30×**（24.5→80.8 obs/s）、B=2 3.15×、B=4 2.97×、B=8 2.35×、B=16 1.92×、B=32 1.31×。eager 全 batch 持平 ~39.5ms（纯 launch 开销：DiT 32 blocks × N 步 × 每步极轻），graph 才随真实 compute 缩放；per-step≈full-loop（步间 host 可忽略，同 pi0.5 结论）。加速比大于 pi0.5（B=1 2.29×）—— DiT 更 launch-bound。

**EngineCore 端到端**（B=1，backbone prefill + 去噪；`bench_gr00t_engine.py`）：backbone 25.1ms + 去噪 → eager 66.6ms（15.0 obs/s）vs graph 37.1ms（27.0 obs/s）= **1.79×**（去噪占 eager 延迟 ~62%，backbone 为剩余固定成本），**完整引擎链路 parity $\max\lvert\Delta a\rvert=0$**。GR00T 未改引擎跑通 `pad`/`encode_prefix`/graph-integrate/`pack`。

**局限**：多 env batched 吞吐（B 个不同图像）需 packed-vision `collate`/`pad`（§5 TODO）；attention backend `eager`/`sdpa`/`flash` 对照（§9 主表，反推 pi0.5 FlexAttention）尚未实测。

## 10. 风险与局限

- **env 冲突**：需独立 `gr00t_env`；隧道连接可能间歇性中断。
- **权重与 backbone 下载**：`nvidia/GR00T-N1.7-3B`(≈6.9GB) + backbone `nvidia/Cosmos-Reason2-2B`，首次加载体量大。
- **cross-attn mask 常数**：diffusers 对布尔 mask → SDPA 的处理若用 `finfo.min` 而非 `-inf`，可能与自持路径有微小差异；box parity 若非 bit-exact 再对齐 mask 常数。
- **collate / packed vision**：Qwen3-VL 变长 `pixel_values`/`image_grid_thw` 的 batch pad 与 cudagraph bucket 需 box 验证；MVP 的 `collate` 未接裸 `Observation`（走官方 processor 边界）。
- **embodiment_id**：GR00T 特异 metadata，选 per-embodiment 投影器权重，EmbodiInfer `Observation` 无此字段，须由上游按 EmbodimentTag 提供，否则选错权重。
- **RoPE inv_freq**：backbone 换 attn 后端时 Qwen3-VL 非持久 RoPE buffer 需按解析公式重建（`qwen3_backbone.py:210-273`），否则 FA2/SDPA 数值漂移——backbone 作黑盒时沿用官方逻辑规避。

## 11. 结果（box 实测，H200 / bf16）

三个已验证维度（脚本在 `dev/scripts/`）：

1. **vendored action head STRICT match**（`gr00t_ah_check.py`）：537 params、0 missing / 0 unexpected、forward OK。
2. **backbone build+load**（`gr00t_backbone_probe.py`）：官方 Qwen3-VL load GR00T `backbone.model.*` 494 params 0/0；标准 load 的 RoPE `inv_freq` 与 gr00t reset 值 delta ≈9.3e-10。
3. **端到端 sdpa 同 env bit-exact**（`gr00t_e2e_parity.py`）：真图像+真权重，EmbodiInfer 自持 backend(sdpa) vs vendored diffusers 路径 $\max\lvert\Delta\rvert=0$（`eager` 4.7e-2）。

**跨环境 ground-truth parity**（gr00t-native tf4.57 vs EmbodiInfer 内集成 tf>=5，`gr00t_ref_run.py` + `gr00t_compare.py` + `gr00t_inject.py`）：

- 初测端到端 $\max\lvert\Delta a\rvert=6.6\times10^{-2}$（rel 1.7%）。**逐层 hidden state 分解**（`gr00t_compare.py` 的 per-layer cosine）定位到：L00（embedding）cos 0.999、L01–L15 cos 0.95–0.99，而 **L16 骤降到 cos 0.62** 且出现 ~1.5e4 的 massive-activation 通道。`state_features` 则 $\max\lvert\Delta\rvert=0$（bit-exact）。
- **归因**：pre/post-final-norm 约定跨 transformers 版本不同——gr00t 取 `hidden_states[-1]` 为 **pre-norm** 末层残差（tf4.57），EmbodiInfer 在 tf>=5 拿到的 `hidden_states[select_layer]` 是 **post-norm**（forward-hook probe：`hs16 vs post-norm cos 1.0`、`ref vs pre-norm cos 0.981`）。这是内集成的真实缺陷，被 ground-truth 对比查出（1.7% 的 action 吻合系 DiT 对该扰动鲁棒的巧合，非正确性证明）。
- **修复**：strict-load 后把 backbone 末层 text norm 换 `nn.Identity()`，恢复 pre-norm 特征（`modeling_gr00t.py`）。修复后 L16 cos 0.62→0.981、`vl_embeds` cos 0.55→0.94、端到端 $\max\lvert\Delta a\rvert$ 6.6e-2→**3.1e-2**（rel 0.8%）。
- **修复后进一步溯源（L01 单层 cos_min 0.77 异常）**：norm 修复后端到端仍 3.1e-2，逐层分解发现发散**完全集中在 image token**（L0 embedding 处 text token cos=1.0000 bit-identical、image token 0.9986；逐层最差 token 全是 image），且 EmbodiInfer 自身 fp32-vs-bf16（同 tf>=5）逐层持平不 compound → 判定为**算法差异非 bf16**。hook `rotary_emb` 抓 `position_ids`：gr00t（tf4.57）image token 用**正确 2D mrope**（height/width 网格），EmbodiInfer（tf>=5）退化成 **1D 顺序**（`max\lvert\Delta\rvert=63`）。**根因**：tf>=5 的 `get_rope_index` 新增必填 `mm_token_type_ids`，GR00T 最小输入集未构造 → backbone forward 退化。
- **修复 2（mrope）**：`encode_prefix` 构造 `mm_token_type_ids`（image→1）+ `get_rope_index` 算 2D `position_ids` 显式传入（与 gr00t 位置 `max\lvert\Delta\rvert=0`）。修复后 `vl_embeds` cos 0.94→**0.99**、端到端 $\max\lvert\Delta a\rvert$ 3.1e-2→**1.6e-2**（rel 0.4%）。
- **隔离归因（终态）**：注入 gr00t-native prefix 进 EmbodiInfer DiT $=1.6\times10^{-2}$，端到端 $=1.6\times10^{-2}$ —— **两者相等**，即 backbone 对齐后贡献可忽略，端到端 delta 收敛到 DiT/action-head 的 bf16 底噪（diffusers 0.35.1 vs 0.35.2、torch 2.7.1 vs tf>=5 env）。两个 bug（pre/post-norm、mrope）此前均被 DiT 鲁棒性掩盖（两 bug 并存时 action 仅偏 1.7%），由 ground-truth 逐层对比逐一查出。

**Open（未判死，待查）**：更多 obs（多相机、真实场景 prompt、含 video token）下端到端 bound 的分布，当前数字基于单条固定 reference；action-level best-of-N / cudagraph 继承等引擎增量（§5 非目标，后续）。

GR00T 的图缓存按 live prefix 的实际 token 长度区分 variant；静态 buffer 从待捕获请求分配，避免下一次 prefill 改写缓存长度后影响前一请求。不同任务的指令保持原生未填充输入，正式 benchmark 在计时前覆盖全部图形状。
