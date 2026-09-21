# 支持的模型

通过 Python 引擎或服务端的 `--policy` 参数选择模型。
加载前，准备对应检查点并安装匹配的依赖环境。

## 模型与检查点

| policy | 权重 | 说明 |
|---|---|---|
| `pi05` | `lerobot/pi05_base` | 流匹配；内置 Gemma 前向实现；FP32 输出与 LeRobot 逐位一致 |
| `dm05` | 用户提供的 DM0.5 检查点和归一化统计量 | 需要单独准备的 OpenDM 环境；参见下文的输入与服务说明 |
| `gr00t` | NVIDIA GR00T N1.7 | 流匹配；内置 DiT 动作头与原版 Qwen3-VL 主干；已与 Isaac-GR00T 做数值一致性验证 |
| `openvla_oft` | `Haozhan72/Openvla-oft-SFT-*` | 非流匹配：单次前向与动作 token 分类头；内置 Llama-2、timm 视觉编码器和 KV 分离解码图 |
| `lingbot_vla` | `robbyant/lingbot-vla-4b` | pi0 风格流匹配；Qwen2.5-VL 与窄 Qwen2 MoT 专家模块；内置 MoT 前向实现 |
| `cosmos` | `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` | **WAM**：视频扩散 DiT 联合去噪动作、未来状态和价值的潜变量帧；内置 DiT、EDM 采样器和 Wan2.1 VAE；支持模型专用的 best-of-N 规划 |
| `activevln` | `Arvil/Qwen2.5-VL-3B_rl_r2r_4000` | **实验性 R2R 适配器**：KV 缓存在任务内持续增长，以 eager 自回归方式解码动作 token；具备 CPU、会话测试及固定版本的一致性检查工具，真实权重的 GPU 一致性验证尚未完成 |
| `qwen2.5-vl-3b-r2r-low-level` | `Vebbern/Qwen2.5-VL-3B-R2R-low-level` | 有状态第一人称 R2R 策略；输出 `Left`/`Right`/`Move`/`Stop`；原生前向支持 CUDA Graph |
| `qwen2.5-vl-3b-r2r-panoramic` | `Vebbern/Qwen2.5-VL-3B-R2R-panoramic` | 有状态全景 + 候选视角选择器；候选元数据位于 `Observation.metadata` |
| `navida` | `waynechu/NaVIDA` | Qwen2.5-VL-3B；沿用官方 v2 的 JPEG 处理、历史管理和生成规则；最多六个原子动作；支持 eager 或基于 StaticCache 的手动 CUDA Graph 解码 |
| `streamvln` | 已发布的本地检查点布局 | 有状态，同一 episode 固定在同一副本上；在 32 步 fast KV 窗口和八特征 slow-memory 前缀上进行 SlowFast 帧选择 |

## 功能支持与运行环境 {#capabilities-and-installation}

HTTP 与 WirelessComm 启动器使用相同的服务适配器。π0.5 支持
通过 `--max-batch` 进行跨会话批处理；DM0.5 和 StreamVLN 目前一次
只服务一个请求。没有网络适配器的策略使用 Python API。

| Policy | 环境 | HTTP | Python 批处理 / 有状态 | CUDA graphs | 通用 RL 解码器 |
|---|---|---|---|---|---|
| π0.5 | `pi05` | 是 | 无状态批处理 | 解码；可选原生前缀 | 是 |
| DM0.5 | 外部 OpenDM 环境 | 是 | 模型特定；HTTP 保留历史 | 解码 | 是 |
| GR00T | 基准测试专用环境；没有专用 uv group | 否 | 无状态批处理 | 解码；可选原生前缀 | 是 |
| OpenVLA-OFT | `openvla-oft` | 否 | 无状态批处理 | 单次 forward 解码 | 是 |
| LingBot-VLA | `lingbot-vla` | 否 | 无状态批处理 | 解码 | 是 |
| Cosmos | `cosmos` | 否 | 模型特定的候选规划 | 扩散步 | 否 |
| ActiveVLN | `activevln` | 否 | 显式有状态会话，B=1 | eager 路径 | 否 |
| Qwen R2R low / panoramic、NaViDA | `qwen25-vln` | 否 | 显式有状态会话，B=1 | 策略原生解码 | 否 |
| StreamVLN | `streamvln` | 是 | 显式有状态会话，B=1 | 策略原生解码 | 否 |

对于 GR00T，请遵循其
[benchmark 指南](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md)中的环境与资产准备步骤。
其他依赖组见[安装指南](installation.md)。下文分别说明各模型的输入格式和预处理要求，
相机、状态向量和 token 排列需与检查点匹配。CUDA Graph 还要求硬件和输入形状兼容。

## 推理优化选项 {#optimization-profiles}

EmbodiInfer 提供模型级优化和通用算子后端，可按模型与 GPU 选择：

- **CUDA Graphs** 捕获并重放静态计算，按策略支持前缀编码、单步解码或完整去噪循环。
- **torch.compile / Inductor** 编译选定的模型路径，包括原生
  π0.5 和 GR00T 计算以及 StreamVLN 语言 prefill。
- **Triton 算子** 提供专用注意力、融合归一化、旋转嵌入和门控激活。
  π0.5 原生实现将这些算子与去噪过程中的前缀 KV 复用结合使用。
- **低精度线性层** 提供 FP8、INT8 和 NVFP4 后端。
  π0.5 和 StreamVLN 的量化基准测试使用以下组合：

| 硬件 | 基线精度 | 量化方案 |
|---|---|---|
| Jetson AGX Orin | BF16 | INT8 |
| RTX 4090 | BF16 | FP8 |
| Jetson AGX Thor | BF16 | FP8、NVFP4 |

这些配置以批次大小 1 运行，启用 CUDA Graphs，并在量化
执行时禁用 Inductor。在 RTX 4090 上，π0.5 使用 Triton weight-only FP8；
StreamVLN 使用原生 W8A8 FP8。Thor 使用原生 W8A8 FP8 和原生 NVFP4。
量化会改变模型数值：请结合延迟
测量结果以及与 BF16 的输出对比来选择方案。

检查点、GPU 软件、后端和数值对比的细节，请参见
[π0.5 精度基准测试](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-quant-benchmark/README.md)和
[StreamVLN 精度基准测试](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-quant-benchmark/README.md)。
[架构指南](architecture.md#cuda-graph-capture)介绍 CUDA Graph 捕获机制；
[多 GPU 并行](parallelism.md)介绍跨设备批处理和张量并行。

## 路线图 {#roadmap}

**SmolVLA** 和基础 **OpenVLA** 模型是计划新增的模型。OpenVLA-OFT
已经实现，并保持为独立的策略。

- [ ] 检查点加载与观测预处理
- [ ] 推理适配器与参考输出检查
- [ ] HTTP / WirelessComm 服务集成
- [ ] [EmbodiRun](https://github.com/BUAA-CI-LAB/EmbodiRun) 中的部署绑定

## 无需检查点的测试策略

`mock_flow_vla` 是无需检查点的合成策略，可在 CPU 或 CUDA 上测试引擎、批处理和 rollout 接口。

## 各模型的使用说明

### Pi0.5

π0.5 支持 Python 批处理以及跨会话的 HTTP/WirelessComm 批处理。
请使用
[适配器 JSON](serving.md#adapter-json)配置相机名称和状态字段，再通过 `--max-batch` 设置批次上限。

对于 PI0.5，`attention="eager"` 还会选择参考投影布局：各相机
视角单独编码，Q/K/V 与 gate/up 投影保持分离。
融合后端则保留批处理与融合路径。这一区别对
部分张量做过类型转换的检查点上的 rollout/actor 对数概率比较很重要：即使
注意力公式相同，改变 GEMM 形状也可能改变舍入。

### DM0.5

提供 DM0.5 检查点及其 `norm_stats.json`；参见
[`examples/dm05_arx5_inference.py`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/examples/dm05_arx5_inference.py)。标准
`embodiinfer-http-serve --policy dm05` 启动器会注册该模型；HTTP 服务
还要求在其 adapter JSON 中设置 `policy_kwargs.is_history: true`。

DM0.5 会把 `action_horizon` 和 `default_num_steps` 转发给 OpenDM loader，因此
LIBERO 实验可以使用其原生的 10 动作 chunk。`liger_kernel=True` 会启用
OpenDM 的融合推理模块；默认值为 `False`。LIBERO
benchmark 会检查所请求的融合模块是否确实已安装。

DM05 支持逐步和整循环的 CUDA Graph 解码。该 graph 存储
32 列内部状态，并为不同的前缀长度缓存各自的布局；
对外的 7/14 列切片和输出转换在重放之后进行。

DM0.5 使用共享的 `FlowDecoder` 和 `engine/rollout/logprob.py`。其策略
提供内部状态形状、对外动作转换和似然掩码；
填充列保留在轨迹中，但不计入 logprob。

DM0.5 接受共享的 `embodiinfer.Observation`：`images` 中的 RGB float 张量、一个状态
张量、`instruction` 中的原始文本，以及用于 `instruction_tokens` 的空 long 张量
（由 OpenDM 执行 tokenization）。可选的 `metadata` 键有 `robot_type`（默认
取策略设置）、`speed`（默认 `"0.5"`）、`control_mode`、`state_desc`、
`history_images` 和 `history_placeholder_text`。相机顺序遵循
检查点的 `image_prompts`。对于不同的相机尺寸，`processor_dm05.pack_images`
会返回一个填充后的张量以及用于 `metadata["image_sizes"]` 的 `(height, width)` 对；
该 processor 会在 OpenDM 预处理前恢复每个原始视图。HTTP 服务
按会话管理历史。只有处理后的 batch 布局 `DM05Batch` 是
模型特定的。

### 3B 导航模型

三个 3B 导航模型使用相同的会话 API，可手动启用 CUDA Graph 重放。
Low-level 和 panoramic 捕获原生的下一 token 前向计算；NaViDA v2 使用 eager 多模态 prefill、
固定地址的 StaticCache 解码 graph，并在图外执行官方 temperature/top-k 采样。
panoramic 的 `images[0]` 为全景图，候选图像可通过
如下形式提供：
`metadata={"candidate_images": [...], "candidates": [{"relative_angle": ..., "distance": ...}]}`。
Python 引擎为每轮导航任务分配独立会话，以 B=1 执行。
对于原始模型的吞吐实验，`benchmarks/3B-navigation/benchmark.py`
会对新生成且对齐的 runner 输入进行批处理。

参考数据生成脚本位于 `scripts/activevln/`。有状态策略目前同步执行，
尚不支持流水线、异步执行、分组采样和 best-of-N。

### StreamVLN

StreamVLN 是一种有状态策略，每个副本分配一个 episode。

`embodiinfer/models/streamvln` 加载分片权重，运行 SigLIP 视觉编码器、投影层和 Qwen2 attention/RoPE/SwiGLU，
处理多模态拼接与 KV 缓存。`embodiinfer/policies/streamvln` 负责确定性的 Habitat 提示词、SlowFast 帧选择、
任务状态、自回归生成和符号动作解析。

每个副本独立维护会话，在主机锁页内存中保存稀疏视觉特征，并维护覆盖 32 步 fast 窗口的 Qwen KV 缓存。
窗口结束时，从完整任务历史中均匀选取八个特征，重建 slow-memory 前缀并开始新的 fast 缓存。
历史特征和 KV 更新都包含在会话事务中，仅在解码成功后提交。

```python
from embodiinfer import EmbodiInfer

model = EmbodiInfer(
    "streamvln",
    checkpoint="/models/streamvln",
    dtype="bfloat16",
    decode_block_size=4,
    cache_history_features=True,
    fast_action_decode=True,
)
```

在独立环境中运行 `uv sync --frozen --no-dev --group streamvln` 安装依赖。
该组固定 PyTorch 2.10.0、Torchvision 0.25.0 和 Transformers 4.51.3；Linux 锁定环境使用
CUDA 13.0 wheel 和 Triton 3.6.0，不能与要求不同 Torch 或 Transformers 版本的模型组混装。
基准测试使用的 Torch 2.13 Thor 和 4090 环境通过
[`benchmarks/streamvln-benchmark/setup_env.py`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/streamvln-benchmark/setup_env.py)
以及该目录下平台特定的 requirements 文件创建。

StreamVLN 每个副本运行一个有状态会话。优化实现使用编译后的视觉编码和 prefill，
每次 CUDA Graph 重放最多生成四个相互依赖的自回归 token。
Q/K/V 与 MLP gate/up 权重采用连续存储，但仍分别执行 GEMM，以保持原有的 BF16 累加顺序。
设置 `cuda_graph=False` 可使用 eager 参考实现。

固定响应加速会先在因果 prefill 中评估官方 assistant 前缀，再用完整词表和常规重复惩罚逐 token 校验。
只有动作 token 和结束符 EOS 全部符合响应格式，才接受四动作结果；否则回滚逻辑 KV 长度，改用普通自回归解码。
测量参考实现时，设置 `fast_action_decode=False`。当前不支持张量并行；上游代码和权重采用非商业许可。

分阶段计时时，`StreamVLNPolicy.prepare_prefix` 负责观测预处理和输入传输到设备，
`encode_prepared_prefix` 负责视觉与文本编码，`StreamVLNDecoder.generate_tokens` 生成完整 token 序列，
`finalize_generation` 再完成文本解码、动作解析及下一步状态和轨迹记录的组装。
常规的 `encode_prefix` 和 `decoder.decode` 按相同顺序调用这些阶段，会话行为不变。
基准测试也按这些阶段计时，历史特征缓存的管理耗时计入 prefill。
