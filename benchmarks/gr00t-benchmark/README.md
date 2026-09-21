# GR00T N1.7 Benchmark

## 4090 原生优化（2026-09-09）

同一单卡 RTX 4090、LIBERO-10 1,600 条观测、BF16、B=1、完整 4 步去噪，
Inference 优化后 **E2E 31.30 ms / Forward 24.17 ms / 31.95 obs/s**。
E2E 相对原实现降低 **31.2%**，相对本次同机复测的 PhyAI 降低 **7.9%**。
这是本轮已测同口径实现中的最低延迟；新优化尚未在 Thor 上验证。

| 实现 | 平均 E2E (ms) | 平均完整 Forward (ms) | E2E 吞吐 (obs/s) | 数值核验 |
| --- | ---: | ---: | ---: | ---: |
| Inference，优化后 | **31.30** | **24.17** | **31.95** | **10/10，逐位一致** |
| PhyAI，本次复测 | 33.97 | 26.97 | 29.44 | 9/10 |
| vla.cpp，前轮 | 38.70 | — | 25.84 | 10/10 |
| Inference，优化前 | 45.51 | 38.94 | 21.97 | 10/10 |
| Embodied.cpp，前轮 | 73.17 | — | 13.67 | 10/10 |

以上每行均为完整 1,600 条回放。E2E 包含 CPU 图像/状态预处理、噪声生成、
传输、完整模型与 CPU 动作后处理；Forward 包含视觉/语言编码及全部去噪。
沿用下文固定的 checkpoint、样本、BF16 噪声与逐样本数值门槛，
内部动作保持 `40×132`，物理输出保持 `16×7`。PhyAI 保留未通过数值门槛的标记。

优化实现位于 `embodiinfer/policies/gr00t/`：跳过未使用的 logits，跨 4 个去噪步
复用前缀 K/V，并捕获完整视觉/DeepStack/语言/状态编码的 CUDA Graph。
不引入 FlashInfer。SDPA、Inductor 与完整去噪图继续开启；本次使用 Torch
`2.13.0+cu129`、Transformers `5.3.0`、Diffusers `0.35.2`，CPU 数值库线程数为 1。
前缀与去噪分别为 10.56 / 13.59 ms；预热覆盖 10 条跨任务观测，正式测量
前后均为 7 个前缀图、7 个去噪图，编译计数不变。

### 批量吞吐

仍按任务内凑批，独立观测与噪声，全部回放 1,600 条；表中延迟为整批耗时。

| 实现 | Batch | 整批 E2E (ms) | 整批 Forward (ms) | 吞吐 (obs/s) | 数值核验 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Inference，优化后 | 16 | 273.18 | 154.57 | 58.57 | 10/10 |
| Inference，优化后 | 32 | **539.33** | **304.66** | **59.33** | **10/10** |
| Inference，优化前 | 32 | 601.76 | 396.00 | 53.18 | 10/10 |
| PhyAI，前轮 | 32 | 584.60 | 328.98 | 54.74 | 9/10 |

B=32 相对原实现吞吐提高 **11.6%**，相对已测 PhyAI 最高吞吐提高 **8.4%**。
B=64 开启前缀图时在预热阶段显存不足，未产生性能结果，见
诊断日志 (`reports/4090-native/batch-64.log`)。B=32 为本轮已完成配置中的最高吞吐。
batch 变化引入的 BF16 舍入差异仍使用原冻结门槛，不将批量结果表述为逐位一致。

### 使用与验证

policy 参数 `native_inference=True, prefix_cuda_graph=True` 启用该路径；
API 默认均为 `False`，当前 `config.yaml` 的 benchmark 示例显式开启。
回跑下文旧 Thor/4090 基线时将这两个参数设为 `False`。

复跑使用 实际检查配置 (`reports/4090-native/native-check-config.json`) 与
实际测量配置 (`reports/4090-native/native-measure-config.json`)，调整本机模型、数据、
参考文件和新输出路径；`gate_report` 指向本次检查输出。保持冻结容差及参考观测不变。

```bash
.venv/bin/python compare.py --config /path/to/check-config.json --mode check
.venv/bin/python compare.py --config /path/to/measure-config.json --mode measure
.venv/bin/python throughput.py --config /path/to/measure-config.json --batch-size 32 --output runs/gr00t-b32.json
```

真实权重回归额外比较 10 条观测的全部 `40×132` 动作，`rtol=atol=0`，
并验证动态图像、文本、state、embodiment、噪声、输出独立性、refit 后重建，
以及 `inference_mode` 预热到 `no_grad` 执行的切换。
GPU 测试 (`reports/4090-native/gpu-tests.log`) 10 项通过；
完整 CPU 测试 (`reports/4090-native/cpu-suite.log`) 407 通过、43 跳过。
修改范围的 Ruff/格式检查通过；全仓 Ruff 的 122 项问题均位于未修改的主线文件，
见 检查记录 (`reports/4090-native/lint-validation.json`)。
独立复算与源码核验 (`reports/4090-native/validation.json`) 确认均值、吞吐、样本和容差一致。
原始报告、配置、测试记录和压缩包均不提交 Git。

本轮最终源码与实验记录打包为 20260909152532.tar.gz (`../snapshots/20260909152532.tar.gz`)，共用一份 SHA256 (`../snapshots/20260909152532.tar.gz.sha256`)。包内不含权重、数据集、虚拟环境或编译产物。

## 实验方法

模型：[nvidia/GR00T-N1.7-LIBERO](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO)，revision `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`。两路 RGB、8 维 EEF/夹爪状态。使用 `libero_10` 子目录及 `libero_sim` embodiment；4 个去噪步，内部计算 40×132，按 checkpoint processor 取前 16 步并反归一化为 **16×7**。图像遵循原生 LIBERO 180° 翻转、area resize、0.95 中心裁剪和 Qwen3-VL processor。

数据沿用 [PI0.5 benchmark](../pi05-benchmark/README.md) 的 `yifengzhu-hf/LIBERO-datasets`，revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`，`libero_10/*.hdf5` 任务名字典序前 10 个、每任务 numeric demo ID 前 10 段、每段含首尾均匀取 16 帧，共 **1,600 帧**。选择规则、sample ID 和 selection SHA256 与 PI0.5/Cosmos 一致。四个采样字段可在 `config.yaml` 调整。

统一单卡、B=1、BF16、seed=42、float32 matmul precision=`highest`、预热 10 次、正式测量 1 遍。输入为已解码 CPU RGB/状态；计时包含预处理、传入 GPU、模型推理、动作后处理和 CPU 输出，两端 CUDA 同步。文件读取和解码、加载、预热（包含编译/图捕获）不计入正式延迟。无需仿真，不测任务成功率。

启用 Inductor、CUDA Graph 和 prefix 复用；SDPA 由运行时选择支持的内核。本轮对 DiT 的 SDPA/eager 做相同观测和种子的实测对比；保留 checkpoint 定义的 backbone attention。正式报告记录实际 graph capture、compiler counters 和注意力配置；测量中出现新图捕获则报错，不能把编译耗时混进稳态结果。

指标：mean/P50/P95/P99 延迟、calls/s、action slots/s、CUDA 峰值 allocated/reserved、进程峰值 RSS、加载/预热时间。calls/s = 完成调用数 / 总推理秒数；action slots/s 不等于机械臂控制频率。不同模型保留原生相机数量和动作长度，比较时同时列明。Thor 的 RSS 与 CUDA 内存不可相加。

本轮分段计时保留原端到端 `latency_ms`，额外记录 `model_timing_ms`：`prefill_ms` 为视觉/文本/状态编码的 CUDA elapsed，`decode_ms` 为全部去噪或 token 生成步骤的 CUDA elapsed，`gpu_inference_ms` 为二者合计；`pure_inference_ms` 为同一完整模型区间的同步墙钟时间，包含模型内部 CPU 调度及采样等待。输入预处理与输入 H2D 在计时前完成，动作反归一化、文本/动作解析及输出 D2H 在计时后执行。两段之间不插入同步；CUDA elapsed 包含流上的等待间隙，不是逐 kernel 时长相加。`model_calls_per_second` = 调用数 / 总纯推理墙钟时间，与原 E2E `calls_per_second` 分开报告。原 E2E 范围保留，但新测量含计时探针开销。新报告 schema 为 `rlinf_offline_performance_v2`，旧报告未测量的字段不回填。

## 运行

除 LIBERO checkpoint 外，当前加载器还需要 `config.yaml` 中的 `Cosmos-Reason2-2B` backbone。该仓库需要 Hugging Face 访问授权；准备资产前应在 Thor 登录已有访问权限的账号，或配置已下载的 backbone 路径。

每个目录拥有独立 `.venv`，不导入其他 benchmark 目录。沿用本平台已验证 Torch 环境建立环境：Thor `2.13.0+cu132`，4090 `2.13.0+cu129`。

```bash
CUDA_VISIBLE_DEVICES=0 python3 setup_env.py --platform thor --runtime-python /path/to/verified-runtime/bin/python --inference-root /path/to/EmbodiInfer
# 修改 config.yaml 的模型、数据和输出路径；输出文件已存在时会拒绝覆盖。
.venv/bin/python prepare_assets.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py --validate-data
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py
```

4090 使用 `setup_env.py --platform 4090`，`CUDA_VISIBLE_DEVICES=0` 选择单卡；传入 `--config /path/to/4090.yaml` 设置该机路径，测量代码与 Thor 相同。

## 实验结果

2026-09-08 在 Thor MAXN 和一台单卡 RTX 4090 工作站分别完成全部 1,600 次正式调用。单卡、batch=1、BF16，预热 10 次，正式测量 1 遍。输入选择与 PI0.5/DM0.5/OpenVLA-OFT/Cosmos 的 LIBERO-10 相同；两台机器的测量源码 SHA256、采样、精度、去噪步数和优化开关一致。

| 硬件 | 数据集 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thor | LIBERO-10 | 1,600 | 92.47 | 38.91 | 41.57 | 80.50 | 10.8141 | 12.4222 |
| RTX 4090 | LIBERO-10 | 1,600 | 47.32 | 24.86 | 15.56 | 40.45 | 21.1339 | 24.7239 |

4090 使用 Torch `2.13.0+cu129`、Transformers `5.3.0`，DiT/backbone attention 均为 SDPA，启用 Inductor、LoopGraph 和 prefix 复用。正式测量前后均为 7 个图，compiler counters 不变；E2E P50/P95/P99 为 46.26/49.35/56.06 ms。均值、分位数、吞吐和样本摘要已从逐调用记录独立复算。普通 `EngineCore.execute` 与分段计时路径在 10 条输入上的动作逐位一致，见 4090 数值核验 (`reports/4090/timing-parity.json`)。4090 原始报告 (`reports/4090/gr00t-libero10.json`) SHA256 为 `c3c112321fd980d886df47c17b913c4cb463d86d33f9bd867a4eff5b941480a2`。

Prefill 包含视觉/文本/状态条件编码，decode 覆盖全部 4 个去噪步，最终输出 16×7 动作块。Thor 完整模型 CUDA elapsed 为 80.47 ms，同一区间的纯推理同步墙钟为 80.50 ms。Thor E2E P50/P95/P99 为 90.14/97.87/109.21 ms。

预热后与正式测量结束均为 7 个图，compiler counters 保持不变；backbone attention 为 `sdpa`。

Thor 的 DiT attention 对比使用 10 条真实观测、每条 3 次重复，两种后端分别按 seed=42+i 重置随机数。SDPA/eager 纯推理均值为 79.37/80.05 ms；公共动作块最大绝对差为 0.110778809，最终选择 `sdpa`。切换标准是在公共动作逐位一致的前提下，纯推理均值改善至少 2%；backbone attention 保持不变。

当前普通 `EngineCore.execute` 与实际分阶段计时调用使用相同权重、观测和种子，在 10 条真实输入上的公共 float32 动作块逐位一致，rtol=atol=0，见调用一致性记录 (`reports/thor/timing-parity.json`)。

Cosmos-Reason2-2B 固定 revision 为 `9ce19a195e423419c349abfc86fd07178b230561`，10 个依赖文件已在 Thor 逐文件校验；backbone 权重 SHA256 为 `fa5a6e6ef4fce40216b185cc48a3b24d31637ac3e2ba69c107ed1f389c1e6ede`。

selection SHA256 为 `92be73f628b54f35c5f0ece7ea0e6babc4a6e3b3a4fd43adc0662bbfa516eb69`；实际 Python 源码 SHA256 为 `c41d2dc8195cd851871d3dac5140e2e4039ab1a56da6513ffd6227f66c86fe2a`。

Thor 原始报告 (`reports/thor/gr00t-libero10.json`) 保留逐调用数据与完整条件，SHA256 为 `84439cfaa5623258cd1ef6f21096f3fd8c275981076c7dd1fbc5b13a3fa718de`；attention 对比记录 (`reports/thor/attention/selection.json`) 关联两种候选的计时和数值检查数组。报告及统一快照不提交 Git；权重、数据和虚拟环境不进入快照。

本轮 8 个 benchmark 共用 20260908223127.tar.gz (`../snapshots/20260908223127.tar.gz`) 和一份 SHA256 (`../snapshots/20260908223127.tar.gz.sha256`)。包内只包含必要脚本、依赖文件、README、实际运行配置、14 组 Thor 与 14 组 RTX 4090 分段计时报告，以及相关数值核验记录。PI0.5 使用修复 tokenizer 后的新结果；历史报告、模型框架、权重、原始数据和虚拟环境不进入本包。报告及快照不提交 Git。

配套 EmbodiInfer 源码基线为 `72cba6e4ae05312bd93c69918709f9b792869e61`，使用本包附带的 benchmark 脚本。报告旁的 `*-config.json` 保存该次实际运行配置；复跑时调整本机模型、数据和输出路径，再传入 `--config`。两平台的采样、配置和源码核验见 协议核验记录 (`reports/4090/cross-hardware-protocol.json`)。

## 4090 竞品对比（2026-09-09）

使用一台 RTX 4090 24 GiB 工作站，驱动 `580.173.02`，GPU 测量串行执行。每组回放 LIBERO-10 的 10 个字典序任务文件、每任务前 10 个数字序 demo、每 demo 含首尾均匀取 16 帧，共 1,600 条离线观测；不启动仿真或机械臂。数据源 `yifengzhu-hf/LIBERO-datasets` revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`。选帧 SHA256：`92be73f628b54f35c5f0ece7ea0e6babc4a6e3b3a4fd43adc0662bbfa516eb69`。

检查点为 `nvidia/GR00T-N1.7-LIBERO` 的 `libero_10` 变体（revision `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`），使用该检查点的 backbone 权重和 Cosmos-Reason2-2B 处理器元数据。两个 256px 相机，4 步去噪，内部 `40×132`、输出物理动作 `16×7`。保持原相机翻转/裁剪、Qwen3-VL 模板和状态 percentile 归一化。

### 计时与配置

E2E 从已解码的 CPU 图像/状态/指令到 CPU 物理 action chunk，包含预处理、噪声生成、H2D/D2H、同步、完整模型和后处理。完整推理只计 GPU 输入就绪后的视觉/文本编码与全部去噪。磁盘读取、图像解码、首次模型加载、初始化预热和凑批等待不计入；请求触发的 C++ 建图/更新/再预热计入 E2E。C++ 内置分段计时范围不同，完整推理列留空，避免混用。

EmbodiInfer 为 BF16、SDPA、Inductor 和完整去噪 CUDA Graph；PhyAI 为原生 autotune、CUDA Graph、非量化运行。测量前覆盖 10 条跨任务参考并预热，batch 核验覆盖相同 10 条输入。正式测量期间 EmbodiInfer 的图/编译状态及 PhyAI 的 kernel 选择缓存保持稳定。各实现独立使用 `.venv`、`.venv-vlacpp`、`.venv-embodied`、`.venv-phyai`。

### B=1 全量结果

下表所有行都完成 1,600 条回放；数值核验通过条数单独列出。吞吐单位为真实 observation/s。

| 实现 | 平均 E2E (ms) | 平均完整推理 (ms) | E2E 吞吐 (obs/s) | 数值核验 |
| --- | ---: | ---: | ---: | ---: |
| EmbodiInfer | 45.51 | 38.94 | 21.97 | 10/10 |
| vla.cpp | 38.70 | — | 25.84 | 10/10 |
| Embodied.cpp | 73.17 | — | 13.67 | 10/10 |
| PhyAI | 34.48 | 27.18 | 29.00 | 9/10 |

### Batch 吞吐扫描

每个 batch 都处理独立观测和独立噪声；末批复制最后一条输入填充，填充槽位不计入吞吐。延迟为整批完成时间。C++ 公开接口仅支持 B=1。

GR00T 按任务内凑批，保证同批文本长度相同，不额外引入文本 padding。每任务 160 条观测，B=64 的第三批补 32 条，B=128 的第二批补 96 条，故有效吞吐同时受填充开销影响。

| 实现 | Batch | 平均整批 E2E (ms) | 平均整批完整推理 (ms) | E2E 吞吐 (obs/s) | 数值核验 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EmbodiInfer | 4 | 89.49 | 64.14 | 44.70 | 10/10 |
| EmbodiInfer | 8 | 166.86 | 110.14 | 47.95 | 10/10 |
| EmbodiInfer | 16 | 311.79 | 204.58 | 51.32 | 10/10 |
| EmbodiInfer | 32 | 601.76 | 396.00 | 53.18 | 10/10 |
| EmbodiInfer | 64 | 1187.16 | 803.66 | 44.92 | 10/10 |
| PhyAI | 2 | 51.21 | 36.55 | 39.05 | 8/10 |
| PhyAI | 4 | 83.15 | 52.88 | 48.10 | 6/10 |
| PhyAI | 8 | 149.31 | 91.62 | 53.58 | 9/10 |
| PhyAI | 16 | 309.23 | 179.97 | 51.74 | 9/10 |
| PhyAI | 32 | 584.60 | 328.98 | 54.74 | 9/10 |
| PhyAI | 64 | 1131.42 | 682.29 | 47.14 | 9/10 |
| PhyAI | 128 | 2195.20 | 1474.43 | 36.44 | 9/10 |

EmbodiInfer 已测最高吞吐为 **B=32，53.18 obs/s**（该配置数值核验 10/10）。最高仅指实际完成测量的配置，不代表理论上限。
PhyAI 已测最高吞吐为 **B=32，54.74 obs/s**（该配置数值核验 9/10）。最高仅指实际完成测量的配置，不代表理论上限。

扫描停止原因保留在 诊断日志 (`reports/4090-competitors/diagnostics`)。

- `phyai-autotune-fresh-batch-128`：FlashInfer 临时工作区不足，后续扩大工作区重试。
- `phyai-autotune-unchecked-batch-128`：FlashInfer 临时工作区不足，后续扩大工作区重试。
- `phyai-autotune-workspace1024-batch-256`：CUDA 显存不足。
- `phyai-autotune-workspace512-batch-256`：FlashInfer 临时工作区不足，后续扩大工作区重试。

EmbodiInfer 前轮 PI0.5 B=4、GR00T B=2 各有一条参考超出数值门槛，未计完整性能；PI0.5 B=64、GR00T B=128 显存不足。本次按用户要求补测 PhyAI，保留 EmbodiInfer 既有结果。

### 数值核验与继续计时

每条观测使用 `np.random.default_rng(42 + 选帧序号)` 产生 FP32 高斯噪声，再舍入到 BF16，显式传给各引擎。先在 EmbodiInfer FP32 算术路径中使用相同 BF16 舍入权重/输入/噪声，测量相对 EmbodiInfer BF16 的差异；在查看竞品输出前，按每条观测分别冻结 max-abs/RMSE 门槛，取该差异两倍，下限 `1e-4/1e-5`。比较归一化动作，门槛与原始报告见 tolerance-contract.json (`reports/4090-competitors/tolerance-contract.json`)。这是数值筛选规则，不是官方任务正确性标准或任务成功率。

`compare.py --mode measure` 默认要求同一配置的 10 条核验通过；`throughput.py` 默认也受该门槛约束。用户明确要求忽略精度问题后，PhyAI 使用 `--allow-numerical-mismatch` 完整测量：仍要求对应配置的核验记录、相同输入、正确输出形状及有限值，报告 `passed` 保留数值结果，`measurement_complete` 单独表示回放完成，`allow_numerical_mismatch` 记录本次选择。全量报告没有将失败核验改为通过。

PhyAI 首次默认图模式和 autotune 均为 9/10 通过；关闭图复用也未解决数值差异。全量结果采用对应新配置的核验记录，具体通过条数见上表；门槛从未放宽。

### 复现依赖与适配

源码固定为 [vla.cpp](https://github.com/VinRobotics/vla.cpp) `f386c16347094eb2cb183edf39677cf1b6bcc750`、[Embodied.cpp](https://github.com/SEU-PAISys/Embodied.cpp) `1dad33f2c87ee1d390808cb5d776cd8c998f4a36`、[PhyAI](https://github.com/mingti-org/phyai) `a0abb211c4b05b21b766d0f6f6840eed72a8fcec`。C++ 构建为 CUDA 12.6.85、sm_89、Release；vla.cpp 使用 llama.cpp b10729、统一 `--use_fast_math` 保留 device LTO；Embodied.cpp 使用 llama.cpp b9016 及上游 pi05/groot-n1/cuda-parity 补丁。实际构建参数、差异和动态库 SHA256 见 build-provenance.json (`reports/4090-competitors/build-provenance.json`)，GGUF dtype 统计见 precision-provenance.json (`reports/4090-competitors/precision-provenance.json`)。

vla.cpp 的 GR00T 路径不读取 `runtime.act_dtype`，实际为 F32 激活，GGUF 的 1,030 个 tensor 均为 BF16；`flash_attn=true` 覆盖 Qwen3-VL ViT、Qwen3 LM 和 VLSA 编码器。Embodied.cpp 使用原生 BF16/F32 存储、启用 Flash Attention 和 CUDA Graph。其动态库附带 PI0.5 常量修复，GR00T 模型源码未修改。

`prepare_cpp.py` 固定转换器并记录原 checkpoint/GGUF 摘要。Python 仅做公开 API 需要的预处理，C++ 做图像 patchification；vla.cpp 接收归一化状态/外部 token，Embodied.cpp 接收物理状态、内部 tokenize 并返回物理动作，避免重复归一化。CPU 状态显式 FP32，模型入口再转 BF16；两种全局默认 dtype 下共 20 条输入逐位核验通过。CPU 预处理对比 24/8/4/2/1 线程未发现减线程收益，见 CPU 记录 (`reports/4090-competitors/cpu-threads.json`)。

PhyAI 独立环境为 Torch 2.11.0/CUDA 13.0、Transformers 5.8.1、h5py 3.16.0。FlashQLA 0.1.2 要求 TVM-FFI 0.1.9，使用兼容的 CUTLASS DSL[cu13] 4.5.2；两个环境均通过 `uv pip check`。EmbodiInfer 沿用 Torch 2.13.0+cu129、Transformers 5.3.0。完整版本见 environments.json (`reports/4090-competitors/environments.json`)。

PhyAI 的持久化 kernel 选择缓存会跳过某些后端预热，导致首次 cuDNN 分配发生在捕获期间。本次 B=1 和 PI0.5 补测设 `autotune_cache: null`，每个进程重新 autotune，初始化不计时，CUDA Graph 保持开启；GR00T B=2–64 保留原缓存路径，B=128 重新调优。默认工作区为 256 MiB，planner 不足时按新配置扩展至 512 MiB 或更大，具体容量写入对应报告；未修改 PhyAI 模型或 kernel 源码。

复跑时将 `*-config.json` 中模型、数据、源码、库、参考目录与输出路径改为本机路径，设置 `PYTHONPATH` 指向 EmbodiInfer、`HF_HOME` 指向缓存。使用对应虚拟环境执行 `compare.py --config <配置> --mode check`，再为同一配置设置新 `gate_report/output` 后执行 `--mode measure --allow-numerical-mismatch`；批量使用 `throughput.py --config <配置> --batch-size N --output <新报告> --allow-numerical-mismatch`。输入、步数、噪声和冻结门槛保持不变。APXinf 本轮未取得可复现执行入口，未测。

### 快照

两模型共用 20260909105724.tar.gz (`../snapshots/20260909105724.tar.gz`) 和一份 SHA256 (`../snapshots/20260909105724.tar.gz.sha256`)。仅包含必要脚本、README、依赖/构建记录、实际配置、全量逐调用报告及相关核验；不含权重、原始数据集、虚拟环境、动态库。报告和压缩包均由 `.gitignore` 排除。

远端实际 HEAD 为 `72cba6e4ae05312bd93c69918709f9b792869e61`，benchmark 使用本包版本；收集清单 (`reports/4090-competitors/collection-manifest.json`) 记录报告与脚本 SHA256，适配器变更另有逐文件差异。
