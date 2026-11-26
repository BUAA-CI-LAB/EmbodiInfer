# GR00T N1.7 Benchmark

## RTX 4090 原生优化（2026-09-09）

同一单卡 RTX 4090、LIBERO-10 1,600 条观测、BF16、B=1、完整 4 步去噪，
EmbodiInfer 优化后 **E2E 31.30 ms / Forward 24.17 ms / 31.95 obs/s**。
E2E 相对优化前降低 **31.2%**，相对同机复测的 PhyAI 降低 **7.9%**。

| 实现 | 平均 E2E (ms) | 平均完整 Forward (ms) | E2E 吞吐 (obs/s) | 数值核验 |
| --- | ---: | ---: | ---: | ---: |
| EmbodiInfer，优化后 | **31.30** | **24.17** | **31.95** | **10/10，逐位一致** |
| PhyAI | 33.97 | 26.97 | 29.44 | 9/10 |
| vla.cpp | 38.70 | — | 25.84 | 10/10 |
| EmbodiInfer，优化前 | 45.51 | 38.94 | 21.97 | 10/10 |
| Embodied.cpp | 73.17 | — | 13.67 | 10/10 |

每行均为完整 1,600 条回放；vla.cpp 与 Embodied.cpp 取同日较早一轮的测量，其 C++ 内置分段计时范围不同，Forward 留空。
E2E 包含 CPU 图像/状态预处理、噪声生成、传输、完整模型与 CPU 动作后处理；Forward 包含视觉/语言编码及全部去噪。
内部动作为 `40×132`，物理输出为 `16×7`。数值核验规则见[竞品对比条件](#竞品对比条件)。

优化实现位于 `embodiinfer/policies/gr00t/`：跳过未使用的 logits，跨 4 个去噪步
复用前缀 K/V，并捕获完整视觉/DeepStack/语言/状态编码的 CUDA Graph；不引入 FlashInfer。
SDPA、Inductor 与完整去噪图继续开启。Torch `2.13.0+cu129`、Transformers `5.3.0`、
Diffusers `0.35.2`，CPU 数值库线程数为 1。前缀与去噪分别为 10.56 / 13.59 ms；
预热覆盖 10 条跨任务观测，正式测量前后均为 7 个前缀图、7 个去噪图，编译计数不变。

### 批量吞吐

按任务内凑批，独立观测与噪声，全部回放 1,600 条；表中延迟为整批耗时。

| 实现 | Batch | 整批 E2E (ms) | 整批 Forward (ms) | 吞吐 (obs/s) | 数值核验 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EmbodiInfer，优化后 | 16 | 273.18 | 154.57 | 58.57 | 10/10 |
| EmbodiInfer，优化后 | 32 | **539.33** | **304.66** | **59.33** | **10/10** |
| EmbodiInfer，优化前 | 32 | 601.76 | 396.00 | 53.18 | 10/10 |
| PhyAI | 32 | 584.60 | 328.98 | 54.74 | 9/10 |

PhyAI 行取同日较早一轮 B=2–128 扫描中吞吐最高的配置。
B=32 相对优化前吞吐提高 **11.6%**，相对 PhyAI 已测最高吞吐提高 **8.4%**。
B=64 开启前缀图时在预热阶段显存不足，未产生性能结果。最高仅指实际完成测量的配置。
batch 变化引入的 BF16 舍入差异仍使用原冻结门槛，批量结果不表述为逐位一致。

### 启用与验证

policy 参数 `native_inference=True, prefix_cuda_graph=True` 启用该路径；
API 默认均为 `False`，当前 `config.yaml` 的 benchmark 示例显式开启。

```bash
.venv/bin/python compare.py --config /path/to/check-config.json --mode check
.venv/bin/python compare.py --config /path/to/measure-config.json --mode measure
.venv/bin/python throughput.py --config /path/to/measure-config.json --batch-size 32 --output runs/gr00t-b32.json
```

真实权重回归比较 10 条观测的全部 `40×132` 动作，`rtol=atol=0`，
并验证动态图像、文本、state、embodiment、噪声、输出独立性、refit 后重建，
以及 `inference_mode` 预热到 `no_grad` 执行的切换。

## Jetson AGX Thor（2026-09-08，原生优化前）

Thor MAXN 完成全部 1,600 次正式调用。单卡、B=1、BF16，预热 10 次，正式测量 1 遍；
DiT/backbone attention 为 SDPA，启用 Inductor、LoopGraph 和 prefix 复用。
上面的原生优化尚未在 Thor 上复测。

| 硬件 | 数据集 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thor | LIBERO-10 | 1,600 | 92.47 | 38.91 | 41.57 | 80.50 | 10.8141 | 12.4222 |

E2E P50/P95/P99 为 90.14/97.87/109.21 ms。Prefill 包含视觉/文本/状态条件编码，
decode 覆盖全部 4 个去噪步，最终输出 16×7 动作块；完整模型 CUDA elapsed 为 80.47 ms。
DiT attention 在 10 条真实观测上对比 SDPA/eager（纯推理 79.37/80.05 ms）后保留 SDPA；
backbone attention 保持 checkpoint 定义。分段计时路径与普通 `EngineCore.execute`
在 10 条真实输入上的动作逐位一致。

## 实验方法

模型：[nvidia/GR00T-N1.7-LIBERO](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO)，revision `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`，加载器还需要 `Cosmos-Reason2-2B` backbone（revision `9ce19a195e423419c349abfc86fd07178b230561`）。两路 RGB、8 维 EEF/夹爪状态，使用 `libero_10` 子目录及 `libero_sim` embodiment；4 个去噪步，内部计算 40×132，按 checkpoint processor 取前 16 步并反归一化为 **16×7**。图像遵循原生 LIBERO 180° 翻转、area resize、0.95 中心裁剪和 Qwen3-VL processor。

数据沿用 [PI0.5 benchmark](../pi05-benchmark/README.md) 的 `yifengzhu-hf/LIBERO-datasets`，revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`，`libero_10/*.hdf5` 任务名字典序前 10 个、每任务 numeric demo ID 前 10 段、每段含首尾均匀取 16 帧，共 **1,600 帧**，selection SHA256 `92be73f628b54f35c5f0ece7ea0e6babc4a6e3b3a4fd43adc0662bbfa516eb69`。四个采样字段可在 `config.yaml` 调整。

统一单卡、B=1、BF16、seed=42、float32 matmul precision=`highest`、预热 10 次、正式测量 1 遍。输入为已解码 CPU RGB/状态；计时包含预处理、传入 GPU、模型推理、动作后处理和 CPU 输出，两端 CUDA 同步。文件读取和解码、加载、预热（包含编译/图捕获）不计入。测量中出现新图捕获则报错。无需仿真，不测任务成功率。

分段计时：`prefill_ms` 为视觉/文本/状态编码的 CUDA elapsed，`decode_ms` 为全部去噪步骤的 CUDA elapsed；`pure_inference_ms`（表中“纯推理”）为同一完整模型区间的同步墙钟时间，包含模型内部 CPU 调度。输入预处理与 H2D 在该区间之前，动作反归一化与 D2H 在之后；CUDA elapsed 包含流上的等待间隙，不是逐 kernel 时长相加。

## 运行

当前加载器需要 `config.yaml` 中的 `Cosmos-Reason2-2B` backbone。该仓库需要 Hugging Face 访问授权；准备资产前应登录已有访问权限的账号，或配置已下载的 backbone 路径。

每个目录拥有独立 `.venv`，不导入其他 benchmark 目录。沿用本平台已验证 Torch 环境建立环境：Thor `2.13.0+cu132`，4090 `2.13.0+cu129`。

```bash
CUDA_VISIBLE_DEVICES=0 python3 setup_env.py --platform thor --runtime-python /path/to/verified-runtime/bin/python --inference-root /path/to/EmbodiInfer
# 修改 config.yaml 的模型、数据和输出路径；输出文件已存在时会拒绝覆盖。
.venv/bin/python prepare_assets.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py --validate-data
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py
```

4090 使用 `setup_env.py --platform 4090`，`CUDA_VISIBLE_DEVICES=0` 选择单卡；传入 `--config /path/to/4090.yaml` 设置该机路径，测量代码与 Thor 相同。回跑原生优化前的路径时，将 `native_inference` 与 `prefix_cuda_graph` 设为 `False`。

## 竞品对比条件

RTX 4090 24 GiB 工作站，驱动 `580.173.02`，GPU 测量串行执行，输入与上文 1,600 条观测相同，不启动仿真或机械臂。

E2E 从已解码的 CPU 图像/状态/指令到 CPU 物理 action chunk，包含预处理、噪声生成、H2D/D2H、同步、完整模型和后处理；请求触发的 C++ 建图/更新/再预热计入 E2E。完整推理只计 GPU 输入就绪后的视觉/文本编码与全部去噪。磁盘读取、图像解码、首次模型加载、初始化预热和凑批等待不计入。各实现使用独立虚拟环境，测量前覆盖 10 条跨任务参考并预热；正式测量期间 EmbodiInfer 的图/编译状态及 PhyAI 的 kernel 选择缓存保持稳定。batch 测试按任务内凑批，保证同批文本长度相同，末批复制最后一条输入填充，填充槽位不计入吞吐；C++ 公开接口仅支持 B=1。

**数值核验。** 每条观测使用 `np.random.default_rng(42 + 选帧序号)` 生成 FP32 高斯噪声，舍入到 BF16 后显式传给各引擎。门槛在查看竞品输出前冻结：在 EmbodiInfer FP32 算术路径中使用相同的 BF16 舍入权重/输入/噪声，测量其相对 EmbodiInfer BF16 的差异，按每条观测取该差异的两倍作为 max-abs/RMSE 门槛（下限 `1e-4/1e-5`），比较归一化动作。这是数值筛选规则，不是官方任务正确性标准或任务成功率。`compare.py --mode measure` 与 `throughput.py` 默认要求同一配置的 10 条核验通过；PhyAI 未全部通过，使用 `--allow-numerical-mismatch` 完成全量测量，报告保留实际通过条数，门槛未放宽。

**版本与构建。** [vla.cpp](https://github.com/VinRobotics/vla.cpp) `f386c16347094eb2cb183edf39677cf1b6bcc750`、[Embodied.cpp](https://github.com/SEU-PAISys/Embodied.cpp) `1dad33f2c87ee1d390808cb5d776cd8c998f4a36`、[PhyAI](https://github.com/mingti-org/phyai) `a0abb211c4b05b21b766d0f6f6840eed72a8fcec`。C++ 构建为 CUDA 12.6.85、sm_89、Release；vla.cpp 使用 llama.cpp b10729，统一 `--use_fast_math` 并保留 device LTO；Embodied.cpp 使用 llama.cpp b9016 及上游 pi05/groot-n1/cuda-parity 补丁。vla.cpp 的 GR00T 路径不读取 `runtime.act_dtype`，实际为 F32 激活、BF16 GGUF，`flash_attn=true` 覆盖 Qwen3-VL ViT、Qwen3 LM 和 VLSA 编码器；Embodied.cpp 使用原生 BF16/F32 存储、Flash Attention 和 CUDA Graph，GR00T 模型源码未修改。`prepare_cpp.py` 固定转换器并记录原 checkpoint/GGUF 摘要；vla.cpp 接收归一化状态/外部 token，Embodied.cpp 接收物理状态、内部 tokenize 并返回物理动作，避免重复归一化。

PhyAI 独立环境为 Torch 2.11.0/CUDA 13.0、Transformers 5.8.1、h5py 3.16.0，FlashQLA 0.1.2 配合 CUTLASS DSL[cu13] 4.5.2；以原生 autotune、CUDA Graph、非量化方式运行。B=1 设 `autotune_cache: null`，每个进程重新 autotune，初始化不计时；batch 扫描沿用持久化缓存路径，B=128 重新调优。默认工作区为 256 MiB，planner 不足时扩展至 512 MiB 或更大；未修改 PhyAI 模型或 kernel 源码。APXinf 未取得可复现执行入口，未测。

复跑时将 `*-config.json` 中模型、数据、源码、库、参考目录与输出路径改为本机路径，设置 `PYTHONPATH` 指向 EmbodiInfer、`HF_HOME` 指向缓存。使用对应虚拟环境执行 `compare.py --config <配置> --mode check`，再为同一配置设置新 `gate_report/output` 后执行 `--mode measure`；batch 使用 `throughput.py --config <配置> --batch-size N --output <新报告>`。输入、步数、噪声和冻结门槛保持不变。
