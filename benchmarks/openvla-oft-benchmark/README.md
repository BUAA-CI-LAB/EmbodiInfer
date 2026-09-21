# OpenVLA-OFT Benchmark

## 实验方法

模型：[Haozhan72/Openvla-oft-SFT-libero10-trajall](https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero10-trajall)，revision `97d6e58bc5fbc402e85bab382e4206bb8e5f0db0`。使用主线适配的 RLinf categorical action-token checkpoint；单路 agentview RGB，不消费 wrist/状态。一次并行生成 56 个动作 token，反归一化输出 **8×7**；不是 L1 regression action head 的 OFT 权重。

数据沿用 [PI0.5 benchmark](../pi05-benchmark/README.md) 的 `yifengzhu-hf/LIBERO-datasets`，revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`，`libero_10/*.hdf5` 任务名字典序前 10 个、每任务 numeric demo ID 前 10 段、每段含首尾均匀取 16 帧，共 **1,600 帧**。选择规则、sample ID 和 selection SHA256 与 PI0.5/Cosmos 一致。四个采样字段可在 `config.yaml` 调整。

统一单卡、B=1、BF16、seed=42、float32 matmul precision=`highest`、预热 10 次、正式测量 1 遍。输入为已解码 CPU RGB/状态；计时包含预处理、传入 GPU、模型推理、动作后处理和 CPU 输出，两端 CUDA 同步。文件读取和解码、加载、预热（包含编译/图捕获）不计入正式延迟。无需仿真，不测任务成功率。

启用 Inductor、CUDA Graph 和 prefix 复用；Thor 的 eager/SDPA 小样本比较后保留 SDPA，由运行时选择支持的内核。比较范围和数值条件见下文。正式报告记录实际 graph capture、compiler counters 和注意力配置；测量中出现新图捕获或编译则报错，不能把编译耗时混进稳态结果。

指标：mean/P50/P95/P99 延迟、calls/s、action slots/s、CUDA 峰值 allocated/reserved、进程峰值 RSS、加载/预热时间。calls/s = 完成调用数 / 总推理秒数；action slots/s 不等于机械臂控制频率。不同模型保留原生相机数量和动作长度，比较时同时列明。Thor 的 RSS 与 CUDA 内存不可相加。

本轮分段计时保留原端到端 `latency_ms`，额外记录 `model_timing_ms`：`prefill_ms` 为视觉/文本/状态编码的 CUDA elapsed，`decode_ms` 为全部去噪或 token 生成步骤的 CUDA elapsed，`gpu_inference_ms` 为二者合计；`pure_inference_ms` 为同一完整模型区间的同步墙钟时间，包含模型内部 CPU 调度及采样等待。输入预处理与输入 H2D 在计时前完成，动作反归一化、文本/动作解析及输出 D2H 在计时后执行。两段之间不插入同步；CUDA elapsed 包含流上的等待间隙，不是逐 kernel 时长相加。`model_calls_per_second` = 调用数 / 总纯推理墙钟时间，与原 E2E `calls_per_second` 分开报告。原 E2E 范围保留，但新测量含计时探针开销。新报告 schema 为 `rlinf_offline_performance_v2`，旧报告未测量的字段不回填。

OFT 按 policy 的两个执行阶段拆分：`prefill_ms` 包含 DINOv2/SigLIP 视觉编码、投影和 Llama 多模态 prompt prefill，产出 KV cache；`decode_ms` 包含基于该 KV 的一次并行动作 query forward 和类别选择。此 checkpoint 没有自回归循环，不能把这里的 decode 当成单 token 解码延迟；两段合计覆盖完整模型推理。

## 运行

每个目录拥有独立 `.venv`，不导入其他 benchmark 目录。沿用本平台已验证 Torch 环境建立环境：Thor `2.13.0+cu132`，4090 `2.13.0+cu129`。

```bash
CUDA_VISIBLE_DEVICES=0 python3 setup_env.py --platform thor --runtime-python /path/to/verified-runtime/bin/python --inference-root /path/to/EmbodiInfer
# 修改 config.yaml 的模型、数据和输出路径；输出文件已存在时会拒绝覆盖。
.venv/bin/python prepare_assets.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py --validate-data
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py
```

4090 使用 `setup_env.py --platform 4090`，通过 `CUDA_VISIBLE_DEVICES` 选择一张空闲 GPU，并传入 `--config /path/to/4090.yaml` 设置该机路径。加载时先在 CPU 将权重转为目标 BF16，再交给 EngineCore 搬入 GPU，避免 FP32 权重临时占满 24 GiB 显存；正式推理仍使用原配置。

## 实验结果

2026-09-08 在 Thor MAXN 完成全部 1,600 次调用，selection SHA256 为 `92be73f628b54f35c5f0ece7ea0e6babc4a6e3b3a4fd43adc0662bbfa516eb69`。预热后及测量结束均为 1 个 ForwardGraph、2 个编译图（`calls_captured=3445`），没有在正式测量中新增编译或图捕获。

| 模型 / 数据 | 硬件 / 日期 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| openvla-oft-libero10 | Thor / 09-08 | 1,600 | 149.36 | 80.09 | 63.55 | 143.67 | 6.6951 | 6.9605 |
| openvla-oft-libero10 | RTX 4090 / 09-08 | 1,600 | 63.90 | 44.04 | 18.20 | 62.26 | 15.6499 | 16.0614 |

Thor E2E P50/P95/P99 为 148.42/154.04/154.78 ms。decode 包含一次完整的 56 个动作 token 并行预测，动作 token 到物理 `8×7` 输出的转换在纯推理计时外。10 个跨任务观测在相同种子下，分段计时路径与普通 `EngineCore.execute` 的物理输出逐元素相同，最大绝对差为 0。

一台单卡 RTX 4090 工作站完成相同 1,600 帧。E2E P50/P95/P99 为 63.34/66.88/69.71 ms；预热后及正式测量结束均为 1 个 ForwardGraph、2 个编译图，测量中没有新捕获或编译。4090 路径一致性记录 (`reports/4090/timing-parity.json`) 覆盖 10 个跨任务观测，与普通 EngineCore 输出逐元素相同。4090 原始报告 SHA256 为 `9f2fdc1c0091248ab3eb3dae9c7b05eca6ad7e59e6cfb515a7a0e23ca979bd8f`。两台机器的完整源码摘要因上述 CPU 权重转换而不同；该改动发生在计时前，BF16 推理设置和数据采样保持一致。

attention 对比从上述 1,600 帧中均匀取 10 帧，预热 10 次、测量 3 遍，每帧固定 `torch.manual_seed(42 + i)`。仅切换 attention，其余权重、B=1、BF16、`highest`、编译及 CUDA Graph 均保持不变。SDPA/eager 纯推理均值分别为 **142.94/144.59 ms**；eager 在 1,680 个动作 token 比较位置中有 6 个不一致，物理动作最大绝对差为 `0.06640625`。预先固定的选择条件要求动作 token 完全一致，且纯推理均值至少降低 2% 才切换，因此保留 SDPA。该结论仅针对上述候选和样本，不表示穷举所有吞吐配置。

Thor 原始报告 (`reports/thor/openvla-oft-libero10.json`) 包含逐调用数据和完整条件，SHA256 为 `e0a96462f05e3e11ed36f4caceaba26db8f6234816498fe8535a2e3bb7c36f5f`；attention 对比记录 (`reports/thor/attention/selection.json`) 关联两种候选的逐调用计时和数值检查数组。统一快照只保留最终配置、必要脚本和最终报告；权重、数据和虚拟环境不进入快照，报告及快照不提交 Git。

本轮 8 个 benchmark 共用 20260908223127.tar.gz (`../snapshots/20260908223127.tar.gz`) 和一份 SHA256 (`../snapshots/20260908223127.tar.gz.sha256`)。包内只包含必要脚本、依赖文件、README、实际运行配置、14 组 Thor 与 14 组 RTX 4090 分段计时报告，以及相关数值核验记录。PI0.5 使用修复 tokenizer 后的新结果；历史报告、模型框架、权重、原始数据和虚拟环境不进入本包。报告及快照不提交 Git。

配套 EmbodiInfer 源码基线为 `72cba6e4ae05312bd93c69918709f9b792869e61`，使用本包附带的 benchmark 脚本。报告旁的 `*-config.json` 保存该次实际运行配置；复跑时调整本机模型、数据和输出路径，再传入 `--config`。两平台的采样、配置和源码核验见 协议核验记录 (`reports/4090/cross-hardware-protocol.json`)。
