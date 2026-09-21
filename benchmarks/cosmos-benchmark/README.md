# Cosmos Policy Benchmark

## 实验方法

模型：[Cosmos-Policy-LIBERO-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)，使用完整 Wan2.1 VAE 权重、dataset statistics 与检查点发布的预计算 T5 task embeddings。

数据：[LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) 的 `libero_10/*.hdf5`。与 PI0.5 同一采样：10 个任务 × 前 10 个 numeric demo × 每段均匀 16 帧，共 **1,600 帧**。输入 agentview、wrist、gripper2 + position3 + quaternion4；quaternion 由记录的 axis-angle 重建。

本轮测 **Policy 动作生成：N=1、5 个去噪步、16 步动作块**，包括图像方向转换、JPEG95、resize224/crop212/resize224、VAE 编码和 DiT，启用逐步 DiT CUDA Graph，预热 10 次。使用预计算 T5 embedding，不解码未来 RGB，不运行 best-of-N 规划；结果不代表完整 WAM 视频生成或规划性能。

统一 B=1、BF16、seed=42，正式测量 1 遍。计时从已解码 CPU RGB/状态开始，到 CPU 输出结束，包含预处理、模型执行、后处理及两端 CUDA 同步；磁盘读取/解码、模型加载和预热（含编译/图捕获）单独排除。**无需仿真**，不计算任务成功率、SR/SPL。

指标：mean / P50 / P95 / P99 延迟、calls/s、action slots/s、tokens/s、CUDA 峰值 allocated/reserved、进程峰值 RSS，以及加载/预热时间。吞吐为总输出数除以总推理时间；action slots/s 不等于机器人控制频率。Thor 的 CPU/GPU 共享物理内存，RSS 与 CUDA 内存不能相加。

本轮分段计时保留原端到端 `latency_ms`，额外记录 `model_timing_ms`：`prefill_ms` 为视觉/文本/状态编码的 CUDA elapsed，`decode_ms` 为全部去噪或 token 生成步骤的 CUDA elapsed，`gpu_inference_ms` 为二者合计；`pure_inference_ms` 为同一完整模型区间的同步墙钟时间，包含模型内部 CPU 调度及采样等待。输入预处理与输入 H2D 在计时前完成，动作反归一化、文本/动作解析及输出 D2H 在计时后执行。两段之间不插入同步；CUDA elapsed 包含流上的等待间隙，不是逐 kernel 时长相加。`model_calls_per_second` = 调用数 / 总纯推理墙钟时间，与原 E2E `calls_per_second` 分开报告。原 E2E 范围保留，但新测量含计时探针开销。新报告 schema 为 `rlinf_offline_performance_v2`，旧报告未测量的字段不回填。

Cosmos 的 prefill 具体包含 Wan VAE 编码、状态注入和条件 mask 构建；decode 包含初始噪声及全部 5 个采样步。T5 使用检查点发布的缓存嵌入，缓存读取和输入传输在纯推理计时前完成，本实验不执行在线 T5 编码。这里的完整纯推理对应上述 N=1 Policy 动作生成范围。

## 运行

在本目录执行，依赖配套的 EmbodiInfer 源码和本平台已验证的 Torch 环境。2026-09-07 的旧结果基于 `b86cce30ba57`；本轮分段计时需使用当前工作树中同时更新的 EmbodiInfer，实际 Python 源码摘要记录在报告的 `environment.source_python_sha256`。`setup_env.py` 为本目录创建独立 `.venv`；`benchmark.py` 自带数据采样和计时代码。

```bash
# Thor: --platform thor；4090: --platform 4090，并选择一张空闲 GPU。
CUDA_VISIBLE_DEVICES=0 python3 setup_env.py --platform thor --runtime-python /path/to/verified-runtime/bin/python --inference-root /path/to/EmbodiInfer
# 修改 config.yaml 中的模型、数据路径；新结果默认写入 runs/。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py --validate-data
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py
```

Thor 使用 Torch 2.13.0+cu132，4090 使用 2.13.0+cu129，模型依赖分别见 `requirements-thor.txt` / `requirements-4090.txt`。`--config /path/to/config.yaml` 可使用另一份配置。选定数据缺失时会报错，不以随机输入替代。

## 最终结果

2026-09-08 在 Thor MAXN 和一台单卡 RTX 4090 工作站分别完成本轮全部正式调用。两平台使用相同权重、测量源码、输入选择和生成设置，但 CUDA 构建与驱动不同；结果表示各自实际环境，未穷举所有 batch size 或优化组合。

| 模型 / 数据 | 硬件 / 日期 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosmos-libero10 | Thor / 09-08 | 1,600 | 1323.97 | 422.35 | 884.53 | 1306.95 | 0.7553 | 0.7651 |
| cosmos-libero10 | RTX 4090 / 09-08 | 1,600 | 444.62 | 130.72 | 302.65 | 433.43 | 2.2491 | 2.3072 |

Thor E2E P50/P95/P99 为 1322.13/1334.26/1336.26 ms；完整模型 CUDA elapsed 为 1306.87 ms，同一区间的纯推理同步墙钟为 1306.95 ms。prefill 包含 VAE 编码及条件构建，decode 覆盖全部 5 个采样步，输出为 16×7 动作块。E2E 与纯推理平均相差 17.02 ms，包含输入处理/传输、动作后处理和计时开销。

两台机器均在预热后及正式测量结束保持 1 个 `CosmosDenoiseGraph`，没有新图捕获；测量沿用逐步 DiT 图。使用预计算 T5 嵌入，N=1、不解码未来 RGB；这些数值对应 Policy 动作生成范围。

selection SHA256 为 `92be73f628b54f35c5f0ece7ea0e6babc4a6e3b3a4fd43adc0662bbfa516eb69`，与 PI0.5 相同。Thor 的全部 1,600 个逐调用动作块 SHA256 与旧 Thor 报告一致。

4090 另取 10 个跨任务真实观测，在相同输入、权重和随机种子下，将分段计时路径与普通 `EngineCore.execute` 对比，完整 5 步生成后的 `16×7` 物理动作逐元素相同，最大绝对差为 0，见 路径一致性记录 (`reports/4090/timing-parity.json`)。

当前 Python 源码 SHA256 为 `e419a9a2814a4dfc4de338649e0984f0b858c18624a4c600b6f5cef0aa419e9e`，Thor 原始 JSON SHA256 为 `acf049be481faefa4b9eb5df712f5f973baa4d3801396f26cb48a314cd50cbef`；4090 原始 JSON SHA256 为 `b5f4c8c6884aff181020914d79abf81f577383d0af8edd331ce2855729d5607f`。4090 的 E2E P50/P95/P99 为 445.03/447.96/449.59 ms；均值、分位数、吞吐和采样摘要已从逐调用记录独立复算。

原始报告包含逐调用 E2E/分段计时、P50/P95/P99、完整配置、样本与源码摘要、内存及实际图运行状态。旧 4090 报告没有测量纯推理字段，已留存为历史记录；本表只列新测的分段计时结果。

本轮 8 个 benchmark 共用 20260908223127.tar.gz (`../snapshots/20260908223127.tar.gz`) 和一份 SHA256 (`../snapshots/20260908223127.tar.gz.sha256`)。包内只包含必要脚本、依赖文件、README、实际运行配置、14 组 Thor 与 14 组 RTX 4090 分段计时报告，以及相关数值核验记录。PI0.5 使用修复 tokenizer 后的新结果；历史报告、模型框架、权重、原始数据和虚拟环境不进入本包。报告及快照不提交 Git。

配套 EmbodiInfer 源码基线为 `72cba6e4ae05312bd93c69918709f9b792869e61`，使用本包附带的 benchmark 脚本。报告旁的 `*-config.json` 保存该次实际运行配置；复跑时调整本机模型、数据和输出路径，再传入 `--config`。两平台的采样、配置和源码核验见 协议核验记录 (`reports/4090/cross-hardware-protocol.json`)。
