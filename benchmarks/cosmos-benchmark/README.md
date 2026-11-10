# Cosmos Policy Benchmark

## 实验结果（2026-09-08）

Thor MAXN 与一台单卡 RTX 4090 工作站分别完成全部 1,600 次正式调用。两平台使用相同权重、测量源码、输入选择和生成设置，但 CUDA 构建与驱动不同；结果表示各自实际环境，未穷举 batch size、best-of-N 或优化组合。这是两台设备上的 BF16 基线；模型专用原生优化与量化档尚未测量。

| 模型 / 数据 | 硬件 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cosmos-libero10 | Thor | 1,600 | 1323.97 | 422.35 | 884.53 | 1306.95 | 0.7553 | 0.7651 |
| cosmos-libero10 | RTX 4090 | 1,600 | 444.62 | 130.72 | 302.65 | 433.43 | 2.2491 | 2.3072 |

E2E P50/P95/P99：Thor 1322.13/1334.26/1336.26 ms，RTX 4090 445.03/447.96/449.59 ms。Thor 完整模型 CUDA elapsed 为 1306.87 ms，E2E 与纯推理平均相差 17.02 ms，包含输入处理/传输、动作后处理和计时开销。两台机器均在预热后及正式测量结束保持 1 个 `CosmosDenoiseGraph`，没有新图捕获。10 个跨任务真实观测在相同输入、权重和随机种子下，分段计时路径与普通 `EngineCore.execute` 的 `16×7` 物理动作逐元素相同。

## 实验方法

模型：[Cosmos-Policy-LIBERO-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)，使用完整 Wan2.1 VAE 权重、dataset statistics 与检查点发布的预计算 T5 task embeddings。

数据：[LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) 的 `libero_10/*.hdf5`。与 PI0.5 同一采样：10 个任务 × 前 10 个 numeric demo × 每段均匀 16 帧，共 **1,600 帧**，selection SHA256 `92be73f628b54f35c5f0ece7ea0e6babc4a6e3b3a4fd43adc0662bbfa516eb69`。输入 agentview、wrist、gripper2 + position3 + quaternion4；quaternion 由记录的 axis-angle 重建。

测量 **Policy 动作生成：N=1、5 个去噪步、16 步动作块**，包括图像方向转换、JPEG95、resize224/crop212/resize224、VAE 编码和 DiT，启用逐步 DiT CUDA Graph，预热 10 次。使用预计算 T5 embedding，不解码未来 RGB，不运行 best-of-N 规划；结果不代表完整 WAM 视频生成或规划性能。

统一 B=1、BF16、seed=42，正式测量 1 遍。计时从已解码 CPU RGB/状态开始，到 CPU 输出结束，包含预处理、模型执行、后处理及两端 CUDA 同步；磁盘读取/解码、模型加载和预热（含编译/图捕获）单独排除。**无需仿真**，不计算任务成功率、SR/SPL。

分段计时：prefill 包含 Wan VAE 编码、状态注入和条件 mask 构建，decode 包含初始噪声及全部 5 个采样步，两者均为 CUDA elapsed；`pure_inference_ms`（表中“纯推理”）为同一完整模型区间的同步墙钟时间。T5 缓存嵌入的读取和输入传输在纯推理计时前完成，本实验不执行在线 T5 编码。Thor 的 CPU/GPU 共享物理内存，RSS 与 CUDA 内存不能相加。

## 运行

在本目录执行，依赖配套的 EmbodiInfer 源码和本平台已验证的 Torch 环境。`setup_env.py` 为本目录创建独立 `.venv`；`benchmark.py` 自带数据采样和计时代码。

```bash
# Thor: --platform thor；4090: --platform 4090，并选择一张空闲 GPU。
CUDA_VISIBLE_DEVICES=0 python3 setup_env.py --platform thor --runtime-python /path/to/verified-runtime/bin/python --inference-root /path/to/EmbodiInfer
# 修改 config.yaml 中的模型、数据路径；新结果默认写入 runs/。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py --validate-data
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py
```

Thor 使用 Torch 2.13.0+cu132，4090 使用 2.13.0+cu129，模型依赖分别见 `requirements-thor.txt` / `requirements-4090.txt`。`--config /path/to/config.yaml` 可使用另一份配置。选定数据缺失时会报错，不以随机输入替代。
