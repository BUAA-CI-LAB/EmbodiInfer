# QwenVL / NaViDA Benchmark

## 实验结果（2026-09-08）

Thor MAXN 与一台单卡 RTX 4090 工作站分别完成三个模型在 R2R / RxR 上的全部正式调用。两平台使用相同权重、输入选择和生成设置，但 CUDA 构建与驱动不同；结果表示各自实际环境，未穷举 batch size、量化或优化组合。

| 模型 / 数据 | 硬件 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low-r2r | Thor | 192 | 212.51 | 157.44 | 0.01 | 157.50 | 4.7056 | 6.3494 |
| low-r2r | RTX 4090 | 192 | 118.75 | 76.27 | 0.01 | 76.31 | 8.4210 | 13.1041 |
| low-rxr | Thor | 192 | 226.52 | 170.03 | 0.01 | 170.08 | 4.4146 | 5.8796 |
| low-rxr | RTX 4090 | 192 | 124.88 | 82.17 | 0.01 | 82.21 | 8.0076 | 12.1639 |
| navida-r2r | Thor | 192 | 1145.61 | 323.83 | 721.24 | 1045.15 | 0.8729 | 0.9568 |
| navida-r2r | RTX 4090 | 192 | 377.54 | 98.19 | 234.25 | 332.49 | 2.6487 | 3.0076 |
| navida-rxr | Thor | 192 | 1131.33 | 318.67 | 710.87 | 1029.60 | 0.8839 | 0.9713 |
| navida-rxr | RTX 4090 | 192 | 376.52 | 102.07 | 229.15 | 331.28 | 2.6559 | 3.0186 |
| panoramic-r2r | Thor | 192 | 773.61 | 642.71 | 0.02 | 642.79 | 1.2926 | 1.5557 |
| panoramic-r2r | RTX 4090 | 192 | 398.82 | 282.24 | 0.01 | 282.30 | 2.5074 | 3.5423 |
| panoramic-rxr | Thor | 192 | 777.25 | 644.03 | 0.02 | 644.10 | 1.2866 | 1.5526 |
| panoramic-rxr | RTX 4090 | 192 | 397.09 | 282.46 | 0.01 | 282.52 | 2.5183 | 3.5396 |

每组先预热全部 192 个输入，随后正式测量 192 次。Low/Panoramic 使用 SDPA、Inductor 和单 token 前向图，NaViDA 使用 StaticCache 自回归解码图；测量中没有新图捕获，编译失败与动作解析失败均为 0。NaViDA 延迟包含直到 EOS 或 512-token 上限的全部生成（每组 192 次调用共生成约 3,700–3,800 个 token），不能解释为单个 decode step。每组另用同进程、相同逐观测种子对比普通 `runner.infer_batch` 与分段计时回调，192 次生成 token 全部一致。

Panoramic 使用前视裁剪和时序候选帧构造的形状兼容输入；结果不代表真实全景导航或任务成功率。

## 实验方法

三个模型共用本目录的一套独立环境，各用单独进程测量：

| 配置名 | 模型 | 输出上限 / 优化 |
| --- | --- | --- |
| `low` | Qwen2.5-VL-3B-R2R-low-level | 1 token；SDPA + Inductor + CUDA Graph |
| `panoramic` | Qwen2.5-VL-3B-R2R-panoramic | 1 token；SDPA + Inductor + CUDA Graph |
| `navida` | NaViDA | 512 tokens；StaticCache 解码图，compile none |

数据：[StreamVLN-Trajectory-Data](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data) 的 R2R、RxR 公开训练轨迹，各按 numeric episode ID 取前 48 段，使用第一条指令。每段从满足 4 帧历史及前后邻帧要求的位置中均匀取 4 个，每模型、每数据集 **192 次调用**。三个模型使用相同位置；RxR 编号不连续，不能按 ID 1–48 取样。

每次调用构建独立的 4 帧历史快照，Qwen 历史响应来自记录的动作标签，NaViDA 只使用历史帧。Panoramic 将公开前视 RGB 裁为 960×240，并以邻近时间帧构造 4 个候选图像：这是**形状兼容输入，不是真实全景或官方候选导航边**。

统一 B=1、BF16、seed=42，正式测量 1 遍。计时从已解码 CPU RGB/状态开始，到 CPU 输出结束，包含预处理、模型执行、后处理及两端 CUDA 同步；磁盘读取/解码、模型加载和预热（含编译/图捕获）单独排除。**无需仿真**，不计算任务成功率、SR/SPL。

分段计时：`prefill_ms` 与 `decode_ms` 为 CUDA elapsed，`pure_inference_ms`（表中“纯推理”）为同一完整模型区间的同步墙钟时间。Qwen Low/Panoramic 的单 token 前向全部记入 prefill，decode 仅含动作 token 的 argmax；NaViDA 在首个多模态前向后划分，decode 包含后续全部自回归生成。Thor 的 CPU/GPU 共享物理内存，RSS 与 CUDA 内存不能相加。

## 运行

在本目录执行，依赖配套的 EmbodiInfer 源码和本平台已验证的 Torch 环境。`setup_env.py` 为本目录创建独立 `.venv`；`benchmark.py` 自带数据采样和计时代码；`prepare_data.py` 提取选中的导航图像。

```bash
# Thor: --platform thor；4090: --platform 4090，并选择一张空闲 GPU。
CUDA_VISIBLE_DEVICES=0 python3 setup_env.py --platform thor --runtime-python /path/to/verified-runtime/bin/python --inference-root /path/to/EmbodiInfer
# 修改 config.yaml 中的模型、数据路径；新结果默认写入 runs/。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py --validate-data
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py
```

Thor 使用 Torch 2.13.0+cu132，4090 使用 2.13.0+cu129，模型依赖分别见 `requirements-thor.txt` / `requirements-4090.txt`。`--config /path/to/config.yaml` 可使用另一份配置。选定数据缺失时会报错，不以随机输入替代。

默认依次启动三个模型；加 `--model low`、`--model panoramic` 或 `--model navida` 可单独运行。Qwen Low/Panoramic 需使用 Qwen2.5-VL processor metadata，不要使用共享模型目录中旧的 Qwen2VL processor 配置。

公开图像归档下载后，可只提取前 48 个 episode。RxR 分卷按 part0、part1 顺序传入：

```bash
.venv/bin/python prepare_data.py --annotations /data/R2R/annotations_v1-3.json --archives /data/R2R/images_v1-3.tar.gz --output-root /data/R2R --episodes 48
.venv/bin/python prepare_data.py --annotations /data/RxR/annotations.json --archives /data/RxR/images.tar.gz.part0 /data/RxR/images.tar.gz.part1 --output-root /data/RxR --episodes 48
```

Qwen2.5-VL-3B 与 NaViDA 的逐样本 eager / CUDA Graph 一致性、算子级与编译缓存检查见 [3B-navigation](../3B-navigation/README.md)。
