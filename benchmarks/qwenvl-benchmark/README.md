# QwenVL / NaViDA Benchmark

## 实验方法

三个模型共用本目录的一套独立环境，各用单独进程测量：

| 配置名 | 模型 | 输出上限 / 优化 |
| --- | --- | --- |
| `low` | Qwen2.5-VL-3B-R2R-low-level | 1 token；SDPA + Inductor + CUDA Graph |
| `panoramic` | Qwen2.5-VL-3B-R2R-panoramic | 1 token；SDPA + Inductor + CUDA Graph |
| `navida` | NaViDA | 512 tokens；StaticCache 解码图，compile none |

数据：[StreamVLN-Trajectory-Data](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data) 的 R2R、RxR 公开训练轨迹，各按 numeric episode ID 取前 48 段，使用第一条指令。每段从满足 4 帧历史及前后邻帧要求的位置中均匀取 4 个，每模型、每数据集 **192 次调用**。三个模型使用相同位置；RxR 编号不连续，不能按 ID 1–48 取样。

每次调用构建独立的 4 帧历史快照，Qwen 历史响应来自记录的动作标签，NaViDA 只使用历史帧。先将全部 192 个输入预热一遍，再正式测量。Panoramic 将公开前视 RGB 裁为 960×240，并以邻近时间帧构造 4 个候选图像：这是**形状兼容输入，不是真实全景或官方候选导航边**。

统一 B=1、BF16、seed=42，正式测量 1 遍。计时从已解码 CPU RGB/状态开始，到 CPU 输出结束，包含预处理、模型执行、后处理及两端 CUDA 同步；磁盘读取/解码、模型加载和预热（含编译/图捕获）单独排除。**无需仿真**，不计算任务成功率、SR/SPL。

指标：mean / P50 / P95 / P99 延迟、calls/s、action slots/s、tokens/s、CUDA 峰值 allocated/reserved、进程峰值 RSS，以及加载/预热时间。吞吐为总输出数除以总推理时间；action slots/s 不等于机器人控制频率。Thor 的 CPU/GPU 共享物理内存，RSS 与 CUDA 内存不能相加。

本轮分段计时保留原端到端 `latency_ms`，额外记录 `model_timing_ms`：`prefill_ms` 为视觉/文本/状态编码的 CUDA elapsed，`decode_ms` 为全部去噪或 token 生成步骤的 CUDA elapsed，`gpu_inference_ms` 为二者合计；`pure_inference_ms` 为同一完整模型区间的同步墙钟时间，包含模型内部 CPU 调度及采样等待。输入预处理与输入 H2D 在计时前完成，动作反归一化、文本/动作解析及输出 D2H 在计时后执行。两段之间不插入同步；CUDA elapsed 包含流上的等待间隙，不是逐 kernel 时长相加。`model_calls_per_second` = 调用数 / 总纯推理墙钟时间，与原 E2E `calls_per_second` 分开报告。原 E2E 范围保留，但新测量含计时探针开销。新报告 schema 为 `rlinf_offline_performance_v2`，旧报告未测量的字段不回填。

Qwen Low/Panoramic 的单 token 前向全部记入 prefill，decode 仅含动作 token 的 argmax；NaViDA 则在首个多模态前向后划分，decode 包含后续全部自回归生成。

## 运行

在本目录执行，依赖配套的 EmbodiInfer 源码和本平台已验证的 Torch 环境。2026-09-07 的旧结果基于 `b86cce30ba57`；本轮分段计时需使用当前工作树中同时更新的 EmbodiInfer，实际 Python 源码摘要记录在报告的 `environment.source_python_sha256`。`setup_env.py` 为本目录创建独立 `.venv`；`benchmark.py` 自带数据采样和计时代码；`prepare_data.py` 提取选中的导航图像。

```bash
# Thor: --platform thor；4090: --platform 4090，并选择一张空闲 GPU。
CUDA_VISIBLE_DEVICES=0 python3 setup_env.py --platform thor --runtime-python /path/to/verified-runtime/bin/python --inference-root /path/to/EmbodiInfer
# 修改 config.yaml 中的模型、数据路径；新结果默认写入 runs/。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py --validate-data
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py
```

Thor 使用 Torch 2.13.0+cu132，4090 使用 2.13.0+cu129，模型依赖分别见 `requirements-thor.txt` / `requirements-4090.txt`。`--config /path/to/config.yaml` 可使用另一份配置。选定数据缺失时会报错，不以随机输入替代。

默认依次启动三个模型；加 `--model low`、`--model panoramic` 或 `--model navida` 可单独运行。

公开图像归档下载后，可只提取前 48 个 episode。RxR 分卷按 part0、part1 顺序传入：

```bash
.venv/bin/python prepare_data.py --annotations /data/R2R/annotations_v1-3.json --archives /data/R2R/images_v1-3.tar.gz --output-root /data/R2R --episodes 48
.venv/bin/python prepare_data.py --annotations /data/RxR/annotations.json --archives /data/RxR/images.tar.gz.part0 /data/RxR/images.tar.gz.part1 --output-root /data/RxR --episodes 48
```

## 最终结果

2026-09-08 在 Thor MAXN 完成本轮全部正式调用，并在一台单卡 RTX 4090 工作站补测同一协议。两台机器的全部 6 组结果均已完成；下表只纳入新的分段计时报告。两平台使用相同权重、输入选择和生成设置，但 CUDA 构建与驱动不同；结果表示各自实际环境，未穷举所有 batch size 或优化组合。

| 模型 / 数据 | 硬件 / 日期 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low-r2r | Thor / 09-08 | 192 | 212.51 | 157.44 | 0.01 | 157.50 | 4.7056 | 6.3494 |
| low-r2r | RTX 4090 / 09-08 | 192 | 118.75 | 76.27 | 0.01 | 76.31 | 8.4210 | 13.1041 |
| low-rxr | Thor / 09-08 | 192 | 226.52 | 170.03 | 0.01 | 170.08 | 4.4146 | 5.8796 |
| low-rxr | RTX 4090 / 09-08 | 192 | 124.88 | 82.17 | 0.01 | 82.21 | 8.0076 | 12.1639 |
| navida-r2r | Thor / 09-08 | 192 | 1145.61 | 323.83 | 721.24 | 1045.15 | 0.8729 | 0.9568 |
| navida-r2r | RTX 4090 / 09-08 | 192 | 377.54 | 98.19 | 234.25 | 332.49 | 2.6487 | 3.0076 |
| navida-rxr | Thor / 09-08 | 192 | 1131.33 | 318.67 | 710.87 | 1029.60 | 0.8839 | 0.9713 |
| navida-rxr | RTX 4090 / 09-08 | 192 | 376.52 | 102.07 | 229.15 | 331.28 | 2.6559 | 3.0186 |
| panoramic-r2r | Thor / 09-08 | 192 | 773.61 | 642.71 | 0.02 | 642.79 | 1.2926 | 1.5557 |
| panoramic-r2r | RTX 4090 / 09-08 | 192 | 398.82 | 282.24 | 0.01 | 282.30 | 2.5074 | 3.5423 |
| panoramic-rxr | Thor / 09-08 | 192 | 777.25 | 644.03 | 0.02 | 644.10 | 1.2866 | 1.5526 |
| panoramic-rxr | RTX 4090 / 09-08 | 192 | 397.09 | 282.46 | 0.01 | 282.52 | 2.5183 | 3.5396 |

每组先预热全部 192 个输入，随后正式测量 192 次。Low/Panoramic 保留 SDPA、Inductor 和单 token 前向图，NaViDA 保留 StaticCache 自回归解码图。以下列出已经完成正式计时的各组；NaViDA 每次首个 token 来自 prefill，所以解码图回放数等于总生成 token 数减去调用数。

| 硬件 | 模型 / 数据 | 图数量（测量前→后） | 正式图回放数 | 总生成 tokens | 解析失败数 |
| --- | --- | --- | ---: | ---: | ---: |
| Thor | low-r2r | 1→1 | 192 | 192 | 0 |
| RTX 4090 | low-r2r | 1→1 | 192 | 192 | 0 |
| Thor | low-rxr | 3→3 | 192 | 192 | 0 |
| RTX 4090 | low-rxr | 3→3 | 192 | 192 | 0 |
| Thor | navida-r2r | 1→1 | 3604 | 3796 | 0 |
| RTX 4090 | navida-r2r | 1→1 | 3605 | 3797 | 0 |
| Thor | navida-rxr | 1→1 | 3547 | 3739 | 0 |
| RTX 4090 | navida-rxr | 1→1 | 3527 | 3719 | 0 |
| Thor | panoramic-r2r | 1→1 | 192 | 192 | 0 |
| RTX 4090 | panoramic-r2r | 1→1 | 192 | 192 | 0 |
| Thor | panoramic-rxr | 1→1 | 192 | 192 | 0 |
| RTX 4090 | panoramic-rxr | 1→1 | 192 | 192 | 0 |

测量中没有新图捕获；Low/Panoramic 的编译次数和缓存项数保持不变，编译失败数为 0。输出 SHA256 比较覆盖完整生成序列；不同摘要本身不量化数值误差。NaViDA 延迟包含直到 EOS 或 512-token 上限的全部生成，不能解释为单个 decode step。

Panoramic 仍使用前视裁剪和时序候选帧构造的形状兼容输入；结果不代表真实全景导航或任务成功率。

Thor low-r2r 有 5/192 个输出摘要与旧报告不同；同一进程中按相同逐观测种子，对比当前普通 `runner.infer_batch` 与实际分段计时回调，192 次生成 token 全部一致，见调用一致性记录 (`reports/thor/low-r2r-timing-parity.json`)。该检查针对当前两条调用路径，不宣称跨进程或跨报告的输出完全一致。

校验对每个观测的两条路径均重置 seed=42+i；正式测量在每遍开始重置 seed=42。该校验运行与正式报告另有 6/192 个输出摘要不同；当前只用同一进程、相同种子的成对比较证明调用路径一致，跨运行差异来源尚未确定。

Thor panoramic-r2r 有 6/192 个输出摘要与旧报告不同；同一进程中按相同逐观测种子，对比当前普通 `runner.infer_batch` 与实际分段计时回调，192 次生成 token 全部一致，见调用一致性记录 (`reports/thor/panoramic-r2r-timing-parity.json`)。该检查针对当前两条调用路径，不宣称跨进程或跨报告的输出完全一致。

校验对每个观测的两条路径均重置 seed=42+i；正式测量在每遍开始重置 seed=42。该校验运行与正式报告另有 5/192 个输出摘要不同；当前只用同一进程、相同种子的成对比较证明调用路径一致，跨运行差异来源尚未确定。

4090 已完成各组的均值、分位数、吞吐及采样摘要均已独立复算。每组另用同进程、相同逐观测种子对比普通 `runner.infer_batch` 与实际分段计时回调，192 条生成 token 全部一致，见 low-r2r (`reports/4090/low-r2r-timing-parity.json`)、low-rxr (`reports/4090/low-rxr-timing-parity.json`)、navida-r2r (`reports/4090/navida-r2r-timing-parity.json`)、navida-rxr (`reports/4090/navida-rxr-timing-parity.json`)、panoramic-r2r (`reports/4090/panoramic-r2r-timing-parity.json`)、panoramic-rxr (`reports/4090/panoramic-rxr-timing-parity.json`)。Qwen Low/Panoramic 使用与 Thor 相同的 Qwen2.5-VL processor metadata；共享模型目录中的旧 Qwen2VL processor 配置未用于本轮测量，实际配置与摘要见 processor 核验 (`reports/4090/processor-metadata-check.json`)。

原始报告包含逐调用 E2E/分段计时、P50/P95/P99、完整配置、样本与源码摘要、内存及实际图运行状态。旧 4090 报告留存为历史记录，不与本次 RTX 4090 分段计时混用。

本轮 8 个 benchmark 共用 20260908223127.tar.gz (`../snapshots/20260908223127.tar.gz`) 和一份 SHA256 (`../snapshots/20260908223127.tar.gz.sha256`)。包内只包含必要脚本、依赖文件、README、实际运行配置、14 组 Thor 与 14 组 RTX 4090 分段计时报告，以及相关数值核验记录。PI0.5 使用修复 tokenizer 后的新结果；历史报告、模型框架、权重、原始数据和虚拟环境不进入本包。报告及快照不提交 Git。

配套 EmbodiInfer 源码基线为 `72cba6e4ae05312bd93c69918709f9b792869e61`，使用本包附带的 benchmark 脚本。报告旁的 `*-config.json` 保存该次实际运行配置；复跑时调整本机模型、数据和输出路径，再传入 `--config`。还需按报告的 `selected_model` 显式传入 `--model low`、`--model panoramic` 或 `--model navida`。两平台的采样、配置和源码核验见 协议核验记录 (`reports/4090/cross-hardware-protocol.json`)。
