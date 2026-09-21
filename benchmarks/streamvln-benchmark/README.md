# StreamVLN Benchmark

## 实验方法

模型：[StreamVLN](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3)。数据：[StreamVLN-Trajectory-Data](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data) 的 R2R、RxR 公开训练轨迹，各按 numeric episode ID 取前 48 段，使用第一条指令，回放全部 RGB：R2R 2,997 帧、RxR 3,879 帧。R2R 是 ID 1–48，RxR 编号不连续，不能替换为 ID 1–48。

逐帧顺序推理，同一 episode 保留模型历史，边界重置；下一帧来自录制轨迹。启用 CUDA Graph、历史特征缓存和 fast action decode。预热 33 帧，使用所选轨迹中首个足够长的 episode（R2R ID 1、RxR ID 13）；每次调用输出容量为 4 个 action slots。

统一 B=1、BF16、seed=42，正式测量 1 遍。计时从已解码 CPU RGB/状态开始，到 CPU 输出结束，包含预处理、模型执行、后处理及两端 CUDA 同步；磁盘读取/解码、模型加载和预热（含编译/图捕获）单独排除。**无需仿真**，不计算任务成功率、SR/SPL。

指标：mean / P50 / P95 / P99 延迟、calls/s、action slots/s、tokens/s、CUDA 峰值 allocated/reserved、进程峰值 RSS，以及加载/预热时间。吞吐为总输出数除以总推理时间；action slots/s 不等于机器人控制频率。Thor 的 CPU/GPU 共享物理内存，RSS 与 CUDA 内存不能相加。

本轮分段计时保留原端到端 `latency_ms`，额外记录 `model_timing_ms`：`prefill_ms` 为视觉/文本/状态编码的 CUDA elapsed，`decode_ms` 为全部去噪或 token 生成步骤的 CUDA elapsed，`gpu_inference_ms` 为二者合计；`pure_inference_ms` 为同一完整模型区间的同步墙钟时间，包含模型内部 CPU 调度及采样等待。输入预处理与输入 H2D 在计时前完成，动作反归一化、文本/动作解析及输出 D2H 在计时后执行。两段之间不插入同步；CUDA elapsed 包含流上的等待间隙，不是逐 kernel 时长相加。`model_calls_per_second` = 调用数 / 总纯推理墙钟时间，与原 E2E `calls_per_second` 分开报告。原 E2E 范围保留，但新测量含计时探针开销。新报告 schema 为 `rlinf_offline_performance_v2`，旧报告未测量的字段不回填。

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

公开图像归档下载后，可只提取前 48 个 episode。RxR 分卷按 part0、part1 顺序传入：

```bash
.venv/bin/python prepare_data.py --annotations /data/R2R/annotations_v1-3.json --archives /data/R2R/images_v1-3.tar.gz --output-root /data/R2R --episodes 48
.venv/bin/python prepare_data.py --annotations /data/RxR/annotations.json --archives /data/RxR/images.tar.gz.part0 /data/RxR/images.tar.gz.part1 --output-root /data/RxR --episodes 48
```

## 最终结果

2026-09-08 在 Thor MAXN 完成 R2R 的全部 2,997 次调用及 RxR 的全部 3,879 次调用。一台单卡 RTX 4090 工作站也完成相同的全部轨迹，本表更新为此次分段计时结果。两平台使用相同权重、输入选择和生成设置，但 CUDA 构建与驱动不同，结果表示两套实际环境；未穷举所有 batch size 或优化组合。

| 模型 / 数据 | 硬件 / 日期 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| streamvln-r2r | Thor / 09-08 | 2,997 | 487.32 | 103.77 | 344.26 | 448.11 | 2.0521 | 2.2316 |
| streamvln-r2r | RTX 4090 / 09-08 | 2,997 | 161.52 | 41.86 | 102.10 | 144.00 | 6.1910 | 6.9446 |
| streamvln-rxr | Thor / 09-08 | 3,879 | 495.08 | 111.19 | 344.77 | 456.04 | 2.0199 | 2.1928 |
| streamvln-rxr | RTX 4090 / 09-08 | 3,879 | 163.33 | 43.18 | 102.40 | 145.63 | 6.1226 | 6.8668 |

Thor R2R 的 E2E P50/P95/P99 为 447.62/966.36/1114.26 ms，RxR 为 453.08/996.38/1215.70 ms。R2R/RxR 的完整模型 CUDA elapsed 分别为 448.03/455.96 ms，同一区间的同步墙钟为 448.11/456.04 ms；decode 覆盖该次调用的全部动作生成步骤。E2E 与纯推理平均相差 39.20/39.03 ms，包含图像/提示词处理、输入传输、输出解析、历史状态更新和计时开销。

R2R selection SHA256 为 `f619f18b95276938068d2f5c3bb86a7e837d17c61024b7935f5d7eb6ccf7246b`，RxR 为 `5478777de23c0065d77d7a84d3690779a5fbf2c4996bb54607ea825c26df837f`。两组均预热 33 帧后冻结图捕获，实际状态如下；各列数字依次对应 vision/prefill/decode。

| Thor 数据集 | 图数量 | 正式回放数 | eager 回退数 | 与旧 Thor 报告相同的动作输出 hash |
| --- | --- | --- | --- | --- |
| R2R | 1/9/1 | 2997/2795/3493 | 0/202/0 | 2997/2997 |
| RxR | 1/7/1 | 3879/3336/4447 | 0/543/0 | 3879/3879 |

该优化配置允许未捕获的 prefill 形状走 eager，不能把“启用 CUDA Graph”解释为每次 prefill 都回放图。上述 Thor 图数量、回放和回退计数均与各自旧报告相同；全部 6,876 个逐调用 action chunk 的 SHA256 与旧报告一致。

旧 Thor R2R/RxR E2E 为 491.75/502.57 ms，仅作为历史参照；本轮快照保留当前 Thor 分段计时报告，不据旧值补算未测量的纯推理字段。当前 Thor R2R/RxR 原始 JSON SHA256 分别为 `cf1bc352a3132ccd0c70c74b7a4271361c9f3a09d736fb4e5ca09f6cf4a4edec` / `ddd43cc3f9b2906705a49a590a8293768ee2842a9cafaa079d7726b9451663fd`，包含逐调用数据、完整配置、采样与源码摘要、内存和图回放状态。

4090 的图捕获数量和正式回放统计如下，各列依次为 vision/prefill/decode。

| 4090 数据集 | 图数量 | 正式回放数 | eager 回退数 | E2E P50/P95/P99 ms |
| --- | --- | --- | --- | --- |
| R2R | 1/9/1 | 2997/2795/3503 | 0/202/0 | 145.82/306.33/365.93 |
| RxR | 1/7/1 | 3879/3336/4447 | 0/543/0 | 148.19/307.79/400.58 |

4090 另在每组正式轨迹的前 4 个 episode、每段至多 64 帧上，对比普通 `encode_prefix + decoder.decode` 与分段计时路径；覆盖 episode 边界和第 32 帧的历史窗口切换，逐调用 token、动作、cache 长度及停止原因全部一致。该配对检查仅验证数值；表中性能仍来自全 48 段正式回放。

R2R：168 次配对核验；正式性能报告 SHA256 `f4810d8c62a7261f375de412e622131cf84cefca2fa9b8117a25aaf4d4b43798`；RxR：116 次配对核验；正式性能报告 SHA256 `7f1b0a22c1d9a1b1c92b02626cd0638f88f1e4ddbefc4e24673d9d80bb8385b3`。核验记录：R2R (`reports/4090/streamvln-r2r-timing-parity.json`)、RxR (`reports/4090/streamvln-rxr-timing-parity.json`)。

两组在启动阶段均出现过显存分配警告，进程随后继续完成测试。结合日志时间、模型加载及预热耗时，已确认警告发生在正式计时之前；最终图数量与 Thor 一致，vision/decode 没有 eager 回退，见 启动显存核查 (`reports/4090/startup-memory-check.json`)。

本轮 8 个 benchmark 共用 20260908223127.tar.gz (`../snapshots/20260908223127.tar.gz`) 和一份 SHA256 (`../snapshots/20260908223127.tar.gz.sha256`)。包内只包含必要脚本、依赖文件、README、实际运行配置、14 组 Thor 与 14 组 RTX 4090 分段计时报告，以及相关数值核验记录。PI0.5 使用修复 tokenizer 后的新结果；历史报告、模型框架、权重、原始数据和虚拟环境不进入本包。报告及快照不提交 Git。

配套 EmbodiInfer 源码基线为 `72cba6e4ae05312bd93c69918709f9b792869e61`，使用本包附带的 benchmark 脚本。报告旁的 `*-config.json` 保存该次实际运行配置；复跑时调整本机模型、数据和输出路径，再传入 `--config`。两平台的采样、配置和源码核验见 协议核验记录 (`reports/4090/cross-hardware-protocol.json`)。
