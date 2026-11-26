# LingBot-VLA Benchmark

## 实验结果（2026-09-08）

Thor MAXN 与一台单卡 RTX 4090 工作站分别完成全部 1,600 次调用。两台机器的 Python 源码、输入选择、精度、去噪步数和优化开关一致：B=1、BF16，eager attention，启用 Inductor、完整去噪循环 CUDA Graph 和 prefix 复用。这是两台设备上的 BF16 基线；模型专用原生优化与量化档尚未测量。

| 硬件 | 数据集 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thor | RoboTwin2.0 | 1,600 | 212.17 | 103.20 | 91.41 | 194.65 | 4.7131 | 5.1374 |
| RTX 4090 | RoboTwin2.0 | 1,600 | 88.36 | 41.00 | 41.14 | 82.17 | 11.3170 | 12.1697 |

E2E P50/P95/P99：Thor 208.94/231.47/242.99 ms，RTX 4090 87.26/92.20/97.13 ms。预热后及测量结束均为 1 个 LoopGraph、2 个编译图，正式测量中没有新增编译或图捕获。纯推理覆盖完整 prefix 编码和 10 步去噪；输出反归一化及关节布局还原在该区间外。10 个跨任务观测在相同种子下，分段计时路径与普通 `EngineCore.execute` 的 `50×14` 输出逐元素相同。

Thor 上对比 eager/SDPA（纯推理 193.47/196.78 ms）后保留 eager。两者的物理动作最大绝对差为 `0.0142653`，归一化模型状态最大绝对差 `0.015625` 低于 `test_lingbot_vla_matches_native_reference` 的 BF16 跨实现阈值 `0.06`。

## 实验方法

模型：[robbyant/lingbot-vla-4b-posttrain-robotwin](https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin)，revision `fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e`，不带 depth 的公开版本。使用官方 `robotwin.yaml` 关节布局与 `robotwin_50.json` 的 bounds_99 统计；该配置对应 14 维关节数据。`robbyant/robotwin-clean-and-aug-lerobot` 中的 EEF 数据为 16 维位姿，不能直接套用这份关节统计。

数据：[TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)，revision `785feb15aa4a4f532395ad2b1d2be5f28cb561ad`。取 `dataset/` 下字典序前 10 个任务的 `aloha-agilex_clean_50.zip`，每任务 numeric episode ID 前 10 段、每段含首尾均匀取 16 帧，共 **1,600 帧**，selection SHA256 `b2d25a3356ee01b328ad1ff8bb3808ff420c7c02a345c979a7fa14c477b327b1`；每段固定使用公开 `instructions/episodeN.json` 的第一条 `seen` 指令。前 10 个任务为 adjust_bottle、beat_block_hammer、blocks_ranking_rgb、blocks_ranking_size、click_alarmclock、click_bell、dump_bin_bigbin、grab_roller、handover_block、handover_mic。

读取 head/left/right RGB 和 `joint_action/vector` 当前帧；按官方映射组成 arm12 + pad2 + gripper2，再补至内部 75 维。图像保持宽高比缩放、左上补黑至 224×224，随后进入 Qwen2.5-VL processor。10 个去噪步，输出 50×75 后按原生布局切片、反归一化并还原为 **50×14** 物理关节动作。

与 [PI0.5 benchmark](../pi05-benchmark/README.md) 统一单卡、B=1、BF16、seed=42、float32 matmul precision=`highest`，预热 10 次、正式测量 1 遍。计时从已解码 CPU RGB/状态到 CPU 动作输出，包含预处理、GPU 传输、模型执行和后处理，两端 CUDA 同步；磁盘读取、JPEG 解码、模型加载和编译/图捕获预热在计时外。无需仿真，不计算任务成功率。与 LIBERO 模型的计时和采样规模一致，但数据集、三相机及双臂动作负载不同，不作为相同任务下的模型排名。

分段计时：`prefill_ms` 为视觉/文本/状态编码的 CUDA elapsed，`decode_ms` 为全部去噪步骤的 CUDA elapsed；`pure_inference_ms`（表中“纯推理”）为同一完整模型区间的同步墙钟时间，包含模型内部 CPU 调度。CUDA elapsed 包含流上的等待间隙，不是逐 kernel 时长相加。Thor 的 CPU/GPU 共享内存，RSS 与 CUDA 内存不相加。

## 运行

本目录使用独立 `.venv`，包含采样与计时，不导入其他 benchmark。Thor 使用已验证的 Torch 2.13.0+cu132，4090 使用 2.13.0+cu129。

```bash
CUDA_VISIBLE_DEVICES=0 python3 setup_env.py --platform thor --runtime-python /path/to/verified-runtime/bin/python --inference-root /path/to/EmbodiInfer
# 修改 config.yaml 模型、backbone、统计文件及数据路径。
.venv/bin/python prepare_data.py --rate 10M
.venv/bin/python prepare_assets.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py --validate-data
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py
```

`prepare_data.py` 先用 HF mirror，失败后用 Hugging Face，串行限速下载，只提取选中段的 HDF5/指令；不运行模拟器、不加载 pickle。四个采样字段均可调整，改变任务数或轨迹数后先重新准备数据。报告已存在会拒绝覆盖。

4090 使用相同代码，`--platform 4090`、`CUDA_VISIBLE_DEVICES=0`，通过 `--config /path/to/4090.yaml` 指定该机路径。
