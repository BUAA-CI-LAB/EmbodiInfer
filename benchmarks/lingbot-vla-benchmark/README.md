# LingBot-VLA Benchmark

## 实验方法

模型：[robbyant/lingbot-vla-4b-posttrain-robotwin](https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin)，revision `fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e`，不带 depth 的公开版本。使用官方 `robotwin.yaml` 关节布局与 `robotwin_50.json` 的 bounds_99 统计；该配置对应 14 维关节数据。`robbyant/robotwin-clean-and-aug-lerobot` 中的 EEF 数据为 16 维位姿，不能直接套用这份关节统计。

数据：[TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)，revision `785feb15aa4a4f532395ad2b1d2be5f28cb561ad`。取 `dataset/` 下字典序前 10 个任务的 `aloha-agilex_clean_50.zip`，每任务 numeric episode ID 前 10 段、每段含首尾均匀取 16 帧，共 **1,600 帧**；每段固定使用公开 `instructions/episodeN.json` 的第一条 `seen` 指令。前 10 个任务为 adjust_bottle、beat_block_hammer、blocks_ranking_rgb、blocks_ranking_size、click_alarmclock、click_bell、dump_bin_bigbin、grab_roller、handover_block、handover_mic。

读取 head/left/right RGB 和 `joint_action/vector` 当前帧；按官方映射组成 arm12 + pad2 + gripper2，再补至内部 75 维。图像保持宽高比缩放、左上补黑至 224×224，随后进入 Qwen2.5-VL processor。10 个去噪步，输出 50×75 后按原生布局切片、反归一化并还原为 **50×14** 物理关节动作。

与 [PI0.5 benchmark](../pi05-benchmark/README.md) 统一单卡、B=1、BF16、seed=42、float32 matmul precision=`highest`，预热 10 次、正式测量 1 遍。计时从已解码 CPU RGB/状态到 CPU 动作输出，包含预处理、GPU 传输、模型执行和后处理，两端 CUDA 同步；磁盘读取、JPEG 解码、模型加载和编译/图捕获预热在计时外。无需仿真，不计算任务成功率。

启用 Inductor、完整去噪循环 CUDA Graph、prefix 复用；Thor 的 eager/SDPA 小样本比较后保留 eager，比较范围和数值检查见下文。报告记录实际 compiler counters 和图捕获状态。与 LIBERO 模型的计时和采样规模一致，但数据集、三相机及双臂动作负载不同，不作为相同任务下的模型排名。

指标：mean/P50/P95/P99 延迟、calls/s、action slots/s、CUDA 峰值 allocated/reserved、进程峰值 RSS、加载/预热时间。吞吐用完成调用数除以总推理秒数；action slots/s 不等于机器人控制频率。Thor 的 CPU/GPU 共享内存，RSS 与 CUDA 内存不相加。

本轮分段计时保留原端到端 `latency_ms`，额外记录 `model_timing_ms`：`prefill_ms` 为视觉/文本/状态编码的 CUDA elapsed，`decode_ms` 为全部去噪或 token 生成步骤的 CUDA elapsed，`gpu_inference_ms` 为二者合计；`pure_inference_ms` 为同一完整模型区间的同步墙钟时间，包含模型内部 CPU 调度及采样等待。输入预处理与输入 H2D 在计时前完成，动作反归一化、文本/动作解析及输出 D2H 在计时后执行。两段之间不插入同步；CUDA elapsed 包含流上的等待间隙，不是逐 kernel 时长相加。`model_calls_per_second` = 调用数 / 总纯推理墙钟时间，与原 E2E `calls_per_second` 分开报告。原 E2E 范围保留，但新测量含计时探针开销。新报告 schema 为 `rlinf_offline_performance_v2`，旧报告未测量的字段不回填。

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

## 实验结果

2026-09-08 在 Thor MAXN 和一台单卡 RTX 4090 工作站分别完成全部 1,600 次调用。两台机器的实际 Python 源码、输入选择、精度、去噪步数和优化开关一致。使用上述原生三相机/关节数据，selection SHA256 为 `b2d25a3356ee01b328ad1ff8bb3808ff420c7c02a345c979a7fa14c477b327b1`。预热后及测量结束均为 1 个 LoopGraph、2 个编译图（`calls_captured=5870`），没有在正式测量中新增编译或图捕获。

| 硬件 | 数据集 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thor | RoboTwin2.0 | 1,600 | 212.17 | 103.20 | 91.41 | 194.65 | 4.7131 | 5.1374 |
| RTX 4090 | RoboTwin2.0 | 1,600 | 88.36 | 41.00 | 41.14 | 82.17 | 11.3170 | 12.1697 |

4090 使用 Torch `2.13.0+cu129` 和 Transformers `5.3.0`，沿用 eager attention，启用 Inductor、完整 LoopGraph 和 prefix 复用。E2E P50/P95/P99 为 87.26/92.20/97.13 ms；均值、分位数、吞吐和采样摘要已从逐调用记录独立复算。4090 原始报告 (`reports/4090/lingbot-vla-robotwin.json`) SHA256 为 `d041afcc891d145464111ff336227e92323493744c198325816b886a0f287001`；4090 路径一致性记录 (`reports/4090/timing-parity.json`) 中的 10 条动作逐位一致，最大绝对差为 0。

Thor E2E P50/P95/P99 为 208.94/231.47/242.99 ms。纯推理覆盖完整 prefix 编码和 10 步去噪；输出反归一化及关节布局还原在该区间外。10 个跨任务观测在相同种子下，分段计时路径与普通 `EngineCore.execute` 的 `50×14` 输出逐元素相同，最大绝对差为 0。

Thor attention 对比从上述 1,600 帧中均匀取 10 帧，预热 10 次、测量 3 遍，每帧固定 `torch.manual_seed(42 + i)`；两边保持 B=1、BF16、`highest`、相同权重、编译及 CUDA Graph。eager/SDPA 纯推理均值分别为 **193.47/196.78 ms**，因此保留 eager。完整去噪后归一化模型状态最大绝对差为 `0.015625`，低于既有 `test_lingbot_vla_matches_native_reference` 的 BF16 跨实现阈值 `0.06`；物理动作最大绝对差为 `0.0142653`。SDPA 可能改变浮点累加顺序，该比较不声称两种 attention 逐位相同。选择规则预先固定为“数值检查通过且纯推理均值至少降低 2% 才切换”；这是该小样本下的选择，不代表穷举全部吞吐配置。

Thor 原始报告 (`reports/thor/lingbot-vla-robotwin.json`) 包含逐调用数据和完整条件，SHA256 为 `6cb71a5661451bdb032af4846a12e8c31a08243954b64645c75c838f9674abcd`；attention 对比记录 (`reports/thor/attention/selection.json`) 关联两种候选的逐调用计时和数值检查数组。统一快照只保留最终配置、必要脚本和最终报告；权重、原始数据和虚拟环境不进入快照，报告及快照不提交 Git。

本轮 8 个 benchmark 共用 20260908223127.tar.gz (`../snapshots/20260908223127.tar.gz`) 和一份 SHA256 (`../snapshots/20260908223127.tar.gz.sha256`)。包内只包含必要脚本、依赖文件、README、实际运行配置、14 组 Thor 与 14 组 RTX 4090 分段计时报告，以及相关数值核验记录。PI0.5 使用修复 tokenizer 后的新结果；历史报告、模型框架、权重、原始数据和虚拟环境不进入本包。报告及快照不提交 Git。

配套 EmbodiInfer 源码基线为 `72cba6e4ae05312bd93c69918709f9b792869e61`，使用本包附带的 benchmark 脚本。报告旁的 `*-config.json` 保存该次实际运行配置；复跑时调整本机模型、数据和输出路径，再传入 `--config`。两平台的采样、配置和源码核验见 协议核验记录 (`reports/4090/cross-hardware-protocol.json`)。
