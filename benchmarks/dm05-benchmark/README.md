# DM0.5 Benchmark

## 实验方法

模型：[Dexmal/DM05-libero](https://huggingface.co/Dexmal/DM05-libero)，revision `25a8e0d38a8eaeaae41a44d7b4a2378fd8ce1088`。两路 RGB；EEF position3 + axis-angle3 + gripper2。遵循官方 LIBERO 配置 `add_state=false`，状态参与输入校验和归一化，不添加状态文本 token。10 个去噪步，输出 **10×7**。

数据沿用 [PI0.5 benchmark](../pi05-benchmark/README.md) 的 `yifengzhu-hf/LIBERO-datasets`，revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`，`libero_10/*.hdf5` 任务名字典序前 10 个、每任务 numeric demo ID 前 10 段、每段含首尾均匀取 16 帧，共 **1,600 帧**。选择规则、sample ID 和 selection SHA256 与 PI0.5/Cosmos 一致。四个采样字段可在 `config.yaml` 调整。

统一单卡、B=1、BF16、seed=42、float32 matmul precision=`highest`、预热 10 次、正式测量 1 遍。输入为已解码 CPU RGB/状态；计时包含预处理、传入 GPU、模型推理、动作后处理和 CPU 输出，两端 CUDA 同步。文件读取和解码、加载、预热（包含编译/图捕获）不计入正式延迟。无需仿真，不测任务成功率。

本次配置启用 Inductor 编译 `denoise_step`、完整去噪循环 CUDA Graph 和 prefix 复用；视觉、语言及动作注意力均使用 SDPA，并启用 Liger。语言/动作 attention 的 eager/SDPA 小样本比较后保留 SDPA，比较范围及数值条件见下文。正式报告记录实际 graph capture、compiler counters 和注意力配置；测量中出现新图捕获或编译则报错，不能把编译耗时混进稳态结果。

指标：mean/P50/P95/P99 延迟、calls/s、action slots/s、CUDA 峰值 allocated/reserved、进程峰值 RSS、加载/预热时间。calls/s = 完成调用数 / 总推理秒数；action slots/s 不等于机械臂控制频率。不同模型保留原生相机数量和动作长度，比较时同时列明。Thor 的 RSS 与 CUDA 内存不可相加。

本轮分段计时保留原端到端 `latency_ms`，额外记录 `model_timing_ms`：`prefill_ms` 为视觉/文本/状态编码的 CUDA elapsed，`decode_ms` 为全部去噪或 token 生成步骤的 CUDA elapsed，`gpu_inference_ms` 为二者合计；`pure_inference_ms` 为同一完整模型区间的同步墙钟时间，包含模型内部 CPU 调度及采样等待。输入预处理与输入 H2D 在计时前完成，动作反归一化、文本/动作解析及输出 D2H 在计时后执行。两段之间不插入同步；CUDA elapsed 包含流上的等待间隙，不是逐 kernel 时长相加。`model_calls_per_second` = 调用数 / 总纯推理墙钟时间，与原 E2E `calls_per_second` 分开报告。原 E2E 范围保留，但新测量含计时探针开销。新报告 schema 为 `rlinf_offline_performance_v2`，旧报告未测量的字段不回填。

## 运行

每个目录拥有独立 `.venv`，不导入其他 benchmark 目录。`--runtime-python` 须指向可导入固定版本 OpenDM 的运行环境：revision `e89fcbaa0408ca0fb04a410bfab50cc15eb73fdb`。该版本的 prefix cache 和时间条件接口与主线 DM0.5 适配层一致；9 月 1 日新增 FP32 mixed precision 后的接口不适用于此 BF16 实验环境。此模型的依赖表只补齐 benchmark 依赖，不代装 OpenDM 训练栈。启用 Liger RMSNorm/GeGLU/RoPE/LayerNorm，缺少实际 fused modules 时直接报错。沿用本平台已验证 Torch 环境建立环境：Thor `2.13.0+cu132`，4090 `2.13.0+cu129`。

```bash
CUDA_VISIBLE_DEVICES=0 python3 setup_env.py --platform thor --runtime-python /path/to/verified-runtime/bin/python --inference-root /path/to/EmbodiInfer
# 修改 config.yaml 的模型、数据和输出路径；输出文件已存在时会拒绝覆盖。
.venv/bin/python prepare_assets.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py --validate-data
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py
```

4090 使用 `setup_env.py --platform 4090`，`CUDA_VISIBLE_DEVICES=0` 选择单卡；传入 `--config /path/to/4090.yaml` 设置该机路径，测量代码与 Thor 相同。

## 实验结果

2026-09-08 在 Thor MAXN 和一台单卡 RTX 4090 工作站分别完成全部 1,600 次调用。Thor/4090 分别使用 Torch `2.13.0+cu132` / `2.13.0+cu129`，均使用 Transformers `5.3.0`、Liger `0.8.2`。EmbodiInfer 与 OpenDM 的实际 Python 源码摘要在两台机器上一致；其余条件同上。正式阶段仅有一个模型进程，下载已完成；两台机器均预热捕获 7 个 CUDA Graph，测量期间图数量和编译计数均未增加。

| 硬件 | 数据集 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thor | LIBERO-10 | 1,600 | 260.93 | 139.69 | 104.38 | 244.11 | 3.8325 | 4.0965 |
| RTX 4090 | LIBERO-10 | 1,600 | 140.37 | 86.23 | 46.40 | 132.67 | 7.1242 | 7.5378 |

4090 的视觉、语言和动作 attention 均为 SDPA，534 个 Liger 模块已在运行时确认，Inductor 和完整 LoopGraph 已启用。E2E P50/P95/P99 为 139.27/142.43/143.78 ms；均值、分位数、吞吐和采样摘要已从逐调用记录独立复算。4090 原始报告 (`reports/4090/dm05-libero10.json`) SHA256 为 `83a28aa95f1412549418d1ad0ec422017c73d5150dabc99652f05aa79f99e91b`；4090 路径一致性记录 (`reports/4090/timing-parity.json`) 中的 10 条动作逐位一致，最大绝对差为 0。

Prefill/decode 是 CUDA elapsed，纯推理是完整模型区间的同步墙钟时间。Thor 的 E2E 与纯推理平均相差 16.82 ms，包含本脚本的输入处理/传输、输出还原和测量开销。Thor 的 10 个跨任务观测在相同随机种子下，新计时路径与普通 `EngineCore.execute` 的 `10×7` 输出逐元素相同，最大绝对差为 0。

Thor attention 对比从上述 1,600 帧中均匀取 10 帧，预热 10 次、测量 3 遍，每帧固定 `torch.manual_seed(42 + i)`。两边均使用 BF16、`highest`、相同权重和 10 个去噪步，保留 Liger、编译及 CUDA Graph；仅切换语言/动作 attention，视觉 attention 保持 SDPA。SDPA/eager 纯推理均值分别为 **238.11/251.21 ms**，因此保留 SDPA。eager 与 SDPA 的物理动作最大绝对差为 `0.0078125`，也未满足预先固定的逐元素完全一致条件；未放宽阈值来选择候选。这是 B=1 的指定候选比较，未穷举全部 batch size 或优化组合。

Thor 原始报告 (`reports/thor/dm05-libero10.json`) 保留逐调用结果、完整配置、采样摘要、实际依赖和 OpenDM 源码摘要；attention 对比记录 (`reports/thor/attention/selection.json`) 关联两种候选的逐调用计时和数值检查数组。报告及后续统一快照不提交 Git；权重、数据和虚拟环境不进入快照。

本轮 8 个 benchmark 共用 20260908223127.tar.gz (`../snapshots/20260908223127.tar.gz`) 和一份 SHA256 (`../snapshots/20260908223127.tar.gz.sha256`)。包内只包含必要脚本、依赖文件、README、实际运行配置、14 组 Thor 与 14 组 RTX 4090 分段计时报告，以及相关数值核验记录。PI0.5 使用修复 tokenizer 后的新结果；历史报告、模型框架、权重、原始数据和虚拟环境不进入本包。报告及快照不提交 Git。

配套 EmbodiInfer 源码基线为 `72cba6e4ae05312bd93c69918709f9b792869e61`，使用本包附带的 benchmark 脚本。报告旁的 `*-config.json` 保存该次实际运行配置；复跑时调整本机模型、数据和输出路径，再传入 `--config`。两平台的采样、配置和源码核验见 协议核验记录 (`reports/4090/cross-hardware-protocol.json`)。
