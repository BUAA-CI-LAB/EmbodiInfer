# StreamVLN Quantization Benchmark

仅展示 **batch=1** 下完整跑完的最终配置与结果。按 `设备/精度/` 查看结果，例如 `agx-orin/int8/`。

| 设备 | BF16 | FP8 | INT8 | NVFP4 |
|---|---|---|---|---|
| AGX Orin 32G | [结果](agx-orin/bf16/README.md) | — | [结果](agx-orin/int8/README.md) | — |
| Thor | [结果](thor/bf16/README.md) | [结果](thor/fp8/README.md) | — | [结果](thor/nvfp4/README.md) |
| RTX 4090 | [结果](4090/bf16/README.md) | [结果](4090/fp8/README.md) | — | — |

模型为 `mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3`。数据为 `cywan/StreamVLN-Trajectory-Data` 公开录制训练轨迹：R2R/RxR 各按数字 episode ID 取前 48 段，使用第一条指令并回放全部 RGB，共 **2,997 / 3,879 条观测**。

seed=42，逐帧推理，各 episode 保留自身生成历史，边界重置。预热 33 帧；联合运行 R2R/RxR 使用 R2R ID 1，独立 RxR 使用 ID 13，正式回放前重置历史。最多生成 128 token，decode block=4，输出容量为 4 个 action slots。纯离线回放，无仿真或导航 SR/SPL。

每个设备/精度目录仅保留一份开启可用优化的 `config.json`，以及对应的 README 和最终全量结果。必要脚本仅放在本模型 benchmark 根目录。

- `config.json`：保留原测试机器的模型和数据路径；新输出写入同目录的 `runs/`。
- `*.result.json`：最终原始结果，包含逐条动作、计时、环境与实际优化信息；仅在本地和 snapshot 中保存。

E2E 是已解码 CPU 观测到 CPU 动作的完整耗时；Forward 是 GPU 输入就绪后的完整模型调用。两者均包含全部去噪或生成步骤，E2E 额外包含预处理、传输和后处理。加载、预热、磁盘读取及图像解码不计入。吞吐为观测数除以总 E2E 时间，内存为峰值 CUDA allocated。

FP8 使用 native W8A8 后端，4090 为逐通道权重/逐行激活 scale，Thor 为 tensorwise scale。INT8/NVFP4 也使用 native 后端。

Orin 为当时系统环境下的测量，尚未证明资源独占；后期资源监控记录到系统换页。

BF16 开启语言 prefill 编译；量化配置关闭语言 prefill 编译。所有配置均开启视觉/decode CUDA Graph、历史特征缓存和 fast action decode，KV 缓存为 32K。

Forward 包含视觉到完整生成结束、文本/动作解析之前的同步时间，以及生成流程内部的 token 传输和历史调度；不能解释为纯 GPU kernel 时间之和。动作比较包含生成历史的累积差异。

每个模型使用独立虚拟环境。在本目录运行：

```bash
python setup_env.py --runtime-python /path/to/working/python --inference-root /path/to/inference
.venv/bin/python benchmark.py --config agx-orin/int8/config.json --validate-data
.venv/bin/python benchmark.py --config agx-orin/int8/config.json
.venv/bin/python compare.py agx-orin/bf16/rxr.result.json agx-orin/int8/rxr.result.json --output agx-orin/int8/runs/comparison.json
```

上例比较最终 BF16 与 INT8 输出；两者 prefill 编译配置不同。

| 设备 | 已测环境 |
|---|---|
| 4090 | Python 3.12，Torch 2.13.0+cu129 |
| Thor | Python 3.12，Torch 2.13.0+cu132 |
| AGX Orin | 主机 Python 3.10，Torch 2.8.0+CUDA 12.6 |

三端使用 Transformers 4.51.3。每份配置均包含 R2R/RxR；4090、Thor 在同一进程依次回放，Orin 使用 `isolate_datasets: true` 自动分进程回放，保持各自原测量的预热来源。可用 `--dataset R2R` 或 `--dataset RxR` 单独选择数据集。

结果、日志、虚拟环境和压缩包均由 `.gitignore` 排除，提交代码时只包含脚本、配置和 README。
