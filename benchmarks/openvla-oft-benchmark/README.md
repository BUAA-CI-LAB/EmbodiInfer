# OpenVLA-OFT Benchmark

## 实验结果（2026-09-08）

Thor MAXN 与一台单卡 RTX 4090 工作站分别完成全部 1,600 次调用。B=1、BF16，启用 Inductor、CUDA Graph 和 prefix 复用。这是两台设备上的 BF16 基线；模型专用原生优化、量化档与竞品对比尚未测量。

| 硬件 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thor | 1,600 | 149.36 | 80.09 | 63.55 | 143.67 | 6.6951 | 6.9605 |
| RTX 4090 | 1,600 | 63.90 | 44.04 | 18.20 | 62.26 | 15.6499 | 16.0614 |

E2E P50/P95/P99：Thor 148.42/154.04/154.78 ms，RTX 4090 63.34/66.88/69.71 ms。两台设备预热后及测量结束均为 1 个 ForwardGraph、2 个编译图，正式测量中没有新增编译或图捕获。10 个跨任务观测在相同种子下，分段计时路径与普通 `EngineCore.execute` 的物理输出逐元素相同。

decode 是一次完整的 56 个动作 token 并行预测，动作 token 到物理 `8×7` 输出的转换在纯推理计时外。Thor 上对比 SDPA/eager（纯推理 142.94/144.59 ms，eager 改变了 1,680 个动作 token 中的 6 个）后保留 SDPA。

RTX 4090 加载时先在 CPU 将权重转为目标 BF16，再交给 EngineCore 搬入 GPU，避免 FP32 权重临时占满 24 GiB 显存；该改动发生在计时前，BF16 推理设置和数据采样与 Thor 一致。

## 实验方法

模型：[Haozhan72/Openvla-oft-SFT-libero10-trajall](https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero10-trajall)，revision `97d6e58bc5fbc402e85bab382e4206bb8e5f0db0`。使用主线适配的 RLinf categorical action-token checkpoint；单路 agentview RGB，不消费 wrist/状态。一次并行生成 56 个动作 token，反归一化输出 **8×7**；不是 L1 regression action head 的 OFT 权重。

数据沿用 [PI0.5 benchmark](../pi05-benchmark/README.md) 的 `yifengzhu-hf/LIBERO-datasets`，revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`，`libero_10/*.hdf5` 任务名字典序前 10 个、每任务 numeric demo ID 前 10 段、每段含首尾均匀取 16 帧，共 **1,600 帧**，selection SHA256 `92be73f628b54f35c5f0ece7ea0e6babc4a6e3b3a4fd43adc0662bbfa516eb69`。四个采样字段可在 `config.yaml` 调整。

统一单卡、B=1、BF16、seed=42、float32 matmul precision=`highest`、预热 10 次、正式测量 1 遍。输入为已解码 CPU RGB/状态；计时包含预处理、传入 GPU、模型推理、动作后处理和 CPU 输出，两端 CUDA 同步。文件读取和解码、加载、预热（包含编译/图捕获）不计入。测量中出现新图捕获或编译则报错。无需仿真，不测任务成功率。

分段计时：`prefill_ms` 包含 DINOv2/SigLIP 视觉编码、投影和 Llama 多模态 prompt prefill，产出 KV cache；`decode_ms` 包含基于该 KV 的一次并行动作 query forward 和类别选择，两者均为 CUDA elapsed。此 checkpoint 没有自回归循环，decode 不是单 token 解码延迟。`pure_inference_ms`（表中“纯推理”）为同一完整模型区间的同步墙钟时间，包含模型内部 CPU 调度。

## 运行

每个目录拥有独立 `.venv`，不导入其他 benchmark 目录。沿用本平台已验证 Torch 环境建立环境：Thor `2.13.0+cu132`，4090 `2.13.0+cu129`。

```bash
CUDA_VISIBLE_DEVICES=0 python3 setup_env.py --platform thor --runtime-python /path/to/verified-runtime/bin/python --inference-root /path/to/EmbodiInfer
# 修改 config.yaml 的模型、数据和输出路径；输出文件已存在时会拒绝覆盖。
.venv/bin/python prepare_assets.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py --validate-data
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py
```

4090 使用 `setup_env.py --platform 4090`，通过 `CUDA_VISIBLE_DEVICES` 选择一张空闲 GPU，并传入 `--config /path/to/4090.yaml` 设置该机路径。
