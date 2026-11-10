# DM0.5 Benchmark

## 实验结果（2026-09-08）

Thor MAXN 与一台单卡 RTX 4090 工作站分别完成全部 1,600 次调用。Thor/4090 分别使用 Torch `2.13.0+cu132` / `2.13.0+cu129`，均使用 Transformers `5.3.0`、Liger `0.8.2`；EmbodiInfer 与 OpenDM 的 Python 源码摘要在两台机器上一致。这是两台设备上的 BF16 基线；模型专用原生优化与量化档尚未测量。

| 硬件 | 数据集 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thor | LIBERO-10 | 1,600 | 260.93 | 139.69 | 104.38 | 244.11 | 3.8325 | 4.0965 |
| RTX 4090 | LIBERO-10 | 1,600 | 140.37 | 86.23 | 46.40 | 132.67 | 7.1242 | 7.5378 |

RTX 4090 的 E2E P50/P95/P99 为 139.27/142.43/143.78 ms。两台机器均预热捕获 7 个 CUDA Graph，测量期间图数量和编译计数均未增加；4090 运行时确认 534 个 Liger 模块生效。Thor 的 E2E 与纯推理平均相差 16.82 ms，包含输入处理/传输、输出还原和测量开销。10 个跨任务观测在相同随机种子下，分段计时路径与普通 `EngineCore.execute` 的 `10×7` 输出逐元素相同。

Thor 上对比语言/动作 attention 的 SDPA/eager（纯推理 238.11/251.21 ms，视觉 attention 保持 SDPA）后保留 SDPA；两者物理动作最大绝对差为 `0.0078125`。

## 实验方法

模型：[Dexmal/DM05-libero](https://huggingface.co/Dexmal/DM05-libero)，revision `25a8e0d38a8eaeaae41a44d7b4a2378fd8ce1088`。两路 RGB；EEF position3 + axis-angle3 + gripper2。遵循官方 LIBERO 配置 `add_state=false`，状态参与输入校验和归一化，不添加状态文本 token。10 个去噪步，输出 **10×7**。

数据沿用 [PI0.5 benchmark](../pi05-benchmark/README.md) 的 `yifengzhu-hf/LIBERO-datasets`，revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`，`libero_10/*.hdf5` 任务名字典序前 10 个、每任务 numeric demo ID 前 10 段、每段含首尾均匀取 16 帧，共 **1,600 帧**。选择规则、sample ID 和 selection SHA256 与 PI0.5/Cosmos 一致。四个采样字段可在 `config.yaml` 调整。

统一单卡、B=1、BF16、seed=42、float32 matmul precision=`highest`、预热 10 次、正式测量 1 遍。输入为已解码 CPU RGB/状态；计时包含预处理、传入 GPU、模型推理、动作后处理和 CPU 输出，两端 CUDA 同步。文件读取和解码、加载、预热（包含编译/图捕获）不计入。测量中出现新图捕获或编译则报错。无需仿真，不测任务成功率。

配置启用 Inductor 编译 `denoise_step`、完整去噪循环 CUDA Graph 和 prefix 复用；视觉、语言及动作注意力均使用 SDPA，并启用 Liger。

分段计时：`prefill_ms` 为视觉/文本/状态编码的 CUDA elapsed，`decode_ms` 为全部去噪步骤的 CUDA elapsed；`pure_inference_ms`（表中“纯推理”）为同一完整模型区间的同步墙钟时间，包含模型内部 CPU 调度。CUDA elapsed 包含流上的等待间隙，不是逐 kernel 时长相加。

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
