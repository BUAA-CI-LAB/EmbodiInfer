# PI0.5 Quantization Benchmark

AGX Orin 32G 上 **batch=1** 的 BF16 与 INT8 LIBERO-10 回放，按 `设备/精度/` 查看结果。
Jetson AGX Thor 与 RTX 4090 的 BF16 / FP8 / NVFP4 预设见 [PI0.5 性能页](../../docs/zh/pi05-performance.md)。

| 设备 | BF16 | INT8 |
|---|---|---|
| AGX Orin 32G | [结果](agx-orin/bf16/README.md) | [结果](agx-orin/int8/README.md) |

使用 `pi05_libero_finetuned_v044` 检查点。LIBERO-10 按任务文件名取 10 个任务，每任务按数字 demo ID 取前 10 段，每段含首尾均匀取 16 帧，共 **1,600 条观测**。输入两路 RGB、状态和指令；seed=42，完整 10 步去噪，输出 50×7 动作。纯离线回放，无仿真。

INT8 预热 10 条，覆盖 10 个任务；BF16 原生优化配置完整预热 1,600 条以覆盖所有输入形状。正式测量前完成编译与图捕获。

每个精度目录保留一份开启可用优化的 `config.json` 和对应 README。`config.json` 保留原测试机器的模型和数据路径，新输出写入同目录的 `runs/`；逐条结果 `*.result.json` 不提交 Git。

E2E 是已解码 CPU 观测到 CPU 动作的完整耗时；Forward 是 GPU 输入就绪后的完整模型调用。两者均包含全部去噪步骤，E2E 额外包含预处理、传输和后处理。加载、预热、磁盘读取及图像解码不计入。吞吐为观测数除以总 E2E 时间，内存为峰值 CUDA allocated。

BF16 保留原生优化、Inductor、prefix/denoise CUDA Graph 和 Triton attention。INT8 使用 native 后端，开启 CUDA Graph、关闭 Inductor 编译。两者均开启低内存加载。

测量时尚未证明资源独占，后期资源监控记录到系统换页。

先从 Orin 上已验证的环境复制 CUDA/PyTorch 依赖创建独立虚拟环境，再选择精度配置：

```bash
python setup_env.py --runtime-python /path/to/working/python --inference-root /path/to/inference
.venv/bin/python benchmark.py --config agx-orin/int8/config.json --validate-data
.venv/bin/python benchmark.py --config agx-orin/int8/config.json
.venv/bin/python compare.py agx-orin/bf16/libero10.result.json agx-orin/int8/libero10.result.json --output agx-orin/int8/runs/comparison.json
```

已测环境为 `vla_keep` 容器，Python 3.10，Torch 2.9.1+CUDA 12.6，LeRobot 0.5.1、Transformers 5.3.0。`prepare_lerobot_py310.py` 为 Python 3.10 回移 LeRobot 的类型标注语法。原生优化 BF16 需使用原测试运行时或包含相应优化的源码；配置中的 `native_inference` 不能直接用于原始量化分支。

结果、日志、虚拟环境和压缩包均由 `.gitignore` 排除。
