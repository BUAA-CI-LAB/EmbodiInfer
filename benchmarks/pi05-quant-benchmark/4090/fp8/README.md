# PI0.5 · 4090 · FP8

仅 batch=1；[实验方法与环境](../../README.md)。本目录仅保留一份开启可用优化的 [config.json](config.json)。

FP8 Triton 权重量化、CUDA Graph；预热 10 条。Inductor 编译关闭。

延迟 ms，吞吐 obs/s，内存为峰值 CUDA allocated GiB。

| 数据集 | 观测数 | E2E | Forward | 吞吐 | 内存 |
|---|---:|---:|---:|---:|---:|
| LIBERO-10 | 1,600 | 121.76 | 118.57 | 8.213 | 5.89 |

在本 benchmark 根目录运行；模型、数据路径按实际机器修改，新结果写入本目录的 `runs/`：

```bash
.venv/bin/python benchmark.py --config 4090/fp8/config.json
```

原始全量结果：LIBERO-10 (`libero10.result.json`)（本地及 snapshot 提供，不提交 Git）。
