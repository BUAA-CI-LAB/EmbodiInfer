# PI0.5 · 4090 · BF16

仅 batch=1；[实验方法与环境](../../README.md)。本目录仅保留一份开启可用优化的 [config.json](config.json)。

原生优化、Inductor、prefix/denoise CUDA Graph、Triton attention；预热 1,600 条。需使用包含 PI0.5 原生优化的独立源码运行时，见上级 README。

延迟 ms，吞吐 obs/s，内存为峰值 CUDA allocated GiB。

| 数据集 | 观测数 | E2E | Forward | 吞吐 | 内存 |
|---|---:|---:|---:|---:|---:|
| LIBERO-10 | 1,600 | 39.77 | 36.74 | 25.145 | 10.80 |

在本 benchmark 根目录运行；模型、数据路径按实际机器修改，新结果写入本目录的 `runs/`：

```bash
.venv/bin/python benchmark.py --config 4090/bf16/config.json
```

原始全量结果：LIBERO-10 (`libero10.result.json`)（本地及 snapshot 提供，不提交 Git）。
