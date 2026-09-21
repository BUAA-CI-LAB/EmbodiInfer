# StreamVLN · thor · BF16

仅 batch=1；[实验方法与环境](../../README.md)。本目录仅保留一份开启可用优化的 [config.json](config.json)。

开启语言 prefill 编译、视觉/decode CUDA Graph、历史特征缓存和 fast action decode。KV 缓存 32K，预热 33 帧。

延迟 ms，吞吐 obs/s，内存为峰值 CUDA allocated GiB。

| 数据集 | 观测数 | E2E | Forward | 吞吐 | 内存 |
|---|---:|---:|---:|---:|---:|
| R2R | 2,997 | 502.19 | 461.51 | 1.991 | 17.25 |
| RxR | 3,879 | 506.80 | 464.98 | 1.973 | 17.46 |

在本 benchmark 根目录运行；模型、数据路径按实际机器修改，新结果写入本目录的 `runs/`：

```bash
.venv/bin/python benchmark.py --config thor/bf16/config.json
```

原始全量结果：R2R (`r2r.result.json`)、RxR (`rxr.result.json`)（本地及 snapshot 提供，不提交 Git）。
