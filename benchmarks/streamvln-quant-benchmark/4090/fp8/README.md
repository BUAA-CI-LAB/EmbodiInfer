# StreamVLN · 4090 · FP8

仅 batch=1；[实验方法与环境](../../README.md)。本目录仅保留一份开启可用优化的 [config.json](config.json)。

FP8 native 后端；关闭语言 prefill 编译；开启视觉/decode CUDA Graph、历史特征缓存和 fast action decode。KV 缓存 32K，预热 33 帧。

延迟 ms，吞吐 obs/s，内存为峰值 CUDA allocated GiB。

| 数据集 | 观测数 | E2E | Forward | 吞吐 | 内存 |
|---|---:|---:|---:|---:|---:|
| R2R | 2,997 | 246.30 | 223.51 | 4.060 | 14.95 |
| RxR | 3,879 | 243.29 | 219.40 | 4.110 | 17.63 |

在本 benchmark 根目录运行；模型、数据路径按实际机器修改，新结果写入本目录的 `runs/`：

```bash
.venv/bin/python benchmark.py --config 4090/fp8/config.json
```

原始全量结果：R2R (`r2r.result.json`)、RxR (`rxr.result.json`)（本地及 snapshot 提供，不提交 Git）。
