# ActiveVLN benchmark lane data

本目录归档 `bench/activevln-rxr-oom-fix` 分支的完整 benchmark 证据（本地生成于独立工作区
`activevln-benchmark-review/`，RTX 5090 32GB）。

## 内容

| 路径 | 说明 |
|---|---|
| `configs/` | 全部运行配置（EmbodiInfer B1 smoke/full、B2/4/8、RxR retry/tuned、vLLM B1） |
| `scripts/` | 驱动脚本（R2R/RxR EmbodiInfer、vLLM modern、OOM fix 验证、post-fix 重跑） |
| `summary/` | `opt_summary.md`（最终矩阵+OOM fix+差距分析）、`opt_before_after.md`、`opt_matrix_final.md`、两份 run manifest、第一轮矩阵 `summary_table.md`、状态板 |
| `results/optimized/` | 优化分支原始报告（25 份 JSON，含 `*-fix.json` 与 RxR B8 OOM 报告） |
| `results/first-matrix/` | 第一轮 main 快照 16 格原始报告（12 份 JSON） |
| `logs/optimized/` | 优化分支运行日志（含 OOM fix 验证与 post-fix 重跑） |
| `logs/first-matrix/` | 第一轮运行/serve 日志 |
| `tools/diff_token_sequences.py` | 两份报告按 (episode, step) 的逐 token 对比工具 |

## 结论摘要（samples/s，48 episodes 全量重放，greedy）

| dataset | batch | EmbodiInfer | vLLM modern | 备注 |
|---|---|---|---|---|
| R2R | 1 | 17.01 | 11.55 | B1 为单会话路径 |
| R2R | 2 | 22.89 | 17.25 | |
| R2R | 4 | 26.60 | 22.33 | |
| R2R | 8 | 27.62 | 27.07 | |
| RxR | 1 | 10.81 | 7.19 | 官方 RxR checkpoint + 30/60/90 |
| RxR | 2 | 12.92 | 10.96 | |
| RxR | 4 | 13.11 | 13.44 | |
| RxR | 8 | OOM（840 obs） | 16.08 | pool+owned 双份 KV 超 32GB |

测量口径、限制与 OOM fix 详情见 `summary/opt_summary.md` 与 `summary/run_manifest_optimized.md`。
本数据为记录帧重放的推理吞吐，不是闭环 SR/SPL。
