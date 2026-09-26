# ActiveVLN 性能对照 — 状态板（优化分支 + OOM fix 已完成）

- 更新: 2026-09-26 04:2x EDT
- 本地: `/Users/lxy/lxygit/activevln-benchmark-review/`；远端: `/home/qy/lxygit/activevln-benchmark-review/`
- 代码 worktree: `EmbodiInfer-opt/`（detached @8b39867 + lane diff；OOM fix 已提交前暂存）
- 交付:
  - 第一轮（main 快照）：`results/summary_table.md` + `results/run_manifest.md`（16 格旧口径）
  - 优化分支：`results/opt_summary.md`（最终矩阵/前后对比/差距分析/OOM fix）、
    `results/run_manifest_optimized.md`（命令/移植/OOM fix）、`results/opt_before_after.md`
  - 原始报告 `results/opt-runs/`（含 `*-fix.json`）、日志 `results/logs/opt/`
  - 归档数据 `EmbodiInfer-opt/benchmarks/activevln-benchmark/lane-data/`（待 push 到 GitHub）
- 交接: DSH claim 已 release；automation `activevln` 仍 PAUSED；远端无遗留进程，GPU 已清空

## 最终矩阵（post-fix，samples/s，48 ep 全量重放，RTX 5090）

| 数据集 | batch | EmbodiInfer | vLLM modern | 快者 |
|---|---|---|---|---|
| R2R | 1 | 17.01 | 11.55 | EmbodiInfer |
| R2R | 2 | 22.89 | 17.25 | EmbodiInfer |
| R2R | 4 | 26.60 | 22.33 | EmbodiInfer |
| R2R | 8 | 27.62 | 27.07 | EmbodiInfer |
| RxR | 1 | 10.81 | 7.19 | EmbodiInfer |
| RxR | 2 | 12.92 | 10.96 | EmbodiInfer |
| RxR | 4 | 13.11 | 13.44 | vLLM |
| RxR | 8 | OOM（840 obs） | 16.08 | vLLM |

## OOM fix（本次）

- `batching_activevln.py`：每行持久 committed-KV 缓冲替代每轮整段 `clone()`；
  同 memory 原地补齐、跨 episode 按 weakref 保护旧 memory；~90% 复用。
- 验证：54 CPU 单测通过；R2R B8 / RxR B4 逐 token 零差异（2997/2997、3879/3879）；
  R2R B8 27.62 sps（+0.9%）、RxR B4 13.11 sps（+0.5%）、RxR B4 显存 -2.1GiB。
- RxR B8 仍 OOM（600→840 obs 推迟；pool 196.6k / workspace 64k 未通过），如实标注；
  根因是 pool + owned 双份 KV，彻底修复需 multi-base attention/冻结段分配器（未做）。

## 交付位置

- 代码 + 数据已提交到分支 `bench/activevln-rxr-oom-fix`（基于 `feat/activevln-inference-optimization` @8b39867）：
  3 个 commit（lane RxR/接线、OOM fix、lane-data 全量证据），推送目标 `github.com/BUAA-CI-LAB/EmbodiInfer`
- 归档数据位于 `EmbodiInfer-opt/benchmarks/activevln-benchmark/lane-data/`（含 configs/scripts/reports/logs/summary）
- 可选后续（未做）：KV 单份化改造、CPU 预处理流水线、尾部占用优化（见对话中的分析）
