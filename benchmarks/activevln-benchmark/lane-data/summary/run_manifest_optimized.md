# 优化分支复现手册（commands / env / 移植说明）

## 代码与环境

- 本地: `/Users/lxy/lxygit/activevln-benchmark-review/EmbodiInfer-opt`
  （`git worktree add --detach ... 8b39867`，origin 主仓库 `EmbodiInfer` 未动）
- 远端: `/home/qy/lxygit/activevln-benchmark-review/EmbodiInfer-opt`（同 commit）
- 远端 venv 复用现有环境，未重装：
  - EmbodiInfer: `../EmbodiInfer/.venv` → torch 2.10.0+cu130 / transformers 4.51.3 / triton 3.6.0
  - vLLM: `../vllm-env` → vLLM 0.30.0 / torch 2.13.0+cu130 / transformers 5.17.0
- 远端必须先置 PATH（否则 flashinfer JIT 找不到 ninja，已知问题）：
  `export PATH=/home/qy/lxygit/activevln-benchmark-review/vllm-env/bin:$PATH`

## 本地移植（上游实现未改，仅接线）

| 文件 | 改动 |
|---|---|
| `prompt_activevln.py` | 增加 `SYSTEM_PROMPT_RXR`、`SYSTEM_PROMPTS`、`DEFAULT_TURN_ANGLE`、`parse_navigation_actions`；`render_turn_text/chat_messages` 增加 `action_space` |
| `processor_activevln.py` | 构造参数 `action_space`，传入 `render_turn_text` |
| `modeling_activevln.py` | policy/factory 增加 `action_space`、`parse_actions()`；trace runner_profile 随 action space |
| `batching_activevln.py` / `speculation_activevln.py` | 解析改用 `policy.parse_actions` |
| `benchmark.py` / `benchmark_batch.py` | 读取 config `action_space`；batch_size 白名单放宽到 8（本地 lane 扩展） |
| `benchmark_vllm_modern.py` / `benchmark_vllm_batch.py` | action-space 解析/渲染接线；batch_size 白名单放宽到 8 |

移植文件 md5 本地=远端一致（9 个文件，2026-09-26 校验）。

## 运行命令（远端根目录 `/home/qy/lxygit/activevln-benchmark-review`）

通用环境：
```bash
cd EmbodiInfer-opt
export PYTHONPATH=.
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7
```

R2R EmbodiInfer（B1 单进程；B2/B4/B8 逐进程）：
```bash
../EmbodiInfer/.venv/bin/python benchmarks/activevln-benchmark/benchmark.py \
  --config benchmarks/activevln-benchmark/r2r-b1-full.yaml
../EmbodiInfer/.venv/bin/python benchmarks/activevln-benchmark/benchmark_batch.py \
  --config benchmarks/activevln-benchmark/r2r-b{2,4,8}.json \
  --output benchmarks/activevln-benchmark/runs/opt-full-r2r-b{2,4,8}/report.json
```
驱动脚本：`scripts/opt_batch_r2r.sh`

RxR EmbodiInfer（B1 前置 smoke 验证 RxR prompt/解析）：
```bash
../EmbodiInfer/.venv/bin/python benchmarks/activevln-benchmark/benchmark.py \
  --config benchmarks/activevln-benchmark/rxr-b1-smoke.yaml   # 2 ep x 6 帧
../EmbodiInfer/.venv/bin/python benchmarks/activevln-benchmark/benchmark.py \
  --config benchmarks/activevln-benchmark/rxr-b1-full.yaml
../EmbodiInfer/.venv/bin/python benchmarks/activevln-benchmark/benchmark_batch.py \
  --config benchmarks/activevln-benchmark/rxr-b{2,4,8}.json --output ...
# B8 两次 OOM：rxr-b8.json（pool 256000）与 rxr-b8-retry.json（pool 196608）
```
驱动脚本：`scripts/opt_rxr_driver.sh`

vLLM modern：
```bash
export PATH=/home/qy/lxygit/activevln-benchmark-review/vllm-env/bin:$PATH
export VLLM_ENABLE_V1_MULTIPROCESSING=0
../vllm-env/bin/python benchmarks/activevln-benchmark/benchmark_vllm_modern.py \
  --config benchmarks/activevln-benchmark/r2r-vllm-b1.json --output .../opt-vllm-r2r-b1/report.json
../vllm-env/bin/python benchmarks/activevln-benchmark/benchmark_vllm_batch.py \
  --config benchmarks/activevln-benchmark/r2r-b{2,4,8}.json --output ...
# RxR 同构：rxr-vllm-b1.json / rxr-b{2,4,8}.json + action_space: rxr
```
驱动脚本：`scripts/opt_vllm_r2r_driver.sh`、`scripts/opt_vllm_r2r_batch_driver.sh`、`scripts/opt_vllm_rxr_driver.sh`

## 配置文件（本地 `results/opt-configs/`，远端 `EmbodiInfer-opt/benchmarks/activevln-benchmark/`）

- `r2r-b1-smoke.yaml`（8 ep × 12 帧，用于同协议前后锚点 55.68ms）
- `r2r-b1-full.yaml`、`rxr-b1-smoke.yaml`、`rxr-b1-full.yaml`
- `r2r-b{2,4,8}.json`、`rxr-b{2,4,8}.json`、`rxr-b8-retry.json`、`*-vllm-b1.json`
- 关键参数：`attention: sdpa; cuda_graph: true; fused_ops: true; split_attention: true;
  tree_decode: true; query_bucket_size: 1; prewarm_context_buckets: [512..128000];
  graph_workspace_tokens: 65536(R2R)/128000(RxR); kv_pool_tokens: R2R 128k(B2/B4)/256k(B8),
  RxR 128k(B2)/153.6k(B4)/256k(B8, 196.6k retry); max_context: 128000`

## 原始结果（本地 `results/opt-runs/`）

- EmbodiInfer 全量：`opt-full-{r2r,rxr}-b{1,2,4,8}.json`（RxR B8 含 OOM 报告 + retry）
- vLLM modern：`opt-vllm-{r2r,rxr}-b{1,2,4,8}.json`
- 日志（远端 `logs/opt-*.log`，本地 `results/logs/` 增量同步）

## OOM fix（lane 追加，2026-09-26）

代码：`embodiinfer/policies/activevln/batching_activevln.py`
- `ActiveVLNBatchedRuntime.__init__`: 新增 `_owned_rows` 每行持久缓冲登记
- 新增 `_commit_row_kv(row, length, memory)`：4096-token 对齐增长；同一 memory 原地补齐；
  跨 episode/行重排按 weakref 判定旧 memory 是否仍被持有
- `_finish_generation`: 用 `_commit_row_kv` 替代 `_row_kv(...).clone().contiguous()`
- `reset_stats`: 新增 `owned_kv_allocations` / `owned_kv_reuses` 计数

验证命令（远端 `EmbodiInfer-opt`）：
```bash
PYTHONPATH=. ../EmbodiInfer/.venv/bin/python -m pytest \
  tests/test_activevln_batching.py tests/test_activevln_batch_serving.py \
  tests/test_activevln_graph_warmup.py -q        # 54 passed（需 pytest，已装入 EmbodiInfer/.venv）
bash ../scripts/opt_oomfix_verify.sh              # R2R B8 / RxR B8 / RxR B4
bash ../scripts/opt_postfix_cells.sh              # R2R B2 / R2R B4 / RxR B2
python ../results/tools/diff_token_sequences.py A B  # 逐 token 回归对比
```
结果：`*-fix.json` 报告；R2R B8 / RxR B4 逐 token 一致；RxR B8 仍 OOM（840 obs，pool 196.6k、
workspace 64k 两次尝试均未通过）。tuned/retry 配置：`rxr-b8-retry.json`、`rxr-b8-tuned.json`。
