# 3B Navigation Real-Image Benchmarks

Formal R2R-VLNCE aligned-RGB throughput results are recorded in [R2R_VLNCE_THROUGHPUT_REPORT.md](R2R_VLNCE_THROUGHPUT_REPORT.md).

本目录只发布使用真实 RGB 文件的三个 3B 导航模型 benchmark。脚本不生成
random、zero、placeholder 或其他合成视觉输入，也没有 synthetic fallback。

支持的 profile：

- `qwen2.5-vl-3b-r2r-low-level`
- `qwen2.5-vl-3b-r2r-panoramic`
- `navida`

目录内容：

- `dataset.py`：严格加载 JSON/JSONL manifest，将真实图像通过 PIL 转为 RGB，
  再转为 `[C,H,W]`、`float32`、`[0,1]` tensor。
- `benchmark.py`：逐个真实 sample 做 B1 full-policy paired eager/manual-graph
  测量和 token/text/action parity。
- `benchmark_cuda_graph.py`：仅用于 Low/Panoramic 的固定 shape next-token kernel，
  对比 Transformers top-level eager、自研 uncaptured forward 和 captured replay。
- `benchmark_compile_cache.py`：纯 stdlib 多进程 launcher，验证空 cache producer、
  same-dir consumer 和 fresh-dir Mega-Cache consumer 的 FX/AOT persistent hit。
- [`TRITON_REPORT.md`](TRITON_REPORT.md)：2026-08-22 RTX 4090 正式 Triton hybrid
  kernel、full-policy 与 NaViDA paired-seeded 结果和正确性门禁。
- [`TORCH_COMPILE_REPORT.md`](TORCH_COMPILE_REPORT.md)：2026-08-22 RTX 4090
  precision-v2 的正式 8/8 kernel/full-policy 数据、正确性门禁与冷启动代价。
- [`HISTORY_IMAGE_CACHE_REPORT.md`](HISTORY_IMAGE_CACHE_REPORT.md)：2026-08-22
  CPU-only 8-step recurrent history RGB cache 正式结果、exact processor parity 与门禁。

## 数据前置条件

R2R-CE 的 episode annotation（例如 `val_unseen.json.gz`）只包含 instruction、
trajectory、pose 和 scene ID，不包含 RGB。它不能直接作为本 benchmark 的视觉输入。
必须先在有授权的 Matterport3D 场景中预渲染 trajectory frames，或提供自有的真实
RGB 图像。不要用黑图、随机图或 placeholder 代替。

所有 manifest 图像路径必须是相对 `--data-root` 的文件路径。绝对路径、`..` 路径、
缺失文件、目录和越过 data root 的 symlink 都会被拒绝。manifest 和选中的所有图像
会在计时前完整读入内存；manifest SHA256、磁盘 I/O、PIL decode 和 tensor 构造均不
计入 benchmark 延迟。

Manifest 可以是 JSON 数组、`{"samples": [...]}`，或每行一个对象的 `.jsonl`。
`id` 在整个 manifest 中必须唯一，`instruction` 必须为非空字符串。`--limit` 按
manifest 顺序选择前 N 个样本，且不能超过样本总数。

## Manifest schema

### Low-level

Low 图像尺寸由官方 policy processor resize，不在 loader 中改尺寸。history frame
和 history response 必须一一对应。

```json
[
  {
    "id": "episode-1-step-0",
    "instruction": "Walk through the doorway and stop by the stairs.",
    "current_image": "low/episode-1/0000.jpg",
    "history_images": [
      "low/episode-1/history-0000.jpg"
    ],
    "history_responses": [
      "Move"
    ],
    "distance_traveled": 0.25,
    "move_possible": true
  }
]
```

必需字段：`id`、`instruction`、`current_image`。

可选字段：`history_images`、`history_responses`、`distance_traveled`、
`move_possible`。两项 history 均省略表示 empty memory。

### NaViDA

当前帧和每个 history frame 必须严格为 `320x240` RGB。

```json
[
  {
    "id": "episode-1-step-0",
    "instruction": "Walk through the doorway and stop by the stairs.",
    "current_image": "navida/episode-1/0000.jpg",
    "history_images": [
      "navida/episode-1/history-0000.jpg"
    ]
  }
]
```

必需字段：`id`、`instruction`、`current_image`。

可选字段：`history_images`。

### Panoramic

`panorama_image` 和每个 history panorama 必须严格为 `960x240` RGB；每个
candidate image 必须严格为 `320x240` RGB。candidate 列表不能为空，角度和距离
来自真实候选导航边。

```json
[
  {
    "id": "episode-1-step-0",
    "instruction": "Take the opening on the left.",
    "panorama_image": "panoramic/episode-1/0000.jpg",
    "candidates": [
      {
        "image": "panoramic/episode-1/candidate-0.jpg",
        "relative_angle": -45.0,
        "distance": 2.1
      },
      {
        "image": "panoramic/episode-1/candidate-1.jpg",
        "relative_angle": 20.0,
        "distance": 3.4
      }
    ],
    "history_panoramas": [
      "panoramic/episode-1/history-0000.jpg"
    ],
    "history_responses": [
      "0"
    ],
    "distance_traveled": 0.0
  }
]
```

必需字段：`id`、`instruction`、`panorama_image`、`candidates`。每个 candidate
必须且只能包含 `image`、`relative_angle`、`distance`。

可选字段：`history_panoramas`、`history_responses`、`distance_traveled`。
两项 history 必须一一对应。

## Full-policy benchmark

```bash
cd /path/to/embodiinfer
export PYTHONPATH=$PWD

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python benchmarks/3B-navigation/benchmark.py \
  --profile qwen2.5-vl-3b-r2r-low-level \
  --attention-backend torch_sdpa \
  --checkpoint /path/to/qwen-r2r-low \
  --manifest /path/to/low.jsonl \
  --data-root /path/to/rendered-rgb \
  --limit 8 \
  --seed 41 \
  --warmup 1 \
  --iters 1 \
  --device cuda:0 \
  --expected-physical-gpu 0 \
  --output /path/to/low-result.json
```

`benchmark.py` 始终逐 sample、B1 执行；不会把不同 episode 拼成 raw batch。
每个 mode、每个 sample 先 warmup。性能 timed loop 内不重置 RNG seed；seed reset、
graph capture 和 paired validation 均在计时区间外。计时包含 prompt 构造、
processor、H2D、prefill、完整 decode、
text decode 和 action parser；不包含 manifest/image I/O、model load、shape planning、
warmup/graph capture 和 JSON serialization。

Low/Panoramic 是确定性单 token policy，继续要求每个 mode 内 signature stable。
NaViDA 使用随机采样，JSON 标记 `stochastic=true`、`stability_required=false`，timed
samples 不要求 signature stable。所有真实 shape 完成 graph capture 后，脚本为每个
sample 派生并记录固定 seed，在计时外分别执行一次 eager 和 graph；两次的
token/text/action 必须完全一致，记录为 `paired_seeded_parity`。所有 timed 与 paired
outputs 的 action 都必须有效，timed window 中不得出现新 capture；这些条件共同决定
退出状态。

输出记录 `visual_source=real_images_from_manifest`、`synthetic=false`、manifest
SHA256、sample IDs、所有 image paths、memory 来源、GPU 映射、逐 sample 输出、
吞吐、token 数、峰值显存、capture/replay/cache 和 parity。

脚本会在计时前用真实 observation 做 encoded-shape planning。若产生超过 8 个不同
shape cohort，会明确失败并要求拆分 manifest；不会依赖 8-entry graph cache 静默
evict。graph capture 必须发生在 warmup，若 timed window 中出现 capture，结果失败。

## Pre-encoded CUDA Graph kernel benchmark

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python benchmarks/3B-navigation/benchmark_cuda_graph.py \
  --profile qwen2.5-vl-3b-r2r-panoramic \
  --attention-backend torch_sdpa \
  --checkpoint /path/to/qwen-r2r-panoramic \
  --manifest /path/to/panoramic.json \
  --data-root /path/to/rendered-rgb \
  --limit 8 \
  --seed 41 \
  --warmup 5 \
  --iters 20 \
  --device cuda:0 \
  --expected-physical-gpu 0 \
  --output /path/to/panoramic-kernel-result.json
```

该脚本仅支持 Low/Panoramic。NaViDA 的实际 CUDA Graph 路径是完整 StaticCache
autoregressive generation，必须使用 `benchmark.py`；不能用单 next-token kernel
数字代表 NaViDA policy。

每个真实 sample 只在计时前编码一次。脚本比较：

- Transformers model top-level eager logits；
- 自研 uncaptured multimodal forward logits；
- captured CUDA Graph replay logits。

三路均记录 top-1、pairwise logits tolerance、最大绝对误差。CUDA Event 只计量
pre-encoded GPU kernel；processor、H2D、generation、parser、capture 和磁盘 I/O
全部排除。JSON 同样记录 manifest SHA256、image paths、encoded tensor shapes、
shape cohorts、capture/replay/cache 和 GPU 映射。

Parity gate 按实际 resolved backend 分层。Torch SDPA 的 HF-vs-native/captured
保持 `rtol=0.02, atol=0.05`；Triton hybrid 的 window attention 在 32 个 vision
layers 中产生归约误差累积，因此 HF-vs-native/captured 使用
`rtol=0.06, atol=0.30`，同时强制 top-50 集合至少重合 48 项，即 96% boundary-set
gate。native-vs-captured 无论 backend 都保持 `rtol=0.02, atol=0.05`，所有 pair
仍要求 top-1 完全一致。JSON 每个 comparison 明确记录实际 `rtol`、`atol`、
`top50_overlap`、`top50_required` 和 `passes_gate`，不会隐藏较宽的 hybrid 门槛。

## 同一真实 manifest 的 Torch SDPA 与 Triton 对比

Low/Panoramic 支持 `--attention-backend torch_sdpa|triton|auto`。显式
`triton` 在 kernel 不可用时直接失败；`auto` 才允许回退。NaViDA 不使用该共享
next-token backend，显式选择 `triton` 或 `auto` 会直接失败。

下面两次运行使用完全相同的真实 Panoramic manifest、图像、checkpoint、seed、
sample 顺序、warmup 和迭代次数，只有 attention backend 不同：

```bash
for ATTENTION_BACKEND in torch_sdpa triton; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  python benchmarks/3B-navigation/benchmark_cuda_graph.py \
    --profile qwen2.5-vl-3b-r2r-panoramic \
    --attention-backend "${ATTENTION_BACKEND}" \
    --checkpoint /path/to/qwen-r2r-panoramic \
    --manifest /path/to/panoramic.json \
    --data-root /path/to/rendered-rgb \
    --limit 8 \
    --seed 41 \
    --warmup 5 \
    --iters 20 \
    --device cuda:0 \
    --expected-physical-gpu 0 \
    --output "/path/to/panoramic-${ATTENTION_BACKEND}-kernel-result.json"
done
```

JSON 的 `attention_backend` 记录 requested/resolved、是否 fallback、fallback reason、
kernel ABI、resolved config、layers backend 和 Triton version。backend resolution、
Triton JIT 与 graph capture 均发生在 warmup/parity gate，排除在 timed window 外。

正式结果必须保留原始 manifest 和真实图像资产的 provenance。仓库不再发布旧的
synthetic/placeholder RTX 4090 数字。

## `torch.compile` / Inductor

两个 benchmark CLI 都接受 `--compile-backend none|inductor`，默认 `none`。
编译路径 fail-loud，不会静默回退 eager。

| Profile | `none` | `inductor` | attention 约束 | kernel 脚本 |
| --- | --- | --- | --- | --- |
| Low | 支持 | 支持 | `inductor` 仅允许 `torch_sdpa` | 支持 |
| Panoramic | 支持 | 支持 | `inductor` 仅允许 `torch_sdpa` | 支持 |
| NaViDA | 支持 | **拒绝** | 真实 3B history stochastic paired-seed token 分叉 | 不支持 |

Low/Panoramic 开启编译但关闭 graph 时标记为 `compiled`，开启 graph 时标记为
`compiled_manual_cudagraph`。NaViDA 只准入 `compile_backend=none`；两个 CLI 都会在
读取 manifest 或加载 checkpoint 前明确拒绝 NaViDA + Inductor。

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/3B-navigation/benchmark.py \
  --profile qwen2.5-vl-3b-r2r-low-level \
  --attention-backend torch_sdpa \
  --compile-backend inductor \
  --checkpoint /path/to/qwen-r2r-low \
  --manifest /path/to/low.jsonl --data-root /path/to/rendered-rgb \
  --limit 8 --seed 41 --warmup 3 --iters 10 \
  --output /path/to/low-inductor.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/3B-navigation/benchmark_cuda_graph.py \
  --profile qwen2.5-vl-3b-r2r-panoramic \
  --attention-backend torch_sdpa \
  --compile-backend inductor \
  --checkpoint /path/to/qwen-r2r-panoramic \
  --manifest /path/to/panoramic.json --data-root /path/to/rendered-rgb \
  --limit 8 --seed 41 --warmup 3 --iters 10 \
  --output /path/to/panoramic-inductor-kernel.json

```

首次 compiled call、Inductor/Triton JIT、compile 内部 warmup、policy warmup 和
manual CUDA Graph capture 全部发生在 timed window 外。每个 timed window 前后都会
冻结并核对 `attempts`、`cache_entries`、`failures`，任何变化都使运行失败。JSON
记录 requested/effective/target/compile ABI、backend、fullgraph、dynamic、
`inductor_cudagraphs`、`emulate_precision_casts`、完整 `inductor_options`、Torch
version、cache entries、attempts、failures、每个 cache entry 的 first-call wall time
以及 `runtime_mode`。结果 JSON 仍是外部 artifact，不入库。

当前 numerics contract 是 `qwen25_vl_next_token_compile_v4`，完整 Inductor options
固定为 `{"triton.cudagraphs": false, "emulate_precision_casts": true}`。其中
`emulate_precision_casts=true` 用于保留 eager BF16 路径中 precision cast 的 rounding
语义，且 Inductor 自身 cudagraphs 关闭，由 EmbodiInfer 的 manual CUDA Graph 负责 capture。

BF16 Inductor correctness 使用独立的分层门禁，不改变上面的 Triton hybrid 门禁：

- HF top-level 与 raw Torch self-authored forward 保持 `rtol=0.02, atol=0.05`。
- HF 或 raw Torch 与 Inductor compiled forward/captured replay 使用
  `rtol=0.06, atol=0.30`，并要求 top-50 集合至少重合 45 项，即 90%。
- 同一个 compiled callable 的 uncaptured execution 与 manual CUDA Graph replay
  保持 `rtol=0.02, atol=0.05`。

较宽的 logits 门限仅覆盖 BF16 Inductor 改变归约分组产生的累积差异，不放宽
top-1、token、decoded text 或 parsed action 的完全一致门禁，也不放宽同一 compiled
kernel 在 uncaptured/captured 两种执行方式之间的严格门禁。changed-image parity
使用完全相同的分层规则。precision-cast emulation 不保证 eager 与 compiled 位级
一致，仍必须满足 top-50 overlap 至少 45/50 及上述语义门禁。

Low/Panoramic 官方 policy 都是 deterministic argmax；top-1、token、decoded text 和
parsed action exact 才是行为正确性门禁，top-50 是额外的分布 guard。90% 即 45/50
是预先定义的原则化边界，不按当前样本观测过拟合：正式 manifest 实际最低为 48/50，
扩展 changed-image parity 实际最低为 47/50。Triton hybrid 继续使用其独立的 48/50
门禁，不受此 Inductor 调整影响。

NaViDA 不使用上述容差获得 compile 准入。真实 3B checkpoint 的 history stochastic
case 在相同 paired seed 下出现 token 分叉，因此不能宣称 compiled decode graph 与
官方 eager policy 正确性等价。NaViDA 的现有 official eager/manual-graph gate 保持
`compile_backend=none`，待 token/text/action paired-seed parity 被独立解决后才能重新
评估 compile 支持。

## Compile text buckets 与 persistent Mega-Cache v3

Low/Panoramic 的两个 benchmark CLI 接受 --compile-text-buckets 和
--compile-cache-dir。固定 text bucket 同时用于 Torch 与 Inductor，使完整真实
manifest 只产生受控的 CUDA Graph shape，并保证两种后端计算量一致。
Persistent compile cache 仍只允许与 --compile-backend=inductor 及显式
--attention-backend=torch_sdpa 组合。`triton` 与 `auto` 在模型加载前
fail-loud，编译路径不会静默回退。

Masked left padding 会改变 BF16 kernel 的 tile 与归约分组，因此跨 shape
parity 分别比较 HF、raw、compiled、captured 的 exact-shape 与 bucketed 输出，
使用 `rtol=.06`、`atol=.30`、top-50 overlap `>=45/50`，同时保持 top1、
token、text、action 完全一致。同 shape 的 HF/raw 仍使用 `.02/.05`，同 shape
raw/Inductor 使用 `.06/.30` 与 Inductor top-50 `>=45/50`，compiled/manual
CUDA Graph 仍使用 `.02/.05`。真实诊断 shape 为 Low `503 -> 512`、Panoramic
`1301 -> 1344`；`45/50` 是预先定义的 90% 分布门槛，不是按这组观测拟合。

文本 bucket 使用 qwen25_vl_left_masked_text_v3，只在左侧 pad input_ids，
对应 attention_mask 为零且绝不截断。pixel_values、image_grid_thw、history、
candidate 数量和 batch 均保持 exact；视觉 cohort 不 padding、不跨 shape 复用。
启用 bucket 时 native/compiled forward 使用 attention mask；默认 exact-shape
路径保持原有无 bucket 契约。

持久缓存使用 qwen25_vl_inductor_execution_cache_v3。每个 execution shape
在 embodiinfer-execution-entries/<fingerprint>.json 拥有独立 manifest，artifact 按
SHA256 存放在 embodiinfer-content-blobs/<sha256>.bin。损坏或身份不一致的内容进入
embodiinfer-quarantine/ 后 cold compile。cold compile 必须发布 artifact；已加载且
FX/AOT hit admission 成功的 consumer 可以记录 artifact_publish_skipped=true。

cache 目录必须在进程启动前存在，使用当前用户拥有的绝对 canonical 路径、
0700 权限且不能经过 symlink。以下 bootstrap token 与四项环境必须在任何
Torch import 之前设置：

    export EMBODIINFER_COMPILE_CACHE_BOOTSTRAP_ASSERTION=qwen25_vl_compile_cache_preimport_v1
    export TORCHINDUCTOR_CACHE_DIR=/absolute/trusted/cache
    export TORCHINDUCTOR_FX_GRAPH_CACHE=1
    export TORCHINDUCTOR_AUTOGRAD_CACHE=1
    export TRITON_LIBDEVICE_PATH=/absolute/path/to/libdevice.10.bc

推荐使用不导入 Torch 的三进程 launcher：

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0     python benchmarks/3B-navigation/benchmark_compile_cache.py       --profile qwen2.5-vl-3b-r2r-low-level       --checkpoint /path/to/qwen-r2r-low       --manifest /path/to/low.jsonl       --data-root /path/to/rendered-rgb       --compile-text-buckets 512,768,1024       --libdevice /absolute/path/to/libdevice.10.bc       --work-dir /absolute/external/compile-cache-run       --output /absolute/external/compile-cache-summary.json       --scope both --limit 1 --seed 41 --warmup 3 --iters 10

Panoramic 使用完整 profile qwen2.5-vl-3b-r2r-panoramic。实际远端 manifest
文件名可以沿用历史命名，但传给代码的 profile 必须是完整 registry key。

launcher 分别运行空缓存 producer、same-directory consumer，以及只复制
per-execution manifests/content blobs 的 fresh-directory consumer。两个
consumer 均要求 artifact_loaded、FX/AOT hit、零 FX/AOT miss。launcher 显式
固定 bootstrap、Inductor cache、libdevice、CUDA 映射和当前源码 PYTHONPATH。

compile/load、首次 compiled call、Inductor warmup、artifact save 和 manual
CUDA Graph capture 均位于 timed window 外。计时前后会比较 attempts、cache
entries、bucket IDs、per-execution artifact 与 counter 状态。结果 JSON 和 cache
artifact 必须存放在仓库外，不提交。

## 准备确定性的 R2R-VLNCE 真实 RGB manifest

`prepare_r2r_vlnce_manifest.py` 按 episode id、完整 instruction text 和 MP3D scene
精确 join 官方 R2R-VLNCE `train.json.gz` 与 StreamVLN R2R observation。默认按
数值 episode id 排序选择 48 个 episode，每个 episode 取 4 个整数分位 step。
每个样本固定 4 帧 history；Panoramic 另固定 4 个 temporal-neighbor candidate。
选择算法、输入 SHA、step index、manifest SHA 和每个 derivative 的 source/output
SHA256 都写入 `provenance.json`。
StreamVLN 的第 0 个 action 是 dummy；builder 严格使用 frame `i` 对应
`actions[i+1]`，history window 同步整体偏移，累计距离只统计
`actions[1:step+1]` 中的 `MoveForward`。

```bash
python benchmarks/3B-navigation/prepare_r2r_vlnce_manifest.py \
  --train-json-gz "$R2R_VLNCE/train/train.json.gz" \
  --images-root "$STREAMVLN/R2R/images" \
  --output-dir "$R2R_BENCH_MANIFESTS"
```

Low 直接引用 StreamVLN JPEG。NaViDA 在源图已经是 `320x240` 时也直接引用；否则
生成带完整 hash provenance 的 `ImageOps.fit` 真实 RGB derivative，以满足严格输入
契约。Panoramic 使用有 provenance 的 `960x240` current/history derivative 和 4 个
`320x240` temporal-neighbor derivative。它必须且只被描述为
`r2r_rgb_shape_compatible_not_official_panorama`：不是 Habitat stitched panorama，
candidate 不是 simulator navigation edge，也不提供 trajectory success、SR、SPL、
NE 或 nDTW 证据，只能用于固定输入的 policy/kernel throughput 与 parity。

StreamVLN Hugging Face 数据集需要接受 gated access，并声明 CC BY-NC-SA 4.0；
R2R 与 Matterport3D 资产仍受各自访问和许可条款约束。准备数据前需遵守全部上游
条款：[StreamVLN trajectory data](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data)
与 [VLN-CE](https://github.com/jacobkrantz/VLN-CE)。

以下命令仅使用物理 GPU0。`LOW_ROOT`/`PANORAMIC_ROOT` 必须取自
`provenance.json` 的 `data_roots`；默认 48x4 manifest 使用 `--limit 192`。
两个 CLI 的正式 R2R admission 都必须显式传 `--provenance`，并在模型加载前验证
provenance JSON、本体 SHA、profile/data-root、R2R/StreamVLN source SHA、全部
source/derivative SHA、`synthetic_pixels=false`、48x4 selection、history4、
responses4，以及 Panoramic candidates4。省略 provenance 的运行只能输出
`status=unverified` 并非零退出，不能作为 formal pass。

Full-policy 的 Inductor correctness gate 会先用短生命周期的
`attention_backend=torch_sdpa + compile_backend=none` runner 在相同 paired seed
生成独立 token/text/action reference，销毁并清空显存后才构造 requested runner。
reference、requested uncaptured 与 requested manual graph 必须三方完全一致；
reference load/inference/释放全部在 timed window 外。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python benchmarks/3B-navigation/benchmark.py \
  --manifest "$R2R_BENCH_MANIFESTS/low.json" --data-root "$LOW_ROOT" \
  --provenance "$R2R_BENCH_MANIFESTS/provenance.json" \
  --checkpoint "$LOW_CHECKPOINT" --profile qwen2.5-vl-3b-r2r-low-level \
  --attention-backend torch_sdpa --compile-backend inductor \
  --limit 192 --seed 41 --warmup 3 --iters 10 \
  --expected-physical-gpu 0 --output "$ARTIFACTS/low-full.json"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python benchmarks/3B-navigation/benchmark_cuda_graph.py \
  --manifest "$R2R_BENCH_MANIFESTS/panoramic.json" \
  --provenance "$R2R_BENCH_MANIFESTS/provenance.json" \
  --data-root "$PANORAMIC_ROOT" --checkpoint "$PANORAMIC_CHECKPOINT" \
  --profile qwen2.5-vl-3b-r2r-panoramic \
  --attention-backend torch_sdpa --compile-backend inductor \
  --limit 192 --seed 41 --warmup 3 --iters 10 \
  --expected-physical-gpu 0 --output "$ARTIFACTS/panoramic-kernel.json"
```

旧 Habitat-document smoke manifest 和对应报告仅保留为历史诊断，不是最终 R2R
throughput 输入，也不得表述为 R2R trajectory evaluation。

## Recurrent history RGB cache preprocessing

`benchmark_history_image_cache.py` 只测连续单 session 的真实图像
preprocessing，不是模型端到端吞吐。manifest 的 `steps` 按时间顺序提供
`image`（Low）或 `panorama`（Panoramic）、`instruction`、`response`。Panoramic
的 `candidates` 是包含 `image`、`relative_angle`、`distance` 的对象列表；相对
图像路径以 manifest 所在目录解析。

```bash
python benchmarks/3B-navigation/benchmark_history_image_cache.py \
  --profile low \
  --checkpoint /absolute/path/to/checkpoint \
  --manifest /absolute/path/to/real-images.json \
  --iterations 5 \
  --output /absolute/path/to/history-cache.json
```

脚本先对每个 mode 做一次不计时 warmup，再按 ABBA/BAAB 顺序交替执行从空
memory 开始的完整 session。轻量 runner 通过 `object.__new__` 复用正式 Low /
Panoramic prompt、image interleave 与 render 方法，不加载模型；processor、chat
template、system prompt 及其 checkpoint SHA 契约与正式 runner 相同。`none` 与
`rgb_bytes` 每一步的全部 processor tensor dtype/shape/hash 必须完全一致。计时包含历史、
当前图和 candidate 的 PIL preparation 以及 processor 调用；checkpoint/processor
加载和 manifest 图像解码排除。结果分别记录 current/history/candidate render、
cache hit/miss/append/eviction、resident bytes/entries、执行顺序、per-session 与
pooled-step mean/p50/p95/steps/s。`none` 不执行 cache lookup，因此 cache miss 为
零；其 raw history render 单独计数。
`rgb_bytes` 每个 session 最多保存 16 MiB 的 immutable post-resize RGB recent
suffix；Panoramic candidate 每一步照常处理且永不进入 cache。

Persistent compile-cache producer/consumer measurements are documented in
[COMPILE_CACHE_REPORT.md](COMPILE_CACHE_REPORT.md).
