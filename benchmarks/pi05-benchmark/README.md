# PI0.5 Benchmark

## Pi05 推理优化的验证入口

新实现通过 `Pi05OptimizationConfig` 显式启用。优化结果见下文独立基线与三平台验证；
其后的历史实验使用各自独立口径。候选算子验证可使用保持原 attention/MLP 的 BF16 配置：

```json
{
  "activation": "gelu_pytorch_tanh",
  "norm_fusion": true,
  "kv_workspace": true
}
```

保存为独立 JSON，在一份新的 benchmark YAML 中设置：

```yaml
dtype: auto
compile_backend: none
native_inference: true
prefix_cuda_graph: true
cuda_graph: true
optimizations: ./fused.json
```

`benchmark.py` 和 `compare.py` 接受 `optimizations`，并记录配方内容；前者相对
YAML 解析配方路径，后者也将配方内容计入 profile 摘要。报告额外记录 CUDA 源码
摘要。不要覆盖现有结果。`compare.py` 保留原 50-action 协议，测量前仍需要
十观测动作检查；它不适用于仅支持 horizon=10 的 Thor/Spark paired-GEMM 配置。

按相同 LeRobot 权重、预处理、随机种子和全部去噪步，使用 recorded 对比入口：

```bash
CUDA_HOME=/path/to/cuda .venv/bin/python compare_recorded.py \
  --checkpoint /models/pi05 --tokenizer /models/tokenizer \
  --samples /data/recorded/sample.json --out runs/fused-parity \
  --warmup 3 --modes lerobot vvla_sdpa_graph vvla_fused_graph \
  --optimizations /path/to/fused.json
```

脚本保存完整归一化/物理动作和逐请求误差，不会自动宣称通过。依次检查
K/V-only（`norm_fusion: false`）、严格融合、alternative attention 和 paired GEMM；
保持权重、dtype、输入/noise、horizon 和去噪步一致。Thor/Spark paired-GEMM profile
限制为匹配 GPU、标准模型尺寸、B1、推理 horizon=10、10-step。

FP8/NVFP4 可分别配置 `prefix_layers` 和 `action_layers`，每层独立选择 gate/up
与 down 的精度；空配置保持该 tower 为 BF16。要求本地 checkpoint SHA256 和
匹配的 activation/numerics 语义。默认 LeRobot 使用 `gelu_pytorch_tanh`，不能直接
复用 exact-GELU profile 的 scale。完整 RLinf profile 和冻结配方见下文。
NVFP4 需要 Blackwell，
mixed GEMM 需要 Torch 2.13 的 `scaled_mm` API；新路径支持 CUDA Graph，暂不组合
Inductor、全局量化或 TP。配置与安装要求见
[模型文档](../../docs/en/models.md#opt-in-fused-operators)。

算子通过 `layers` 的契约与注册表选择，CUDA/Triton/Torch 实现在各自 backend；
通用 Projection 已移出 Pi05。JSON 可增加 `operators` 映射，例如
`{"normalization": "torch", "quantization": "torch", "gelu_mul": "torch", "norm_quant": null}`
选择参考路径。CUDA residual/norm/quant 融合需要 CUDA norm 与 quantizer 配套，
参考组合使用 `norm_quant: null` 进行分开执行。硬件 tile/tail 参数归 Triton backend。
Torch quantizer 仅支持 FP8；NVFP4 的分开执行对照使用 Torch normalization/GELU、
CUDA quantizer 与原生 GEMM，不能将其描述为纯 Torch NVFP4 实现。

本地验证包括 4090 上的严格算子逐位检查、FP8 完整 10-step graph、更新输入、
graph eviction/capture failure、refit 失效及参数身份保持。它们使用小型模型，
不作为 LeRobot 全模型延迟或任务质量结果。NVFP4 正向检查需在 Blackwell 执行。

### PR 加速比的基线

本 PR 的性能基线是加入新优化之前的 EmbodiInfer BF16，保留其原有
Inductor、Triton attention、prefix/full-loop CUDA Graph 和缓存优化。使用 PR 前
提交 `05d603714dd3dcb48c5210810847395da13272aa` 的独立源码目录及独立进程；
基线不导入新增的 ops、`Pi05OptimizationConfig` 或修改后的 policy/runtime。
沿用原 checkpoint 的 BF16 transformer 与 FP32 视觉/projector/norm/action/time
精度区域，不额外改变基线 dtype。

下文的 `optimized_rlinf_bf16` / “完整优化 BF16 runtime” 是候选配置。FP8/NVFP4
相对它的加速只表示量化的额外收益；整个 PR 的加速比应计算为上述原 EmbodiInfer
延迟除以候选延迟。RLinf profile 改变了视觉精度、GELU 和 time/RoPE 数值配置，性能对照需
同时报告动作差异，不能自动视为无损优化。

`compare_optimizations.py` 对独立 checkout 进行计时，检查实际 package 导入路径，
拒绝包含新增优化实现的 baseline 目录，保存完整动作、实际配方、源码/权重/input 摘要
和每次分段计时。B1/H10/完整十步，六条核验观测逐条预热三次，roundrobin 两轮
各 30 次；`highest` 和允许 TF32 的 `high` 分别运行。两种角色使用同一 Python
环境，候选通过设备 preset 展开并记录完整配置：

```bash
python benchmarks/pi05-benchmark/compare_optimizations.py \
  --role baseline --source /reference/embodiinfer-before-pr \
  --baseline-revision 05d603714dd3dcb48c5210810847395da13272aa \
  --checkpoint /models/RLinf-Pi05-LIBERO-SFT --inputs /data/inputs.pt \
  --matmul-precision highest --out runs/original-bf16-highest

python benchmarks/pi05-benchmark/compare_optimizations.py \
  --role candidate --source /workspace/EmbodiInfer \
  --device thor \
  --checkpoint /models/RLinf-Pi05-LIBERO-SFT --inputs /data/inputs.pt \
  --matmul-precision highest --out runs/optimized-highest
```

### 原 EmbodiInfer BF16 与优化配置的对照（2026-10-10）

按上述独立源码/进程协议完成 Thor/Spark 补测，以下均为六观测、60 次请求的同步
wall P50。加速比统一为 **原 EmbodiInfer BF16 wall P50 / 候选 wall P50**。
原有 Inductor、Triton attention 和两个 CUDA Graph 在 baseline 中均开启；运行时
检查所有 EmbodiInfer 模块来自冻结的 PR 前源码，未导入新增 optimization 模块。
“原数值配置 BF16 ops”保留 LeRobot 原 dtype/GELU/time/RoPE；其余候选使用完整
RLinf 优化 profile。两组的 CUDA Graph 缓存计数在测量前后均不变，六观测输出
均为有限值且重复逐位一致；这不是候选与原 baseline 逐位一致的声明。

FP32 GEMM precision = `high`，baseline 保留原有的 TF32 选项：

| 配置 | Thor ms | 相对原 BF16 | Spark ms | 相对原 BF16 |
|---|---:|---:|---:|---:|
| 原 EmbodiInfer BF16 / Inductor | 102.787 | 1.000× | 109.329 | 1.000× |
| 原数值配置 BF16 ops | 93.468 | 1.100× | 109.098 | 1.002× |
| 完整优化 BF16 | 70.189 | 1.464× | 81.308 | 1.345× |
| 完整优化 / FP8 action | 63.494 | 1.619× | 75.744 | 1.443× |
| 完整优化 / NVFP4 prefix | 64.593 | 1.591× | 71.021 | 1.539× |
| 完整优化 / NVFP4 prefix + FP8 action | 58.054 | 1.771× | 65.758 | 1.663× |

FP32 GEMM precision = `highest`，不与上一组交叉计算加速比：

| 配置 | Thor ms | 相对原 BF16 | Spark ms | 相对原 BF16 |
|---|---:|---:|---:|---:|
| 原 EmbodiInfer BF16 / Inductor | 189.137 | 1.000× | 129.545 | 1.000× |
| 原数值配置 BF16 ops | 178.819 | 1.058× | 129.791 | 0.998× |
| 完整优化 BF16 | 69.787 | 2.710× | 83.391 | 1.553× |
| 完整优化 / FP8 action | 63.056 | 2.999× | 75.342 | 1.719× |
| 完整优化 / NVFP4 prefix | 64.715 | 2.923× | 69.109 | 1.874× |
| 完整优化 / NVFP4 prefix + FP8 action | 58.025 | 3.260× | 65.038 | 1.992× |

完整优化的收益包含 BF16 视觉路径、camera batching、prefix/context 优化和数值
profile 变化，不能全部归因于新增 kernel。分段计时中，`highest` 的 Thor BF16
prefix 均值从 153.19 降至 35.79 ms，diffusion 从 35.95 降至 34.03 ms；Spark
分别为 90.08 → 44.41 ms、39.28 → 38.61 ms。这只能定位到 prefix 阶段，尚未
进一步分离其中视觉精度、布局和算子的贡献。

相对同设备/precision 的原 BF16，六观测中完整优化 BF16 的 full32 动作相对 L2
最大值为 0.197–0.254%，组合低精度为 1.966–2.192%；LIBERO7 对应为
0.177–0.240% / 1.944–2.170%。这些是归一化动作差异，没有任务成功率结论。
原始动作、每次计时、实际导入文件清单、两份源码归档、launch 脚本及 SHA256
清单保存在本地忽略目录 `runs/pr5-original-baseline-20261010/`，运行
`collect_summary.py` 可重新计算全部对照。两轮在同一进程内连续测量，未据此
估计独立运行的置信区间，也不将几个百分点的差异解释为稳定加速。

## openpi_rlinf 数值约定与推理验证（2026-10-10）

完整配方显式设置 `numerics="openpi_rlinf"`，对应 RLinf 的 `openpi_rlinf` 推理实现，使用 BF16 视觉 encoder/projector、exact GELU、FP32
time/RoPE、BF16 query prescaling、separate Q/K/V、连续 BF16 down 矩阵布局、
action arithmetic / prefix lookup GEGLU，并启用 active-camera batching、prefix
compaction、末层仅生成 K/V、action context 复用。仅设置 `numerics` 不会自动启用
所有优化开关；完整配方明确选择各项优化。既有 LeRobot 默认语义保持不变。

部署直接选择设备和完整预设：

```python
make_policy(
    "pi05", checkpoint="/models/RLinf-Pi05-LIBERO-SFT",
    device_type="thor", preset="nvfp4-fp8",
)
```

Spark 使用对应设备名称。预设 `bf16`、`bf16-fp8`、`nvfp4-bf16`、`nvfp4-fp8`
分别对应本节四组完整配方；组合名称按 prefix MLP、action MLP 排列。
预设包含执行组合、明确的 `openpi_rlinf` 数值约定和校准选择，factory 自动配置
native 路径、默认 prefix capture、horizon=10 和完整十步。不需要部署方再组合
这些内部选项。其他精度组合可使用显式格式对名称，但仍需要对应校准。
`inference/calibration/rlinf_libero.json` 的 schema 2 按设备、prefix/action、目标
格式组织保护选择和激活范围，并保存 checkpoint identity 与数值语义。内置数据
只覆盖已测的 prefix NVFP4、受保护 action FP8；可分别或组合选用。其他部分/格式
组合需要自己的校准数据，缺失时明确报错。文件名指实际权重来源，不是优化 preset。

自定义权重通过 `calibration` 传入独立校准数据路径。`from_json`/`to_json` 保留完整
配置导入导出，供消融和记录实验使用。`strict` preset 保持 LeRobot tanh-GELU
及 dtype 语义，支持 Thor/Spark/Orin/4090 的严格融合与 K/V 复用。更细的算子、
数值约定和逐层精度选择可以直接构造 `Pi05OptimizationConfig`；
`from_preset(device, name)` 用于检查预设的完整展开结果。校准与 checkpoint、
硬件、部分、格式和数值语义绑定，不能混用。本节性能只覆盖已测组合，
没有推及所有可表达配置。

Thor/Spark 使用相同权重、fixture、noise、B1/H10/完整十步与 `highest` 进行冻结
参考核验。四种精度配置分别为 BF16、受保护 action FP8、prefix NVFP4，以及
prefix NVFP4/action FP8 组合。FP8 保留八组 BF16 投影保护选择；prefix NVFP4 使用
冻结激活范围。没有按核验输出重新校准或调整门限。

六条核验观测的有效 prefix K/V、十步完整 32 维 velocity、最终动作、graph/eager
以及改用相反 noise 后的 graph 回放均与**对应精度的冻结参考**逐位一致。
这不表示低精度与 BF16 等价，也没有任务成功率结论。性能以本文件上一节的
独立原 EmbodiInfer BF16 对照为准；不同 profile、计时范围及统计量不得交叉相除。

运行环境为下文记录的 Python 3.12、Torch 2.13、Triton 3.7.1、Transformers 5.3
及 LeRobot 0.5.1。最终算子/CUDA Graph 回归在 Thor/Spark 各为 87 passed、
1 个按架构跳过。完整 tensor、逐层/逐步误差、冻结参考源码、配方、graph 检查
及环境摘要保存在本地忽略目录 `runs/pr5-complete-migration-20261010/`。
Orin 不支持本次原生 FP8/NVFP4 profile，使用其支持的 BF16 路径。

后续三平台实验保留 LeRobot 数值语义，量化范围也不同，单独报告其条件和失败结果。

## 三平台 LeRobot-profile 验证（2026-10-10）

验证源码基于 `8555ef48ee2901f2e70193e2c441916629612510`，另加 query-major
的旧版 BMM API capability 检查。三台机器使用同一份真实 RLinf-Pi05-LIBERO-SFT
权重，812 个 tensor 严格映射到 LeRobot adapter；权重 SHA256 为
`4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7`。
这不是历史实验中的 LeRobot-SFT 权重。执行语义保持当前 adapter 的 tanh GELU、
原 RoPE 与原生 dtype，**不验证 RLinf/JAX 对齐或任务成功率**。

统一 B1、完整 10 个去噪步，两个真实 224×224 相机和一个 masked 空相机。
从 LIBERO-10 每任务首段首帧取得十条真实观测；0/3/6/9 用于量化校准，其余六条
用于核验。输入 fixture SHA256 为
`264f9cc39cc9500c815095215ff3905a1ebf2973670795a1caad7d5285e60a2c`。
图像数值先按 BF16 舍入，再以 FP32 存入共享 fixture；三台设备读取同一份数据。
噪声由 NumPy `default_rng(1000 + observation_index)` 生成，三设备一致。
每条核验观测预热两次，随后 roundrobin 计时 30 次；CUDA Event 覆盖视觉/前缀
和全部 diffusion，**不含 CPU 预处理、输入传输、动作后处理、加载、编译或预热**。
这些结果不是机器人请求端到端延迟，也不与下文 1,600 帧实验混用。

模型保留 BF16 文本/action transformer，以及 FP32 视觉塔、projector、norm、
action/time 投影等原生精度区域。`highest` / `high` 是 Torch FP32 GEMM precision；
后者允许 TF32，包含该组的 FP32-tensor 参考。两组分别比较，未统一将视觉塔转成 BF16。
Thor/Spark 为 Python 3.12、Torch 2.13.0（分别 cu132/cu130）、Triton 3.7.1、
Transformers 5.3.0、LeRobot 0.5.1。Thor/Orin 均为 MAXN mode 0，未修改时钟/功耗设置。
Orin 的完整模型使用已有 Python 3.12、Torch 2.6.0/cu128、Triton 3.2.0 容器及
相同模型依赖；**Torch 低于 LeRobot 声明的最低版本，因此属于旧环境兼容实测，
不是受支持环境认证**。主机 Torch 2.8/cu126 另行验证算子，未混入模型延迟。

### 性能对照

以下延迟为 GPU elapsed 均值。`existing_inductor` 未启用新增优化配方；“新路径”已经
使用新增 ops，不能用它充当原 EmbodiInfer BF16 基线。

FP32 GEMM precision = `highest`

| 设备 | Horizon | 原 Inductor ms | 新路径 ms | 延迟降低 |
|---|---:|---:|---:|---:|
| AGX Thor | 10 | 189.02 | 178.68 | +5.47% |
| AGX Thor | 50 | 205.59 | 202.05 | +1.72% |
| DGX Spark | 10 | 131.11 | 132.20 | -0.83% |
| DGX Spark | 50 | 135.19 | 143.65 | -6.26% |
| AGX Orin | 10 | 525.08 | 532.62 | -1.44% |
| AGX Orin | 50 | 558.82 | 540.74 | +3.23% |

FP32 GEMM precision = `high`（允许 TF32）

| 设备 | Horizon | 原 Inductor ms | 新路径 ms | 延迟降低 |
|---|---:|---:|---:|---:|
| AGX Thor | 10 | 103.08 | 93.25 | +9.53% |
| AGX Thor | 50 | 119.81 | 116.43 | +2.82% |
| DGX Spark | 10 | 109.39 | 110.48 | -1.00% |
| DGX Spark | 50 | 113.16 | 121.66 | -7.51% |
| AGX Orin | 10 | 363.41 | 373.38 | -2.74% |
| AGX Orin | 50 | 399.62 | 381.51 | +4.53% |

`existing_inductor` 是原有 Inductor + 原 Triton attention + prefix/full-loop
CUDA Graph；新路径也开启 graphs，但不组合 Inductor。新路径在 H10 的 Thor/Spark
使用 `paired_flash`，其余使用 `folded_flash`。H10 是显式改变推理 horizon，
不是把默认 H50 的动作截断；checkpoint 名义 horizon 仍为 50。
单组 30 次测量不能证明跨进程稳定收益，约几个百分点的差异需进一步复测。
Spark 上原有 Inductor 更优，不能把相对 native SDPA 的收益宣称为超越原有最佳路径。

### 正确性与限制

在查看优化候选输出前，按每条观测的 native BF16 与同 precision 的 FP32-tensor
参考差异冻结门限：max-abs / RMSE 各取两倍，下限 `1e-4` / `1e-5`。保存全部
32 维归一化动作，数值门限检查 LIBERO 的前七维。K/V-only 在三台设备的全部
32 维输出均逐位一致；所列 BF16 路径通过六条门限检查，重复输出逐位一致，
计时前后图缓存数量保持不变。融合 norm/MLP 与 alternative attention 的完整模型
输出存在浮点差异，不能称为逐位无损，也不能据此推断任务成功率。

Orin 本地 FP32 模型遇到大块 CUDA 分配失败；后续运行复用对应 Thor run 的
FP32 参考，逐项核对权重/input hash、horizon、B、steps、noise 和 matmul precision。
已有 H10 的 Orin/Thor FP32 参考在全部 32 维最大差异为 `1.49e-6`；更换参考未改变
Orin 的 BF16 baseline 输出。原始失败日志保留，未将失败尝试计入性能表。

统一 action-MLP FP8 在 Thor/Spark 分别仅通过 2/6、1/6 条核验，最大误差
0.02301/0.02279；NVFP4 均为 0/6，最大误差 0.04663/0.04528。校准固定使用四条
独立观测和 10% headroom，未按候选结果调整门限。**这些配方不推荐部署**。
该实验量化全部 action MLP，与保护八组敏感投影的 FP8 配置、以及
NVFP4 prefix / BF16 action 配置的量化范围不同，其误差结果需分别报告。
Orin 不具备本实现要求的 FP8/NVFP4 原生 GEMM 能力。硬件 profile 未覆盖 Orin 的
paired GEMM，query-major 在其 Torch 2.6 模型环境中因 BMM API 缺失明确不可用。

首轮 attention/operator 回归：Thor/Spark 各 73 passed、1 skipped；Orin 主机
70 passed、4 skipped；Orin 模型容器 68 passed、6 skipped。可选能力按实际硬件/API
跳过，包含 NVFP4、scaled_mm 和旧版 query-major；不是把未执行的能力记为通过。

### 低精度范围补测

`prefix_layers` 与 `action_layers` 独立选择每层精度。原有全局 `quantization` 在
这些实验中为 `None`，且 policy 会在加载前拒绝它与 `optimizations` 同时开启。

补测固定 B1/H10/完整十步、`highest`、相同权重/input/noise 和原冻结门限。
FP8 使用 Thor/Spark 各自冻结的八组 BF16 保护选择（28/36 组 FP8），
全部激活范围按当前 adapter 的四条独立观测重新校准；不复用旧 scale，
也不按六条核验结果调整保护选择。NVFP4 只量化 prefix MLP，action 保持 BF16。
本轮 LeRobot-profile 执行全部 18 层 prefix MLP；RLinf profile 省略最后一层
不影响输出的 MLP，因此两组计时不能作为同一数值 profile 的对照。

| 配置 | Thor 通过数 | Thor 最大动作差 | Spark 通过数 | Spark 最大动作差 |
|---|---:|---:|---:|---:|
| 受保护 action FP8 | 5/6 | 0.008395 | 5/6 | 0.012037 |
| prefix NVFP4 / action BF16 | 0/6 | 0.083918 | 0/6 | 0.062972 |

二者均可执行完整 CUDA Graph，重复动作逐元素一致，计时期间图缓存稳定。
CUDA 融合与分开执行对照的完整 32 维动作在两台设备上均逐位一致；FP8 使用
Torch norm/GELU/encoder 对照，NVFP4 使用 Torch norm/GELU 与同一 CUDA encoder/GEMM。
后者只能排除 pointwise 融合带来的额外误差，不能视为独立 NVFP4 编码器验证。
本轮 native BF16 动作与同设备首轮 baseline 也逐位一致。

**能够运行低精度、通过算子回归、通过数值门限是不同结论。** 补测配方仍未
全部通过门限，未作为默认或部署推荐。LeRobot 与 RLinf profile 的 GELU 和视觉
精度不同，激活范围不能混用。本次补测没有运行闭环任务质量实验。

补齐后的回归在 Thor/Spark 各为 80 passed、1 skipped，包含 prefix/action
FP8/NVFP4 图回放、NVFP4 分开执行对照、组合 prefix NVFP4/action FP8，以及 refit 失效。
原始报告、冻结格式选择、重新校准配方和环境记录保存在
`runs/pr5-lowp-audit-20261010/`。Spark 首次选择 Torch 4-bit encoder 的显式失败
单独保留；有效 NVFP4 分开执行对照来自 `lowp-prefix-reference-v1`，未覆盖失败报告。

### 复跑与报告

在仓库根目录及匹配的隔离环境执行，先准备兼容的 checkpoint JSON、完整 tokenizer
和真实权重。不要改用缩小词表或随机输入：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/path/to/cuda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python benchmarks/pi05-benchmark/validate_optimizations.py \
  --checkpoint /models/pi05 --dataset /data/libero/libero_10 \
  --inputs /data/pi05-recorded.pt --export
python benchmarks/pi05-benchmark/validate_optimizations.py \
  --checkpoint /models/pi05 --inputs /data/pi05-recorded.pt \
  --out /reports/pi05-h50-high --horizon 50 --matmul-precision high \
  --warmup 2 --iterations 30 --modes native strict folded_flash existing_inductor
```

分别以 `--matmul-precision highest` 和 `high` 运行。H10 使用 `--horizon 10`，
在对应 Thor/Spark 上可增加 `paired` / `paired_flash`；`fp8` / `nvfp4` 会用独立
校准观测生成本次配方并保留失败结果。小内存设备可指定
`--fp32-reference /reports/matching-thor-run`，脚本会验证兼容性并记录外部参考 hash。
设置 CUDA_HOME，并为 Inductor 提供匹配 Python 开发头文件；Spark 本次通过独立
CPATH 提供头文件，未修改原环境。

补测可追加 `--precision-map /recipes/guarded-action-fp8.json` 或
`--precision-map /recipes/prefix-nvfp4.json`。文件内 `action_layers` / `prefix_layers`
各为 18 条只含 `gate_up` / `down` 格式选择的列表；脚本拒绝传入旧激活范围，
使用独立校准观测重新采集，并另存 checkpoint-bound `*-recipe.json`。
文件名作为模式名，多个映射可在同一进程中比较；对照算子可通过 `operators` 指定。

原始 JSON、完整动作、逐样本门限、分段/墙钟时间、环境 spec/ledger 与 SHA256
清单位于 `runs/pr5-multigpu-20261010/`；该目录按 benchmark 约定被 Git 忽略。
`summary.json` 明确选用成功的复测轮次，`report.md` 为汇总；先前的依赖/分配器失败
另行保留，不回填成成功结果。保留基准 commit 源码快照、兼容性补丁和实际执行的 benchmark 摘要；新报告还记录源码树 hash。

## 实验方法

模型：LeRobot `pi05_libero_finetuned_v044` 检查点，使用其 tokenizer、状态归一化及动作反归一化。数据：[LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) 的 `libero_10/*.hdf5`：任务文件名字典序取 10 个，每任务按 numeric demo ID 取前 10 段，每段包含首尾均匀取 16 帧，共 **1,600 帧**。

输入 agentview、wrist RGB、EEF position3 + axis-angle3 + gripper2 及任务指令。启用 Inductor 与完整去噪循环 CUDA Graph，10 个去噪步，每次输出 50×7 动作块。预热 10 次，在所选样本中均匀取点以覆盖任务。

统一 B=1、BF16、seed=42，正式测量 1 遍。计时从已解码 CPU RGB/状态开始，到 CPU 输出结束，包含预处理、模型执行、后处理及两端 CUDA 同步；磁盘读取/解码、模型加载和预热（含编译/图捕获）单独排除。**无需仿真**，不计算任务成功率、SR/SPL。

指标：mean / P50 / P95 / P99 延迟、calls/s、action slots/s、tokens/s、CUDA 峰值 allocated/reserved、进程峰值 RSS，以及加载/预热时间。吞吐为总输出数除以总推理时间；action slots/s 不等于机器人控制频率。Thor 的 CPU/GPU 共享物理内存，RSS 与 CUDA 内存不能相加。

本轮分段计时保留原端到端 `latency_ms`，额外记录 `model_timing_ms`：`prefill_ms` 为视觉/文本/状态编码的 CUDA elapsed，`decode_ms` 为全部去噪或 token 生成步骤的 CUDA elapsed，`gpu_inference_ms` 为二者合计；`pure_inference_ms` 为同一完整模型区间的同步墙钟时间，包含模型内部 CPU 调度及采样等待。输入预处理与输入 H2D 在计时前完成，动作反归一化、文本/动作解析及输出 D2H 在计时后执行。两段之间不插入同步；CUDA elapsed 包含流上的等待间隙，不是逐 kernel 时长相加。`model_calls_per_second` = 调用数 / 总纯推理墙钟时间，与原 E2E `calls_per_second` 分开报告。原 E2E 范围保留，但新测量含计时探针开销。新报告 schema 为 `rlinf_offline_performance_v2`，旧报告未测量的字段不回填。

## 运行

在本目录执行，依赖配套的 EmbodiInfer 源码和本平台已验证的 Torch 环境。2026-09-07 的旧结果基于 `b86cce30ba57`；本轮分段计时需使用当前工作树中同时更新的 EmbodiInfer，实际 Python 源码摘要记录在报告的 `environment.source_python_sha256`。`setup_env.py` 为本目录创建独立 `.venv`；`benchmark.py` 自带数据采样和计时代码。

```bash
# Thor: --platform thor；4090: --platform 4090，并选择一张空闲 GPU。
CUDA_VISIBLE_DEVICES=0 python3 setup_env.py --platform thor --runtime-python /path/to/verified-runtime/bin/python --inference-root /path/to/EmbodiInfer
# 修改 config.yaml 中的模型、数据路径；新结果默认写入 runs/。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py --validate-data
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark.py
```

Thor 使用 Torch 2.13.0+cu132，4090 使用 2.13.0+cu129，模型依赖分别见 `requirements-thor.txt` / `requirements-4090.txt`。`--config /path/to/config.yaml` 可使用另一份配置。选定数据缺失时会报错，不以随机输入替代。

Transformers 5 的 GemmaTokenizer 需要包含完整词表的 `tokenizer.json`。只有 `tokenizer.model` 时可能静默加载成 5-token 词表。首次运行前，在实际使用的 Hugging Face 缓存目录执行下面的检查；缺少 JSON 时，脚本从已有的原始 SentencePiece 模型生成它，已有 JSON 时只核验。检查包含 10 个 LIBERO 任务指令、状态文本、空白和 Unicode，共 119 条文本，要求 token ID 与 SentencePiece 完全一致。`benchmark.py` 还会在加载权重前检查词表和三条不同指令，并把实际词表大小及 token ID 写入报告。

```bash
export HF_HOME=/path/to/huggingface-cache
.venv/bin/python prepare_tokenizer.py \
  --tokenizer-dir /path/to/huggingface-cache/hub/models--google--paligemma-3b-pt-224/snapshots/ACTUAL_SNAPSHOT \
  --dataset-root /path/to/libero_10 \
  --output runs/tokenizer-validation.json
```

## 实验结果

2026-09-08 修复 tokenizer 后，在 Thor MAXN 完成 1,600 帧分段计时。输入词表为 257,152 个 token；119 条文本与原始 SentencePiece 编码完全一致，见 tokenizer 核验 (`reports/thor/tokenizer-validation.json`)。保留 SDPA、Inductor 和完整 10 步去噪 CUDA Graph。预热后及测量结束均为 1 个 LoopGraph、1 个编译视觉编码器、1 个编译 prefix 编码器和 4 个编译推理 helper，测量中没有新图捕获。selection SHA256 为 `92be73f628b54f35c5f0ece7ea0e6babc4a6e3b3a4fd43adc0662bbfa516eb69`。

| 模型 / 数据 | 硬件 / 日期 | Calls | E2E mean ms | Prefill mean ms | Decode mean ms | 纯推理 mean ms | E2E calls/s | 纯推理 calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pi05-libero10 | Thor / 09-08 | 1,600 | 164.05 | 62.45 | 98.42 | 160.90 | 6.0956 | 6.2152 |
| pi05-libero10 | RTX 4090 / 09-08 | 1,600 | 74.77 | 36.69 | 34.87 | 71.59 | 13.3735 | 13.9694 |

Thor E2E P50/P95/P99 为 163.95/165.01/165.50 ms；完整模型 CUDA elapsed 为 160.87 ms，纯推理同步墙钟为 160.90 ms。两者都覆盖全部 10 个去噪步，不能当成单个去噪 step 的时间。一台单卡 RTX 4090 工作站已按相同完整词表、10 个去噪步和测量代码完成 1,600 帧；E2E P50/P95/P99 为 74.71/75.50/77.11 ms。两平台图捕获和编译数量在正式测量中均保持不变。

另取 10 个跨任务观测，用相同权重、输入和逐观测随机种子，对比分段计时路径与当前普通 `EngineCore.execute`；`50×7` 物理动作逐元素完全一致，最大绝对差为 0，见 路径一致性记录 (`reports/thor/timing-parity.json`)。该检查针对同一进程内的两条执行路径，不宣称与旧报告或另一硬件的输出 SHA256 完全相同。

4090 的 tokenizer 核验 (`reports/4090/tokenizer-validation.json`) 同样通过全部 119 条文本；路径一致性记录 (`reports/4090/timing-parity.json`) 的 10 个跨任务输出最大绝对差为 0。4090 原始 JSON SHA256 为 `2ab09a20c81d460da37755d96d2620aac25f1c4d8e2b763771724ae115e67571`，源码摘要为 `1ce73cd69d5fbfb03430eced40a58b89de474633a717505f0a469a45ccc4d0ec`。Thor 重跑后新增了独立 `prepare_tokenizer.py`，使完整目录摘要发生变化；两平台 `benchmark.py` 和 EmbodiInfer 推理源码一致。

此前 Thor 的 tokenizer 缓存仅有 `tokenizer.model`，缺少 Transformers 5 使用的 `tokenizer.json`，实际加载为 5-token 词表，普通指令被编码为 `[2, 3]`。旧的 165.38 ms Thor 结果只保留为历史记录，不作为有效基线；同进程两条路径输出一致不能证明 tokenizer 本身正确。09-07 的旧 4090 环境没有复核，也不纳入当前对照。

当前 Thor 报告包含逐调用数据、完整配置、样本摘要、实际词表、内存和编译/图回放状态。实际 Python 源码 SHA256 为 `c8c4a5785c3ab1a7539fdcd4ade75568dd9ecad6c559a99bcd539fbef76cfd45`，原始 JSON SHA256 为 `514ac8ff86eaafda56cf32905cf90dada812aa3c61390e767fd5c790aa1c5726`。这是 B=1、固定完整步数下的结果，未穷举所有 batch size 或优化组合。

本轮 8 个 benchmark 共用 20260908223127.tar.gz (`../snapshots/20260908223127.tar.gz`) 和一份 SHA256 (`../snapshots/20260908223127.tar.gz.sha256`)。包内只包含必要脚本、依赖文件、README、实际运行配置、14 组 Thor 与 14 组 RTX 4090 分段计时报告，以及相关数值核验记录。PI0.5 使用修复 tokenizer 后的新结果；历史报告、模型框架、权重、原始数据和虚拟环境不进入本包。报告及快照不提交 Git。

配套 EmbodiInfer 源码基线为 `72cba6e4ae05312bd93c69918709f9b792869e61`，使用本包附带的 benchmark 脚本。报告旁的 `*-config.json` 保存该次实际运行配置；复跑时调整本机模型、数据和输出路径，再传入 `--config`。同时设置 `HF_HOME` 指向通过 tokenizer 核验的缓存；原运行路径记录在配置的 `benchmark_context.HF_HOME`。两平台的采样、配置和源码核验见 协议核验记录 (`reports/4090/cross-hardware-protocol.json`)。

## 4090 竞品对比（2026-09-09）

使用一台 RTX 4090 24 GiB 工作站，驱动 `580.173.02`，GPU 测量串行执行。每组回放 LIBERO-10 的 10 个字典序任务文件、每任务前 10 个数字序 demo、每 demo 含首尾均匀取 16 帧，共 1,600 条离线观测；不启动仿真或机械臂。数据源 `yifengzhu-hf/LIBERO-datasets` revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`。选帧 SHA256：`92be73f628b54f35c5f0ece7ea0e6babc4a6e3b3a4fd43adc0662bbfa516eb69`。

检查点为 `pi05_libero_finetuned_v044_gitcode`，输入两个 224px 相机、状态和指令，10 步去噪，内部 `50×32`、输出物理动作 `50×7`。保持 MEAN_STD 归一化和完整 257,152 词表；119 条文本的 SentencePiece 对照来自既有 tokenizer 核验。EmbodiInfer 的额外空相机完全 masked，C++/PhyAI 仅计算两个有效视角。

### 计时与配置

E2E 从已解码的 CPU 图像/状态/指令到 CPU 物理 action chunk，包含预处理、噪声生成、H2D/D2H、同步、完整模型和后处理。完整推理只计 GPU 输入就绪后的视觉/文本编码与全部去噪。磁盘读取、图像解码、首次模型加载、初始化预热和凑批等待不计入；请求触发的 C++ 建图/更新/再预热计入 E2E。C++ 内置分段计时范围不同，完整推理列留空，避免混用。

EmbodiInfer 为 BF16、SDPA、Inductor 和完整去噪 CUDA Graph；PhyAI 为原生 autotune、CUDA Graph、非量化运行。测量前覆盖 10 条跨任务参考并预热，batch 核验覆盖相同 10 条输入。正式测量期间 EmbodiInfer 的图/编译状态及 PhyAI 的 kernel 选择缓存保持稳定。各实现独立使用 `.venv`、`.venv-vlacpp`、`.venv-embodied`、`.venv-phyai`。

### B=1 全量结果

下表所有行都完成 1,600 条回放；数值核验通过条数单独列出。吞吐单位为真实 observation/s。

| 实现 | 平均 E2E (ms) | 平均完整推理 (ms) | E2E 吞吐 (obs/s) | 数值核验 |
| --- | ---: | ---: | ---: | ---: |
| EmbodiInfer | 74.33 | 71.84 | 13.45 | 10/10 |
| vla.cpp | 79.36 | — | 12.60 | 10/10 |
| Embodied.cpp + correctness patch | 89.99 | — | 11.11 | 10/10 |
| PhyAI | 41.31 | 39.14 | 24.21 | 9/10 |

### Batch 吞吐扫描

每个 batch 都处理独立观测和独立噪声；末批复制最后一条输入填充，填充槽位不计入吞吐。延迟为整批完成时间。C++ 公开接口仅支持 B=1。

| 实现 | Batch | 平均整批 E2E (ms) | 平均整批完整推理 (ms) | E2E 吞吐 (obs/s) | 数值核验 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EmbodiInfer | 2 | 118.01 | 112.21 | 16.95 | 10/10 |
| EmbodiInfer | 8 | 423.01 | 408.01 | 18.91 | 10/10 |
| EmbodiInfer | 16 | 848.15 | 819.54 | 18.86 | 10/10 |
| EmbodiInfer | 32 | 1736.32 | 1676.58 | 18.43 | 10/10 |
| PhyAI | 2 | 67.01 | 62.82 | 29.85 | 9/10 |
| PhyAI | 4 | 120.37 | 112.10 | 33.23 | 9/10 |
| PhyAI | 8 | 227.31 | 211.26 | 35.19 | 9/10 |
| PhyAI | 16 | 446.48 | 414.31 | 35.84 | 10/10 |
| PhyAI | 32 | 898.85 | 818.31 | 35.60 | 10/10 |

EmbodiInfer 已测最高吞吐为 **B=8，18.91 obs/s**（该配置数值核验 10/10）。最高仅指实际完成测量的配置，不代表理论上限。
PhyAI 已测最高吞吐为 **B=16，35.84 obs/s**（该配置数值核验 10/10）。最高仅指实际完成测量的配置，不代表理论上限。

扫描停止原因保留在 诊断日志 (`reports/4090-competitors/diagnostics`)。

- `phyai-autotune-fresh-batch-8`：FlashInfer 临时工作区不足，后续扩大工作区重试。
- `phyai-autotune-unchecked-batch-2`：持久化 autotune 缓存导致捕获期间首次初始化，后续重新调优重试。
- `phyai-autotune-workspace1024-batch-32`：FlashInfer 临时工作区不足，后续扩大工作区重试。
- `phyai-autotune-workspace2048-batch-64`：FlashInfer 临时工作区不足，后续扩大工作区重试。
- `phyai-autotune-workspace4096-batch-64`：CUDA 显存不足。
- `phyai-autotune-workspace512-batch-16`：FlashInfer 临时工作区不足，后续扩大工作区重试。

EmbodiInfer 前轮 PI0.5 B=4、GR00T B=2 各有一条参考超出数值门槛，未计完整性能；PI0.5 B=64、GR00T B=128 显存不足。本次按用户要求补测 PhyAI，保留 EmbodiInfer 既有结果。

### 数值核验与继续计时

每条观测使用 `np.random.default_rng(42 + 选帧序号)` 产生 FP32 高斯噪声，再舍入到 BF16，显式传给各引擎。先在 EmbodiInfer FP32 算术路径中使用相同 BF16 舍入权重/输入/噪声，测量相对 EmbodiInfer BF16 的差异；在查看竞品输出前，按每条观测分别冻结 max-abs/RMSE 门槛，取该差异两倍，下限 `1e-4/1e-5`。比较归一化动作，门槛与原始报告见 tolerance-contract.json (`reports/4090-competitors/tolerance-contract.json`)。这是数值筛选规则，不是官方任务正确性标准或任务成功率。

`compare.py --mode measure` 默认要求同一配置的 10 条核验通过；`throughput.py` 默认也受该门槛约束。用户明确要求忽略精度问题后，PhyAI 使用 `--allow-numerical-mismatch` 完整测量：仍要求对应配置的核验记录、相同输入、正确输出形状及有限值，报告 `passed` 保留数值结果，`measurement_complete` 单独表示回放完成，`allow_numerical_mismatch` 记录本次选择。全量报告没有将失败核验改为通过。

PhyAI 首次默认图模式和 autotune 均为 9/10 通过；关闭图复用也未解决数值差异。全量结果采用对应新配置的核验记录，具体通过条数见上表；门槛从未放宽。

PhyAI 的 patch stem 使用 FP32 计算、Euler 状态使用 FP32 累加，EmbodiInfer 本配置为 BF16 状态；这是已确认的实现差别，尚未通过逐层对照证明它解释了全部误差。

### 复现依赖与适配

源码固定为 [vla.cpp](https://github.com/VinRobotics/vla.cpp) `f386c16347094eb2cb183edf39677cf1b6bcc750`、[Embodied.cpp](https://github.com/SEU-PAISys/Embodied.cpp) `1dad33f2c87ee1d390808cb5d776cd8c998f4a36`、[PhyAI](https://github.com/mingti-org/phyai) `a0abb211c4b05b21b766d0f6f6840eed72a8fcec`。C++ 构建为 CUDA 12.6.85、sm_89、Release；vla.cpp 使用 llama.cpp b10729、统一 `--use_fast_math` 保留 device LTO；Embodied.cpp 使用 llama.cpp b9016 及上游 pi05/groot-n1/cuda-parity 补丁。实际构建参数、差异和动态库 SHA256 见 build-provenance.json (`reports/4090-competitors/build-provenance.json`)，GGUF dtype 统计见 precision-provenance.json (`reports/4090-competitors/precision-provenance.json`)。

`prepare_cpp.py` 记录转换摘要：vla.cpp 转换前仅移除原检查点 `model.` 命名空间，812 个 tensor 数值保持不变，并修正其硬编码的归一化元数据为实际 MEAN_STD；Embodied.cpp 直接接受原检查点。C++ 使用原生 BF16/F32 权重存储与 F32 激活，PI0.5 路径不支持通过全局开关切换 BF16 激活或统一 Flash Attention；Embodied.cpp SigLIP attention 为 auto。

Embodied.cpp 应用 [常量存储修复](embodied-pi05-constants.patch)：保留持久图位置/mask/时间输入的存储，防止分配器复用覆盖；不改变权重、算术和步数。标注 `+ correctness patch`，另有 12 次重复/变形状回放核验 (`reports/4090-competitors/embodied-constants-replay.json`)，同一输入/噪声输出逐位一致。

PhyAI 独立环境为 Torch 2.11.0/CUDA 13.0、Transformers 5.8.1、h5py 3.16.0。FlashQLA 0.1.2 要求 TVM-FFI 0.1.9，使用兼容的 CUTLASS DSL[cu13] 4.5.2；两个环境均通过 `uv pip check`。EmbodiInfer 沿用 Torch 2.13.0+cu129、Transformers 5.3.0。完整版本见 environments.json (`reports/4090-competitors/environments.json`)。

PhyAI 的持久化 kernel 选择缓存会跳过某些后端预热，导致首次 cuDNN 分配发生在捕获期间。本次 B=1 和 PI0.5 补测设 `autotune_cache: null`，每个进程重新 autotune，初始化不计时，CUDA Graph 保持开启；GR00T B=2–64 保留原缓存路径，B=128 重新调优。默认工作区为 256 MiB，planner 不足时按新配置扩展至 512 MiB 或更大，具体容量写入对应报告；未修改 PhyAI 模型或 kernel 源码。

复跑时将 `*-config.json` 中模型、数据、源码、库、参考目录与输出路径改为本机路径，设置 `PYTHONPATH` 指向 EmbodiInfer、`HF_HOME` 指向缓存。使用对应虚拟环境执行 `compare.py --config <配置> --mode check`，再为同一配置设置新 `gate_report/output` 后执行 `--mode measure --allow-numerical-mismatch`；批量使用 `throughput.py --config <配置> --batch-size N --output <新报告> --allow-numerical-mismatch`。输入、步数、噪声和冻结门槛保持不变。APXinf 本轮未取得可复现执行入口，未测。

### 快照

两模型共用 20260909105724.tar.gz (`../snapshots/20260909105724.tar.gz`) 和一份 SHA256 (`../snapshots/20260909105724.tar.gz.sha256`)。仅包含必要脚本、README、依赖/构建记录、实际配置、全量逐调用报告及相关核验；不含权重、原始数据集、虚拟环境、动态库。报告和压缩包均由 `.gitignore` 排除。

远端实际 HEAD 为 `72cba6e4ae05312bd93c69918709f9b792869e61`，benchmark 使用本包版本；收集清单 (`reports/4090-competitors/collection-manifest.json`) 记录报告与脚本 SHA256，适配器变更另有逐文件差异。

## PI0.5 原生优化（2026-09-09）

分支 `perf/pi05-native-inference` 基于 `main@06e82ca80f6385079ef7390e32466180088b4d80`。
迁入上游优化分支的固定时间表/AdaRMS 缓存、prefix graph、融合算子及梯度路径回退；
模型布局、调度与缓存位于 `embodiinfer/policies/pi05/`，可复用张量算子位于
`embodiinfer/backend/triton/`；分段 K/V attention 通过 `embodiinfer/layers/attention.py` 注册和选择。
参考 PhyAI 的布局与缓存方法，增加有效相机裁剪、语言长度分档、自行实现的 Triton
prefix/分段 KV 注意力，以及 Q/K RoPE 融合。运行环境**没有安装或导入 FlashInfer**；
不用 PhyAI 或其他推理框架执行 EmbodiInfer 模型。

### 相同 B=1 口径的完整回放

复用上文同一 RTX 4090、检查点、BF16 输入/权重、10 个去噪步、50×7 物理动作和
LIBERO-10 的 1,600 个样本。噪声为每条观测 `42+index`，固定数值阈值不变。
所有布局在计时前预热，完整测量期间图数量/缓存数量不变；不计磁盘读取、解码、
模型加载、编译与预热。新结果仅代表此 GPU 的 B=1，尚未重测 Thor 或 batch 吞吐。

| 实现 | 平均 E2E (ms) | 平均完整推理 (ms) | E2E 吞吐 (obs/s) | 数值门槛 |
| --- | ---: | ---: | ---: | ---: |
| 原 EmbodiInfer | 74.33 | 71.84 | 13.45 | 10/10 |
| 本分支 EmbodiInfer | 38.89 | 36.54 | 25.71 | 10/10 |
| PhyAI（前述固定版本/环境） | 41.31 | 39.14 | 24.21 | 9/10 |

新结果的 prefix / 完整去噪 CUDA elapsed 均值为 **22.31 / 14.21 ms**。
相对原 EmbodiInfer，E2E 降低 **47.7%**，吞吐为原先的
**1.91×**。
动作核验使用既有逐样本冻结门槛；10/10 通过不等于与原实现逐位一致，也不代表
仿真任务成功率。真实权重的同路径 eager/graph、后续请求输出隔离以及 refit 后重建
另由 测试记录 (`reports/4090-native/native-v6/tests.log`) 验证（16 项通过）。
完整 CPU 回归 (`reports/4090-native/cpu-final.log`) 为 407 通过、49 跳过；Ruff 与 diff 检查通过。

运行源码 SHA256：`47236c7c5a86013b034b7467cb0815110ee3dc1a5c334f620323e1c2066ce147`；完整结果 SHA256：`b803b23302d5b67375c84b092d7aec58e2124b3d39337a530c32c249a77f0124`。
收集及协议核验 (`reports/4090-native/verification.json`) 检查逐帧 ID、冻结门槛、配置、
全部 1,600 行、均值/吞吐、图缓存稳定性，以及本地代码与实际运行版本的一致性。

### 开启方式

`make_policy("pi05", ...)` 或本目录 `benchmark.py` 配置增加以下参数；默认关闭，便于
与原始路径对照。完整去噪图同时需要 `EngineConfig(use_cuda_graph=True,
capture_full_loop=True)`。优化只用于 CUDA eval 推理；训练/梯度和任意时间的原始
`denoise_step` 保留参考路径。`return_hidden=True` 保留原始 prefix 布局。

```yaml
native_inference: true
prefix_cuda_graph: true
prefix_attention: triton
denoise_attention: triton
compile_backend: inductor
cuda_graph: true
```

`runtime.py` 拥有按形状/stream 缓存的 prefix/完整去噪图和每个时间步的 AdaRMS 投影；
`backend/triton/split_kv_attention.py` 直接读取两段 K/V，按 KV 分片计算并稳定合并 softmax；
`backend/triton/norm.py`、`activation.py`、`rotary.py` 分别提供 AdaRMS/门控残差、GELU 和 RoPE 融合。
PI0.5 policy 负责构造模型所需的 mask、选择算子和管理缓存。语言 bucket 为
16/32/48/64/96/128/160/200，只删除公共尾部 padding；完全 masked 的相机不再经过 SigLIP。
残差与 RoPE 禁止跨 BF16 中间舍入做 FMA 融合，防止当前 Triton 版本消除舍入。
refit、设备/dtype 迁移与 train/eval 切换会清理派生缓存。

复现本表使用 实际配置 (`reports/4090-native/native-v6/check-config.json`)，修改本机路径后：

```bash
# 与上述独立 .venv 相同，不安装 FlashInfer。
PYTHONPATH=/path/to/EmbodiInfer HF_HOME=/path/to/cache \
  .venv/bin/python compare.py --config /path/to/check-config.json --mode check
# 保持配置相同，仅把 output 改为新文件、gate_report 指向刚生成的 check 报告。
PYTHONPATH=/path/to/EmbodiInfer HF_HOME=/path/to/cache \
  .venv/bin/python compare.py --config /path/to/measure-config.json --mode measure
```

固定阈值与参考样本沿用竞品实验归档，首次重建流程仍使用 `compare.py` 的
`reference` / `calibrate` 模式（用原始 baseline 配置关闭 native_inference，不能用待测优化校准）；
比较前必须冻结阈值，不能根据新输出调整。

本次 PI0.5 优化单独归档为 20260909123408.tar.gz (`../snapshots/20260909123408.tar.gz`)
及 SHA256 (`../snapshots/20260909123408.tar.gz.sha256`)。只包含最终结果相关的脚本、
本次 PI0.5 优化源码、实际配置、校验参考、原 VVLA/PhyAI 对照和逐帧报告；
不含 checkpoint、完整数据集、虚拟环境或编译库。之前多模型快照保留。

## AGX Orin mixed precision (2026-09-17)

这是三相机、12 维动作的 `pi05-chips-30000` 微调 checkpoint 对照，与上面的
LIBERO/Thor/4090 实验独立。AGX Orin 32GB、30W，PyTorch 2.8.0 / CUDA 12.6、
Transformers 5.3.0、LeRobot 0.5.1。Gemma 权重 BF16，视觉、归一化、动作和时间
投影 FP32；B=1、10 次去噪、内部 50×32、输出 50×12，未量化或缩短动作块。

同一 episode 242 的 frame 0、18、23、27，种子 1000、0、2026，每配置 12 次。
参考和候选显式检查 FP32 初始噪声逐元素一致；全部 600 个动作值均参与比较。
各配置先预热一次，再顺序测量，当时无其他模型推理。所有模式在同一进程内
保持相同 matmul precision；原始日志未单独记录该值，新复现脚本显式设置并记录。
视觉 attention 各配置均为 SDPA，表中 eager/SDPA 指
Gemma 骨干和动作专家。实际动作是反归一化、控制层裁剪前的 checkpoint 坐标。

| 后端 | P50 ms | P95 ms | 最大归一化动作差 | 最大实际动作差 | GPU 峰值分配 MiB |
|---|---:|---:|---:|---:|---:|
| LeRobot | 2751.8 | 2823.7 | 0 | 0 | 7082 |
| EmbodiInfer eager | 2565.7 | 2568.3 | 0 | 0 | 8072 |
| EmbodiInfer eager + 完整去噪图 | **2311.7** | **2319.8** | **0** | **0** | 8106 |
| EmbodiInfer SDPA | 2889.9 | 2897.5 | 0.06924 | 2.9954 | 8100 |
| EmbodiInfer SDPA + 完整去噪图 | 2787.0 | 2792.0 | 0.06924 | 2.9954 | 8125 |

测量从已解码 CPU 图像/状态进入预处理，到完整动作块反归一化并回到 CPU。
包含设备传输、预处理、完整推理和后处理；排除磁盘/图像解码、加载、首次图捕获、
相机采集、HTTP/Wi-Fi 和执行器。12 次样本的 P95 不代表长时间尾延迟保证。

所选 eager + 图的中位延迟降低 **15.99%（1.19×）**；prefix 约 1964 ms，
去噪约 341 ms，普通 EmbodiInfer eager 去噪约 592 ms。计算结构和模型规模未减小。
单独插桩的线性层/卷积乘加计数约 4.787 TFLOPs，排除 attention 矩阵乘、归一化
等算子，不是完整硬件 FLOPs。显存增加约 1 GiB 主要来自 Engine 将原先 CPU
offload 的词嵌入搬到 GPU；图本身相对 EmbodiInfer eager 约增加 33 MiB。这里记录的是
PyTorch GPU 分配，不能与 Jetson 进程 RSS 相加。

种子 1000 的右夹爪完整轨迹逐元素一致，四帧峰值依次为 64.9038、69.2670、
63.6015、67.3609。张开/闭合方向由实际标定决定；本实验不证明实机抓取成功率。
SDPA 在此条件下既没有提速，也不满足逐位一致，因此没有作为这次默认替代。

**代码与证据范围：** 上表在 EmbodiInfer 部署分支 `fix/pi05-mixed-inference-20260917`
（基于提交 `6f65961`，加本次混合精度和原生嵌入改动）测得。迁入 inference
主线后保留已有融合 SDPA / compact-layout 优化，故不能把表中 SDPA 数据当成
当前主线所有优化组合的速度。迁入主线后已重新运行同一
30,000-step checkpoint：4 帧 × 3 种子，完整 50×12 归一化和反归一化动作均
与 LeRobot 逐元素一致（最大差 0）。本次回放复用已验证的部署流式加载器，
候选执行的是 inference 仓库的 `LeRobotPi05Adapter`；不据此声称新加载器或
新复现脚本已完成全量性能测量。AGX 小型组件和 CUDA 图验证通过，CPU 全套
445 项通过、62 项按硬件/依赖跳过。性能数据不会因源码迁移自动视为重新测量。原始 60 次动作数组及统计保存在本次
部署实验工作目录，不将 checkpoint、录制图像或机器人配置提交仓库。

### 本地录制数据复现

`compare_recorded.py` 复用 checkpoint 自带预处理/后处理，不依赖机器人服务。
在独立、空闲的 GPU 上运行，使用匹配 checkpoint 的 LeRobot 环境：

```bash
PYTHONPATH=. python benchmarks/pi05-benchmark/compare_recorded.py \
  --checkpoint /path/to/pretrained_model \
  --tokenizer /path/to/paligemma-tokenizer \
  --samples /path/to/frame-000/sample.json /path/to/frame-018/sample.json \
            /path/to/frame-023/sample.json /path/to/frame-027/sample.json \
  --seeds 1000 0 2026 --warmup 1 \
  --out /path/to/new-run
```

每个 `sample.json` 包含 `state` 数组、`task` 字符串、`images` 对象；`images`
将 checkpoint 相机角色名映射到同目录图片文件，如 `{"camera1": "camera1.png"}`。
图片解码为 RGB CHW [0,1]。结果目录含运行条件、每次计时/动作误差和完整动作数组。
当前脚本将全部权重放在 CUDA，参考和候选共享同一加载模型；每个观测形状都预热。
这与历史部署实验的参考 CPU 词嵌入 offload、仅首帧预热不同，应以新生成条件和
结果为准。`--modes lerobot vvla_eager vvla_eager_graph` 可只测逐位一致的候选；
SDPA 结果始终单独报告误差。冷启动、CPU offload 和正式服务请求延迟不在该脚本
的推理计时范围内。
