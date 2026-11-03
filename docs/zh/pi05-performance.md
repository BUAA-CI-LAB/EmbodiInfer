# PI0.5：Jetson AGX Thor 与 RTX 4090 性能对比

在两台设备上，用同一个基准脚本、同一套设置，对比 EmbodiInfer 的 PI0.5 预设与 FlashRT、APXinf、
Embodied.cpp 以及 OpenPI 参考实现。

![EmbodiInfer 在 Jetson AGX Thor 与 RTX 4090 上相对 FlashRT、APXinf、Embodied.cpp 和 OpenPI 的 PI0.5 加速比](assets/pi05/speedup.svg)

| 设备 | 精度档 | EmbodiInfer | 加速比 |
|---|---|---:|---|
| Jetson AGX Thor | FP4 | 20.04 ms · 49.9 Hz | 比 FlashRT 快 1.05× |
| Jetson AGX Thor | FP8 | 33.60 ms · 29.8 Hz | 比 FlashRT 快 1.21×，比 APXinf 快 1.22× |
| Jetson AGX Thor | 16 位 | 61.63 ms · 16.2 Hz | 比 APXinf 快 1.09×，比 FlashRT 快 1.12×，比 Embodied.cpp 快 3.63×，比 OpenPI 快 4.24× |
| RTX 4090 | 8 位 | 17.06 ms · 58.6 Hz | 比 FlashRT 快 1.29×，比 APXinf 快 2.38× |
| RTX 4090 | 16 位 | 30.08 ms · 33.2 Hz | 比 FlashRT 快 1.06×，比 APXinf 快 1.07×，比 Embodied.cpp 快 1.91× |

在 RTX 4090 上使用 OpenPI 官方权重和官方 LIBERO 评测时，`fp8-fp8` 请求在 ρ = 3 的调度下只做 2 步去噪，
延迟降到 13.34 ms（75.0 Hz），LIBERO-spatial 成功率 99.0%，OpenPI 为 98.8%；详见下文
「RTX 4090 上的 LIBERO：OpenPI 官方权重与官方评测」一节。

## 测试方法

所有引擎都运行 FlashRT 自己的 PI0.5 基准脚本，即 FlashRT v0.2.0 的 `examples/thor/pi05.py`。
每个引擎只替换模型加载和每次调用的 `run()`；输入、prompt、预热、固定种子的动作采样、计时循环和
报告格式都沿用 FlashRT 的实现。

| 设置 | 取值 |
|---|---|
| 权重 | `RLinf-Pi05-LIBERO-SFT`，所有引擎使用同一个文件；动作按其分位数 `norm_stats.json` 反归一化 |
| 输入 | FlashRT `pi05_libero.npz` 中的 8 个 LIBERO 观测，依次轮流使用：主相机与腕部相机，224×224 uint8 |
| Prompt | "pick up the black bowl and place it on the plate"，13 个 PaliGemma token |
| 模型 | batch 1、2 路相机、10 步 flow matching、动作长度 10 |
| 计时 | 先预热 200 次，再计时 1,000 次（`synchronize → run → synchronize`），每轮取 P50 |
| 轮次 | 同一设备上的各引擎交替运行 3 轮；表中给出每轮的 P50 及其平均值 |

每次计时调用都包括图像上传、图像归一化、噪声生成、整次推理、动作拷回主机和反归一化。权重加载、
校准、prompt 绑定和图捕获在计时之前完成。8 次固定种子调用对所有引擎都使用 FlashRT 的 float16
NumPy 噪声，因此各引擎的动作可以直接比较。

## Jetson AGX Thor

Jetson AGX Thor（SM110，20 个 SM，128 GB LPDDR5X），L4T R39.2.1、CUDA 13.2、PyTorch 2.13.0、
cuDNN 9.20；MAXN 模式，GPU、内存和 CPU 频率分别锁定在 1575 MHz、4266 MHz 和 2.6 GHz。每次运行前
都会确认 GPU 空闲且频率已锁定。

| 精度档 | 引擎 · 配置 | 第 1 轮 | 第 2 轮 | 第 3 轮 | P50 平均（ms） | P95 最大（ms） | 频率 | 动作与 OpenPI 的余弦相似度（平均 / 最差） |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FP4 | **EmbodiInfer · `nvfp4-nvfp4`** | 19.938 | 20.157 | 20.033 | **20.04** | 20.34 | **49.9 Hz** | 0.997182 / 0.995833 |
| FP4 | FlashRT · `fp4`（NVFP4 + FA4） | 21.070 | 21.122 | 21.031 | 21.07 | 21.20 | 47.5 Hz | 0.998025 / 0.996257 |
| FP8 | **EmbodiInfer · `fp8-fp8`** | 33.386 | 33.529 | 33.894 | **33.60** | 34.25 | **29.8 Hz** | 0.999785 / 0.999716 |
| FP8 | FlashRT · `fp8` | 40.687 | 40.781 | 40.630 | 40.70 | 40.92 | 24.6 Hz | 0.999773 / 0.999602 |
| FP8 | APXinf · `fp8_static` | 40.875 | 40.980 | 40.796 | 40.88 | 41.06 | 24.5 Hz | 0.997992 / 0.997391 |
| 16 位 | **EmbodiInfer · `bf16`** | 61.360 | 61.867 | 61.655 | **61.63** | 62.06 | **16.2 Hz** | 0.999995 / 0.999993 |
| 16 位 | APXinf · `bf16` | 66.997 | 67.188 | 67.179 | 67.12 | 67.39 | 14.9 Hz | 0.999982 / 0.999977 |
| 16 位 | FlashRT · `fp16` | 69.220 | 69.244 | 69.244 | 69.24 | 69.46 | 14.4 Hz | 0.999891 / 0.999791 |
| 16 位 | Embodied.cpp · BF16 GGUF | 223.895 | 224.080 | 223.939 | 223.97 | 224.50 | 4.5 Hz | 0.999992 / 0.999981 |
| 16 位 | OpenPI · PyTorch BF16 | 258.443 | 265.667 | 259.420 | 261.18 | 271.41 | 3.8 Hz | 参考 |

余弦相似度比较的是第 1 轮 8 次固定种子调用的动作与 OpenPI 的动作。所有引擎在 3 轮中返回的动作
逐位一致。两个 FP4 引擎在另一次交替运行中测得，同样 3 轮、同一套设置。

## RTX 4090

GeForce RTX 4090（SM89，128 个 SM，24 GB），CUDA 12.9、PyTorch 2.13.0、cuDNN 9.20，默认频率
（测试期间 SM 约 2.79 GHz）。SM89 没有 FP4 Tensor Core，因此没有 FP4 档。

| 精度档 | 引擎 · 配置 | 第 1 轮 | 第 2 轮 | 第 3 轮 | P50 平均（ms） | P95 最大（ms） | 频率 |
|---|---|---:|---:|---:|---:|---:|---:|
| 8 位 | **EmbodiInfer · `fp8-fp8`** | 17.055 | 17.066 | 17.063 | **17.06** | 17.09 | **58.6 Hz** |
| 8 位 | FlashRT · `fp8` | 22.012 | 22.061 | 22.102 | 22.06 | 22.15 | 45.3 Hz |
| 8 位 | APXinf · `int8_dynamic` | 40.612 | 40.615 | 40.643 | 40.62 | 40.85 | 24.6 Hz |
| 16 位 | **EmbodiInfer · `bf16`** | 30.060 | 30.079 | 30.095 | **30.08** | 30.53 | **33.2 Hz** |
| 16 位 | FlashRT · `fp16` | 31.913 | 31.937 | 31.946 | 31.93 | 32.25 | 31.3 Hz |
| 16 位 | FlashRT · `bf16` | 32.881 | 33.314 | 32.933 | 33.04 | 33.42 | 30.3 Hz |
| 16 位 | APXinf · `bf16` | 32.033 | 32.051 | 32.085 | 32.06 | 32.30 | 31.2 Hz |
| 16 位 | Embodied.cpp · BF16 GGUF | 57.452 | 57.479 | 57.500 | 57.48 | 57.68 | 17.4 Hz |

在 RLinf LIBERO 验证集（10 个观测 × 5 个噪声种子）上，4090 的 `bf16` 与 `fp8-fp8` 预设和 strict
BF16 预设的动作余弦相似度分别为 0.999997 和 0.99996。

## 各引擎配置

| 引擎 | 版本 | 配置 |
|---|---|---|
| EmbodiInfer | 本仓库 | `Pi05Policy.static_request`，使用表中列出的 `thor` 或 `4090` 预设；从 uint8 图像到动作是一张 CUDA 图 |
| FlashRT | v0.2.0（`f4ddc90`） | Thor：`fp4` 为其 `docs/thor/pi05.md` 中的 NVFP4 + FlashAttention-4 配置，`fp8` 为默认配置，`fp16` 设置 `use_fp16=True, use_fp8=False`。RTX 4090：`rtx_sm89` 路线，`fp8`、`bf16`（`use_fp8=False`）与 `fp16`（`use_fp16=True`）。均在同样 8 个观测上校准 |
| APXinf | `4de8f5c` | `ModelRunner.infer_rgb`，prompt 只分词一次，并按其基准命令开启 `autotune`（其 tactic 表未覆盖这个 prompt 长度）。Thor 的 `fp8_static` 由其 `calibrate_pi05.py` 在同样 8 个观测上校准；静态 FP8 需要 SM100 及以上，因此 4090 使用其 `int8_dynamic` 模式 |
| Embodied.cpp | `1dad33f2`，llama.cpp `b9016` | 由同一份 safetensors 转出的 BF16 GGUF（动作长度 10），ggml CUDA 后端，进程内调用；其原始动作经过同样的反归一化 |
| OpenPI | `pi05_libero` | FlashRT 脚本自带的参考模式：OpenPI 自己的策略，eager PyTorch。仅在 Thor 上测量 |

APXinf 在加载时为这个 prompt 长度挑选 GEMM kernel。在这个形状下，有一次全新调优为 FP8 GateUp 选中了
一个会输出错误动作的 kernel，因此表中 APXinf 的结果固定使用一份动作与参考一致的调优结果。

## Thor 上的 LIBERO 闭环

LIBERO-spatial 任务 0（"pick up the black bowl between the plate and the ramekin and place it on
the plate"），每次从同一初始状态出发，每个控制步都重新决策，每个引擎的 flow-matching 采样保持
固定。仿真端用 OpenPI 的 Pillow `resize_with_pad` 预处理一次图像，三个引擎收到完全相同的输入。
视频以 2 倍速播放，画面中的计时为真实时间；每一步持续的时间等于该步的策略延迟加上 50 ms 控制周期。

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="../assets/pi05/libero-thor-comparison.mp4" type="video/mp4">
</video>

| 引擎 | 决策次数 | 策略延迟 P50（ms） | 频率 | 任务 | 完成用时 |
|---|---:|---:|---:|---|---:|
| OpenPI · PyTorch BF16 | 74 | 265.52 | 3.8 Hz | 完成 | 23.68 s |
| FlashRT · NVFP4 | 71 | 22.79 | 43.9 Hz | 完成 | 5.17 s |
| **EmbodiInfer · `nvfp4-nvfp4`** | 68 | 20.00 | 50.0 Hz | 完成 | 4.76 s |

策略延迟指推理服务的一次模型调用，从共享内存中的图像到反归一化后的动作。完成用时是每一步的
策略延迟加 50 ms 控制周期的累计。

### Thor 上的 NVFP4 校准

内置的 `RLinf-Pi05-LIBERO-SFT` `nvfp4-nvfp4` 校准让两个塔的所有 MLP 投影都保持 NVFP4，相比直接对
checkpoint 权重做就近舍入打包，改动有两处：

- **gate/up 的 [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html)。** 每个 MLP 的 gate/up 权重按 W·s 打包，
  s<sub>j</sub> = (a<sub>j</sub> / mean(a))<sup>0.5</sup> 并截断到 [0.25, 4]，a<sub>j</sub> 是输入通道 j
  在 40 帧 LIBERO-spatial 录制画面（任务 0–3、初始状态 10）上的取值范围。前一层 norm 的输出相应除以
  s：prefix 通过 RMSNorm 的 scale，action expert 通过 AdaRMS 投影。只改变打包后的权重，kernel 不变。
- **BF16 离群通道。** prefix 第 16 层 down 投影输入中取值范围最大的 128 个通道（最大 14.7k，均值 28）
  在 NVFP4 主路径旁用 BF16 计算，主路径其余通道的范围随之从 14.1k 降到 213。这些通道在该层输出中
  的范数约为其余通道的 3 倍，而该层输出决定 prefix 最后一层的 K/V。旁路额外耗时约 0.04 ms。

同一任务的初始状态 0–29 各跑一局、输入画面相同，与就近舍入打包和 FlashRT 的对比：

| 引擎 · 配置 | 成功 | 2 秒后碗仍在盘上 | 碗与盘中心距离 |
|---|---:|---:|---:|
| EmbodiInfer · `nvfp4-nvfp4`，就近舍入打包 | 18/30 | 10/30 | 2.45 cm |
| **EmbodiInfer · `nvfp4-nvfp4`** | **25/30** | **21/30** | **1.85 cm** |
| FlashRT · NVFP4 | 26/30 | 24/30 | 1.91 cm |

成功按 LIBERO 自带判定，上限 220 次策略调用；成功后再执行 40 个控制步检查碗是否还在盘上；距离取
LIBERO 首次判定成功时碗心与盘心的水平距离，对成功的局取平均。30 局能分辨的差别约为 2 次成功，因此
这里 EmbodiInfer 与 FlashRT 没有可区分的差异。

## RTX 4090 上的 LIBERO：OpenPI 官方权重与官方评测

这一节使用 OpenPI 官方的 `pi05_libero` 权重（`gs://openpi-assets/checkpoints/pi05_libero`，用 OpenPI 的
`examples/convert_jax_model_to_pytorch.py` 转为 PyTorch），评测直接运行 OpenPI 的
`examples/libero/main.py`，不做任何修改：LIBERO-spatial 共 10 个任务 × 50 个初始状态，seed 7，先等待
10 步让物体落稳，之后每次推理执行 10 步动作块中的前 5 步，控制频率 20 Hz。OpenPI 用它自己的 PyTorch
策略服务，EmbodiInfer 用
[`libero/serve.py`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/libero/serve.py)
走同一套 websocket 协议；两者每次调用都重新采样 flow-matching 噪声。

| 引擎 · 配置 | 去噪步数 | 各任务成功次数 | 成功率 |
|---|---|---|---:|
| OpenPI · PyTorch BF16 | 10 | 50 50 50 50 47 48 50 50 49 50 | 494/500 · 98.8% |
| **EmbodiInfer · `fp8-fp8`** | **2，ρ = 3** | 50 50 50 50 47 48 50 50 50 50 | **495/500 · 99.0%** |
| EmbodiInfer · `fp8-fp8` | 3，ρ = 1 | 50 50 50 50 48 47 50 50 50 50 | 495/500 · 99.0% |
| EmbodiInfer · `fp8-fp8` | 2，ρ = 1 | 50 50 50 49 47 46 50 50 50 49 | 491/500 · 98.2% |
| EmbodiInfer · `fp8-fp8` | 3，ρ = 2 | 50 50 50 49 47 46 50 50 50 48 | 490/500 · 98.0% |
| EmbodiInfer · `fp8-fp8` | 4，ρ = 1 | 50 50 50 50 47 44 48 50 50 50 | 489/500 · 97.8% |

少步配置直接沿用训练好的权重，不重新训练。`Pi05Policy.flow_schedule_rho` 把 N 个 Euler 步放在幂次网格
t<sub>i</sub> = (1 − i/N)<sup>ρ</sup> 上：ρ = 1 就是 OpenPI 的均匀网格；ρ > 1 时靠近噪声端（t = 1）步长大、
靠近数据端（t = 0）步长小。2 步、ρ = 3 时，速度场在 t = 1 和 t = 0.125 处求值。成功率在 98% 附近时，
500 次评测的标准误约为 0.6 个百分点，因此这几个配置彼此之间、以及与 OpenPI 相比都没有可区分的差异；
它们与 OpenPI 10 步的成功率都相差不到 1 个百分点。

| EmbodiInfer `fp8-fp8`，RTX 4090 | 延迟 P50 | 延迟 P95 | 频率 | 加速比 |
|---|---:|---:|---:|---:|
| 10 步 | 17.08 ms | 17.19 ms | 58.5 Hz | 1.00× |
| 4 步，ρ = 1 | 14.21 ms | 14.27 ms | 70.4 Hz | 1.20× |
| 3 步，ρ = 1 | 13.79 ms | 13.84 ms | 72.5 Hz | 1.24× |
| 3 步，ρ = 2 | 13.80 ms | 13.86 ms | 72.5 Hz | 1.24× |
| 2 步，ρ = 1 | 13.33 ms | 13.37 ms | 75.0 Hz | 1.28× |
| **2 步，ρ = 3** | **13.34 ms** | **13.40 ms** | **75.0 Hz** | **1.28×** |

[`libero/latency.py`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/libero/latency.py)
在空闲 GPU 上预热 50 次后计时 1,000 次，prompt 为 LIBERO 任务 0 的 20 个 token，输入为录制的 LIBERO 画面；
每次调用与推理服务一样，包括拷贝画面、采样噪声、运行 static request 和反归一化动作。ρ 不影响延迟。

仓库自带的 `fp8-fp8` 标定属于 `RLinf-Pi05-LIBERO-SFT`，并绑定了该文件的 SHA256，因此这份权重需要自己的
标定：[`libero/calibrate.py`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/pi05-benchmark/libero/calibrate.py)
在 80 个 LIBERO 观测上记录每个 MLP 层的输入范围并留 10% 余量；这 80 个观测是 LIBERO-spatial、-object、
-goal、-10 共 40 个任务各 2 个初始状态落稳后的第一帧。prefix 第 16、17 层的 down projection 输入最大分别
达到 13k 和 20k（RLinf 权重中为 1.2k 和 9.7k），保留 BF16；两个塔的其余部分全部使用 FP8。

下面的视频是同一个 episode 在各引擎上的表现，均为 10 步去噪：任务 6（"pick up the black bowl next to the
cookie box and place it on the plate"），初始状态 2；每个引擎单独占用 GPU 录制，按真实时间播放：每个控制步
持续 50 ms，请求新动作块的那一步再加上该次推理的耗时。在 16 个候选 episode（任务 1、3、5、6，初始状态
0–3）中，各引擎都一次抓取成功，这一个 episode 各引擎的成功步数最接近（96–97 步），所以各引擎的差别只在
速度，不在动作本身。FlashRT 使用 `rtx_sm89` FP8 路线，经 `predict()` 调用，在每个 prompt 的第一个观测上标定。

<video controls muted loop playsinline preload="metadata" width="100%">
  <source src="../assets/pi05/libero-4090-comparison.mp4" type="video/mp4">
</video>

| 引擎 · 配置 | 推理延迟 P50 | 频率 | 成功所用控制步数 | 完成用时 |
|---|---:|---:|---:|---:|
| OpenPI · PyTorch BF16 | 202.0 ms | 4.9 Hz | 97 | 8.89 s |
| FlashRT · FP8 | 23.6 ms | 42.3 Hz | 97 | 5.32 s |
| **EmbodiInfer · `fp8-fp8`** | **18.6 ms** | **53.8 Hz** | 96 | **5.17 s** |

这里的推理延迟是闭环中推理服务的一次模型调用，此时仿真也在同一张 GPU 上渲染，比空闲 GPU 上的数字高约
1 ms。完成用时是每一步 50 ms 控制周期与请求动作块那些步的推理耗时之和。
