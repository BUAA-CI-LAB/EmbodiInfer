# 多 GPU 并行推理

数据并行（DP）在多张 GPU 上分别加载完整模型，将独立请求分发给各副本，侧重提高总吞吐。
张量并行（TP）把一个模型的计算分到多张 GPU 上，共同处理同一请求，需要权衡显存占用、计算和通信开销。

通过下面的 Python API 配置副本与多 GPU 执行。

## 配置数据并行

在各设备上创建完整模型副本，再传给 `DataParallelEngine`。无状态请求会分发到正常运行的副本。
有状态请求需提供 `SessionKey`：首次调用分配副本，后续调用固定使用该副本，
直到 `reset_sessions()` 清除状态并解除绑定。每个有状态会话以 B=1 执行，独立任务可跨设备并发。

会话状态保存在所属副本内，失败后不能迁移到其他副本重试。
`cancel_sessions()` 也会发往所属副本，但不会解除会话与副本的绑定。

```python
from embodiinfer import DataParallelEngine
from embodiinfer.types import SessionKey

dp = DataParallelEngine([core_on_cuda0, core_on_cuda1])
sessions = [SessionKey("env-0", "ep-0"), SessionKey("env-1", "ep-1")]
chunks = dp.execute([obs0, obs1], session_ids=sessions)
dp.reset_sessions(sessions)
```

这里的 `core_on_cuda0` 和 `core_on_cuda1` 是已经构造好的引擎核心，
各自拥有自己的模型，而 `obs0` 和 `obs1` 属于相互独立的 episode。
无状态 π0.5 的完整示例见
[`examples/pi05_inference.py`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/examples/pi05_inference.py)，
使用 `--gpus 2` 调用时，它会为每个 GPU 构造一个副本。

对于有状态策略，每个副本同一时间只使用一个会话，并将相互
独立的 episode 分发到多个副本上。异步/流水线执行、group 采样
以及 best-of-N 目前仅支持无状态策略。

## 双 GPU 性能对比

测试使用两张 NVIDIA A800 80GB PCIe GPU（`SYS` 拓扑，无 NVLink），
采用 BF16 eager、批次大小 1，预热两次后测量十个样本，不启用 CUDA Graph。
DP 吞吐统计两个完整模型副本的总处理量；TP 则由两张卡共同处理一个请求。

三个导航策略使用来自 R2R-VLNCE 对齐基准测试输入的
同一真实 RGB 观测。Pi0.5 使用确定性的三摄像头、
48-token 输入及其标准的 10 步
去噪循环。

| 策略 | 单 GPU samples/s | DP=2 samples/s | DP 加速比 | TP=2 samples/s | TP / 单 GPU |
|---|---:|---:|---:|---:|---:|
| Qwen R2R Low | 12.440 | 19.955 | 1.604x | 10.583 | 0.851x |
| Qwen R2R Panoramic | 4.639 | 8.407 | 1.812x | 3.849 | 0.830x |
| NaViDA | 1.614 | 2.248 | 1.393x | 1.304 | 0.808x |
| Pi0.5 | 3.969 | 6.817 | 1.717x | 3.370 | 0.849x |

所有导航 DP/TP 输出在全部十个样本上都与单 GPU 动作输出完全一致。
Pi0.5 DP 也完全一致；Pi0.5 TP 的两个 rank 输出完全一致，并且
在 `rtol=0.02`、`atol=0.05` 范围内与单 GPU 参考一致
（`max_abs=0.015625`）。在此 batch-one、无 NVLink 的设置下，DP 将聚合
吞吐提升 1.39-1.81 倍，而 TP 通信开销使 TP 比
单 GPU 慢 15-19%。
