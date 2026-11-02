"""GRPO closed loop on a mock flow VLA, with a rollout-throughput panel.

Two things this demonstrates:

  1. A working GRPO loop with embodiinfer as the rollout backend — group sampling,
     group-relative advantage, a differentiable flow log-prob recompute, a
     clipped policy-gradient step, and weight sync. The toy reach reward climbs,
     confirming the loop learns.

  2. Why the generation path is worth optimizing: inside the loop, action
     generation is a large share of wall-clock (and the whole share, once the
     learner runs in a separate process). We time embodiinfer's batched group sampling
     (one prefix encode broadcast across all candidates + envs, one denoise
     pass) against an HF-style per-env baseline (one call per env) and report the
     speedup. Cross-env batching amortizes the compute-bound backbone across
     envs, and the DataParallelEngine scales across replicas; the exact gain
     depends on the model and hardware.

    python examples/grpo_demo.py

Runs on CPU (small + slow) or a single GPU; larger GPU-scale numbers come from
running it on GPU hardware. This is the runnable-anywhere version.
"""

from __future__ import annotations

import time

import torch

from embodiinfer import EngineConfig, ToyReachEnv, EmbodiInfer, preset_config
from embodiinfer.engine.rollout.demo import GRPOConfig, GRPOTrainer, ToyReachReward
from embodiinfer.engine.rollout.logprob import flow_sample_with_logprob


def _time_call(fn, *, warmup: int = 1, iters: int = 3) -> float:
    """Return mean wall-clock seconds for ``fn`` (with CUDA sync + warmup)."""
    for _ in range(warmup):
        fn()
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def _hf_style_group(backend, observations, group_size: int, num_steps: int, sigma: float) -> None:
    """HF-style baseline: one call per env (no cross-env batching)."""
    policy = backend.policy
    for o in observations:
        batch = policy.collate([o], ["hf"]).to(backend.device, backend.dtype)
        prefix = policy.encode_prefix(batch).expand(group_size)
        x0 = policy.new_noise(group_size)
        flow_sample_with_logprob(policy, prefix, x0, num_steps, sigma)


def train(trainer: GRPOTrainer, env: ToyReachEnv, iterations: int) -> None:
    print(f"[grpo] training {iterations} iterations  "
          f"(envs={env.num_envs}, group={trainer.cfg.group_size}, "
          f"inner_epochs={trainer.cfg.num_inner_epochs})")
    obs = env.reset()  # fixed observations, so the reward trend isolates learning
    ema = None
    first = 0.0
    for it in range(iterations):
        stats = trainer.step(obs)
        ema = stats["reward_mean"] if ema is None else 0.9 * ema + 0.1 * stats["reward_mean"]
        if it == 0:
            first = ema
        if it == 0 or (it + 1) % max(1, iterations // 10) == 0:
            print(
                f"[grpo] it={it + 1:3d}  reward_ema={ema:.4f}  "
                f"reward_max={stats['reward_max']:.4f}  ratio={stats['ratio_mean']:.3f}  "
                f"grad_norm={stats.get('grad_norm', 0.0):.1f}"
            )
    print(f"[grpo] reward_ema {first:.4f} -> {ema:.4f}  (Δ={ema - first:+.4f}; reward in (0,1], 1 at the goal)")


def throughput_panel(trainer: GRPOTrainer, env: ToyReachEnv, num_steps: int) -> None:
    backend = trainer.backend
    G = trainer.cfg.group_size
    sigma = trainer.cfg.sigma
    obs = env.reset()
    print(f"\n[throughput]  batch = {len(obs)} envs x {G} candidates = {len(obs) * G}")

    t_embodiinfer = _time_call(lambda: backend.sample_group(obs, G, sigma=sigma, num_steps=num_steps))
    t_hf = _time_call(lambda: _hf_style_group(backend, obs, G, num_steps, sigma))

    # rollout vs update share within one GRPO step
    t_rollout = _time_call(lambda: backend.sample_group(obs, G, sigma=sigma, num_steps=num_steps))
    t_step = _time_call(lambda: trainer.step(obs))
    gen_frac = t_rollout / max(t_step, 1e-9)

    print(f"  embodiinfer (cross-env batched) : {t_embodiinfer * 1e3:8.1f} ms")
    print(f"  HF-style (per-env loop)  : {t_hf * 1e3:8.1f} ms")
    print(f"  speedup                  : {t_hf / max(t_embodiinfer, 1e-9):8.2f}x")
    print(f"  rollout share of a step  : {gen_frac * 100:8.1f} %")
    ndev = torch.cuda.device_count()
    if ndev > 1:
        print(f"  (+ DataParallelEngine scales rollout ~linearly across {ndev} GPUs)")
    else:
        print("  (+ DataParallelEngine adds ~linear scaling across multiple GPUs)")


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        preset, overrides = "tiny", dict(image_size=32, patch_size=16)  # cheap CPU forward
        train_envs, throughput_envs, num_steps, iterations = 8, 32, 6, 200
    else:
        # On GPU, real 224px vision so the throughput panel is meaningful.
        preset, overrides = "small", {}
        train_envs, throughput_envs, num_steps, iterations = 8, 64, 10, 200

    cfg = preset_config(preset, **overrides)
    engine = EmbodiInfer(
        "mock_flow_vla",
        preset=preset,
        engine_config=EngineConfig(device=device, use_cuda_graph=False, max_batch_size=2048),
        **overrides,
    )
    # A shared fixed goal at the origin keeps the task a small learnable set (every
    # position is one-step reachable), so the reward climbs within a short CPU run;
    # real rollout randomizes goals per env.
    reward = ToyReachReward(step_scale=1.0, width=1.0)
    train_env = ToyReachEnv(num_envs=train_envs, cfg=cfg, seed=0, fixed_goal=(0.0, 0.0))
    trainer = GRPOTrainer(
        engine.backend,
        reward,
        GRPOConfig(group_size=16, lr=1e-3, sigma=0.4, num_steps=num_steps, num_inner_epochs=1),
    )

    print(f"[grpo-demo] device={device}  preset={cfg.name}  action_dim={cfg.action_dim}")
    train(trainer, train_env, iterations=iterations)

    # Throughput on a larger batch (random goals; timing doesn't need learning).
    throughput_env = ToyReachEnv(num_envs=throughput_envs, cfg=cfg, seed=1)
    throughput_panel(trainer, throughput_env, num_steps)


if __name__ == "__main__":
    main()
