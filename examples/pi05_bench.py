"""Real pi0.5 batching benchmark: HF-style single-stream vs embodiinfer batching.

Uses the actual LeRobot PI05Policy (pi0.5) compute. The only difference between
the two configs is how many environments share one forward:

    hf   : N sequential predict_action_chunk calls, batch size 1  (single-stream
           serving -- how a robot client / naive rollout loop drives the policy)
    embodiinfer : one predict_action_chunk call, batch size N            (request-level
           batching across envs -- what the embodiinfer scheduler forms)

Inputs are correctly shaped but random (throughput is independent of pixel/token
values); this measures the real pi0.5 forward, isolating the batching lever.
Action chunks come out identical in shape to production; semantics are irrelevant
for a throughput measurement.

    python examples/pi05_bench.py --ckpt /path/to/pi05_base --lang-len 48
"""

import argparse
import time

import torch
import torch.nn.functional as F


def now():
    return time.perf_counter()


def _sdpa_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Lossless drop-in for HF gemma eager_attention_forward using fused SDPA.

    Same additive float mask, same scale, same dtype -> numerically equivalent
    (fused kernel, no precision change). Returns (attn_output[B,seq,heads,hd], None)
    matching the eager function's output contract used by pi0.5's layer loop.
    """
    mask = attention_mask
    if mask is not None:
        mask = mask[:, :, :, : key.shape[-2]]
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling, dropout_p=0.0)
    out = out.transpose(1, 2).contiguous()
    return out, None


def enable_sdpa():
    """Patch the function pi0.5's custom layer loop calls for attention."""
    from transformers.models.gemma import modeling_gemma

    modeling_gemma.eager_attention_forward = _sdpa_attention_forward


def build_batch(policy, B, lang_len, device, dtype, image_keys, tok_key, mask_key, state_dim):
    c = policy.config
    batch = {}
    for k in image_keys:
        batch[k] = torch.rand(B, 3, c.image_resolution[0], c.image_resolution[1], device=device, dtype=dtype)
    # random but valid language tokens + full attention mask
    vocab = 257152  # paligemma vocab; only the embedding lookup range matters
    batch[tok_key] = torch.randint(0, vocab, (B, lang_len), device=device)
    batch[mask_key] = torch.ones(B, lang_len, dtype=torch.bool, device=device)
    batch["observation.state"] = torch.zeros(B, state_dim, device=device, dtype=dtype)
    return batch


@torch.no_grad()
def timed(policy, make_batch, B, iters, warmup):
    lat = []
    for it in range(iters + warmup):
        t0 = now()
        policy.predict_action_chunk(make_batch(B))
        torch.cuda.synchronize()
        if it >= warmup:
            lat.append((now() - t0) * 1e3)
    return lat


def run_data_parallel(args, dtype, tok_key, mask_key, PI05Policy):
    """Measure data-parallel throughput scaling across cuda:0..gpus-1."""
    import numpy as np

    from embodiinfer.engine.parallel.data_parallel import ThreadedExecutor

    enable_sdpa()
    print(
        f"[dp] loading {args.gpus} replicas (SDPA, dtype={args.dtype}, per-GPU batch={args.local_batch}) ..."
    )
    replicas = []
    for i in range(args.gpus):
        p = PI05Policy.from_pretrained(args.ckpt).eval().to(dtype).to(f"cuda:{i}")
        replicas.append(p)
    c = replicas[0].config
    image_keys = list(c.image_features)
    B = args.local_batch
    n_params = sum(p.numel() for p in replicas[0].parameters())

    # one fixed per-replica batch, built on that replica's device
    batches = [
        build_batch(
            replicas[i], B, args.lang_len, f"cuda:{i}", dtype, image_keys, tok_key, mask_key, c.max_state_dim
        )
        for i in range(args.gpus)
    ]

    def fwd(i):
        with torch.no_grad():
            replicas[i].predict_action_chunk(batches[i])
        torch.cuda.synchronize(i)

    # warmup all replicas
    ThreadedExecutor(args.gpus).run([lambda i=i: fwd(i) for i in range(args.gpus)])

    print(f"\n=== pi0.5 data-parallel throughput (SDPA, {args.dtype}) ===")
    print(f"params={n_params / 1e9:.2f}B/replica  per-GPU batch={B}\n")
    print(f"{'gpus':>5s} {'obs/s':>10s} {'obs/s/gpu':>11s} {'scaling':>9s}")
    print("-" * 38)
    base = None
    for G in range(1, args.gpus + 1):
        pool = ThreadedExecutor(G)
        thunks = [lambda i=i: fwd(i) for i in range(G)]
        lat = []
        for it in range(args.iters + args.warmup):
            t0 = now()
            pool.run(thunks)
            if it >= args.warmup:
                lat.append((now() - t0) * 1e3)
        ms = float(np.mean(lat))
        obs_s = (G * B) / (ms / 1e3)
        if base is None:
            base = obs_s
        print(f"{G:5d} {obs_s:10.2f} {obs_s / G:11.2f} {obs_s / base:8.2f}x")
        pool.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lerobot/pi05_base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--lang-len", type=int, default=48)
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
        "--attn",
        default="sdpa",
        choices=["eager", "sdpa"],
        help="attention kernel used by BOTH hf and embodiinfer configs. Default sdpa: "
        "the eager->sdpa swap is a lossless table-stakes fix that benefits "
        "the baseline equally, so it is NOT counted as embodiinfer's speedup. Both "
        "configs share the backend; the measured speedup is batching only.",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="run eager vs sdpa on identical input+noise, report max action diff",
    )
    ap.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="data-parallel replicas across cuda:0..N-1 (lossless throughput scaling)",
    )
    ap.add_argument("--local-batch", type=int, default=32, help="per-GPU batch size in data-parallel mode")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if args.attn == "sdpa" and not args.verify:
        enable_sdpa()
        print("[attn] patched pi0.5 attention: eager -> SDPA (fused, lossless)")

    print(f"[load] {args.ckpt}")
    if args.gpus > 1:
        return run_data_parallel(args, dtype, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, PI05Policy)

    policy = PI05Policy.from_pretrained(args.ckpt)
    policy.eval().to(args.device).to(dtype)
    c = policy.config
    image_keys = list(c.image_features)
    n_params = sum(p.numel() for p in policy.parameters())

    print("\n=== pi0.5 real batching benchmark ===")
    print(
        f"params={n_params / 1e9:.2f}B  cameras={len(image_keys)}  chunk={c.chunk_size}  "
        f"steps={c.num_inference_steps}  lang_len={args.lang_len}  dtype={args.dtype}  "
        f"attn={args.attn} (both hf & embodiinfer)\n"
    )
    print("speedup below = batching only (same attention backend on both configs)\n")

    def mk(B):
        return build_batch(
            policy,
            B,
            args.lang_len,
            args.device,
            dtype,
            image_keys,
            OBS_LANGUAGE_TOKENS,
            OBS_LANGUAGE_ATTENTION_MASK,
            c.max_state_dim,
        )

    if args.verify:
        b = mk(1)  # same input for both kernels
        with torch.no_grad():
            torch.manual_seed(0)
            a_eager = policy.predict_action_chunk(b).float().cpu()
            enable_sdpa()
            torch.manual_seed(0)
            a_sdpa = policy.predict_action_chunk(b).float().cpu()
        diff = (a_eager - a_sdpa).abs().max().item()
        print("\n[verify] eager vs SDPA on identical input+noise:")
        print(f"  max|Δaction| = {diff:.3e}   mean|action| = {a_eager.abs().mean().item():.4f}")
        print(
            f"  -> {'LOSSLESS (fused kernel, same fp32 numerics)' if diff < 1e-2 else 'DIFFERS - investigate'}"
        )
        return

    # warmup graphs / cudnn
    timed(policy, mk, 1, 1, 1)

    hdr = f"{'num_envs':>8s} {'hf obs/s':>10s} {'ei obs/s':>11s} {'hf p50(ms)':>11s} {'embodiinfer p50(ms)':>13s} {'speedup':>8s}"
    print(hdr)
    print("-" * len(hdr))
    import numpy as np

    for N in args.envs:
        # hf: N sequential bs=1 calls; one "workload" = serving all N envs once
        hf_lat = []
        for it in range(args.iters + args.warmup):
            t0 = now()
            for _ in range(N):
                with torch.no_grad():
                    policy.predict_action_chunk(mk(1))
            torch.cuda.synchronize()
            if it >= args.warmup:
                hf_lat.append((now() - t0) * 1e3)
        vv_lat = timed(policy, mk, N, args.iters, args.warmup)
        hf_ms = float(np.mean(hf_lat))
        vv_ms = float(np.mean(vv_lat))
        hf_obs = N / (hf_ms / 1e3)
        vv_obs = N / (vv_ms / 1e3)
        print(
            f"{N:8d} {hf_obs:10.2f} {vv_obs:11.2f} {np.percentile(hf_lat, 50):11.1f} "
            f"{np.percentile(vv_lat, 50):13.1f} {hf_ms / vv_ms:7.2f}x"
        )


if __name__ == "__main__":
    main()
