"""Quickstart: single and batched action prediction on the synthetic flow VLA.

Runs anywhere (CPU or CUDA) with no checkpoint — ``MockFlowVLA`` mirrors the
pi0.5 / Cosmos compute pattern (encode a multimodal prefix once, integrate a
flow-matching field for N steps) so the engine path is exercised end-to-end.

    python examples/quickstart.py
"""

from __future__ import annotations

import torch

from embodiinfer import EngineConfig, Observation, EmbodiInfer, preset_config


def make_obs(cfg, env_id: int = 0) -> Observation:
    return Observation(
        images=torch.rand(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
        state=torch.rand(cfg.state_dim),
        instruction_tokens=torch.randint(0, cfg.vocab_size, (cfg.max_lang_len,)),
        instruction="pick up the cube",
        env_id=env_id,
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = preset_config("small")
    engine = EmbodiInfer(
        "mock_flow_vla",
        preset="small",
        engine_config=EngineConfig(device=device, use_cuda_graph=(device == "cuda")),
    )
    print(f"[quickstart] device={device}")

    # single observation
    chunk = engine.act(make_obs(cfg))
    print(f"single : action chunk {tuple(chunk.actions.shape)}  latency={chunk.latency_ms:.2f} ms")

    # 8 parallel environments coalesced into one forward pass
    chunks = engine.act([make_obs(cfg, i) for i in range(8)])
    print(
        f"batched: {len(chunks)} chunks, each {tuple(chunks[0].actions.shape)}, "
        f"batch latency={chunks[0].latency_ms:.2f} ms "
        f"({chunks[0].latency_ms / 8:.2f} ms/req amortized)"
    )


if __name__ == "__main__":
    main()
