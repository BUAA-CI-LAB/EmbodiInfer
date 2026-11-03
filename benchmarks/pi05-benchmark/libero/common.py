"""EmbodiInfer setup shared by the LIBERO tools: OpenPI checkpoint, optimized policy, static requests.

``pi05_libero.openpi_config.json`` is the training recipe of OpenPI's ``pi05_libero`` config:
two model cameras (OpenPI's third, right-wrist slot is zero-padded and masked for PI0.5, so it adds
no attention keys or positions), a stateless prompt, quantile normalization and absolute actions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

RECIPE = Path(__file__).resolve().parent / "pi05_libero.openpi_config.json"


def load_recipe(path: str | Path = RECIPE, tokenizer: str | None = None) -> dict:
    """Read an OpenPI recipe; ``tokenizer`` replaces its checkpoint-relative tokenizer path."""
    recipe = json.loads(Path(path).read_text())
    if tokenizer is not None:
        recipe["tokenizer"] = str(Path(tokenizer).expanduser().resolve())
    return recipe


def load_model(checkpoint: str, recipe: dict):
    """Load an OpenPI PyTorch checkpoint on the GPU with its processor."""
    from embodiinfer.policies.pi05.checkpoints import load_checkpoint
    from embodiinfer.policies.pi05.processor_pi05 import OpenPiProcessor

    model, config = load_checkpoint(checkpoint, load_device="cuda", checkpoint_config=recipe, low_cpu_mem_usage=True)
    return model, OpenPiProcessor(config)


def build_policy(model, processor, checkpoint: str, *, device: str, preset: str, calibration: str | None, steps: int):
    """One optimized policy; static requests of any step count or schedule share its weights."""
    from embodiinfer.engine.config import EngineConfig
    from embodiinfer.engine.core import EngineCore
    from embodiinfer.policies.config import VLAPolicyConfig
    from embodiinfer.policies.pi05 import Pi05OptimizationConfig
    from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy

    config = processor.config
    policy = Pi05Policy(
        VLAPolicyConfig(
            name="pi0.5", action_dim=config.action_dim, action_horizon=config.action_horizon, default_num_steps=steps
        ),
        model,
        native_embeddings=True,
        native_inference=True,
        prefix_cuda_graph=True,
        request_cuda_graph=True,
        compile_backend="none",
        prefix_attention="sdpa",
        denoise_attention="sdpa",
        optimizations=Pi05OptimizationConfig.from_preset(device, preset, calibration=calibration),
    )
    policy.checkpoint = checkpoint
    EngineCore(
        policy,
        EngineConfig(
            device="cuda", dtype="auto", max_batch_size=1, batch_buckets=(1,), use_cuda_graph=True,
            capture_full_loop=True,
        ),
    )
    return policy


class Request:
    """One prompt bound to a (steps, rho) schedule: uint8 camera frames in, physical actions out."""

    def __init__(self, policy, processor, prompt: str, steps: int, rho: float):
        config = processor.config
        if config.discrete_state_input:
            raise ValueError("A static request binds one prompt; discrete-state prompts change every step")
        tokens, mask = processor.tokenize(prompt, np.zeros(config.action_dim, np.float32))
        self.tokens = int(mask.sum())
        # The request captures the schedule of the rho set here.
        policy.flow_schedule_rho = float(rho)
        with torch.inference_mode():
            self.session = policy.static_request(
                tokens[0, : self.tokens].tolist(), cameras=len(config.image_keys), steps=steps, frame_size=(224, 224)
            )
        self.frames = self.session.frames.numpy()
        self.processor = processor

    def noise(self, generator: torch.Generator) -> torch.Tensor:
        return torch.randn(self.session.noise.shape, generator=generator)

    def run(self, noise: torch.Tensor) -> torch.Tensor:
        """Run on the frames already in ``self.frames``; returns the normalized action chunk."""
        return self.session.run(noise=noise)[0]

    def actions(self, raw: torch.Tensor, state) -> np.ndarray:
        state = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        return self.processor.restore_actions(raw, self.processor.prepare_state(state)).numpy()

    def close(self) -> None:
        self.session.close()
