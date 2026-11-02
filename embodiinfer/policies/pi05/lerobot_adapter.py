"""Use EmbodiInfer inference in an existing LeRobot preprocessing/control application."""

from __future__ import annotations

from typing import Any

import torch

from ...engine.config import EngineConfig
from ...engine.core import EngineCore
from ..config import VLAPolicyConfig
from .modeling_pi05 import Pi05Policy
from .processor_pi05 import Pi05Batch


class LeRobotPi05Adapter:
    """Keep the application's batch/action API while all model compute uses EmbodiInfer.

    The loaded policy supplies weights, image preprocessing and config only.
    Its predict_action_chunk/sample_actions/transformer forwards are not called.
    """

    def __init__(self, loaded_policy: Any, *, attention: str = "eager", cuda_graph: bool = True) -> None:
        self.config = loaded_policy.config
        self._reference = loaded_policy
        cfg = VLAPolicyConfig(
            name="pi0.5",
            action_dim=self.config.max_action_dim,
            action_horizon=self.config.chunk_size,
            default_num_steps=self.config.num_inference_steps,
        )
        policy = Pi05Policy(cfg, loaded_policy, attention=attention, native_embeddings=True)
        self.engine = EngineCore(
            policy,
            EngineConfig(
                device=self.config.device,
                dtype="auto",
                max_batch_size=1,
                batch_buckets=(1,),
                use_cuda_graph=cuda_graph,
                capture_full_loop=cuda_graph,
            ),
        )
        self.backend_name = f"embodiinfer-pi05-{attention}-{'cuda-graph' if cuda_graph else 'eager'}"
        self.last_timing: dict[str, float] = {}

    def reset(self) -> None:
        """Clear timing; every request already computes a fresh observation prefix."""
        self.last_timing = {}

    @torch.inference_mode()
    def predict_action_chunk(
        self, batch: dict[str, Any], *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Return a single normalized action chunk for the caller's postprocessor."""
        native_batch = Pi05Batch.from_lerobot_batch(self._reference, batch)
        if native_batch.batch_size != 1:
            raise ValueError("this deployment adapter expects one observation")
        native_batch.request_ids = ["request"]
        chunk = self.engine.execute(native_batch, generator=generator)[0]
        self.last_timing = dict(chunk.timing)
        dim = self.config.output_features["action"].shape[0]
        return chunk.actions[None, :, :dim].to(self.engine.device)
