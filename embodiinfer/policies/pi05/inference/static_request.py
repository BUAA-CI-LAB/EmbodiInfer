"""Fixed-layout PI0.5 requests served by one CUDA graph, from uint8 camera frames to actions."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from ....backend.cuda.camera_frames import FrameResize, normalize_frames
from ....backend.torch.request_graph import CapturedRequest
from ..processor_pi05 import Pi05Batch

if TYPE_CHECKING:
    from ..modeling_pi05 import Pi05Policy


class Pi05StaticRequest:
    """Serve B1 observations for one bound prompt and camera layout with a single graph launch.

    ``frames`` (uint8 [cameras, frame height, frame width, 3]) and ``noise`` are pinned host
    buffers that the graph uploads itself. It resizes frames of another size than the model's
    exactly like OpenPI's Pillow ``resize_with_pad``, normalizes them on the device, runs the
    whole request and writes model-space actions to the pinned ``actions`` buffer, so the host
    only launches the graph and waits for it. ``run`` returns after the graph finished, so the
    staging buffers may be refilled immediately. Refit, device moves and ``train()`` close the
    request.
    """

    def __init__(
        self,
        policy: Pi05Policy,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None,
        *,
        cameras: int,
        height: int,
        width: int,
        steps: int,
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        frame_height, frame_width = frame_size or (height, width)
        if min(cameras, height, width, steps, frame_height, frame_width) < 1:
            raise ValueError("Pi05 static requests need positive cameras, image and frame sizes and steps")
        runtime = policy._runtime
        device = policy._m.action_in_proj.weight.device
        dtype = policy.execution_dtype
        shape = policy.flow_state_shape(1)
        tokens = torch.as_tensor(tokens).reshape(1, -1).to(device=device, dtype=torch.long)
        mask = (
            torch.ones_like(tokens, dtype=torch.bool)
            if token_mask is None
            else torch.as_tensor(token_mask).reshape(tokens.shape).to(device=device, dtype=torch.bool)
        )
        self.frames = torch.zeros((cameras, frame_height, frame_width, 3), dtype=torch.uint8).pin_memory()
        self.noise = torch.zeros(shape, dtype=dtype).pin_memory()
        self.actions = torch.zeros(shape, dtype=dtype).pin_memory()
        self._frames = torch.empty_like(self.frames, device=device)
        self._images = torch.empty((cameras, 3, height, width), dtype=torch.float32, device=device)
        self._resize = None
        if (frame_height, frame_width) != (height, width):
            self._resize = FrameResize((frame_height, frame_width), (height, width), device)
            self._rows = self._resize.rows_buffer(cameras, device)
        # The engine hands images to the model in the execution dtype; keep that contract.
        self._model_images = (
            self._images if dtype == torch.float32 else torch.empty_like(self._images, dtype=dtype)
        )
        self._state = torch.empty(shape, dtype=dtype, device=device)
        self._batch, all_valid = runtime._prepare_batch(
            Pi05Batch(
                [self._model_images[i : i + 1] for i in range(cameras)],
                [torch.ones(1, dtype=torch.bool, device=device) for _ in range(cameras)],
                tokens,
                mask,
            )
        )

        def forward() -> torch.Tensor:
            # Capture keeps addresses, not Python owners of cached graph inputs.
            self._frames.copy_(self.frames, non_blocking=True)
            if self._resize is None:
                normalize_frames(self._frames, self._images)
            else:
                self._resize(self._frames, self._rows, self._images)
            if self._model_images is not self._images:
                self._model_images.copy_(self._images)
            self._state.copy_(self.noise, non_blocking=True)
            self._schedule = runtime.schedule(self._state, steps)
            self._prefix = policy._encode_prefix_impl(self._batch, all_valid=all_valid)
            actions = runtime.integrate(self._state, self._prefix, steps)
            self._suffix_attention = getattr(policy, "_cached_suffix_att", None)
            self.actions.copy_(policy.finalize_actions(actions, self._prefix), non_blocking=True)
            return self.actions

        operators = policy._get_optimizations()
        scope = object()
        self._captured = CapturedRequest(
            forward,
            device,
            execution_context=(
                nullcontext if operators is None else lambda: operators.execution_scope(scope)
            ),
            release=None if operators is None else lambda: operators.release_scope(scope),
        )
        self._lock = threading.Lock()
        runtime.static_requests.add(self)

    def run(
        self,
        frames: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return model-space actions [1, horizon, action_dim] on the host.

        Given ``frames`` or ``noise`` are copied into the staging buffers. Without ``frames`` the
        caller has already written ``self.frames``; without ``noise`` it is drawn on the host
        from ``generator`` (or the default CPU generator).
        """
        with self._lock, torch.inference_mode():
            if frames is not None:
                frames = torch.as_tensor(frames)
                if frames.shape != self.frames.shape:
                    raise ValueError(f"frames must have shape {tuple(self.frames.shape)}")
                self.frames.copy_(frames)
            if noise is None:
                torch.randn(self.noise.shape, generator=generator, dtype=self.noise.dtype, out=self.noise)
            else:
                noise = torch.as_tensor(noise)
                if noise.shape != self.noise.shape:
                    raise ValueError(f"noise must have shape {tuple(self.noise.shape)}")
                self.noise.copy_(noise)
            self._captured.replay()
            self._captured.stream.synchronize()
            return self.actions.clone()

    def close(self) -> None:
        """Release the graph and its operator workspaces after pending work; later runs raise."""
        self._captured.close()

    def __enter__(self) -> Pi05StaticRequest:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
