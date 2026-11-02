"""Serialized ActiveVLN serving using the engine's recurrent transactions."""

from __future__ import annotations

import io
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from ...engine.serve.contracts import ModelAction, ModelResult, RawPolicyRequest
from ...types import Observation, SessionKey
from .execution_activevln import EXECUTION_PREFIX_SCHEMA, ExecutionPrefix

if TYPE_CHECKING:
    from ...engine.core import EngineCore


class ActiveVLNServingAdapter:
    """Serve B=1 R2R turns with session-local history and execution acknowledgements."""

    action_space = "activevln.r2r.discrete.v1"
    image_field = "observation.images.rgb"

    def __init__(
        self,
        *,
        core: EngineCore,
        checkpoint: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        del checkpoint
        if config:
            raise ValueError(f"unknown ActiveVLN serving options: {sorted(config)}")
        if not core.policy.is_recurrent or core.policy.action_space != "r2r":
            raise ValueError("ActiveVLN HTTP serving requires a recurrent R2R policy")
        self._core = core
        self._lock = threading.RLock()
        self._committed: dict[str, tuple[int, str]] = {}

    def capabilities(self) -> dict[str, Any]:
        """Advertise eager B=1 inference, without claiming benchmark optimizations."""
        return {
            "model": "activevln",
            "action_space": self.action_space,
            "image_fields": [self.image_field],
            "recurrent": True,
            "max_batch_size": 1,
            "execution_mode": "serialized_b1",
            "execution_prefix": EXECUTION_PREFIX_SCHEMA,
        }

    def infer(self, request: RawPolicyRequest) -> ModelResult:
        """Decode one RGB frame and commit model memory only through EngineCore."""
        if len(request.images) != 1 or request.images[0].name != self.image_field:
            raise ValueError(f"expected exactly one {self.image_field} image")
        image = request.images[0]
        if image.mime_type not in {"image/jpeg", "image/png"}:
            raise ValueError("ActiveVLN requires JPEG or PNG")
        try:
            with Image.open(io.BytesIO(image.data)) as opened:
                if opened.width * opened.height > 8_000_000:
                    raise ValueError("ActiveVLN image exceeds eight million pixels")
                rgb = np.array(opened.convert("RGB"))
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("invalid ActiveVLN image") from error
        metadata = {**request.metadata, "step_id": request.step_id}
        prefix = None
        if "activevln_execution" in metadata:
            prefix = ExecutionPrefix.parse(metadata["activevln_execution"], step_id=request.step_id)
            metadata["activevln_execution"] = prefix.to_wire()
        observation = Observation(
            images=torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255),
            state=torch.empty(0),
            instruction_tokens=torch.empty(0, dtype=torch.long),
            instruction=request.instruction,
            metadata=metadata,
        )
        with self._lock:
            previous = self._committed.get(request.session_id)
            expected_step = 0 if previous is None else previous[0] + 1
            if request.step_id != expected_step:
                raise ValueError("ActiveVLN steps must be consecutive within a session")
            if previous is not None and request.instruction != previous[1]:
                raise ValueError("reset the ActiveVLN session before changing its instruction")
            if prefix is not None and previous is None:
                raise ValueError("execution prefix requires committed session history")
            batch = self._core.policy.collate([observation], [request.request_id])
            chunks = self._core.execute(batch, session_ids=[self._session_key(request.session_id)])
            chunk = chunks[0]
            trace = chunk.trace
            values: dict[str, Any] = {
                "rows": chunk.actions.tolist(),
                "parsed_valid": bool(trace is not None and getattr(trace.parsed_actions, "valid", False)),
                "text": trace.text if trace is not None else "",
                "stop_reason": trace.stop_reason if trace is not None else None,
            }
            if prefix is not None:
                values["execution_prefix_sha256"] = prefix.digest
            self._committed[request.session_id] = (request.step_id, request.instruction)
            return ModelResult(
                action_space=self.action_space,
                actions=(ModelAction("discrete_chunk", values),),
                timing={"policy_ms": float(chunk.latency_ms), **chunk.timing},
                policy_revision=str(chunk.policy_version),
            )

    def reset(self, session_id: str) -> None:
        """Release model state and continuation metadata for one HTTP session."""
        with self._lock:
            self._core.reset_sessions([self._session_key(session_id)])
            self._committed.pop(session_id, None)

    @staticmethod
    def _session_key(session_id: str) -> SessionKey:
        return SessionKey(env_id=session_id, episode_id="http")
