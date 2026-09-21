"""Model- and robot-independent contracts for raw policy serving."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_policy
from typing import Any, Protocol, runtime_checkable

_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class ServeError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self.message = message


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ServeError(400, "invalid_request", f"{name} is invalid")
    return value


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ServeError(400, "invalid_request", f"{name} must be an object")
    return dict(value)


@dataclass(frozen=True, slots=True)
class RawImage:
    name: str
    mime_type: str
    data: bytes


@dataclass(frozen=True, slots=True)
class RawPolicyRequest:
    session_id: str
    request_id: str
    step_id: int
    instruction: str
    state: Mapping[str, Any]
    images: tuple[RawImage, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelAction:
    kind: str
    values: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelResult:
    action_space: str
    actions: tuple[ModelAction, ...]
    timing: Mapping[str, float]
    policy_revision: str | None = None


class ServingAdapter(Protocol):
    action_space: str

    def capabilities(self) -> dict[str, Any]: ...

    def infer(self, request: RawPolicyRequest) -> ModelResult: ...

    def reset(self, session_id: str) -> None: ...


@runtime_checkable
class BatchServingAdapter(ServingAdapter, Protocol):
    """Opt-in cross-session batching; results preserve request order.

    Return an exception in a row for invalid input without failing its peers.
    A raised exception fails the whole model call. The adapter owns preparation,
    tensor collation, and per-row restoration; the scheduler sees only requests.
    """

    def infer_batch(self, requests: Sequence[RawPolicyRequest]) -> Sequence[ModelResult | Exception]:
        """Execute one batch and return exactly one outcome per request."""
        ...


def payload_fingerprint(value: Mapping[str, Any]) -> str:
    """Return a stable digest for an idempotent JSON operation."""

    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ServeError(400, "invalid_request", "request contains non-JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


def request_fingerprint(request: RawPolicyRequest) -> str:
    """Hash semantic step content, independent of multipart boundary bytes."""

    return payload_fingerprint(
        {
            "session_id": request.session_id,
            "request_id": request.request_id,
            "step_id": request.step_id,
            "instruction": request.instruction,
            "state": request.state,
            "metadata": request.metadata,
            "images": [
                {
                    "name": image.name,
                    "mime_type": image.mime_type,
                    "sha256": hashlib.sha256(image.data).hexdigest(),
                }
                for image in request.images
            ],
        }
    )


def parse_json(body: bytes, name: str = "request") -> dict[str, Any]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServeError(400, "invalid_json", f"{name} is not valid JSON") from exc
    return _object(value, name)


def parse_step(
    content_type: str,
    body: bytes,
    *,
    route_session_id: str,
    idempotency_key: str | None,
    maximum_images: int = 8,
    maximum_image_bytes: int = 16 * 1024 * 1024,
) -> RawPolicyRequest:
    if not content_type.lower().startswith("multipart/form-data"):
        raise ServeError(415, "unsupported_media_type", "steps require multipart/form-data")
    try:
        prefix = b"Content-Type: " + content_type.encode("ascii") + b"\r\nMIME-Version: 1.0\r\n\r\n"
    except UnicodeEncodeError as exc:
        raise ServeError(400, "invalid_multipart", "invalid Content-Type") from exc
    message = BytesParser(policy=email_policy).parsebytes(prefix + body)
    if not message.is_multipart():
        raise ServeError(400, "invalid_multipart", "body is not valid multipart data")

    metadata_body: bytes | None = None
    image_parts: dict[int, tuple[str, bytes]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        if part.get_content_disposition() != "form-data" or not isinstance(payload, bytes):
            raise ServeError(400, "invalid_multipart", "invalid multipart part")
        if name == "metadata":
            if metadata_body is not None:
                raise ServeError(400, "invalid_multipart", "duplicate metadata part")
            metadata_body = payload
            continue
        match = re.fullmatch(r"image_(\d+)", str(name))
        if match is None:
            raise ServeError(400, "invalid_multipart", f"unexpected multipart field {name!r}")
        index = int(match.group(1))
        if index in image_parts:
            raise ServeError(400, "invalid_multipart", f"duplicate image_{index}")
        if len(payload) > maximum_image_bytes:
            raise ServeError(413, "image_too_large", f"image_{index} is too large")
        image_parts[index] = (part.get_content_type().lower(), payload)

    if metadata_body is None:
        raise ServeError(400, "invalid_multipart", "metadata part is required")
    metadata = parse_json(metadata_body, "metadata")
    if metadata.get("schema") != "vvla.policy.step.v1":
        raise ServeError(400, "unsupported_schema", "unsupported step schema")
    session_id = _identifier(metadata.get("session_id"), "session_id")
    request_id = _identifier(metadata.get("request_id"), "request_id")
    if session_id != route_session_id:
        raise ServeError(400, "session_mismatch", "route and body session IDs differ")
    if request_id != idempotency_key:
        raise ServeError(400, "idempotency_mismatch", "Idempotency-Key must equal request_id")
    step_id = metadata.get("step_id")
    if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0:
        raise ServeError(400, "invalid_request", "step_id must be a non-negative integer")
    instruction = metadata.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ServeError(400, "invalid_request", "instruction must not be empty")
    if metadata.get("reset", False):
        raise ServeError(400, "invalid_request", "use the reset endpoint before step 0")

    declared = metadata.get("images")
    if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
        raise ServeError(400, "invalid_request", "images must be a list")
    if not 1 <= len(declared) <= maximum_images:
        raise ServeError(400, "invalid_request", f"images must contain 1..{maximum_images} entries")
    if set(image_parts) != set(range(len(declared))):
        raise ServeError(400, "invalid_multipart", "image parts do not match metadata")
    images: list[RawImage] = []
    for index, item in enumerate(declared):
        info = _object(item, f"images[{index}]")
        mime, data = image_parts[index]
        if mime not in {"image/jpeg", "image/png"} or info.get("mime_type") != mime:
            raise ServeError(415, "unsupported_image", f"image_{index} must be JPEG or PNG")
        images.append(RawImage(_identifier(info.get("name"), "image.name"), mime, data))

    return RawPolicyRequest(
        session_id=session_id,
        request_id=request_id,
        step_id=step_id,
        instruction=instruction,
        state=_object(metadata.get("state"), "state"),
        images=tuple(images),
        metadata=_object(metadata.get("metadata", {}), "metadata.metadata"),
    )


def parse_structured_step(
    payload: object,
    *,
    maximum_images: int = 8,
    maximum_image_bytes: int = 16 * 1024 * 1024,
) -> RawPolicyRequest:
    """Validate a transport-neutral structured policy step payload."""

    metadata = _object(payload, "step request")
    if metadata.get("schema") != "vvla.policy.step.v1":
        raise ServeError(400, "unsupported_schema", "unsupported step schema")
    session_id = _identifier(metadata.get("session_id"), "session_id")
    request_id = _identifier(metadata.get("request_id"), "request_id")
    step_id = metadata.get("step_id")
    if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0:
        raise ServeError(400, "invalid_request", "step_id must be a non-negative integer")
    instruction = metadata.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ServeError(400, "invalid_request", "instruction must not be empty")
    if metadata.get("reset", False):
        raise ServeError(400, "invalid_request", "use reset before step 0")

    declared = metadata.get("images")
    if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
        raise ServeError(400, "invalid_request", "images must be a list")
    if not 1 <= len(declared) <= maximum_images:
        raise ServeError(400, "invalid_request", f"images must contain 1..{maximum_images} entries")
    images: list[RawImage] = []
    for index, item in enumerate(declared):
        image = _object(item, f"images[{index}]")
        mime_type = image.get("mime_type")
        if mime_type not in {"image/jpeg", "image/png"}:
            raise ServeError(415, "unsupported_image", f"images[{index}] must be JPEG or PNG")
        data = image.get("data")
        if not isinstance(data, bytes) or not data:
            raise ServeError(400, "invalid_request", f"images[{index}].data must contain bytes")
        if len(data) > maximum_image_bytes:
            raise ServeError(413, "image_too_large", f"images[{index}] is too large")
        images.append(
            RawImage(
                _identifier(image.get("name"), "image.name"),
                mime_type,
                data,
            )
        )

    return RawPolicyRequest(
        session_id=session_id,
        request_id=request_id,
        step_id=step_id,
        instruction=instruction,
        state=_object(metadata.get("state"), "state"),
        images=tuple(images),
        metadata=_object(metadata.get("metadata", {}), "metadata.metadata"),
    )


__all__ = [
    "BatchServingAdapter",
    "ModelAction",
    "ModelResult",
    "RawImage",
    "RawPolicyRequest",
    "ServeError",
    "ServingAdapter",
    "payload_fingerprint",
    "parse_json",
    "parse_step",
    "parse_structured_step",
    "request_fingerprint",
]
