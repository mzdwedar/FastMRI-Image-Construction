"""Versioned API routes."""

import asyncio

import numpy as np
import structlog
from fastapi import APIRouter, Request, Response

from fastmri_knee.gateway.errors import ProblemError
from fastmri_knee.gateway.triton_client import (
    InvalidOutputError,
    TritonClient,
    TritonError,
    TritonTimeoutError,
    infer_volume,
)

log = structlog.get_logger()
router = APIRouter(prefix="/v1")

SLICE_SHAPE = (1, 320, 320)
SLICE_BYTES = int(np.prod(SLICE_SHAPE)) * 4
OCTET_STREAM = "application/octet-stream"


async def run_inference(request: Request, volume: np.ndarray) -> np.ndarray:
    """Call Triton and translate failures into API errors. Internals are logged, never returned."""
    client: TritonClient = request.app.state.triton
    try:
        return await infer_volume(client, volume, request.app.state.settings.triton_timeout_s)
    except TritonTimeoutError:
        raise ProblemError(504, "Inference backend timed out") from None
    except TritonError as e:
        if e.code == "UNAVAILABLE":
            raise ProblemError(503, "Inference backend unavailable", {"Retry-After": "1"}) from None
        log.error("triton_error", code=e.code)
        raise ProblemError(500, "Internal server error") from None
    except InvalidOutputError as e:
        log.error("invalid_model_output", reason=str(e))  # TODO(O1): increment the error metric
        raise ProblemError(500, "Internal server error") from None


async def read_capped(request: Request, limit: int) -> bytes | None:
    """Read the body, stopping as soon as it exceeds `limit` bytes (returns None in that case)."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        return None
    buf = bytearray()
    async for part in request.stream():
        buf += part
        if len(buf) > limit:
            return None
    return bytes(buf)


@router.post(
    "/infer",
    summary="Low-level inference (benchmark path)",
    response_class=Response,
    responses={200: {"content": {OCTET_STREAM: {}}}},
)
async def infer(request: Request) -> Response:
    """Body: float32 bytes, N×1×320×320 with 1 ≤ N ≤ MAX_SLICES. Returns the same shape."""
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != OCTET_STREAM:
        raise ProblemError(415, f"Content-Type must be {OCTET_STREAM}")
    max_slices = request.app.state.settings.max_slices
    body = await read_capped(request, max_slices * SLICE_BYTES)
    if body is None:
        raise ProblemError(413, f"At most {max_slices} slices per request")
    if not body or len(body) % SLICE_BYTES:
        raise ProblemError(400, f"Body must be a non-empty multiple of {SLICE_BYTES} bytes (N×1×320×320 float32)")
    volume = np.frombuffer(body, dtype="<f4").reshape(-1, *SLICE_SHAPE)
    if not np.isfinite(volume).all():
        raise ProblemError(400, "Input contains non-finite values")
    out = await run_inference(request, volume)
    payload = await asyncio.to_thread(lambda: np.ascontiguousarray(out, dtype="<f4").tobytes())
    return Response(payload, media_type=OCTET_STREAM)
