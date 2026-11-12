"""Versioned API routes."""

import asyncio
import io
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import structlog
from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import JSONResponse

from fastmri_knee.gateway import telemetry
from fastmri_knee.gateway.auth import require_api_key
from fastmri_knee.gateway.errors import ProblemError
from fastmri_knee.gateway.limits import limit_inflight
from fastmri_knee.gateway.triton_client import (
    F32,
    InvalidOutputError,
    TritonClient,
    TritonError,
    TritonTimeoutError,
    infer_volume,
)
from fastmri_knee.gateway.undersample import random_cartesian_mask
from fastmri_knee.gateway.uploads import read_kspace, stream_file_part, temp_upload
from fastmri_knee.preprocess import PREPROCESS_VERSION, kspace_to_image_scaled

log = structlog.get_logger()
router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
probes = APIRouter()  # unversioned, unauthenticated

SLICE_SHAPE = (1, 320, 320)
SLICE_BYTES = int(np.prod(SLICE_SHAPE)) * 4
OCTET_STREAM = "application/octet-stream"
HDF5 = "application/x-hdf5"


async def run_inference(request: Request, volume: F32) -> F32:
    """Call Triton and translate failures into API errors. Internals are logged, never returned."""
    client: TritonClient = request.app.state.triton
    try:
        with telemetry.stage("triton"):
            return await infer_volume(client, volume, request.app.state.settings.triton_timeout_s)
    except TritonTimeoutError:
        telemetry.triton_error("DEADLINE_EXCEEDED")
        raise ProblemError(504, "Inference backend timed out") from None
    except TritonError as e:
        telemetry.triton_error(e.code)
        if e.code == "UNAVAILABLE":
            raise ProblemError(503, "Inference backend unavailable", {"Retry-After": "1"}) from None
        log.error("triton_error", code=e.code)
        raise ProblemError(500, "Internal server error") from None
    except InvalidOutputError as e:
        telemetry.triton_error(telemetry.INVALID_OUTPUT)
        log.error("invalid_model_output", reason=str(e))
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
    dependencies=[Depends(limit_inflight)],
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
    telemetry.UPLOAD_BYTES.observe(len(body))
    telemetry.SLICES.labels("/v1/infer").inc(len(volume))
    request.state.slices = len(volume)
    out = await run_inference(request, volume)
    with telemetry.stage("encode"), telemetry.span("response.encode"):
        payload = await asyncio.to_thread(lambda: np.ascontiguousarray(out, dtype="<f4").tobytes())
    return Response(payload, media_type=OCTET_STREAM)


@router.get("/model", summary="Served model version and Triton-reported details")
async def model_info(request: Request) -> dict[str, Any]:
    state = request.app.state.readiness.current
    if not state.ready or state.model is None:
        raise ProblemError(503, "Model is not ready", {"Retry-After": "1"})
    m = state.model
    return {
        "model_name": request.app.state.settings.triton_model_name,
        "version": m.version,
        "platform": m.platform,
        "max_batch_size": m.max_batch_size,
        "inputs": m.inputs,
        "outputs": m.outputs,
        "preprocess_version": PREPROCESS_VERSION,
    }


@probes.get("/readyz", include_in_schema=False)
async def readyz(request: Request) -> JSONResponse:
    """Readiness: Triton reachable and the model ready. Reads the cache the background loop maintains."""
    state = request.app.state.readiness.current
    if state.ready and state.model:
        return JSONResponse({"status": "ready", "model_version": state.model.version})
    return JSONResponse({"status": "not_ready", "reason": state.reason}, status_code=503)


def preprocess_volume(kspace: npt.NDArray[np.complex64], acceleration: int | None) -> tuple[F32, F32]:
    """(S, H, W) complex64 -> ((S, 1, 320, 320) float32 model input, (S,) float32 per-slice scale).

    One call per volume; each slice is normalized by its own max. Blocking: run in a worker thread.
    """
    mask = random_cartesian_mask(kspace.shape[-1], acceleration) if acceleration else None
    pairs = [kspace_to_image_scaled(s, mask) for s in kspace]
    return np.stack([p[0] for p in pairs]), np.array([p[1] for p in pairs], dtype=np.float32)


def encode_reconstruction(recon: F32, scale: F32, model_version: str, request_id: str | None) -> bytes:
    """HDF5 response: dataset `reconstruction` (S, 320, 320) float32 and root attrs."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("reconstruction", data=recon[:, 0].astype(np.float32, copy=False))
        f.attrs["model_version"] = model_version
        f.attrs["preprocess_version"] = PREPROCESS_VERSION
        f.attrs["request_id"] = request_id or ""
        f.attrs["scale"] = scale
    return buf.getvalue()


@router.post(
    "/reconstructions",
    summary="Reconstruct a k-space volume",
    response_class=Response,
    responses={200: {"content": {HDF5: {}}}},
    dependencies=[Depends(limit_inflight)],
)
async def reconstructions(
    request: Request,
    acceleration: int | None = Query(
        None,
        ge=2,
        le=16,
        description="DEMO FEATURE. Simulate undersampling of a fully sampled file with a random Cartesian mask "
        "at this acceleration factor. Off by default.",
    ),
) -> Response:
    """Upload: multipart field `file`, a single-coil `.h5` with `kspace` (slices, H, W) complex64.

    Returns `application/x-hdf5`: dataset `reconstruction` (slices, 320, 320) float32 and root attributes
    `model_version`, `preprocess_version`, `request_id`, and `scale` (per-slice max the model input was divided
    by, so `reconstruction * scale[:, None, None]` restores relative intensity).
    """
    settings = request.app.state.settings
    with temp_upload(settings.tmp_dir) as path:
        with telemetry.span("upload.read") as sp:
            await stream_file_part(request, path, settings.max_upload_mb << 20)
            size = path.stat().st_size
            sp.set_attribute("upload.bytes", size)
        telemetry.UPLOAD_BYTES.observe(size)
        with telemetry.span("h5.validate"):
            kspace = await asyncio.to_thread(read_kspace, path, settings.max_slices)
    telemetry.SLICES.labels("/v1/reconstructions").inc(len(kspace))
    request.state.slices = len(kspace)
    with telemetry.stage("preprocess"), telemetry.span("preprocess", slices=len(kspace)):
        images, scale = await asyncio.to_thread(preprocess_volume, kspace, acceleration)
    recon = await run_inference(request, images)
    version = request.app.state.readiness.model_version or "unknown"
    with telemetry.stage("encode"), telemetry.span("response.encode"):
        body = await asyncio.to_thread(
            encode_reconstruction, recon, scale, version, getattr(request.state, "request_id", None)
        )
    return Response(body, media_type=HDF5)
