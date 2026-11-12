"""Triton gRPC client: chunking, deadline, retry of UNAVAILABLE only, output validation.

`TritonClient` is the narrow protocol the routes depend on; `GrpcTritonClient` is the real implementation and
tests use a fake. Retry/chunk/validate logic lives in `infer_volume` so it is exercised with the fake too.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import structlog
from opentelemetry import propagate

from fastmri_knee.gateway import telemetry

log = structlog.get_logger()

F32 = npt.NDArray[np.float32]

INPUT_NAME = "input"
OUTPUT_NAME = "output"
MAX_RETRIES = 2
BACKOFF_BASE_S = 0.05


class TritonError(Exception):
    """A failed Triton call, carrying the gRPC status name (e.g. `UNAVAILABLE`)."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


class TritonTimeoutError(Exception):
    """The request deadline passed before Triton answered (-> 504)."""


class InvalidOutputError(Exception):
    """Triton returned a wrongly shaped or non-finite output. Never returned to clients (-> 500)."""


@dataclass(frozen=True)
class ModelState:
    """A ready model as reported by Triton."""

    version: str
    platform: str = ""
    max_batch_size: int = 0
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)


class TritonClient(Protocol):
    async def model_state(self, timeout_s: float) -> ModelState:
        """The served, ready model. Raises `TritonError`: UNAVAILABLE/DEADLINE_EXCEEDED when Triton cannot be
        reached, any other code (e.g. NOT_FOUND) when the model is not loaded or not ready."""
        ...

    async def get_max_batch_size(self) -> int:
        """Max slices per Triton request (>= 1). Raises `TritonError` if Triton cannot be reached."""
        ...

    async def infer(self, batch: F32, timeout_s: float) -> F32:
        """One Triton request. Raises `TritonError` on a gRPC failure."""
        ...

    async def close(self) -> None: ...


def _validate_output(out: F32, expected_shape: tuple[int, ...]) -> F32:
    if out.shape != expected_shape:
        raise InvalidOutputError(f"unexpected output shape {out.shape}, wanted {expected_shape}")
    if not np.isfinite(out).all():
        raise InvalidOutputError("non-finite values in model output")
    return out


async def _infer_chunk(client: TritonClient, chunk: F32, deadline: float) -> F32:
    with telemetry.span("triton.infer", slices=len(chunk)) as sp:  # one span per chunk, retries included
        out = await _infer_chunk_attempts(client, chunk, deadline, sp)
    return out


async def _infer_chunk_attempts(client: TritonClient, chunk: F32, deadline: float, sp: Any) -> F32:
    attempt = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TritonTimeoutError
        try:
            out = await client.infer(chunk, remaining)
        except TritonError as e:
            if e.code == "DEADLINE_EXCEEDED":
                raise TritonTimeoutError from e
            if e.code != "UNAVAILABLE" or attempt >= MAX_RETRIES:
                raise
            delay = BACKOFF_BASE_S * 2**attempt * random.uniform(0.5, 1.5)
            if time.monotonic() + delay >= deadline:
                raise TritonTimeoutError from e
            log.warning("triton_retry", attempt=attempt + 1, code=e.code)
            await asyncio.sleep(delay)
            attempt += 1
            sp.set_attribute("triton.retries", attempt)
            continue
        return _validate_output(out, chunk.shape)


async def infer_volume(client: TritonClient, volume: F32, timeout_s: float) -> F32:
    """Run an (N, 1, H, W) float32 volume through Triton in chunks of <= max_batch_size, sent concurrently.

    `timeout_s` is the overall deadline for the whole call, retries included.
    """
    deadline = time.monotonic() + timeout_s
    try:
        async with asyncio.timeout(timeout_s):
            size = max(1, await client.get_max_batch_size())
            chunks = [volume[i : i + size] for i in range(0, len(volume), size)]
            outs = await asyncio.gather(*(_infer_chunk(client, c, deadline) for c in chunks))
    except TimeoutError as e:
        raise TritonTimeoutError from e
    return np.concatenate(outs, axis=0)


class GrpcTritonClient:
    """`tritonclient.grpc.aio` wrapper. One channel per worker; created in the app lifespan.

    Construction does not touch the network, so the gateway starts (and stays live) while Triton is down.
    """

    def __init__(self, url: str, model_name: str):
        from tritonclient.grpc import aio as grpcclient

        self._grpc = grpcclient
        self._client = grpcclient.InferenceServerClient(url=url)
        self._model = model_name
        self._max_batch: int | None = None

    @staticmethod
    def _code(e: Exception) -> str:
        status = e.status() if hasattr(e, "status") else None  # "StatusCode.UNAVAILABLE"
        return str(status or "UNKNOWN").removeprefix("StatusCode.")

    async def get_max_batch_size(self) -> int:
        if self._max_batch is None:
            try:
                cfg = await self._client.get_model_config(self._model)
            except Exception as e:
                raise TritonError(self._code(e), "model config unavailable") from e
            self._max_batch = max(1, cfg.config.max_batch_size)  # 0 = batching disabled -> 1 slice per call
        return self._max_batch

    async def infer(self, batch: F32, timeout_s: float) -> F32:
        inp = self._grpc.InferInput(INPUT_NAME, list(batch.shape), "FP32")
        inp.set_data_from_numpy(np.ascontiguousarray(batch, dtype=np.float32))
        # W3C trace context rides in gRPC metadata (lowercase `traceparent`, O9); Triton parents its spans on it.
        headers: dict[str, str] = {}
        propagate.inject(headers)
        try:
            res = await self._client.infer(
                self._model,
                [inp],
                outputs=[self._grpc.InferRequestedOutput(OUTPUT_NAME)],
                client_timeout=timeout_s,
                headers=headers,
            )
        except Exception as e:
            raise TritonError(self._code(e), "infer failed") from e
        out = res.as_numpy(OUTPUT_NAME)
        if out is None:
            raise InvalidOutputError("response has no output tensor")
        return cast(F32, out)

    async def model_state(self, timeout_s: float) -> ModelState:
        try:
            meta = await self._client.get_model_metadata(self._model, as_json=True, client_timeout=timeout_s)
            versions = sorted(meta.get("versions", []), key=int)
            if not versions:
                raise TritonError("NOT_FOUND", "no versions listed")
            version = versions[-1]
            if not await self._client.is_model_ready(self._model, version, client_timeout=timeout_s):
                raise TritonError("NOT_FOUND", "model not ready")
            cfg = await self._client.get_model_config(self._model, version, as_json=True, client_timeout=timeout_s)
        except TritonError:
            raise
        except Exception as e:
            raise TritonError(self._code(e), "model state unavailable") from e
        return ModelState(
            version=version,
            platform=meta.get("platform", ""),
            max_batch_size=int(cfg.get("config", {}).get("max_batch_size", 0)),
            inputs=meta.get("inputs", []),
            outputs=meta.get("outputs", []),
        )

    async def close(self) -> None:
        await self._client.close()
