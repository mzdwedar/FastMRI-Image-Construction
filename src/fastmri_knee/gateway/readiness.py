"""Background readiness loop: polls Triton model state and caches it, so probes never make a gRPC call."""

import asyncio
import contextlib
import time
from dataclasses import dataclass

import structlog

from fastmri_knee.gateway.triton_client import ModelState, TritonClient, TritonError

log = structlog.get_logger()

TRITON_UNREACHABLE = "triton_unreachable"
MODEL_NOT_READY = "model_not_ready"
_UNREACHABLE_CODES = {"UNAVAILABLE", "DEADLINE_EXCEEDED"}
STALE_AFTER_INTERVALS = 3  # a loop that stopped updating must not leave a stale "ready" behind


@dataclass(frozen=True)
class Readiness:
    ready: bool
    reason: str | None = None
    model: ModelState | None = None
    checked_at: float = 0.0  # time.monotonic()


class ReadinessMonitor:
    def __init__(self, client: TritonClient, interval_s: float):
        self._client = client
        self._interval = interval_s
        self._task: asyncio.Task[None] | None = None
        self._last = Readiness(ready=False, reason=TRITON_UNREACHABLE)  # until the first poll says otherwise

    @property
    def current(self) -> Readiness:
        if self._last.checked_at and time.monotonic() - self._last.checked_at > STALE_AFTER_INTERVALS * self._interval:
            return Readiness(ready=False, reason=TRITON_UNREACHABLE, checked_at=self._last.checked_at)
        return self._last

    @property
    def model_version(self) -> str | None:
        r = self.current
        return r.model.version if r.ready and r.model else None

    async def poll_once(self) -> Readiness:
        try:
            model = await self._client.model_state(self._interval)
            new = Readiness(ready=True, model=model, checked_at=time.monotonic())
        except TritonError as e:
            reason = TRITON_UNREACHABLE if e.code in _UNREACHABLE_CODES else MODEL_NOT_READY
            new = Readiness(ready=False, reason=reason, checked_at=time.monotonic())
        except Exception:
            log.exception("readiness_poll_failed")
            new = Readiness(ready=False, reason=TRITON_UNREACHABLE, checked_at=time.monotonic())
        old = self._last
        if (old.ready, old.reason, old.model and old.model.version) != (
            new.ready,
            new.reason,
            new.model and new.model.version,
        ):
            log.info("readiness_changed", ready=new.ready, reason=new.reason, version=new.model and new.model.version)
        self._last = new
        return new

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            await self.poll_once()

    async def start(self) -> None:
        await self.poll_once()  # /readyz is accurate from the first request
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
