"""Backpressure: a global in-flight limit that rejects immediately instead of queueing."""

from collections.abc import AsyncIterator

from fastapi import Request

from fastmri_knee.gateway import telemetry
from fastmri_knee.gateway.errors import ProblemError


class InflightLimiter:
    """Counts volumes being processed. Single event loop per worker, so plain integers are race-free."""

    def __init__(self, limit: int):
        self.limit = limit
        self.active = 0

    def try_acquire(self) -> bool:
        if self.active >= self.limit:
            return False
        self.active += 1
        telemetry.INFLIGHT.inc()
        return True

    def release(self) -> None:
        self.active -= 1
        telemetry.INFLIGHT.dec()


async def limit_inflight(request: Request) -> AsyncIterator[None]:
    limiter: InflightLimiter = request.app.state.limiter
    if not limiter.try_acquire():
        telemetry.rejected("overloaded")
        raise ProblemError(503, "Server busy, retry shortly", {"Retry-After": "1"})
    try:
        yield
    finally:
        limiter.release()
