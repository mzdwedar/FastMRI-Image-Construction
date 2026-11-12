"""structlog JSON baseline and the per-request access log. Never log request bodies, filenames or API keys."""

import logging
import sys
import time

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from fastmri_knee.gateway import telemetry

access_log = structlog.get_logger("access")
_UNLOGGED = frozenset({telemetry.METRICS_PATH, "/healthz"})  # scrape and liveness polls would drown real traffic


def _trace_ids() -> tuple[str | None, str | None]:
    """W3C ids of the current span, or (None, None) while tracing is not installed or no span is active."""
    try:
        from opentelemetry import trace
    except ImportError:
        return None, None
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None, None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level]),
        cache_logger_on_first_use=False,
    )


class AccessLogMiddleware:
    """One JSON line per request, written after the response (or the unhandled error) is known. Sits inside
    `RequestIdMiddleware`, which has bound `request_id` and turns an unhandled error into the 500 response.

    `route` is the matched template, never the raw path. `slices` and `client` (a key-id hash, never the key)
    are set on `request.state` by the route and by auth; they stay null where unknown (401, 404, probes)."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in _UNLOGGED:
            await self.app(scope, receive, send)
            return
        status = 500
        start = time.perf_counter()

        async def send_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_status)
        finally:
            state = scope.get("state", {})
            monitor = getattr(scope["app"].state, "readiness", None) if "app" in scope else None
            trace_id, span_id = _trace_ids()
            access_log.info(
                "request",
                trace_id=trace_id,
                span_id=span_id,
                route=getattr(scope.get("route"), "path", None) or telemetry.UNMATCHED,
                method=scope["method"] if scope["method"] in telemetry.METHODS else "OTHER",
                status=status,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
                slices=state.get("slices"),
                model_version=monitor.model_version if monitor else None,
                client=state.get("client"),
            )
