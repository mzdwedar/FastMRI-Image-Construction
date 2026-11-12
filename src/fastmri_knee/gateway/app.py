"""App factory."""

import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Response
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from fastmri_knee.gateway import telemetry
from fastmri_knee.gateway.auth import key_hashes_for
from fastmri_knee.gateway.errors import install_handlers, problem_response
from fastmri_knee.gateway.limits import InflightLimiter
from fastmri_knee.gateway.logging import AccessLogMiddleware, configure_logging
from fastmri_knee.gateway.readiness import ReadinessMonitor
from fastmri_knee.gateway.routes import probes, router
from fastmri_knee.gateway.settings import Settings
from fastmri_knee.gateway.triton_client import GrpcTritonClient, TritonClient

log = structlog.get_logger()
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")


class RequestIdMiddleware:
    """Echo a valid client `X-Request-ID` or generate one; also turns unhandled errors into a 500 problem
    response here (not in an `Exception` handler, which runs outside middleware and would lose the header)."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = MutableHeaders(scope=scope).get("x-request-id")
        rid = supplied if supplied and _VALID_REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid
        trace.get_current_span().set_attribute("request_id", rid)  # the server span (instrumentation is outermost)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=rid)
        started = False

        async def send_with_id(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = rid
                monitor = getattr(scope["app"].state, "readiness", None) if "app" in scope else None
                if monitor and (version := monitor.model_version):
                    headers["X-Model-Version"] = version
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            log.exception("unhandled_error", path=scope["path"])
            if started:
                raise
            response = problem_response(500, "Internal server error", rid)
            await response(scope, receive, send_with_id)


def create_app(
    settings: Settings | None = None,
    triton: TritonClient | None = None,
    tracer_provider: TracerProvider | None = None,
) -> FastAPI:
    """`tracer_provider` is for tests; in service `GATEWAY_OTLP_ENDPOINT` builds the exporting provider."""
    settings = settings or Settings()  # invalid settings raise here, at startup
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One gRPC channel per worker. Creating it makes no network call, so the gateway starts without Triton.
        app.state.triton = triton or GrpcTritonClient(settings.triton_url, settings.triton_model_name)
        app.state.readiness = ReadinessMonitor(app.state.triton, settings.readiness_interval_s)
        await app.state.readiness.start()
        log.info("gateway_started", env=settings.env)
        try:
            yield
        finally:
            await app.state.readiness.stop()
            await app.state.triton.close()
            log.info("gateway_stopped")

    app = FastAPI(title="fastmri-knee gateway", version="1", lifespan=lifespan)
    app.state.settings = settings
    app.state.key_hashes = key_hashes_for(settings)  # missing/empty keys file is a startup error
    app.state.limiter = InflightLimiter(settings.max_inflight)
    app.add_middleware(telemetry.MetricsMiddleware)  # inner: sees unhandled errors before RequestIdMiddleware
    app.add_middleware(AccessLogMiddleware)  # inside RequestIdMiddleware: request_id is bound, 500s are visible
    app.add_middleware(RequestIdMiddleware)
    install_handlers(app)
    provider = tracer_provider or (telemetry.otlp_provider(settings.otlp_endpoint) if settings.otlp_endpoint else None)
    if provider:
        telemetry.setup_tracing(app, provider)
    else:
        telemetry.disable_tracing()
    app.include_router(router)
    app.include_router(probes)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up. Never checks dependencies."""
        return {"status": "ok"}

    @app.get(telemetry.METRICS_PATH, include_in_schema=False)
    async def metrics() -> Response:
        """Prometheus exposition. Unauthenticated: reach it only over the internal network (SPEC-gateway)."""
        body, content_type = telemetry.render()
        return Response(body, media_type=content_type)

    return app
