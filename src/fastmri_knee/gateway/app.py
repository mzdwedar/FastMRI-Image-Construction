"""App factory."""

import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from fastmri_knee.gateway.errors import install_handlers, problem_response
from fastmri_knee.gateway.logging import configure_logging
from fastmri_knee.gateway.routes import router
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
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=rid)
        started = False

        async def send_with_id(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                MutableHeaders(scope=message)["X-Request-ID"] = rid
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            log.exception("unhandled_error", path=scope["path"])
            if started:
                raise
            response = problem_response(500, "Internal server error", rid)
            await response(scope, receive, send_with_id)


def create_app(settings: Settings | None = None, triton: TritonClient | None = None) -> FastAPI:
    settings = settings or Settings()  # invalid settings raise here, at startup
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One gRPC channel per worker. Creating it makes no network call, so the gateway starts without Triton.
        app.state.triton = triton or GrpcTritonClient(settings.triton_url, settings.triton_model_name)
        log.info("gateway_started", env=settings.env)
        try:
            yield
        finally:
            await app.state.triton.close()
            log.info("gateway_stopped")

    app = FastAPI(title="fastmri-knee gateway", version="1", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(RequestIdMiddleware)
    install_handlers(app)
    app.include_router(router)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up. Never checks dependencies."""
        return {"status": "ok"}

    return app
