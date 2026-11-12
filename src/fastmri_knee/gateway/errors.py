"""RFC 9457 problem+json errors. Internal failures never leak details to clients."""

from http import HTTPStatus

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastmri_knee.gateway import telemetry

PROBLEM_JSON = "application/problem+json"
log = structlog.get_logger()


class ProblemError(Exception):
    """Raise from routes to produce a problem+json response with this status."""

    def __init__(self, status: int, detail: str, headers: dict[str, str] | None = None):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.headers = headers


def problem_response(
    status: int, detail: str, request_id: str | None, headers: dict[str, str] | None = None
) -> JSONResponse:
    title = HTTPStatus(status).phrase
    body = {"type": "about:blank", "title": title, "status": status, "detail": detail, "request_id": request_id}
    return JSONResponse(body, status_code=status, media_type=PROBLEM_JSON, headers=headers)


def _rid(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProblemError)
    async def _problem(request: Request, exc: ProblemError) -> JSONResponse:
        telemetry.rejected_status(exc.status)
        return problem_response(exc.status, exc.detail, _rid(request), exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        telemetry.rejected_status(exc.status_code)
        headers = dict(exc.headers) if exc.headers else None
        return problem_response(exc.status_code, str(exc.detail), _rid(request), headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Field locations and messages only; never echo the submitted input.
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        telemetry.rejected_status(422)
        return problem_response(422, detail, _rid(request))
