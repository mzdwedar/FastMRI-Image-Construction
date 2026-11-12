"""Prometheus metrics (SPEC-observability, "Metrics") and the `/metrics` exposition.

Multiprocess: when `PROMETHEUS_MULTIPROC_DIR` is set (gunicorn, several workers) it must be set before this module
is imported; prometheus_client then writes per-process files there and `/metrics` aggregates them. Without it the
default in-process registry is used (tests, single process).

Labels are bounded: `route` is the matched path template (or "unmatched"), never the raw path; no request ids,
filenames or keys.

Tracing (SPEC-observability, "Traces"): `setup_tracing` instruments the app and points `span()` at a provider. The
gateway exports every span and does not head-sample: the keep/drop decision (errors, slow requests, a probabilistic
share) is made in Alloy's tail sampling, which also sees Triton's spans (Triton ignores the sampled flag, see O9).
Without a provider `span()` is a no-op, so tests and a gateway with no OTLP endpoint pay nothing.
"""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send

if TYPE_CHECKING:
    from fastapi import FastAPI
    from opentelemetry.sdk.trace import TracerProvider

DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
UPLOAD_BUCKETS = tuple(float(1 << n) for n in range(10, 31, 2))  # 1 KiB ... 1 GiB
STAGES = ("preprocess", "triton", "encode")
REJECT_REASONS = ("overloaded", "too_large", "unauthorized", "invalid")
_REJECT_BY_STATUS = {400: "invalid", 415: "invalid", 422: "invalid", 401: "unauthorized", 413: "too_large"}
GRPC_CODES = frozenset(
    [
        "CANCELLED",
        "UNKNOWN",
        "INVALID_ARGUMENT",
        "DEADLINE_EXCEEDED",
        "NOT_FOUND",
        "ALREADY_EXISTS",
        "PERMISSION_DENIED",
        "RESOURCE_EXHAUSTED",
        "FAILED_PRECONDITION",
        "ABORTED",
        "OUT_OF_RANGE",
        "UNIMPLEMENTED",
        "INTERNAL",
        "UNAVAILABLE",
        "DATA_LOSS",
        "UNAUTHENTICATED",
    ]
)
INVALID_OUTPUT = "INVALID_OUTPUT"  # not a gRPC status: Triton answered OK but the tensor failed validation
UNMATCHED = "unmatched"
METRICS_PATH = "/metrics"
METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})

REQUESTS = Counter("gateway_requests_total", "HTTP requests", ["route", "method", "status_class"])
REQUEST_DURATION = Histogram(
    "gateway_request_duration_seconds", "HTTP request duration", ["route"], buckets=DURATION_BUCKETS
)
INFLIGHT = Gauge(
    "gateway_inflight_requests", "Inference requests holding a backpressure slot", multiprocess_mode="livesum"
)
REJECTED = Counter("gateway_rejected_total", "Requests rejected before inference", ["reason"])
UPLOAD_BYTES = Histogram("gateway_upload_bytes", "Request payload size in bytes", buckets=UPLOAD_BUCKETS)
SLICES = Counter("gateway_slices_total", "Slices processed", ["route"])
STAGE_DURATION = Histogram(
    "gateway_stage_duration_seconds", "Duration of a processing stage", ["stage"], buckets=DURATION_BUCKETS
)
TRITON_ERRORS = Counter("gateway_triton_errors_total", "Failed Triton calls", ["code"])
MODEL_READY = Gauge(
    "gateway_model_ready", "1 when Triton is reachable and the model is ready", multiprocess_mode="livemax"
)
MODEL_INFO = Gauge(
    "gateway_model_info",
    "1 for the served model version",
    ["version", "preprocess_version"],
    multiprocess_mode="livemax",
)

TRACER_NAME = "fastmri_knee.gateway"
UNTRACED_PATHS = "healthz,readyz,metrics"  # probes and scrapes would be most of the spans and none of the signal
_tracer: trace.Tracer = trace.NoOpTracer()

_served: dict[str, str] = {}  # this process's currently published model_info labels


def rejected(reason: str) -> None:
    REJECTED.labels(reason).inc()


def rejected_status(status: int) -> None:
    """Count a client-error response under its rejection reason, if it has one."""
    if reason := _REJECT_BY_STATUS.get(status):
        rejected(reason)


def triton_error(code: str) -> None:
    TRITON_ERRORS.labels(code if code in GRPC_CODES or code == INVALID_OUTPUT else "UNKNOWN").inc()


@contextmanager
def stage(name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        STAGE_DURATION.labels(name).observe(time.perf_counter() - start)


def set_model_state(ready: bool, version: str | None, preprocess_version: str) -> None:
    """Publish readiness; keeps exactly one `gateway_model_info` series at 1 (superseded versions go to 0)."""
    MODEL_READY.set(1 if ready else 0)
    if not (ready and version):
        return
    old = _served.get("version")
    if old and old != version:
        MODEL_INFO.labels(old, preprocess_version).set(0)
    MODEL_INFO.labels(version, preprocess_version).set(1)
    _served["version"] = version


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    """A child span of the current one (an error inside marks it as failed). Attributes must be bounded and
    never carry filenames, keys or payload values."""
    with _tracer.start_as_current_span(name, attributes=attributes or None) as sp:
        yield sp


def setup_tracing(app: "FastAPI", provider: "TracerProvider") -> None:
    """Instrument `app` (one server span per request, probes excluded) and route `span()` to `provider`.

    Instrumentation wraps the whole middleware stack, so the trace context is active in `AccessLogMiddleware`
    and the access-log line carries the request's `trace_id`.
    """
    global _tracer
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    _tracer = provider.get_tracer(TRACER_NAME)
    # receive/send spans are one per body chunk (about 170 for a 3-slice upload) and say nothing the server span lacks
    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=provider, excluded_urls=UNTRACED_PATHS, exclude_spans=["receive", "send"]
    )


def disable_tracing() -> None:
    """Make `span()` a no-op again (a new app without a provider must not keep exporting to an old one)."""
    global _tracer
    _tracer = trace.NoOpTracer()


def otlp_provider(endpoint: str) -> "TracerProvider":
    """Tracer provider that exports every span over OTLP/gRPC. Built per worker, after the fork: the batch
    exporter owns a thread that would not survive a fork."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "gateway"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    return provider


def render() -> tuple[bytes, str]:
    """Exposition body and content type, aggregated over workers when multiprocess is on."""
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
    else:
        registry = REGISTRY
    return generate_latest(registry), CONTENT_TYPE_LATEST


class MetricsMiddleware:
    """Request count and duration per matched route. Sits inside `RequestIdMiddleware`, so an unhandled error is
    counted as a 500 here and re-raised for that middleware to turn into the response."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == METRICS_PATH:
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
            route = getattr(scope.get("route"), "path", None) or UNMATCHED  # set by the router during dispatch
            REQUESTS.labels(
                route, scope["method"] if scope["method"] in METHODS else "OTHER", f"{status // 100}xx"
            ).inc()
            REQUEST_DURATION.labels(route).observe(time.perf_counter() - start)
