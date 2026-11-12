import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from fastmri_knee.gateway.app import create_app
from fastmri_knee.gateway.settings import Settings
from fastmri_knee.gateway.triton_client import GrpcTritonClient
from tests.fixtures.synthetic import make_kspace
from tests.unit.gateway.fakes import FakeTritonClient, unavailable
from tests.unit.gateway.test_reconstructions import h5_bytes

MANUAL = {"upload.read", "h5.validate", "preprocess", "triton.infer", "response.encode"}


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def make(exporter, tmp_path):
    open_clients = []

    def build(fake=None, **kw):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        settings = Settings(env="dev", auth_disabled=True, tmp_dir=str(tmp_path), readiness_interval_s=30, **kw)
        c = TestClient(
            create_app(settings, fake or FakeTritonClient(), tracer_provider=provider), raise_server_exceptions=False
        )
        open_clients.append(c.__enter__())  # runs the lifespan, which creates app.state.triton
        return c

    yield build
    for c in open_clients:
        c.__exit__(None, None, None)


def reconstruct(c, **kw):
    kspace = make_kspace(slices=2)
    return c.post("/v1/reconstructions", files={"file": ("patient-7.h5", h5_bytes(kspace), "application/x-hdf5")}, **kw)


def by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def test_one_request_one_trace_with_spec_spans(make, exporter):
    assert reconstruct(make()).status_code == 200
    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert names >= MANUAL
    assert len({s.context.trace_id for s in spans}) == 1
    server = next(s for s in spans if s.parent is None)
    assert server.name.startswith("POST /v1/reconstructions")
    for name in MANUAL:
        assert by_name(exporter)[name].parent.span_id == server.context.span_id  # flat under the server span


def test_no_per_chunk_asgi_receive_or_send_spans(make, exporter):
    reconstruct(make())
    assert not [s for s in exporter.get_finished_spans() if s.name.endswith((" http receive", " http send"))]


def test_triton_infer_is_one_span_per_chunk(make, exporter):
    fake = FakeTritonClient(max_batch_size=1)  # 2 slices -> 2 chunks
    reconstruct(make(fake))
    assert [s.name for s in exporter.get_finished_spans()].count("triton.infer") == 2


def test_server_span_carries_request_id_and_log_line_matches_trace(make, exporter, capsys):
    r = reconstruct(make(), headers={"X-Request-ID": "req-123"})
    server = next(s for s in exporter.get_finished_spans() if s.parent is None)
    assert server.attributes["request_id"] == "req-123"
    log = next(
        json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{") and '"event": "request"' in x
    )
    assert log["trace_id"] == format(server.context.trace_id, "032x")
    assert log["span_id"] == format(server.context.span_id, "016x")
    assert r.headers["x-request-id"] == "req-123"


def test_incoming_traceparent_is_continued(make, exporter):
    tid = "0af7651916cd43dd8448eb211c80319c"
    reconstruct(make(), headers={"traceparent": f"00-{tid}-b7ad6b7169203331-01"})
    assert {format(s.context.trace_id, "032x") for s in exporter.get_finished_spans()} == {tid}


def test_triton_failure_marks_the_trace_failed(make, exporter):
    fake = FakeTritonClient(script=[unavailable(), unavailable(), unavailable()])
    assert reconstruct(make(fake)).status_code == 503
    spans = by_name(exporter)
    assert spans["triton.infer"].status.status_code == StatusCode.ERROR
    server = next(s for s in exporter.get_finished_spans() if s.parent is None)
    assert server.status.status_code == StatusCode.ERROR


def test_retries_are_recorded_on_the_chunk_span(make, exporter):
    fake = FakeTritonClient(script=[unavailable()])
    assert reconstruct(make(fake)).status_code == 200
    assert by_name(exporter)["triton.infer"].attributes["triton.retries"] == 1


def test_probes_and_metrics_are_not_traced(make, exporter):
    c = make()
    for path in ("/healthz", "/readyz", "/metrics"):
        c.get(path)
    assert exporter.get_finished_spans() == ()


def test_no_filename_or_key_in_any_span(make, exporter):
    reconstruct(make(), headers={"X-API-Key": "s3cret-key-value"})
    dump = json.dumps([dict(s.attributes) | {"name": s.name} for s in exporter.get_finished_spans()], default=str)
    assert "patient-7" not in dump and "s3cret" not in dump


def test_no_provider_is_a_noop(tmp_path, capsys):
    settings = Settings(env="dev", auth_disabled=True, tmp_dir=str(tmp_path), readiness_interval_s=30)
    with TestClient(create_app(settings, FakeTritonClient())) as c:
        assert reconstruct(c).status_code == 200
    log = next(json.loads(x) for x in capsys.readouterr().out.splitlines() if '"event": "request"' in x)
    assert log["trace_id"] is None


def test_grpc_client_forwards_traceparent_as_lowercase_metadata(exporter):
    seen = {}

    class Out:
        def as_numpy(self, name):
            return np.zeros((1, 1, 2, 2), "float32")

    class Stub:
        async def infer(self, *a, **kw):
            seen.update(kw)
            return Out()

    async def call(sp_holder):
        client = GrpcTritonClient("localhost:1", "m")  # the aio channel needs a running loop
        client._client = Stub()
        with TracerProvider().get_tracer("t").start_as_current_span("parent") as sp:
            sp_holder.append(sp)
            await client.infer(np.zeros((1, 1, 2, 2), "float32"), 1.0)

    holder = []
    asyncio.run(call(holder))
    sp = holder[0]
    ctx = sp.get_span_context()
    assert seen["headers"]["traceparent"].startswith(f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-")
    assert all(k == k.lower() for k in seen["headers"])
