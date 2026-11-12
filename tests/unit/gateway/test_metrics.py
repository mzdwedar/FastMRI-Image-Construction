import re
import subprocess
import sys

import numpy as np
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from fastmri_knee.gateway import telemetry
from fastmri_knee.gateway.app import create_app
from fastmri_knee.gateway.settings import Settings
from fastmri_knee.gateway.triton_client import ModelState
from tests.unit.gateway.fakes import FakeTritonClient, unavailable

SPEC_METRICS = {  # name -> (type, label names) from SPEC-observability
    "gateway_requests": ("counter", {"route", "method", "status_class"}),
    "gateway_request_duration_seconds": ("histogram", {"route"}),
    "gateway_inflight_requests": ("gauge", set()),
    "gateway_rejected": ("counter", {"reason"}),
    "gateway_upload_bytes": ("histogram", set()),
    "gateway_slices": ("counter", {"route"}),
    "gateway_stage_duration_seconds": ("histogram", {"stage"}),
    "gateway_triton_errors": ("counter", {"code"}),
    "gateway_model_ready": ("gauge", set()),
    "gateway_model_info": ("gauge", {"version", "preprocess_version"}),
}


def value(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def client_for(fake=None, **settings):
    app = create_app(Settings(env="dev", auth_disabled=True, **settings), triton=fake or FakeTritonClient())
    return TestClient(app, raise_server_exceptions=False)


def infer(c, n=1):
    return c.post(
        "/v1/infer",
        content=np.zeros((n, 1, 320, 320), "<f4").tobytes(),
        headers={"content-type": "application/octet-stream"},
    )


def test_registered_names_types_and_labels_match_spec():
    found = {m.name: m for m in REGISTRY.collect() if m.name.startswith("gateway_")}
    assert set(found) == set(SPEC_METRICS)
    for name, (kind, labels) in SPEC_METRICS.items():
        assert found[name].type == kind, name
        sample_labels = {k for s in found[name].samples for k in s.labels} - {"le"}
        assert sample_labels <= labels, name


def test_request_counted_by_route_template_not_raw_path():
    with client_for() as c:
        before = value("gateway_requests_total", route="/v1/infer", method="POST", status_class="2xx")
        assert infer(c, 3).status_code == 200
        assert c.get("/definitely/not/a/route/123").status_code == 404
        assert c.request("FROB", "/healthz").status_code == 405
    assert value("gateway_requests_total", route="/v1/infer", method="POST", status_class="2xx") == before + 1
    assert value("gateway_requests_total", route="unmatched", method="GET", status_class="4xx") >= 1
    assert value("gateway_requests_total", route="/healthz", method="OTHER", status_class="4xx") >= 1
    assert value("gateway_request_duration_seconds_count", route="/v1/infer") >= 1
    routes = {s.labels["route"] for m in REGISTRY.collect() if m.name == "gateway_requests" for s in m.samples}
    assert not any("123" in r for r in routes)


def test_slices_upload_bytes_and_stages():
    with client_for() as c:
        s0, b0 = value("gateway_slices_total", route="/v1/infer"), value("gateway_upload_bytes_sum")
        t0, e0 = (
            value("gateway_stage_duration_seconds_count", stage="triton"),
            value("gateway_stage_duration_seconds_count", stage="encode"),
        )
        infer(c, 4)
    assert value("gateway_slices_total", route="/v1/infer") == s0 + 4
    assert value("gateway_upload_bytes_sum") == b0 + 4 * 320 * 320 * 4
    assert value("gateway_stage_duration_seconds_count", stage="triton") == t0 + 1
    assert value("gateway_stage_duration_seconds_count", stage="encode") == e0 + 1


def test_rejections_by_reason():
    before = {r: value("gateway_rejected_total", reason=r) for r in telemetry.REJECT_REASONS}
    with client_for(max_inflight=1) as c:
        assert c.app.state.limiter.try_acquire()  # the only slot is taken
        assert infer(c).status_code == 503  # overloaded
        c.app.state.limiter.release()
    with client_for(max_slices=1) as c:
        assert infer(c, 2).status_code == 413  # too_large
        assert c.post("/v1/infer", content=b"x", headers={"content-type": "text/plain"}).status_code == 415  # invalid
        assert (
            c.post("/v1/infer", content=b"x", headers={"content-type": "application/octet-stream"}).status_code == 400
        )
    with TestClient(create_app(Settings(env="dev", auth_disabled=True)), raise_server_exceptions=False) as c:
        assert c.post("/v1/reconstructions?acceleration=1").status_code == 422  # invalid
    after = {r: value("gateway_rejected_total", reason=r) for r in telemetry.REJECT_REASONS}
    assert after["overloaded"] == before["overloaded"] + 1
    assert after["too_large"] == before["too_large"] + 1
    assert after["invalid"] >= before["invalid"] + 3
    assert after["unauthorized"] == before["unauthorized"]


def test_triton_errors_by_code_and_invalid_output():
    u0 = value("gateway_triton_errors_total", code="UNAVAILABLE")
    i0 = value("gateway_triton_errors_total", code=telemetry.INVALID_OUTPUT)
    with client_for(FakeTritonClient(script=[unavailable()] * 3)) as c:
        assert infer(c).status_code == 503
    with client_for(FakeTritonClient(script=[lambda b: np.full_like(b, np.nan)])) as c:
        assert infer(c).status_code == 500
    assert value("gateway_triton_errors_total", code="UNAVAILABLE") == u0 + 1
    assert value("gateway_triton_errors_total", code=telemetry.INVALID_OUTPUT) == i0 + 1


def test_unknown_error_code_is_bucketed():
    telemetry.triton_error("SOMETHING_INVENTED")
    assert value("gateway_triton_errors_total", code="UNKNOWN") >= 1
    assert value("gateway_triton_errors_total", code="SOMETHING_INVENTED") == 0


def test_inflight_gauge_returns_to_baseline():
    with client_for() as c:
        base = value("gateway_inflight_requests")
        infer(c)
    assert value("gateway_inflight_requests") == base


def test_model_gauges_follow_readiness_and_version_change():
    fake = FakeTritonClient()
    with client_for(fake) as c:
        assert value("gateway_model_ready") == 1
        assert value("gateway_model_info", version="1", preprocess_version="1") == 1
        fake.state = ModelState("2", "onnxruntime_onnx", 16)
        c.portal.call(c.app.state.readiness.poll_once)
        assert value("gateway_model_info", version="2", preprocess_version="1") == 1
        assert value("gateway_model_info", version="1", preprocess_version="1") == 0
        fake.state = RuntimeError("down")
        c.portal.call(c.app.state.readiness.poll_once)
        assert value("gateway_model_ready") == 0


def test_metrics_endpoint_exposition_and_not_self_counted():
    with client_for() as c:
        infer(c)
        before = value("gateway_requests_total", route="/metrics", method="GET", status_class="2xx")
        r = c.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "gateway_requests_total" in r.text and "gateway_model_info" in r.text
    assert value("gateway_requests_total", route="/metrics", method="GET", status_class="2xx") == before
    assert "/metrics" not in c.get("/openapi.json").text


WORKER = """
import sys
from fastmri_knee.gateway import telemetry
n = int(sys.argv[1])
for _ in range(n):
    telemetry.REQUESTS.labels("/v1/infer", "POST", "2xx").inc()
    telemetry.REQUEST_DURATION.labels("/v1/infer").observe(0.02)
telemetry.SLICES.labels("/v1/infer").inc(n * 2)
telemetry.INFLIGHT.inc()  # a live worker's gauge counts; dead ones are dropped by gunicorn's child_exit
"""


def test_counts_aggregate_across_two_worker_processes(tmp_path):
    import os

    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}
    procs = [subprocess.Popen([sys.executable, "-c", WORKER, str(n)], env=env) for n in (3, 5)]
    assert [p.wait(timeout=60) for p in procs] == [0, 0]
    old = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(tmp_path)
    try:
        body, _ = telemetry.render()
    finally:
        if old is None:
            del os.environ["PROMETHEUS_MULTIPROC_DIR"]
        else:
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = old
    text = body.decode()
    assert 'gateway_requests_total{method="POST",route="/v1/infer",status_class="2xx"} 8.0' in text
    assert 'gateway_request_duration_seconds_count{route="/v1/infer"} 8.0' in text
    assert 'gateway_slices_total{route="/v1/infer"} 16.0' in text
    assert re.search(r"^gateway_request_duration_seconds_bucket\{le=\"0.025\",route=\"/v1/infer\"\} 8.0$", text, re.M)
