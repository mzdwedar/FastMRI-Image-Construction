import json

import numpy as np
from fastapi.testclient import TestClient

from fastmri_knee.gateway.app import create_app
from fastmri_knee.gateway.auth import hash_key
from fastmri_knee.gateway.settings import Settings
from tests.unit.gateway.fakes import FakeTritonClient

KEY = "s3cret-key-value"
SPEC_FIELDS = {
    "ts", "level", "event", "request_id", "trace_id", "span_id", "route", "method", "status", "duration_ms",
    "slices", "model_version", "client",
}  # fmt: skip


def lines(capsys):
    out = capsys.readouterr().out
    return [json.loads(x) for x in out.splitlines() if x.startswith("{")]


def access(capsys):
    return [x for x in lines(capsys) if x["event"] == "request"]


def make(tmp_path, **kw):
    keys = tmp_path / "api_keys"
    keys.write_text(hash_key(KEY) + "\n")
    settings = Settings(env="prod", api_keys_file=str(keys), readiness_interval_s=30, **kw)
    return TestClient(create_app(settings, FakeTritonClient()), raise_server_exceptions=False)


def post_infer(c, n=2, **headers):
    return c.post(
        "/v1/infer",
        content=np.zeros((n, 1, 320, 320), "<f4").tobytes(),
        headers={"content-type": "application/octet-stream", **headers},
    )


def test_one_line_per_request_with_spec_fields(tmp_path, capsys):
    with make(tmp_path) as c:
        lines(capsys)  # drop startup output
        r = post_infer(c, 2, **{"X-API-Key": KEY, "X-Request-ID": "req-123"})
    (rec,) = access(capsys)
    assert r.status_code == 200
    assert set(rec) == SPEC_FIELDS
    assert rec["request_id"] == "req-123" == r.headers["X-Request-ID"]
    assert (rec["route"], rec["method"], rec["status"], rec["slices"]) == ("/v1/infer", "POST", 200, 2)
    assert rec["model_version"] == "1" and rec["level"] == "info" and rec["duration_ms"] >= 0
    assert rec["client"] == hash_key(KEY)[:12]


def test_unauthorized_unmatched_and_unhandled_each_log_once(tmp_path, capsys):
    with make(tmp_path) as c:
        lines(capsys)
        r401 = post_infer(c, 1, **{"X-API-Key": "wrong-key"})
        r404 = c.get("/no/such/route/42", headers={"X-Request-ID": "gone"})
    recs = access(capsys)
    assert [x["status"] for x in recs] == [401, 404]
    assert recs[0]["client"] is None and recs[0]["request_id"] == r401.headers["X-Request-ID"]
    assert recs[1]["route"] == "unmatched" and recs[1]["request_id"] == "gone" == r404.headers["X-Request-ID"]


def test_unhandled_error_logs_500_access_line(tmp_path, capsys):
    app = create_app(
        Settings(env="dev", auth_disabled=True, readiness_interval_s=30), FakeTritonClient(script=[ValueError("boom")])
    )
    with TestClient(app, raise_server_exceptions=False) as c:
        lines(capsys)
        r = post_infer(c, 1)
    recs = access(capsys)
    assert r.status_code == 500 and len(recs) == 1 and recs[0]["status"] == 500
    assert recs[0]["request_id"] == r.headers["X-Request-ID"]


def test_probes_and_metrics_are_not_logged(tmp_path, capsys):
    with make(tmp_path) as c:
        lines(capsys)
        c.get("/healthz")
        c.get("/metrics")
    assert access(capsys) == []


def test_no_key_or_filename_in_output(tmp_path, capsys):
    with make(tmp_path) as c:
        c.post(
            "/v1/reconstructions",
            files={"file": ("patient_jane_doe.h5", b"not hdf5", "application/x-hdf5")},
            headers={"X-API-Key": KEY},
        )
        post_infer(c, 1, **{"X-API-Key": KEY})
    out = capsys.readouterr().out
    assert out
    assert KEY not in out and hash_key(KEY) not in out and "patient_jane_doe" not in out
