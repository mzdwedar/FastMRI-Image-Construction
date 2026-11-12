import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from fastmri_knee.gateway.app import create_app
from fastmri_knee.gateway.readiness import MODEL_NOT_READY, TRITON_UNREACHABLE, ReadinessMonitor
from fastmri_knee.gateway.settings import Settings
from fastmri_knee.gateway.triton_client import ModelState, TritonError
from tests.unit.gateway.fakes import FakeTritonClient


def make_client(fake, interval=0.05):
    app = create_app(Settings(env="dev", readiness_interval_s=interval), triton=fake)
    return TestClient(app, raise_server_exceptions=False)


def wait_for(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_ready_when_model_ready():
    with make_client(FakeTritonClient()) as c:
        r = c.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "model_version": "1"}


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (TritonError("UNAVAILABLE"), TRITON_UNREACHABLE),
        (TritonError("DEADLINE_EXCEEDED"), TRITON_UNREACHABLE),
        (TritonError("NOT_FOUND"), MODEL_NOT_READY),
        (TritonError("INVALID_ARGUMENT"), MODEL_NOT_READY),
        (RuntimeError("boom"), TRITON_UNREACHABLE),
    ],
)
def test_not_ready_reasons(error, reason):
    fake = FakeTritonClient()
    fake.state = error
    with make_client(fake) as c:
        r = c.get("/readyz")
    assert r.status_code == 503
    assert r.json() == {"status": "not_ready", "reason": reason}
    assert "boom" not in r.text


def test_gateway_starts_and_stays_live_with_triton_down():
    fake = FakeTritonClient()
    fake.state = TritonError("UNAVAILABLE")
    with make_client(fake) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 503


def test_readyz_flips_both_ways_without_per_probe_calls():
    fake = FakeTritonClient()
    calls = 0
    original = fake.model_state

    async def counting(timeout_s):
        nonlocal calls
        calls += 1
        return await original(timeout_s)

    fake.model_state = counting
    with make_client(fake) as c:
        assert c.get("/readyz").status_code == 200
        fake.state = TritonError("UNAVAILABLE")
        assert wait_for(lambda: c.get("/readyz").status_code == 503)
        assert c.get("/readyz").json()["reason"] == TRITON_UNREACHABLE
        fake.state = TritonError("NOT_FOUND")
        assert wait_for(lambda: c.get("/readyz").json().get("reason") == MODEL_NOT_READY)
        fake.state = ModelState("2")
        assert wait_for(lambda: c.get("/readyz").status_code == 200)
        assert c.get("/readyz").json()["model_version"] == "2"
        before = calls
        for _ in range(50):
            c.get("/readyz")
        assert calls - before < 50  # probes read the cache


def test_default_interval_flips_within_10_seconds():
    assert Settings().readiness_interval_s * 3 < 10  # poll + stale window stays well inside the 10 s budget


def test_stale_cache_is_not_ready():
    async def run():
        fake = FakeTritonClient()
        m = ReadinessMonitor(fake, interval_s=0.02)
        await m.poll_once()  # polled once, loop not running
        assert m.current.ready
        await asyncio.sleep(0.1)  # > 3 intervals
        assert not m.current.ready
        assert m.current.reason == TRITON_UNREACHABLE
        assert m.model_version is None

    asyncio.run(run())


def test_model_endpoint():
    fake = FakeTritonClient(max_batch_size=16)
    fake.state = ModelState("3", "onnxruntime_onnx", 16, [{"name": "input"}], [{"name": "output"}])
    with make_client(fake) as c:
        r = c.get("/v1/model")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "3"
    assert body["model_name"] == "unet_model"
    assert body["max_batch_size"] == 16
    assert body["preprocess_version"]
    assert body["inputs"] == [{"name": "input"}]


def test_model_endpoint_503_when_not_ready():
    fake = FakeTritonClient()
    fake.state = TritonError("NOT_FOUND")
    with make_client(fake) as c:
        r = c.get("/v1/model")
    assert r.status_code == 503
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.headers["retry-after"] == "1"


def test_model_version_header_on_responses():
    with make_client(FakeTritonClient()) as c:
        assert c.get("/healthz").headers["X-Model-Version"] == "1"
        assert c.get("/nope").headers["X-Model-Version"] == "1"  # error responses too
        assert c.get("/readyz").headers["X-Model-Version"] == "1"


def test_model_version_header_absent_when_not_ready():
    fake = FakeTritonClient()
    fake.state = TritonError("UNAVAILABLE")
    with make_client(fake) as c:
        r = c.get("/healthz")
    assert "X-Model-Version" not in r.headers
    assert r.headers["X-Request-ID"]
