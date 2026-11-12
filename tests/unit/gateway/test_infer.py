import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fastmri_knee.gateway.app import create_app
from fastmri_knee.gateway.settings import Settings
from fastmri_knee.gateway.triton_client import TritonError, infer_volume
from tests.unit.gateway.fakes import FakeTritonClient, unavailable

SLICE = 320 * 320


def make_client(fake, **settings):
    app = create_app(Settings(env="dev", **settings), triton=fake)
    return TestClient(app, raise_server_exceptions=False)


def post(client, n, content_type="application/octet-stream", data=None):
    body = data if data is not None else np.zeros((n, 1, 320, 320), "<f4").tobytes()
    return client.post("/v1/infer", content=body, headers={"content-type": content_type})


@pytest.mark.parametrize("n", [1, 16, 17, 64])
def test_chunking_and_roundtrip(n):
    fake = FakeTritonClient(max_batch_size=16)
    with make_client(fake) as c:
        vol = np.arange(n, dtype="<f4").reshape(n, 1, 1, 1) * np.ones((n, 1, 320, 320), "<f4")
        r = post(c, n, data=vol.tobytes())
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    out = np.frombuffer(r.content, "<f4").reshape(n, 1, 320, 320)
    np.testing.assert_array_equal(out, vol + 1.0)  # order preserved across chunks
    assert sorted(fake.calls) == sorted([16] * (n // 16) + ([n % 16] if n % 16 else []))
    assert fake.closed


def test_retry_unavailable_then_success():
    fake = FakeTritonClient(script=[unavailable(), unavailable()])
    with make_client(fake) as c:
        assert post(c, 1).status_code == 200
    assert fake.calls == [1, 1, 1]


def test_unavailable_exhausts_retries_503():
    fake = FakeTritonClient(script=[unavailable()] * 3)
    with make_client(fake) as c:
        r = post(c, 1)
    assert r.status_code == 503
    assert r.headers["retry-after"] == "1"
    assert r.headers["content-type"].startswith("application/problem+json")
    assert fake.calls == [1, 1, 1]  # 1 try + 2 retries


@pytest.mark.parametrize("code", ["INVALID_ARGUMENT", "INTERNAL", "UNKNOWN"])
def test_other_codes_not_retried_and_hidden(code):
    fake = FakeTritonClient(script=[TritonError(code, "secret detail")])
    with make_client(fake) as c:
        r = post(c, 1)
    assert r.status_code == 500
    assert "secret" not in r.text and code not in r.text
    assert fake.calls == [1]


def test_deadline_exceeded_is_504_and_not_retried():
    fake = FakeTritonClient(script=[TritonError("DEADLINE_EXCEEDED")])
    with make_client(fake) as c:
        r = post(c, 1)
    assert r.status_code == 504
    assert fake.calls == [1]


def test_slow_triton_hits_overall_deadline_504():
    fake = FakeTritonClient(delay_s=1.0)
    with make_client(fake, triton_timeout_s=0.1) as c:
        r = post(c, 1)
    assert r.status_code == 504


def test_non_finite_output_is_500_and_not_returned():
    fake = FakeTritonClient(script=[lambda b: np.full_like(b, np.nan)])
    with make_client(fake) as c:
        r = post(c, 1)
    assert r.status_code == 500
    assert "nan" not in r.text.lower()


def test_wrong_output_shape_is_500():
    fake = FakeTritonClient(script=[lambda b: b[:, :, :10]])
    with make_client(fake) as c:
        assert post(c, 1).status_code == 500


def test_too_many_slices_413_without_calling_triton():
    fake = FakeTritonClient()
    with make_client(fake) as c:
        r = post(c, 65)
    assert r.status_code == 413
    assert fake.calls == []


def test_wrong_content_type_415():
    with make_client(FakeTritonClient()) as c:
        assert post(c, 1, content_type="application/json").status_code == 415


@pytest.mark.parametrize("data", [b"", b"\x00" * (SLICE * 4 + 1)])
def test_malformed_length_400(data):
    with make_client(FakeTritonClient()) as c:
        assert post(c, 0, data=data).status_code == 400


def test_non_finite_input_400():
    bad = np.zeros((1, 1, 320, 320), "<f4")
    bad[0, 0, 0, 0] = np.inf
    with make_client(FakeTritonClient()) as c:
        assert post(c, 1, data=bad.tobytes()).status_code == 400


def test_nobatch_model_sends_one_slice_per_call():
    fake = FakeTritonClient(max_batch_size=0)
    out = asyncio.run(infer_volume(fake, np.zeros((3, 1, 4, 4), "f4"), 5))
    assert out.shape == (3, 1, 4, 4)
    assert fake.calls == [1, 1, 1]
