"""The gateway against a real Triton over gRPC: parity with local ONNX Runtime, readiness, and Triton going away."""

import io
import time

import h5py
import numpy as np
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient

from fastmri_knee.gateway.app import create_app
from fastmri_knee.gateway.settings import Settings
from tests.fixtures.synthetic import make_kspace, write_h5

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client(gateway_triton):
    settings = Settings(
        env="dev", auth_disabled=True, triton_url=gateway_triton.grpc_url, readiness_interval_s=0.25,
        triton_timeout_s=10,
    )  # fmt: skip
    with TestClient(create_app(settings)) as c:
        wait_for(c, 200)
        yield c


def wait_for(client: TestClient, status: int, timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while client.get("/readyz").status_code != status:
        assert time.monotonic() < deadline, f"/readyz did not reach {status} in {timeout_s}s"
        time.sleep(0.1)


def reference(onnx, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    return session.run(["output"], {"input": x})[0]


def test_infer_matches_local_onnx_runtime(client, gateway_triton):
    x = np.random.default_rng(0).random((20, 1, 320, 320), dtype=np.float32)  # > max_batch_size 16: two chunks
    r = client.post("/v1/infer", content=x.tobytes(), headers={"content-type": "application/octet-stream"})
    assert r.status_code == 200, r.text
    assert r.headers["X-Model-Version"] == "1"
    got = np.frombuffer(r.content, dtype="<f4").reshape(x.shape)
    np.testing.assert_allclose(got, reference(gateway_triton.onnx, x), atol=1e-5)


def test_reconstruction_matches_local_onnx_runtime(client, gateway_triton, tmp_path):
    from fastmri_knee.gateway.routes import preprocess_volume

    kspace = make_kspace(slices=3, seed=1)
    path = write_h5(tmp_path / "vol.h5", kspace)
    with path.open("rb") as f:
        r = client.post("/v1/reconstructions", files={"file": ("vol.h5", f, "application/x-hdf5")})
    assert r.status_code == 200, r.text
    x, scale = preprocess_volume(kspace.astype(np.complex64), None)
    with h5py.File(io.BytesIO(r.content)) as out:
        np.testing.assert_allclose(out["reconstruction"][:], reference(gateway_triton.onnx, x)[:, 0], atol=1e-5)
        np.testing.assert_allclose(out.attrs["scale"], scale, rtol=1e-6)
        assert out.attrs["model_version"] == "1"


def test_model_endpoint_reports_triton_state(client):
    body = client.get("/v1/model").json()
    assert body["version"] == "1"
    assert body["platform"] == "onnxruntime_onnx"
    assert body["max_batch_size"] == 16


def test_triton_stop_and_restart(client, gateway_triton):
    """Run last: the gateway stays live and turns unready/503 while Triton is down, and recovers by itself."""
    x = np.zeros((1, 1, 320, 320), dtype=np.float32)
    gateway_triton.stop()
    try:
        wait_for(client, 503)
        assert client.get("/healthz").status_code == 200
        r = client.post("/v1/infer", content=x.tobytes(), headers={"content-type": "application/octet-stream"})
        assert r.status_code == 503 and r.headers["Retry-After"] == "1"
        assert r.headers["content-type"].startswith("application/problem+json")
    finally:
        gateway_triton.start()
    wait_for(client, 200)
    r = client.post("/v1/infer", content=x.tobytes(), headers={"content-type": "application/octet-stream"})
    assert r.status_code == 200
