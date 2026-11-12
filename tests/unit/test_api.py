import io

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

SLICE = (1, 320, 320)


@pytest.fixture(scope="module")
def client(checkpoint):
    import os

    os.environ["CHECKPOINT_PATH"] = str(checkpoint)
    from deploy.services.fastapi.main import app

    with TestClient(app) as c:  # runs the lifespan: model load + hardware sampler
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("n", [1, 4])
def test_infer_returns_one_reconstruction_per_slice(client, n):
    x = np.random.default_rng(0).random((n, *SLICE), dtype=np.float32)
    resp = client.post("/infer", content=x.tobytes(), headers={"Content-Type": "application/octet-stream"})
    assert resp.status_code == 200
    out = np.frombuffer(resp.content, dtype=np.float32).reshape(n, *SLICE)
    assert np.isfinite(out).all()


def test_infer_matches_direct_forward(client):
    x = np.random.default_rng(1).random((2, *SLICE), dtype=np.float32)
    resp = client.post("/infer", content=x.tobytes())
    served = np.frombuffer(resp.content, dtype=np.float32).reshape(2, *SLICE)
    direct = client.app.state.model.infer(x)
    np.testing.assert_allclose(served, direct, atol=1e-6)


@pytest.mark.parametrize("body", [b"", b"\x00" * 100])
def test_infer_rejects_malformed_body(client, body):
    assert client.post("/infer", content=body).status_code == 400


def test_predict_returns_volume(client, h5_file):
    with open(h5_file, "rb") as f:
        resp = client.post("/predict", files={"file": (h5_file.name, f, "application/octet-stream")})
    assert resp.status_code == 200
    with h5py.File(io.BytesIO(resp.content), "r") as out:
        assert out["pred"].shape == (3, *SLICE)


def test_predict_rejects_non_h5(client):
    resp = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_errors_do_not_leak_tracebacks(client, tmp_path):
    bad = tmp_path / "broken.h5"
    bad.write_bytes(b"not really hdf5")
    with open(bad, "rb") as f:
        resp = client.post("/predict", files={"file": (bad.name, f, "application/octet-stream")})
    assert resp.status_code == 400
    assert "Traceback" not in resp.text


def test_evaluate_reports_image_metrics(client, h5_file):
    with open(h5_file, "rb") as f:
        resp = client.post("/evaluate", files=[("files", (h5_file.name, f, "application/octet-stream"))])
    assert resp.status_code == 200
    metrics = resp.json()["metrics"][0]
    assert {"psnr_test", "ssim_test", "nmse_test"} <= metrics.keys()


def test_metrics_exposes_inference_series(client):
    text = client.get("/metrics").text
    for name in (
        "inference_requests_total",
        "inference_latency_seconds_bucket",
        "inference_batch_size_bucket",
        "process_rss_bytes",
        "http_requests_total",
    ):
        assert name in text, name
