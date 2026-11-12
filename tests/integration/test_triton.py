"""Triton on its own: strict readiness, parity with ONNX Runtime, and load shedding under overload."""

from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import onnxruntime as ort
import pytest
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from tests.integration.conftest import MODEL

pytestmark = pytest.mark.integration


def test_empty_repository_is_live_but_model_not_ready(empty_triton):
    """Startup on an empty repo succeeds. Note Triton 24.08 keeps /v2/health/ready at 200 here (a model that was
    never found is not tracked as failed), so the per-model endpoint is what gates readiness."""
    assert httpx.get(f"{empty_triton}/v2/health/live", timeout=5).status_code == 200
    r = httpx.get(f"{empty_triton}/v2/models/{MODEL}/ready", timeout=5)
    assert r.status_code != 200, "an empty repository must not report the model ready"
    assert httpx.post(f"{empty_triton}/v2/repository/index", timeout=5).json() == []


def _infer(url: str, x: np.ndarray, timeout: float = 60) -> np.ndarray:
    with grpcclient.InferenceServerClient(url) as c:
        inp = grpcclient.InferInput("input", list(x.shape), "FP32")
        inp.set_data_from_numpy(x)
        return c.infer(MODEL, [inp], client_timeout=timeout).as_numpy("output")


def test_output_matches_onnx_runtime(gateway_triton):
    x = np.random.default_rng(1).random((4, 1, 320, 320), dtype=np.float32)
    session = ort.InferenceSession(str(gateway_triton.onnx), providers=["CPUExecutionProvider"])
    expected = session.run(["output"], {"input": x})[0]
    np.testing.assert_allclose(_infer(gateway_triton.grpc_url, x), expected, atol=1e-5)


def test_overload_is_rejected_and_server_stays_live(gateway_triton):
    """512 concurrent requests exceed max_queue_size 256: some are rejected, the server keeps serving."""
    x = np.random.default_rng(2).random((16, 1, 320, 320), dtype=np.float32)

    def one(_: int) -> str:
        try:
            _infer(gateway_triton.grpc_url, x, timeout=120)
            return "ok"
        except InferenceServerException as e:
            return "rejected" if "UNAVAILABLE" in str(e) else f"error: {e}"

    with ThreadPoolExecutor(max_workers=512) as pool:
        outcomes = list(pool.map(one, range(512)))

    errors = [o for o in outcomes if o.startswith("error")]
    assert not errors, errors[:3]
    assert "rejected" in outcomes, "no request was shed under 512 concurrent requests"
    assert "ok" in outcomes
    http = f"http://127.0.0.1:{gateway_triton.http_port}"
    assert httpx.get(f"{http}/v2/health/live", timeout=5).status_code == 200
    # and it recovers: a fresh request succeeds once the queue drains
    assert _infer(gateway_triton.grpc_url, x[:1]).shape == (1, 1, 320, 320)
