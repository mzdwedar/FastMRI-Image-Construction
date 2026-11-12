"""Smoke: a synthetic volume goes in through the gateway and a reconstruction comes out of the real Triton."""

import io

import h5py
import numpy as np
import pytest

from tests.fixtures.synthetic import make_kspace, write_h5

pytestmark = pytest.mark.e2e

SLICES = 3


def test_requires_an_api_key(anonymous, gateway):
    assert anonymous.get("/v1/model").status_code == 401


def test_model_endpoint(gateway):
    body = gateway.get("/v1/model").json()
    assert body["version"] == "1"
    assert body["platform"] == "onnxruntime_onnx"


def test_reconstruct_a_synthetic_volume(gateway, tmp_path):
    path = write_h5(tmp_path / "vol.h5", make_kspace(slices=SLICES, seed=7))
    with path.open("rb") as f:
        r = gateway.post("/v1/reconstructions", files={"file": ("vol.h5", f, "application/x-hdf5")})
    assert r.status_code == 200, r.text
    assert r.headers["X-Model-Version"] == "1"
    with h5py.File(io.BytesIO(r.content)) as out:
        recon = out["reconstruction"][:]
        assert recon.shape == (SLICES, 320, 320)
        assert recon.dtype == np.float32
        assert np.isfinite(recon).all()
        assert out.attrs["model_version"] == "1"
        assert np.isfinite(out.attrs["scale"]).all()


def test_bad_upload_is_a_clean_4xx(gateway):
    r = gateway.post("/v1/reconstructions", files={"file": ("vol.h5", b"not an hdf5 file", "application/x-hdf5")})
    assert 400 <= r.status_code < 500
