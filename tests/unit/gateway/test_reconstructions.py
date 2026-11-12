import asyncio
import io
import tracemalloc

import h5py
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fastmri_knee.gateway.app import create_app
from fastmri_knee.gateway.settings import Settings
from fastmri_knee.gateway.undersample import random_cartesian_mask
from fastmri_knee.preprocess import kspace_to_image, kspace_to_image_scaled
from tests.fixtures.synthetic import make_kspace
from tests.unit.gateway.fakes import FakeTritonClient, unavailable


def h5_bytes(kspace, name="kspace") -> bytes:
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset(name, data=kspace)
    return buf.getvalue()


@pytest.fixture
def tmp_dir(tmp_path):
    d = tmp_path / "uploads"
    d.mkdir()
    return d


@pytest.fixture
def fake():
    return FakeTritonClient(script=[])


def client_for(fake, tmp_dir, **settings):
    app = create_app(
        Settings(env="dev", auth_disabled=True, tmp_dir=str(tmp_dir), readiness_interval_s=30, **settings), triton=fake
    )
    return TestClient(app, raise_server_exceptions=False)


def upload(c, kspace=None, data=None, params=None, field="file", **kw):
    data = data if data is not None else h5_bytes(kspace)
    return c.post("/v1/reconstructions", files={field: ("x.h5", data, "application/x-hdf5")}, params=params, **kw)


def read_response(r):
    with h5py.File(io.BytesIO(r.content), "r") as f:
        return f["reconstruction"][...], dict(f.attrs)


def identity(batch):
    return batch


def test_roundtrip_output_attrs_and_scale(tmp_dir):
    fake = FakeTritonClient(script=[identity])
    ks = make_kspace(3)
    ks[1] *= 5.0  # slices with different intensity -> different scales
    with client_for(fake, tmp_dir) as c:
        r = upload(c, ks)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/x-hdf5"
    recon, attrs = read_response(r)
    assert recon.shape == (3, 320, 320) and recon.dtype == np.float32
    assert attrs["model_version"] == "1"
    assert attrs["preprocess_version"] == "1"
    assert attrs["request_id"] == r.headers["X-Request-ID"]
    expected = [kspace_to_image_scaled(s) for s in ks]
    np.testing.assert_allclose(attrs["scale"], [e[1] for e in expected], rtol=1e-6)
    assert attrs["scale"][1] > 4 * attrs["scale"][0]
    for i, (img, _scale) in enumerate(expected):
        np.testing.assert_array_equal(recon[i], img[0])
        assert recon[i].max() == pytest.approx(1.0, abs=1e-5)  # per-slice max normalization
    assert list(tmp_dir.iterdir()) == []


def test_minimum_size_320x320_accepted(fake, tmp_dir):
    with client_for(fake, tmp_dir) as c:
        assert upload(c, make_kspace(1, 320, 320)).status_code == 200


def test_acceleration_demo_changes_input_deterministically(tmp_dir):
    seen = []

    def capture(batch):
        seen.append(batch.copy())
        return batch

    fake = FakeTritonClient(script=[capture, capture, capture])
    ks = make_kspace(2)
    with client_for(fake, tmp_dir) as c:
        assert upload(c, ks).status_code == 200
        assert upload(c, ks, params={"acceleration": 4}).status_code == 200
        assert upload(c, ks, params={"acceleration": 4}).status_code == 200
    plain, accel, accel2 = seen
    assert not np.allclose(plain, accel)
    np.testing.assert_array_equal(accel, accel2)
    mask = random_cartesian_mask(ks.shape[-1], 4)
    np.testing.assert_array_equal(accel[0], kspace_to_image(ks[0], mask))


@pytest.mark.parametrize("value", ["1", "0", "17", "-4", "abc"])
def test_bad_acceleration_422_before_reading_body(fake, tmp_dir, value):
    with client_for(fake, tmp_dir) as c:
        r = upload(c, make_kspace(1), params={"acceleration": value})
    assert r.status_code == 422
    assert r.headers["content-type"].startswith("application/problem+json")
    assert fake.calls == [] and list(tmp_dir.iterdir()) == []


def bad_files():
    k = make_kspace(1)
    return {
        "not_hdf5": b"this is not hdf5",
        "missing_kspace": h5_bytes(k, name="other"),
        "rank2": h5_bytes(k[0]),
        "rank4": h5_bytes(k[None]),
        "float32": h5_bytes(np.abs(k)),
        "complex128": h5_bytes(k.astype(np.complex128)),
        "too_small": h5_bytes(make_kspace(1, 100, 100)),
        "too_large": h5_bytes(np.zeros((1, 320, 1100), np.complex64)),
        "no_slices": h5_bytes(np.zeros((0, 320, 320), np.complex64)),
    }


@pytest.mark.parametrize("name", list(bad_files()))
def test_malformed_input_400(fake, tmp_dir, name):
    with client_for(fake, tmp_dir) as c:
        r = upload(c, data=bad_files()[name])
    assert r.status_code == 400, r.text
    assert r.headers["content-type"].startswith("application/problem+json")
    assert fake.calls == []
    assert list(tmp_dir.iterdir()) == []


def test_missing_file_field_400(fake, tmp_dir):
    with client_for(fake, tmp_dir) as c:
        r = upload(c, make_kspace(1), field="other")
    assert r.status_code == 400
    assert list(tmp_dir.iterdir()) == []


def test_broken_multipart_400(fake, tmp_dir):
    with client_for(fake, tmp_dir) as c:
        r = c.post(
            "/v1/reconstructions", content=b"garbage", headers={"content-type": "multipart/form-data; boundary=zzz"}
        )
    assert r.status_code == 400
    assert list(tmp_dir.iterdir()) == []


@pytest.mark.parametrize("content_type", ["application/json", "application/octet-stream", "multipart/form-data"])
def test_wrong_content_type_415(fake, tmp_dir, content_type):  # the last has no boundary
    with client_for(fake, tmp_dir) as c:
        r = c.post("/v1/reconstructions", content=b"{}", headers={"content-type": content_type})
    assert r.status_code == 415
    assert list(tmp_dir.iterdir()) == []


def test_too_many_slices_413(fake, tmp_dir):
    with client_for(fake, tmp_dir, max_slices=2) as c:
        r = upload(c, make_kspace(3, 320, 320))
    assert r.status_code == 413
    assert fake.calls == [] and list(tmp_dir.iterdir()) == []


def test_upload_over_limit_413(fake, tmp_dir):
    with client_for(fake, tmp_dir, max_upload_mb=1) as c:
        r = upload(c, data=b"\x00" * (2 << 20))
    assert r.status_code == 413
    assert list(tmp_dir.iterdir()) == []


def test_declared_content_length_over_limit_rejected_without_reading(fake, tmp_dir):
    with client_for(fake, tmp_dir, max_upload_mb=1) as c:
        r = c.post(
            "/v1/reconstructions",
            content=b"x",
            headers={"content-type": "multipart/form-data; boundary=zzz", "content-length": str(10 << 20)},
        )
    assert r.status_code in (413, 400)  # the transport may reject the mismatched length itself


def test_300mb_stream_rejected_without_buffering(fake, tmp_dir):
    app = create_app(
        Settings(env="dev", auth_disabled=True, tmp_dir=str(tmp_dir), max_upload_mb=64, readiness_interval_s=30),
        triton=fake,
    )
    chunk = b"\x00" * (1 << 20)
    boundary = "bnd"
    sent = 0

    async def body():
        nonlocal sent
        yield (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="x.h5"\r\n\r\n').encode()
        for _ in range(300):
            sent += 1
            yield chunk
        yield f"\r\n--{boundary}--\r\n".encode()

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=transport, base_url="http://t") as c,
        ):
            return await c.post(
                "/v1/reconstructions",
                content=body(),
                headers={"content-type": f"multipart/form-data; boundary={boundary}"},
            )

    tracemalloc.start()
    r = asyncio.run(go())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert r.status_code == 413
    assert sent < 80  # stopped reading shortly after the 64 MB limit instead of consuming all 300
    assert peak < 30 << 20  # never buffered the body
    assert list(tmp_dir.iterdir()) == []


def test_temp_file_removed_on_client_disconnect(fake, tmp_dir):
    app = create_app(
        Settings(env="dev", auth_disabled=True, tmp_dir=str(tmp_dir), readiness_interval_s=30), triton=fake
    )
    head = b'--b\r\nContent-Disposition: form-data; name="file"; filename="x.h5"\r\n\r\n' + b"\x00" * 1000
    messages = iter(
        [
            {"type": "http.request", "body": head, "more_body": True},
            {"type": "http.disconnect"},
        ]
    )
    seen_in_dir = []

    async def receive():
        m = next(messages)
        if m["type"] == "http.disconnect":
            seen_in_dir.extend(tmp_dir.iterdir())  # the partial upload exists at disconnect time
        return m

    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/reconstructions",
        "raw_path": b"/v1/reconstructions",
        "query_string": b"",
        "headers": [(b"content-type", b"multipart/form-data; boundary=b")],
        "client": ("t", 1),
        "server": ("t", 80),
        "scheme": "http",
        "root_path": "",
    }

    async def go():
        async with app.router.lifespan_context(app):
            scope["app"] = app
            await app(scope, receive, send)

    asyncio.run(go())
    assert len(seen_in_dir) == 1
    assert list(tmp_dir.iterdir()) == []
    assert fake.calls == []


def test_triton_unavailable_503_and_cleanup(tmp_dir):
    fake = FakeTritonClient(script=[unavailable()] * 3)
    with client_for(fake, tmp_dir) as c:
        r = upload(c, make_kspace(1))
    assert r.status_code == 503
    assert r.headers["retry-after"] == "1"
    assert list(tmp_dir.iterdir()) == []


def test_non_finite_model_output_500(tmp_dir):
    fake = FakeTritonClient(script=[lambda b: np.full_like(b, np.nan)])
    with client_for(fake, tmp_dir) as c:
        r = upload(c, make_kspace(1))
    assert r.status_code == 500
    assert "nan" not in r.text.lower()


def test_internal_error_hides_details(fake, tmp_dir, monkeypatch):
    monkeypatch.setattr(
        "fastmri_knee.gateway.routes.preprocess_volume", lambda *a: (_ for _ in ()).throw(RuntimeError("secret"))
    )
    with client_for(fake, tmp_dir) as c:
        r = upload(c, make_kspace(1))
    assert r.status_code == 500 and "secret" not in r.text
    assert list(tmp_dir.iterdir()) == []


def test_openapi_documents_acceleration_as_demo(fake, tmp_dir):
    with client_for(fake, tmp_dir) as c:
        spec = c.get("/openapi.json").json()
    params = spec["paths"]["/v1/reconstructions"]["post"]["parameters"]
    assert "DEMO" in next(p for p in params if p["name"] == "acceleration")["description"]


@pytest.mark.parametrize("acc", [2, 4, 8, 16])
def test_mask_density_center_and_determinism(acc):
    w = 372
    m = random_cartesian_mask(w, acc)
    assert m.shape == (w,) and m.dtype == np.float32 and set(np.unique(m)) <= {0.0, 1.0}
    assert m.mean() == pytest.approx(1 / acc, abs=0.06)
    num_low = round(w * 0.32 / acc)
    pad = (w - num_low + 1) // 2
    assert m[pad : pad + num_low].all()
    np.testing.assert_array_equal(m, random_cartesian_mask(w, acc))
    assert not np.array_equal(m, random_cartesian_mask(w, acc, seed=1))


def test_mask_rejects_acceleration_below_2():
    with pytest.raises(ValueError):
        random_cartesian_mask(320, 1)


def test_scaled_preprocess_parity_and_scale():
    ks = make_kspace(1)[0]
    img, scale = kspace_to_image_scaled(ks)
    np.testing.assert_array_equal(img, kspace_to_image(ks))
    full = np.abs(np.fft.ifft2(ks, norm="ortho"))
    assert scale == pytest.approx(float(full.max()), rel=1e-6)
    # restoring intensity inside the crop
    cropped = full[160:480, 26:346]
    np.testing.assert_allclose(img[0] * scale, cropped, atol=1e-4 * scale)
