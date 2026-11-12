import numpy as np
import pytest
from fastapi.testclient import TestClient

from fastmri_knee.gateway.app import create_app
from fastmri_knee.gateway.auth import hash_key, load_key_hashes
from fastmri_knee.gateway.auth import main as mint_key
from fastmri_knee.gateway.settings import Settings
from tests.unit.gateway.fakes import FakeTritonClient

KEY = "k-correct"
BODY = np.zeros((1, 1, 320, 320), "<f4").tobytes()
OCTET = {"content-type": "application/octet-stream"}


@pytest.fixture
def keys_file(tmp_path):
    f = tmp_path / "api_keys"
    f.write_text(f"# comment\n\n{hash_key('other')}\n{hash_key(KEY).upper()}\n")
    return f


@pytest.fixture
def client(keys_file):
    app = create_app(Settings(env="prod", api_keys_file=str(keys_file), readiness_interval_s=30), FakeTritonClient())
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_valid_key_accepted(client):
    r = client.post("/v1/infer", content=BODY, headers={**OCTET, "X-API-Key": KEY})
    assert r.status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": ""}, {"X-API-Key": "wrong"}, {"X-API-Key": KEY + " "}])
def test_missing_or_bad_key_401(client, headers):
    r = client.post("/v1/infer", content=BODY, headers={**OCTET, **headers})
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.headers["www-authenticate"] == "ApiKey"
    assert "X-Request-ID" in r.headers


@pytest.mark.parametrize("method,path", [("get", "/v1/model"), ("post", "/v1/reconstructions"), ("post", "/v1/infer")])
def test_every_v1_route_requires_key(client, method, path):
    assert getattr(client, method)(path).status_code == 401


def test_probes_stay_open(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code in (200, 503)


def test_auth_checked_before_body_validation(client):
    r = client.post("/v1/infer", content=b"x", headers={"content-type": "text/plain"})
    assert r.status_code == 401  # not 415


def test_bypass_outside_dev_fails_startup():
    with pytest.raises(ValueError, match="only allowed"):
        Settings(env="prod", auth_disabled=True)


def test_bypass_in_dev_skips_auth():
    app = create_app(Settings(env="dev", auth_disabled=True, readiness_interval_s=30), FakeTritonClient())
    with TestClient(app) as c:
        assert c.post("/v1/infer", content=BODY, headers=OCTET).status_code == 200


def test_missing_keys_file_fails_startup(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        create_app(Settings(env="prod", api_keys_file=str(tmp_path / "nope")), FakeTritonClient())


@pytest.mark.parametrize("content,match", [("", "no keys"), ("# only\n", "no keys"), ("notahash\n", "not a SHA-256")])
def test_bad_keys_file_fails_startup(tmp_path, content, match):
    f = tmp_path / "k"
    f.write_text(content)
    with pytest.raises(ValueError, match=match):
        load_key_hashes(str(f))


def test_mint_key_appends_hash_not_key(tmp_path, capsys):
    path = tmp_path / "sub" / "api_keys"
    assert mint_key([str(path)]) == 0
    assert mint_key([str(path)]) == 0
    key = capsys.readouterr().out.split()[-1]
    assert load_key_hashes(str(path))[1] == hash_key(key).encode()
    assert key not in path.read_text()
    assert oct(path.stat().st_mode & 0o777) == "0o600"
