import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fastmri_knee.gateway.app import create_app
from fastmri_knee.gateway.errors import ProblemError
from fastmri_knee.gateway.settings import Settings


@pytest.fixture
def app():
    app = create_app(Settings(env="dev"))

    @app.get("/boom")
    async def boom():
        raise RuntimeError("secret internal detail")

    @app.get("/teapot")
    async def teapot():
        raise ProblemError(503, "busy", {"Retry-After": "1"})

    @app.get("/num/{n}")
    async def num(n: int):
        return n

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.headers["X-Request-ID"]


def test_request_id_echoed_when_valid(client):
    assert client.get("/healthz", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"] == "abc-123"


@pytest.mark.parametrize("bad", ["has space", "x" * 129, "semi;colon"])
def test_request_id_generated_when_invalid(client, bad):
    rid = client.get("/healthz", headers={"X-Request-ID": bad}).headers["X-Request-ID"]
    assert rid != bad and len(rid) == 32


def test_unknown_error_is_500_problem_without_internals(client):
    r = client.get("/boom", headers={"X-Request-ID": "rid-1"})
    assert r.status_code == 500
    assert r.headers["content-type"] == "application/problem+json"
    assert r.headers["X-Request-ID"] == "rid-1"
    body = r.json()
    assert body["status"] == 500 and body["request_id"] == "rid-1"
    assert "secret" not in r.text and "RuntimeError" not in r.text


def test_problem_error_with_headers(client):
    r = client.get("/teapot")
    assert r.status_code == 503 and r.headers["Retry-After"] == "1"
    assert r.json()["detail"] == "busy"


def test_404_and_422_are_problem_json(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.headers["content-type"] == "application/problem+json"
    assert r.json()["request_id"] == r.headers["X-Request-ID"]
    r = client.get("/num/abc")
    assert r.status_code == 422 and r.headers["content-type"] == "application/problem+json"
    assert "abc" not in r.json()["detail"]


def test_invalid_settings_fail_startup(monkeypatch):
    monkeypatch.setenv("GATEWAY_MAX_SLICES", "0")
    with pytest.raises(ValidationError):
        create_app()


def test_auth_disabled_outside_dev_fails_startup(monkeypatch):
    monkeypatch.setenv("GATEWAY_AUTH_DISABLED", "true")
    monkeypatch.setenv("GATEWAY_ENV", "prod")
    with pytest.raises(ValidationError):
        create_app()
    monkeypatch.setenv("GATEWAY_ENV", "dev")
    create_app()


def test_logs_are_json(client, capsys):
    client.get("/boom", headers={"X-Request-ID": "rid-9"})
    out = capsys.readouterr().out
    assert '"request_id": "rid-9"' in out and '"event": "unhandled_error"' in out
