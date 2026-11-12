"""End-to-end tests run against the real Compose stack (`make ci-e2e` brings it up, or `make up-dev` locally).

Environment: E2E_GATEWAY_URL (default http://localhost:8000) and E2E_API_KEY (from `make api-key`).
Skips when the stack is not reachable, so a plain `pytest -m e2e` never fails for lack of Docker; CI treats a skip
as a failure (see .github/workflows/e2e.yml).
"""

import os
import time

import httpx
import pytest

GATEWAY_URL = os.environ.get("E2E_GATEWAY_URL", "http://localhost:8000").rstrip("/")


def wait_ready(client: httpx.Client, timeout_s: float = 120) -> None:
    deadline = time.monotonic() + timeout_s
    last = "no response"
    while time.monotonic() < deadline:
        try:
            r = client.get("/readyz")
            if r.status_code == 200:
                return
            last = f"{r.status_code} {r.text[:200]}"
        except httpx.HTTPError as e:
            last = repr(e)
        time.sleep(1)
    raise AssertionError(f"/readyz did not reach 200 in {timeout_s}s; last: {last}")


@pytest.fixture(scope="session")
def api_key() -> str:
    key = os.environ.get("E2E_API_KEY")
    if not key:
        pytest.skip("E2E_API_KEY is not set (mint one with `make api-key`)")
    return key


@pytest.fixture(scope="session")
def gateway(api_key: str):
    """An authenticated client for a gateway whose model is ready."""
    with httpx.Client(base_url=GATEWAY_URL, headers={"X-API-Key": api_key}, timeout=60) as client:
        try:
            client.get("/healthz")
        except httpx.ConnectError:
            pytest.skip(f"no gateway at {GATEWAY_URL}; bring the stack up with `make up-dev`")
        wait_ready(client)
        yield client


@pytest.fixture(scope="session")
def anonymous() -> httpx.Client:
    with httpx.Client(base_url=GATEWAY_URL, timeout=30) as client:
        yield client
