"""Fixtures for tests against the running Compose stack (`make up-dev`, a promoted model).

Needs: gateway :8000, Prometheus :9090, Loki :3100, Tempo :3200, Grafana :3000 on localhost, `docker compose` for
the alert sink (not published) and for stopping Triton, and `E2E_API_KEY` (a key whose hash is in secrets/api_keys).
The tests skip when the gateway is not reachable and fail when the stack is up but not usable, so a half-started
stack is never reported green.
"""

import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

URLS = {
    "gateway": os.environ.get("E2E_GATEWAY_URL", "http://localhost:8000"),
    "prometheus": os.environ.get("E2E_PROMETHEUS_URL", "http://localhost:9090"),
    "loki": os.environ.get("E2E_LOKI_URL", "http://localhost:3100"),
    "tempo": os.environ.get("E2E_TEMPO_URL", "http://localhost:3200"),
    "grafana": os.environ.get("E2E_GRAFANA_URL", "http://localhost:3000"),
}
REPO = Path(__file__).resolve().parents[2]


def compose(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", "compose", *args], cwd=REPO, capture_output=True, text=True, check=check)


def wait_until(check: Callable[[], Any], timeout: float, what: str, every: float = 1.0) -> Any:
    """Poll `check` until it returns something truthy; fail with `what` after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while True:
        try:
            got = check()
            if got:
                return got
        except (httpx.HTTPError, ValueError, KeyError) as e:  # service not answering yet
            last = e
        if time.monotonic() > deadline:
            pytest.fail(
                f"timed out after {timeout:.0f}s waiting for {what}" + (f" (last error: {last!r})" if last else "")
            )
        time.sleep(every)


@pytest.fixture(scope="session")
def stack() -> dict[str, str]:
    try:
        httpx.get(f"{URLS['gateway']}/healthz", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"no Compose stack at {URLS['gateway']} (run `make up-dev`, `make promote`, `make sync`)")
    if httpx.get(f"{URLS['gateway']}/readyz", timeout=5).status_code != 200:
        pytest.fail("gateway is up but not ready: promote and sync a model first (make model-random promote sync)")
    return URLS


@pytest.fixture(scope="session")
def api_key(stack: dict[str, str]) -> str:
    key = os.environ.get("E2E_API_KEY")
    if not key:
        pytest.fail("set E2E_API_KEY to a key whose hash is in secrets/api_keys (make api-key)")
    return key


@pytest.fixture(scope="session")
def full_sampling(stack: dict[str, str]) -> None:
    """Alloy drops ~90 % of healthy traces by default; the trace test needs every one kept."""
    got = compose("exec", "-T", "alloy", "printenv", "TRACE_SAMPLE_PERCENT", check=False).stdout.strip()
    if got != "100":
        pytest.fail(
            f"alloy has TRACE_SAMPLE_PERCENT={got or 'unset'}; run `make test-e2e` (sets 100) or "
            "`TRACE_SAMPLE_PERCENT=100 docker compose up -d alloy`"
        )


def wait_serving(client: httpx.Client, timeout: float = 120, in_a_row: int = 8) -> None:
    """Wait until `in_a_row` consecutive inferences succeed.

    /readyz is answered by one gunicorn worker, and after a Triton restart the workers recover at different times,
    so a single 200 does not mean every worker can serve.
    """
    body = bytes(4 * 320 * 320)  # one zero slice, float32
    streak = 0

    def ok() -> bool:
        nonlocal streak
        r = client.post("/v1/infer", content=body, headers={"content-type": "application/octet-stream"})
        streak = streak + 1 if r.status_code == 200 else 0
        return streak >= in_a_row

    wait_until(ok, timeout, f"{in_a_row} successful inferences in a row", every=0.5)


@pytest.fixture(scope="session")
def gateway(stack: dict[str, str], api_key: str) -> httpx.Client:
    with httpx.Client(base_url=stack["gateway"], headers={"X-API-Key": api_key}, timeout=60) as c:
        wait_serving(c, timeout=60)
        yield c


def prom_value(stack: dict[str, str], query: str) -> float:
    """First value of an instant query, 0 when the series does not exist yet."""
    r = httpx.get(f"{stack['prometheus']}/api/v1/query", params={"query": query}, timeout=10).json()
    assert r["status"] == "success", r
    return float(r["data"]["result"][0]["value"][1]) if r["data"]["result"] else 0.0


def sink_alerts() -> list[dict[str, Any]]:
    """Alerts the alert-sink has received (its port is not published, so ask from inside the container)."""
    import json

    code = "import urllib.request as u; print(u.urlopen('http://127.0.0.1:8080/alerts', timeout=3).read().decode())"
    out = compose("exec", "-T", "alert-sink", "python", "-c", code).stdout
    return json.loads(out)
