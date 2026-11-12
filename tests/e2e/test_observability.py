"""Observability end to end: the signals a request leaves, the dashboards that read them, and the Triton outage alert.

`test_triton_outage_pages_and_recovers` stops Triton, so it stays last in this file.
"""

import io
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from tests.e2e.conftest import compose, prom_value, sink_alerts, wait_serving, wait_until
from tests.fixtures.synthetic import make_kspace, write_h5

pytestmark = pytest.mark.e2e
REQUESTS = 10
DASHBOARDS = Path(__file__).resolve().parents[2] / "monitoring/grafana/dashboards"
RECON = 'sum(gateway_requests_total{route="/v1/reconstructions",status_class="2xx"})'


@pytest.fixture(scope="module")
def volume(tmp_path_factory) -> bytes:
    path = write_h5(tmp_path_factory.mktemp("e2e") / "patient-7.h5", make_kspace(slices=2))
    return path.read_bytes()


def _post(gateway: httpx.Client, volume: bytes) -> httpx.Response:
    r = gateway.post("/v1/reconstructions", files={"file": ("patient-7.h5", io.BytesIO(volume), "application/x-hdf5")})
    assert r.status_code == 200, r.text
    return r


def _loki_lines(stack, request_id: str) -> list[dict]:
    now = time.time_ns()
    r = httpx.get(
        f"{stack['loki']}/loki/api/v1/query_range",
        params={
            "query": f'{{service="gateway"}} | json | request_id="{request_id}"',
            "start": now - 10 * 60 * 10**9,
            "end": now + 60 * 10**9,
            "limit": 20,
        },
        timeout=10,
    ).json()
    return [json.loads(v[1]) for s in r["data"]["result"] for v in s["values"]]  # fmt: skip


def _span_names(trace: dict) -> set[str]:
    return {
        s["name"]
        for b in trace.get("batches", [])
        for ss in b.get("scopeSpans", b.get("instrumentationLibrarySpans", []))
        for s in ss["spans"]
    }


@pytest.fixture(scope="module")
def sent(stack, gateway, volume, full_sampling) -> dict:
    """Ten reconstructions; the first one's request id is the one the log and trace tests look up."""
    before = prom_value(stack, RECON)
    request_ids = [_post(gateway, volume).headers["X-Request-ID"] for _ in range(REQUESTS)]
    return {"before": before, "request_id": request_ids[0]}


def test_request_counter_reaches_prometheus(stack, sent):
    got = wait_until(
        lambda: (v := prom_value(stack, RECON) - sent["before"]) >= REQUESTS and v,
        timeout=60,
        what=f"gateway_requests_total to rise by {REQUESTS}",
    )
    assert got >= REQUESTS


def test_log_line_for_a_known_request_id_is_in_loki(stack, sent):
    lines = wait_until(lambda: _loki_lines(stack, sent["request_id"]), 60, "the request's log line in Loki")
    access = next(line for line in lines if line.get("event") == "request")
    assert access["route"] == "/v1/reconstructions" and access["status"] == 200
    assert re.fullmatch(r"[0-9a-f]{32}", access["trace_id"])
    assert "patient-7" not in json.dumps(lines)  # filenames never reach the logs


def test_trace_has_gateway_and_triton_spans(stack, sent):
    access = next(line for line in _loki_lines(stack, sent["request_id"]) if line.get("event") == "request")

    def find() -> dict | None:
        r = httpx.get(f"{stack['tempo']}/api/traces/{access['trace_id']}", timeout=10)
        return r.json() if r.status_code == 200 else None  # 404 until Alloy's 10 s decision window has passed

    names = _span_names(wait_until(find, 90, "the request's trace in Tempo", every=3))
    assert {"upload.read", "h5.validate", "preprocess", "triton.infer", "response.encode"} <= names
    assert {"InferRequest", "compute"} <= names  # Triton's own spans, parented on triton.infer

    def searchable() -> bool:  # search indexes a trace a little after it can be fetched by id
        found = httpx.get(
            f"{stack['tempo']}/api/search",
            params={
                "q": f'{{ span.request_id = "{sent["request_id"]}" }}',
                "start": int(time.time()) - 900,
                "end": int(time.time()) + 60,
            },
            timeout=10,
        ).json()
        return access["trace_id"] in {t["traceID"] for t in found.get("traces", [])}

    wait_until(searchable, 60, "the trace to be searchable by request_id", every=3)


def test_every_dashboard_query_runs(stack):
    files = sorted(DASHBOARDS.glob("*.json"))
    assert len(files) >= 3
    queries = 0
    for path in files:
        for panel in json.loads(path.read_text())["panels"]:
            for t in panel["targets"]:
                r = httpx.get(
                    f"{stack['prometheus']}/api/v1/query",
                    params={"query": t["expr"].replace("$__rate_interval", "1m")},
                    timeout=10,
                ).json()
                assert r["status"] == "success", f"{path.name} / {panel['title']}: {r}"
                queries += 1
    assert queries >= 30
    titles = {d["title"] for d in httpx.get(f"{stack['grafana']}/api/search", timeout=10).json()}
    assert {"Service overview", "Triton", "Infrastructure"} <= titles  # provisioned, not just valid JSON


def test_triton_outage_pages_and_recovers(stack, gateway):
    """Stop Triton: /readyz turns 503 within 10 s, TritonDown reaches the alert sink within 2 min; restart recovers."""
    started = datetime.now(UTC)
    compose("stop", "triton")
    try:
        t0 = time.monotonic()
        wait_until(
            lambda: httpx.get(f"{stack['gateway']}/readyz", timeout=3).status_code == 503, 10, "/readyz 503", 0.5
        )
        assert time.monotonic() - t0 <= 10

        def fired() -> bool:
            return any(
                a["alertname"] == "TritonDown"
                and a["status"] == "firing"
                and datetime.fromisoformat(a["starts_at"]) >= started
                for a in sink_alerts()
            )

        wait_until(fired, 120, "TritonDown at the alert sink", every=3)
    finally:
        compose("start", "triton")
    wait_serving(gateway)  # every worker, not just the one that answered /readyz
