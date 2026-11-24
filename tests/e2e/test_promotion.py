"""Promote a new model version and roll back while requests are in flight: nothing may fail (SPEC-model-packaging)."""

import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx
import mlflow
import pytest
from mlflow.tracking import MlflowClient

from fastmri_knee.packaging.promote import PRODUCTION, promote, rollback
from fastmri_knee.packaging.registry import MODEL_NAME, register_random

pytestmark = pytest.mark.e2e

MLFLOW = os.environ.get("E2E_MLFLOW_URL", "http://localhost:5001")
TRITON = os.environ.get("E2E_TRITON_URL", "http://localhost:8100")
CONCURRENCY = 8
MAX_SWITCH_S = 60
CONVERGE_S = 10  # the version header trails a switch: it comes from the readiness loop
SETTLE_S = 4  # load before, between and after the two switches


def production_version() -> int:
    mlflow.set_tracking_uri(MLFLOW)
    return int(MlflowClient().get_model_version_by_alias(MODEL_NAME, PRODUCTION).version)


def test_promote_and_rollback_under_load_drop_no_requests(gateway: httpx.Client, api_key: str):
    original = production_version()
    new = register_random(tracking_uri=MLFLOW)
    body = bytes(4 * 320 * 320)
    end = 0.0
    stop = threading.Event()
    log: list[tuple[float, int | str, str | None]] = []  # (time, status or exception name, X-Model-Version)

    def worker() -> None:
        with httpx.Client(base_url=str(gateway.base_url), headers={"X-API-Key": api_key}, timeout=60) as c:
            while not stop.is_set():
                try:
                    r = c.post("/v1/infer", content=body, headers={"content-type": "application/octet-stream"})
                    log.append((time.monotonic(), r.status_code, r.headers.get("X-Model-Version")))
                except httpx.HTTPError as e:
                    log.append((time.monotonic(), type(e).__name__, None))

    switches: dict[str, tuple[float, float]] = {}
    pool = ThreadPoolExecutor(max_workers=CONCURRENCY)
    try:
        for _ in range(CONCURRENCY):
            pool.submit(worker)
        time.sleep(SETTLE_S)
        t0 = time.monotonic()
        promote(new, allow_random=True, force=True, tracking_uri=MLFLOW, triton_url=TRITON)
        switches["promote"] = (t0, time.monotonic())
        time.sleep(SETTLE_S)
        t0 = time.monotonic()
        rollback(tracking_uri=MLFLOW, triton_url=TRITON)
        switches["rollback"] = (t0, time.monotonic())
        time.sleep(CONVERGE_S + 2)
        end = time.monotonic()
    finally:
        stop.set()
        pool.shutdown(wait=True)
        if production_version() != original:  # leave the stack as it was found, whatever happened above
            promote(original, allow_random=True, force=True, tracking_uri=MLFLOW, triton_url=TRITON)

    statuses = Counter(s for _, s, _ in log)
    print(
        f"\npromotion under load: {len(log)} requests, statuses {dict(statuses)}, "
        + ", ".join(f"{n} {e - b:.1f}s" for n, (b, e) in switches.items())
    )
    assert len(log) > 20 * CONCURRENCY // 4, f"load generator barely ran: {len(log)} requests"
    assert set(statuses) == {200}, f"failed requests during promote/rollback: {dict(statuses)}"
    for name, (start, end) in switches.items():
        assert end - start <= MAX_SWITCH_S, f"{name} took {end - start:.0f}s (budget {MAX_SWITCH_S}s)"

    def versions_between(a: float, b: float) -> set[str | None]:
        return {v for t, _, v in log if a <= t <= b}

    # X-Model-Version comes from the readiness loop (polled every 2 s), so it trails a switch; it must catch up quickly.
    def lag(after: float, version: int) -> float:
        first = next((t for t, _, v in log if t >= after and v == str(version)), None)
        assert first is not None, f"gateway never reported v{version} after the switch"
        return first - after

    lags = {"promote": lag(switches["promote"][1], new), "rollback": lag(switches["rollback"][1], original)}
    print("version header lag after the switch: " + ", ".join(f"{k} {v:.1f}s" for k, v in lags.items()))
    assert all(v <= CONVERGE_S for v in lags.values()), f"gateway slow to report the new version: {lags}"
    assert {v for t, _, v in log if t >= end - 1.5} == {str(original)}, "gateway did not settle on the original version"
