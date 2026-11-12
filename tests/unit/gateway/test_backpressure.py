import asyncio
import functools

import httpx
import numpy as np

from fastmri_knee.gateway.app import create_app
from fastmri_knee.gateway.settings import Settings
from tests.unit.gateway.fakes import FakeTritonClient

BODY = np.zeros((1, 1, 320, 320), "<f4").tobytes()
OCTET = {"content-type": "application/octet-stream"}


def run_async(fn):
    """No pytest-asyncio in this repo: run the coroutine test on its own event loop."""

    @functools.wraps(fn)
    def wrapper():
        asyncio.run(fn())

    return wrapper


@run_async
async def test_saturation_returns_503_immediately_and_recovers():
    fake = FakeTritonClient(delay_s=0.5)
    app = create_app(Settings(env="dev", auth_disabled=True, max_inflight=2, readiness_interval_s=30), fake)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            slow = [asyncio.create_task(c.post("/v1/infer", content=BODY, headers=OCTET)) for _ in range(2)]
            while app.state.limiter.active < 2:
                await asyncio.sleep(0.01)
            t0 = asyncio.get_running_loop().time()
            r = await c.post("/v1/infer", content=BODY, headers=OCTET)
            assert asyncio.get_running_loop().time() - t0 < 0.2  # rejected, not queued behind the slow ones
            assert r.status_code == 503
            assert r.headers["retry-after"] == "1"
            assert r.headers["content-type"].startswith("application/problem+json")
            assert [(await t).status_code for t in slow] == [200, 200]
            assert app.state.limiter.active == 0
            assert (await c.post("/v1/infer", content=BODY, headers=OCTET)).status_code == 200


@run_async
async def test_slot_released_when_request_fails():
    app = create_app(
        Settings(env="dev", auth_disabled=True, max_inflight=1, readiness_interval_s=30), FakeTritonClient()
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            for _ in range(3):
                r = await c.post("/v1/infer", content=b"bad", headers={"content-type": "text/plain"})
                assert r.status_code == 415
            assert app.state.limiter.active == 0


@run_async
async def test_model_and_probes_not_limited():
    app = create_app(
        Settings(env="dev", auth_disabled=True, max_inflight=1, readiness_interval_s=30), FakeTritonClient()
    )
    async with app.router.lifespan_context(app):
        app.state.limiter.active = 1  # full
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            assert (await c.get("/v1/model")).status_code == 200
            assert (await c.get("/healthz")).status_code == 200
