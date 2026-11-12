"""A minimal fake of Triton's explicit-mode HTTP API, served through httpx.MockTransport."""

from pathlib import Path

import httpx

from fastmri_knee.packaging.sync import served_version


class FakeTriton:
    """Just enough of Triton's explicit-mode HTTP API."""

    def __init__(self, repo: Path):
        self.repo = repo
        self.calls: list[str] = []
        self.fail_loads = False  # load request rejected
        self.never_ready = False  # load accepted, version never becomes ready
        self.loaded: int | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(f"{request.method} {path}")
        if path.endswith("/load"):
            if self.fail_loads:
                return httpx.Response(503, text="load failed")
            config = (self.repo / "unet_model" / "config.pbtxt").read_text()
            self.loaded = served_version(config)
            return httpx.Response(200)
        if path.endswith("/unload"):
            self.loaded = None
            return httpx.Response(200)
        version = int(path.split("/versions/")[1].split("/")[0])
        ready = not self.never_ready and self.loaded == version
        return httpx.Response(200 if ready else 400)

    def client(self) -> httpx.Client:
        return httpx.Client(base_url="http://triton", transport=httpx.MockTransport(self.handler))

    def count(self, suffix: str) -> int:
        return sum(call.endswith(suffix) for call in self.calls)
