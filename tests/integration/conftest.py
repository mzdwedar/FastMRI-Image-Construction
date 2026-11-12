"""Real Triton (explicit control mode, empty model repository) and a throwaway MLflow registry.

Skips when Docker or the Triton image is unavailable. Override the image with TRITON_IMAGE.
"""

import os
import shutil
import subprocess
import time
import uuid

import httpx
import pytest

from fastmri_knee.packaging.export import random_init_checkpoint
from fastmri_knee.packaging.registry import register

TRITON_IMAGE = os.environ.get("TRITON_IMAGE", "nvcr.io/nvidia/tritonserver:24.08-py3")
MODEL = "unet_model"


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=False)


@pytest.fixture(scope="session")
def triton_repo(tmp_path_factory):
    repo = tmp_path_factory.mktemp("model_repository")
    repo.chmod(0o755)
    return repo


@pytest.fixture(scope="session")
def triton_url(triton_repo):
    if shutil.which("docker") is None or _docker("info").returncode != 0:
        pytest.skip("Docker is not available")
    if _docker("image", "inspect", TRITON_IMAGE).returncode != 0:
        pytest.skip(f"{TRITON_IMAGE} is not present locally (docker pull it, or set TRITON_IMAGE)")
    name = f"fastmri-it-triton-{uuid.uuid4().hex[:8]}"
    started = _docker(
        "run", "-d", "--rm", "--name", name, "-p", "127.0.0.1::8000",
        "-v", f"{triton_repo}:/models",
        TRITON_IMAGE,
        "tritonserver", "--model-repository=/models", "--model-control-mode=explicit",
        f"--load-model={MODEL}", "--strict-readiness=true", "--exit-on-error=false",
    )  # fmt: skip
    if started.returncode != 0:
        pytest.skip(f"could not start Triton: {started.stderr.strip()}")
    try:
        port = _docker("port", name, "8000/tcp").stdout.strip().splitlines()[0].rsplit(":", 1)[1]
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 60
        while True:
            try:
                if httpx.get(f"{url}/v2/health/live", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"Triton not live after 60s:\n{_docker('logs', name).stdout[-2000:]}")
            time.sleep(1)
        yield url
    finally:
        _docker("rm", "-f", name)


@pytest.fixture(scope="session")
def registry(tmp_path_factory):
    """Three registered versions of a random-init U-Net (each exported to its own model.onnx)."""
    uri = f"file://{tmp_path_factory.mktemp('mlruns')}"
    ckpt = tmp_path_factory.mktemp("ckpt") / "model.ckpt"
    random_init_checkpoint(ckpt)
    for _ in range(3):
        register(ckpt, tracking_uri=uri, source="random-init", extra_tags={"random_init": "true"})
    return uri


def version_ready(url: str, version: int) -> bool:
    return httpx.get(f"{url}/v2/models/{MODEL}/versions/{version}/ready", timeout=5).status_code == 200


# --- gateway tests: a Triton with gRPC published on fixed ports, so it can be stopped and restarted ---------------


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class GatewayTriton:
    """A Triton serving random-init `unet_model` v1 from its own repository. `stop()`/`start()` keep the ports."""

    def __init__(self, name: str, grpc_port: int, http_port: int, onnx: "os.PathLike[str]"):
        self.name, self.grpc_port, self.http_port, self.onnx = name, grpc_port, http_port, onnx

    @property
    def grpc_url(self) -> str:
        return f"127.0.0.1:{self.grpc_port}"

    def wait_ready(self, timeout_s: float = 90) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if (
                    httpx.get(f"http://127.0.0.1:{self.http_port}/v2/models/{MODEL}/ready", timeout=2).status_code
                    == 200
                ):
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"Triton not ready after {timeout_s}s:\n{_docker('logs', self.name).stdout[-2000:]}")
            time.sleep(1)

    def stop(self) -> None:
        assert _docker("stop", "-t", "1", self.name).returncode == 0

    def start(self) -> None:
        assert _docker("start", self.name).returncode == 0
        self.wait_ready()


@pytest.fixture(scope="session")
def gateway_onnx(tmp_path_factory):
    """A random-init U-Net exported to ONNX; also the reference the gateway output is compared with."""
    from fastmri_knee.packaging.export import export

    d = tmp_path_factory.mktemp("gateway_model")
    module = random_init_checkpoint(d / "model.ckpt")
    export(module, d / "model.onnx")
    return d / "model.onnx"


@pytest.fixture(scope="session")
def gateway_triton(gateway_onnx, tmp_path_factory):
    from fastmri_knee.packaging.triton_repo import render

    if shutil.which("docker") is None or _docker("info").returncode != 0:
        pytest.skip("Docker is not available")
    if _docker("image", "inspect", TRITON_IMAGE).returncode != 0:
        pytest.skip(f"{TRITON_IMAGE} is not present locally (docker pull it, or set TRITON_IMAGE)")
    repo = tmp_path_factory.mktemp("gateway_repo")
    (repo / MODEL / "1").mkdir(parents=True)
    shutil.copy(gateway_onnx, repo / MODEL / "1" / "model.onnx")
    (repo / MODEL / "config.pbtxt").write_text(render("cpu", version=1, threads=2))
    for p in [repo, *repo.rglob("*")]:
        p.chmod(0o755)
    triton = GatewayTriton(f"fastmri-it-gw-{uuid.uuid4().hex[:8]}", _free_port(), _free_port(), gateway_onnx)
    started = _docker(
        "run", "-d", "--name", triton.name,
        "-p", f"127.0.0.1:{triton.grpc_port}:8001", "-p", f"127.0.0.1:{triton.http_port}:8000",
        "-v", f"{repo}:/models:ro",
        TRITON_IMAGE, "tritonserver", "--model-repository=/models", "--strict-readiness=true",
    )  # fmt: skip
    if started.returncode != 0:
        pytest.skip(f"could not start Triton: {started.stderr.strip()}")
    try:
        triton.wait_ready()
        yield triton
    finally:
        _docker("rm", "-f", triton.name)


# --- T3: a Triton with an empty repository (strict readiness) ---------------------------------------------------


@pytest.fixture(scope="module")
def empty_triton(tmp_path_factory):
    """Triton started on an empty model repository, as `make up` does before the first promote."""
    if shutil.which("docker") is None or _docker("info").returncode != 0:
        pytest.skip("Docker is not available")
    if _docker("image", "inspect", TRITON_IMAGE).returncode != 0:
        pytest.skip(f"{TRITON_IMAGE} is not present locally (docker pull it, or set TRITON_IMAGE)")
    repo = tmp_path_factory.mktemp("empty_repo")
    repo.chmod(0o755)
    name, http_port = f"fastmri-it-empty-{uuid.uuid4().hex[:8]}", _free_port()
    started = _docker(
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "-p",
        f"127.0.0.1:{http_port}:8000",
        "-v",
        f"{repo}:/models:ro",
        TRITON_IMAGE,
        "tritonserver",
        "--model-repository=/models",
        "--model-control-mode=explicit",
        f"--load-model={MODEL}",
        "--strict-readiness=true",
        "--exit-on-error=false",
    )  # fmt: skip  (the same flags docker-compose.yml uses)
    if started.returncode != 0:
        pytest.skip(f"could not start Triton: {started.stderr.strip()}")
    url = f"http://127.0.0.1:{http_port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                if httpx.get(f"{url}/v2/health/live", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"Triton not live after 60s:\n{_docker('logs', name).stdout[-2000:]}")
            time.sleep(1)
        yield url
    finally:
        _docker("rm", "-f", name)
