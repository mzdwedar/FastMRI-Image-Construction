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
