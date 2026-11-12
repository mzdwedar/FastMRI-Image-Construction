"""The gateway image: torch-free, small, non-root. Builds the image (or uses GATEWAY_IMAGE)."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 500 * 1024 * 1024


def docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def image() -> str:
    if shutil.which("docker") is None or docker("info").returncode != 0:
        pytest.skip("Docker is not available")
    if tag := os.environ.get("GATEWAY_IMAGE"):
        return tag
    tag = "fastmri-gateway:it"
    built = docker("build", "-f", "deploy/gateway/Dockerfile", "-t", tag, str(ROOT))
    assert built.returncode == 0, built.stderr[-2000:]
    return tag


def inspect(image: str) -> dict:
    return json.loads(docker("image", "inspect", image).stdout)[0]


def test_image_is_small(image):
    assert inspect(image)["Size"] <= MAX_BYTES


def test_image_runs_as_non_root(image):
    user = inspect(image)["Config"]["User"]
    assert user and user not in ("0", "root") and not user.startswith("0:")
    uid = docker("run", "--rm", "--entrypoint", "id", image, "-u").stdout.strip()
    assert uid != "0"


def test_image_has_no_torch(image):
    r = docker("run", "--rm", "--entrypoint", "python", image, "-c", "import torch")
    assert r.returncode != 0 and "ModuleNotFoundError" in r.stderr
    r = docker("run", "--rm", "--entrypoint", "python", image, "-c", "import fastmri_knee.gateway.app")
    assert r.returncode == 0, r.stderr
