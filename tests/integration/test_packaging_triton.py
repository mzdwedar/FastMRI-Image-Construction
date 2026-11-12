"""promote -> rollback -> failed promote against a real Triton in explicit control mode."""

from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from fastmri_knee.packaging.promote import PromoteError, promote, rollback
from fastmri_knee.packaging.registry import MODEL_NAME
from tests.integration.conftest import version_ready

pytestmark = pytest.mark.integration


def production() -> int:
    return int(MlflowClient().get_model_version_by_alias(MODEL_NAME, "production").version)


def run(fn, registry, repo, url, *args, **kwargs):
    return fn(*args, tracking_uri=registry, repo=repo, triton_url=url, threads=2, timeout=120.0, **kwargs)


def test_promote_rollback_and_failed_promote(registry, triton_repo, triton_url):
    # random-init models: --force is how the Phase 3 checkpoint promotes one
    assert run(promote, registry, triton_repo, triton_url, 1, force=True) == 1
    assert production() == 1 and version_ready(triton_url, 1)

    assert run(promote, registry, triton_repo, triton_url, 2, force=True) == 2
    assert production() == 2 and version_ready(triton_url, 2)
    assert not version_ready(triton_url, 1)  # the config pins the served version

    assert run(rollback, registry, triton_repo, triton_url) == 1
    assert production() == 1 and version_ready(triton_url, 1) and not version_ready(triton_url, 2)

    # corrupt v3's model.onnx in the registry: the sha256 check must refuse it and v1 must keep serving
    mlflow.set_tracking_uri(registry)
    client = MlflowClient()
    run_id = client.get_model_version(MODEL_NAME, "3").run_id
    artifact_root = client.get_run(run_id).info.artifact_uri.removeprefix("file://")
    onnx = next(Path(artifact_root).rglob("model.onnx"))
    onnx.write_bytes(onnx.read_bytes()[:-16] + b"corrupted-bytes!")

    with pytest.raises(PromoteError, match="production restored.*sha256"):
        run(promote, registry, triton_repo, triton_url, 3, force=True)
    assert production() == 1
    assert version_ready(triton_url, 1)
    assert not (triton_repo / "unet_model" / "3").exists()
