import json
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from fastmri_knee.packaging import registry
from fastmri_knee.packaging.card import sha256_file
from fastmri_knee.packaging.registry import MODEL_NAME, RegistryError, register, register_random
from fastmri_knee.preprocess import PREPROCESS_VERSION

CARD_FIELDS = {
    "registry_version", "source", "source_run_id", "git_sha", "checkpoint_sha256", "onnx_sha256", "opset",
    "input_names", "output_names", "input_shape", "output_shape", "dtype", "preprocess_version",
    "parity_max_diff", "val_metrics", "hyperparameters", "created_at",
}  # fmt: skip


@pytest.fixture
def store(tmp_path):
    uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(uri)
    return uri


def fetch(version, name):
    return mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}/{version}/{name}")


def summary_for(checkpoint, **overrides):
    return {
        "source": "colab",
        "git_sha": "abc123",
        "preprocess_version": PREPROCESS_VERSION,
        "checkpoint_sha256": sha256_file(checkpoint),
        "val_metrics": {"psnr": 30.5, "ssim": 0.9, "nmse": 0.01},
        "hyperparameters": {"lr": 0.001},
        **overrides,
    }


def test_register_writes_all_artifacts_card_and_tags(store, checkpoint):
    version = register(checkpoint, tracking_uri=store, summary=summary_for(checkpoint))
    assert version == 1

    card = json.loads(Path(fetch(version, "model_card.json")).read_text())
    assert set(card) == CARD_FIELDS
    assert card["registry_version"] == 1
    assert card["checkpoint_sha256"] == sha256_file(checkpoint)
    assert card["onnx_sha256"] == sha256_file(Path(fetch(version, "model.onnx")))
    assert (card["source"], card["git_sha"], card["preprocess_version"]) == ("colab", "abc123", PREPROCESS_VERSION)
    assert card["val_metrics"] == {"psnr": 30.5, "ssim": 0.9, "nmse": 0.01}
    assert card["parity_max_diff"] < 1e-4 and card["opset"] == 17 and card["dtype"] == "float32"
    assert sha256_file(Path(fetch(version, "model.ckpt"))) == card["checkpoint_sha256"]

    tags = MlflowClient().get_model_version(MODEL_NAME, str(version)).tags
    assert tags == {"preprocess_version": PREPROCESS_VERSION, "git_sha": "abc123"}


def test_second_registration_is_a_new_version(store, checkpoint):
    assert register(checkpoint, tracking_uri=store) == 1
    assert register(checkpoint, tracking_uri=store) == 2


def test_parity_failure_refuses_and_registers_nothing(store, checkpoint, monkeypatch):
    def diverge(*args, **kwargs):
        raise AssertionError("ONNX output differs from PyTorch by 1.00e-02 (> 0.0001)")

    monkeypatch.setattr(registry, "check_parity", diverge)
    with pytest.raises(RegistryError, match="refusing to register"):
        register(checkpoint, tracking_uri=store)
    assert MlflowClient().search_model_versions(f"name='{MODEL_NAME}'") == []


def test_summary_hash_mismatch_refuses(store, checkpoint):
    with pytest.raises(RegistryError, match="sha256"):
        register(checkpoint, tracking_uri=store, summary=summary_for(checkpoint, checkpoint_sha256="0" * 64))


def test_summary_preprocess_version_mismatch_refuses(store, checkpoint):
    with pytest.raises(RegistryError, match="preprocess_version"):
        register(checkpoint, tracking_uri=store, summary=summary_for(checkpoint, preprocess_version="999"))


def test_register_random_needs_no_checkpoint_and_is_tagged(store):
    version = register_random(tracking_uri=store)
    tags = MlflowClient().get_model_version(MODEL_NAME, str(version)).tags
    assert tags["random_init"] == "true" and tags["preprocess_version"] == PREPROCESS_VERSION
    card = json.loads(Path(fetch(version, "model_card.json")).read_text())
    assert card["source"] == "random-init" and card["val_metrics"] == {}


def test_trained_registration_has_no_random_init_tag(store, checkpoint):
    version = register(checkpoint, tracking_uri=store)
    assert "random_init" not in MlflowClient().get_model_version(MODEL_NAME, str(version)).tags
