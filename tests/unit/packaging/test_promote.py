import contextlib

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from fastmri_knee.packaging.card import sha256_file
from fastmri_knee.packaging.promote import PromoteError, promote, rollback
from fastmri_knee.packaging.registry import MODEL_NAME, register
from fastmri_knee.packaging.sync import served_version
from fastmri_knee.preprocess import PREPROCESS_VERSION
from tests.unit.packaging.fake_triton import FakeTriton


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from fastmri_knee.packaging.export import random_init_checkpoint

    path = tmp_path_factory.mktemp("ckpt") / "model.ckpt"
    random_init_checkpoint(path)
    return path


def summary(checkpoint, psnr: float | None):
    return {
        "checkpoint_sha256": sha256_file(checkpoint),
        "preprocess_version": PREPROCESS_VERSION,
        "val_metrics": {} if psnr is None else {"psnr": psnr},
    }


@pytest.fixture(scope="module")
def registry(checkpoint, tmp_path_factory):
    """v1 psnr 30, v2 psnr 31, v3 psnr 29 (>0.5 below v2), v4 no psnr, v5 random-init, v6 wrong preprocess,
    v7 psnr 30.5 (exactly 0.5 below v2)."""
    uri = f"file://{tmp_path_factory.mktemp('mlruns')}"
    for psnr in (30.0, 31.0, 29.0, None):
        register(checkpoint, tracking_uri=uri, summary=summary(checkpoint, psnr))
    register(checkpoint, tracking_uri=uri, source="random-init", extra_tags={"random_init": "true"})
    register(checkpoint, tracking_uri=uri, extra_tags={"preprocess_version": "999"})
    register(checkpoint, tracking_uri=uri, summary=summary(checkpoint, 30.5))
    return uri


@pytest.fixture
def store(registry):
    """The shared registry with every alias and promotion tag cleared, so each test starts from nothing."""
    mlflow.set_tracking_uri(registry)
    client = MlflowClient()
    for alias in ("production", "staging"):
        with contextlib.suppress(mlflow.exceptions.MlflowException):
            client.delete_registered_model_alias(MODEL_NAME, alias)
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        for key in ("previous_production", "promote_forced"):
            if key in mv.tags:
                client.delete_model_version_tag(MODEL_NAME, mv.version, key)
    return registry


def alias_version(alias: str = "production") -> int | None:
    try:
        return int(MlflowClient().get_model_version_by_alias(MODEL_NAME, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def tag(version: int, key: str) -> str | None:
    return MlflowClient().get_model_version(MODEL_NAME, str(version)).tags.get(key)


def go(store, version, **kwargs):
    return promote(version, tracking_uri=store, do_sync=False, **kwargs)


def test_promote_sets_alias_and_records_previous_production(store):
    go(store, 1)
    assert alias_version() == 1 and tag(1, "previous_production") is None
    go(store, 2)
    assert alias_version() == 2 and tag(2, "previous_production") == "1"


def test_preprocess_mismatch_refuses_even_with_force(store):
    with pytest.raises(PromoteError, match="preprocess_version='999'"):
        go(store, 6, force=True)
    assert alias_version() is None


def test_random_init_refused_for_production_unless_allowed(store):
    with pytest.raises(PromoteError, match="random-init.*--allow-random"):
        go(store, 5)
    assert alias_version() is None
    go(store, 5, allow_random=True)
    assert alias_version() == 5


def test_random_init_is_fine_for_staging(store):
    go(store, 5, alias="staging")
    assert alias_version("staging") == 5


def test_psnr_regression_refused_and_force_overrides_with_a_tag(store, caplog):
    go(store, 2)
    with pytest.raises(PromoteError, match=r"29\.00 dB.*31\.00 dB.*--force"):
        go(store, 3)
    assert alias_version() == 2
    with caplog.at_level("WARNING"):
        go(store, 3, force=True)
    assert alias_version() == 3 and tag(3, "promote_forced") == "psnr"
    assert "--force" in caplog.text


def test_psnr_drop_of_exactly_the_tolerance_passes(store):
    go(store, 2)
    go(store, 7)
    assert alias_version() == 7 and tag(7, "promote_forced") is None


def test_version_without_psnr_cannot_replace_one_that_has_it(store):
    go(store, 1)
    with pytest.raises(PromoteError, match="no val PSNR"):
        go(store, 4)


def test_unknown_version_is_a_clear_error(store):
    with pytest.raises(PromoteError, match="no version 99"):
        go(store, 99)


def test_rollback_restores_previous_and_skips_the_psnr_gate(store):
    go(store, 2)
    go(store, 3, force=True)
    assert rollback(tracking_uri=store, do_sync=False) == 2
    assert alias_version() == 2


def test_rollback_without_history_is_refused(store):
    with pytest.raises(PromoteError, match="no version has the production alias"):
        rollback(tracking_uri=store, do_sync=False)
    go(store, 1)
    with pytest.raises(PromoteError, match="previous_production"):
        rollback(tracking_uri=store, do_sync=False)


def test_promote_runs_sync_and_triton_serves_it(store, tmp_path):
    repo = tmp_path / "repo"
    triton = FakeTriton(repo)

    def kwargs():  # sync closes the client it is given
        return {"repo": repo, "http": triton.client(), "sleep": lambda _: None}

    promote(1, tracking_uri=store, **kwargs())
    promote(2, tracking_uri=store, **kwargs())
    assert triton.loaded == 2
    rollback(tracking_uri=store, **kwargs())
    assert triton.loaded == 1
    assert served_version((repo / "unet_model" / "config.pbtxt").read_text()) == 1


def test_failed_sync_puts_alias_and_tag_back(store, tmp_path):
    repo = tmp_path / "repo"
    triton = FakeTriton(repo)

    def kwargs():  # sync closes the client it is given
        return {"repo": repo, "http": triton.client(), "sleep": lambda _: None}

    promote(1, tracking_uri=store, **kwargs())
    triton.fail_loads = True
    with pytest.raises(PromoteError, match="production restored"):
        promote(2, tracking_uri=store, **kwargs())
    assert alias_version() == 1 and tag(2, "previous_production") is None


def test_failed_first_sync_leaves_no_production_alias(store, tmp_path):
    triton = FakeTriton(tmp_path / "repo")
    triton.fail_loads = True
    with pytest.raises(PromoteError):
        promote(1, tracking_uri=store, repo=tmp_path / "repo", http=triton.client(), sleep=lambda _: None)
    assert alias_version() is None
