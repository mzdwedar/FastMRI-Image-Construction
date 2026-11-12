import hashlib
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from fastmri_knee.packaging.registry import MODEL_NAME, register
from fastmri_knee.packaging.sync import SyncError, served_version, sync
from fastmri_knee.packaging.triton_repo import render
from tests.unit.packaging.fake_triton import FakeTriton


@pytest.fixture(scope="module")
def store(checkpoint_module, tmp_path_factory):
    uri = f"file://{tmp_path_factory.mktemp('mlruns')}"
    for _ in range(3):
        register(checkpoint_module, tracking_uri=uri)
    return uri


@pytest.fixture(scope="module")
def checkpoint_module(tmp_path_factory):
    from fastmri_knee.packaging.export import random_init_checkpoint

    path = tmp_path_factory.mktemp("ckpt") / "model.ckpt"
    random_init_checkpoint(path)
    return path


@pytest.fixture
def env(store, tmp_path):
    repo = tmp_path / "model_repository"
    triton = FakeTriton(repo)
    mlflow.set_tracking_uri(store)

    def run(version: int | None = None, **kwargs) -> int:
        if version is not None:
            MlflowClient().set_registered_model_alias(MODEL_NAME, "production", str(version))
        return sync(tracking_uri=store, repo=repo, http=triton.client(), sleep=lambda _: None, **kwargs)

    return repo, triton, run


def snapshot(repo: Path) -> dict[str, tuple[int, str]]:
    return {
        str(p.relative_to(repo)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(repo.rglob("*"))
        if p.is_file()
    }


def test_sync_writes_version_dir_and_rendered_config_then_loads(env):
    repo, triton, run = env
    assert run(2) == 2
    assert (repo / "unet_model" / "2" / "model.onnx").stat().st_size > 0
    assert (repo / "unet_model" / "config.pbtxt").read_text() == render("cpu", version=2)
    assert triton.loaded == 2 and triton.count("/load") == 1
    assert {p.name for p in (repo / "unet_model").iterdir()} == {"2", "config.pbtxt"}


def test_second_sync_changes_nothing_and_does_not_reload(env):
    repo, triton, run = env
    run(2)
    before, loads = snapshot(repo), triton.count("/load")
    assert run() == 2
    assert snapshot(repo) == before
    assert triton.count("/load") == loads


def test_sha_mismatch_aborts_before_touching_the_repository(env, store, checkpoint_module):
    repo, triton, run = env
    version = register(checkpoint_module, tracking_uri=store)
    run_id = MlflowClient().get_model_version(MODEL_NAME, str(version)).run_id
    stored = next(Path(store.removeprefix("file://")).rglob(f"{run_id}/artifacts/model/model.onnx"))
    stored.write_bytes(stored.read_bytes() + b"corrupt")
    with pytest.raises(SyncError, match="sha256"):
        run(version)
    assert not repo.exists() or not any(repo.rglob("*"))
    assert triton.calls == []


def test_load_failure_restores_previous_version_config_and_alias(env):
    repo, triton, run = env
    run(1)
    config_before = (repo / "unet_model" / "config.pbtxt").read_text()
    triton.fail_loads = True
    with pytest.raises(SyncError, match="previous state restored"):
        run(2)
    assert (repo / "unet_model" / "config.pbtxt").read_text() == config_before
    assert not (repo / "unet_model" / "2").exists()
    assert (repo / "unet_model" / "1" / "model.onnx").exists()
    assert int(MlflowClient().get_model_version_by_alias(MODEL_NAME, "production").version) == 1


def test_version_never_ready_times_out_and_restores(env, store):
    repo, triton, run = env
    run(1)
    triton.never_ready = True
    MlflowClient().set_registered_model_alias(MODEL_NAME, "production", "2")
    ticks = iter(range(0, 1000, 20))  # fake clock: every reading advances 20 s
    with pytest.raises(SyncError, match="previous state restored"):
        sync(
            tracking_uri=store, repo=repo, http=triton.client(), sleep=lambda _: None, clock=lambda: float(next(ticks))
        )
    assert served_version((repo / "unet_model" / "config.pbtxt").read_text()) == 1
    assert int(MlflowClient().get_model_version_by_alias(MODEL_NAME, "production").version) == 1


def test_prunes_to_current_and_previous_only(env):
    repo, _, run = env
    run(1)
    run(2)
    run(3)
    assert sorted(p.name for p in (repo / "unet_model").iterdir() if p.is_dir()) == ["2", "3"]


def test_missing_alias_is_a_clear_error(env, store):
    _, _, run = env
    MlflowClient().delete_registered_model_alias(MODEL_NAME, "production")
    with pytest.raises(SyncError, match="alias"):
        run()
