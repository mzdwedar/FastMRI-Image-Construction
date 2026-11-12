"""K1: a CPU run on synthetic .h5 writes run_summary.json, and a rerun in the same dir resumes."""

import json
from pathlib import Path

import pytest
from conftest import write_fastmri_h5

pytest.importorskip("pytorch_lightning")
from hydra import compose, initialize_config_dir  # noqa: E402

import fastmri_knee.paths  # noqa: E402
from fastmri_knee.packaging.card import sha256_file  # noqa: E402
from fastmri_knee.paths import CONFIG_PATH  # noqa: E402
from fastmri_knee.preprocess import PREPROCESS_VERSION  # noqa: E402
from fastmri_knee.train import train_fn  # noqa: E402


@pytest.fixture
def workspace(tmp_path):
    for split in ("train", "val"):
        (tmp_path / split).mkdir()
        write_fastmri_h5(tmp_path / split / f"{split}.h5", num_slices=2)
    return tmp_path


def run(workspace: Path, *, max_steps: int, max_epochs: int, num_samples: int = -1) -> dict:
    overrides = [
        f"datamodule.data_root={workspace}",
        f"datamodule.train_root={workspace / 'train'}",
        f"datamodule.val_root={workspace / 'val'}",
        "datamodule.batch_size=1",
        "datamodule.num_workers=0",
        f"datamodule.num_samples={num_samples}",
        f"trainer.checkpoint.path={workspace / 'ckpt'}",
        f"trainer.MLFLOW_TRACKING_URI=file:{workspace / 'mlruns'}",
        "trainer.accelerator=cpu",
        "trainer.source=colab",
        f"trainer.train.max_steps={max_steps}",
        f"trainer.train.max_epochs={max_epochs}",
        "model.num_chans=4",
        "model.num_pools=2",
    ]
    with initialize_config_dir(
        config_dir=str((Path(fastmri_knee.paths.__file__).parent / CONFIG_PATH).resolve()), version_base=None
    ):
        cfg = compose(config_name="config", overrides=overrides)
    return train_fn(cfg)


def test_summary_written_then_rerun_resumes(workspace):
    run(workspace, max_steps=-1, max_epochs=1)  # 2 slices, batch 1 = 2 steps

    ckpt_dir = workspace / "ckpt"
    summary = json.loads((ckpt_dir / "run_summary.json").read_text())
    assert summary["checkpoint"] == "best.ckpt"
    best = ckpt_dir / summary["checkpoint"]
    assert summary["checkpoint_sha256"] == sha256_file(best)
    assert summary["preprocess_version"] == PREPROCESS_VERSION
    assert summary["source"] == "colab"
    assert {"psnr", "ssim", "nmse"} <= summary["val_metrics"].keys()
    assert summary["hyperparameters"]["batch_size"] == 1
    assert summary["global_step"] == 2
    assert summary["resumed_from"] is None
    for key in ("git_sha", "torch_version", "cuda_version", "gpu_name"):
        assert key in summary
    assert (ckpt_dir / "last.ckpt").is_file()

    run(workspace, max_steps=-1, max_epochs=2)

    resumed = json.loads((ckpt_dir / "run_summary.json").read_text())
    assert resumed["resumed_from"] == str(ckpt_dir / "last.ckpt")
    assert resumed["global_step"] == 4  # continued from step 2, did not restart at 0
    assert resumed["checkpoint"] == "best.ckpt"  # the rerun overwrote best in place, no best-v1.ckpt
    assert not (ckpt_dir / "best-v1.ckpt").exists()


def test_num_samples_limits_the_slices_used(workspace):
    run(workspace, max_steps=-1, max_epochs=1, num_samples=1)  # 1 of 2 slices, batch 1 = 1 step

    summary = json.loads((workspace / "ckpt" / "run_summary.json").read_text())
    assert summary["global_step"] == 1


def test_rerunning_a_finished_run_keeps_the_summary_and_its_metrics(workspace):
    run(workspace, max_steps=-1, max_epochs=1)
    summary_file = workspace / "ckpt" / "run_summary.json"
    before = summary_file.read_text()
    assert json.loads(before)["val_metrics"]

    run(workspace, max_steps=-1, max_epochs=1)  # same max_epochs: resumes at the end, trains nothing

    assert summary_file.read_text() == before  # not overwritten by an empty-metrics, resumed_from summary
