import tarfile

import h5py
import numpy as np
import pytest

from fastmri_knee.data import DataNotFoundError, FastMRIDataModule, check_data_present
from fastmri_knee.data.prepare import extract, prepare, split_singlecoil_subjects


def write_volume(path, patient_id):
    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=np.zeros((1, 4, 4), dtype=np.complex64))
        f.attrs["patient_id"] = patient_id


def test_missing_data_gives_actionable_error(tmp_path):
    with pytest.raises(DataNotFoundError, match="make data"):
        check_data_present(str(tmp_path / "train"), str(tmp_path / "val"))


def test_datamodule_prepare_data_only_checks(tmp_path, monkeypatch):
    # Training must never shell out to a downloader
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("prepare_data must not run subprocesses"))
    dm = FastMRIDataModule(str(tmp_path), str(tmp_path / "train"), str(tmp_path / "val"), str(tmp_path / "test"))
    with pytest.raises(DataNotFoundError):
        dm.prepare_data()


def test_present_data_passes(tmp_path):
    for d in ("train", "val"):
        (tmp_path / d).mkdir()
        write_volume(tmp_path / d / "a.h5", "p")
    check_data_present(str(tmp_path / "train"), str(tmp_path / "val"))


def test_split_keeps_subjects_together_and_is_reproducible(tmp_path):
    def run(root):
        root.mkdir()
        for i in range(10):
            write_volume(root / f"s{i}_a.h5", f"p{i}")
            write_volume(root / f"s{i}_b.h5", f"p{i}")
        split_singlecoil_subjects(root, root / "train", root / "val")
        return {d: sorted(p.name for p in (root / d).glob("*.h5")) for d in ("train", "val")}

    a, b = run(tmp_path / "a"), run(tmp_path / "b")
    assert a == b
    assert len(a["train"]) == 16 and len(a["val"]) == 4
    train_subjects = {n.split("_")[0] for n in a["train"]}
    assert train_subjects.isdisjoint({n.split("_")[0] for n in a["val"]})


def test_extract_strips_top_level_and_rejects_traversal(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_volume(src / "v.h5", "p")
    good = tmp_path / "good.tar.xz"
    with tarfile.open(good, "w:xz") as t:
        t.add(src / "v.h5", arcname="top/v.h5")
    extract(good, tmp_path / "out")
    assert (tmp_path / "out" / "v.h5").exists()

    bad = tmp_path / "bad.tar.xz"
    with tarfile.open(bad, "w:xz") as t:
        t.add(src / "v.h5", arcname="top/../../evil.h5")
    with pytest.raises(RuntimeError, match="Unsafe"):
        extract(bad, tmp_path / "out2")


def test_prepare_from_local_archive(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(5):
        write_volume(src / f"v{i}.h5", f"p{i}")
    archive = tmp_path / "a.tar.xz"
    with tarfile.open(archive, "w:xz") as t:
        for f in src.glob("*.h5"):
            t.add(f, arcname=f"knee/{f.name}")
    root = tmp_path / "data"
    prepare(root, url=None, archive=archive)
    assert len(list((root / "train").glob("*.h5"))) == 4 and len(list((root / "val").glob("*.h5"))) == 1
    prepare(root, url=None, archive=archive)  # idempotent: nothing left to move, no error
    assert archive.exists()  # a user-supplied archive is never deleted


def test_prepare_without_url_fails_clearly(tmp_path):
    with pytest.raises(SystemExit, match="url"):
        prepare(tmp_path, url=None)
