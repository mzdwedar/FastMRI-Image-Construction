"""Download, extract and split the FastMRI single-coil knee data.

    python -m fastmri_knee.data.prepare --root data            # url from the `url` environment variable
    python -m fastmri_knee.data.prepare --root data --archive knee_singlecoil_val.tar.xz   # already downloaded

Training never does this itself: `FastMRIDataModule.prepare_data` only checks the data is there.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path

import h5py

log = logging.getLogger(__name__)
ARCHIVE_NAME = "knee_singlecoil_val.tar.xz"


def download(url: str, dest: Path) -> None:
    """Resumable download. Uses aria2c when installed (parallel), otherwise curl."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("aria2c"):
        cmd = ["aria2c", "-c", "-x", "16", "-s", "16", "-k", "1M", "--max-tries=5", "--retry-wait=10"]
        cmd += ["-d", str(dest.parent), "-o", dest.name, url]
    elif shutil.which("curl"):
        cmd = ["curl", "-L", "--fail", "-C", "-", "--retry", "5", "-o", str(dest), url]
    else:
        raise RuntimeError("Install aria2c or curl to download the dataset, or pass --archive with a local file.")
    subprocess.run(cmd, check=True)


def extract(archive: Path, root: Path) -> None:
    """Extract into `root`, dropping the archive's top-level folder (same as tar --strip-components=1)."""
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = []
        for m in tar.getmembers():
            parts = Path(m.name).parts[1:]
            if not parts:
                continue
            target = (root / Path(*parts)).resolve()
            if root.resolve() not in target.parents:  # refuse path traversal
                raise RuntimeError(f"Unsafe path in archive: {m.name}")
            m.name = str(Path(*parts))
            members.append(m)
        tar.extractall(root, members=members, filter="data")


def split_singlecoil_subjects(
    data_root: Path, train_root: Path, val_root: Path, train_ratio: float = 0.8, seed: int = 42
) -> None:
    """Move volumes into train/val folders, keeping every subject's volumes together (no patient leakage).

    Subjects are sorted before shuffling, so the split is reproducible for a given seed and file set.
    """
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    subject_to_files: dict[str, list[str]] = defaultdict(list)
    for fname in sorted(os.listdir(data_root)):
        if not fname.endswith(".h5"):
            continue
        with h5py.File(data_root / fname, "r") as f:
            subject_to_files[str(f.attrs["patient_id"])].append(fname)

    subjects = sorted(subject_to_files)
    random.Random(seed).shuffle(subjects)
    num_train = int(train_ratio * len(subjects))

    for subject_ids, target in ((subjects[:num_train], train_root), (subjects[num_train:], val_root)):
        for sid in subject_ids:
            for fname in subject_to_files[sid]:
                shutil.move(str(data_root / fname), str(target / fname))


def prepare(root: Path, url: str | None, archive: Path | None = None, seed: int = 42) -> None:
    train, val = root / "train", root / "val"
    if any(train.glob("*.h5")) and any(val.glob("*.h5")):
        log.info("Data already prepared in %s", root)
        return
    if archive is None:
        if not url:
            raise SystemExit("Set the `url` environment variable (signed FastMRI link) or pass --url / --archive.")
        archive = root / ARCHIVE_NAME
        download(url, archive)
        delete_archive = True
    else:
        delete_archive = False
    extract(archive, root)
    if delete_archive:
        archive.unlink()
    split_singlecoil_subjects(root, train, val, seed=seed)
    log.info("Split into %s and %s", train, val)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data"), help="where data/train and data/val are created")
    parser.add_argument("--url", default=os.getenv("url"), help="signed FastMRI archive URL (default: $url)")
    parser.add_argument("--archive", type=Path, help="use this already-downloaded .tar.xz instead of downloading")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare(args.root, args.url, args.archive, args.seed)


if __name__ == "__main__":
    main()
