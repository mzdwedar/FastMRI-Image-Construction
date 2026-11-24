import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import h5py


def split_singlecoil_subjects(data_root, train_root, val_root, train_ratio=0.8, seed=42):
    data_root = Path(data_root)
    train_root = Path(train_root)
    val_root = Path(val_root)

    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)

    # Step 1: Collect subject -> files
    subject_to_files = defaultdict(list)
    for fname in os.listdir(data_root):
        if not fname.endswith(".h5"):
            continue
        fpath = data_root / fname
        with h5py.File(fpath, "r") as f:
            sid = str(f.attrs["patient_id"])

        if sid is None:
            raise RuntimeError(f"No subject ID found in {fname}")
        subject_to_files[sid].append(fname)

    # Step 2: Shuffle subjects and split
    subjects = list(subject_to_files.keys())
    random.seed(seed)
    random.shuffle(subjects)

    num_train = int(train_ratio * len(subjects))
    train_subjects = set(subjects[:num_train])
    val_subjects = set(subjects[num_train:])

    # Step 3: Move files by subject
    def move_files(subjects, target_root):
        for sid in subjects:
            for fname in subject_to_files[sid]:
                shutil.move(str(data_root / fname), str(target_root / fname))

    move_files(train_subjects, train_root)
    move_files(val_subjects, val_root)


os.makedirs("./data/train", exist_ok=True)
os.makedirs("./data/val", exist_ok=True)

split_singlecoil_subjects("./data", "./data/train", "./data/val")
