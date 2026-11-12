"""Synthetic single-coil k-space HDF5 files for gateway tests and `curl` demos.

uv run python -m tests.fixtures.synthetic out.h5 --slices 35
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def make_kspace(slices: int = 4, h: int = 640, w: int = 372, seed: int = 0) -> np.ndarray:
    """Smooth random phantoms (a few Gaussian blobs per slice) transformed to centred k-space, complex64."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    images = np.zeros((slices, h, w), dtype=np.float32)
    for i in range(slices):
        for _ in range(5):
            cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
            sy, sx = rng.uniform(20, 70), rng.uniform(20, 70)
            images[i] += rng.uniform(0.3, 1.0) * np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2) / 2)
    return np.fft.fft2(images, norm="ortho").astype(np.complex64)


def write_h5(path: Path, kspace: np.ndarray, dataset: str = "kspace") -> Path:
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset, data=kspace)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--slices", type=int, default=35)
    args = p.parse_args()
    write_h5(args.out, make_kspace(args.slices))


if __name__ == "__main__":
    main()
