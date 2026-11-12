"""Serving tests run on an untrained U-Net and synthetic FastMRI-shaped files,
so they need neither the dataset nor a GPU (this is what CI runs)."""

import h5py
import numpy as np
import pytest

from fastmri_knee.packaging.export import export, random_init_checkpoint


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "model.ckpt"
    random_init_checkpoint(path)
    return path


@pytest.fixture(scope="session")
def onnx_model(checkpoint, tmp_path_factory):
    from fastmri_knee.model import UnetModel

    module = UnetModel.load_from_checkpoint(checkpoint, map_location="cpu")
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export(module, path)
    return module, path


def write_fastmri_h5(path, num_slices=3, with_target=True):
    """Single-coil knee layout: complex k-space (slices, 640, 368) + 320x320 RSS target."""
    rng = np.random.default_rng(0)
    kspace = (rng.standard_normal((num_slices, 640, 368)) + 1j * rng.standard_normal((num_slices, 640, 368)))
    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=kspace.astype(np.complex64))
        if with_target:
            f.create_dataset("reconstruction_rss", data=rng.random((num_slices, 320, 320), dtype=np.float32))
        f.attrs["patient_id"] = "synthetic"
    return path


@pytest.fixture
def h5_file(tmp_path):
    return write_fastmri_h5(tmp_path / "file_synthetic.h5")
