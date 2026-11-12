import numpy as np
import pytest
import torch

from fastmri_knee.data import FastMRITransform
from fastmri_knee.preprocess import PREPROCESS_VERSION, kspace_to_image, preprocess_target


def old_torch_transform(kspace: np.ndarray, mask=None) -> np.ndarray:
    """The pre-F7 implementation, frozen here as the reference for parity."""
    k = torch.from_numpy(kspace).to(torch.complex64)
    if k.ndim == 2:
        k = k[None]
    if mask is not None:
        k = k * torch.from_numpy(mask)
    img = torch.abs(torch.fft.ifft2(k, norm="ortho"))
    img = img / (img.max() + 1e-12)
    h, w = img.shape[-2:]
    hf, wf = (h - 320) // 2, (w - 320) // 2
    return img[..., hf : hf + 320, wf : wf + 320].numpy()


def synth_kspace(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((h, w)) + 1j * rng.standard_normal((h, w))).astype(np.complex64)


@pytest.mark.parametrize("shape", [(640, 368), (640, 372), (321, 321), (400, 333)])
def test_parity_with_torch(shape):
    k = synth_kspace(*shape)
    np.testing.assert_allclose(kspace_to_image(k), old_torch_transform(k), atol=1e-6)


def test_parity_with_mask():
    k = synth_kspace(640, 368)
    mask = (np.random.default_rng(1).random((1, 1, 368)) < 0.3).astype(np.float32)
    np.testing.assert_allclose(kspace_to_image(k, mask=mask), old_torch_transform(k, mask), atol=1e-6)


def test_output_contract():
    out = kspace_to_image(synth_kspace(640, 368))
    assert out.shape == (1, 320, 320)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_accepts_leading_channel_axis():
    k = synth_kspace(640, 368)
    np.testing.assert_array_equal(kspace_to_image(k[None]), kspace_to_image(k))


def test_zero_kspace_is_finite():
    out = kspace_to_image(np.zeros((640, 368), dtype=np.complex64))
    assert np.isfinite(out).all() and out.max() == 0.0


def test_target_normalized_and_cropped():
    t = np.random.default_rng(0).random((640, 372)).astype(np.float32) * 5
    out = preprocess_target(t)
    assert out.shape == (1, 320, 320) and out.dtype == np.float32
    assert out.max() <= 1.0


def test_transform_uses_shared_implementation():
    k = synth_kspace(640, 368)
    t = np.abs(np.fft.ifft2(k)).astype(np.float32)
    image, target, fname, n = FastMRITransform()(torch.from_numpy(k), torch.from_numpy(t), "f.h5", 3)
    np.testing.assert_array_equal(image.numpy(), kspace_to_image(k))
    np.testing.assert_array_equal(target.numpy(), preprocess_target(t))
    assert (fname, n) == ("f.h5", 3)


def test_version_is_string():
    assert PREPROCESS_VERSION == "1"
