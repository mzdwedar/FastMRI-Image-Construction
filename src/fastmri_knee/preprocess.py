"""k-space -> normalized model input, numpy only.

Single source of truth for preprocessing: training (via `FastMRITransform`) and the serving gateway both call
this, so they cannot drift apart. It must not import torch (enforced by tests/unit/test_preprocess_no_torch.py).

Version "1" reproduces the original behaviour exactly, including the missing ifftshift/fftshift around the FFT
(SPEC-foundation open question 1). Do not change semantics here without bumping PREPROCESS_VERSION: models are
tagged with it and promotion refuses a mismatch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

PREPROCESS_VERSION = "1"
CROP_SIZE: tuple[int, int] = (320, 320)
_EPS = 1e-12

FloatArray = npt.NDArray[np.floating[Any]]
ComplexArray = npt.NDArray[np.complexfloating[Any, Any]]
F32Array = npt.NDArray[np.float32]


def normalize(image: FloatArray) -> F32Array:
    """Divide by the global max so values land in [0, 1]. The epsilon keeps all-zero input finite (all zeros)."""
    image = image.astype(np.float32, copy=False)
    return np.asarray(image / (image.max() + np.float32(_EPS)), dtype=np.float32)


def center_crop(data: npt.NDArray[Any], shape: tuple[int, int] = CROP_SIZE) -> npt.NDArray[Any]:
    """Crop the last two axes to `shape`, centered (same offsets as the original torch implementation)."""
    h, w = data.shape[-2:]
    th, tw = shape
    h_from = (h - th) // 2
    w_from = (w - tw) // 2
    return data[..., h_from : h_from + th, w_from : w_from + tw]


def kspace_to_image_scaled(
    kspace: ComplexArray,
    mask: FloatArray | None = None,
    crop_size: tuple[int, int] = CROP_SIZE,
) -> tuple[F32Array, float]:
    """Like `kspace_to_image`, but also returns the scale the image was divided by (its pre-crop max).

    `normalize` uses the max of whatever array it is given, so pass one slice at a time to get a per-slice
    scale; `image * scale` then restores the original intensity (up to the epsilon).
    """
    if kspace.ndim == 2:
        kspace = kspace[None, ...]
    kspace = kspace.astype(np.complex64, copy=False)
    if mask is not None:
        kspace = kspace * mask
    image = np.abs(np.fft.ifft2(kspace, norm="ortho")).astype(np.float32, copy=False)
    scale = float(image.max())
    return center_crop(normalize(image), crop_size).astype(np.float32, copy=False), scale


def kspace_to_image(
    kspace: ComplexArray,
    mask: FloatArray | None = None,
    crop_size: tuple[int, int] = CROP_SIZE,
) -> F32Array:
    """Single-coil k-space (H, W) or (1, H, W) -> normalized magnitude image of shape (1, *crop_size), float32.

    `mask` (broadcastable against k-space) simulates undersampling.
    """
    return kspace_to_image_scaled(kspace, mask, crop_size)[0]


def preprocess_target(target: FloatArray, crop_size: tuple[int, int] = CROP_SIZE) -> F32Array:
    """Ground-truth reconstruction (H, W) or (1, H, W) -> normalized, cropped (1, *crop_size), float32."""
    if target.ndim == 2:
        target = target[None, ...]
    return center_crop(normalize(target), crop_size).astype(np.float32, copy=False)
