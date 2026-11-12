"""Demo-only k-space undersampling for `/v1/reconstructions?acceleration=N` (SPEC-gateway OQ2)."""

import numpy as np
import numpy.typing as npt


def random_cartesian_mask(width: int, acceleration: int, seed: int = 0) -> npt.NDArray[np.float32]:
    """fastMRI-style random Cartesian column mask, shape (width,), float32.

    A fully sampled centre band of `0.32 / acceleration` of the columns (0.08 at 4x, 0.04 at 8x) plus random
    other columns, so about `width / acceleration` columns are kept. The same seed gives the same mask, so
    results are reproducible. It is meant to be shared by every slice of a volume, like a scanner protocol.
    """
    if acceleration < 2:
        raise ValueError("acceleration must be >= 2")
    num_low = round(width * 0.32 / acceleration)
    prob = (width / acceleration - num_low) / (width - num_low)
    mask = np.random.default_rng(seed).uniform(size=width) < prob
    pad = (width - num_low + 1) // 2
    mask[pad : pad + num_low] = True
    return mask.astype(np.float32)
