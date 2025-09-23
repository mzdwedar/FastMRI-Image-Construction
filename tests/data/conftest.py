import os
import hashlib
from pathlib import Path
import numpy as np
import torch
import h5py  # type: ignore

from src.data import ifft2c as torch_ifft2c

# ---------- Config: tune these per your project ----------
DATA_DIR = os.environ.get("FASTMRI_DATA_DIR", "data")
SAMPLE_LIMIT = 200
ROUNDTRIP_TOL = 1e-5
NMSE_TOL = 1e-3
PSNR_MIN = 35.0
EMPTY_STD_THRESHOLD = 1e-4
DUPLICATE_HASHES_FAIL = True



def open_k_and_recon(h5path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with h5py.File(h5path, 'r') as f:
        k = f['kspace'][()]
        recon_key = next(k for k in ('reconstruction_esc','reconstruction_rss','recon') if k in f)
        recon = f[recon_key][()]
        attrs = dict(f.attrs)
    return k, recon, attrs

def compute_ifft_magnitude(k: np.ndarray) -> np.ndarray:
    # Use torch-centered IFFT from src.data then convert to numpy magnitude
    kk = k
    if kk.ndim == 3 and kk.shape[0] == 1:
        kk = kk[0]
    k_t = torch.from_numpy(kk).to(torch.complex64)
    img_t = torch_ifft2c(k_t)
    mag_t = torch.abs(img_t)
    mag = mag_t.cpu().numpy()
    return mag

__all__ = [
    "DATA_DIR",
    "SAMPLE_LIMIT",
    "open_k_and_recon",

]
