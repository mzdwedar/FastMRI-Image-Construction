import numpy as np
import pytest
from pathlib import Path
import h5py 
from src.utils import list_files

import conftest as cfg

# ---------- Tests ----------
def get_train_val_h5_files():
    root = Path(cfg.DATA_DIR)
    files = []
    for split in ("train", "val"):
        split_dir = root / split
        if split_dir.exists():
            files.extend(list_files(str(split_dir), limit=cfg.SAMPLE_LIMIT))
    return files

@pytest.mark.parametrize("h5path", get_train_val_h5_files())
def test_h5_keys_and_shapes(h5path):
    """File opens and has expected keys & single-coil layout for train and val splits.""" 
    with h5py.File(h5path, 'r') as f:
        assert 'kspace' in f, f"{h5path} missing 'kspace'"
        
        assert any(k in f for k in ('reconstruction_esc', 'reconstruction_rss', 'recon')), \
            f"{h5path} missing reconstruction key (reconstruction_esc/reconstruction_rss/recon)"
        
        k = f['kspace'] #all slices shape: [*,H,W]

        for n_slice in range(k.shape[0]):
            # singlecoil fastMRI sometimes stores (H,W) or (1,H,W)
            kslice = k[n_slice]

            assert kslice.ndim in (2, 3), f"unexpected kspace ndim {k.ndim} in {h5path}"
            if kslice.ndim == 3:
                # expect first dim==1 for single-coil
                assert kslice.shape[0] == 1, f"expected single-coil file but first dim !=1 in {h5path}"
            
            # dtype should be complex
            assert np.iscomplexobj(kslice), f"kspace not complex in {h5path}"
            # check reconstruction dims roughly match image size
            # read matching reconstruction for this slice
            recon_key = next(k for k in ('reconstruction_esc', 'reconstruction_rss', 'recon') if k in f)
            recon = f[recon_key][n_slice]
            assert recon.ndim == 2, f"expected 2D reconstruction, got {recon.shape} in {h5path}"
            # shapes test: if k is (1,H,W) then recon shape should be (H,W) or close after cropping
            if kslice.ndim == 3:
                _, H, W = kslice.shape
            else:
                H, W = kslice.shape
            # allow that recon may be cropped/padded, so just sanity check sizes not wildly different
            assert abs(recon.shape[0] - H) <= H, "reconstruction height suspiciously different"
            assert abs(recon.shape[1] - W) <= W, "reconstruction width suspiciously different"


@pytest.mark.parametrize("h5path", list_files(cfg.DATA_DIR, limit=cfg.SAMPLE_LIMIT))
def test_no_empty_or_nearzero_slices(h5path):
    k, _, _ = cfg.open_k_and_recon(h5path)
    mag = cfg.compute_ifft_magnitude(k)
    std = float(np.std(mag))
    assert std > cfg.EMPTY_STD_THRESHOLD, f"Slice has near-zero std {std:.3e} possibly empty in {h5path}"


def test_subject_split_consistency():
    """
    If patient/subject id attr exists across files, ensure no overlap between train/val/test folders.
    This is a soft check — it only runs if it can find directories named train/val under DATA_DIR.
    """
    root = Path(cfg.DATA_DIR)
    splits = {}
    split_files = {}
    for split in ("train", "val"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        ids = set()
        id_to_files = {}
        for p in split_dir.rglob("*.h5"):
            with h5py.File(p, 'r') as f:
                # common metadata keys: 'patient_id', 'subject_id'
                pid = None
                for k in ('patient_id', 'subject_id', 'patientid'):
                    if k in f.attrs:
                        pid = str(f.attrs[k])
                        break
                if pid:
                    ids.add(pid)
                    id_to_files.setdefault(pid, []).append(str(p))
        if ids:
            splits[split] = ids
            split_files[split] = id_to_files
    if len(splits) >= 2:
        # ensure disjoint
        for a in splits:
            for b in splits:
                if a == b:
                    continue
                inter = splits[a].intersection(splits[b])
                if len(inter) > 0:
                    overlap_info = []
                    for pid in list(inter)[:10]:
                        files_a = split_files[a].get(pid, [])
                        files_b = split_files[b].get(pid, [])
                        overlap_info.append(
                            f"Subject ID: {pid}\n  {a} files: {files_a}\n  {b} files: {files_b}"
                        )
                    msg = f"Subject overlap between {a} and {b}:\n" + "\n".join(overlap_info)
                    assert False, msg
