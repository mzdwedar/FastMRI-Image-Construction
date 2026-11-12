from collections import defaultdict

import numpy as np
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from fastmri_knee.model import UnetModel


class ModelDeployment:
    """Holds the U-Net in eval mode and runs plain no-grad forward passes.

    Inference skips the Lightning Trainer: building a Trainer per request costs more
    than the forward pass of a single slice. The Trainer is kept only for /evaluate,
    which reuses the LightningModule's test metrics (PSNR, SSIM, NMSE).
    """

    def __init__(self, checkpoint_path: str, num_threads: int | None = None):
        if num_threads:
            torch.set_num_threads(num_threads)
        self.module = UnetModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
        self.module.eval()

    @torch.inference_mode()
    def infer(self, images: np.ndarray) -> np.ndarray:
        """images: float32 (N, 1, H, W) -> reconstructions, same shape."""
        return self.module(torch.from_numpy(images)).numpy()

    def predict_volumes(self, data_loader: DataLoader) -> dict[str, np.ndarray]:
        """Reconstruct every slice and reassemble per-file volumes ordered by slice index."""
        by_file = defaultdict(list)
        for batch in data_loader:
            images, fnames, slice_nums = batch[0], batch[-2], batch[-1]
            preds = self.infer(images.numpy())
            for fname, slice_num, pred in zip(fnames, slice_nums.tolist(), preds):
                by_file[fname].append((slice_num, pred))
        return {f: np.stack([p for _, p in sorted(s, key=lambda x: x[0])]) for f, s in by_file.items()}

    def evaluate(self, data_loader: DataLoader) -> list[dict]:
        trainer = Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False)
        return trainer.test(self.module, data_loader, verbose=False)
