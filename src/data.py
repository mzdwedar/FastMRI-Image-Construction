import os
import subprocess
import tarfile
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader


data_url = os.getenv('url')
tar_path = "knee_singlecoil_val.tar.xz"


class FastMRICustomDataset(Dataset):
    """
    PyTorch Dataset for FastMRI singlecoil .h5 files.

    - Indexes every slice in every file under the given root folder.
    - Returns (kspace, target, fname, slice_num).
    - Designed to be used with a transform like FastMRITransform.

    Args:
        root (str): Path to the folder containing .h5 files.
        transform (callable, optional): Transform applied to (kspace, target, fname, slice_num).
    """
    def __init__(self, root: str, transform=None):
        self.files = [
            os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(".h5")
        ]
        self.examples = []
        for fname in self.files:
            with h5py.File(fname, "r") as hf:
                num_slices = hf["kspace"].shape[0]
                self.examples += [(fname, slice_num) for slice_num in range(num_slices)]
        self.transform = transform

    def __len__(self) -> int:
        """Return the total number of slices across all volumes."""
        return len(self.examples)

    def __getitem__(self, i: int):
        """
        Retrieve one slice of data.

        Args:
            i (int): Index of the slice.

        Returns:
            tuple:
                - kspace (torch.Tensor): Raw k-space data (H, W).
                - target (torch.Tensor or None): Ground truth image (H, W).
                - fname (str): Path to the HDF5 file containing this slice.
                - slice_num (int): Slice index within the volume.
        """
        fname, slice_num = self.examples[i]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][slice_num]
            target = hf["reconstruction_rss"][slice_num] if "reconstruction_rss" in hf else None
        kspace = torch.from_numpy(kspace)
        target = torch.from_numpy(target) if target is not None else None

        if self.transform is not None:
            return self.transform(kspace, target, fname, slice_num)

        return kspace, target, fname, slice_num


# FFT utilities
def fft2c(img: torch.Tensor) -> torch.Tensor:
    """
    Apply centered 2D Fast Fourier Transform (FFT) to a complex image.

    Args:
        img (torch.Tensor): Complex image tensor of shape (..., H, W).
                           Must be torch.complex64 or complex128.

    Returns:
        torch.Tensor: K-space representation of the input image 
                      with the same shape as input.
    """
    return torch.fft.fft2(img, norm="ortho") # pylint: disable=not-callable

def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    """
    Apply centered 2D Inverse Fast Fourier Transform (IFFT) to k-space data.

    Args:
        kspace (torch.Tensor): Complex k-space tensor of shape (..., H, W).
                               Must be torch.complex64 or complex128.

    Returns:
        torch.Tensor: Complex image reconstructed from k-space 
                      with the same shape as input.
    """
    return torch.fft.ifft2(kspace, norm="ortho") # pylint: disable=not-callable

def complex_center_crop(data: torch.Tensor, shape: tuple) -> torch.Tensor:
    """
    Crop a complex tensor to the specified shape, centered in both dimensions.

    Args:
        data (torch.Tensor): Complex tensor of shape (..., H, W).
        shape (tuple): Target crop size (target_height, target_width).

    Returns:
        torch.Tensor: Center-cropped complex tensor of shape (..., th, tw).
    """
    h, w = data.shape[-2:]
    th, tw = shape
    w_from = (w - tw) // 2
    h_from = (h - th) // 2
    return data[..., h_from:h_from+th, w_from:w_from+tw]


# Transform for k-space → input/target
class FastMRITransform:
    """
    Transformation pipeline for FastMRI k-space data.

    - Converts raw k-space to complex torch tensors.
    - Applies an optional undersampling mask (to simulate accelerated scans).
    - Transforms k-space to image domain via IFFT.
    - Returns magnitude image (input) and normalized target.

    Args:
        mask_func (callable, optional): Function that generates an undersampling 
                                        mask given a k-space shape. Defaults to None.
    """
    def __init__(self, mask_func=None):
        self.mask_func = mask_func

    def __call__(self, kspace: torch.Tensor, target: torch.Tensor, fname: str, slice_num: int):
        """
        Apply the transform to one slice.

        Args:
            kspace (torch.Tensor): Raw singlecoil k-space, shape (H, W).
            target (torch.Tensor or None): Ground truth reconstruction_rss for the same slice.
            fname (str): Path to the HDF5 file containing this slice.
            slice_num (int): Slice index inside the volume.

        Returns:
            tuple:
                image (torch.Tensor): Input magnitude image after IFFT (float32, H, W).
                target (torch.Tensor or None): Normalized ground truth magnitude image.
                fname (str): Source filename.
                slice_num (int): Slice index.
        """
        # Convert to complex tensor
        kspace = torch.view_as_complex(kspace.to(torch.float32))

        # Apply undersampling mask if provided
        if self.mask_func is not None:
            mask = self.mask_func(kspace.shape)
            kspace = kspace * mask

        # Inverse FFT to image domain
        image = ifft2c(kspace)
        image = torch.abs(image)  # magnitude image

        # Normalize input and target to [0,1]
        image = image / (image.max() + 1e-12)            
        if target is not None:
            target = target.to(torch.float32)
            target = target / (target.max() + 1e-12)

        return image, target, fname, slice_num



class FastMRIDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for FastMRI singlecoil datasets.

    - Handles train, validation, and test datasets.
    - Applies k-space to image transforms.
    - Supports optional undersampling masks during training.

    Args:
        train_root (str): Path to folder containing training .h5 files.
        val_root (str): Path to folder containing validation .h5 files.
        test_root (str): Path to folder containing test .h5 files.
        mask_func (callable, optional): Function to generate undersampling masks.
        batch_size (int): Batch size for all dataloaders.
        num_workers (int): Number of workers for dataloaders.
    """
    def __init__(
        self,
        data_root:str,
        train_root: str,
        val_root: str,
        test_root: str,
        mask_func=None,
        batch_size: int = 4,
        num_workers: int = 4,

    ):
        super().__init__()
        self.data_root = data_root
        self.train_root = train_root
        self.val_root = val_root
        self.test_root = test_root
        self.mask_func = mask_func
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Placeholder for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """download the dataset from cloud"""
        if not os.path.exists(os.path.join(self.data_root, 'val')):
            cmd = [
                "aria2c",
                "-c",                # continue downloads
                "-x", "16",          # max connections per server
                "-s", "16",          # split downloads
                "-k", "1M",          # piece size
                "--max-tries=5",
                "--retry-wait=10",
                "-o", tar_path,
                data_url
            ]

            # Execute the command
            subprocess.run(cmd, check=True)

            # Path to your downloaded tar.xz file
            extract_path = "data/val"  # directory to extract to

            # Make sure the extraction directory exists
            os.makedirs(extract_path, exist_ok=True)

            # Extract the .tar.xz file
            with tarfile.open(tar_path, "r:xz") as tar:
                tar.extractall(path=extract_path)

            print(f"Extracted {tar_path} to {extract_path}")

            # Remove the original tar.xz file
            os.remove(tar_path)
            print(f"Removed {tar_path}")

    def setup(self, stage=None):
        """
        Create datasets for each stage: 'fit', 'validate', 'test'.
        """
        # TODO: Add data splitting
        
        if stage == "fit" or stage is None:
            self.train_dataset = FastMRICustomDataset(
                root=self.train_root,
                transform=FastMRITransform(mask_func=self.mask_func)
            )
            self.val_dataset = FastMRICustomDataset(
                root=self.val_root,
                transform=FastMRITransform(mask_func=None)  # No mask for validation
            )

        if stage == "test" or stage is None:
            self.test_dataset = FastMRICustomDataset(
                root=self.test_root,
                transform=FastMRITransform(mask_func=None)  # No mask for testing
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )