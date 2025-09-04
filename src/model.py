import torch
import torch.nn.functional as F
from fastmri.models.unet import Unet
import pytorch_lightning as pl
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from utils import nmse

class DeepLab(pl.LightningModule):
    """
    A PyTorch Lightning module for MRI image reconstruction using a UNet architecture.
    
    This model predicts fully-sampled images from undersampled MRI measurements and
    supports training, validation, and testing with metrics including NMSE, PSNR, and SSIM.
    """
    
    def __init__(self):
        """
        Initializes the DeepLab model with UNet, loss functions, and image quality metrics.
        """
        super().__init__()

        self.save_hyperparameters()
        self.model = Unet(
            in_chans=1,
            out_chans=1,
            chans=32,
            num_pool_layers=4,
            drop_prob=0.0,
        )

        # Metrics for validation
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        # Metrics for testing
        self.psnr_test = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_test = StructuralSimilarityIndexMeasure(data_range=1.0)
    
    def forward(self, x):
        """
        Performs a forward pass through the UNet model.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Reconstructed image tensor of shape (B, C, H, W)
        """
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        
        Args:
            batch (tuple): Tuple containing input images and ground truth (x, y)
            batch_idx (int): Index of the batch
            
        Returns:
            torch.Tensor: Computed L1 loss for the batch
        """
        x, y = batch
        logits = self.model(x)
        loss = F.l1_loss(logits, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step and updates metrics.
        
        Args:
            batch (tuple): Tuple containing input images and ground truth (x, y)
            batch_idx (int): Index of the batch
        """
        x, y = batch
        logits = self.model(x)
        loss = F.l1_loss(logits, y)

        nmse_val = nmse(logits, y)
        self.psnr.update(logits, y)
        self.ssim.update(logits, y)
    
        self.log("val_loss", loss, prog_bar=True)
        self.log("nmse", nmse_val, prog_bar=True)
    
    def validation_epoch_end(self):
        """
        Computes and logs PSNR and SSIM at the end of the validation epoch.
        Resets the metrics for the next epoch.
        """
        self.log("psnr", self.psnr.compute(), prog_bar=True)
        self.log("ssim", self.ssim.compute(), prog_bar=True)
        self.psnr.reset()
        self.ssim.reset()    
    
    def test_step(self, batch, batch_idx):
        """
        Performs a single test step and updates metrics.
        
        Args:
            batch (tuple): Tuple containing input images and ground truth (x, y)
            batch_idx (int): Index of the batch
        """
        x, y = batch
        logits = self(x)
        loss = F.l1_loss(logits, y)

        nmse_test = nmse(logits, y)
        self.psnr_test.update(logits, y)
        self.ssim_test.update(logits, y)
      
        self.log("test_loss", loss, prog_bar=True)
        self.log("nmse_test", nmse_test, prog_bar=True)
    
    def on_test_epoch_end(self):
        """
        Computes and logs PSNR and SSIM at the end of the test epoch.
        Resets the metrics for future test evaluations.
        """
        self.log("psnr_test", self.psnr_test.compute(), prog_bar=True)
        self.log("ssim_test", self.ssim_test.compute(), prog_bar=True)
        self.psnr_test.reset()
        self.ssim_test.reset()
        
    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        
        Returns:
            torch.optim.Optimizer: Adam optimizer with learning rate 1e-3
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer