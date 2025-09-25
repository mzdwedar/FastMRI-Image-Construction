import torch
import torch.nn.functional as F
from fastmri.models.unet import Unet
import pytorch_lightning as pl
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from .utils import nmse

class UnetModel(pl.LightningModule):
    """
    A PyTorch Lightning module for MRI image reconstruction using a UnetModel architecture.
    
    This model predicts fully-sampled images from undersampled MRI measurements and
    supports training, validation, and testing with metrics including NMSE, PSNR, and SSIM.
    """
    
    def __init__(self, lr, weight_decay, optimizer_name, num_chans, num_pools, dropout):
        """
        Initializes the UnetModel model with UnetModel, loss functions, and image quality metrics.
        """
        super().__init__()

        self.save_hyperparameters(
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "optimizer_name": optimizer_name,
                "num_chans": num_chans,
                "num_pools": num_pools,
                "dropout": dropout,
            }
        )
        
        self.model = Unet(
            in_chans=1,
            out_chans=1,
            chans=num_chans,
            num_pool_layers=num_pools,
            drop_prob=dropout,
        )

        # Metrics for validation
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        # Metrics for testing
        self.psnr_test = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_test = StructuralSimilarityIndexMeasure(data_range=1.0)
    
    def forward(self, x):
        """
        Performs a forward pass through the UnetModel model.
        
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
        x, y, _, _ = batch
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
        x, y, _, _ = batch
        logits = self.model(x)
        loss = F.l1_loss(logits, y)

        nmse_val = nmse(logits, y)
        self.psnr.update(logits, y)
        self.ssim.update(logits, y)
    
        self.log("val_loss", loss, prog_bar=True)
        self.log("nmse", nmse_val, prog_bar=True)
    
    def on_validation_epoch_end(self):
        """
        Computes and logs PSNR and SSIM at the end of the validation epoch.
        Resets the metrics for the next epoch.
        """
        self.log("psnr", self.psnr.compute(), prog_bar=True, on_epoch=True)
        self.log("ssim", self.ssim.compute(), prog_bar=True, on_epoch=True)
        self.psnr.reset()
        self.ssim.reset()    
    
    def test_step(self, batch, batch_idx):
        """
        Performs a single test step and updates metrics.
        
        Args:
            batch (tuple): Tuple containing input images and ground truth (x, y)
            batch_idx (int): Index of the batch
        """
        x, y, _, _ = batch
        logits = self(x)
        loss = F.l1_loss(logits, y)

        nmse_test = nmse(logits, y)
        self.psnr_test.update(logits, y)
        self.ssim_test.update(logits, y)
      
        self.log("test_loss", loss, batch_size=x.size(0), prog_bar=True)
        self.log("nmse_test", nmse_test, batch_size=x.size(0), prog_bar=True)

        return {'test_loss': loss,
                'nmse_test': nmse_test,
                'psnr_test': self.psnr_test.compute(),
                'ssim_test': self.ssim_test.compute()
                }
    
    def on_test_epoch_end(self):
        """
        Computes and logs PSNR and SSIM at the end of the test epoch.
        Resets the metrics for future test evaluations.
        """
        self.log("psnr_test", self.psnr_test.compute(), prog_bar=True, on_epoch=True)
        self.log("ssim_test", self.ssim_test.compute(), prog_bar=True, on_epoch=True)
        self.psnr_test.reset()
        self.ssim_test.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, fname, slice_num = batch
        pred = self(x)

        return {'pred':pred, 'fname':fname, 'slice_num':slice_num}
        
    
    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        
        Returns:
            torch.optim.Optimizer
        """
        if self.hparams.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        return optimizer