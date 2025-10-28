import os
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from scripts.predict import predict_fn, TorchPredictor
from scripts.evaluate import evaluate_fn

class ModelDeployment:
    def __init__(self, checkpoint_path:str):
        
        trainer_cfg = OmegaConf.load("./configs/trainer.yaml")
        self.cfg = OmegaConf.create({"trainer": trainer_cfg.trainer})
        self.trainer = Trainer(accelerator=self.cfg.trainer.accelerator,
                      devices=self.cfg.trainer.gpus
                    )
        os.makedirs('results', exist_ok=True)
        self.predictor = TorchPredictor.from_checkpoint(checkpoint_path)


    async def predict(self, data_loader: DataLoader):
        predictions = predict_fn(self.trainer, data_loader, self.predictor)

        return predictions

    async def evaluate(self, dataloader: DataLoader):
        metrics = evaluate_fn(self.trainer, self.predictor, dataloader)

        return metrics
