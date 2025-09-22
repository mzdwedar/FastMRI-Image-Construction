import json
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from src.data import FastMRITransform, FastMRICustomDataset
from .predict import TorchPredictor, get_best_checkpoint, get_best_run_id
from hydra.utils import to_absolute_path

def evaluate_fn(trainer:Trainer, predictor: TorchPredictor,
                data_loader:DataLoader, run_id:str = None):
    """
    Evaluate the model using the provided trainer and data loader.
    Args:
        trainer (Trainer): The trainer instance used for testing the model.
        predictor (TorchPredictor): The predictor instance that contains the model to be evaluated.
        data_loader (DataLoader): The data loader providing the test dataset.
        run_id (str): A unique identifier for the current evaluation run. Default is None.
    Returns:
        dict: A dictionary containing the evaluation metrics, including a timestamp and run ID.
    """
    
    test_metrics = trainer.test(predictor.model, data_loader, verbose=True)

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id if run_id else "None",
        "overall": test_metrics
    }

    logger.info(json.dumps(metrics, indent=4))

    return metrics

logger = logging.getLogger(__name__)
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logging.info("\n%s", OmegaConf.to_yaml(cfg))
    data_path = to_absolute_path(cfg.trainer.predict.batch.data_path)
    dataset = FastMRICustomDataset(data_path, transform=FastMRITransform(mask_func=None))
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    run_id = get_best_run_id(cfg)
    checkpoint_path = get_best_checkpoint(run_id)
    predictor = TorchPredictor.from_checkpoint(checkpoint_path)

    trainer = Trainer(accelerator=cfg.trainer.accelerator,
                      devices=cfg.trainer.gpus
                    )
    
    metrics = evaluate_fn(trainer, predictor, data_loader, run_id)
    
    return metrics

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter