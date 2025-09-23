import json
import datetime
import os
import hydra
import logging
import mlflow
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from mlflow.exceptions import MlflowException
from fastmri.data.subsample import RandomMaskFunc

from src.model import UnetModel
from src.data import FastMRIDataModule

def train_fn(cfg: DictConfig):
    """Function to train the model"""
    checkpoint_config = ModelCheckpoint(monitor=cfg.trainer.checkpoint.monitor,
                                        mode=cfg.trainer.checkpoint.mode,
                                        save_top_k=cfg.trainer.checkpoint.top_k,
                                        every_n_epochs=cfg.trainer.checkpoint.n_epochs,
                                        dirpath=cfg.trainer.checkpoint.path,
                                        filename=cfg.trainer.checkpoint.filename,
                                        auto_insert_metric_name=False
                                        )
    
    # Initialize MLflow logger with fallback to local file store if remote is inaccessible
    try:
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.trainer.train.experiment_name,
            tracking_uri=cfg.trainer.MLFLOW_TRACKING_URI,
            save_dir=cfg.trainer.MLFLOW_SAVE_DIR,
            log_model=False
        )
    except MlflowException as exc:
        print(f"Warning: Failed to connect to MLflow at {cfg.trainer.MLFLOW_TRACKING_URI} ({exc}). Falling back to local file store.")
        local_tracking_uri = f"file:{os.path.abspath('mlruns')}"
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.trainer.train.experiment_name,
            tracking_uri=local_tracking_uri,
            save_dir=cfg.trainer.MLFLOW_SAVE_DIR,
            log_model=False
        )
    
    # variable-density random mask (accelerated factor 4, center fraction 0.08)
    mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[4])

    data_module = FastMRIDataModule(
        cfg.datamodule.data_root,
        cfg.datamodule.train_root,
        cfg.datamodule.val_root,
        cfg.datamodule.test_root,
        mask_func,
        cfg.datamodule.batch_size,
        cfg.datamodule.num_workers
    )

    model = UnetModel(cfg.trainer.train.lr, cfg.trainer.train.weight_decay,
                    cfg.trainer.train.optimizer_name, cfg.model.num_chans,
                    cfg.model.num_pools, cfg.model.dropout)
    
    trainer = Trainer(accelerator=cfg.trainer.accelerator, devices=cfg.trainer.gpus,
                      max_epochs=cfg.trainer.train.max_epochs, logger=mlflow_logger,
                      strategy=cfg.trainer.train.strategy, callbacks=[checkpoint_config],
                      )
    
    trainer.fit(model, data_module)

    best_checkpoint = checkpoint_config.best_model_path
    if best_checkpoint:
        with mlflow.start_run(experiment_id=mlflow_logger.experiment_id):
            mlflow.log_artifact(best_checkpoint, artifact_path="checkpoints")

    return trainer.callback_metrics

@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    """Main train function to train our model"""
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    train_fn(cfg)

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter