import json
import datetime
import os
import hydra
import logging
import mlflow
from typing import Dict, Optional
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from mlflow.exceptions import MlflowException
from fastmri.data.subsample import RandomMaskFunc

from src.model import UnetModel
from src.data import FastMRIDataModule
import src.utils as utils

class StopOnFirstImprovement(Callback):
    """Stop training immediately after the first improvement in the monitored metric."""

    def __init__(self, monitor: str, mode: str = "min", min_delta: float = 0.0) -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: Optional[float] = None

    def _is_improvement(self, current: float) -> bool:
        if self.best_value is None:
            return False
        if self.mode == "min":
            return (self.best_value - current) > self.min_delta
        return (current - self.best_value) > self.min_delta

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor]
        try:
            current_val = float(current.detach().cpu().item() if hasattr(current, "detach") else float(current))
        except Exception:
            return
        if self.best_value is None:
            self.best_value = current_val
            return
        if self._is_improvement(current_val):
            logging.info("Stopping on first improvement of %s: %.6f -> %.6f", self.monitor, self.best_value, current_val)
            trainer.should_stop = True
        else:
            # update best if improved according to mode for subsequent comparisons
            if (self.mode == "min" and current_val < self.best_value) or (self.mode == "max" and current_val > self.best_value):
                self.best_value = current_val


def train_fn(cfg: DictConfig) -> Dict:
    """Function to train the model"""
    checkpoint_config = ModelCheckpoint(monitor=cfg.trainer.checkpoint.monitor,
                                        mode=cfg.trainer.checkpoint.mode,
                                        save_top_k=cfg.trainer.checkpoint.top_k,
                                        every_n_epochs=cfg.trainer.checkpoint.n_epochs,
                                        dirpath=cfg.trainer.checkpoint.path,
                                        filename=cfg.trainer.checkpoint.filename,
                                        auto_insert_metric_name=False
                                        )

    callbacks = [checkpoint_config]
    if getattr(cfg.trainer.stop_on_first_improvement, "enabled", False):
        callbacks.append(
            StopOnFirstImprovement(
                monitor=cfg.trainer.stop_on_first_improvement.monitor,
                mode=cfg.trainer.stop_on_first_improvement.mode,
                min_delta=cfg.trainer.stop_on_first_improvement.min_delta,
            )
        )
    
    # Initialize MLflow logger with fallback to local file store if remote is inaccessible
    try:
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.trainer.train.experiment_name,
            tracking_uri=cfg.trainer.MLFLOW_TRACKING_URI,
            log_model=False
        )
    except MlflowException as exc:
        print(f"Warning: Failed to connect to MLflow at {cfg.trainer.MLFLOW_TRACKING_URI} ({exc}). Falling back to local file store.")
        local_tracking_uri = f"file:{os.path.abspath('mlruns')}"
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.trainer.train.experiment_name,
            tracking_uri=local_tracking_uri,
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
                      strategy=cfg.trainer.train.strategy, callbacks=callbacks,
                      )
    
    trainer.fit(model, data_module)

    hparams = {"lr": cfg.trainer.train.lr, 
               "weight_decay": cfg.trainer.train.weight_decay,
               "optimizer_name": cfg.trainer.train.optimizer_name,
               "num_chans": cfg.model.num_chans,
               "num_pools": cfg.model.num_pools, 
               "dropout": cfg.model.dropout,
               "batch_size": cfg.datamodule.batch_size, 
               "max_epochs": cfg.trainer.train.max_epochs,
               "accelerator": cfg.trainer.accelerator, 
               "devices": cfg.trainer.gpus,
               "strategy": cfg.trainer.train.strategy
               }
    
    metrics = {
        k: (float(v.detach().cpu().item()) if hasattr(v, "detach") else float(v))
        for k, v in trainer.callback_metrics.items()
    }

    best_checkpoint = checkpoint_config.best_model_path
    if best_checkpoint:
        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.log_artifact(best_checkpoint, artifact_path="checkpoints")
            mlflow.log_metrics(metrics)
            mlflow.log_params(hparams)

    artifact_dir = os.path.join("mlruns", mlflow_logger.experiment_id, mlflow_logger.run_id, "artifacts", "checkpoints")
    checkpoint_path = os.path.join(artifact_dir, os.path.basename(best_checkpoint))

    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "experiment_id": mlflow_logger.experiment_id,
        "run_id": mlflow_logger.run_id,
        "checkpoint_path": checkpoint_path,
        "params": hparams,
        "metrics": metrics,
    }
    
    return d

@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    """Main train function to train our model"""
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    results_fp = cfg.trainer.TRAIN_RESULTS_FILE
    results = train_fn(cfg)

    logging.info(json.dumps(results, indent=2))
    if results_fp:  # pragma: no cover, saving results`
        utils.save_dict(results, results_fp)

    return results['metrics']
if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter