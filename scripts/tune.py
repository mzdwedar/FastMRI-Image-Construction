import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from mlflow.exceptions import MlflowException
from fastmri.data.subsample import RandomMaskFunc
from src.model import UnetModel
from src.data import FastMRIDataModule
import optuna
from optuna.samplers import TPESampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _objective(trial: optuna.Trial, cfg: DictConfig) -> float:
    try:
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.trainer.train.experiment_name,
            tracking_uri=cfg.trainer.MLFLOW_TRACKING_URI,
            save_dir=cfg.trainer.MLFLOW_SAVE_DIR,
            log_model=False,
        )
    except MlflowException as exc:
        print(f"Warning: Failed to connect to MLflow at {cfg.trainer.MLFLOW_TRACKING_URI} ({exc}). Falling back to local file store.")
        local_tracking_uri = f"file:{os.path.abspath('mlruns')}"
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.trainer.train.experiment_name,
            tracking_uri=local_tracking_uri,
            save_dir=cfg.trainer.MLFLOW_SAVE_DIR,
            log_model=False,
        )

    checkpoint_cb = ModelCheckpoint(
        monitor=cfg.trainer.checkpoint.monitor,
        mode=cfg.trainer.checkpoint.mode,
        save_top_k=cfg.trainer.checkpoint.top_k,
        every_n_epochs=cfg.trainer.checkpoint.n_epochs,
        dirpath=cfg.trainer.checkpoint.path,
        filename=cfg.trainer.checkpoint.filename,
        auto_insert_metric_name=False,
    )

    # Search space
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    num_chans = trial.suggest_categorical("num_chans", [16, 32, 64])
    num_pools = trial.suggest_categorical("num_pools", [3, 4, 5])
    dropout = trial.suggest_float("dropout", 0.0, 0.3, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

    model = UnetModel(lr=lr, weight_decay=weight_decay, optimizer_name=optimizer_name, num_chans=num_chans, num_pools=num_pools, dropout=dropout)
    mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[4])

    datamodule = FastMRIDataModule(
        cfg.datamodule.data_root,
        cfg.datamodule.train_root,
        cfg.datamodule.val_root,
        cfg.datamodule.test_root,
        mask_func,
        batch_size,
        cfg.datamodule.num_workers,
    )
    

    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.gpus,
        max_epochs=cfg.trainer.train.max_epochs,
        logger=mlflow_logger,
        strategy=cfg.trainer.train.strategy,
        callbacks=[checkpoint_cb, PyTorchLightningPruningCallback(trial, monitor=cfg.trainer.checkpoint.monitor)],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    trainer.fit(model, datamodule)

    # Use the same metric as checkpoint monitor
    monitor_metric = cfg.trainer.checkpoint.monitor
    best_score = trainer.callback_metrics.get(monitor_metric)

    # Save checkpoint path on the trial for later retrieval
    try:
        trial.set_user_attr("best_checkpoint", checkpoint_cb.best_model_path)
    except Exception:  # noqa: BLE001
        pass
    return float(best_score.detach().cpu().item()) if hasattr(best_score, "detach") else float(best_score)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def tune_models(cfg: DictConfig):
    logging.info("\n%s", OmegaConf.to_yaml(cfg))

    direction = "minimize" if cfg.trainer.checkpoint.mode == "min" else "maximize"
    study = optuna.create_study(direction=direction, sampler=TPESampler())
    n_trials = cfg.trainer.tune.n_trials
    timeout = cfg.trainer.tune.timeout
    study.optimize(lambda t: _objective(t, cfg), n_trials=n_trials, timeout=timeout)

    print("Best trial:")
    print(f"  Metric ({cfg.trainer.checkpoint.monitor}): {study.best_trial.value}")
    print("  Params:")

    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    best_ckpt = study.best_trial.user_attrs.get("best_checkpoint")
    if best_ckpt:
        print(f"  Best checkpoint path: {best_ckpt}")
    else:
        print("  Best checkpoint path: <not captured>")

    return {
        "metric_name": cfg.trainer.checkpoint.monitor,
        "metric_value": float(study.best_trial.value),
        "params": dict(study.best_trial.params),
        "checkpoint_path": best_ckpt,
        "direction": direction,
    }


if __name__ == "__main__":
    tune_models()  # pylint: disable=no-value-for-parameter