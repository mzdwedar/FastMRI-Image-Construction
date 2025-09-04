import json
import datetime
import hydra
from omegaconf import DictConfig
from typing_extensions import Annotated
from src.model import DeepLab
from src.data import FastMRIDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger


@hydra.main(config_path='configs', config_name='config')
def main(cfg: DictConfig):
    """Main train function to train our model"""


    checkpoint_config = ModelCheckpoint(monitor=cfg.trainer.checkpoint.monitor,
                                        save_top_k=cfg.trainer.checkpoint.top_k,
                                        every_n_epochs=cfg.trainer.checkpoint.n_epochs,
                                        dirpath=cfg.trainer.checkpoint.path,
                                        filename=cfg.trainer.checkpoint.filename
                                        )
    
    mlflow_logger = MLFlowLogger(experiment_name=cfg.trainer.train.experiment_name,
                                tracking_uri=cfg.trainer.MLFLOW_TRACKING_URI,
                                save_dir=cfg.trainer.MLFLOW_SAVE_DIR
                                )

    mask_func = None
    data_module = FastMRIDataModule(cfg.datamodule.train.data_root, cfg.datamodule.train_root,
                                    cfg.datamodule.val_root, cfg.datamodule.test_root, 
                                    mask_func, cfg.datamodule.batch_size,
                                    cfg.datamodule.num_workers
                                    )

    model = DeepLab()
    
    trainer = Trainer(accelerator=cfg.trainer.accelerator, devices=cfg.trainer.gpus,
                      max_epochs=cfg.trainer.train.max_epochs, logger=mlflow_logger,
                      strategy=cfg.trainer.train.strategy, callbacks=[checkpoint_config],
                      )
    
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter