import os
import pytest
from omegaconf import OmegaConf
from hydra import compose, initialize
from pathlib import Path
from scripts.train import train_fn
import utils

@pytest.mark.training
def test_train_model(dataset_loc: str=None):
    experiment_name = utils.generate_experiment_name(prefix="test_train")
    os.environ['experiment_name'] = experiment_name

    if dataset_loc:
        os.environ["data_root"] = dataset_loc

    with initialize(config_path='../../configs', version_base=None):
        cfg = compose(config_name="config")
        
    metrics = train_fn(cfg)
    utils.delete_experiment(experiment_name=experiment_name)

    assert metrics['val_loss'] < 0.5