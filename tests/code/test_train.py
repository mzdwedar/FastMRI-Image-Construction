import os

import pytest
import utils
from hydra import compose, initialize

from scripts.train import train_fn


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