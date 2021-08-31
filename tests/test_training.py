
from predict_pv_yield.training import train
from predict_pv_yield.utils import load_config
import os
from moto import mock_s3

from hydra import compose, initialize


@mock_s3
def test_train():

    # creat fake s3 bucket

    os.environ["NEPTUNE_API_TOKEN"] = "not_at_token"

    initialize(config_path="../configs", job_name="test_app")
    config = compose(config_name="config", overrides=['logger=csv',
                                                      'experiment=example_simple',
                                                      'datamodule.fake_data=true',
                                                      'trainer.fast_dev_run=true'])

    train(config=config)


