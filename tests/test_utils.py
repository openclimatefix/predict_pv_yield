from predict_pv_yield.utils import extras, print_config

from hydra import compose, initialize
import hydra

import os


def test_utils():
    """
    Test that util functions work. This just runs them. Perhaps slightly harder to check they work how they should.
    """
    os.environ["NEPTUNE_API_TOKEN"] = "not_at_token"

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path="../configs", job_name="test_app")
    config = compose(config_name="config")

    extras(config)

    print_config(config)
