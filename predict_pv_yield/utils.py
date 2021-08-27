import yaml
import os
import predict_pv_yield


def load_config(config_file):
    """
    Open yam configruation file, and get rid eof '_target_' line
    """

    # get full path of config file
    path = os.path.dirname(predict_pv_yield.__file__)
    config_file = f"{path}/../{config_file}"


    with open(config_file, "r") as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)

    config.pop("_target_")  # This is only for Hydra

    return config
