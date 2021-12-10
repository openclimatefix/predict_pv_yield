import pytest

from nowcasting_dataset.config.model import Configuration
from predict_pv_yield.utils import load_config
from nowcasting_dataset.consts import SAT_VARIABLE_NAMES


@pytest.fixture()
def configuration():
    configuration = Configuration()
    configuration.input_data = configuration.input_data.set_all_to_defaults()
    configuration.process.batch_size = 32
    configuration.input_data.default_history_minutes = 30
    configuration.input_data.default_forecast_minutes = 60
    configuration.input_data.nwp.nwp_image_size_pixels = 16

    return configuration


@pytest.fixture()
def configuration_conv3d():

    config_file = "tests/configs/model/conv3d.yaml"
    config = load_config(config_file)

    dataset_configuration = Configuration()
    dataset_configuration.input_data = dataset_configuration.input_data.set_all_to_defaults()
    dataset_configuration.input_data.nwp.nwp_image_size_pixels = 2
    dataset_configuration.input_data.satellite.satellite_image_size_pixels = config['image_size_pixels']
    dataset_configuration.input_data.default_history_minutes = config['history_minutes']
    dataset_configuration.input_data.default_forecast_minutes = config['forecast_minutes']
    dataset_configuration.input_data.satellite.satellite_channels = SAT_VARIABLE_NAMES[1:]

    return dataset_configuration


@pytest.fixture()
def configuration_perceiver():

    dataset_configuration = Configuration()
    dataset_configuration.input_data = dataset_configuration.input_data.set_all_to_defaults()
    dataset_configuration.process.batch_size = 2
    dataset_configuration.input_data.nwp.nwp_image_size_pixels = 16
    dataset_configuration.input_data.satellite.satellite_image_size_pixels = 16
    dataset_configuration.input_data.default_history_minutes = 30
    dataset_configuration.input_data.default_forecast_minutes = 120
    dataset_configuration.input_data.nwp.nwp_channels = dataset_configuration.input_data.nwp.nwp_channels[0:10]

    return dataset_configuration
