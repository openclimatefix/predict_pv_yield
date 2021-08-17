from predict_pv_yield.models.conv3d.model import Model
import torch

from nowcasting_dataset.data_sources.satellite_data_source import SAT_VARIABLE_NAMES


def test_init():

    m = Model()


def test_model_forward():

    model = Model()

    # setup parameters TODO should take parameters from yaml file (PD 2021-08-17)
    batch_size = 32
    seq_length = 17
    width = 16
    height = 16
    channel = len(SAT_VARIABLE_NAMES)

    # set up fake data
    # satelite data
    x = {'sat_data': torch.randn(batch_size, seq_length, width, height, channel)}

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == batch_size
    assert y.shape[1] == 1
