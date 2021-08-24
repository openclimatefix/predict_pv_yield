from predict_pv_yield.models.conv3d.model import Model
import torch
import pytorch_lightning as pl
import numpy as np


from nowcasting_dataset.data_sources.satellite_data_source import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.nwp_data_source import NWP_VARIABLE_NAMES


def test_init():

    m = Model()


def test_model_forward():

    data_configruation = dict(
        batch_size=32,
        history_len=6,  #: Number of timesteps of history, not including t0.
        forecast_len=12,  #: Number of timesteps of forecast.
        image_size_pixels=16,
        nwp_channels=NWP_VARIABLE_NAMES,
        sat_channels=SAT_VARIABLE_NAMES,
    )

    # model configuration
    model_configuration = dict(conv3d_channels=16, kennel=3)

    # start model
    model = Model(data_configruation=data_configruation, model_configuration=model_configuration)

    # create fake data loader
    train_dataset = FakeDataset(
        batch_size=data_configruation["batch_size"],
        width=data_configruation["image_size_pixels"],
        height=data_configruation["image_size_pixels"],
        channels=len(data_configruation["sat_channels"]),
        seq_length=data_configruation["history_len"] + data_configruation["forecast_len"] + 1,
    )

    x = next(iter(train_dataset))

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == data_configruation['batch_size']
    assert y.shape[1] == data_configruation['forecast_len']


class FakeDataset(torch.utils.data.Dataset):
    """Fake dataset."""

    def __init__(self, batch_size=32, seq_length=3, width=16, height=16, channels=8, length=32):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.width = width
        self.height = height
        self.channels = channels
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "sat_data": torch.randn(self.batch_size, self.seq_length, self.width, self.height, self.channels),
            "pv_yield": torch.randn(self.batch_size, self.seq_length, 128),
        }


def test_train():

    # set up data configuration
    data_configruation = dict(
        batch_size=32,
        history_len=6,  #: Number of timesteps of history, not including t0.
        forecast_len=12,  #: Number of timesteps of forecast.
        image_size_pixels=16,
        nwp_channels=NWP_VARIABLE_NAMES,
        sat_channels=SAT_VARIABLE_NAMES,
    )

    # model configuration
    model_configuration = dict(conv3d_channels=16, kennel=3)

    # start model
    model = Model(data_configruation=data_configruation, model_configuration=model_configuration)

    # create fake data loader
    train_dataset = FakeDataset(
        batch_size=data_configruation["batch_size"],
        width=data_configruation["image_size_pixels"],
        height=data_configruation["image_size_pixels"],
        channels=len(data_configruation["sat_channels"]),
        seq_length=data_configruation["history_len"] + data_configruation["forecast_len"] + 1,
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # fit model
    trainer = pl.Trainer(gpus=0, max_epochs=1)
    trainer.fit(model, train_dataloader)

    # predict over training set
    y = trainer.predict(model, train_dataloader)
