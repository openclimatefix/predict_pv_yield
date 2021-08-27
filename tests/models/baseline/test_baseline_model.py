from predict_pv_yield.models.baseline.last_value import Model
import torch
import pytorch_lightning as pl
import numpy as np


from nowcasting_dataset.data_sources.satellite_data_source import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.nwp_data_source import NWP_VARIABLE_NAMES


def test_init():

    m = Model()


class FakeDataset(torch.utils.data.Dataset):
    """Fake dataset."""

    def __init__(self, batch_size=32, seq_length=3, length=32):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "pv_yield": torch.randn(self.batch_size, self.seq_length, 128),
        }


def test_model_forward():

    data_configruation = dict(
        batch_size=32,
        history_len=6,  #: Number of timesteps of history, not including t0.
        forecast_len=12,  #: Number of timesteps of forecast.
        image_size_pixels=16,
        nwp_channels=NWP_VARIABLE_NAMES,
        sat_channels=SAT_VARIABLE_NAMES,
    )

    # start model
    model = Model()

    # set up fake data
    train_dataset = iter(FakeDataset(
        batch_size=data_configruation["batch_size"],
        seq_length=data_configruation["history_len"] + data_configruation["forecast_len"] + 1,
    ))
    # satellite data
    x = next(train_dataset)

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == data_configruation['batch_size']
    assert y.shape[1] == data_configruation['forecast_len']


def test_trainer():

    # set up data configuration
    data_configruation = dict(
        batch_size=32,
        history_len=6,  #: Number of timesteps of history, not including t0.
        forecast_len=12,  #: Number of timesteps of forecast.
        image_size_pixels=16,
        nwp_channels=NWP_VARIABLE_NAMES,
        sat_channels=SAT_VARIABLE_NAMES,
    )

    # start model
    model = Model()

    # create fake data loader
    train_dataset = FakeDataset(
        batch_size=data_configruation["batch_size"],
        seq_length=data_configruation["history_len"] + data_configruation["forecast_len"] + 1,
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # fit model
    trainer = pl.Trainer(gpus=0, max_epochs=1)
    # trainer.fit(model, train_dataloader)

    # test over training set
    r = trainer.test(model, train_dataloader)
    print(r)
