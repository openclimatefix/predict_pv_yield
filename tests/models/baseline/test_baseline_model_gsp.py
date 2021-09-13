from predict_pv_yield.models.baseline.last_value import Model
import torch
import pytorch_lightning as pl
from predict_pv_yield.data.dataloader import FakeDataset


from nowcasting_dataset.data_sources.satellite_data_source import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.nwp_data_source import NWP_VARIABLE_NAMES


def test_init():

    m = Model(output_variable="gsp_yield")


def test_model_forward():

    data_configuration = dict(
        batch_size=32,
        history_minutes=30,  #: Number of minutes of history, not including t0.
        forecast_minutes=60,  #: Number of minutes of forecast.
        image_size_pixels=16,
        nwp_channels=NWP_VARIABLE_NAMES,
        sat_channels=SAT_VARIABLE_NAMES,
    )

    # start model
    model = Model(
        forecast_minutes=data_configuration["forecast_minutes"],
        history_minutes=data_configuration["history_minutes"],
        output_variable="gsp_yield",
    )

    # set up fake data
    train_dataset = iter(
        FakeDataset(
            batch_size=data_configuration["batch_size"],
            seq_length_30=model.history_len_30 + model.forecast_len_30 + 1,
        )
    )
    # satellite data
    x = next(train_dataset)

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == data_configuration["batch_size"]
    assert y.shape[1] == data_configuration["forecast_minutes"] // 30


def test_trainer():

    # set up data configuration
    data_configuration = dict(
        batch_size=32,
        history_minutes=30,  #: Number of minutes of history, not including t0.
        forecast_minutes=60,  #: Number of minutes of forecast.
        image_size_pixels=16,
        nwp_channels=NWP_VARIABLE_NAMES,
        sat_channels=SAT_VARIABLE_NAMES,
    )

    # start model
    model = Model(
        forecast_minutes=data_configuration["forecast_minutes"],
        history_minutes=data_configuration["history_minutes"],
        output_variable="gsp_yield",
    )

    # create fake data loader
    train_dataset = FakeDataset(
        batch_size=data_configuration["batch_size"],
        seq_length_5=model.history_len_30 + model.forecast_len_30 + 1,
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # set up trainer
    trainer = pl.Trainer(gpus=0, max_epochs=1)

    # test over training set
    r = trainer.test(model, train_dataloader)
