from predict_pv_yield.models.baseline.last_value import Model
import torch
import pytorch_lightning as pl
from nowcasting_dataloader.fake import FakeDataset
from nowcasting_dataset.config.model import Configuration


def test_init():

    _ = Model()


def test_model_forward(configuration):

    # start model
    model = Model(forecast_minutes=configuration.input_data.default_forecast_minutes)

    # create fake data loader
    train_dataset = FakeDataset(configuration=configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # satellite data
    x = next(iter(train_dataloader))

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == configuration.process.batch_size
    assert y.shape[1] == configuration.input_data.default_forecast_minutes // 5


def test_trainer(configuration):

    # start model
    model = Model(forecast_minutes=configuration.input_data.default_forecast_minutes)

    # create fake data loader
    train_dataset = FakeDataset(configuration=configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # set up trainer
    trainer = pl.Trainer(gpus=0, max_epochs=1)

    # test over training set
    _ = trainer.test(model, train_dataloader)
