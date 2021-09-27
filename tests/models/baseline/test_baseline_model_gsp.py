from predict_pv_yield.models.baseline.last_value import Model
import torch
import pytorch_lightning as pl
from nowcasting_dataset.dataset.validate import FakeDataset
from nowcasting_dataset.config.model import Configuration



def test_init():

    _ = Model(output_variable="gsp_yield")


def test_model_forward():
    configuration = Configuration()
    configuration.process.batch_size = 32
    configuration.process.history_minutes = 30
    configuration.process.forecast_minutes = 60
    configuration.process.nwp_image_size_pixels = 16

    # start model
    model = Model(
        forecast_minutes=configuration.process.forecast_minutes,
        history_minutes=configuration.process.history_minutes,
        output_variable="gsp_yield",
    )

    # set up fake data
    train_dataset = iter(FakeDataset(configuration=configuration))
    # satellite data
    x = next(train_dataset)

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == configuration.process.batch_size
    assert y.shape[1] == configuration.process.forecast_minutes // 30


def test_trainer():

    configuration = Configuration()
    configuration.process.batch_size = 32
    configuration.process.history_minutes = 30
    configuration.process.forecast_minutes = 60
    configuration.process.nwp_image_size_pixels = 16

    # start model
    model = Model(
        forecast_minutes=configuration.process.forecast_minutes,
        history_minutes=configuration.process.history_minutes,
        output_variable="gsp_yield",
    )

    # create fake data loader
    train_dataset = FakeDataset(configuration=configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # set up trainer
    trainer = pl.Trainer(gpus=0, max_epochs=1)

    # test over training set
    _ = trainer.test(model, train_dataloader)
