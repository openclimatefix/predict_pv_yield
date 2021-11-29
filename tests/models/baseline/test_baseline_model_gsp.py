from predict_pv_yield.models.baseline.last_value import Model
import torch
import pytorch_lightning as pl
import pandas as pd
from nowcasting_dataloader.fake import FakeDataset
from nowcasting_dataset.config.model import Configuration
import tempfile



def test_init():

    _ = Model(output_variable="gsp_yield")


def test_model_forward(configuration):

    # start model
    model = Model(
        forecast_minutes=configuration.input_data.default_forecast_minutes,
        history_minutes=configuration.input_data.default_history_minutes,
        output_variable="gsp_yield",
    )

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
    assert y.shape[1] == configuration.input_data.default_forecast_minutes // 30


def test_model_validation(configuration):

    # start model
    model = Model(
        forecast_minutes=configuration.input_data.default_forecast_minutes,
        history_minutes=configuration.input_data.default_history_minutes,
        output_variable="gsp_yield",
    )

    # create fake data loader
    train_dataset = FakeDataset(configuration=configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # satellite data
    x = next(iter(train_dataloader))

    # run data through model
    model.validation_step(x, 0)


def test_trainer(configuration):

    # start model
    model = Model(
        forecast_minutes=configuration.input_data.default_forecast_minutes,
        history_minutes=configuration.input_data.default_history_minutes,
        output_variable="gsp_yield",
    )

    # create fake data loader
    train_dataset = FakeDataset(configuration=configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # set up trainer
    trainer = pl.Trainer(gpus=0, max_epochs=1)

    # test over training set
    _ = trainer.test(model, train_dataloader)


def test_trainer_validation(configuration):

    # start model
    model = Model(
        forecast_minutes=configuration.input_data.default_forecast_minutes,
        history_minutes=configuration.input_data.default_history_minutes,
        output_variable="gsp_yield",
    )

    # create fake data loader
    train_dataset = FakeDataset(configuration=configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # set up trainer
    trainer = pl.Trainer(gpus=0, max_epochs=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model.results_file_name = f'{tmpdirname}/temp'

        # test over validation set
        _ = trainer.validate(model, train_dataloader)

        # check csv file of validation results has been made
        results_df = pd.read_csv(f'{model.results_file_name}_0.csv')

        assert len(results_df) == len(train_dataloader) * configuration.process.batch_size
        assert 't0_datetime_utc' in results_df.keys()
        assert 'gsp_id' in results_df.keys()
        for i in range(model.forecast_len_30):
            assert f'truth_{i}' in results_df.keys()
            assert f'prediction_{i}' in results_df.keys()
