from predict_pv_yield.models.conv3d.model import Model
import torch
import pytorch_lightning as pl
from predict_pv_yield.utils import load_config
from nowcasting_dataset.dataset.validate import FakeDataset
from nowcasting_dataset.config.model import Configuration


def test_init():

    config_file = "configs/model/conv3d.yaml"
    config = load_config(config_file)

    _ = Model(**config)


def test_model_forward():

    config_file = "tests/configs/model/conv3d.yaml"
    config = load_config(config_file)

    dataset_configuration = Configuration()
    dataset_configuration.process.nwp_image_size_pixels = 2
    dataset_configuration.process.satellite_image_size_pixels = config['image_size_pixels']
    dataset_configuration.process.history_minutes = config['history_minutes']
    dataset_configuration.process.forecast_minutes = config['forecast_minutes']

    # start model
    model = Model(**config)

    # create fake data loader
    train_dataset = iter(FakeDataset(configuration=dataset_configuration))
    x = next(train_dataset)

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == 32
    assert y.shape[1] == model.forecast_len_5


def test_train():

    config_file = "tests/configs/model/conv3d.yaml"
    config = load_config(config_file)

    dataset_configuration = Configuration()
    dataset_configuration.process.nwp_image_size_pixels = 2
    dataset_configuration.process.satellite_image_size_pixels = config['image_size_pixels']
    dataset_configuration.process.history_minutes = config['history_minutes']
    dataset_configuration.process.forecast_minutes = config['forecast_minutes']

    # start model
    model = Model(**config)

    # create fake data loader
    train_dataset = FakeDataset(configuration=dataset_configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # fit model
    trainer = pl.Trainer(gpus=0, max_epochs=1)
    trainer.fit(model, train_dataloader)

    # predict over training set
    _ = trainer.predict(model, train_dataloader)
