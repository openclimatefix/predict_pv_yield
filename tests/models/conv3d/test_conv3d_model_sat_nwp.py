from predict_pv_yield.models.conv3d.model_sat_nwp import Model
import torch
import pytorch_lightning as pl
from predict_pv_yield.utils import load_config
from predict_pv_yield.data.dataloader import FakeDataset


def test_init():

    config_file = "tests/configs/model/conv3d_sat_nwp.yaml"
    config = load_config(config_file)

    _ = Model(**config)


def test_model_forward():

    config_file = "tests/configs/model/conv3d_sat_nwp.yaml"
    config = load_config(config_file)

    # start model
    model = Model(**config)

    # create fake data loader
    train_dataset = FakeDataset(
        width=config["image_size_pixels"],
        height=config["image_size_pixels"],
        number_sat_channels=config["number_sat_channels"],
        seq_length_5=model.history_len_5 + model.forecast_len_5 + 1,
        nwp_image_size=config["image_size_pixels"],
    )

    x = next(iter(train_dataset))

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == 32
    assert y.shape[1] == model.forecast_len_30


def test_train():

    config_file = "tests/configs/model/conv3d_sat_nwp.yaml"
    config = load_config(config_file)

    # start model
    model = Model(**config)

    # create fake data loader
    train_dataset = FakeDataset(
        width=config["image_size_pixels"],
        height=config["image_size_pixels"],
        number_sat_channels=config["number_sat_channels"],
        seq_length_5=model.history_len_5 + model.forecast_len_5 + 1,
        nwp_image_size=config["image_size_pixels"]
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # fit model
    trainer = pl.Trainer(gpus=0, max_epochs=1)
    trainer.fit(model, train_dataloader)

    # predict over training set
    _ = trainer.predict(model, train_dataloader)
