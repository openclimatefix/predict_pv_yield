from predict_pv_yield.models.conv3d.model import Model
import torch
import pytorch_lightning as pl
import yaml
import predict_pv_yield
from predict_pv_yield.utils import load_config


class FakeDataset(torch.utils.data.Dataset):
    """Fake dataset."""

    def __init__(self, batch_size=32, seq_length=3, width=16, height=16, number_sat_channels=8, length=32):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.width = width
        self.height = height
        self.number_sat_channels = number_sat_channels
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        x = {
            "sat_data": torch.randn(
                self.batch_size, self.seq_length, self.width, self.height, self.number_sat_channels
            ),
            "pv_yield": torch.randn(self.batch_size, self.seq_length, 128),
            "nwp": torch.randn(self.batch_size, 10, self.seq_length, 2, 2),
        }

        # add a nan
        x["pv_yield"][0, 0, :] = float("nan")

        return x


def test_init():

    config_file = f"configs/model/conv3d.yaml"
    config = load_config(config_file)

    _ = Model(**config)


def test_model_forward():

    config_file = f"tests/configs/model/conv3d.yaml"
    config = load_config(config_file)

    # start model
    model = Model(**config)

    # create fake data loader
    train_dataset = FakeDataset(
        batch_size=config["batch_size"],
        width=config["image_size_pixels"],
        height=config["image_size_pixels"],
        number_sat_channels=config["number_sat_channels"],
        seq_length=config["history_len"] + config["forecast_len"] + 1,
    )

    x = next(iter(train_dataset))

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == config["batch_size"]
    assert y.shape[1] == config["forecast_len"]


def test_train():

    config_file = f"tests/configs/model/conv3d.yaml"
    config = load_config(config_file)

    # start model
    model = Model(**config)

    # create fake data loader
    train_dataset = FakeDataset(
        batch_size=config["batch_size"],
        width=config["image_size_pixels"],
        height=config["image_size_pixels"],
        number_sat_channels=config["number_sat_channels"],
        seq_length=config["history_len"] + config["forecast_len"] + 1,
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # fit model
    trainer = pl.Trainer(gpus=0, max_epochs=1)
    trainer.fit(model, train_dataloader)

    # predict over training set
    _ = trainer.predict(model, train_dataloader)
