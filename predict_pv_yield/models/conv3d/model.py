#!/usr/bin/env python3

import numpy as np
import os

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from nowcasting_dataset.data_sources.satellite_data_source import SAT_VARIABLE_NAMES
from predict_pv_yield.data.dataloader import get_dataloaders

from predict_pv_yield.netcdf_dataset import NetCDFDataset, worker_init_fn
from predict_pv_yield.visualisation import plot_example

from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from neptune.new.types import File

import logging

logging.basicConfig()
_LOG = logging.getLogger("predict_pv_yield")
_LOG.setLevel(logging.DEBUG)


params = dict(
    batch_size=32,
    history_len=6,  #: Number of timesteps of history, not including t0.
    forecast_len=12,  #: Number of timesteps of forecast.
    image_size_pixels=64,
    nwp_channels=("t", "dswrf", "prate", "r", "sde", "si10", "vis", "lcc", "mcc", "hcc"),
    sat_channels=SAT_VARIABLE_NAMES,
)

CHANNELS = 16
KERNEL = 3

EMBEDDING_DIM = 16
NWP_SIZE = 10 * 2 * 2  # channels x width x height
N_DATETIME_FEATURES = 4
CNN_OUTPUT_SIZE = CHANNELS * ((params['image_size_pixels']) ** 2) * 19


class Model(pl.LightningModule):
    def __init__(
        self,
        history_len=6,
        forecast_len=12,
    ):
        """
        Fairly simply 3d conv model.
        - 3 conv 3d layers,
        - 6 fully connected layers.
        """
        super().__init__()
        self.forecast_len = forecast_len
        self.history_len = history_len

        # TODO should take parameters from yaml file (PD 2021-08-17)

        self.sat_conv1 = nn.Conv3d(
            in_channels=len(params["sat_channels"]), out_channels=CHANNELS, kernel_size=(3, 3, 3), padding=1
        )
        self.sat_conv2 = nn.Conv3d(
            in_channels=CHANNELS, out_channels=CHANNELS, kernel_size=(3, 3, 3), padding=1
        )

        self.sat_conv3 = nn.Conv3d(
            in_channels=CHANNELS, out_channels=CHANNELS, kernel_size=(3, 3, 3), padding=1
        )

        self.fc1 = nn.Linear(
            in_features=CNN_OUTPUT_SIZE,
            out_features=128)

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=128)

        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=self.forecast_len)
        # self.fc5 = nn.Linear(in_features=32, out_features=8)
        # self.fc6 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):


        # ******************* Satellite imagery *************************
        # Shape: batch_size, seq_length, width, height, channel
        sat_data = x['sat_data']
        batch_size, seq_len, width, height, n_chans = sat_data.shape

        # Conv3d expects channels to be the 2nd dim, https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        sat_data = sat_data.permute(0, 4, 1, 3, 2)
        # Now shape: batch_size, n_chans, seq_len, height, width

        # Pass data through the network :)
        out = F.relu(self.sat_conv1(sat_data))
        out = F.relu(self.sat_conv2(out))
        out = F.relu(self.sat_conv3(out))

        out = out.reshape(batch_size, CNN_OUTPUT_SIZE)
        out = F.relu(self.fc1(out))

        # Fully connected layers.
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        # print(out.shape)
        # out = F.relu(self.fc5(out))
        # print(out.shape)
        # final_out = self.fc6(out)
        # print(f'{final_out.shape=}')

        out = out.reshape(batch_size, self.forecast_len)

        return out

    def _training_or_validation_step(self, batch, is_train_step: bool = True):
        # put the batch data through the model
        y_hat = self(batch)

        # get the true result out. Select the first data point, as this is the pv system in the center of the image
        y = batch['pv_yield'][:, -self.forecast_len:, 0]

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = (y_hat - y).abs().mean()

        tag = "Train" if is_train_step else "Validation"
        self.log_dict(
            {f'MSE/{tag}': mse_loss}, on_step=is_train_step, on_epoch=True)
        self.log_dict(
            {f'MAE/{tag}': mae_loss}, on_step=is_train_step, on_epoch=True)

        return mae_loss

    def training_step(self, batch, batch_idx):
        return self._training_or_validation_step(batch)

    def validation_step(self, batch, batch_idx):
        return self._training_or_validation_step(batch, is_train_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer





