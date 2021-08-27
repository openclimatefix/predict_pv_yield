import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nowcasting_dataset.data_sources.satellite_data_source import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.nwp_data_source import NWP_VARIABLE_NAMES
from torch import nn

from predict_pv_yield.models.base_model import BaseModel

logging.basicConfig()
_LOG = logging.getLogger("predict_pv_yield")
_LOG.setLevel(logging.DEBUG)

data_configruation_default = dict(
    batch_size=32,
    history_len=6,  #: Number of timesteps of history, not including t0.
    forecast_len=12,  #: Number of timesteps of forecast.
    image_size_pixels=64,
    nwp_channels=NWP_VARIABLE_NAMES,
    sat_channels=SAT_VARIABLE_NAMES,
)

model_configuration_default = dict(conv3d_channels=8, kennel=3, number_of_conv3d_layers=4)


class Model(BaseModel):
    def __init__(
        self,
        data_configruation: dict = data_configruation_default,
        model_configuration: dict = model_configuration_default,
        include_pv_yield: bool = True,
        include_nwp: bool = True,
    ):
        """
        Fairly simply 3d conv model.
        - 3 conv 3d layers,
        - 6 fully connected layers.
        """
        super().__init__()

        self.forecast_len = data_configruation["forecast_len"]
        self.history_len = data_configruation["history_len"]
        self.include_pv_yield = include_pv_yield
        self.include_nwp = include_nwp
        self.number_of_conv3d_layers = model_configuration["number_of_conv3d_layers"]
        self.number_of_nwp_features = 10 * 19 * 2 * 2
        conv3d_channels = model_configuration["conv3d_channels"]

        self.cnn_output_size = (
            conv3d_channels
            * ((data_configruation["image_size_pixels"] - 2 * self.number_of_conv3d_layers) ** 2)
            * (self.forecast_len + self.history_len + 1 - 2 * self.number_of_conv3d_layers)
        )

        self.sat_conv1 = nn.Conv3d(
            in_channels=len(data_configruation["sat_channels"]),
            out_channels=conv3d_channels,
            kernel_size=(3, 3, 3),
            padding=0,
        )
        self.sat_conv2 = nn.Conv3d(
            in_channels=conv3d_channels, out_channels=conv3d_channels, kernel_size=(3, 3, 3), padding=0
        )

        self.fc1 = nn.Linear(in_features=self.cnn_output_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)

        fc3_in_features = 128
        if include_pv_yield:
            fc3_in_features += 128 * 7  # 7 could be (history_len + 1)
        if include_nwp:
            self.fc_nwp = nn.Linear(in_features=self.number_of_nwp_features, out_features=128)
            fc3_in_features += 128

        self.fc3 = nn.Linear(in_features=fc3_in_features, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=self.forecast_len)
        # self.fc5 = nn.Linear(in_features=32, out_features=8)
        # self.fc6 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        # ******************* Satellite imagery *************************
        # Shape: batch_size, seq_length, width, height, channel
        sat_data = x["sat_data"]
        batch_size, seq_len, width, height, n_chans = sat_data.shape

        # Conv3d expects channels to be the 2nd dim, https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        sat_data = sat_data.permute(0, 4, 1, 3, 2)
        # Now shape: batch_size, n_chans, seq_len, height, width

        # :) Pass data through the network :)
        out = F.relu(self.sat_conv1(sat_data))
        for i in range(0, self.number_of_conv3d_layers - 1):
            out = F.relu(self.sat_conv2(out))

        out = out.reshape(batch_size, self.cnn_output_size)

        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # which has shape (batch_size, 128)

        # add pv yield
        if self.include_pv_yield:
            pv_yield_history = x["pv_yield"][:, : self.history_len + 1].nan_to_num(nan=0.0)

            pv_yield_history = pv_yield_history.reshape(
                pv_yield_history.shape[0], pv_yield_history.shape[1] * pv_yield_history.shape[2]
            )
            out = torch.cat((out, pv_yield_history), dim=1)

        # *********************** NWP Data ************************************
        if self.include_nwp:
            # Shape: batch_size, channel, seq_length, width, height
            nwp_data = x["nwp"].float()
            nwp_data = nwp_data.flatten(start_dim=1)

            # fully connected layer
            out_nwp = F.relu(self.fc_nwp(nwp_data))

            # join with other FC layer
            out = torch.cat((out, out_nwp), dim=1)

        # Fully connected layers.
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        out = out.reshape(batch_size, self.forecast_len)

        return out
