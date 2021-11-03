import logging

import torch
import torch.nn.functional as F
from torch import nn

from predict_pv_yield.models.base_model import BaseModel
from nowcasting_dataloader.batch import BatchML

logging.basicConfig()
_LOG = logging.getLogger("predict_pv_yield")


class Model(BaseModel):

    name = "conv3d_sat_nwp"

    def __init__(
        self,
        include_pv_yield: bool = True,
        include_nwp: bool = True,
        include_time: bool = True,
        forecast_minutes: int = 30,
        history_minutes: int = 60,
        number_of_conv3d_layers: int = 4,
        conv3d_channels: int = 32,
        image_size_pixels: int = 64,
        number_sat_channels: int = 12,
        number_nwp_channels: int = 10,
        fc1_output_features: int = 128,
        fc2_output_features: int = 128,
        fc3_output_features: int = 64,
        output_variable: str = "pv_yield",
    ):
        """
        3d conv model, that takes in different data streams

        architecture is roughly
        1. satellite image time series goes into many 3d convolution layers.
        2. nwp time series goes into many 3d convolution layers.
        3. Final convolutional layer goes to full connected layer. This is joined by other data inputs like
        - pv yield
        - time variables
        Then there ~4 fully connected layers which end up forecasting the pv yield / gsp into the future

        include_pv_yield: include pv yield data
        include_nwp: include nwp data
        include_time: include hour of data, and day of year as sin and cos components
        forecast_len: the amount of minutes that should be forecasted
        history_len: the amount of historical minutes that are used
        number_of_conv3d_layers, number of convolution 3d layers that are use
        conv3d_channels, the amount of convolution 3d channels
        image_size_pixels: the input satellite image size
        number_sat_channels: number of nwp channels
        fc1_output_features: number of fully connected outputs nodes out of the the first fully connected layer
        fc2_output_features: number of fully connected outputs nodes out of the the second fully connected layer
        fc3_output_features: number of fully connected outputs nodes out of the the third fully connected layer
        output_variable: the output variable to be predicted
        number_nwp_channels: The number of nwp channels there are
        """

        self.include_pv_yield = include_pv_yield
        self.include_nwp = include_nwp
        self.include_time = include_time
        self.number_of_conv3d_layers = number_of_conv3d_layers
        self.number_of_nwp_features = 128
        self.fc1_output_features = fc1_output_features
        self.fc2_output_features = fc2_output_features
        self.fc3_output_features = fc3_output_features
        self.forecast_minutes = forecast_minutes
        self.history_minutes = history_minutes
        self.output_variable = output_variable
        self.number_nwp_channels = number_nwp_channels

        super().__init__()

        conv3d_channels = conv3d_channels

        self.cnn_output_size = (
            conv3d_channels
            * ((image_size_pixels - 2 * self.number_of_conv3d_layers) ** 2)
            * (self.forecast_len_5 + self.history_len_5 + 1 - 2 * self.number_of_conv3d_layers)
        )

        # conv0
        self.sat_conv0 = nn.Conv3d(
            in_channels=number_sat_channels,
            out_channels=conv3d_channels,
            kernel_size=(3, 3, 3),
            padding=0,
        )
        for i in range(0, self.number_of_conv3d_layers - 1):
            layer = nn.Conv3d(
                in_channels=conv3d_channels, out_channels=conv3d_channels, kernel_size=(3, 3, 3), padding=0
            )
            setattr(self, f"sat_conv{i + 1}", layer)

        self.fc1 = nn.Linear(in_features=self.cnn_output_size, out_features=self.fc1_output_features)
        self.fc2 = nn.Linear(in_features=self.fc1_output_features, out_features=self.fc2_output_features)

        # nwp
        if include_nwp:
            self.nwp_conv0 = nn.Conv3d(
                in_channels=number_nwp_channels,
                out_channels=conv3d_channels,
                kernel_size=(3, 3, 3),
                padding=0,
            )
            for i in range(0, self.number_of_conv3d_layers - 1):
                layer = nn.Conv3d(
                    in_channels=conv3d_channels, out_channels=conv3d_channels, kernel_size=(3, 3, 3), padding=0
                )
                setattr(self, f"nwp_conv{i + 1}", layer)

            self.nwp_fc1 = nn.Linear(in_features=self.cnn_output_size, out_features=self.fc1_output_features)
            self.nwp_fc2 = nn.Linear(in_features=self.fc1_output_features, out_features=self.number_of_nwp_features)

        fc3_in_features = self.fc2_output_features
        if include_pv_yield:
            fc3_in_features += self.number_of_samples_per_batch * (self.history_len_30 + 1)
        if include_nwp:
            fc3_in_features += 128
        if include_time:
            fc3_in_features += 4

        self.fc3 = nn.Linear(in_features=fc3_in_features, out_features=self.fc3_output_features)
        self.fc4 = nn.Linear(in_features=self.fc3_output_features, out_features=self.forecast_len)
        # self.fc5 = nn.Linear(in_features=32, out_features=8)
        # self.fc6 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):

        if type(x) == dict:
            x = BatchML(**x)

        # ******************* Satellite imagery *************************
        # Shape: batch_size, channel, seq_length, height, width
        sat_data = x.satellite.data
        batch_size, n_chans, seq_len, height, width = sat_data.shape

        # :) Pass data through the network :)
        out = F.relu(self.sat_conv0(sat_data))
        for i in range(0, self.number_of_conv3d_layers - 1):
            layer = getattr(self, f"sat_conv{i + 1}")
            out = F.relu(layer(out))

        out = out.reshape(batch_size, self.cnn_output_size)

        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # which has shape (batch_size, 128)

        # add pv yield
        if self.include_pv_yield:
            pv_yield_history = x[self.output_variable][:, : self.history_len_30 + 1].nan_to_num(nan=0.0)

            pv_yield_history = pv_yield_history.reshape(
                pv_yield_history.shape[0], pv_yield_history.shape[1] * pv_yield_history.shape[2]
            )
            # join up
            out = torch.cat((out, pv_yield_history), dim=1)

        # *********************** NWP Data ************************************
        if self.include_nwp:

            # shape: batch_size, n_chans, seq_len, height, width
            nwp_data = x.nwp.data

            out_nwp = F.relu(self.nwp_conv0(nwp_data))
            for i in range(0, self.number_of_conv3d_layers - 1):
                layer = getattr(self, f"nwp_conv{i + 1}")
                out_nwp = F.relu(layer(out_nwp))

            # fully connected layers
            out_nwp = out_nwp.reshape(batch_size, self.cnn_output_size)
            out_nwp = F.relu(self.nwp_fc1(out_nwp))
            out_nwp = F.relu(self.nwp_fc2(out_nwp))

            # join with other FC layer
            out = torch.cat((out, out_nwp), dim=1)

        # ########## include time variables #########
        if self.include_time:
            # just take the value now
            x_sin_hour = x.datetime.hour_of_day_sin[:, self.history_len_5 + 1].unsqueeze(dim=1)
            x_cos_hour = x.datetime.hour_of_day_cos[:, self.history_len_5 + 1].unsqueeze(dim=1)
            x_sin_day = x.datetime.day_of_year_sin[:, self.history_len_5 + 1].unsqueeze(dim=1)
            x_cos_day = x.datetime.day_of_year_cos[:, self.history_len_5 + 1].unsqueeze(dim=1)

            # join up
            out = torch.cat((out, x_sin_hour, x_cos_hour, x_sin_day, x_cos_day), dim=1)

        # Fully connected layers.
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        out = out.reshape(batch_size, self.forecast_len)

        return out
