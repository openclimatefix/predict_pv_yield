import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

logging.basicConfig()
_LOG = logging.getLogger("predict_pv_yield")


class Model(pl.LightningModule):
    def __init__(
        self,
        include_pv_yield: bool = True,
        include_nwp: bool = True,
        forecast_len: int = 6,
        history_len: int = 12,
        number_of_conv3d_layers: int = 4,
        conv3d_channels: int = 32,
        image_size_pixels: int = 64,
        number_sat_channels: int = 12,
        batch_size: int = 64
    ):
        """
        Fairly simply 3d conv model.
        - 3 conv 3d layers,
        - 6 fully connected layers.
        """
        super().__init__()

        self.forecast_len = forecast_len
        self.history_len = history_len
        self.include_pv_yield = include_pv_yield
        self.include_nwp = include_nwp
        self.number_of_conv3d_layers = number_of_conv3d_layers
        self.number_of_nwp_features = 10*19*2*2
        conv3d_channels = conv3d_channels

        self.cnn_output_size = (
            conv3d_channels
            * ((image_size_pixels - 2*self.number_of_conv3d_layers) ** 2)
            * (self.forecast_len + self.history_len + 1 - 2*self.number_of_conv3d_layers)
        )

        self.sat_conv1 = nn.Conv3d(
            in_channels=number_sat_channels,
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
            fc3_in_features += 128*7 # 7 could be (history_len + 1)
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
        for i in range(0, self.number_of_conv3d_layers-1):
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
            nwp_data = x['nwp']
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

    def _training_or_validation_step(self, batch, is_train_step: bool = True):
        # put the batch data through the model
        y_hat = self(batch)

        # get the true result out. Select the first data point, as this is the pv system in the center of the image
        y = batch["pv_yield"][:, -self.forecast_len :, 0]

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = (y_hat - y).abs().mean()

        tag = "Train" if is_train_step else "Validation"
        self.log_dict({f"MSE/{tag}": mse_loss}, on_step=True, on_epoch=True)
        self.log_dict({f"MAE/{tag}": mae_loss}, on_step=True, on_epoch=True)

        return mae_loss

    def training_step(self, batch, batch_idx):
        return self._training_or_validation_step(batch)

    def validation_step(self, batch, batch_idx):
        return self._training_or_validation_step(batch, is_train_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
