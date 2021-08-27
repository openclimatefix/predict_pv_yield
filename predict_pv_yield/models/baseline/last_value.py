import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F


logging.basicConfig()
_LOG = logging.getLogger("predict_pv_yield")
_LOG.setLevel(logging.DEBUG)


class Model(pl.LightningModule):
    def __init__(
        self,
        forecast_len: int = 12,
        history_len: int = 6,
    ):
        """
        Simple baseline model that takes the last pv yield value and copies it forward
        """
        super().__init__()

        self.forecast_len = forecast_len
        self.history_len = history_len

    def forward(self, x):
        # Shape: batch_size, seq_length, n_sites
        pv_yield = x["pv_yield"]

        # take the last value non forecaster value and the first in the pv yeild
        # (this is the pv site we are preditcting for)
        y_hat = pv_yield[:, -self.forecast_len - 1, 0]

        # expand the last valid forward n predict steps
        out = y_hat.unsqueeze(1).repeat(1, self.forecast_len)
        # shape: batch_size, forecast_len

        return out

    def _training_or_validation_step(self, batch, on_step: bool = True, tag: str = "Train"):
        # put the batch data through the model
        y_hat = self(batch)

        # get the true result out. Select the first data point, as this is the pv system in the center of the image
        y = batch["pv_yield"][:, -self.forecast_len :, 0]

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = (y_hat - y).abs().mean()

        self.log_dict({f"MSE/{tag}": mse_loss}, on_step=on_step, on_epoch=True)
        self.log_dict({f"MAE/{tag}": mae_loss}, on_step=on_step, on_epoch=True)

        return mae_loss

    def training_step(self, batch, batch_idx):
        return self._training_or_validation_step(batch)

    def validation_step(self, batch, batch_idx):
        return self._training_or_validation_step(batch, on_step=False, tag="Validation")

    def test_step(self, batch, batch_idx):
        self._training_or_validation_step(batch, on_step=True, tag="Test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
