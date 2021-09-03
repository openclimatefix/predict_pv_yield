import logging

from predict_pv_yield.models.base_model import BaseModel


logging.basicConfig()
_LOG = logging.getLogger("predict_pv_yield")
_LOG.setLevel(logging.DEBUG)


class Model(BaseModel):
    name = 'last_value'

    def __init__(
        self,
        forecast_len: int = 12,
        history_len: int = 6,
    ):
        """
        Simple baseline model that takes the last pv yield value and copies it forward
        """

        self.forecast_len = forecast_len
        self.history_len = history_len

        super().__init__()

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
