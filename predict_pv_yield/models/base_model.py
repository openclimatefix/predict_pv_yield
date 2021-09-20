import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from nowcasting_utils.visualization.visualization import plot_example
from nowcasting_utils.visualization.line import plot_batch_results
from nowcasting_dataset.data_sources.nwp_data_source import NWP_VARIABLE_NAMES
from nowcasting_utils.models.loss import WeightedLosses
from nowcasting_utils.models.metrics import mae_each_forecast_horizon, mse_each_forecast_horizon

import pandas as pd

import logging

logger = logging.getLogger(__name__)

activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)

default_output_variable = "pv_yield"


class BaseModel(pl.LightningModule):

    # default batch_size
    batch_size = 32

    def __init__(self):
        super().__init__()

        self.history_len_5 = self.history_minutes // 5  # the number of historic timestemps for 5 minutes data
        self.forecast_len_5 = self.forecast_minutes // 5  # the number of forecast timestemps for 5 minutes data

        self.history_len_30 = self.history_minutes // 30  # the number of historic timestemps for 5 minutes data
        self.forecast_len_30 = self.forecast_minutes // 30  # the number of forecast timestemps for 5 minutes data

        if not hasattr(self, "output_variable"):
            print("setting")
            self.output_variable = default_output_variable

        if self.output_variable == "pv_yield":
            self.forecast_len = self.forecast_len_5
            self.history_len = self.history_len_5
        else:
            self.forecast_len = self.forecast_len_30
            self.history_len = self.history_len_30

        self.weighted_losses = WeightedLosses(forecast_length=self.forecast_len)

    def _training_or_validation_step(self, batch, tag: str):
        """
        batch: The batch data
        tag: either 'Train', 'Validation' , 'Test'
        """

        # put the batch data through the model
        y_hat = self(batch)

        # get the true result out. Select the first data point, as this is the pv system in the center of the image
        y = batch[self.output_variable][0 : self.batch_size, -self.forecast_len :, 0]

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()

        # calculate mse, mae with exp weighted loss
        mse_exp = self.weighted_losses.get_mse_exp(output=y_hat, target=y)
        mae_exp = self.weighted_losses.get_mae_exp(output=y_hat, target=y)

        # TODO: Compute correlation coef using np.corrcoef(tensor with
        # shape (2, num_timesteps))[0, 1] on each example, and taking
        # the mean across the batch?
        self.log_dict(
            {f"MSE/{tag}": mse_loss, f"NMAE/{tag}": nmae_loss, f"MSE_EXP/{tag}": mse_exp, f"MAE_EXP/{tag}": mae_exp},
            on_step=True,
            on_epoch=True,
            sync_dist=True  # Required for distributed training
            # (even multi-GPU on signle machine).
        )

        if tag != "Train":
            # add metrics for each forecast horizon
            mse_each_forecast_horizon_metric = mse_each_forecast_horizon(output=y_hat, target=y)
            mae_each_forecast_horizon_metric = mae_each_forecast_horizon(output=y_hat, target=y)

            metrics_mse = {
                f"MSE_forecast_horizon_{i}/{tag}": mse_each_forecast_horizon_metric[i]
                for i in range(self.forecast_len_30)
            }
            metrics_mae = {
                f"MSE_forecast_horizon_{i}/{tag}": mae_each_forecast_horizon_metric[i]
                for i in range(self.forecast_len_30)
            }

            self.log_dict(
                {**metrics_mse, **metrics_mae},
                on_step=True,
                on_epoch=True,
                sync_dist=True  # Required for distributed training
                # (even multi-GPU on signle machine).
            )

        return nmae_loss

    def training_step(self, batch, batch_idx):

        if (batch_idx == 0) and (self.current_epoch == 0):
            return self._training_or_validation_step(batch, tag="Train")
        else:
            return self._training_or_validation_step(batch, tag="Train")

    def validation_step(self, batch, batch_idx):
        INTERESTING_EXAMPLES = (1, 5, 6, 7, 9, 11, 17, 19)
        name = f"validation/plot/epoch{self.current_epoch}"
        if batch_idx == 0:

            # get model outputs
            model_output = self(batch)

            # make sure the interesting example doesnt go above the batch size
            INTERESTING_EXAMPLES = (i for i in INTERESTING_EXAMPLES if i < self.batch_size)

            for example_i in INTERESTING_EXAMPLES:
                # 1. Plot example
                fig = plot_example(
                    batch,
                    model_output,
                    history_minutes=self.history_len_5*5,
                    forecast_minutes=self.forecast_len_5*5,
                    nwp_channels=NWP_VARIABLE_NAMES,
                    example_i=example_i,
                    epoch=self.current_epoch,
                    output_variable='gsp_yield'
                )

                # save fig to log
                self.logger.experiment[-1].log_image(name, fig)
                try:
                    fig.close()
                except Exception as _:
                    # could not close figure
                    pass

                # 2. plot summary batch of predictions and results
                # make x,y data
                y = batch[self.output_variable][0 : self.batch_size, :, 0].cpu().numpy()
                y_hat = model_output[0 : self.batch_size].cpu().numpy()
                time = [
                    pd.to_datetime(x, unit="s") for x in batch["gsp_datetime_index"][0 : self.batch_size].cpu().numpy()
                ]
                time_hat = [
                    pd.to_datetime(x, unit="s")
                    for x in batch["gsp_datetime_index"][0 : self.batch_size, self.history_len_30 + 1 :].cpu().numpy()
                ]

                # plot and save to logger
                fig = plot_batch_results(model_name=self.name, y=y, y_hat=y_hat, x=time, x_hat=time_hat)
                fig.write_html(f"temp.html")
                self.logger.experiment[-1].log_artifact(f"temp.html", f"{name}.html")

        return self._training_or_validation_step(batch, tag="Validation")

    def test_step(self, batch, batch_idx):
        self._training_or_validation_step(batch, tag="Test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer
