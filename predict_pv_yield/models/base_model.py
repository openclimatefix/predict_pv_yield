import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from predict_pv_yield.visualisation.visualisation import plot_example
from predict_pv_yield.visualisation.line import plot_batch_results
from nowcasting_dataset.data_sources.nwp_data_source import NWP_VARIABLE_NAMES
from predict_pv_yield.models.loss import WeightedLosses
from predict_pv_yield.models.metrics import mae_each_forecast_horizon, mse_each_forecast_horizon

import pandas as pd


class BaseModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.weighted_losses = WeightedLosses(forecast_length=self.forecast_len)

    def _training_or_validation_step(self, batch, tag: str):

        # put the batch data through the model
        y_hat = self(batch)

        # get the true result out. Select the first data point, as this is the pv system in the center of the image
        y = batch["pv_yield"][:, -self.forecast_len :, 0]

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
            {f"MSE/{tag}": mse_loss,
             f"NMAE/{tag}": nmae_loss,
             f"MSE_EXP/{tag}": mse_exp,
             f"MAE_EXP/{tag}": mae_exp},
            on_step=True,
            on_epoch=True,
            sync_dist=True  # Required for distributed training
            # (even multi-GPU on signle machine).
        )

        if tag != 'Train':
            # add metrics for each forecast horizon
            mse_each_forecast_horizon_metric = mse_each_forecast_horizon(output=y_hat, target=y)
            mae_each_forecast_horizon_metric = mae_each_forecast_horizon(output=y_hat, target=y)

            metrics_mse = {f"MSE_forecast_horizon_{i}/{tag}": mse_each_forecast_horizon_metric[i]
                           for i in range(self.forecast_len)}
            metrics_mae = {f"MSE_forecast_horizon_{i}/{tag}": mae_each_forecast_horizon_metric[i]
                           for i in range(self.forecast_len)}

            self.log_dict(
                {**metrics_mse, **metrics_mae},
                on_step=True,
                on_epoch=True,
                sync_dist=True  # Required for distributed training
                # (even multi-GPU on signle machine).
            )

        return nmae_loss

    def training_step(self, batch, batch_idx):
        return self._training_or_validation_step(batch, tag="Train")

    def validation_step(self, batch, batch_idx):
        INTERESTING_EXAMPLES = (1, 5, 6, 7, 9, 11, 17, 19)
        name = f"validation/plot/epoch{self.current_epoch}"
        if batch_idx == 0:
            # Plot example
            model_output = self(batch)
            for example_i in INTERESTING_EXAMPLES:
                fig = plot_example(
                    batch,
                    model_output,
                    history_len=self.history_len,
                    forecast_len=self.forecast_len,
                    nwp_channels=NWP_VARIABLE_NAMES,
                    example_i=example_i,
                    epoch=self.current_epoch,
                )

                self.logger.experiment[-1].log_image(name, fig)
                try:
                    fig.close()
                except Exception as _:
                    # could not close figure
                    pass

                # plot summary batch of predictions and results
                y = batch["pv_yield"][:, :, 0].detach().numpy()
                y_hat = model_output.detach().numpy()
                time = [pd.to_datetime(x, unit="s") for x in batch["sat_datetime_index"].detach().numpy()]
                time_hat = [pd.to_datetime(x, unit="s") for x in batch["sat_datetime_index"][:, self.history_len + 1:].detach().numpy()]

                fig = plot_batch_results(model_name=self.name, y=y, y_hat=y_hat, x=time, x_hat=time_hat)
                fig.write_html(f"temp.html")
                self.logger.experiment[-1].log_artifact(f"temp.html", f"{name}.html")

        return self._training_or_validation_step(batch, tag="Validation")

    def test_step(self, batch, batch_idx):
        self._training_or_validation_step(batch, tag="Test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer
