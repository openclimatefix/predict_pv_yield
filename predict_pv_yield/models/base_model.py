import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from nowcasting_utils.visualization.visualization import plot_example
from nowcasting_utils.visualization.line import plot_batch_results
from nowcasting_dataset.data_sources.nwp.nwp_data_source import NWP_VARIABLE_NAMES
from nowcasting_utils.models.loss import WeightedLosses
from nowcasting_utils.models.metrics import mae_each_forecast_horizon, mse_each_forecast_horizon
from nowcasting_dataloader.batch import BatchML

import pandas as pd
import numpy as np

import logging

logger = logging.getLogger(__name__)

activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)

default_output_variable = "pv_yield"


class BaseModel(pl.LightningModule):

    # default batch_size
    batch_size = 32

    # results file name
    results_file_name = 'results_epoch'

    # list of results dataframes. This is used to save validation results
    results_dfs = []

    def __init__(self):
        super().__init__()

        self.history_len_5 = self.history_minutes // 5  # the number of historic timestemps for 5 minutes data
        self.forecast_len_5 = self.forecast_minutes // 5  # the number of forecast timestemps for 5 minutes data

        self.history_len_30 = self.history_minutes // 30  # the number of historic timestemps for 5 minutes data
        self.forecast_len_30 = self.forecast_minutes // 30  # the number of forecast timestemps for 5 minutes data

        # the number of historic timesteps for 60 minutes data
        # Note that ceil is taken as for 30 minutes of history data, one history value will be used
        self.history_len_60 = int(np.ceil(self.history_minutes / 60))
        self.forecast_len_60 = self.forecast_minutes // 60 # the number of forecast timestemps for 60 minutes data

        if not hasattr(self, "output_variable"):
            print("setting")
            self.output_variable = default_output_variable

        if self.output_variable == "pv_yield":
            self.forecast_len = self.forecast_len_5
            self.history_len = self.history_len_5
            self.number_of_samples_per_batch = 128
        else:
            self.forecast_len = self.forecast_len_30
            self.history_len = self.history_len_30
            self.number_of_samples_per_batch = 32
        self.number_of_pv_samples_per_batch = 128

        self.weighted_losses = WeightedLosses(forecast_length=self.forecast_len)

    def _training_or_validation_step(self, batch, tag: str):
        """
        batch: The batch data
        tag: either 'Train', 'Validation' , 'Test'
        """

        if type(batch) == dict:
            batch = BatchML(**batch)

        # put the batch data through the model
        y_hat = self(batch)


        # get the true result out. Select the first data point, as this is the pv system in the center of the image
        if self.output_variable == 'gsp_yield':
            y = batch.gsp.gsp_yield
        else:
            y = batch.pv.pv_yield
        y = y[0 : self.batch_size, -self.forecast_len :, 0]

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

    def validation_step(self, batch: BatchML, batch_idx):

        if type(batch) == dict:
            batch = BatchML(**batch)

        INTERESTING_EXAMPLES = (1, 5, 6, 7, 9, 11, 17, 19)
        name = f"validation/plot/epoch_{self.current_epoch}_{batch_idx}"
        if batch_idx in [0, 1, 2, 3, 4]:

            # get model outputs
            model_output = self(batch)

            # make sure the interesting example doesnt go above the batch size
            INTERESTING_EXAMPLES = (i for i in INTERESTING_EXAMPLES if i < self.batch_size)

            for example_i in INTERESTING_EXAMPLES:
                # 1. Plot example
                if 0:
                    fig = plot_example(
                        batch,
                        model_output,
                        history_minutes=self.history_len_5*5,
                        forecast_minutes=self.forecast_len_5*5,
                        nwp_channels=NWP_VARIABLE_NAMES,
                        example_i=example_i,
                        epoch=self.current_epoch,
                        output_variable=self.output_variable
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
                if self.output_variable == 'gsp_yield':
                    y = batch.gsp.gsp_yield[0: self.batch_size, :, 0].cpu().numpy()
                else:
                    y = batch.pv.pv_yield[0 : self.batch_size, :, 0].cpu().numpy()
                y_hat = model_output[0 : self.batch_size].cpu().numpy()
                time = [
                    pd.to_datetime(x, unit="ns") for x in batch.gsp.gsp_datetime_index[0 : self.batch_size].cpu().numpy()
                ]
                time_hat = [
                    pd.to_datetime(x, unit="ns")
                    for x in batch.gsp.gsp_datetime_index[0 : self.batch_size, self.history_len_30 + 1 :].cpu().numpy()
                ]

                # plot and save to logger
                fig = plot_batch_results(model_name=self.name, y=y, y_hat=y_hat, x=time, x_hat=time_hat)
                fig.write_html(f"temp.html")
                try:
                    self.logger.experiment[-1][f'validation/plot/{self.current_epoch}_{batch_idx}'].upload(
                        f"temp.html")
                except:
                    pass

        # ## save to file ###
        # create dataframe with the following columns:
        # t0_datetime_utc, gsp_id,
        # prediction_0, prediction_1, .....
        # truth_0, truth_1, ....
        # get model outputs
        model_output = self(batch).cpu().numpy()
        results = pd.DataFrame(model_output,
                               columns=[f'prediction_{i}' for i in range(model_output.shape[1])])
        results.index.name = 'example_index'
        for i in range(model_output.shape[1]):
            results[f'truth_{i}'] = batch.gsp.gsp_yield[:, -self.forecast_len_30 + i, 0].cpu()
        results['t0_datetime_utc'] = batch.metadata.t0_datetime_utc
        results['gsp_id'] = batch.gsp.gsp_id[:,0].cpu()
        results['batch_index'] = batch_idx

        # append
        if batch_idx == 0:
            self.results_dfs = []
        self.results_dfs.append(results)

        return self._training_or_validation_step(batch, tag="Validation")

    def validation_epoch_end(self, outputs):

        logger.info('Saving results of validation to logger')

        # join all validation step results together
        results_df = pd.concat(self.results_dfs)
        results_df.reset_index(inplace=True)

        # save to csv file
        name_csv = f"{self.results_file_name}_{self.current_epoch}.csv"
        results_df.to_csv(name_csv)

        # upload csv to neptune
        try:
            self.logger.experiment[-1][f'validation/results/epoch_{self.current_epoch}'].upload(name_csv)
        except:
            pass

    def test_step(self, batch, batch_idx):
        self._training_or_validation_step(batch, tag="Test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer
