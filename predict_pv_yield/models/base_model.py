import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from predict_pv_yield.visualisation.visualisation import plot_example
from nowcasting_dataset.data_sources.nwp_data_source import NWP_VARIABLE_NAMES
from neptune.new.types import File


class BaseModel(pl.LightningModule):
    def _training_or_validation_step(self, batch, is_train_step, tag: str):

        # put the batch data through the model
        y_hat = self(batch)

        # get the true result out. Select the first data point, as this is the pv system in the center of the image
        y = batch["pv_yield"][:, -self.forecast_len :, 0]

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()
        # TODO: Compute correlation coef using np.corrcoef(tensor with
        # shape (2, num_timesteps))[0, 1] on each example, and taking
        # the mean across the batch?
        self.log_dict(
            {f"MSE/{tag}": mse_loss, f"NMAE/{tag}": nmae_loss},
            on_step=is_train_step,
            on_epoch=True,
            sync_dist=True  # Required for distributed training
            # (even multi-GPU on signle machine).
        )

        return nmae_loss

    def training_step(self, batch, batch_idx):
        return self._training_or_validation_step(batch, is_train_step=True, tag="Train")

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
                self.logger.experiment[name].log(File.as_image(fig))
                try:
                    fig.close()
                except Exception as _:
                    # could not close figure
                    pass

        return self._training_or_validation_step(batch, is_train_step=False, tag="Validation")

    def test_step(self, batch, batch_idx):
        self._training_or_validation_step(batch, is_train_step=False, tag="Test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer
