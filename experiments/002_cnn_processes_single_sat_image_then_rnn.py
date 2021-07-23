#!/usr/bin/env python3

import numpy as np
import os

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from nowcasting_dataset.dataset import worker_init_fn, NetCDFDataset
from predict_pv_yield.visualisation import plot_example

from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from neptune.new.types import File

import logging
logging.basicConfig()
_LOG = logging.getLogger('predict_pv_yield')
_LOG.setLevel(logging.DEBUG)


params = dict(
    batch_size=32,
    history_len=6,  #: Number of timesteps of history, not including t0.
    forecast_len=12,  #: Number of timesteps of forecast.
    image_size_pixels=32,
    nwp_channels=(
        't', 'dswrf', 'prate', 'r', 'sde', 'si10', 'vis', 'lcc', 'mcc', 'hcc'),
    sat_channels=(
        'HRV', 'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120',
        'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073')
)


SAT_X_MEAN = np.float32(309000)
SAT_X_STD = np.float32(316387.42073603)
SAT_Y_MEAN = np.float32(519000)
SAT_Y_STD = np.float32(406454.17945938)


TOTAL_SEQ_LEN = params['history_len'] + params['forecast_len'] + 1
CHANNELS = 32
N_CHANNELS_LAST_CONV = 4
KERNEL = 3
EMBEDDING_DIM = 16
NWP_SIZE = 10 * 2 * 2  # channels x width x height
N_DATETIME_FEATURES = 4
CNN_OUTPUT_SIZE = N_CHANNELS_LAST_CONV * ((params['image_size_pixels'] - 6) ** 2)
FC_OUTPUT_SIZE = 8
RNN_HIDDEN_SIZE = 16


def get_dataloaders():
    DATA_PATH = 'gs://solar-pv-nowcasting-data/prepared_ML_training_data/v2/'
    TEMP_PATH = '/home/jack/temp/'

    train_dataset = NetCDFDataset(
        12_500,
        os.path.join(DATA_PATH, 'train'),
        os.path.join(TEMP_PATH, 'train'))

    #validation_dataset = NetCDFDataset(1_000, 'gs://solar-pv-nowcasting-data/prepared_ML_training_data/v2/validation/', '/home/jack/temp/validation')

    dataloader_config = dict(
        pin_memory=True,
        num_workers=24,
        prefetch_factor=8,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,

        # Disable automatic batching because dataset
        # returns complete batches.
        batch_size=None,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, **dataloader_config)

    return train_dataloader


class LitModel(pl.LightningModule):
    def __init__(
        self,
        history_len=params['history_len'],
        forecast_len=params['forecast_len'],
    ):
        super().__init__()
        self.history_len = history_len
        self.forecast_len = forecast_len

        self.sat_conv1 = nn.Conv2d(
            in_channels=len(params['sat_channels'])+5,
            out_channels=CHANNELS, kernel_size=KERNEL)
        self.sat_conv2 = nn.Conv2d(
            in_channels=CHANNELS,
            out_channels=CHANNELS, kernel_size=KERNEL)
        self.sat_conv3 = nn.Conv2d(
            in_channels=CHANNELS,
            out_channels=N_CHANNELS_LAST_CONV, kernel_size=KERNEL)

        self.fc1 = nn.Linear(
            in_features=CNN_OUTPUT_SIZE,
            out_features=256)

        self.fc2 = nn.Linear(
            in_features=256 + EMBEDDING_DIM,
            out_features=128)

        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=32)
        self.fc5 = nn.Linear(in_features=32, out_features=FC_OUTPUT_SIZE)

        if EMBEDDING_DIM:
            self.pv_system_id_embedding = nn.Embedding(
                num_embeddings=940,
                embedding_dim=EMBEDDING_DIM)

        self.encoder_rnn = nn.GRU(
            # plus 1 for history
            input_size=FC_OUTPUT_SIZE + N_DATETIME_FEATURES + 1 + NWP_SIZE,
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=2,
            batch_first=True)
        self.decoder_rnn = nn.GRU(
            input_size=FC_OUTPUT_SIZE + N_DATETIME_FEATURES + NWP_SIZE,
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=2,
            batch_first=True)

        self.decoder_fc1 = nn.Linear(
            in_features=RNN_HIDDEN_SIZE,
            out_features=8)
        self.decoder_fc2 = nn.Linear(
            in_features=8,
            out_features=1)

        # EXTRA CHANNELS
        # Center marker
        new_batch_size = params['batch_size'] * TOTAL_SEQ_LEN
        self.center_marker = torch.zeros(
            (
                new_batch_size,
                1,
                params['image_size_pixels'],
                params['image_size_pixels']
            ),
            dtype=torch.float32, device=self.device)
        half_width = params['image_size_pixels'] // 2
        self.center_marker[
            ..., half_width-2:half_width+2, half_width-2:half_width+2] = 1

        # pixel x & y
        pixel_range = (
            torch.arange(params['image_size_pixels'], device=self.device)
            - 64) / 37
        pixel_range = pixel_range.unsqueeze(0).unsqueeze(0)
        self.pixel_x = pixel_range.unsqueeze(-2).expand(
            new_batch_size, 1, params['image_size_pixels'], -1)
        self.pixel_y = pixel_range.unsqueeze(-1).expand(
            new_batch_size, 1, -1, params['image_size_pixels'])

    def forward(self, x):
        # ******************* Satellite imagery *************************
        # Shape: batch_size, seq_length, width, height, channel
        # TODO: Use optical flow, not actual sat images of the future!
        sat_data = x['sat_data']
        batch_size, seq_len, width, height, n_chans = sat_data.shape

        # Stack timesteps as extra examples
        new_batch_size = batch_size * seq_len
        #                                 0           1       2      3
        sat_data = sat_data.reshape(new_batch_size, width, height, n_chans)

        # Conv2d expects channels to be the 2nd dim!
        sat_data = sat_data.permute(0, 3, 1, 2)
        # Now shape: new_batch_size, n_chans, width, height

        # EXTRA CHANNELS
        # geo-spatial x
        x_coords = x['sat_x_coords']  # shape:  batch_size, image_size_pixels
        x_coords = x_coords - SAT_X_MEAN
        x_coords = x_coords / SAT_X_STD
        x_coords = x_coords.unsqueeze(1).expand(-1, width, -1).unsqueeze(1)
        x_coords = x_coords.repeat_interleave(repeats=TOTAL_SEQ_LEN, dim=0)

        # geo-spatial y
        y_coords = x['sat_y_coords']  # shape:  batch_size, image_size_pixels
        y_coords = y_coords - SAT_Y_MEAN
        y_coords = y_coords / SAT_Y_STD
        y_coords = y_coords.unsqueeze(-1).expand(-1, -1, height).unsqueeze(1)
        y_coords = y_coords.repeat_interleave(repeats=TOTAL_SEQ_LEN, dim=0)

        # Concat
        if sat_data.device != self.center_marker.device:
            self.center_marker = self.center_marker.to(sat_data.device)
            self.pixel_x = self.pixel_x.to(sat_data.device)
            self.pixel_y = self.pixel_y.to(sat_data.device)

        sat_data = torch.cat(
            (
                sat_data, self.center_marker,
                x_coords, y_coords, self.pixel_x, self.pixel_y
            ),
            dim=1)

        del x_coords, y_coords

        # Pass data through the network :)
        out = F.relu(self.sat_conv1(sat_data))
        out = F.relu(self.sat_conv2(out))
        out = F.relu(self.sat_conv3(out))

        out = out.reshape(new_batch_size, CNN_OUTPUT_SIZE)
        out = F.relu(self.fc1(out))

        # ********************** Embedding of PV system ID ********************
        if EMBEDDING_DIM:
            pv_row = x['pv_system_row_number'].repeat_interleave(TOTAL_SEQ_LEN)
            pv_embedding = self.pv_system_id_embedding(pv_row)
            out = torch.cat(
                (
                    out,
                    pv_embedding
                ),
                dim=1)

        # Fully connected layers.
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))

        # ******************* PREP DATA FOR RNN *******************************
        out = out.reshape(batch_size, TOTAL_SEQ_LEN, FC_OUTPUT_SIZE)

        # The RNN encoder gets recent history: satellite, NWP,
        # datetime features, and recent PV history.  The RNN decoder
        # gets what we know about the future: satellite, NWP, and
        # datetime features.

        # *********************** NWP Data ************************************
        # Shape: batch_size, channel, seq_length, width, height
        nwp_data = x['nwp'].float()
        # RNN expects seq_len to be dim 1.
        nwp_data = nwp_data.permute(0, 2, 1, 3, 4)
        batch_size, nwp_seq_len, n_nwp_chans, nwp_width, nwp_height = (
            nwp_data.shape)
        nwp_data = nwp_data.reshape(
            batch_size, nwp_seq_len, n_nwp_chans * nwp_width * nwp_height)

        # Concat
        rnn_input = torch.cat(
            (
                out,
                nwp_data,
                x['hour_of_day_sin'].unsqueeze(-1),
                x['hour_of_day_cos'].unsqueeze(-1),
                x['day_of_year_sin'].unsqueeze(-1),
                x['day_of_year_cos'].unsqueeze(-1),
            ),
            dim=2)

        pv_yield_history = x['pv_yield'][:, :self.history_len+1].unsqueeze(-1)
        encoder_input = torch.cat(
            (
                rnn_input[:, :self.history_len+1],
                pv_yield_history
            ),
            dim=2)

        encoder_output, encoder_hidden = self.encoder_rnn(encoder_input)
        decoder_output, _ = self.decoder_rnn(
            rnn_input[:, -self.forecast_len:], encoder_hidden)
        # decoder_output is shape batch_size, seq_len, rnn_hidden_size

        decoder_output = F.relu(self.decoder_fc1(decoder_output))
        decoder_output = self.decoder_fc2(decoder_output)

        return decoder_output.squeeze()

    def _training_or_validation_step(self, batch, is_train_step):
        y_hat = self(batch)
        y = batch['pv_yield'][:, -self.forecast_len:]
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()
        # TODO: Compute correlation coef using np.corrcoef(tensor with
        # shape (2, num_timesteps))[0, 1] on each example, and taking
        # the mean across the batch?
        tag = "Train" if is_train_step else "Validation"
        self.log_dict(
            {f'MSE/{tag}': mse_loss}, on_step=is_train_step, on_epoch=True)
        self.log_dict(
            {f'NMAE/{tag}': nmae_loss}, on_step=is_train_step, on_epoch=True)

        return nmae_loss

    def training_step(self, batch, batch_idx):
        return self._training_or_validation_step(batch, is_train_step=True)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            # Plot example
            model_output = self(batch)
            fig = plot_example(
                batch, model_output, history_len=params['history_len'],
                forecast_len=params['forecast_len'],
                nwp_channels=params['nwp_channels'])
            self.logger.experiment['validation/plot'].log(File.as_image(fig))

        return self._training_or_validation_step(batch, is_train_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


def main():
    train_dataloader = get_dataloaders()
    model = LitModel()
    logger = NeptuneLogger(project='OpenClimateFix/predict-pv-yield')
    logger.log_hyperparams(params)
    _LOG.info(f'logger.version = {logger.version}')
    trainer = pl.Trainer(gpus=1, max_epochs=10_000, logger=logger)
    trainer.fit(model, train_dataloader)


if __name__ == '__main__':
    main()
