#!/usr/bin/env python3

import numpy as np
import os

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from predict_pv_yield.netcdf_dataset import NetCDFDataset, worker_init_fn
from predict_pv_yield.visualisation import plot_example

from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from neptune.new.types import File

from perceiver_pytorch import Perceiver

import logging
logging.basicConfig()
_LOG = logging.getLogger('predict_pv_yield')
_LOG.setLevel(logging.DEBUG)


params = dict(
    # DATA
    # TODO: Everything that relates to the dataset should come automatically
    # from a yaml file stored with the dataset.
    batch_size=32,
    history_len=6,  #: Number of timesteps of history, not including t0.
    forecast_len=12,  #: Number of timesteps of forecast.
    image_size_pixels=32,
    nwp_channels=(
        't', 'dswrf', 'prate', 'r', 'sde', 'si10', 'vis', 'lcc', 'mcc', 'hcc'),
    sat_channels=(
        'HRV', 'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120',
        'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073'),

    # TRAINING
    precision=16  # 16, 32, or 64-bit precision for data.
)


SAT_X_MEAN = np.float32(309000)
SAT_X_STD = np.float32(316387.42073603)
SAT_Y_MEAN = np.float32(519000)
SAT_Y_STD = np.float32(406454.17945938)


TOTAL_SEQ_LEN = params['history_len'] + params['forecast_len'] + 1
EMBEDDING_DIM = 16
NWP_SIZE = len(params['nwp_channels']) * 2 * 2  # channels x width x height
N_DATETIME_FEATURES = 4
PERCEIVER_OUTPUT_SIZE = 512
FC_OUTPUT_SIZE = 8
RNN_HIDDEN_SIZE = 16


def get_dataloaders():
    DATA_PATH = 'gs://solar-pv-nowcasting-data/prepared_ML_training_data/v2/'
    TEMP_PATH = '/home/jack/temp/'

    train_dataset = NetCDFDataset(
        24_900,
        os.path.join(DATA_PATH, 'train'),
        os.path.join(TEMP_PATH, 'train'))

    validation_dataset = NetCDFDataset(
        900,
        os.path.join(DATA_PATH, 'validation'),
        os.path.join(TEMP_PATH, 'validation'))

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

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, **dataloader_config)

    return train_dataloader, validation_dataloader


class LitModel(pl.LightningModule):
    def __init__(
        self,
        history_len=params['history_len'],
        forecast_len=params['forecast_len'],
    ):
        super().__init__()
        self.history_len = history_len
        self.forecast_len = forecast_len

        self.perceiver = Perceiver(
            input_channels=len(params['sat_channels']),
            input_axis=2,
            num_freq_bands=6,
            max_freq=10,
            depth=2,
            num_latents=128,
            latent_dim=64,
            num_classes=PERCEIVER_OUTPUT_SIZE,
        )

        self.fc1 = nn.Linear(
            in_features=PERCEIVER_OUTPUT_SIZE,
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

    def forward(self, x):
        # ******************* Satellite imagery *************************
        # Shape: batch_size, seq_length, width, height, channel
        # TODO: Use optical flow, not actual sat images of the future!
        sat_data = x['sat_data']
        batch_size, seq_len, width, height, n_chans = sat_data.shape

        # Stack timesteps as examples (to make a large batch)
        new_batch_size = batch_size * seq_len
        #                                 0           1       2      3
        sat_data = sat_data.reshape(new_batch_size, width, height, n_chans)

        # Pass data through the network :)
        out = self.perceiver(sat_data)

        out = out.reshape(new_batch_size, PERCEIVER_OUTPUT_SIZE)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer


def main():
    train_dataloader, validation_dataloader = get_dataloaders()
    model = LitModel()
    logger = NeptuneLogger(project='OpenClimateFix/predict-pv-yield')
    logger.log_hyperparams(params)
    _LOG.info(f'logger.version = {logger.version}')
    trainer = pl.Trainer(
        gpus=1, max_epochs=10_000, logger=logger,
        precision=params['precision'])
    trainer.fit(model, train_dataloader)


if __name__ == '__main__':
    main()
