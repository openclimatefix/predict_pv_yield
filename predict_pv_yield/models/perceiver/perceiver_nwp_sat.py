from typing import Iterable
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from perceiver_pytorch import Perceiver

from predict_pv_yield.models.base_model import BaseModel


params = dict(
    # DATA
    # TODO: Everything that relates to the dataset should come automatically
    # from a yaml file stored with the dataset.
    batch_size=32,
    history_minutes=60,  #: Number of timesteps of history, not including t0.
    forecast_minutes=30,  #: Number of timesteps of forecast.
    image_size_pixels=64,
    nwp_channels=("t", "dswrf", "prate", "r", "sde", "si10", "vis", "lcc", "mcc", "hcc"),
    sat_channels=(
        "HRV",
        "IR_016",
        "IR_039",
        "IR_087",
        "IR_097",
        "IR_108",
        "IR_120",
        "IR_134",
        "VIS006",
        "VIS008",
        "WV_062",
        "WV_073",
    ),
)


SAT_X_MEAN = np.float32(309000)
SAT_X_STD = np.float32(316387.42073603)
SAT_Y_MEAN = np.float32(519000)
SAT_Y_STD = np.float32(406454.17945938)


TOTAL_SEQ_LEN = params["history_minutes"] // 5 + params["forecast_minutes"] // 5 + 1
NWP_SIZE = len(params["nwp_channels"]) * 2 * 2  # channels x width x height
N_DATETIME_FEATURES = 4
PERCEIVER_OUTPUT_SIZE = 512
FC_OUTPUT_SIZE = 8
RNN_HIDDEN_SIZE = 16


class Model(BaseModel):

    name = "perceiver_nwp_sat"

    def __init__(
        self,
        history_minutes: int,
        forecast_minutes: int,
        nwp_channels: Iterable[str] = params["nwp_channels"],
        batch_size: int = 32,
        num_latents: int = 128,
        latent_dim: int = 64,
        embedding_dem: int = 16,
        output_variable: str = "pv_yield",
    ):
        self.history_minutes = history_minutes
        self.forecast_minutes = forecast_minutes
        self.nwp_channels = nwp_channels
        self.batch_size = batch_size
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.embedding_dem = embedding_dem
        self.output_variable = output_variable

        super().__init__()

        self.perceiver = Perceiver(
            input_channels=len(params["sat_channels"]) + len(nwp_channels),
            input_axis=2,
            num_freq_bands=6,
            max_freq=10,
            depth=TOTAL_SEQ_LEN,
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            num_classes=PERCEIVER_OUTPUT_SIZE,
            weight_tie_layers=True,
        )

        self.fc1 = nn.Linear(in_features=PERCEIVER_OUTPUT_SIZE, out_features=256)

        self.fc2 = nn.Linear(in_features=256 + self.embedding_dem, out_features=128)

        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=32)
        self.fc5 = nn.Linear(in_features=32, out_features=FC_OUTPUT_SIZE)

        if self.embedding_dem:
            self.pv_system_id_embedding = nn.Embedding(num_embeddings=940, embedding_dim=self.embedding_dem)

        # TODO: Get rid of RNNs!
        self.encoder_rnn = nn.GRU(
            # plus 1 for history
            input_size=FC_OUTPUT_SIZE + N_DATETIME_FEATURES + 1,
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
        )
        self.decoder_rnn = nn.GRU(
            input_size=FC_OUTPUT_SIZE + N_DATETIME_FEATURES,
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
        )

        self.decoder_fc1 = nn.Linear(in_features=RNN_HIDDEN_SIZE, out_features=8)
        self.decoder_fc2 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        # ******************* Satellite imagery *************************
        # Shape: batch_size, seq_length, width, height, channel
        # TODO: Use optical flow, not actual sat images of the future!
        sat_data = x["sat_data"][0 : self.batch_size]
        batch_size, seq_len, width, height, n_chans = sat_data.shape

        # Stack timesteps as examples (to make a large batch)
        new_batch_size = batch_size * seq_len
        #                                 0           1       2      3
        sat_data = sat_data.reshape(new_batch_size, width, height, n_chans)

        # *********************** NWP Data ************************************
        # Shape: batch_size, channel, seq_length, width, height
        nwp_data = x["nwp"][0 : self.batch_size].float()
        # Perciever expects seq_len to be dim 1, and channels at the end
        nwp_data = nwp_data.permute(0, 2, 3, 4, 1)
        batch_size, nwp_seq_len, nwp_width, nwp_height, n_nwp_chans = nwp_data.shape
        nwp_data = nwp_data.reshape(new_batch_size, nwp_width, nwp_height, n_nwp_chans)

        assert nwp_width == width
        assert nwp_height == height

        data = torch.cat((sat_data, nwp_data), dim=-1)

        # Perceiver
        # Pass data through the network :)
        out = self.perceiver(data)

        out = out.reshape(new_batch_size, PERCEIVER_OUTPUT_SIZE)
        out = F.relu(self.fc1(out))

        # ********************** Embedding of PV system ID ********************
        if self.embedding_dem:
            pv_row = (
                x["pv_system_row_number"][0 : self.batch_size, 0].type(torch.IntTensor).repeat_interleave(TOTAL_SEQ_LEN)
            )
            pv_row = pv_row.to(out.device)
            pv_embedding = self.pv_system_id_embedding(pv_row)
            out = torch.cat((out, pv_embedding), dim=1)

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

        ####### Time inputs

        # Concat
        rnn_input = torch.cat(
            (
                out,
                x["hour_of_day_sin"][0 : self.batch_size].unsqueeze(-1),
                x["hour_of_day_cos"][0 : self.batch_size].unsqueeze(-1),
                x["day_of_year_sin"][0 : self.batch_size].unsqueeze(-1),
                x["day_of_year_cos"][0 : self.batch_size].unsqueeze(-1),
            ),
            dim=2,
        )

        if self.output_variable == 'pv_yield':
            # take the history of the pv yield of this system,
            pv_yield_history = x["pv_yield"][0 : self.batch_size][:, : self.history_len_5 + 1, 0].unsqueeze(-1)
            encoder_input = torch.cat((rnn_input[:, : self.history_len_5 + 1], pv_yield_history), dim=2)
        elif self.output_variable == 'gsp_yield':
            # take the history of the gsp yield of this system,
            gsp_history = x[self.output_variable][0: self.batch_size][:, : self.history_len_30 + 1, 0].unsqueeze(-1)
            encoder_input = torch.cat((rnn_input[:, : self.history_len_30 + 1], gsp_history), dim=2)


        encoder_output, encoder_hidden = self.encoder_rnn(encoder_input)
        decoder_output, _ = self.decoder_rnn(rnn_input[:, -self.forecast_len :], encoder_hidden)
        # decoder_output is shape batch_size, seq_len, rnn_hidden_size

        decoder_output = F.relu(self.decoder_fc1(decoder_output))
        decoder_output = self.decoder_fc2(decoder_output)

        return decoder_output.squeeze(dim=-1)