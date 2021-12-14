from typing import Iterable
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from perceiver_pytorch import Perceiver

from predict_pv_yield.models.base_model import BaseModel
from nowcasting_dataloader.batch import BatchML

from nowcasting_dataset.consts import NWP_VARIABLE_NAMES, SAT_VARIABLE_NAMES, DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE, DEFAULT_N_GSP_PER_EXAMPLE


params = dict(
    # DATA
    # TODO: Everything that relates to the dataset should come automatically
    # from a yaml file stored with the dataset.
    batch_size=32,
    history_minutes=30,  #: Number of timesteps of history, not including t0.
    forecast_minutes=60,  #: Number of timesteps of forecast.
    image_size_pixels=64,
    nwp_channels=NWP_VARIABLE_NAMES[0:10],
    sat_channels=SAT_VARIABLE_NAMES[1:11],
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


class Conv3dMaxPool(nn.Module):

    def __init__(self, out_channels:int, in_channels:int):
        super().__init__()
        # convultion later, and pad so the output is the same size
        self.sat_conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3), padding=(1, 1, 1)
        )
        # take max pool, keep time sequence the same length
        self.sat_maxpool = nn.MaxPool3d(3, stride=(1, 2, 2), padding=(1, 1, 1))
    def forward(self, x):

        x = self.sat_conv3d(x)
        return self.sat_maxpool(x)


class Model(BaseModel):

    name = "perceiver_conv3d_nwp_sat"

    def __init__(
        self,
        history_minutes: int,
        forecast_minutes: int,
        nwp_channels: Iterable[str] = params["nwp_channels"],
        batch_size: int = 32,
        num_latents: int = 128,
        latent_dim: int = 64,
        embedding_dem: int = 0,
        output_variable: str = "pv_yield",
        conv3d_channels: int = 16,
        use_future_satellite_images: bool = False,  # option not to use future sat images
        include_pv_or_gsp_yield_history: bool = False,
        include_pv_yield_history: int = True,
        include_pv_gsp_coordinates: int = False,
        number_pv_systems: int = DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
        number_gsps: int = DEFAULT_N_GSP_PER_EXAMPLE,
    ):
        """
        Idea is to have a conv3d (+max pool) layer before both sat and nwp data go into perceiver model.
        """
        self.history_minutes = history_minutes
        self.forecast_minutes = forecast_minutes
        self.nwp_channels = nwp_channels
        self.batch_size = batch_size
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.embedding_dem = embedding_dem
        self.output_variable = output_variable
        self.use_future_satellite_images = use_future_satellite_images
        self.include_pv_yield_history = include_pv_yield_history
        self.include_pv_or_gsp_yield_history = include_pv_or_gsp_yield_history
        self.include_pv_gsp_coordinates = include_pv_gsp_coordinates
        self.number_pv_systems= number_pv_systems
        self.number_gsps = number_gsps

        super().__init__()

        self.sat_conv3d_maxpool = Conv3dMaxPool(out_channels=conv3d_channels, in_channels=len(params['sat_channels']))
        self.nwp_conv3d_maxpool = Conv3dMaxPool(out_channels=conv3d_channels, in_channels=len(nwp_channels))

        self.perceiver = Perceiver(
            input_channels=2*conv3d_channels,
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

        rnn_input_size = FC_OUTPUT_SIZE
        if self.include_pv_or_gsp_yield_history:
            rnn_input_size += 1
        if self.include_pv_yield_history:
            rnn_input_size += 128
        if self.include_pv_gsp_coordinates:
            rnn_input_size += 2*(self.number_pv_systems + self.number_gsps)

        # TODO: Get rid of RNNs!
        self.encoder_rnn = nn.GRU(
            # plus 1 for history
            input_size=rnn_input_size,
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
        )
        self.decoder_rnn = nn.GRU(
            input_size=FC_OUTPUT_SIZE,
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
        )

        self.decoder_fc1 = nn.Linear(in_features=RNN_HIDDEN_SIZE, out_features=8)
        self.decoder_fc2 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):

        if type(x) == dict:
            x = BatchML(**x)

        # ******************* Satellite imagery *************************
        # Shape: batch_size, channel, seq_length, height, width
        # TODO: Use optical flow, not actual sat images of the future!
        sat_data = x.satellite.data[0 : self.batch_size].float()

        if not self.use_future_satellite_images:
            sat_data[:, -self.forecast_len_5: ] = 0  # This might not be the best way to do it

        sat_data = self.sat_conv3d_maxpool(sat_data)
        sat_data = sat_data.permute(0, 2, 3, 4, 1)

        # Stack timesteps as examples (to make a large batch)
        batch_size, seq_len, width, height, n_chans = sat_data.shape
        new_batch_size = batch_size * seq_len
        #                                 0           1       2      3
        sat_data = sat_data.reshape(new_batch_size, width, height, n_chans)

        # *********************** NWP Data ************************************
        # Shape: batch_size, seq_length, width, height, channel
        nwp_data = x.nwp.data[0 : self.batch_size].float()
        nwp_data = self.nwp_conv3d_maxpool(nwp_data)
        # Perciever expects seq_len to be dim 1, and channels at the end
        nwp_data = nwp_data.permute(0, 2, 3, 4, 1)
        batch_size, nwp_seq_len, nwp_width, nwp_height, n_nwp_chans = nwp_data.shape

        # nwp to have the same sel_len as sat. I think there is a better solution than this
        nwp_data_zeros = torch.zeros(size=(batch_size, seq_len - nwp_seq_len, nwp_width, nwp_height, n_nwp_chans), device=nwp_data.device)
        nwp_data = torch.cat([nwp_data, nwp_data_zeros], dim=1)

        nwp_data = nwp_data.reshape(new_batch_size, nwp_width, nwp_height, n_nwp_chans)

        # v15 the width and height are a lot less, so lets expand the sat data. There should be a better way
        sat_data_zeros = torch.zeros(size=(new_batch_size, nwp_width - width, height, n_chans),
                                     device=sat_data.device)
        sat_data = torch.cat([sat_data, sat_data_zeros], dim=1)
        sat_data_zeros = torch.zeros(size=(new_batch_size, nwp_width, nwp_height - height, n_chans),
                                     device=sat_data.device)
        sat_data = torch.cat([sat_data, sat_data_zeros], dim=2)
        new_batch_size, sat_width, sat_height, sat_n_chans = sat_data.shape

        assert nwp_width == sat_height, f'widths should be the same({nwp_width},{sat_width})'
        assert nwp_height == sat_height, f'heights should be the same({nwp_height},{sat_height})'

        data = torch.cat((sat_data, nwp_data), dim=-1)

        # Perceiver
        # Pass data through the network :)
        out = self.perceiver(data)

        out = out.reshape(new_batch_size, PERCEIVER_OUTPUT_SIZE)
        out = F.relu(self.fc1(out))

        # ********************** Embedding of PV system ID ********************
        if self.embedding_dem:
            pv_row = (
                x.pv.pv_system_row_number[0 : self.batch_size, 0].type(torch.IntTensor).repeat_interleave(TOTAL_SEQ_LEN)
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
            ),
            dim=2,
        )

        if self.include_pv_or_gsp_yield_history:
            if self.output_variable == 'pv_yield':
                # take the history of the pv yield of this system,
                pv_yield_history = x.pv.pv_yield[0 : self.batch_size][:, : self.history_len_5 + 1, 0].unsqueeze(-1).float()
                encoder_input = torch.cat((rnn_input[:, : self.history_len_5 + 1], pv_yield_history), dim=2)
            elif self.output_variable == 'gsp_yield':
                # take the history of the gsp yield of this system,
                gsp_history = x.gsp.gsp_yield[0: self.batch_size][:, : self.history_len_30 + 1, 0].unsqueeze(-1).float()
                encoder_input = torch.cat((rnn_input[:, : self.history_len_30 + 1], gsp_history), dim=2)

        # add the pv yield history. This can be used if trying to predict gsp
        if self.include_pv_yield_history:
            pv_yield_history = (
                x.pv.pv_yield[:self.batch_size].nan_to_num(nan=0.0).float()
            )
            # remove future pv
            pv_yield_history[:, self.history_len_5 + 1:] = 0.0

            encoder_input = torch.cat((rnn_input, pv_yield_history), dim=2)

        if self.include_pv_gsp_coordinates:

            pv_x_coordiantes = x.pv.pv_system_x_coords[:self.batch_size].nan_to_num(nan=0.0).float()
            pv_y_coordiantes = x.pv.pv_system_y_coords[:self.batch_size].nan_to_num(nan=0.0).float()

            gsp_x_coordiantes = x.gsp.gsp_x_coords[:self.batch_size].nan_to_num(nan=0.0).float()
            gsp_y_coordiantes = x.gsp.gsp_y_coords[:self.batch_size].nan_to_num(nan=0.0).float()

            input = torch.cat((pv_x_coordiantes, pv_y_coordiantes, gsp_x_coordiantes, gsp_y_coordiantes), dim=1)

            # currently this doesnt work TODO
            encoder_input = torch.cat((rnn_input, input), dim=2)

        encoder_output, encoder_hidden = self.encoder_rnn(encoder_input)
        decoder_output, _ = self.decoder_rnn(rnn_input[:, -self.forecast_len :], encoder_hidden)
        # decoder_output is shape batch_size, seq_len, rnn_hidden_size

        decoder_output = F.relu(self.decoder_fc1(decoder_output))
        decoder_output = self.decoder_fc2(decoder_output)

        return decoder_output.squeeze(dim=-1)
