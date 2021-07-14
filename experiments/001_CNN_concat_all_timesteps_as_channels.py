#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nowcasting_dataset.datamodule import NowcastingDataModule
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from neptune.new.integrations.pytorch_lightning import NeptuneLogger

import logging
logging.basicConfig()
logger = logging.getLogger('nowcasting_dataset')
logger.setLevel(logging.DEBUG)


# In[2]:


import numpy as np


# In[3]:


BUCKET = Path('solar-pv-nowcasting-data')

# Solar PV data
PV_PATH = BUCKET / 'PV/PVOutput.org'
PV_DATA_FILENAME = PV_PATH / 'UK_PV_timeseries_batch.nc'
PV_METADATA_FILENAME = PV_PATH / 'UK_PV_metadata.csv'

# SAT_FILENAME = BUCKET / 'satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep_quarter_geospatial.zarr'
SAT_FILENAME = BUCKET / 'satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep.zarr'

# Numerical weather predictions
#NWP_BASE_PATH = BUCKET / 'NWP/UK_Met_Office/UKV_zarr'
#NWP_BASE_PATH = BUCKET / 'NWP/UK_Met_Office/UKV_single_step_and_single_timestep_all_vars.zarr'
NWP_BASE_PATH = BUCKET / 'NWP/UK_Met_Office/UKV_single_step_and_single_timestep_all_vars_full_spatial_2018_7-12_float32.zarr'


# In[4]:


params = dict(
    batch_size=32,
    history_len=6,  #: Number of timesteps of history, not including t0.
    forecast_len=12,  #: Number of timesteps of forecast.
    nwp_channels=(
        't', 'dswrf', 'prate', 'r', 'sde', 'si10', 'vis', 'lcc', 'mcc', 'hcc')
)


# In[5]:


data_module = NowcastingDataModule(
    pv_power_filename=PV_DATA_FILENAME,
    pv_metadata_filename=f'gs://{PV_METADATA_FILENAME}',
    sat_filename = f'gs://{SAT_FILENAME}',
    # sat_channels =('HRV', 'WV_062', 'WV_073'),
    nwp_base_path = f'gs://{NWP_BASE_PATH}',
    pin_memory = True,  #: Passed to DataLoader.
    num_workers = 22,  #: Passed to DataLoader.
    prefetch_factor = 256,  #: Passed to DataLoader.
    n_samples_per_timestep = 8,  #: Passed to NowcastingDataset
    **params
)


# In[6]:


data_module.prepare_data()


# In[7]:


data_module.setup()


# ## Define very simple ML model

# In[8]:


import tilemapbase
from nowcasting_dataset.geospatial import osgb_to_lat_lon


# In[9]:


tilemapbase.init(create=True)


# In[10]:


def plot_example(batch, model_output, example_i: int=0, border: int=0):
    fig = plt.figure(figsize=(20, 20))
    ncols=4
    nrows=2
    
    # Satellite data
    extent = (
        float(batch['sat_x_coords'][example_i, 0].cpu().numpy()), 
        float(batch['sat_x_coords'][example_i, -1].cpu().numpy()), 
        float(batch['sat_y_coords'][example_i, -1].cpu().numpy()), 
        float(batch['sat_y_coords'][example_i, 0].cpu().numpy()))  # left, right, bottom, top
    
    def _format_ax(ax):
        #ax.set_xlim(extent[0]-border, extent[1]+border)
        #ax.set_ylim(extent[2]-border, extent[3]+border)
        # ax.coastlines(color='black')
        ax.scatter(
            batch['x_meters_center'][example_i].cpu(), 
            batch['y_meters_center'][example_i].cpu(), 
            s=500, color='white', marker='x')

    ax = fig.add_subplot(nrows, ncols, 1) #, projection=ccrs.OSGB(approx=False))
    sat_data = batch['sat_data'][example_i, :, :, :, 0].cpu().numpy()
    sat_min = np.min(sat_data)
    sat_max = np.max(sat_data)
    ax.imshow(sat_data[0], extent=extent, interpolation='none', vmin=sat_min, vmax=sat_max)
    ax.set_title('t = -{}'.format(params['history_len']))
    _format_ax(ax)

    ax = fig.add_subplot(nrows, ncols, 2)
    ax.imshow(sat_data[params['history_len']+1], extent=extent, interpolation='none', vmin=sat_min, vmax=sat_max)
    ax.set_title('t = 0')
    _format_ax(ax)
    
    ax = fig.add_subplot(nrows, ncols, 3)
    ax.imshow(sat_data[-1], extent=extent, interpolation='none', vmin=sat_min, vmax=sat_max)
    ax.set_title('t = {}'.format(params['forecast_len']))
    _format_ax(ax)
    
    ax = fig.add_subplot(nrows, ncols, 4)
    lat_lon_bottom_left = osgb_to_lat_lon(extent[0], extent[2])
    lat_lon_top_right = osgb_to_lat_lon(extent[1], extent[3])
    tiles = tilemapbase.tiles.build_OSM()
    lat_lon_extent = tilemapbase.Extent.from_lonlat(
        longitude_min=lat_lon_bottom_left[1],
        longitude_max=lat_lon_top_right[1],
        latitude_min=lat_lon_bottom_left[0],
        latitude_max=lat_lon_top_right[0])
    plotter = tilemapbase.Plotter(lat_lon_extent, tile_provider=tiles, zoom=6)
    plotter.plot(ax, tiles)

    ############## TIMESERIES ##################
    # NWP
    ax = fig.add_subplot(nrows, ncols, 5)
    nwp_dt_index = pd.to_datetime(batch['nwp_target_time'][example_i].cpu().numpy(), unit='s')
    pd.DataFrame(
        batch['nwp'][example_i, :, :, 0, 0].T.cpu().numpy(), 
        index=nwp_dt_index,
        columns=params['nwp_channels']).plot(ax=ax)
    ax.set_title('NWP')

    # datetime features
    ax = fig.add_subplot(nrows, ncols, 6)
    ax.set_title('datetime features')
    datetime_feature_cols = ['hour_of_day_sin', 'hour_of_day_cos', 'day_of_year_sin', 'day_of_year_cos']
    datetime_features_df = pd.DataFrame(index=nwp_dt_index, columns=datetime_feature_cols)
    for key in datetime_feature_cols:
        datetime_features_df[key] = batch[key][example_i].cpu().numpy()
    datetime_features_df.plot(ax=ax)
    ax.legend()
    ax.set_xlabel(nwp_dt_index[0].date())

    # PV yield
    ax = fig.add_subplot(nrows, ncols, 7)
    ax.set_title('PV yield for PV ID {:,d}'.format(batch['pv_system_id'][example_i].cpu()))
    pv_actual = pd.Series(
        batch['pv_yield'][example_i].cpu().numpy(),
        index=nwp_dt_index,
        name='actual')
    pv_pred = pd.Series(
        model_output[example_i].detach().cpu().numpy(),
        index=nwp_dt_index[params['history_len']+1:],
        name='prediction')
    pd.concat([pv_actual, pv_pred], axis='columns').plot(ax=ax)
    ax.legend()

    # fig.tight_layout()
    
    return fig


# In[11]:


# plot_example(batch, model_output, example_i=20);  


# In[12]:


SAT_X_MEAN = np.float32(309000)
SAT_X_STD = np.float32(316387.42073603)
SAT_Y_MEAN = np.float32(519000)
SAT_Y_STD = np.float32(406454.17945938)


# In[13]:


from neptune.new.types import File


# In[14]:


TOTAL_SEQ_LEN = params['history_len'] + params['forecast_len'] + 1
CHANNELS = 144
KERNEL = 3
EMBEDDING_DIM = 16
NWP_SIZE = 10 * 2 * 2 * TOTAL_SEQ_LEN  # channels x width x height
N_DATETIME_FEATURES = 4 * TOTAL_SEQ_LEN

class LitAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        history_len = params['history_len'],
        forecast_len = params['forecast_len'],
        
    ):
        super().__init__()
        self.history_len = history_len
        self.forecast_len = forecast_len
        
        self.sat_conv1 = nn.Conv2d(in_channels=history_len+6, out_channels=CHANNELS, kernel_size=KERNEL)#, groups=history_len+1)
        self.sat_conv2 = nn.Conv2d(in_channels=CHANNELS, out_channels=CHANNELS, kernel_size=KERNEL) #, groups=CHANNELS//2)
        self.sat_conv3 = nn.Conv2d(in_channels=CHANNELS, out_channels=CHANNELS, kernel_size=KERNEL) #, groups=CHANNELS)

        self.maxpool = nn.MaxPool2d(kernel_size=KERNEL)
        
        self.fc1 = nn.Linear(
            in_features=CHANNELS * 11 * 11, 
            out_features=256)
        
        self.fc2 = nn.Linear(in_features=256 + EMBEDDING_DIM + NWP_SIZE + N_DATETIME_FEATURES + history_len+1, out_features=128)
        #self.fc2 = nn.Linear(in_features=EMBEDDING_DIM + N_DATETIME_FEATURES, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=128)
        self.fc5 = nn.Linear(in_features=128, out_features=params['forecast_len'])
        
        if EMBEDDING_DIM:
            self.pv_system_id_embedding = nn.Embedding(
                num_embeddings=len(data_module.pv_data_source.pv_metadata),
                embedding_dim=EMBEDDING_DIM)
        
    def forward(self, x):
        # ******************* Satellite imagery *************************
        # Shape: batch_size, seq_length, width, height, channel
        sat_data = x['sat_data'][:, :self.history_len+1]
        batch_size, seq_len, width, height, n_chans = sat_data.shape
        
        # Move seq_length to be the last dim, ready for changing the shape
        sat_data = sat_data.permute(0, 2, 3, 4, 1)
        
        # Stack timesteps into the channel dimension
        sat_data = sat_data.view(batch_size, width, height, seq_len * n_chans)
        
        sat_data = sat_data.permute(0, 3, 1, 2)  # Conv2d expects channels to be the 2nd dim!
        
        ### EXTRA CHANNELS
        # Center marker
        center_marker = torch.zeros((batch_size, 1, width, height), dtype=torch.float32, device=self.device)
        half_width = width // 2
        center_marker[..., half_width-2:half_width+2, half_width-2:half_width+2] = 1
        
        # geo-spatial x
        x_coords = x['sat_x_coords'] - SAT_X_MEAN
        x_coords /= SAT_X_STD
        x_coords = x_coords.unsqueeze(1).expand(-1, width, -1).unsqueeze(1)
        
        # geo-spatial y
        y_coords = x['sat_y_coords'] - SAT_Y_MEAN
        y_coords /= SAT_Y_STD
        y_coords = y_coords.unsqueeze(-1).expand(-1, -1, height).unsqueeze(1)
        
        # pixel x & y
        pixel_range = (torch.arange(width, device=self.device) - 64) / 37
        pixel_range = pixel_range.unsqueeze(0).unsqueeze(0)
        pixel_x = pixel_range.unsqueeze(-2).expand(batch_size, 1, width, -1)
        pixel_y = pixel_range.unsqueeze(-1).expand(batch_size, 1, -1, height)
        
        # Concat
        sat_data = torch.cat((sat_data, center_marker, x_coords, y_coords, pixel_x, pixel_y), dim=1)
        
        del center_marker, x_coords, y_coords, pixel_x, pixel_y
        
        # Pass data through the network :)
        out = F.relu(self.sat_conv1(sat_data))
        out = self.maxpool(out)
        out = F.relu(self.sat_conv2(out))
        out = self.maxpool(out)
        out = F.relu(self.sat_conv3(out))
        
        out = out.view(-1, CHANNELS * 11 * 11)
        out = F.relu(self.fc1(out))
        
        # *********************** NWP Data **************************************
        nwp_data = x['nwp'].float() # Shape: batch_size, channel, seq_length, width, height
        batch_size, n_nwp_chans, nwp_seq_len, nwp_width, nwp_height = nwp_data.shape
        nwp_data = nwp_data.reshape(batch_size, n_nwp_chans * nwp_seq_len * nwp_width * nwp_height)
        
        # Concat
        out = torch.cat(
            (
                out,
                x['pv_yield'][:, :self.history_len+1],
                nwp_data,
                x['hour_of_day_sin'],
                x['hour_of_day_cos'],
                x['day_of_year_sin'],
                x['day_of_year_cos'],
            ),
            dim=1)
        del nwp_data
        
        # Embedding of PV system ID
        if EMBEDDING_DIM:
            pv_embedding = self.pv_system_id_embedding(x['pv_system_row_number'])
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
        out = F.relu(self.fc5(out)) # PV yield is in range [0, 1].  ReLU should train more cleanly than sigmoid.

        return out
    
    def _training_or_validation_step(self, batch, is_train_step):
        y_hat = self(batch)
        y = batch['pv_yield'][:, -self.forecast_len:]
        #y = torch.rand((32, 1), device=self.device)
        mse_loss = F.mse_loss(y_hat, y)
        nmae_loss = (y_hat - y).abs().mean()
        # TODO: Compute correlation coef using np.corrcoef(tensor with shape (2, num_timesteps))[0, 1]
        # on each example, and taking the mean across the batch?
        tag = "Train" if is_train_step else "Validation"
        self.log_dict({f'MSE/{tag}': mse_loss}, on_step=is_train_step, on_epoch=True)
        self.log_dict({f'NMAE/{tag}': nmae_loss}, on_step=is_train_step, on_epoch=True)
        
        return nmae_loss

    def training_step(self, batch, batch_idx):
        return self._training_or_validation_step(batch, is_train_step=True)
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            # Plot example
            model_output = self(batch)
            fig = plot_example(batch, model_output)
            self.logger.experiment['validation/plot'].log(File.as_image(fig))
            
        return self._training_or_validation_step(batch, is_train_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


# In[15]:


model = LitAutoEncoder()


# In[16]:


#train_ds = data_module.train_dataset
#train_ds.per_worker_init(0)
#for batch in train_ds:
#    break


# In[17]:


#model_output = model(batch)


# In[18]:


#plot_example(batch, model_output, example_i=2);


# In[19]:


logger = NeptuneLogger(
    project='OpenClimateFix/predict-pv-yield',
    #params=params,
    #experiment_name='climatology',
    #experiment_id='PRED-1'
)


# In[20]:


logger.version


# In[21]:


trainer = pl.Trainer(gpus=1, max_epochs=10_000, logger=logger)


# In[ ]:


trainer.fit(model, data_module)


# In[ ]: