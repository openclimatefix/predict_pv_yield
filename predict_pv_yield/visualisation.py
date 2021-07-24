import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable
import tilemapbase
from nowcasting_dataset.geospatial import osgb_to_lat_lon
from nowcasting_dataset.example import DATETIME_FEATURE_NAMES


def plot_example(
        batch, model_output, history_len: int, forecast_len: int,
        nwp_channels: Iterable[str], example_i: int = 0, border: int = 0
) -> plt.Figure:

    fig = plt.figure(figsize=(20, 20))
    ncols = 4
    nrows = 2

    # ******************* SATELLITE IMAGERY ***********************************
    extent = (  # left, right, bottom, top
        float(batch['sat_x_coords'][example_i, 0].cpu().numpy()),
        float(batch['sat_x_coords'][example_i, -1].cpu().numpy()),
        float(batch['sat_y_coords'][example_i, -1].cpu().numpy()),
        float(batch['sat_y_coords'][example_i, 0].cpu().numpy()))

    def _format_ax(ax):
        if 'x_meters_center' in batch:
            ax.scatter(
                batch['x_meters_center'][example_i].cpu(),
                batch['y_meters_center'][example_i].cpu(),
                s=500, color='white', marker='x')

    ax = fig.add_subplot(nrows, ncols, 1)
    sat_data = batch['sat_data'][example_i, :, :, :, 0].cpu().numpy()
    sat_min = np.min(sat_data)
    sat_max = np.max(sat_data)
    ax.imshow(
        sat_data[0], extent=extent, interpolation='none',
        vmin=sat_min, vmax=sat_max)
    ax.set_title('t = -{}'.format(history_len))
    _format_ax(ax)

    ax = fig.add_subplot(nrows, ncols, 2)
    ax.imshow(
        sat_data[history_len+1], extent=extent, interpolation='none',
        vmin=sat_min, vmax=sat_max)
    ax.set_title('t = 0')
    _format_ax(ax)

    ax = fig.add_subplot(nrows, ncols, 3)
    ax.imshow(
        sat_data[-1], extent=extent, interpolation='none',
        vmin=sat_min, vmax=sat_max)
    ax.set_title('t = {}'.format(forecast_len))
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

    # ******************* TIMESERIES ******************************************
    # NWP
    ax = fig.add_subplot(nrows, ncols, 5)
    nwp_dt_index = pd.to_datetime(
        batch['nwp_target_time'][example_i].cpu().numpy(), unit='s')
    pd.DataFrame(
        batch['nwp'][example_i, :, :, 0, 0].T.cpu().numpy(),
        index=nwp_dt_index,
        columns=nwp_channels).plot(ax=ax)
    ax.set_title('NWP')

    # datetime features
    ax = fig.add_subplot(nrows, ncols, 6)
    ax.set_title('datetime features')
    datetime_features_df = pd.DataFrame(
        index=nwp_dt_index, columns=DATETIME_FEATURE_NAMES)
    for key in DATETIME_FEATURE_NAMES:
        datetime_features_df[key] = batch[key][example_i].cpu().numpy()
    datetime_features_df.plot(ax=ax)
    ax.legend()
    ax.set_xlabel(nwp_dt_index[0].date())

    # ************************ PV YIELD ***************************************
    ax = fig.add_subplot(nrows, ncols, 7)
    ax.set_title('PV yield for PV ID {:,d}'.format(
        batch['pv_system_id'][example_i].cpu()))
    pv_actual = pd.Series(
        batch['pv_yield'][example_i].cpu().numpy(),
        index=nwp_dt_index, name='actual')
    pv_pred = pd.Series(
        model_output[example_i].detach().cpu().numpy(),
        index=nwp_dt_index[history_len+1:], name='prediction')
    pd.concat([pv_actual, pv_pred], axis='columns').plot(ax=ax)
    ax.legend()

    return fig
