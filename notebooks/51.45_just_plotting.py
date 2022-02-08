#!/usr/bin/env python
# coding: utf-8


import xarray as xr


# Load a mapping from satellite pixel x and y coords to GSP ID
gsp_id_per_satellite_pixel = xr.open_dataset("GSP_ID_per_satellite_pixel.nc")["data"]



from pathlib import Path

import fsspec
import torch
from torch.utils import data
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
import einops

from dataclasses import dataclass
from typing import Iterable, Optional, Union
import xarray as xr
from concurrent import futures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from itertools import chain
from datetime import timedelta


from pathy import Pathy



SAT_MEAN = {
    "HRV": 236.13257536395903,
    "IR_016": 291.61620182554185,
    "IR_039": 858.8040610176552,
    "IR_087": 738.3103442750336,
    "IR_097": 773.0910794778366,
    "IR_108": 607.5318145165666,
    "IR_120": 860.6716261423857,
    "IR_134": 925.0477987594331,
    "VIS006": 228.02134593063957,
    "VIS008": 257.56333202381205,
    "WV_062": 633.5975770915588,
    "WV_073": 543.4963868823854,
}

SAT_STD = {
    "HRV": 935.9717382401759,
    "IR_016": 172.01044433112992,
    "IR_039": 96.53756504807913,
    "IR_087": 96.21369354283686,
    "IR_097": 86.72892737648276,
    "IR_108": 156.20651744208888,
    "IR_120": 104.35287930753246,
    "IR_134": 104.36462050405994,
    "VIS006": 150.2399269307514,
    "VIS008": 152.16086321818398,
    "WV_062": 111.8514878214775,
    "WV_073": 106.8855172848904,
}

# Means and std computed with
# nowcasting_dataset/scripts/compute_stats_from_batches.py
# using v15 training batches on 2021-11-24.
NWP_MEAN = {
    "t": 285.7799539185846,
    "dswrf": 294.6696933986283,
    "prate": 3.6078121378638696e-05,
    "r": 75.57106712435926,
    "sde": 0.0024915961594965614,
    "si10": 4.931356852411006,
    "vis": 22321.762918384553,
    "lcc": 47.90454236572895,
    "mcc": 44.22781694449808,
    "hcc": 32.87577371914454,
}

NWP_STD = {
    "t": 5.017000766747606,
    "dswrf": 233.1834250473355,
    "prate": 0.00021690701537950742,
    "r": 15.705370079694358,
    "sde": 0.07560040052148084,
    "si10": 2.664583614352396,
    "vis": 12963.802514945439,
    "lcc": 40.06675870700349,
    "mcc": 41.927221148316384,
    "hcc": 39.05157559763763,
}

def make_validation_results(
    predictions_mw,
    truths_mw,
    capacity_mwp,
    gsp_ids: list[int],
    t0_datetimes_utc: pd.DatetimeIndex,
    batch_idx: Optional[int] = None,
    forecast_sample_period: timedelta = timedelta(minutes=30),
) -> pd.DataFrame:
    """
    Make validations results.
    Args:
        predictions_mw: predictions in mw, shape [batch_size, n_forecast_horizons]
        truths_mw: truths in mw, shape [batch_size, n_forecast_horizons]
        capacity_mwp: the capacity of each gsp at that time
        gsp_ids: the gsp ids for eahc prediction, shape [batch_size]
        t0_datetimes_utc: list of date times when the predictions are for
        batch_idx: optional index of the batch
        forecast_sample_period: the different between each forecast horizon
    The following columns are made:
    - t0_datetime_utc
    - target_datetime_utc
    - gsp_id
    - actual_gsp_pv_outturn_mw
    - forecast_gsp_pv_outturn_mw
    - capacity_mwp
    - batch_index (optional)
    - example_index (optional)
    return: Dataframe of predictions and truths
    """

    assert predictions_mw.shape == truths_mw.shape

    results_per_forecast_horizon = []

    # TODO #64 vectorize
    n_forecast_timesteps = predictions_mw.shape[1]
    for i in range(n_forecast_timesteps):
        predictions_mw_df = pd.DataFrame(
            predictions_mw[:, i], columns=["forecast_gsp_pv_outturn_mw"]
        )
        predictions_mw_df["actual_gsp_pv_outturn_mw"] = truths_mw[:, i]
        predictions_mw_df["target_datetime_utc"] = (
            t0_datetimes_utc + (i + 1) * forecast_sample_period
        )
        predictions_mw_df["gsp_id"] = gsp_ids
        predictions_mw_df["t0_datetime_utc"] = t0_datetimes_utc
        predictions_mw_df["capacity_mwp"] = capacity_mwp[:, i]

        results_per_forecast_horizon.append(predictions_mw_df)

    # join truths and predictions for each forecast horizon
    results = pd.concat(results_per_forecast_horizon)

    # join truths and predictions
    results.index.name = "example_index"

    # add batch index
    if batch_idx is not None:
        results["batch_index"] = batch_idx

    return results


def save_validation_results_to_logger(
    results_dfs: list[pd.DataFrame],
    results_file_name: str,
    current_epoch: Union[int, str],
    logger: Optional[NeptuneLogger] = None,
):
    # join all validation step results together
    results_df = pd.concat(results_dfs)
    results_df.reset_index(inplace=True)

    # save to csv file
    name_csv = f"{results_file_name}_{current_epoch}.csv"
    results_df.to_csv(name_csv)

    # upload csv to neptune
    logger.experiment[f"validation/results/epoch_{current_epoch}"].upload(name_csv)



def get_pv_system_id_index_lut():
    # In get_gsp_id_and_pv_system_index_for_each_pv_system(), we really don't want to have to loop round each example and each PV system.  
    # So, instead, we create a "look up table" (LUT) where the entry at the i^th index
    # is the index of the PV system (and len(passiv_2_minutely_pv_system_ids) elsewhere)
    # (i.e. len(passiv_2_minutely_pv_system_ids) is used as a marker for "NaN"... it has to be a real int
    # otherwise the Embedding blows up).
    # That is, `lut[pv_system_id]` is the `index+1` of the PV system.
    passiv_2_minutely_pv_system_ids = pd.read_csv("passiv_2_minutely_pv_system_IDs.csv", index_col="index").squeeze("columns").to_list()
    missing_pv_maker = len(passiv_2_minutely_pv_system_ids)+1
    pv_system_id_index_lut = np.full(
        shape=max(passiv_2_minutely_pv_system_ids)+1, 
        fill_value=missing_pv_maker, 
        dtype=np.int16
    )
    pv_system_id_index_lut[passiv_2_minutely_pv_system_ids] = np.arange(0, len(passiv_2_minutely_pv_system_ids), dtype=np.int16)
    
    # test
    assert pv_system_id_index_lut[0] == missing_pv_maker  # No PV system pv_system_id 0
    for i in [0, 1, 100, 1000, len(passiv_2_minutely_pv_system_ids)-1]:
        pv_system_id = passiv_2_minutely_pv_system_ids[i]
        assert pv_system_id_index_lut[pv_system_id] == i
    return pv_system_id_index_lut


pv_system_id_index_lut = get_pv_system_id_index_lut()


# we use the number MAX_GSP_ID to indicate a "missing GSP" to the GSP ID embedding
MAX_GSP_ID = int(np.nanmax(gsp_id_per_satellite_pixel.values) + 1)
gsp_id_per_satellite_pixel = gsp_id_per_satellite_pixel.fillna(MAX_GSP_ID).astype(np.int32)



def get_pv_system_id_to_gsp_lut():
    """Get a look up table (LUT) which maps from PV system ID to GSP ID.
    """
    passiv_pv_system_id_to_gsp_id = pd.read_csv("passiv_pv_system_id_to_gsp_id.csv", index_col="system_id").squeeze("columns")
    missing_gsp_marker = MAX_GSP_ID,  # Use max(gsp_id)+1 as the marker for "missing" GSP
    pv_system_id_to_gsp_lut = np.full(
        shape=max(passiv_pv_system_id_to_gsp_id.index)+1, 
        fill_value=missing_gsp_marker,
        dtype=np.int16
    )
    pv_system_id_to_gsp_lut[passiv_pv_system_id_to_gsp_id.index] = passiv_pv_system_id_to_gsp_id
    
    # test
    assert pv_system_id_to_gsp_lut[0] == missing_gsp_marker
    for i in [0, 1, 100, 1000, len(passiv_pv_system_id_to_gsp_id)-1]:
        pv_system_id = passiv_pv_system_id_to_gsp_id.index[i]
        gsp_id = passiv_pv_system_id_to_gsp_id.loc[pv_system_id]
        assert pv_system_id_to_gsp_lut[pv_system_id] == gsp_id

    return pv_system_id_to_gsp_lut


pv_system_id_to_gsp_lut = get_pv_system_id_to_gsp_lut()


def get_gsp_id_and_pv_system_index_for_each_pv_system(pv_system_id_tensor: torch.Tensor) -> tuple[np.ndarray, np.array]:
    """For each PV system ID in batch['pv_system_id'], get the GSP ID and the "index" of the PV system (used for embedding the PV system ID)."""
   
    pv_system_id_tensor = torch.nan_to_num(
        pv_system_id_tensor, 
        nan=0  # No PV systems have the ID "0".  So 0 is safe to use as a marker for "missing".
    ).to(torch.int16)

    pv_system_id_index = pv_system_id_index_lut[pv_system_id_tensor]
    gsp_id_for_each_pv_system = pv_system_id_to_gsp_lut[pv_system_id_tensor]
    return gsp_id_for_each_pv_system, pv_system_id_index


#DATA_PATH = Path("/mnt/storage_ssd_4tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/prepared_ML_training_data/v15")
DATA_PATH = Pathy("gs://solar-pv-nowcasting-data/prepared_ML_training_data/v15")
#DATA_PATH = Path("/mnt/ssd/data/v15")

SATELLITE_CHANNELS = (
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
)

NWP_CHANNELS = (
    "t",
    "dswrf",
    "prate",
    "r",
    "sde",
    "si10",
    "vis",
    "lcc",
    "mcc",
    "hcc"
)



def satellite_normalisation_stats_to_data_array(stat: dict, channel_names: tuple[str]) -> xr.DataArray:
    return xr.DataArray(
        [stat[chan_name] for chan_name in channel_names],
        dims="channels_index",
    ).astype(np.float32).assign_coords(channels=("channels_index", list(channel_names)))

SAT_MEAN = satellite_normalisation_stats_to_data_array(SAT_MEAN, SATELLITE_CHANNELS)
SAT_STD = satellite_normalisation_stats_to_data_array(SAT_STD, SATELLITE_CHANNELS)
NWP_MEAN = satellite_normalisation_stats_to_data_array(NWP_MEAN, NWP_CHANNELS)
NWP_STD = satellite_normalisation_stats_to_data_array(NWP_STD, NWP_CHANNELS)


def align_time(src: xr.Dataset, dst: xr.Dataset) -> xr.Dataset:
    """Align `dst` to have the same `time` coords as `src.time`.
    
    For example, use this to ensure that batch['opticalflow'] has the same
    time coords as batch['gsp'].
    """
    # I tried a bunch of "vectorised" ways of doing this.  This appears to be the
    # only way of doing it.  The issue is that each example ends up having different
    # time_index coords so, in order to align, we must reset the "time_index" of each example.
    # We take the satellite image _15 minutes_ before the GSP data, because the GSP data is
    # the average power for the half-hour-ending.
    time_index = src.time - pd.Timedelta("15 minutes")
    n_timesteps = src.time.shape[1]
    time_index_bool_mask = dst.time.isin(time_index)
    data_arrays_for_examples = []
    n_examples = len(time_index_bool_mask)
    for example_i in range(n_examples):
        selection = dst.isel(example=example_i, time_index=time_index_bool_mask[example_i])
        # Ensure there are never too many timesteps
        selection = selection.isel(time_index=slice(None, n_timesteps))
        selection.__setitem__("time_index", np.arange(len(selection["time_index"])))
        data_arrays_for_examples.append(selection)
        
    new_dst = xr.concat(data_arrays_for_examples, dim="example")
    return new_dst



N_DATETIME_FEATURES = 4  # sin & cos for time of day and day of year


def create_datetime_features(t0_datetimes: np.ndarray) -> dict[str, np.ndarray]:
    t0_datetimes = pd.DatetimeIndex(t0_datetimes)
    n_examples = len(t0_datetimes)

    datetime_features = np.full(shape=(n_examples, N_DATETIME_FEATURES), fill_value=np.NaN, dtype=np.float32)

    hour_of_day = t0_datetimes.hour + (t0_datetimes.minute / 60)
    day_of_year = t0_datetimes.day_of_year + (hour_of_day / 24)

    hour_of_day_radians = (hour_of_day / 24.0) * 2 * np.pi
    day_of_year_radians = (day_of_year / 366 ) * 2 * np.pi  #  366 for leap years!

    datetime_features[:, 0] = np.sin(hour_of_day_radians)
    datetime_features[:, 1] = np.cos(hour_of_day_radians)
    datetime_features[:, 2] = np.sin(day_of_year_radians)
    datetime_features[:, 3] = np.cos(day_of_year_radians)
    
    return {
        "t0_datetime_features": datetime_features,
        "t0_hour_of_day": hour_of_day.values,
        "t0_day_of_year": day_of_year.values,
        "t0_month": t0_datetimes.month.values,
        "t0_datetime_UNIX_epoch": t0_datetimes.values.astype("datetime64[s]").astype(np.int32),
    }


def copy_x_and_y_and_time_coords(data_from_all_sources: dict[str, object], data_source_name: str):
    data = data_from_all_sources[data_source_name]
    data_from_all_sources[f"{data_source_name}_time"] = torch.from_numpy(data["time"].values.astype("datetime64[s]").astype(np.int32))
    
    if data_source_name in ["satellite", "opticalflow"]:
        # satellite and opticalflow mistakenly include x and y coords for every timestep (even though they don't change across time!)
        x_coords = data["x_osgb"]
        y_coords = data["y_osgb"]
    elif data_source_name in ["gsp", "pv"]:
        # The x and y coords DataArrays are called "x_coords" and "y_coords" in gsp and pv.  And "x" and "y" in the others.
        x_coords = data["x_coords"]
        y_coords = data["y_coords"]
        if data_source_name == "pv":
            # Set the x_coords, y_coords and time for missing PV systems to NaN
            mask_of_missing_pv_systems = np.isnan(data["data"].values).any(axis=1)
            x_coords.values[mask_of_missing_pv_systems] = np.NaN
            y_coords.values[mask_of_missing_pv_systems] = np.NaN
    else:
        x_coords = data["x"]
        y_coords = data["y"]
        
    data_from_all_sources[f"{data_source_name}_x_coords"] = torch.from_numpy(x_coords.values.astype(np.float32))
    data_from_all_sources[f"{data_source_name}_y_coords"] = torch.from_numpy(y_coords.values.astype(np.float32))




def load_dataset(filename, engine="h5netcdf", *args, **kwargs) -> xr.Dataset:
    """Load a NetCDF dataset from local file system or cloud bucket."""
    with fsspec.open(filename, mode="rb") as file:
        dataset = xr.load_dataset(file, engine=engine, *args, **kwargs)
    return dataset



import fsspec.asyn
def set_fsspec_for_multiprocess() -> None:
    """
    Clear reference to the loop and thread.
    This is necessary otherwise
    gcsfs hangs in the ML training loop.  Only required for fsspec >= 0.9.0
    See https://github.com/dask/gcsfs/issues/379#issuecomment-839929801
    TODO: Try deleting this two lines to make sure this is still relevant.
    """
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None
    
set_fsspec_for_multiprocess()


@dataclass
class SimpleNowcastingDataset(data.Dataset):
    """
    Attributes:
        data_path: Base path to the pre-prepared dataset.  e.g. .../v15/train/
        data_source_names: The names of the data sources.  Must also be the names of the subdirectory.  
            Must include "gsp".
        gsp_first_time_index_of_future: The index into the GSP time_index dimension that marks the start of the "future".
        n_batches: The number of available batches. If the user sets this to an int then
            this int will be the max number of batches used per epoch.
    """
    data_path: Path
    data_source_names: Iterable[str]
    gsp_first_time_index_of_future: int = 2
    n_batches: Optional[int] = None
    
    def __post_init__(self):
        # Sanity checks
        set_fsspec_for_multiprocess()
        assert self.data_path.exists()
        assert len(self.data_source_names) > 0
        assert "gsp" in self.data_source_names
        self._set_number_of_batches()
        
    def _set_number_of_batches(self) -> None:
        """Set number of batches.  Check every data source."""
        for data_source_name in self.data_source_names:
            path_for_data_source = self.data_path / data_source_name
            n_batches_for_data_source = len(list(path_for_data_source.glob("*.nc")))
            print(data_source_name, "has", n_batches_for_data_source, "batches. ", end="")
            if self.n_batches is None:
                self.n_batches = n_batches_for_data_source
            else:
                if n_batches_for_data_source != self.n_batches:
                    print(
                        "Warning!", 
                        data_source_name, 
                        "has a different number of batches to at least one other modality or the requested number of batches! We'll use the minimum of the two values.")
                    self.n_batches = min(self.n_batches, n_batches_for_data_source)
            print()
        assert self.n_batches is not None
        assert self.n_batches > 0
    
    def __len__(self) -> int:
        return self.n_batches
    
    def __getitem__(self, idx: int):
        """
        Returned shapes:
            gsp: batch_size, n_timesteps
            opticalflow: "example", "time_index", "channels_index",  "y_geostationary_index", "x_geostationary_index"
            nwp: "example", "time_index", "channels_index", "y_index", "x_index"
            pv:
        """
        set_fsspec_for_multiprocess()
        data_from_all_sources = {}
        # Parallelising this with concurrent.futures.ThreadPoolExecutor actually
        # appears to be _slower_ than the simple loop approach!
        for data_source_name in self.data_source_names:
            filename = self.data_path / data_source_name / f"{idx:06d}.nc"
            dataset = load_dataset(filename)
            
            # Select just the "future" timesteps at half-hour intervals (t1, t2, etc.)
            # for the first GSP (the "target").
            if data_source_name == "gsp":
                dataset = dataset.isel(
                    time_index=slice(self.gsp_first_time_index_of_future, None),
                    id_index=0
                ).load()
                # Normalise GSP
                dataset["data"] = dataset["power_mw"] / dataset["capacity_mwp"]
                dataset["data"] = dataset["data"].astype(np.float32)
                
                # Datetime features
                t0_datetimes = dataset.time.values[:, 0] - pd.Timedelta("30 minutes")
                
            if data_source_name == "pv":
                try:
                    dataset = dataset.isel(time_index=slice(None, 7))
                except ValueError as e:
                    print(f"Exception raised when reading PV: {filename}", flush=True)
                    raise
                dataset["data"] = dataset["power_mw"] / dataset["capacity_mwp"]
            
            if data_source_name in ["satellite", "opticalflow"]:
                # Normalise satellite and opticalflow
                dataset["data"] = dataset["data"].astype(np.float32)
                dataset["data"] -= SAT_MEAN.drop("channels")
                dataset["data"] /= SAT_STD.drop("channels")
                dataset["data"] = dataset["data"].transpose("example", "time_index", "channels_index", "y_geostationary_index", "x_geostationary_index")
                
            if data_source_name == "nwp":
                # Normalise satellite and opticalflow
                dataset["data"] -= NWP_MEAN.drop("channels")
                dataset["data"] /= NWP_STD.drop("channels")
                dataset["data"] = dataset["data"].transpose("example", "time_index", "channels_index", "y_index", "x_index")

            data_from_all_sources[data_source_name] = dataset

        if ("satellite" in self.data_source_names) and ("opticalflow" in self.data_source_names):
            # Concatenate half an hour of satellite data to the start of the opticalflow data
            # so we can use imagery 15 minutes before the GSP timestep, because the GSP data
            # is half-hour-ending.
            data_from_all_sources["opticalflow"] = xr.concat(
                (
                    data_from_all_sources["satellite"].isel(time_index=slice(None, 7)),
                    data_from_all_sources["opticalflow"]
                ),
                dim="time_index",
            )
            del data_from_all_sources["satellite"]

        # Conform Satellite data sources to have the same time index as GSP
        for data_source_name in ["satellite", "opticalflow", "hrvsatellite", "hrvopticalflow"]:
            if data_source_name in data_from_all_sources:
                data_from_all_sources[data_source_name] = align_time(
                    src=data_from_all_sources["gsp"],
                    dst=data_from_all_sources[data_source_name].load())
                
                
        # Get GSP ID for each opticalflow pixel
        data_from_all_sources["gsp_id_per_satellite_pixel"] = gsp_id_per_satellite_pixel.sel(
            x=data_from_all_sources['opticalflow']['x_osgb'], 
            y=data_from_all_sources['opticalflow']['y_osgb'], 
            method="nearest",
        )
        data_from_all_sources["gsp_id_per_satellite_pixel"] = torch.from_numpy(
            data_from_all_sources["gsp_id_per_satellite_pixel"].values
        )
                
        # Select just the data.  Grab other useful DataArrays.
        new_data_source_names = list(data_from_all_sources.keys())
        for data_source_name in new_data_source_names:
            if data_source_name == "gsp_id_per_satellite_pixel":
                continue

            try:
                copy_x_and_y_and_time_coords(data_from_all_sources, data_source_name)
            except:
                print("Exception raised while processing", data_source_name)
                print(data_from_all_sources[data_source_name])
                raise

            data = data_from_all_sources[data_source_name]

            if data_source_name == "pv":
                data_from_all_sources["pv_system_id"] = torch.from_numpy(data["id"].values.astype(np.float32))
                pv_gsp_id, pv_system_id_index = get_gsp_id_and_pv_system_index_for_each_pv_system(data_from_all_sources["pv_system_id"])
                data_from_all_sources["pv_gsp_id"] = torch.from_numpy(pv_gsp_id)
                data_from_all_sources["pv_system_id_index"] = torch.from_numpy(pv_system_id_index)
                
            if data_source_name == "gsp":
                data_from_all_sources["gsp_id"] = torch.from_numpy(data["id"].values.astype(np.float32))
                data_from_all_sources["capacity_mwp"] = torch.from_numpy(data["capacity_mwp"].values.astype(np.float32))

            data_from_all_sources[data_source_name] = torch.from_numpy(data["data"].values)
            
        # Datetime features.
        t0_datetime_features = create_datetime_features(t0_datetimes)
        for feature_name, feature_values in t0_datetime_features.items():
            data_from_all_sources[feature_name] = torch.from_numpy(feature_values)
        
        # Check for NaNs
        for data_source_name, data in data_from_all_sources.items():
            if data_source_name != "gsp_id" and not data_source_name.startswith("pv") and np.isnan(data).any():
                raise RuntimeError(f"NaNs in {data_source_name} batch {idx}")

        return data_from_all_sources




DATA_SOURCE_NAMES = ("gsp", "satellite", "pv", "opticalflow", "nwp")


train_dataset = SimpleNowcastingDataset(
    data_path=DATA_PATH / "train",
    data_source_names=DATA_SOURCE_NAMES,
    n_batches=100,
)


def tensor_nanmin(tensor, dim):
    return torch.from_numpy(np.nanmin(tensor.cpu(), axis=dim)).to(device=tensor.device)


def tensor_nanmax(tensor, dim):
    return torch.from_numpy(np.nanmax(tensor.cpu(), axis=dim)).to(device=tensor.device)


def rescale_tensors_to_0_to_1(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rescales multiple tensors using the same min and max across all tensors.
    
    Args:
        tensors: A dictionary of Tensors.  The dictionary keys should be the name of each tensor.
            The values of dictionary should be Tensors of shape [batch_size, length].  All tensors
            must have the same batch_size.
            
    Returns:
        rescaled_tensors: A dict with the same keys as the input `tensors` dict but where each
            tensor has had its values rescaled to be in the range [0, 1].
    """
   
    # Compute the maximum and the range, across all the tensors.
    list_of_tensors = [torch.flatten(t, start_dim=1) for t in tensors.values()]
    tensors_concatenated = torch.cat(list_of_tensors, dim=1)
    del list_of_tensors
    minimum = tensor_nanmin(tensors_concatenated, dim=1)
    maximum = tensor_nanmax(tensors_concatenated, dim=1)
    min_max_range = maximum - minimum
    del maximum

    minimum = minimum.unsqueeze(-1)
    min_max_range = min_max_range.unsqueeze(-1)
    
    # Rescale each tensor
    rescaled_tensors = {}
    for name, tensor in tensors.items():
        if len(tensor.shape) == 3:
            # 2D OSGB coords
            rescaled_tensors[name] = (tensor - minimum.unsqueeze(-1)) / min_max_range.unsqueeze(-1)
        else:
            rescaled_tensors[name] = (tensor - minimum) / min_max_range
        
    return rescaled_tensors


def compute_fourier_features(array: torch.Tensor, n_fourier_features: int = 4, min_freq: float = 1, max_freq: float = 4) -> torch.Tensor:
    """Compute fourier features for a single dimension, across all examples in a batch.
    
    Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    
    Args:
        array: Tensor of shape [batch_size, length] with values in the range [0, 1].
            For the time dimension, `length` will be the timeseries sequence length.
            For spatial dimensions, `length` will be the width or height.
        n_fourer_features: Total number of requested fourier features
        
    Returns:
        fourer_features: A tensor of the same dtype and device as `array`,
            with shape [batch_size, length, n_fourier_features].  Fourier features with even indexes
            are sine.  Odd indexes are cosine.
    """
    #assert np.nanmax(array) <= 1
    #assert np.nanmin(array) >= 0
    assert n_fourier_features % 2 == 0
    assert min_freq > 0
    assert max_freq > min_freq

    batch_size, length = array.shape[:2]
    array = array * np.pi / 2
    array = array.unsqueeze(-1)
    div_term = torch.linspace(start=min_freq, end=max_freq, steps=n_fourier_features // 2, dtype=array.dtype, device=array.device)
    fourier_features = torch.full(
        size=(batch_size, length, n_fourier_features), 
        fill_value=np.NaN, 
        dtype=array.dtype, 
        device=array.device,
    )

    fourier_features[:, :, 0::2] = torch.sin(array * div_term)
    fourier_features[:, :, 1::2] = torch.cos(array * div_term)
    return fourier_features



def get_spatial_and_temporal_fourier_features(
    batch: dict[str, torch.Tensor], 
    num_fourier_time_features: int = 4,
    num_fourier_spatial_features_total: int = 8,  # Total for both height and width
) -> None:
    for dim_name in ('_x_coords', '_y_coords', 'time'):
        all_coords = {key: tensor for key, tensor in batch.items() if key.endswith(dim_name)}
        if f"gsp{dim_name}" in all_coords and dim_name != "time":
            # We're only selecting a single GSP.  So GSP x_coords and y_coords are 1D.  We want 2D.
            all_coords[f"gsp{dim_name}"] = all_coords[f"gsp{dim_name}"].unsqueeze(-1)
        all_coords = rescale_tensors_to_0_to_1(all_coords)
        if dim_name == "time":
            n_fourier_features = num_fourier_time_features
        else:
            n_fourier_features = num_fourier_spatial_features_total // 2
        for key, coords in all_coords.items():
            if len(coords.shape) == 3:
                # Handle the 2D OSGB coords
                reshape = coords.shape
                coords = torch.flatten(coords, start_dim=1)
            else:
                reshape = False
            
            batch[f"{key}_fourier_features"] = compute_fourier_features(
                coords,
                n_fourier_features=n_fourier_features,
                min_freq=4,
                max_freq=32,
            )

            if reshape:
                batch[f"{key}_fourier_features"] = batch[f"{key}_fourier_features"].reshape(reshape + (n_fourier_features,))



def plot_imagery(tensor, example_i, channel_i):
    fig, axes = plt.subplots(ncols=4, figsize=(20, 10))
    # axes = list(chain.from_iterable(axes))
    # "example", "time_index", "channels_index", "y_index", "x_index"
    data_to_plot = tensor[example_i, :4, channel_i, :, :]
    vmin = data_to_plot.min()
    vmax = data_to_plot.max()
    for time_index, ax in enumerate(axes):
        ax.imshow(data_to_plot[time_index].numpy()[::-1], vmin=vmin, vmax=vmax)
        ax.set_title(time_index)
    return fig, axes



dataloader_kwargs = dict(
    batch_size=None,
    num_workers=10,
    pin_memory=True,
    persistent_workers=False,
)


train_dataloader = data.DataLoader(
    train_dataset,
    **dataloader_kwargs
)



test_dataloader = data.DataLoader(
    SimpleNowcastingDataset(
        data_path=DATA_PATH / "test",
        data_source_names=DATA_SOURCE_NAMES,
        n_batches=10,
    ),
    **dataloader_kwargs
)


# Mixture density network
PI = 'PI'
MU = 'MU'
SIGMA = 'SIGMA'
MEAN = 'MEAN'

class MixtureDensityNetwork(nn.Module):
    def __init__(self, in_features: int, num_gaussians: int = 2):
        super().__init__()
        self.pi = nn.Linear(in_features=in_features, out_features=num_gaussians)
        self.mu = nn.Linear(in_features=in_features, out_features=num_gaussians)
        self.sigma = nn.Linear(in_features=in_features, out_features=num_gaussians)
        
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pi = self.pi(x)
        pi = F.softmax(pi, dim=-1)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = torch.exp(sigma)
        return {PI: pi, MU: mu, SIGMA: sigma}
    
    
def get_distribution(
    network_output: dict[str, torch.Tensor], 
    example_i: Optional[int] = None
) -> torch.distributions.MixtureSameFamily:

    pi = network_output[PI]
    mu = network_output[MU]
    sigma = network_output[SIGMA]
    
    if example_i is not None:
        pi = pi[example_i]
        mu = mu[example_i]
        sigma = sigma[example_i]

    mixture_distribution = torch.distributions.Categorical(probs=pi)
    component_distribution = torch.distributions.Normal(loc=mu, scale=sigma)
    gaussian_mixture_model = torch.distributions.MixtureSameFamily(mixture_distribution, component_distribution)
    return gaussian_mixture_model



def plot_probs(pi, mu, sigma, ax, left, right, example_i=0):
    sweep_n_steps = 100
    sweep_start = 1
    sweep_stop = 0
       
    n_timesteps = pi.shape[1]

    # Define a 'sweep' matrix which we pass into log_prob to get probabilities 
    # for a range of values at each timestep.  Those values range from sweep_start to sweep_stop
    sweep = torch.linspace(start=sweep_start, end=sweep_stop, steps=sweep_n_steps, dtype=torch.float32, device=pi.device)
    sweep = sweep.unsqueeze(-1).expand(sweep_n_steps, n_timesteps)
    
    # Get probabilities.
    distribution = get_distribution({PI: pi, MU: mu, SIGMA: sigma}, example_i=example_i)
    log_probs = distribution.log_prob(sweep)
    probs = torch.exp(log_probs).detach().cpu().numpy()
    
    # Plot!
    extent = (left, right, sweep_stop, sweep_start) # left, right, bottom, top
    ax.imshow(probs, aspect='auto', interpolation='none', extent=extent, cmap='Greys')

    return ax


def plot_timeseries(batch: dict[str, torch.Tensor], network_output: dict[str, torch.Tensor]) -> plt.Figure:
    actual = batch["gsp"].detach().cpu()
    
    # get the mean prediction
    mu = network_output[MU]
    pi = network_output[PI]
    predicted = torch.sum((mu * pi), dim=-1).squeeze().detach().cpu()

    gsp_id = batch["gsp_id"].squeeze().detach().cpu()
    historical_pv = batch["pv"].squeeze().detach().cpu().numpy()
    if "nwp" in batch:
        nwp_chan = NWP_CHANNELS.index("dswrf")
        nwp = batch["nwp"][:, :, nwp_chan].mean(dim=[2, 3]).detach().cpu().numpy()
        nwp_time = batch["nwp_time"].squeeze().detach().cpu().numpy()
    t0_datetimes = batch["t0_datetime_UNIX_epoch"].squeeze().detach().cpu().numpy()
    t0_datetimes = pd.to_datetime(t0_datetimes, unit='s')
    fig, axes = plt.subplots(nrows=8, ncols=4, figsize=(20, 20))
    axes = list(chain.from_iterable(axes))
    FIFTEEN_MINUTES = (15 / (60 * 24))
    for example_i, ax in enumerate(axes):
        t0_datetime = t0_datetimes[example_i]
        t1_datetime = t0_datetime + pd.Timedelta("30 minutes")
        forecast_datetimes = pd.date_range(t1_datetime, periods=4, freq="30 min")

        # Plot historical PV yield
        historical_pv_datetimes = pd.date_range(t0_datetime - pd.Timedelta("30 minutes"), periods=7, freq="5 min")
        # plot_probs(
        #     pi=network_output[PI],
        #     mu=network_output[MU],
        #     sigma=network_output[SIGMA],
        #     ax=ax,
        #     left=mdates.date2num(forecast_datetimes[0]) - FIFTEEN_MINUTES,
        #     right=mdates.date2num(forecast_datetimes[-1]) + FIFTEEN_MINUTES,
        #     example_i=example_i,
        # )
        # ax.plot(
        #     historical_pv_datetimes,
        #     historical_pv[example_i],
        #     color="grey",
        #     alpha=0.5
        # )
        ax.plot(
            historical_pv_datetimes,
            np.nanmean(historical_pv, axis=2)[example_i],
            label="Historical mean PV",
            linewidth=3,
            alpha=0.8,
            color="red",
        )
        
        # Plot prediction for GSP PV yield and actual GSP PV yield
        # ax.plot(forecast_datetimes, predicted[example_i], label="Predicted GSP PV", color="orange", linewidth=3, alpha=0.8)
        # ax.plot(forecast_datetimes, actual[example_i], label="Actual GSP PV", linewidth=3, alpha=0.8)
        
        # Plot NWP params:
        # if "nwp" in batch:
            # ax2 = ax.twinx()
            # nwp_time_for_example = pd.to_datetime(nwp_time[example_i], unit="s")
            # ax2.plot(nwp_time_for_example, nwp[example_i], label="NWP irradiance", color="green", alpha=0.8)
            # ax2.yaxis.set_ticks([])
            # ax2.yaxis.set_ticks_position('none')
            # ax2.set_ylim(-2, 2)
        
        # Formatting
        # ax.xaxis.set_major_locator(mdates.HourLocator())
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        # ax.set_title("GSP {:.0f} on {}".format(gsp_id[example_i], t0_datetime.strftime("%Y-%m-%d")), y=0.8)
        # ax.set_ylim(0, 1)
        # ax.set_xlim(mdates.date2num(historical_pv_datetimes[0]), mdates.date2num(forecast_datetimes[-1]))
        # if example_i == 0:
        #     fig.legend(framealpha=0, loc="center right")

    return fig



SATELLITE_IMAGE_SIZE_PIXELS = 24

class Model(pl.LightningModule):
    
    # 2. **********
    # list of results dataframes. This is used to save validation results
    results_dfs = []
    # **********
    
    def __init__(
        self, 
        embed_dim_query: int = 256,  # TODO: Play with these numbers
        num_elements_query: int = 64,
        num_heads: int = 16,
        num_latent_transformer_encoder_layers: int = 4,
        dropout: float = 0.0,
        byte_array_dim: int = 64,  # The dimensionality of each key & value element.
        gsp_id_embedding_dim: int = 16,
        num_fourier_time_features: int = 4,
        num_fourier_spatial_features: int = 8,  # Total for both height and width
        satellite_patch_size_per_dim: int = 2,  # The total patch size will be satellite_patch_size_per_dim x satellite_patch_size_per_dim
        satellite_channel_embedding_dim: int = 4,  # The size of the embedding of the satellite channel ID
        nwp_channel_embedding_dim: int = 4,
        share_weights_across_latent_transformer_layers: bool = True,
        pv_system_id_embedding_dim: int = 16,
        results_file_name: str = "results.csv",
        num_gaussians: int = 2,
        nwp_downsample_factor: int = 4,
    ):
        super().__init__()
        self.embed_dim_query = embed_dim_query
        self.num_elements_query = num_elements_query
        self.num_fourier_time_features = num_fourier_time_features
        self.num_fourier_spatial_features = num_fourier_spatial_features
        self.satellite_patch_size_per_dim = satellite_patch_size_per_dim
        self.results_file_name = results_file_name
        self.gsp_id_embedding_dim = gsp_id_embedding_dim
        self.num_gaussians = num_gaussians
        self.nwp_downsample_factor = nwp_downsample_factor

        #self.query_norm1 = nn.LayerNorm(embed_dim_query)
        #self.query_norm2 = nn.LayerNorm(embed_dim_query)
        #self.byte_array_norm = nn.LayerNorm(byte_array_dim)
        
        # PERCEIVER #################################
        # Layers for the Perceiver model:
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim_query,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=byte_array_dim,
            vdim=byte_array_dim
        )
        
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim_query,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        if share_weights_across_latent_transformer_layers:
            self.latent_transformer = nn.Sequential(
                *[transformer_encoder_layer for _ in range (num_latent_transformer_encoder_layers)]
            )
        else:
            self.latent_transformer = nn.TransformerEncoder(
                encoder_layer=transformer_encoder_layer,
                num_layers=num_latent_transformer_encoder_layers,
            )

        # Feed forwards output layers #################################
        self.output_layers = nn.Sequential(
            nn.Linear(in_features=num_elements_query * embed_dim_query, out_features=embed_dim_query),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim_query, out_features=128),
        )
        
        self.mixture_density_network = MixtureDensityNetwork(
            in_features=128,
            num_gaussians=num_gaussians,
        )
        
        self.gsp_id_embedding = nn.Embedding(
            num_embeddings=MAX_GSP_ID + 1,
            embedding_dim=gsp_id_embedding_dim,
        )
        
        self.gsp_id_embedding_query_generator = nn.Sequential(
            nn.Linear(in_features=gsp_id_embedding_dim, out_features=gsp_id_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=gsp_id_embedding_dim * 2, out_features=gsp_id_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=gsp_id_embedding_dim * 2, out_features=gsp_id_embedding_dim),
            nn.ReLU(),
        )
        
        self.time_fourier_features_query_generator = nn.Sequential(
            nn.Linear(in_features=num_fourier_time_features, out_features=num_fourier_time_features * 2),
            nn.ReLU(),
            nn.Linear(in_features=num_fourier_time_features * 2, out_features=num_fourier_time_features * 2),
            nn.ReLU(),
            nn.Linear(in_features=num_fourier_time_features * 2, out_features=num_fourier_time_features),
            nn.ReLU(),
        )
        
        self.gsp_id_embedding_key_generator = nn.Sequential(
            nn.Linear(in_features=gsp_id_embedding_dim, out_features=gsp_id_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=gsp_id_embedding_dim * 2, out_features=gsp_id_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(in_features=gsp_id_embedding_dim * 2, out_features=gsp_id_embedding_dim),
            nn.ReLU(),
        )
        
        self.satellite_chan_embedding = nn.Embedding(
            num_embeddings=len(SATELLITE_CHANNELS),
            embedding_dim=satellite_channel_embedding_dim,
        )
        self.nwp_chan_embedding = nn.Embedding(
            num_embeddings=len(NWP_CHANNELS),
            embedding_dim=nwp_channel_embedding_dim,
        )
        self.pv_system_id_embedding = nn.Embedding(
            num_embeddings=max(pv_system_id_index_lut) + 1,
            embedding_dim=pv_system_id_embedding_dim,
        )
        
        # Learnable parametrs
        self.query = nn.Parameter(
                torch.randn(
                    num_elements_query, 
                    embed_dim_query
                    - gsp_id_embedding_dim
                    - num_fourier_time_features
                )
            )
        
        # 36 = 4x4 patch +  + 4 channel embedding
        self.learnt_modality_identifier_for_optical_flow = nn.Parameter(
                torch.randn(
                    byte_array_dim 
                    - num_fourier_time_features
                    - num_fourier_spatial_features
                    - gsp_id_embedding_dim # 16
                    - (satellite_patch_size_per_dim ** 2) 
                    - satellite_channel_embedding_dim
                )
            )

        # 7 timesteps of PV data
        self.learnt_modality_identifier_for_pv_yield = nn.Parameter(
                torch.randn(
                    byte_array_dim   # 64
                    - num_fourier_time_features # 4
                    - num_fourier_spatial_features # 8
                    - gsp_id_embedding_dim # 16
                    - pv_system_id_embedding_dim # 16
                    - 7
                )
            )
        
        
        """
            nwp = torch.cat(
                (
                    # batch["nwp_time_fourier_features"] is a tensor of shape [batch_size, seq_length, n_fourier_features]
                    einops.repeat(
                        batch["nwp_time_fourier_features"],
                        "b t n_fourier_features -> b c t h w n_fourier_features",
                        c=nwp_c,
                        t=nwp_t,
                        h=nwp_h, 
                        w=nwp_w,
                    ),
                    einops.repeat(
                        batch["nwp_x_coords_fourier_features"],
                        "b w d -> b c t h w d",
                        c=nwp_c,
                        t=nwp_t,
                        h=nwp_h, 
                        w=nwp_w,
                    ),
                    einops.repeat(
                        batch["nwp_y_coords_fourier_features"],
                        "b h d -> b c t h w d",
                        c=nwp_c,
                        t=nwp_t,
                        h=nwp_h, 
                        w=nwp_w,
                    ),
                    nwp,
                    embedded_nwp_chan_list,
                    einops.repeat(
                        self.learnt_modality_identifier_for_nwp,
                        "d -> original_batch_size c t h w d",
                        original_batch_size=original_batch_size,
                        c=nwp_c,
                        t=nwp_t,
                        h=nwp_h, 
                        w=nwp_w,
                    ),
                ),
                dim=-1,
            )
        """
        
        # 4 timesteps of NWP data
        self.learnt_modality_identifier_for_nwp = nn.Parameter(
                torch.randn(
                    byte_array_dim 
                    - num_fourier_time_features
                    - num_fourier_spatial_features
                    - (satellite_patch_size_per_dim ** 2) 
                    - nwp_channel_embedding_dim
                )
            )
        
        self.learnt_modality_identifier_for_t0_datetime_features = nn.Parameter(torch.randn(byte_array_dim - N_DATETIME_FEATURES))
        
        self.learnt_modality_identifier_for_gsp_id = nn.Parameter(torch.randn(byte_array_dim - gsp_id_embedding_dim))

        #self.query_generator = nn.Sequential(
        #    nn.Linear(in_features=16, out_features=16),
        #    nn.ReLU(),
        #    nn.Linear(in_features=16, out_features=16),
        #    nn.ReLU(),
        #    nn.Linear(in_features=16, out_features=num_elements_query * embed_dim_query),
        #    nn.ReLU(),
        #)
        
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: Contains these keys:
                - opticalflow: Shape: batch_size, time, "channels_index", "y_geostationary_index", "x_geostationary_index"
                
        Returns:
            Tensor of shape batch_size, time
        """
        # byte_arrays is the list of modalities to be concatenated and fed into the perceiver
        byte_arrays: list[torch.Tensor] = []

        #######################################################
        # GENERATE POSITION ENCODING
        get_spatial_and_temporal_fourier_features(
            batch, 
            num_fourier_time_features=self.num_fourier_time_features,
            num_fourier_spatial_features_total=self.num_fourier_spatial_features,  # Total for both height and width
        )
        # batch now has keys like "pv_x_coords_fourier_features"
        
        #######################################################
        # GET DATA INTO THE RIGHT SHAPE #######################
        # satellite data starts with shape "example", "time_index", "channels_index", "y_geostationary_index", "x_geostationary_index"
        if "opticalflow" in batch:
            opticalflow = batch["opticalflow"]
            original_batch_size, original_satellite_seq_len = opticalflow.shape[:2]

            # Reshape so each timestep is seen as a separate example
            opticalflow = einops.rearrange(opticalflow, "b t c h w -> (b t) c h w")

            # Take patches of pixels:
            # Adapted from https://github.com/openclimatefix/satflow/blob/main/satflow/models/utils.py#L54
            opticalflow = einops.rearrange(
                opticalflow, 
                "b c (h dh) (w dw) -> b c h w (dh dw)", 
                dh=self.satellite_patch_size_per_dim, 
                dw=self.satellite_patch_size_per_dim,
            )

            b, c, h, w, d = opticalflow.shape
            
            # Do the same for the temporal & spatial encoding
            
            # Reshape so each timestep is seen as a separate example
            batch["opticalflow_time_fourier_features"] = einops.rearrange(
                batch["opticalflow_time_fourier_features"], 
                "b t n_fourier_features -> (b t) n_fourier_features",
            )
            
            # Take patches of pixels:
            for dim_name in ("x_coords", "y_coords"):
                key_name = f"opticalflow_{dim_name}_fourier_features"
                batch[key_name] = einops.reduce(
                    batch[key_name], 
                    "b (height dh) (width dw) n_fourier_features -> b height width n_fourier_features",
                    reduction="mean",
                    dw=self.satellite_patch_size_per_dim,
                    dh=self.satellite_patch_size_per_dim,
                )
                # Repeat across all timesteps
                batch[key_name] = torch.repeat_interleave(batch[key_name], repeats=original_satellite_seq_len, dim=0)
                # Repeate for each channel
                batch[key_name] = einops.repeat(batch[key_name], "b h w d -> b c h w d", b=b, c=c, h=h, w=w)
                
            # GSP ID per pixel
            gsp_id_per_pixel = batch["gsp_id_per_satellite_pixel"]
            middle_index = (self.satellite_patch_size_per_dim**2 // 2) + (self.satellite_patch_size_per_dim // 2)
            if middle_index > 0:
                middle_index -= 1
            gsp_id_per_pixel = einops.rearrange(
                gsp_id_per_pixel,
                "b (h dh) (w dw) -> b h w (dh dw)",
                dh=self.satellite_patch_size_per_dim, 
                dw=self.satellite_patch_size_per_dim,
            )[..., middle_index]
            gsp_id_per_pixel = self.gsp_id_embedding_key_generator(self.gsp_id_embedding(gsp_id_per_pixel))
            gsp_id_per_pixel = torch.repeat_interleave(gsp_id_per_pixel, repeats=original_satellite_seq_len, dim=0)

            # Encode the channel
            sat_chan_list = torch.arange(c, device=self.device)
            embedded_sat_chan_list = self.satellite_chan_embedding(sat_chan_list)    

            # Concat position encoding for the height and width:
            opticalflow = torch.cat(
                (
                    einops.repeat(batch["opticalflow_time_fourier_features"], "b n_fourier_features -> b c h w n_fourier_features", b=b, c=c, h=h, w=w),
                    batch["opticalflow_x_coords_fourier_features"],
                    batch["opticalflow_y_coords_fourier_features"],
                    einops.repeat(
                        gsp_id_per_pixel,
                        "b h w d-> b c h w d", b=b, c=c, h=h, w=w,
                    ),
                    opticalflow,
                    einops.repeat(embedded_sat_chan_list, "c d -> b c h w d", b=b, c=c, h=h, w=w),
                    einops.repeat(self.learnt_modality_identifier_for_optical_flow, "d -> b c h w d", b=b, c=c, h=h, w=w),
                ),
                dim=-1,
            )

            # opticalflow now has shape: b, c, h, w, perceiver element dim
            # e.g. 128, 10, 6, 6, 32

            # Reshape into batch_size, perceiver element, perceiver element dim, ready for the perceiver
            opticalflow = einops.rearrange(opticalflow, "b c h w d -> b (c h w) d")
            byte_arrays.append(opticalflow)
        
        #######################################################
        # CROSS ATTENTION! ####################################
        
        # Create query
        # num_elements is the number of elements in the set that's fed into cross attention
        # Embed the GSP ID 
        ##query = self.gsp_id_embedding(batch["gsp_id"])
        # Then put the embedding through a linear layer so the model can learn different queries
        # for each element of the Transformer's input.
        ##query = self.query_generator(query)
        # Repeat for each timestep:
        ##query = torch.repeat_interleave(query, repeats=original_satellite_seq_len, dim=0)
        ##query = query.squeeze()
        ##query = einops.repeat(
        ##    query, 
        ##    "b (num_elements embed_dim) -> b num_elements embed_dim", 
        ##    b=b, 
        ##    num_elements=self.num_elements_query,
        ##    embed_dim=self.embed_dim_query,
        ##)
        
        # GSP ID embedding
        # TODO: Reduce reduncancy between this and datetime features.
        #gsp_embedding = self.gsp_id_embedding(batch["gsp_id"])
        #gsp_embedding = torch.cat(
        #    (
        #        gsp_embedding,
        #        einops.repeat(
        #            self.learnt_modality_identifier_for_gsp_id,
        #            "d -> original_batch_size d",
        #            original_batch_size=original_batch_size,
        #        ),
        #    ), 
        #    dim=-1,
        #)
        #gsp_embedding = torch.repeat_interleave(gsp_embedding, repeats=original_satellite_seq_len, dim=0)
        #byte_arrays.append(gsp_embedding.unsqueeze(1))
        
        # Datetime features
        #t0_datetime_features = torch.cat(
        #    (
        #        batch["t0_datetime_features"],
        #        einops.repeat(
        #            self.learnt_modality_identifier_for_t0_datetime_features,
        #            "d -> original_batch_size d",
        #            original_batch_size=original_batch_size,
        #        ),
        #    ),
        #    dim=-1,
        #)
        #t0_datetime_features = torch.repeat_interleave(t0_datetime_features, repeats=original_satellite_seq_len, dim=0)
        #byte_arrays.append(t0_datetime_features.unsqueeze(1))

        # Historical PV data
        # input shape: batch=32, timestep=7, pv system=128
        # output shape: batch=128, n_pv_systems, d_model
        key_padding_mask = None
        if "pv" in batch:
            pv_yield = batch["pv"]
            pv_yield = einops.rearrange(pv_yield, "b t pv_system -> b pv_system t")
            n_pv_systems = pv_yield.shape[1]
            #RuntimeError: Sizes of tensors must match except in dimension 2. Expected size 1 but got size 128 for tensor number 1 in the list.
            pv_yield = torch.cat(
                (
                    # fourier_time_features["pv_time"] is a tensor of shape [batch_size, seq_length, n_fourier_features]
                    einops.repeat(
                        batch["pv_time_fourier_features"][:, -2:-1],  # Just take the t0 timestep
                        "b 1 d -> b n_pv_systems d",
                        n_pv_systems=n_pv_systems,
                    ),
                    batch["pv_x_coords_fourier_features"],
                    batch["pv_y_coords_fourier_features"],
                    self.gsp_id_embedding_key_generator(
                        self.gsp_id_embedding(batch["pv_gsp_id"].to(torch.int32)),
                    ),
                    pv_yield,
                    # Convert pv_system_id to ints for embedding
                    self.pv_system_id_embedding(
                        batch["pv_system_id_index"].to(torch.int32),
                    ),
                    einops.repeat(
                        self.learnt_modality_identifier_for_pv_yield,
                        "d -> original_batch_size n_pv_systems d",
                        original_batch_size=original_batch_size,
                        n_pv_systems=n_pv_systems,
                    ),
                ),
                dim=-1,
            )
            # We will mask missing PV systems later
            pv_yield = pv_yield.float()
            pv_yield = torch.nan_to_num(pv_yield, nan=-1)
            pv_yield = torch.repeat_interleave(pv_yield, repeats=original_satellite_seq_len, dim=0)
            key_padding_mask = torch.repeat_interleave(batch["pv"].isnan().any(dim=1), repeats=original_satellite_seq_len, dim=0)
            byte_arrays.insert(0, pv_yield)
        
        # NWP data
        # Original shape: "example", "time_index", "channels_index", "y_index", "x_index"
        # We want to end up with each element being one timestep for a single channel and a single patch
        if "nwp" in batch:
            # Downsample NWP
            nwp = einops.reduce(
                batch["nwp"],
                "b t c (downsample_factor_h h) (downsample_factor_w w) -> b t c h w",
                "mean",
                downsample_factor_h=self.nwp_downsample_factor,
                downsample_factor_w=self.nwp_downsample_factor,
            )

            # Swap time and channel dims:
            nwp = einops.rearrange(nwp, "b t c h w -> b c t h w")
            
            # Take patches of pixels:
            # Adapted from https://github.com/openclimatefix/satflow/blob/main/satflow/models/utils.py#L54
            nwp = einops.rearrange(
                nwp, 
                "b c t (h dh) (w dw) -> b c t h w (dh dw)", 
                dh=self.satellite_patch_size_per_dim, 
                dw=self.satellite_patch_size_per_dim,
            )
            
            nwp_b, nwp_c, nwp_t, nwp_h, nwp_w, nwp_d = nwp.shape
            
            # Take patches of pixels for the temporal & spatial encoding:
            for dim_name in ("x_coords", "y_coords"):
                key_name = f"nwp_{dim_name}_fourier_features"
                
                batch[key_name] = einops.reduce(
                    batch[key_name], 
                    "b (length dl) n_fourier_features -> b length n_fourier_features",
                    reduction="mean",
                    dl=self.satellite_patch_size_per_dim * self.nwp_downsample_factor,
                )
            
            # Encode the channel
            nwp_chan_list = torch.arange(nwp_c, device=self.device)
            embedded_nwp_chan_list = self.nwp_chan_embedding(nwp_chan_list)
            embedded_nwp_chan_list = einops.repeat(
                embedded_nwp_chan_list, 
                "c d -> b c t h w d", 
                b=nwp_b, 
                c=nwp_c,
                t=nwp_t,
                h=nwp_h, 
                w=nwp_w,
            )

            nwp = torch.cat(
                (
                    # batch["nwp_time_fourier_features"] is a tensor of shape [batch_size, seq_length, n_fourier_features]
                    einops.repeat(
                        batch["nwp_time_fourier_features"],
                        "b t n_fourier_features -> b c t h w n_fourier_features",
                        c=nwp_c,
                        t=nwp_t,
                        h=nwp_h, 
                        w=nwp_w,
                    ),
                    einops.repeat(
                        batch["nwp_x_coords_fourier_features"],
                        "b w d -> b c t h w d",
                        c=nwp_c,
                        t=nwp_t,
                        h=nwp_h, 
                        w=nwp_w,
                    ),
                    einops.repeat(
                        batch["nwp_y_coords_fourier_features"],
                        "b h d -> b c t h w d",
                        c=nwp_c,
                        t=nwp_t,
                        h=nwp_h, 
                        w=nwp_w,
                    ),
                    nwp,
                    embedded_nwp_chan_list,
                    einops.repeat(
                        self.learnt_modality_identifier_for_nwp,
                        "d -> original_batch_size c t h w d",
                        original_batch_size=original_batch_size,
                        c=nwp_c,
                        t=nwp_t,
                        h=nwp_h, 
                        w=nwp_w,
                    ),
                ),
                dim=-1,
            )
            # Reshape into batch_size, perceiver element, perceiver element dim, ready for the perceiver
            nwp = einops.rearrange(nwp, "b c t h w d -> b (c t h w) d")
            nwp = torch.repeat_interleave(nwp, repeats=original_satellite_seq_len, dim=0)
            byte_arrays.append(nwp)

        # Final preparation of data for Perceiver
        # Shape of data going into perciever: [batch_size, perceiver element, perceiver element dim]
        #         # RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 64 but got size 49 for tensor number 1 in the list.
        byte_array = torch.cat(byte_arrays, dim=1)
        
        #print(f"{byte_array.shape}")
        
        if key_padding_mask is not None and key_padding_mask.shape[1] < byte_array.shape[1]:
            # If PV is present, then PV is always the first set of 128 elements (in the 2nd dim).
            # We need to extend the key_padding_mask so it has the same number of elements as byte_array:
            key_padding_mask = torch.cat(
                (
                    key_padding_mask,
                    torch.full(
                        size=(
                            byte_array.shape[0], 
                            byte_array.shape[1] - key_padding_mask.shape[1]
                        ), 
                        fill_value=False, dtype=torch.bool, device=byte_array.device),
                ),
                dim=1
            )

        # Query
        query = einops.repeat(
            self.query, 
            "num_elements embed_dim -> b num_elements embed_dim", 
            b=b, 
            num_elements=self.num_elements_query,
        )
        # Append the GSP ID and forecast_datetime
        query = torch.cat(
            (
                query,
                # The opticalflow time is the time that we want the forecast for.
                einops.repeat(
                    self.time_fourier_features_query_generator(
                        batch["opticalflow_time_fourier_features"],
                    ),
                    "b d -> b num_elements d",
                    b=b,
                    num_elements=self.num_elements_query,
                    d=self.num_fourier_time_features
                ),
                einops.repeat(
                    torch.repeat_interleave(
                        self.gsp_id_embedding_query_generator(
                            self.gsp_id_embedding(
                                    torch.nan_to_num(
                                        batch["gsp_id"],
                                        nan=MAX_GSP_ID,
                                    ).to(torch.int32)
                                ),
                        ),
                        repeats=original_satellite_seq_len,
                        dim=0,
                    ),
                    "b d -> b num_elements d",
                    b=b,
                    num_elements=self.num_elements_query,
                    d=self.gsp_id_embedding_dim
                ),
            ),
            dim=-1
        )
        
        #query = self.query_norm1(query)
        #byte_array = self.byte_array_norm(byte_array)
        
        attn_output, attn_weights = self.cross_attention(query, key=byte_array, value=byte_array, need_weights=False, key_padding_mask=key_padding_mask)
        del attn_weights  # Not used yet.
        
        # LATENT TRANSFORMER
        #attn_output = attn_output + query
        #attn_output = self.query_norm2(attn_output)
        attn_output = self.latent_transformer(attn_output)
        
        # LINEAR LAYERS
        out = self.output_layers(einops.rearrange(attn_output, "b s d -> b (s d)"))
        out = self.mixture_density_network(out)
        
        # Reshape back to b, t, 1
        return {
            key: einops.rearrange(
                tensor,
                "(b t) num_gaussians -> b t num_gaussians", 
                b=original_batch_size, 
                t=original_satellite_seq_len,
                num_gaussians=self.num_gaussians,
            )
            for key, tensor in out.items()
        }
    
    def _training_or_validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int, tag: str) -> dict[str, object]:
        """
        Args:
            batch: The training or validation batch.  A dictionary.
            tag: Either "train" or "validation"
            batch_idx: The index of the batch.
        """
        actual_gsp_power = batch["gsp"]
        network_output = self(batch)
        distribution = get_distribution(network_output)
        neg_log_prob_loss = torch.mean(-distribution.log_prob(actual_gsp_power))
        
        # if batch_idx < 3:
        # Log timeseries of actual GSP power and predicted GSP power
        figure_name = f"{tag}/plot/timeseries/epoch={self.current_epoch};batch_idx={batch_idx}"
        fig = plot_timeseries(batch=batch, network_output=network_output)
        self.logger.experiment[figure_name].log(fig)
        plt.close(fig)

        return neg_log_prob_loss
  
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self._training_or_validation_step(batch=batch, batch_idx=batch_idx, tag="train")
    
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self._training_or_validation_step(batch=batch, batch_idx=batch_idx, tag="validation")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


model = Model()
#model = Model.load_from_checkpoint("/home/jack/dev/ocf/predict_pv_yield/notebooks/.neptune/Untitled/PRED-747/checkpoints/epoch=44-step=377999.ckpt")


neptune_logger = NeptuneLogger(
    project="OpenClimateFix/predict-pv-yield",
    prefix=""
)


trainer = pl.Trainer(
    # gpus=[0],
    logger=neptune_logger,
)

trainer.fit(model, train_dataloader, test_dataloader)