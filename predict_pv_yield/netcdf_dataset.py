import gcsfs
import os
import xarray as xr
from nowcasting_dataset import utils as nd_utils
from nowcasting_dataset import example
import torch


class NetCDFDataset(torch.utils.data.Dataset):
    """Loads data saved by the `prepare_ml_training_data.py` script."""

    def __init__(self, n_batches: int, src_path: str, tmp_path: str):
        """
        Args:
          n_batches: Number of batches available on disk.
          src_path: The full path (including 'gs://') to the data on
            Google Cloud storage.
          tmp_path: The full path to the local temporary directory
            (on a local filesystem).
        """
        self.n_batches = n_batches
        self.src_path = src_path
        self.tmp_path = tmp_path

    def per_worker_init(self, worker_id: int):
        self.gcs = gcsfs.GCSFileSystem()

    def __len__(self):
        return self.n_batches

    def __getitem__(self, batch_idx: int) -> example.Example:
        """Returns a whole batch at once.

        Args:
          batch_idx: The integer index of the batch. Must be in the range
          [0, self.n_batches).

        Returns:
            NamedDict where each value is a numpy array. The size of this
            array's first dimension is the batch size.
        """
        if not 0 <= batch_idx < self.n_batches:
            raise IndexError(
                'batch_idx must be in the range'
                f' [0, {self.n_batches}), not {batch_idx}!')
        netcdf_filename = nd_utils.get_netcdf_filename(batch_idx)
        remote_netcdf_filename = os.path.join(self.src_path, netcdf_filename)
        local_netcdf_filename = os.path.join(self.tmp_path, netcdf_filename)
        self.gcs.get(remote_netcdf_filename, local_netcdf_filename)
        netcdf_batch = xr.load_dataset(local_netcdf_filename)
        os.remove(local_netcdf_filename)

        batch = example.Example(
            sat_datetime_index=netcdf_batch.sat_time_coords,
            nwp_target_time=netcdf_batch.nwp_time_coords)
        for key in [
            'nwp', 'nwp_x_coords', 'nwp_y_coords',
            'sat_data', 'sat_x_coords', 'sat_y_coords',
            'pv_yield', 'pv_system_id', 'pv_system_row_number'
        ] + example.DATETIME_FEATURE_NAMES:
            batch[key] = netcdf_batch[key]
        batch = example.to_numpy(batch)

        return batch


def worker_init_fn(worker_id):
    """Configures each dataset worker process.

    Just has one job!  To call NowcastingDataset.per_worker_init().
    """
    # get_worker_info() returns information specific to each worker process.
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        print('worker_info is None!')
    else:
        # The NowcastingDataset copy in this worker process.
        dataset_obj = worker_info.dataset
        dataset_obj.per_worker_init(worker_info.id)
