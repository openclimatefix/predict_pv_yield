import os
from predict_pv_yield.netcdf_dataset import NetCDFDataset, worker_init_fn
import torch
from typing import Tuple


def get_dataloaders(n_train_data: int = 24900, n_validation_data: int = 900) -> Tuple:
    DATA_PATH = 'gs://solar-pv-nowcasting-data/prepared_ML_training_data/v4/'
    TEMP_PATH = '.'

    train_dataset = NetCDFDataset(
        n_train_data,
        os.path.join(DATA_PATH, 'train'),
        os.path.join(TEMP_PATH, 'train'))

    validation_dataset = NetCDFDataset(
        n_validation_data,
        os.path.join(DATA_PATH, 'validation'),
        os.path.join(TEMP_PATH, 'validation'))

    dataloader_config = dict(
        pin_memory=True,
        num_workers=8,
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