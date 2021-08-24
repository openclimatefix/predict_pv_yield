import os
from nowcasting_dataset.dataset import NetCDFDataset, worker_init_fn
import torch
from typing import Tuple
import logging

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.DEBUG)


def get_dataloaders(
    n_train_data: int = 24900,
    n_validation_data: int = 900,
    cloud: str = "gcp",
    temp_path=".",
    data_path="prepared_ML_training_data/v4/",
) -> Tuple:

    train_dataset = NetCDFDataset(
        n_train_data, os.path.join(data_path, "train"), os.path.join(temp_path, "train"), cloud=cloud
    )

    validation_dataset = NetCDFDataset(
        n_validation_data, os.path.join(data_path, "validation"), os.path.join(temp_path, "validation"), cloud=cloud
    )

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

    train_dataloader = torch.utils.data.DataLoader(train_dataset, **dataloader_config)

    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, **dataloader_config)

    return train_dataloader, validation_dataloader
