import os
from nowcasting_dataset.dataset import NetCDFDataset, worker_init_fn
import torch
from typing import Tuple, Optional
import logging
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split


_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.DEBUG)


def get_dataloaders(
    n_train_data: int = 24900,
    n_validation_data: int = 900,
    cloud: str = "gcp",
    temp_path=".",
    data_path="prepared_ML_training_data/v4/",
) -> Tuple:

    data_module = NetCDFDataModule(
        temp_path=temp_path, data_path=data_path, cloud=cloud, n_train_data=n_train_data, n_val_data=n_validation_data
    )

    train_dataloader = data_module.train_dataloader()

    validation_dataloader = data_module.val_dataloader()

    return train_dataloader, validation_dataloader


class NetCDFDataModule(LightningDataModule):
    """
    Example of LightningDataModule for NETCDF dataset.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        temp_path: str = ".",
        n_train_data: int = 24900,
        n_val_data: int = 1000,
        cloud: str = "aws",
        num_workers: int = 8,
        pin_memory: bool = True,
        data_path="prepared_ML_training_data/v4/",
    ):
        super().__init__()

        self.temp_path = temp_path
        self.data_path = data_path
        self.cloud = cloud
        self.n_train_data = n_train_data
        self.n_val_data = n_val_data
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataloader_config = dict(
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=8,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )

    def train_dataloader(self):
        return NetCDFDataset(
            self.n_train_data,
            os.path.join(self.data_path, "train"),
            os.path.join(self.temp_path, "train"),
            cloud=self.cloud,
        )

    def val_dataloader(self):
        return NetCDFDataset(
            self.n_train_data,
            os.path.join(self.data_path, "validation"),
            os.path.join(self.temp_path, "validation"),
            cloud=self.cloud,
        )

    def test_dataloader(self):
        # TODO need to change this to a test folder
        return NetCDFDataset(
                self.n_train_data,
                os.path.join(self.data_path, "validation"),
                os.path.join(self.temp_path, "validation"),
                cloud=self.cloud,
            )
