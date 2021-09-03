from predict_pv_yield.models.conv3d.model import Model

from predict_pv_yield.data.dataloader import get_dataloaders
from pytorch_lightning.utilities.cloud_io import load as pl_load
import torch

weights = './weights/conv3d/last.ckpt'
checkpoint = pl_load(weights, map_location=torch.device('cpu'))

model = Model(conv3d_channels=32,
              fc1_output_features=32,
              fc2_output_features=16,
              fc3_output_features=16,
              include_time=False,
              number_of_conv3d_layers=4)
model.load_from_checkpoint(weights)

train_dataset, validation_dataset = get_dataloaders()



