from predict_pv_yield.models.conv3d.model import Model, params

import os

import torch.nn.functional as F
import pytorch_lightning as pl

from predict_pv_yield.data.dataloader import get_dataloaders

from predict_pv_yield.visualisation.visualisation import plot_example

from neptune.new.integrations.pytorch_lightning import NeptuneLogger

import logging

logging.basicConfig()
_LOG = logging.getLogger("predict_pv_yield")
_LOG.setLevel(logging.DEBUG)



def main():
    train_dataloader, validation_dataloader = get_dataloaders()
    model = Model()
    logger = NeptuneLogger(project='OpenClimateFix/predict-pv-yield')
    logger.log_hyperparams(params)
    _LOG.info(f'logger.version = {logger.version}')
    trainer = pl.Trainer(gpus=0, max_epochs=1, logger=logger)
    trainer.fit(model, train_dataloader)

if __name__ == '__main__':
    main()
