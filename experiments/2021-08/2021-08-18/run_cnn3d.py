import logging

import pytorch_lightning as pl
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from predict_pv_yield.data.dataloader import get_dataloaders
from predict_pv_yield.models.conv3d.model import Model, model_configuration_default

logging.basicConfig()
_LOG = logging.getLogger("predict_pv_yield")
_LOG.setLevel(logging.DEBUG)


def main():
    train_dataloader, validation_dataloader = get_dataloaders(n_train_data=10, n_validation_data=10)
    model = Model()
    logger = NeptuneLogger(project='OpenClimateFix/predict-pv-yield')
    logger.log_hyperparams(model_configuration_default)
    _LOG.info(f'logger.version = {logger.version}')
    trainer = pl.Trainer(gpus=1, max_epochs=10, logger=logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # run validation
    trainer.validate(model, validation_dataloader)


if __name__ == '__main__':
    main()


# Managed to run it on GCP.
#  Results are logged to https://app.neptune.ai/o/OpenClimateFix/org/predict-pv-yield/e/PRED-120/monitoring
# Notes:
# 1. Large training set, and one epoch took a day, so should use GPU for this model. I was a bit suprised as I didnt
# think the model was so big.
# 2. Need to work on validationm general validation method. Good to base line against a really simple model. For
# validation might need to think carefully about metrics that will be used.
