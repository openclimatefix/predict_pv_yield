import logging

import pytorch_lightning as pl
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from predict_pv_yield.data.dataloader import get_dataloaders
from predict_pv_yield.models.conv3d.model import Model, model_configuration_default

logging.basicConfig()
_LOG = logging.getLogger("predict_pv_yield")
_LOG.setLevel(logging.DEBUG)

_LOG = logging.getLogger("nowcasting_dataset")
_LOG.setLevel(logging.DEBUG)


def main():
    train_dataloader, validation_dataloader = get_dataloaders(n_train_data=24900, n_validation_data=1000, cloud='aws')
    model = Model()

    logger = NeptuneLogger(project='OpenClimateFix/predict-pv-yield')
    logger.log_hyperparams(model_configuration_default)
    _LOG.info(f'logger.version = {logger.version}')
    trainer = pl.Trainer(gpus=0, max_epochs=10)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # run validation
    trainer.validate(model, validation_dataloader)


if __name__ == '__main__':
    main()
