from predict_pv_yield.models.baseline.last_value import Model
from predict_pv_yield.data.dataloader import get_dataloaders
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

import pytorch_lightning as pl
import logging

logging.basicConfig()
_LOG = logging.getLogger("predict_pv_yield")
_LOG.setLevel(logging.DEBUG)


def main():
    train_dataloader, validation_dataloader = get_dataloaders(n_train_data=10, n_validation_data=10)
    model = Model()
    logger = NeptuneLogger(project="OpenClimateFix/predict-pv-yield")
    _LOG.info(f"logger.version = {logger.version}")
    trainer = pl.Trainer(gpus=0, max_epochs=10, logger=logger)

    # dont need to train baseline model
    # trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    trainer.validate(model, validation_dataloader)


if __name__ == "__main__":
    main()


# https://app.neptune.ai/o/OpenClimateFix/org/predict-pv-yield/e/PRED-124/charts
#
# {'Validation: MAE': 0.08886486291885376, 'Validation: MSE': 0.02136283740401268}
#
