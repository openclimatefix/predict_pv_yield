import logging

import pytorch_lightning as pl
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from predict_pv_yield.data.dataloader import get_dataloaders
from predict_pv_yield.models.conv3d.model import Model, model_configuration_default

logging.basicConfig()
_LOG = logging.getLogger("predict_pv_yield")
_LOG.setLevel(logging.DEBUG)

_LOG = logging.getLogger("nowcasting_dataset")
_LOG.setLevel(logging.INFO)


def main():
    train_dataloader, validation_dataloader = get_dataloaders(
        n_train_data=24900,
        n_validation_data=1000,
        data_path="gs://solar-pv-nowcasting-data/prepared_ML_training_data/v4/",
        cloud="gcp",
    )

    model_configuration = dict(conv3d_channels=8, kennel=3, number_of_conv3d_layers=6)
    model = Model(model_configuration=model_configuration)

    logger = NeptuneLogger(project="OpenClimateFix/predict-pv-yield")
    logger.log_hyperparams(model_configuration_default)
    _LOG.info(f"logger.version = {logger.version}")
    trainer = pl.Trainer(gpus=1, max_epochs=10, logger=logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # run validation
    trainer.validate(model, validation_dataloader)


if __name__ == "__main__":
    main()
