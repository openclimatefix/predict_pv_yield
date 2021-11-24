from predict_pv_yield.models.conv3d.model import Model

from predict_pv_yield.data.dataloader import get_dataloaders
from pytorch_lightning.utilities.cloud_io import load as pl_load
import torch
import pandas as pd

from predict_pv_yield.visualisation.line import plot_one_result, plot_batch_results

weights = "./weights/conv3d/epoch_009.ckpt"
checkpoint = pl_load(weights, map_location=torch.device("cpu"))

model = Model(
    conv3d_channels=32,
    fc1_output_features=128,
    fc2_output_features=128,
    fc3_output_features=64,
    include_time=True,
    forecast_len=12,
    history_len=6,
    number_of_conv3d_layers=6,
)
model.load_state_dict(checkpoint["state_dict"])

train_dataset, validation_dataset = get_dataloaders(
    cloud="gcp", data_path="gs://solar-pv-nowcasting-data/prepared_ML_training_data/v4/"
)
validation_dataset = iter(validation_dataset)
x = next(validation_dataset)

y_hat_all = model(x)

# plot one
batch_index = 0
y = x["pv_yield"][batch_index][7:, 0].detach().numpy()
y_hat = y_hat_all[batch_index].detach().numpy()
time = pd.to_datetime(x["sat_datetime_index"][batch_index][7:].detach().numpy(), unit="s")

fig = plot_one_result(x=time, y=y, y_hat=y_hat)
fig.show(renderer="browser")

# plot all of batch
y = x["pv_yield"][:, 7:, 0].detach().numpy()
y_hat = y_hat_all.detach().numpy()
time = [pd.to_datetime(x, unit="s") for x in x["sat_datetime_index"][:, 7:].detach().numpy()]

fig = plot_batch_results(x=time, y=y, y_hat=y_hat, model_name=model.name)
fig.show(renderer="browser")
