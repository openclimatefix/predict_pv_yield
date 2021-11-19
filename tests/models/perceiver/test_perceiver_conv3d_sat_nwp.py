from predict_pv_yield.models.perceiver.perceiver_conv3d_nwp_sat import Model, params, TOTAL_SEQ_LEN
from predict_pv_yield.data.dataloader import FakeDataset
import torch
from nowcasting_dataset.config.model import Configuration




def test_init_model():
    """Initilize the model"""
    _ = Model(
        history_minutes=3, forecast_minutes=3, nwp_channels=params["nwp_channels"], output_variable="gsp_yield"
    )


def test_model_forward(configuration_perceiver):

    dataset_configuration = configuration_perceiver

    model = Model(
        history_minutes=params["history_minutes"],
        forecast_minutes=params["forecast_minutes"],
        nwp_channels=params["nwp_channels"],
        output_variable="gsp_yield",
    )  # doesnt do anything

    # set up fake data
    train_dataset = FakeDataset(configuration=dataset_configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    # get data
    x = next(iter(train_dataloader))

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == dataset_configuration.process.batch_size
    assert y.shape[1] == params["forecast_minutes"] // 30


def test_model_forward_no_forward_satelite(configuration_perceiver):

    dataset_configuration = configuration_perceiver

    model = Model(
        history_minutes=params["history_minutes"],
        forecast_minutes=params["forecast_minutes"],
        nwp_channels=params["nwp_channels"],
        output_variable="gsp_yield",
        use_future_satellite_images=False
    )  # doesnt do anything

    train_dataset = FakeDataset(configuration=dataset_configuration)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    # get data
    x = next(iter(train_dataloader))

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == dataset_configuration.process.batch_size
    assert y.shape[1] == params["forecast_minutes"] // 30
