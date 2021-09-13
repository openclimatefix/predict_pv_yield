from predict_pv_yield.models.perceiver.perceiver import PerceiverRNN, params, TOTAL_SEQ_LEN
from predict_pv_yield.data.dataloader import FakeDataset
import torch


def test_init_model():
    """Initilize the model"""
    _ = PerceiverRNN(
        history_minutes=3, forecast_minutes=3, nwp_channels=params["nwp_channels"], output_variable="gsp_yield"
    )


def test_model_forward():

    model = PerceiverRNN(
        history_minutes=params["history_minutes"],
        forecast_minutes=params["forecast_minutes"],
        nwp_channels=params["nwp_channels"],
        output_variable="gsp_yield",
    )  # doesnt do anything

    # setup parameters TODO, should take this from a config file
    batch_size = 2
    seq_length = TOTAL_SEQ_LEN
    seq_length_30 = 4
    width = 16  # this doesnt seem to matter
    height = 16  # this doesnt seem to matterO
    channel = len(params["sat_channels"])
    nwp_channels = len(params["nwp_channels"])

    # set up fake data
    # satelite data
    x = {"sat_data": torch.randn(batch_size, seq_length, width, height, channel)}

    # numerical weather predictions
    x["nwp"] = torch.randn(batch_size, nwp_channels, seq_length, 2, 2)

    # setup fake hour of the day, an dat of the year parameters
    for time_variable in ["hour_of_day_sin", "hour_of_day_cos", "day_of_year_sin", "day_of_year_cos"]:
        x[time_variable] = torch.randn(batch_size, seq_length)

    # setup pv index number, make suer model can handle it as floats
    x["pv_system_row_number"] = torch.randint(high=940, size=(batch_size, 1)).type(torch.FloatTensor)

    # pv yield data
    x["pv_yield"] = torch.randn(batch_size, seq_length, 1)
    x["gsp_yield"] = torch.randn(batch_size, 4, 1)

    # run data through model
    y = model(x)

    # check out put is the correct shape
    assert len(y.shape) == 2
    assert y.shape[0] == batch_size
    assert y.shape[1] == params["forecast_minutes"] // 30
