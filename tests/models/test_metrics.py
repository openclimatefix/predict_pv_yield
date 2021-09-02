from predict_pv_yield.models.metrics import mse_each_forecast_horizon, mae_each_forecast_horizon
import torch
import pytest


def test_mse_each_forecast_horizon():

    output = torch.Tensor([[1, 3], [1, 6]])
    target = torch.Tensor([[1, 5], [1, 3]])

    loss = mse_each_forecast_horizon(output=output, target=target)

    assert loss.cpu().numpy()[0] == 0
    assert loss.cpu().numpy()[1] == 2*2 + 3*3


def test_mae_each_forecast_horizon():

    output = torch.Tensor([[1, 3], [1, 6]])
    target = torch.Tensor([[1, 5], [1, 3]])

    loss = mae_each_forecast_horizon(output=output, target=target)

    assert loss.cpu().numpy()[0] == 0
    assert loss.cpu().numpy()[1] == 2 + 3

