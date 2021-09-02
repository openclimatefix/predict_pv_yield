from predict_pv_yield.models.loss import WeightedLosses
import torch
import numpy as np
import pytest


def test_weight_losses_weights():
    forecast_length = 2
    w = WeightedLosses(forecast_length=forecast_length)

    assert w.weights.cpu().numpy()[0] == pytest.approx(2 / 3)
    assert w.weights.cpu().numpy()[1] == pytest.approx(1 / 3)


def test_mae_exp():
    forecast_length = 2
    w = WeightedLosses(forecast_length=forecast_length)

    output = torch.Tensor([1, 3])
    target = torch.Tensor([1, 5])

    loss = w.get_mae_exp(output=output, target=target)

    assert loss == pytest.approx(2 / 3)  # (1-1)*2/3 + (5-3)*1/4


def test_mse_exp():
    forecast_length = 2
    w = WeightedLosses(forecast_length=forecast_length)

    output = torch.Tensor([1, 3])
    target = torch.Tensor([1, 5])

    loss = w.get_mse_exp(output=output, target=target)

    assert loss == pytest.approx(4 / 3)  # (1-1)*2/3 + (5-3)^2*1/4


def test_mae_exp_rand():
    forecast_length = 6
    batch_size = 32

    w = WeightedLosses(forecast_length=6)

    output = torch.randn(batch_size, forecast_length)
    target = torch.randn(batch_size, forecast_length)

    loss = w.get_mae_exp(output=output, target=target)
    assert loss > 0


def test_mse_exp_rand():
    forecast_length = 6
    batch_size = 32

    w = WeightedLosses(forecast_length=6)

    output = torch.randn(batch_size, forecast_length)
    target = torch.randn(batch_size, forecast_length)

    loss = w.get_mse_exp(output=output, target=target)
    assert loss > 0
