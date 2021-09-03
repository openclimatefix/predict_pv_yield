import torch


def mse_each_forecast_horizon(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Get MSE for each forecast horizon
    output - The model estimate of size (batch_size, forecast_length)
    target - The truth of size (batch_size, forecast_length)

    Return a tensor of size (forecast_length)
    """

    return torch.sum((output - target) ** 2, dim=0)


def mae_each_forecast_horizon(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Get MAE for each forecast horizon
    output - The model estimate of size (batch_size, forecast_length)
    target - The truth of size (batch_size, forecast_length)

    Return a tensor of size (forecast_length)
    """

    return torch.sum(torch.abs(output - target), dim=0)
