import torch
import torch.nn as nn


@torch.no_grad()
def update_target_model(
    online_model: nn.Module, target_model: nn.Module, momentum: float = 0.996
) -> None:
    """Exponential Moving Average (EMA) update of target model parameters from online model."""
    for online_param, target_param in zip(
        online_model.parameters(), target_model.parameters()
    ):
        target_param.data.mul_(momentum)
        target_param.data.add_(online_param.data, alpha=1.0 - momentum)
