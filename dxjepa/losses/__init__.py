from dxjepa.losses.psa import XJEPAPSALoss
from dxjepa.losses.vicreg import (
    decoupled_vicreg_loss,
    vicreg_loss,
    split_common_unique,
    cross_covariance_penalty,
)
from dxjepa.losses.criterion import l2_prediction_loss, compute_loss

__all__ = [
    "XJEPAPSALoss",
    "decoupled_vicreg_loss",
    "vicreg_loss",
    "split_common_unique",
    "cross_covariance_penalty",
    "l2_prediction_loss",
    "compute_loss",
]
