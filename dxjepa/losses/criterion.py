from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from dxjepa.losses.vicreg import decoupled_vicreg_loss


def l2_prediction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error prediction loss."""
    return F.mse_loss(prediction, target)


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    model: nn.Module,
    lambda_inv: float = 1.0,
    lambda_var: float = 25.0,
    lambda_cov: float = 25.0,
    lambda_excl: float = 5.0,
    lambda_psa: float = 0.1,
    d_common: int = 512,
) -> Dict[str, torch.Tensor]:
    """Compute total X-JEPA loss: L2 prediction loss + PSA loss + Decoupled VICReg loss."""
    sar_to_opt_pred = outputs["sar_to_opt_pred"]
    opt_to_sar_pred = outputs["opt_to_sar_pred"]
    sar_target = outputs["sar_target"]
    opt_target = outputs["optical_target"]
    sar_context = outputs["sar_context"]
    optical_context = outputs["optical_context"]

    # 1. Symmetric L2 Prediction Loss
    l2_sar_opt = F.mse_loss(sar_to_opt_pred, opt_target)
    l2_opt_sar = F.mse_loss(opt_to_sar_pred, sar_target)
    loss_l2 = 0.5 * (l2_sar_opt + l2_opt_sar)

    # 2. Symmetric Predictive Structure Alignment (PSA) Loss
    psa_loss_fn = getattr(model, "psa_loss_fn")
    psa_sar_opt = psa_loss_fn(sar_to_opt_pred, opt_target)
    psa_opt_sar = psa_loss_fn(opt_to_sar_pred, sar_target)
    loss_psa = 0.5 * (psa_sar_opt + psa_opt_sar)

    loss_pred = loss_l2 + lambda_psa * loss_psa

    # 3. Decoupled VICReg on pooled context embeddings
    mean_pool = getattr(model, "mean_pool", lambda t: t.mean(dim=1))
    z_sar = mean_pool(sar_context)
    z_opt = mean_pool(optical_context)

    loss_vicreg, sim_loss, var_loss, cov_loss, excl_loss = decoupled_vicreg_loss(
        z_sar,
        z_opt,
        d_common=d_common,
        lambda_inv=lambda_inv,
        lambda_var=lambda_var,
        lambda_cov=lambda_cov,
        lambda_excl=lambda_excl,
    )

    total_loss = loss_pred + loss_vicreg

    return {
        "total_loss": total_loss,
        "loss_l2": loss_l2,
        "loss_psa": loss_psa,
        "loss_pred": loss_pred,
        "loss_vicreg": loss_vicreg,
        "loss_inv": sim_loss,
        "loss_var": var_loss,
        "loss_cov": cov_loss,
        "loss_excl": excl_loss,
    }
