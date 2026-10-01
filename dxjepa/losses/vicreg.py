from typing import Tuple
import torch
import torch.nn.functional as F


def split_common_unique(
    z: torch.Tensor, d_common: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split representation into modality-shared (common) and modality-specific (unique) subspaces."""
    return z[:, :d_common], z[:, d_common:]


def cross_covariance_penalty(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute normalized Frobenius norm of cross-covariance between two feature spaces."""
    n = a.size(0)
    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    cross_cov = a_c.T @ b_c / max(1, n - 1)
    return cross_cov.pow(2).sum() / (a.size(1) * b.size(1))


def decoupled_vicreg_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    d_common: int = 512,
    lambda_inv: float = 1.0,
    lambda_var: float = 25.0,
    lambda_cov: float = 25.0,
    lambda_excl: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decoupled VICReg loss with invariance on shared subspaces, variance/covariance on both,

    and cross-subspace exclusion penalty.
    """
    z_a_2d = z_a.mean(dim=1) if z_a.ndim == 3 else z_a
    z_b_2d = z_b.mean(dim=1) if z_b.ndim == 3 else z_b

    c_a, u_a = split_common_unique(z_a_2d, d_common)
    c_b, u_b = split_common_unique(z_b_2d, d_common)

    # Invariance loss only on common features
    sim_loss = F.mse_loss(c_a, c_b)

    def variance_loss(z: torch.Tensor) -> torch.Tensor:
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1.0 - std_z))

    var_loss = 0.25 * (
        variance_loss(c_a)
        + variance_loss(c_b)
        + variance_loss(u_a)
        + variance_loss(u_b)
    )

    def covariance_loss(z: torch.Tensor) -> torch.Tensor:
        batch_size, dim = z.shape
        zc = z - z.mean(dim=0, keepdim=True)
        cov = zc.T @ zc / max(1, batch_size - 1)
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag / (dim * dim)

    cov_loss = 0.25 * (
        covariance_loss(c_a)
        + covariance_loss(c_b)
        + covariance_loss(u_a)
        + covariance_loss(u_b)
    )

    # Mutual exclusion penalty across common and unique subspaces
    excl_loss = (
        cross_covariance_penalty(c_a, u_a)
        + cross_covariance_penalty(c_b, u_b)
        + cross_covariance_penalty(u_a, u_b)
    ) / 3.0

    total = (
        lambda_inv * sim_loss
        + lambda_var * var_loss
        + lambda_cov * cov_loss
        + lambda_excl * excl_loss
    )
    return total, sim_loss, var_loss, cov_loss, excl_loss


def vicreg_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard VICReg return tuple (total, sim, var, cov)."""
    res = decoupled_vicreg_loss(z_a, z_b, **kwargs)
    return res[0], res[1], res[2], res[3]
