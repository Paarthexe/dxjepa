import torch
import torch.nn as nn


class XJEPAPSALoss(nn.Module):
    """Predictive Structure Alignment (PSA) Loss with learned Mahalanobis-like metric."""

    def __init__(self, embed_dim: int = 768, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.M_weight = nn.Parameter(torch.eye(embed_dim) * 0.1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = pred.device
        target = target.to(device)
        delta = pred - target
        M_w = self.M_weight.to(device)
        W_delta = torch.matmul(delta, M_w)
        quad_form = torch.sum(W_delta ** 2, dim=-1) + self.eps * torch.sum(delta ** 2, dim=-1)
        quad_form_clamped = torch.clamp(quad_form, min=0.0, max=10000.0)
        psa_loss = torch.mean(torch.sqrt(quad_form_clamped + 1e-6))
        return psa_loss
