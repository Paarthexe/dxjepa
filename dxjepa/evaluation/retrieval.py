from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_cross_retrieval_matrix(
    sar_latents: torch.Tensor, optical_latents: torch.Tensor
) -> torch.Tensor:
    """Compute dot-product / cosine similarity matrix between SAR and Optical normalized latents."""
    return torch.matmul(sar_latents, optical_latents.T)


@torch.no_grad()
def evaluate_retrieval_f1(
    model: nn.Module,
    val_loader: DataLoader,
    device: Optional[torch.device] = None,
    k_list: List[int] = [5, 10],
    d_common: int = 512,
) -> Dict[str, float]:
    """Evaluate cross-modal and within-modal retrieval using graded multi-label F1@K metrics.

    Evaluates:
      - S1->S2 (SAR -> Optical) cross-retrieval on shared common subspaces
      - S2->S1 (Optical -> SAR) cross-retrieval on shared common subspaces
      - S1->S1 (SAR -> SAR) retrieval on full representations (excluding self-match)
      - S2->S2 (Optical -> Optical) retrieval on full representations (excluding self-match)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    sar_full = []
    opt_full = []
    sar_common = []
    opt_common = []
    labels_list = []

    for batch in val_loader:
        sar = batch[0].to(device)
        optical = batch[1].to(device)
        lbls = batch[2]

        sar_tokens = model.online_encoder(model.sar_stem(sar))
        opt_tokens = model.online_encoder(model.optical_stem(optical))

        z_sar = model.mean_pool(sar_tokens)
        z_opt = model.mean_pool(opt_tokens)

        sar_full.append(F.normalize(z_sar, dim=-1).cpu())
        opt_full.append(F.normalize(z_opt, dim=-1).cpu())

        sar_common.append(F.normalize(z_sar[:, :d_common], dim=-1).cpu())
        opt_common.append(F.normalize(z_opt[:, :d_common], dim=-1).cpu())

        labels_list.append(lbls.cpu())

    if not sar_full:
        return {}

    z_sar_full = torch.cat(sar_full, dim=0)
    z_opt_full = torch.cat(opt_full, dim=0)
    z_sar_common = torch.cat(sar_common, dim=0)
    z_opt_common = torch.cat(opt_common, dim=0)

    y = torch.cat(labels_list, dim=0).float()
    n = y.size(0)
    if n < 2:
        return {}

    y_sum = y.sum(dim=1, keepdim=True).clamp(min=1.0)
    intersection = torch.matmul(y, y.T)
    graded_p_mat = intersection / y_sum.T
    graded_r_mat = intersection / y_sum

    results = {}
    tasks = [
        ("S1->S2", z_sar_common, z_opt_common, False),
        ("S2->S1", z_opt_common, z_sar_common, False),
        ("S1->S1", z_sar_full, z_sar_full, True),
        ("S2->S2", z_opt_full, z_opt_full, True),
    ]

    for task_name, q, g, exclude_self in tasks:
        sim_mat = torch.matmul(q, g.T)
        if exclude_self:
            sim_mat.fill_diagonal_(-float("inf"))

        for k in k_list:
            topk_limit = min(k, n - 1 if exclude_self else n)
            topk_idx = sim_mat.topk(topk_limit, dim=1).indices
            p_k = torch.gather(graded_p_mat, 1, topk_idx).mean(dim=1)
            r_k = torch.gather(graded_r_mat, 1, topk_idx).mean(dim=1)
            f1_k = (
                torch.where(
                    p_k + r_k > 0,
                    2 * p_k * r_k / (p_k + r_k),
                    torch.zeros_like(p_k),
                )
                .mean()
                .item()
                * 100.0
            )
            results[f"{task_name}_F1@{k}"] = f1_k

    return results
