from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPooling(nn.Module):
    """Mean pooling over token sequence dimension."""

    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        self.dim = dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, D) -> (B, D)
        return tokens.mean(dim=1)


def compute_latent_embeddings(context_tokens: torch.Tensor) -> torch.Tensor:
    """Compute L2-normalized mean-pooled latent embeddings."""
    emb = context_tokens.mean(dim=1)
    return F.normalize(emb, dim=-1)
