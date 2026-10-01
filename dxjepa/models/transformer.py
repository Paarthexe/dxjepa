from typing import Optional
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron with GELU activation."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = self.fc1.weight.device
        x = x.to(device)
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Standard pre-LayerNorm Transformer encoder block."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = self.norm1.weight.device
        x = x.to(device)
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + attn_out
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x


class SharedTransformer(nn.Module):
    """Shared Transformer encoder trunk with layer normalization at output."""

    def __init__(self, depth: int = 12, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.blocks) > 0:
            x = x.to(self.blocks[0].norm1.weight.device)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


def build_encoder_trunk(
    depth: int = 12, embed_dim: int = 768, num_heads: int = 12
) -> SharedTransformer:
    """Build shared encoder trunk with the given depth and dimension."""
    return SharedTransformer(depth=depth, embed_dim=embed_dim, num_heads=num_heads)
