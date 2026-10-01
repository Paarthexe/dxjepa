import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """2D Patch Embedding layer converting image tensors into patch token sequences."""

    def __init__(self, in_channels: int, embed_dim: int = 768, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> proj: (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class ModalityStem(nn.Module):
    """Modality-specific stem that projects raw channel inputs and adds learned positional embeddings."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        num_patches: int = 196,
        patch_size: int = 16,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = self.pos_embed.device
        x = x.to(device)
        return self.patch_embed(x) + self.pos_embed
