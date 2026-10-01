from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from dxjepa.models.stem import ModalityStem
from dxjepa.models.transformer import build_encoder_trunk
from dxjepa.models.predictor import XJEPAPredictor
from dxjepa.models.pooling import MeanPooling
from dxjepa.losses.psa import XJEPAPSALoss

try:
    from dxjepa.data.masking import generate_disjoint_masks, split_visible_masked
except ImportError:
    def generate_disjoint_masks(
        batch_size: int,
        device: torch.device,
        num_patches: int = 196,
        context_ratio: float = 0.5,
        target_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_context = int(num_patches * context_ratio)
        num_target = int(num_patches * target_ratio)
        context_visible_indices = []
        target_masked_indices = []
        for _ in range(batch_size):
            perm = torch.randperm(num_patches, device=device)
            context_visible_indices.append(perm[:num_context])
            target_masked_indices.append(perm[num_context : num_context + num_target])
        return (torch.stack(context_visible_indices), torch.stack(target_masked_indices))

    def split_visible_masked(
        tokens: torch.Tensor,
        visible_idx: torch.Tensor,
        masked_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, d = tokens.shape
        device = tokens.device
        visible_idx = visible_idx.to(device)
        masked_idx = masked_idx.to(device)
        return (
            torch.gather(tokens, 1, visible_idx.unsqueeze(-1).expand(-1, -1, d)),
            torch.gather(tokens, 1, masked_idx.unsqueeze(-1).expand(-1, -1, d)),
        )


class XJEPA(nn.Module):
    """Decoupled Cross-Modal Joint Embedding Predictive Architecture (X-JEPA).

    Processes multi-modal Earth Observation data (e.g., Sentinel-1 SAR and Sentinel-2 Optical)
    using modality stems, a shared Transformer encoder trunk, cross-modal cross-attention
    prediction heads, and Decoupled VICReg + PSA objectives.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        predictor_depth: int = 12,
        predictor_heads: int = 12,
        predictor_embed_dim: int = 384,
        num_shared_queries: int = 16,
        num_patches: int = 196,
        patch_size: int = 16,
        sar_channels: int = 2,
        optical_channels: int = 10,
        context_mask_ratio: float = 0.5,
        target_mask_ratio: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.context_mask_ratio = context_mask_ratio
        self.target_mask_ratio = target_mask_ratio

        # Modality stems
        self.sar_stem = ModalityStem(
            in_channels=sar_channels,
            embed_dim=embed_dim,
            num_patches=num_patches,
            patch_size=patch_size,
        )
        self.optical_stem = ModalityStem(
            in_channels=optical_channels,
            embed_dim=embed_dim,
            num_patches=num_patches,
            patch_size=patch_size,
        )

        # Shared Online Encoder
        self.online_encoder = build_encoder_trunk(
            depth=depth, embed_dim=embed_dim, num_heads=num_heads
        )

        # Learnable Shared Cross-Modal Queries
        self.shared_queries = nn.Parameter(
            torch.randn(1, num_shared_queries, predictor_embed_dim) * 0.02
        )

        # Cross-Modal Predictor
        self.predictor = XJEPAPredictor(
            in_dim=embed_dim,
            pred_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            num_patches=num_patches,
        )

        # PSA Loss Function
        self.psa_loss_fn = XJEPAPSALoss(embed_dim=embed_dim)

        # Representation Pooler
        self.mean_pool = MeanPooling(embed_dim)

    def forward(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
        sar_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        optical_masks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = sar.size(0)
        device = sar.device

        # Mask Generation (Disjoint context visible and target masked)
        if sar_masks is None:
            sar_context_visible_idx, sar_target_masked_idx = generate_disjoint_masks(
                batch_size,
                device,
                num_patches=self.num_patches,
                context_ratio=self.context_mask_ratio,
                target_ratio=self.target_mask_ratio,
            )
        else:
            sar_context_visible_idx, sar_target_masked_idx = sar_masks

        if optical_masks is None:
            optical_context_visible_idx, optical_target_masked_idx = generate_disjoint_masks(
                batch_size,
                device,
                num_patches=self.num_patches,
                context_ratio=self.context_mask_ratio,
                target_ratio=self.target_mask_ratio,
            )
        else:
            optical_context_visible_idx, optical_target_masked_idx = optical_masks

        # Modality Stems
        sar_tokens = self.sar_stem(sar)
        optical_tokens = self.optical_stem(optical)

        # Split Visible vs Masked
        sar_visible, _ = split_visible_masked(
            sar_tokens, sar_context_visible_idx, sar_target_masked_idx
        )
        optical_visible, _ = split_visible_masked(
            optical_tokens, optical_context_visible_idx, optical_target_masked_idx
        )

        # Encode Visible Context
        sar_context = self.online_encoder(sar_visible)
        optical_context = self.online_encoder(optical_visible)

        # Target Encoder Pass (No Gradient)
        with torch.no_grad():
            sar_target_full = self.online_encoder(sar_tokens)
            opt_target_full = self.online_encoder(optical_tokens)
            _, sar_target = split_visible_masked(
                sar_target_full, sar_context_visible_idx, sar_target_masked_idx
            )
            _, opt_target = split_visible_masked(
                opt_target_full, optical_context_visible_idx, optical_target_masked_idx
            )
            sar_target = sar_target.detach()
            opt_target = opt_target.detach()

        # Cross-Modal Prediction
        sar_to_opt_pred = self.predictor(
            context_tokens=sar_context,
            opposite_context=optical_context,
            masked_indices=optical_target_masked_idx,
            shared_queries=self.shared_queries,
            target_modality="optical",
        )
        opt_to_sar_pred = self.predictor(
            context_tokens=optical_context,
            opposite_context=sar_context,
            masked_indices=sar_target_masked_idx,
            shared_queries=self.shared_queries,
            target_modality="sar",
        )

        return {
            "sar_context": sar_context,
            "optical_context": optical_context,
            "sar_target": sar_target,
            "optical_target": opt_target,
            "sar_to_opt_pred": sar_to_opt_pred,
            "opt_to_sar_pred": opt_to_sar_pred,
            "sar_context_visible_idx": sar_context_visible_idx,
            "sar_target_masked_idx": sar_target_masked_idx,
            "optical_context_visible_idx": optical_context_visible_idx,
            "optical_target_masked_idx": optical_target_masked_idx,
        }
