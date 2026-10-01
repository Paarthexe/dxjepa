import math
import torch
import torch.nn as nn
from dxjepa.models.transformer import MLP


class XJEPAPredictorBlock(nn.Module):
    """Predictor block consisting of self-attention, cross-attention with opposite context, and MLP."""

    def __init__(self, dim: int = 384, num_heads: int = 12):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, queries: torch.Tensor, opposite_context: torch.Tensor) -> torch.Tensor:
        device = self.norm1.weight.device
        queries = queries.to(device)
        opposite_context = opposite_context.to(device)

        # Self-attention over queries
        res = queries
        x = self.norm1(queries)
        x, _ = self.self_attn(x, x, x, need_weights=False)
        x = res + x

        # Cross-attention to condition on opposite modality's visible context
        res = x
        x = self.norm2(x)
        x, _ = self.cross_attn(x, opposite_context, opposite_context, need_weights=False)
        x = res + x

        # Feed-forward MLP
        res = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = res + x
        return x


class XJEPAPredictor(nn.Module):
    """Cross-Modal JEPA Predictor predicting masked patch representations from cross-modal context."""

    def __init__(
        self,
        in_dim: int = 768,
        pred_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        num_patches: int = 196,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.pred_dim = pred_dim
        self.num_patches = num_patches

        self.proj_in_context = nn.Linear(in_dim, pred_dim)
        self.proj_out = nn.Linear(pred_dim, in_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.pos_embed_sar = nn.Parameter(torch.zeros(1, num_patches, pred_dim))
        self.pos_embed_optical = nn.Parameter(torch.zeros(1, num_patches, pred_dim))
        self.blocks = nn.ModuleList(
            [XJEPAPredictorBlock(dim=pred_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(pred_dim)

        # Match exact notebook initialization
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_sar, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_optical, std=0.02)

    def forward(
        self,
        context_tokens: torch.Tensor,
        opposite_context: torch.Tensor,
        masked_indices: torch.Tensor,
        shared_queries: torch.Tensor,
        target_modality: str = "sar",
    ) -> torch.Tensor:
        B = context_tokens.size(0)
        device = context_tokens.device
        N_target = masked_indices.size(1)

        if masked_indices.size(0) != B:
            if masked_indices.size(0) > B:
                masked_indices = masked_indices[:B]
            else:
                reps = math.ceil(B / masked_indices.size(0))
                masked_indices = masked_indices.repeat(reps, 1)[:B]

        opposite_context = opposite_context.to(device)
        masked_indices = masked_indices.to(device)
        shared_queries = shared_queries.to(device)

        ctx_proj = self.proj_in_context(opposite_context)
        mask_tokens = self.mask_token.to(device).expand(B, N_target, self.pred_dim)
        pos_table = (
            self.pos_embed_sar if target_modality == "sar" else self.pos_embed_optical
        ).to(device)
        target_pos = torch.gather(
            pos_table.expand(B, self.num_patches, self.pred_dim),
            1,
            masked_indices.unsqueeze(-1).expand(-1, -1, self.pred_dim),
        )
        routed_mask_tokens = mask_tokens + target_pos

        s_expanded = shared_queries.expand(B, -1, -1)
        num_queries = s_expanded.size(1)
        queries = torch.cat([s_expanded, routed_mask_tokens], dim=1)

        x = queries
        for block in self.blocks:
            x = block(x, ctx_proj)
        x = self.norm(x)

        target_preds = x[:, num_queries:, :]
        predicted_targets = self.proj_out(target_preds)
        return predicted_targets
