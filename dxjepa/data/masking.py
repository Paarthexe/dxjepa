from typing import Tuple
import torch


def generate_disjoint_masks(
    batch_size: int,
    device: torch.device,
    num_patches: int = 196,
    context_ratio: float = 0.5,
    target_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate disjoint context visible indices and target masked indices for each sample in batch."""
    num_context = int(num_patches * context_ratio)
    num_target = int(num_patches * target_ratio)
    context_visible_indices = []
    target_masked_indices = []

    for _ in range(batch_size):
        perm = torch.randperm(num_patches, device=device)
        ctx_idx = perm[:num_context]
        tgt_idx = perm[num_context : num_context + num_target]
        context_visible_indices.append(ctx_idx)
        target_masked_indices.append(tgt_idx)

    return (torch.stack(context_visible_indices), torch.stack(target_masked_indices))


def split_visible_masked(
    tokens: torch.Tensor,
    visible_idx: torch.Tensor,
    masked_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather visible context tokens and masked target tokens based on patch index tensors."""
    _, _, d = tokens.shape
    device = tokens.device
    visible_idx = visible_idx.to(device)
    masked_idx = masked_idx.to(device)

    visible_tokens = torch.gather(
        tokens, 1, visible_idx.unsqueeze(-1).expand(-1, -1, d)
    )
    target_tokens = torch.gather(
        tokens, 1, masked_idx.unsqueeze(-1).expand(-1, -1, d)
    )
    return visible_tokens, target_tokens
