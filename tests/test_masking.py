import torch
from dxjepa.data.masking import generate_disjoint_masks, split_visible_masked


def test_disjoint_masks():
    batch_size = 8
    num_patches = 196
    ctx_idx, tgt_idx = generate_disjoint_masks(
        batch_size=batch_size,
        device=torch.device("cpu"),
        num_patches=num_patches,
        context_ratio=0.5,
        target_ratio=0.5,
    )

    assert ctx_idx.shape == (batch_size, 98)
    assert tgt_idx.shape == (batch_size, 98)

    for i in range(batch_size):
        ctx_set = set(ctx_idx[i].tolist())
        tgt_set = set(tgt_idx[i].tolist())
        assert len(ctx_set) == 98
        assert len(tgt_set) == 98
        assert len(ctx_set & tgt_set) == 0, "Context and target masks must be strictly disjoint!"


def test_split_visible_masked():
    batch_size = 2
    num_patches = 16
    embed_dim = 32
    tokens = torch.randn(batch_size, num_patches, embed_dim)
    ctx_idx, tgt_idx = generate_disjoint_masks(
        batch_size=batch_size,
        device=torch.device("cpu"),
        num_patches=num_patches,
        context_ratio=0.5,
        target_ratio=0.5,
    )
    vis, tgt = split_visible_masked(tokens, ctx_idx, tgt_idx)
    assert vis.shape == (batch_size, 8, embed_dim)
    assert tgt.shape == (batch_size, 8, embed_dim)
