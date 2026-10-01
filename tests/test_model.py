import pytest
import torch
from dxjepa.models.stem import PatchEmbedding, ModalityStem
from dxjepa.models.transformer import MLP, TransformerBlock, SharedTransformer, build_encoder_trunk
from dxjepa.models.predictor import XJEPAPredictorBlock, XJEPAPredictor
from dxjepa.models.pooling import MeanPooling, compute_latent_embeddings
from dxjepa.models.xjepa import XJEPA


def test_patch_embedding():
    patch_embed = PatchEmbedding(in_channels=2, embed_dim=768, patch_size=16)
    x = torch.randn(2, 2, 224, 224)
    out = patch_embed(x)
    assert out.shape == (2, 196, 768)


def test_modality_stem():
    sar_stem = ModalityStem(in_channels=2, embed_dim=768, num_patches=196)
    optical_stem = ModalityStem(in_channels=10, embed_dim=768, num_patches=196)
    sar = torch.randn(2, 2, 224, 224)
    optical = torch.randn(2, 10, 224, 224)
    assert sar_stem(sar).shape == (2, 196, 768)
    assert optical_stem(optical).shape == (2, 196, 768)


def test_transformer_block():
    block = TransformerBlock(embed_dim=768, num_heads=12)
    x = torch.randn(2, 98, 768)
    out = block(x)
    assert out.shape == (2, 98, 768)


def test_shared_transformer():
    encoder = build_encoder_trunk(depth=2, embed_dim=768, num_heads=12)
    assert len(encoder.blocks) == 2
    x = torch.randn(2, 98, 768)
    out = encoder(x)
    assert out.shape == (2, 98, 768)


def test_predictor():
    predictor = XJEPAPredictor(
        in_dim=768, pred_dim=384, depth=2, num_heads=6, num_patches=196
    )
    context_tokens = torch.randn(2, 98, 768)
    opposite_context = torch.randn(2, 98, 768)
    masked_indices = torch.randint(0, 196, (2, 98))
    shared_queries = torch.randn(1, 16, 384)

    out = predictor(
        context_tokens, opposite_context, masked_indices, shared_queries, target_modality="optical"
    )
    assert out.shape == (2, 98, 768)


def test_mean_pooling():
    pool = MeanPooling(dim=768)
    x = torch.randn(4, 98, 768)
    pooled = pool(x)
    assert pooled.shape == (4, 768)
    normalized = compute_latent_embeddings(x)
    assert normalized.shape == (4, 768)
    norms = torch.norm(normalized, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_xjepa_full_forward():
    # Use smaller depth for quick unit test
    model = XJEPA(
        embed_dim=128,
        depth=2,
        num_heads=4,
        predictor_depth=2,
        predictor_heads=4,
        predictor_embed_dim=64,
        num_shared_queries=8,
        num_patches=196,
        patch_size=16,
        sar_channels=2,
        optical_channels=10,
    )
    sar = torch.randn(2, 2, 224, 224)
    optical = torch.randn(2, 10, 224, 224)

    outputs = model(sar, optical)
    assert "sar_context" in outputs
    assert "optical_context" in outputs
    assert "sar_target" in outputs
    assert "optical_target" in outputs
    assert "sar_to_opt_pred" in outputs
    assert "opt_to_sar_pred" in outputs

    assert outputs["sar_context"].shape == (2, 98, 128)
    assert outputs["optical_context"].shape == (2, 98, 128)
    assert outputs["sar_target"].shape == (2, 98, 128)
    assert outputs["optical_target"].shape == (2, 98, 128)
    assert outputs["sar_to_opt_pred"].shape == (2, 98, 128)
    assert outputs["opt_to_sar_pred"].shape == (2, 98, 128)


def test_xjepa_backward_pass():
    model = XJEPA(
        embed_dim=128,
        depth=1,
        num_heads=2,
        predictor_depth=1,
        predictor_heads=2,
        predictor_embed_dim=64,
        num_shared_queries=4,
    )
    sar = torch.randn(2, 2, 224, 224)
    optical = torch.randn(2, 10, 224, 224)
    outputs = model(sar, optical)

    loss = outputs["sar_to_opt_pred"].sum() + outputs["opt_to_sar_pred"].sum()
    loss.backward()

    # Check gradients on trainable components
    assert model.sar_stem.patch_embed.proj.weight.grad is not None
    assert model.optical_stem.patch_embed.proj.weight.grad is not None
    assert model.online_encoder.blocks[0].norm1.weight.grad is not None
    assert model.predictor.proj_in_context.weight.grad is not None
