import torch
import torch.nn.functional as F
from dxjepa.losses.psa import XJEPAPSALoss
from dxjepa.losses.vicreg import (
    split_common_unique,
    cross_covariance_penalty,
    decoupled_vicreg_loss,
)
from dxjepa.losses.criterion import compute_loss
from dxjepa.models.xjepa import XJEPA


def test_split_common_unique():
    z = torch.randn(4, 768)
    common, unique = split_common_unique(z, d_common=512)
    assert common.shape == (4, 512)
    assert unique.shape == (4, 256)
    assert torch.equal(torch.cat([common, unique], dim=1), z)


def test_cross_covariance_penalty():
    a = torch.randn(16, 64)
    b = torch.randn(16, 64)
    penalty = cross_covariance_penalty(a, b)
    assert penalty.ndim == 0
    assert penalty.item() >= 0.0


def test_decoupled_vicreg_loss():
    z_a = torch.randn(16, 768)
    z_b = torch.randn(16, 768)
    total, sim, var, cov, excl = decoupled_vicreg_loss(
        z_a, z_b, d_common=512, lambda_inv=1.0, lambda_var=25.0, lambda_cov=25.0, lambda_excl=5.0
    )
    assert total.item() > 0.0
    assert sim.item() >= 0.0
    assert var.item() >= 0.0
    assert cov.item() >= 0.0
    assert excl.item() >= 0.0


def test_psa_loss():
    psa = XJEPAPSALoss(embed_dim=128)
    pred = torch.randn(4, 98, 128)
    target = torch.randn(4, 98, 128)
    loss = psa(pred, target)
    assert loss.ndim == 0
    assert loss.item() > 0.0

    # Test backward pass
    loss.backward()
    assert psa.M_weight.grad is not None


def test_compute_loss_e2e():
    model = XJEPA(
        embed_dim=64,
        depth=1,
        num_heads=2,
        predictor_depth=1,
        predictor_heads=2,
        predictor_embed_dim=32,
        num_shared_queries=4,
    )
    sar = torch.randn(4, 2, 224, 224)
    optical = torch.randn(4, 10, 224, 224)
    outputs = model(sar, optical)

    loss_dict = compute_loss(
        outputs,
        model,
        lambda_inv=1.0,
        lambda_var=25.0,
        lambda_cov=25.0,
        lambda_excl=5.0,
        lambda_psa=0.1,
        d_common=32,
    )

    for k in ["total_loss", "loss_l2", "loss_psa", "loss_pred", "loss_vicreg"]:
        assert k in loss_dict
        assert torch.isfinite(loss_dict[k])

    loss_dict["total_loss"].backward()
    assert model.online_encoder.blocks[0].norm1.weight.grad is not None
