import torch
from dxjepa.evaluation.retrieval import compute_cross_retrieval_matrix, evaluate_retrieval_f1
from dxjepa.data.builder import SyntheticBigEarthNetDataset
from dxjepa.models.xjepa import XJEPA
from torch.utils.data import DataLoader


def test_cross_retrieval_matrix():
    a = torch.randn(10, 64)
    b = torch.randn(10, 64)
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    sim = compute_cross_retrieval_matrix(a, b)
    assert sim.shape == (10, 10)


def test_evaluate_retrieval_f1():
    model = XJEPA(
        embed_dim=64,
        depth=1,
        num_heads=2,
        predictor_depth=1,
        predictor_heads=2,
        predictor_embed_dim=32,
        num_shared_queries=4,
    )
    dataset = SyntheticBigEarthNetDataset(num_samples=32, seed=123)
    loader = DataLoader(dataset, batch_size=16)

    metrics = evaluate_retrieval_f1(model, loader, device=torch.device("cpu"), k_list=[2, 5], d_common=32)
    assert "S1->S2_F1@2" in metrics
    assert "S1->S2_F1@5" in metrics
    assert "S2->S1_F1@2" in metrics
    assert "S1->S1_F1@2" in metrics
    assert "S2->S2_F1@2" in metrics
    for k, v in metrics.items():
        assert 0.0 <= v <= 100.0
