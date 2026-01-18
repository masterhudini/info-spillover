import numpy as np
import torch

from src.modules.models.static_gnn import StaticGNNClassifier


def test_static_gnn_shapes():
    adj = torch.eye(3)
    model = StaticGNNClassifier(
        num_features=4,
        hidden_dim=8,
        gnn_dim=8,
        num_classes=3,
        adjacency=adj,
    )
    x = torch.randn(2, 3, 4)
    out = model(x)
    assert out.shape == (2, 3)


def test_static_gnn_rejects_bad_adjacency():
    try:
        StaticGNNClassifier(
            num_features=4,
            hidden_dim=8,
            gnn_dim=8,
            num_classes=3,
            adjacency=torch.randn(2, 3),
        )
    except ValueError as exc:
        assert "square" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-square adjacency")
