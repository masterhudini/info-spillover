from typing import Optional

import torch
import torch.nn as nn


class StaticGNNClassifier(nn.Module):
    """MLP encoder + static-weight GNN + graph-level classifier."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        gnn_dim: int,
        num_classes: int,
        adjacency: torch.Tensor,
    ):
        super().__init__()
        if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
            raise ValueError("Adjacency must be square (N x N)")

        self.register_buffer("adjacency", adjacency)

        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
        )
        self.gnn_linear = nn.Linear(hidden_dim, gnn_dim, bias=False)
        self.self_linear = nn.Linear(hidden_dim, gnn_dim)
        self.classifier = nn.Linear(gnn_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F)
        if x.dim() != 3:
            raise ValueError("Input must be (batch, nodes, features)")

        h = self.encoder(x)  # (B, N, H)
        msg = torch.matmul(self.adjacency, self.gnn_linear(h))  # (B, N, G)
        out = torch.relu(self.self_linear(h) + msg)
        graph_emb = out.mean(dim=1)
        return self.classifier(graph_emb)


class StaticGNNMultiTaskClassifier(nn.Module):
    """Shared encoder + static-weight GNN + per-task classifier heads."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        gnn_dim: int,
        num_classes: int,
        num_tasks: int,
        adjacency: torch.Tensor,
    ):
        super().__init__()
        if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
            raise ValueError("Adjacency must be square (N x N)")
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")

        self.register_buffer("adjacency", adjacency)

        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
        )
        self.gnn_linear = nn.Linear(hidden_dim, gnn_dim, bias=False)
        self.self_linear = nn.Linear(hidden_dim, gnn_dim)
        self.heads = nn.ModuleList([nn.Linear(gnn_dim, num_classes) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F)
        if x.dim() != 3:
            raise ValueError("Input must be (batch, nodes, features)")

        h = self.encoder(x)  # (B, N, H)
        msg = torch.matmul(self.adjacency, self.gnn_linear(h))  # (B, N, G)
        out = torch.relu(self.self_linear(h) + msg)
        graph_emb = out.mean(dim=1)
        logits = [head(graph_emb) for head in self.heads]
        return torch.stack(logits, dim=1)  # (B, T, C)
