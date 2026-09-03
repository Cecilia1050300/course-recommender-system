"""Small user-level gate for UAF-V1."""

from __future__ import annotations

import torch
from torch import nn


class UserAdaptiveGate(nn.Module):
    """alpha_u = sigmoid(Linear(ReLU(Linear(x_u, 16)), 1))."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
