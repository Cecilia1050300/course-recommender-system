"""Trainable fusion components for OAF-V2."""

from __future__ import annotations

import math

import torch
from torch import nn


class GlobalAlpha(nn.Module):
    def __init__(self, initial_alpha: float) -> None:
        super().__init__()
        logit = math.log(initial_alpha / (1.0 - initial_alpha))
        self.logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)


class UserCountAlpha(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def joint_losses(alpha_rating: torch.Tensor, alpha_pair: torch.Tensor,
                 mf_rating: torch.Tensor, lg_rating: torch.Tensor,
                 rating_targets: torch.Tensor,
                 mf_pair_difference: torch.Tensor, lg_pair_difference: torch.Tensor,
                 pair_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    rating_predictions = alpha_rating * mf_rating + (1.0 - alpha_rating) * lg_rating
    rating_loss = torch.mean(torch.square(rating_predictions - rating_targets))
    fused_difference = alpha_pair * mf_pair_difference + (1.0 - alpha_pair) * lg_pair_difference
    pair_losses = torch.nn.functional.softplus(-fused_difference)
    if pair_weights is None:
        ranking_loss = pair_losses.mean()
    else:
        ranking_loss = torch.sum(pair_weights * pair_losses) / torch.sum(pair_weights)
    return rating_loss, ranking_loss
