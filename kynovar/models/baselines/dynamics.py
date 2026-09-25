"""Baselines that predict acceleration from observable state."""

from __future__ import annotations

import torch
from torch import nn

from kynovar.data.normalize import Normalizer
from kynovar.models.features import design_matrix


class ConstantVelocity(nn.Module):
    """Zero acceleration. Position advances with the current velocity."""

    def __init__(self) -> None:
        super().__init__()
        self.history = 1

    def acceleration(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, bodies = states.shape[0], states.shape[-2]
        return torch.zeros(batch, bodies, 2, device=states.device, dtype=states.dtype)


class LinearDynamics(nn.Module):
    def __init__(self, normalizer: Normalizer, in_features: int = 12) -> None:
        super().__init__()
        self.history = 1
        self.normalizer = normalizer
        self.linear = nn.Linear(in_features, 2)

    def acceleration(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = design_matrix(states[:, -1], mask, self.normalizer)
        predicted = self.normalizer.decode_accel(self.linear(features))
        return predicted * mask.unsqueeze(-1).to(dtype=predicted.dtype)


class MLPDynamics(nn.Module):
    def __init__(self, normalizer: Normalizer, hidden_dim: int, in_features: int = 12) -> None:
        super().__init__()
        self.history = 1
        self.normalizer = normalizer
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def acceleration(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = design_matrix(states[:, -1], mask, self.normalizer)
        predicted = self.normalizer.decode_accel(self.network(features))
        return predicted * mask.unsqueeze(-1).to(dtype=predicted.dtype)


class GRUDynamics(nn.Module):
    def __init__(self, normalizer: Normalizer, hidden_dim: int, history: int, in_features: int = 12) -> None:
        super().__init__()
        self.history = history
        self.normalizer = normalizer
        self.gru = nn.GRU(in_features, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2)

    def acceleration(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if states.shape[1] < self.history:
            raise ValueError(f"GRU expected at least {self.history} frames, got {states.shape[1]}.")
        window = states[:, -self.history :]
        frames = [design_matrix(window[:, time], mask, self.normalizer) for time in range(window.shape[1])]
        sequence = torch.stack(frames, dim=2)
        batch, bodies, length, features = sequence.shape
        flat = sequence.reshape(batch * bodies, length, features)
        output, _hidden = self.gru(flat)
        normalized = self.head(output[:, -1]).reshape(batch, bodies, 2)
        predicted = self.normalizer.decode_accel(normalized)
        return predicted * mask.unsqueeze(-1).to(dtype=predicted.dtype)
