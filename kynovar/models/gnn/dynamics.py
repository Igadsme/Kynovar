"""Message-passing dynamics model. Each body is a node. Acceleration is the output."""

from __future__ import annotations

import torch
from torch import nn

from kynovar.data.normalize import Normalizer
from kynovar.models.features import edge_raw, pair_mask


class DynamicsGNN(nn.Module):
    def __init__(self, normalizer: Normalizer, hidden_dim: int, layers: int = 2) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("GNN layers must be at least 1.")
        self.history = 1
        self.layers = layers
        self.normalizer = normalizer
        self.node_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.messages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, 2)

    def acceleration(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        frame = states[:, -1]
        node = torch.cat([frame[..., 0:4], frame[..., 6:8]], dim=-1)
        hidden = self.node_encoder(self.normalizer.encode_nodes(node))
        edges = self.edge_encoder(self.normalizer.encode_edges(edge_raw(frame[..., 0:2], frame[..., 2:4])))
        valid = pair_mask(mask).to(dtype=hidden.dtype)
        for message_fn, update_fn in zip(self.messages, self.updates, strict=True):
            source = hidden[:, :, None, :].expand(-1, -1, hidden.shape[1], -1)
            target = hidden[:, None, :, :].expand(-1, hidden.shape[1], -1, -1)
            incoming = message_fn(torch.cat([source, target, edges], dim=-1))
            incoming = incoming * valid[..., None]
            aggregated = incoming.sum(dim=2)
            hidden = update_fn(torch.cat([hidden, aggregated], dim=-1))
            hidden = hidden * mask.unsqueeze(-1).to(dtype=hidden.dtype)
        predicted = self.normalizer.decode_accel(self.head(hidden))
        return predicted * mask.unsqueeze(-1).to(dtype=predicted.dtype)
