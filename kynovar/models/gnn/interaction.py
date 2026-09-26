"""Interaction GNN with physically structured edge outputs and in-context law inference.

Each directed edge emits a scalar g_ij. The pairwise part of the acceleration is
sum_j g_ij * u_ij, where u_ij is the unit vector from i to j. A node head adds a
per-body term for forces that are not pairwise (drag, uniform fields).

Hidden coefficients differ between universes, and one frame cannot identify
them. With `history > 1` a context encoder reads past observed frames of the
same experiment: past accelerations are finite differences of observed
velocities, v(s+1) - v(s) = a(s) dt under the simulator's update. The encoder
pools those measurements against the observed geometry into a latent z that
conditions every message. Nothing hidden is supplied.
"""

from __future__ import annotations

import torch
from torch import nn

from kynovar.data.normalize import Normalizer
from kynovar.models.features import pair_mask

EDGE_WIDTH = 10
MIN_DISTANCE = 0.05


def _mlp(width_in: int, hidden: int, width_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, width_out))


def interaction_edges(frame: torch.Tensor, normalizer: Normalizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Relative features for every directed pair. Returns (features, unit vectors)."""
    position = frame[..., 0:2]
    velocity = frame[..., 2:4]
    mass = frame[..., 6].clamp_min(1e-6)
    delta = position[:, None, :, :] - position[:, :, None, :]
    distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(MIN_DISTANCE)
    unit = delta / distance
    relative = velocity[:, None, :, :] - velocity[:, :, None, :]
    speed_scale = normalizer.node_std[2:4].mean().clamp_min(1e-6)
    radial = (relative * unit).sum(dim=-1, keepdim=True) / speed_scale
    log_mass = torch.log(mass)
    bodies = frame.shape[1]
    features = torch.cat(
        [
            unit,
            torch.log(distance),
            (1.0 / distance).clamp_max(1.0 / MIN_DISTANCE) * MIN_DISTANCE * 4.0,
            relative / speed_scale,
            radial,
            log_mass[:, :, None, None].expand(-1, bodies, bodies, 1),
            log_mass[:, None, :, None].expand(-1, bodies, bodies, 1),
            torch.log(distance).square() * 0.25,
        ],
        dim=-1,
    )
    return features, unit


class InteractionGNN(nn.Module):
    def __init__(self, normalizer: Normalizer, hidden_dim: int, layers: int = 2, history: int = 4) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("GNN layers must be at least 1.")
        if history < 1:
            raise ValueError("history must be at least 1.")
        self.history = history
        self.normalizer = normalizer
        self.hidden_dim = hidden_dim
        self.use_context = history > 1
        self.edge_encoder = _mlp(EDGE_WIDTH, hidden_dim, hidden_dim)
        self.node_encoder = _mlp(4, hidden_dim, hidden_dim)
        if self.use_context:
            self.context_edge = _mlp(EDGE_WIDTH + 3, hidden_dim, hidden_dim)
            self.context_out = _mlp(hidden_dim, hidden_dim, hidden_dim)
        context_width = hidden_dim if self.use_context else 0
        self.messages = nn.ModuleList([_mlp(hidden_dim * 3 + context_width, hidden_dim, hidden_dim) for _ in range(layers)])
        self.updates = nn.ModuleList([_mlp(hidden_dim * 2, hidden_dim, hidden_dim) for _ in range(layers)])
        self.edge_head = _mlp(hidden_dim * 3 + context_width, hidden_dim, 2)
        self.node_head = _mlp(hidden_dim + context_width, hidden_dim, 2)
        self.last_diagnostics: dict[str, float] = {}

    def _nodes(self, frame: torch.Tensor) -> torch.Tensor:
        velocity = frame[..., 2:4] / self.normalizer.node_std[2:4].clamp_min(1e-6)
        speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
        return torch.cat([velocity, speed, torch.log(frame[..., 6:7].clamp_min(1e-6))], dim=-1)

    def context(self, states: torch.Tensor, mask: torch.Tensor, dt: float | None = None) -> torch.Tensor | None:
        if not self.use_context:
            return None
        window = states[:, -self.history :]
        steps = window.shape[1] - 1
        scale = self.normalizer.accel_std.mean().clamp_min(1e-6)
        valid = pair_mask(mask).to(window.dtype)
        pooled = window.new_zeros(window.shape[0], self.hidden_dim)
        count = valid.sum(dim=(1, 2)).clamp_min(1.0) * steps
        for step in range(steps):
            frame = window[:, step]
            if dt is None:
                # Recorded acceleration at a past frame is itself an observation of that frame.
                observed = frame[..., 4:6]
            else:
                observed = (window[:, step + 1, :, 2:4] - frame[..., 2:4]) / dt
            features, unit = interaction_edges(frame, self.normalizer)
            along = (observed[:, :, None, :] * unit).sum(dim=-1, keepdim=True) / scale
            across = (observed[:, :, None, 0:1] * unit[..., 1:2] - observed[:, :, None, 1:2] * unit[..., 0:1]) / scale
            magnitude = torch.log1p(torch.linalg.norm(observed, dim=-1) / scale)[:, :, None, None].expand_as(along)
            token = self.context_edge(torch.cat([features, torch.asinh(along), torch.asinh(across), magnitude], dim=-1))
            pooled = pooled + (token * valid[..., None]).sum(dim=(1, 2))
        return self.context_out(pooled / count[:, None])

    def acceleration(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        frame = states[:, -1]
        edges, unit = interaction_edges(frame, self.normalizer)
        edge_hidden = self.edge_encoder(edges)
        hidden = self.node_encoder(self._nodes(frame))
        z = self.context(states, mask)
        valid = pair_mask(mask).to(hidden.dtype)
        bodies = hidden.shape[1]
        extra = [] if z is None else [z[:, None, None, :].expand(-1, bodies, bodies, -1)]
        aggregate_norm = 0.0
        message_norm = 0.0
        for message_fn, update_fn in zip(self.messages, self.updates, strict=True):
            source = hidden[:, :, None, :].expand(-1, -1, bodies, -1)
            target = hidden[:, None, :, :].expand(-1, bodies, -1, -1)
            incoming = message_fn(torch.cat([source, target, edge_hidden, *extra], dim=-1)) * valid[..., None]
            aggregated = incoming.sum(dim=2)
            hidden = hidden + update_fn(torch.cat([hidden, aggregated], dim=-1))
            hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
            message_norm = float(incoming.detach().norm(dim=-1).mean())
            aggregate_norm = float(aggregated.detach().norm(dim=-1).mean())
        source = hidden[:, :, None, :].expand(-1, -1, bodies, -1)
        target = hidden[:, None, :, :].expand(-1, bodies, -1, -1)
        raw = self.edge_head(torch.cat([source, target, edge_hidden, *extra], dim=-1))
        magnitude = torch.exp(raw[..., 0].clamp(-12.0, 8.0)) * torch.tanh(raw[..., 1])
        scale = self.normalizer.accel_std.mean().clamp_min(1e-6)
        pairwise = (magnitude[..., None] * unit * valid[..., None]).sum(dim=2) * scale
        node_extra = [] if z is None else [z[:, None, :].expand(-1, bodies, -1)]
        local = self.node_head(torch.cat([hidden, *node_extra], dim=-1)) * self.normalizer.accel_std
        predicted = pairwise + local
        self.last_diagnostics = {
            "node_embedding_norm": float(hidden.detach().norm(dim=-1).mean()),
            "edge_embedding_norm": float(edge_hidden.detach().norm(dim=-1).mean()),
            "message_norm": message_norm,
            "aggregate_norm": aggregate_norm,
            "context_norm": 0.0 if z is None else float(z.detach().norm(dim=-1).mean()),
        }
        return predicted * mask.unsqueeze(-1).to(predicted.dtype)
