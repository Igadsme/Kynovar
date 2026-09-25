"""Observable node and edge features. Acceleration is a target, not an input."""

from __future__ import annotations

import torch

from kynovar.data.normalize import Normalizer


def edge_raw(position: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Directed pairwise features. Entry [i, j] is the vector from i to j."""
    delta = position[:, None, :, :] - position[:, :, None, :]
    distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
    relative_velocity = velocity[:, None, :, :] - velocity[:, :, None, :]
    return torch.cat([delta, distance, relative_velocity], dim=-1)


def pair_mask(mask: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(mask.shape[1], device=mask.device, dtype=torch.bool)
    return mask[:, :, None] & mask[:, None, :] & ~eye


def neighbor_summary(position: torch.Tensor, velocity: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edges = edge_raw(position, velocity)
    valid = pair_mask(mask).to(dtype=edges.dtype)
    denom = valid.sum(dim=2).clamp_min(1.0)
    summary = (edges * valid[..., None]).sum(dim=2) / denom[..., None]
    count = valid.sum(dim=2, keepdim=True)
    return summary, count


def design_matrix(frame: torch.Tensor, mask: torch.Tensor, normalizer: Normalizer) -> torch.Tensor:
    """Per-body features: normalized kinematics plus the mean relative edge."""
    node = torch.cat([frame[..., 0:4], frame[..., 6:8]], dim=-1)
    encoded_node = normalizer.encode_nodes(node)
    summary, count = neighbor_summary(frame[..., 0:2], frame[..., 2:4], mask)
    encoded_summary = normalizer.encode_edges(summary)
    scaled_count = count / 4.0
    features = torch.cat([encoded_node, encoded_summary, scaled_count], dim=-1)
    return features * mask.unsqueeze(-1).to(dtype=features.dtype)
