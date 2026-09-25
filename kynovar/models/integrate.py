"""Semi-implicit Euler on torch tensors. Matches the simulator update."""

from __future__ import annotations

import torch


def semi_implicit_euler(
    position: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """v <- v + a dt; x <- x + v dt. `dt` is a scalar or shape (batch,)."""
    step = dt.to(dtype=position.dtype, device=position.device)
    while step.ndim < position.ndim:
        step = step.unsqueeze(-1)
    new_velocity = velocity + acceleration * step
    new_position = position + new_velocity * step
    return new_position, new_velocity
