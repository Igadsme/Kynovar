"""Autoregressive rollouts and horizon metrics."""

from __future__ import annotations

import numpy as np
import torch

from kynovar.data.dataset import TrajectoryStore
from kynovar.evaluation.metrics import RunningScore
from kynovar.models.integrate import semi_implicit_euler


def prediction_loss(
    model: torch.nn.Module,
    states: torch.Tensor,
    mask: torch.Tensor,
    dt: torch.Tensor,
    weights: dict[str, float],
) -> torch.Tensor:
    """One-step loss, plus later position and velocity terms when rollout weight is positive."""
    history_length = int(model.history)
    steps = states.shape[1] - history_length
    if steps < 1:
        raise ValueError("The batch does not contain a future frame.")
    if weights["rollout"] == 0.0:
        steps = 1
    position = states[:, history_length - 1, :, 0:2]
    velocity = states[:, history_length - 1, :, 2:4]
    history = states[:, :history_length]
    total = states.new_zeros(())
    later_position = []
    later_velocity = []
    denom = mask.float().sum().clamp_min(1.0)
    for step in range(steps):
        predicted_acc = model.acceleration(history[:, -history_length:], mask)
        if step == 0:
            true_acc = states[:, history_length - 1, :, 4:6]
            total = total + weights["acceleration"] * _normalized_huber(
                predicted_acc, true_acc, model.normalizer.accel_std, mask, denom
            )
        position, velocity = semi_implicit_euler(position, velocity, predicted_acc, dt)
        truth = states[:, history_length + step]
        position_loss = _normalized_huber(position, truth[..., 0:2], model.normalizer.node_std[:2], mask, denom)
        velocity_loss = _normalized_huber(velocity, truth[..., 2:4], model.normalizer.node_std[2:4], mask, denom)
        if step == 0:
            total = total + weights["position"] * position_loss + weights["velocity"] * velocity_loss
        else:
            later_position.append(position_loss)
            later_velocity.append(velocity_loss)
        frame = history[:, -1].clone()
        frame[..., 0:2] = position
        frame[..., 2:4] = velocity
        frame[..., 4:6] = predicted_acc
        history = torch.cat([history, frame.unsqueeze(1)], dim=1)
    if later_position:
        extra = torch.stack(later_position).mean() + torch.stack(later_velocity).mean()
        total = total + weights["rollout"] * extra
    return total


@torch.no_grad()
def rollout_future(
    model: torch.nn.Module,
    history: torch.Tensor,
    mask: torch.Tensor,
    dt: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return future states `(batch, steps, bodies, 8)` and the accelerations that produced them."""
    position = history[:, -1, :, 0:2]
    velocity = history[:, -1, :, 2:4]
    window = history
    frames = []
    accelerations = []
    for _ in range(steps):
        predicted_acc = model.acceleration(window[:, -int(model.history) :], mask)
        position, velocity = semi_implicit_euler(position, velocity, predicted_acc, dt)
        frame = window[:, -1].clone()
        frame[..., 0:2] = position
        frame[..., 2:4] = velocity
        frame[..., 4:6] = predicted_acc
        frames.append(frame)
        accelerations.append(predicted_acc)
        window = torch.cat([window, frame.unsqueeze(1)], dim=1)
    return torch.stack(frames, dim=1), torch.stack(accelerations, dim=1)


def evaluate_horizons(
    model: torch.nn.Module,
    store: TrajectoryStore,
    universe_ids: list[str],
    horizons: list[int] | tuple[int, ...],
    stride: int,
    device: torch.device,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Endpoint error at each horizon on the selected universes."""
    model.eval()
    scores = {
        int(horizon): {
            "position": RunningScore(),
            "velocity": RunningScore(),
            "acceleration": RunningScore(),
            "position_bounded": RunningScore(),
            "velocity_bounded": RunningScore(),
            "acceleration_bounded": RunningScore(),
        }
        for horizon in horizons
    }
    dt = torch.tensor(store.dt, dtype=torch.float32, device=device)
    history_length = int(model.history)
    for _universe_id, _experiment, states_np in store.iter_split(universe_ids):
        states = torch.from_numpy(states_np).to(device=device, dtype=torch.float32)
        frames, bodies, _channels = states.shape
        body_mask = np.ones(bodies, dtype=bool)
        for horizon in scores:
            times = list(range(history_length - 1, frames - horizon, stride))
            if not times:
                continue
            window = torch.stack(
                [states[time - history_length + 1 : time + 1] for time in times],
                dim=0,
            )
            mask = torch.ones(len(times), bodies, dtype=torch.bool, device=device)
            future, accelerations = rollout_future(model, window, mask, dt, horizon)
            predicted_states = future[:, horizon - 1].detach().cpu().numpy()
            predicted_acc = accelerations[:, horizon - 1].detach().cpu().numpy()
            for index, time in enumerate(times):
                truth = states_np[time + horizon]
                true_acc = states_np[time + horizon - 1, :, 4:6]
                scores[horizon]["position"].update(predicted_states[index, :, 0:2], truth[:, 0:2], body_mask)
                scores[horizon]["velocity"].update(predicted_states[index, :, 2:4], truth[:, 2:4], body_mask)
                scores[horizon]["acceleration"].update(predicted_acc[index], true_acc, body_mask)
                inside = bounded_body_mask(truth[:, 0:2], truth[:, 2:4], true_acc)
                scores[horizon]["position_bounded"].update(predicted_states[index, :, 0:2], truth[:, 0:2], inside)
                scores[horizon]["velocity_bounded"].update(predicted_states[index, :, 2:4], truth[:, 2:4], inside)
                scores[horizon]["acceleration_bounded"].update(predicted_acc[index], true_acc, inside)
    summarized: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for horizon, channels in scores.items():
        summarized[str(horizon)] = {}
        for channel, score in channels.items():
            if score.count == 0:
                continue
            summarized[str(horizon)][channel] = score.as_dict()
    return summarized


# A target is "bounded" when the true state has not been ejected by a stiff force.
BOUNDED_POSITION = 8.0
BOUNDED_VELOCITY = 8.0
BOUNDED_ACCELERATION = 80.0


def bounded_body_mask(position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    """True for bodies whose true target is still in the bulk of the training box."""
    return (
        (np.linalg.norm(position, axis=-1) <= BOUNDED_POSITION)
        & (np.linalg.norm(velocity, axis=-1) <= BOUNDED_VELOCITY)
        & (np.linalg.norm(acceleration, axis=-1) <= BOUNDED_ACCELERATION)
    )


def _normalized_huber(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    scale: torch.Tensor,
    mask: torch.Tensor,
    denom: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss after dividing by the training-set scale. Large ejections stay finite."""
    safe = scale.to(device=predicted.device, dtype=predicted.dtype).clamp_min(1e-6)
    residual = (predicted - truth) / safe
    quadratic = 0.5 * residual.square()
    linear = delta * (residual.abs() - 0.5 * delta)
    robust = torch.where(residual.abs() <= delta, quadratic, linear)
    per_body = robust.sum(dim=-1)
    return (per_body * mask.float()).sum() / (denom * predicted.shape[-1])
