"""Fit and train dynamics models. Checkpoints keep the best validation position RMSE."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kynovar.data.dataset import TrajectoryStore, WindowDataset, collate_windows
from kynovar.data.normalize import Normalizer
from kynovar.evaluation.predict import evaluate_horizons, prediction_loss
from kynovar.models.factory import build_model, trainable_parameter_count
from kynovar.models.features import design_matrix
from kynovar.utils.device import select_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    model_name: str,
    store: TrajectoryStore,
    train_ids: list[str],
    validation_ids: list[str],
    normalizer: Normalizer,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device_name: str,
    sequence_length: int,
    stride: int,
    rollout_steps: int,
    hidden_dim: int,
    gnn_layers: int,
    patience: int,
    loss_weights: dict[str, float],
    checkpoint_path: Path,
    num_workers: int = 0,
) -> dict[str, object]:
    set_seed(seed)
    device = select_device(device_name)
    model = build_model(model_name, normalizer, hidden_dim, gnn_layers, sequence_length).to(device)
    horizon = max(rollout_steps, 1)
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    if model_name == "linear":
        loader = _loader(store, train_ids, model.history, horizon, stride, batch_size, seed, shuffle=False, workers=num_workers)
        fit_linear(model, loader, weight_decay, device)
        validation = evaluate_horizons(model, store, validation_ids, [1], stride, device)
        record = _epoch_record(0, 0.0, float(validation["1"]["position"]["rmse"]), 0.0, time.perf_counter() - started)
        history.append(record)
        best_rmse = record["val_position_rmse"]
        best_epoch = 0
    elif model_name == "constant_velocity":
        validation = evaluate_horizons(model, store, validation_ids, [1], stride, device)
        record = _epoch_record(0, 0.0, float(validation["1"]["position"]["rmse"]), 0.0, time.perf_counter() - started)
        history.append(record)
        best_rmse = record["val_position_rmse"]
        best_epoch = 0
    else:
        train_loader = _loader(
            store, train_ids, model.history, horizon, stride, batch_size, seed, shuffle=True, workers=num_workers
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_rmse = float("inf")
        best_epoch = -1
        stale = 0
        best_state = None
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            model.train()
            losses = []
            for states, mask in train_loader:
                states = states.to(device)
                mask = mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = prediction_loss(model, states, mask, _dt(store, device), loss_weights)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{model_name} produced a non-finite loss at epoch {epoch}.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation = evaluate_horizons(model, store, validation_ids, [1], stride, device)
            val_rmse = float(validation["1"]["position"]["rmse"])
            elapsed = time.perf_counter() - epoch_started
            train_loss = float(np.mean(losses)) if losses else float("nan")
            history.append(_epoch_record(epoch, train_loss, val_rmse, elapsed, time.perf_counter() - started))
            print(
                f"model={model_name} epoch={epoch} train_loss={train_loss:.6g} "
                f"val_position_rmse={val_rmse:.6g} seconds={elapsed:.3f}",
                flush=True,
            )
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_epoch = epoch
                stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError(f"{model_name} did not produce a checkpoint.")
        model.load_state_dict(best_state)
    payload = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "history_length": int(model.history),
        "hidden_dim": hidden_dim,
        "gnn_layers": gnn_layers,
        "sequence_length": sequence_length,
        "best_epoch": best_epoch,
        "best_val_position_rmse": best_rmse,
        "trainable_parameters": trainable_parameter_count(model),
        "device": str(device),
        "epochs_ran": len(history),
        "train_seconds": time.perf_counter() - started,
        "epoch_history": history,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return payload


def load_trained_model(
    checkpoint_path: Path,
    normalizer: Normalizer,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(
        payload["model_name"],
        normalizer,
        int(payload["hidden_dim"]),
        int(payload["gnn_layers"]),
        int(payload["sequence_length"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def fit_linear(model, loader: DataLoader, weight_decay: float, device: torch.device) -> None:
    """Ridge regression in normalized acceleration space, accumulated from disk batches."""
    width = model.linear.in_features
    gram = torch.zeros(width + 1, width + 1, dtype=torch.float64)
    rhs = torch.zeros(width + 1, 2, dtype=torch.float64)
    for states, mask in loader:
        states = states.to(device)
        mask = mask.to(device)
        frame = states[:, model.history - 1]
        features = design_matrix(frame, mask, model.normalizer)
        target = model.normalizer.encode_accel(frame[..., 4:6])
        chosen_features = features[mask].detach().cpu().double()
        chosen_target = target[mask].detach().cpu().double()
        if chosen_features.shape[0] == 0:
            continue
        # Drop ejected targets so one stiff step does not own the normal equations.
        keep = chosen_target.abs().amax(dim=-1) <= 8.0
        if int(keep.sum()) == 0:
            continue
        chosen_features = chosen_features[keep]
        chosen_target = chosen_target[keep]
        ones = torch.ones(chosen_features.shape[0], 1, dtype=torch.float64)
        design = torch.cat([chosen_features, ones], dim=1)
        gram += design.T @ design
        rhs += design.T @ chosen_target
    regularizer = np.eye(width + 1, dtype=np.float64) * weight_decay
    regularizer[-1, -1] = 0.0
    solution = np.linalg.solve(gram.numpy() + regularizer + np.eye(width + 1) * 1e-6, rhs.numpy())
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor(solution[:-1].T, dtype=torch.float32, device=device))
        model.linear.bias.copy_(torch.tensor(solution[-1], dtype=torch.float32, device=device))


def _loader(store, universe_ids, sequence_length, horizon, stride, batch_size, seed, shuffle, workers):
    dataset = WindowDataset(store, universe_ids, sequence_length, horizon, stride)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_windows,
        generator=generator,
    )


def _dt(store: TrajectoryStore, device: torch.device) -> torch.Tensor:
    return torch.tensor(store.dt, dtype=torch.float32, device=device)


def _epoch_record(epoch: int, train_loss: float, val_rmse: float, epoch_seconds: float, elapsed: float) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_position_rmse": val_rmse,
        "epoch_seconds": epoch_seconds,
        "elapsed_seconds": elapsed,
    }
