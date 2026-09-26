#!/usr/bin/env python3
"""Measure why a trained GNN does or does not move away from constant velocity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from kynovar.data.config import load_run_config  # noqa: E402
from kynovar.data.dataset import WindowDataset, collate_windows  # noqa: E402
from kynovar.evaluation.benchmark import load_prepared  # noqa: E402
from kynovar.evaluation.predict import prediction_loss  # noqa: E402
from kynovar.models.train import load_trained_model  # noqa: E402


def _norm_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default="gnn")
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args(argv)
    config = load_run_config(args.config)
    store, splits, normalizer = load_prepared(config)
    device = torch.device("cpu")
    model, payload = load_trained_model(ROOT / "checkpoints" / config.name / f"{args.model}.pt", normalizer, device)
    dataset = WindowDataset(store, splits["test"], int(model.history), 1, stride=3)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_windows)
    captured: dict[str, list[float]] = {}

    def hook(name):
        def record(_module, _inputs, output):
            captured.setdefault(name, []).append(float(output.detach().norm(dim=-1).mean()))
        return record

    handles = []
    for name, module in model.named_children():
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.Linear)):
            if isinstance(module, torch.nn.ModuleList):
                for index, child in enumerate(module):
                    handles.append(child.register_forward_hook(hook(f"{name}.{index}")))
            else:
                handles.append(module.register_forward_hook(hook(name)))
    predicted, actual = [], []
    gradient_norms: dict[str, list[float]] = {}
    model.train()
    weights = dict(config.loss_weights)
    weights["rollout"] = 0.0
    for batch_index, (states, mask) in enumerate(loader):
        if batch_index >= args.batches:
            break
        model.zero_grad(set_to_none=True)
        loss = prediction_loss(model, states, mask, torch.tensor(store.dt), weights)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                top = name.split(".")[0]
                gradient_norms.setdefault(top, []).append(float(parameter.grad.norm()))
        with torch.no_grad():
            acc = model.acceleration(states[:, : model.history], mask)
        predicted.append(acc[mask].numpy())
        actual.append(states[:, model.history - 1][mask][:, 4:6].numpy())
    for handle in handles:
        handle.remove()
    predicted_np = np.concatenate(predicted)
    actual_np = np.concatenate(actual)
    pred_norm = np.linalg.norm(predicted_np, axis=-1)
    true_norm = np.linalg.norm(actual_np, axis=-1)
    correlation = float(np.corrcoef(predicted_np.ravel(), actual_np.ravel())[0, 1])
    residual = predicted_np - actual_np
    report = {
        "config": config.name,
        "model": args.model,
        "trainable_parameters": int(payload["trainable_parameters"]),
        "samples": int(pred_norm.size),
        "predicted_acceleration_norm": _norm_stats(pred_norm),
        "actual_acceleration_norm": _norm_stats(true_norm),
        "prediction_to_truth_norm_ratio_median": float(np.median(pred_norm / np.maximum(true_norm, 1e-9))),
        "component_correlation": correlation,
        "acceleration_rmse": float(np.sqrt(np.mean(residual**2))),
        "zero_prediction_rmse": float(np.sqrt(np.mean(actual_np**2))),
        "gradient_norm_mean_by_module": {name: float(np.mean(values)) for name, values in gradient_norms.items()},
        "activation_norm_mean_by_module": {name: float(np.mean(values)) for name, values in captured.items()},
        "normalizer_accel_scale": normalizer.accel_std.tolist(),
        "normalizer_position_scale": normalizer.node_std[:2].tolist(),
    }
    out = ROOT / "results" / "diagnostics" / "stable-v1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"gnn_{config.name}_{args.model}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    figure, axis = plt.subplots(figsize=(5, 5))
    limit = float(np.percentile(np.abs(actual_np), 99)) or 1.0
    axis.scatter(actual_np[:, 0], predicted_np[:, 0], s=3, alpha=0.4, label="ax")
    axis.scatter(actual_np[:, 1], predicted_np[:, 1], s=3, alpha=0.4, label="ay")
    axis.plot([-limit, limit], [-limit, limit], color="black", linewidth=0.8)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel("actual acceleration component")
    axis.set_ylabel("predicted acceleration component")
    axis.set_title(f"{config.name} {args.model} (test)")
    axis.legend()
    figure.tight_layout()
    plot_dir = ROOT / "results" / "plots" / "stable-v1"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_dir / f"acceleration_scatter_{config.name}_{args.model}.png", dpi=140)
    plt.close(figure)
    print(json.dumps(report, indent=2))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
