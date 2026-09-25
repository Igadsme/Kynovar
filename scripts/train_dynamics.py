#!/usr/bin/env python3
"""Train one dynamics model on a prepared dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kynovar.data.config import load_run_config  # noqa: E402
from kynovar.evaluation.benchmark import prepare_dataset  # noqa: E402
from kynovar.models.factory import MODEL_NAMES  # noqa: E402
from kynovar.models.train import train_model  # noqa: E402
from kynovar.utils.paths import find_repo_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a Kynovar dynamics model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments" / "development.yaml",
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_run_config(args.config)
    store, splits, normalizer = prepare_dataset(config)
    root = find_repo_root()
    payload = train_model(
        args.model,
        store,
        splits["train"],
        splits["validation"],
        normalizer,
        epochs=args.epochs or config.epochs,
        batch_size=args.batch_size or config.batch_size,
        learning_rate=args.learning_rate or config.learning_rate,
        weight_decay=config.weight_decay if args.weight_decay is None else args.weight_decay,
        seed=args.seed or config.train_seed,
        device_name=args.device or config.device,
        sequence_length=config.sequence_length,
        stride=config.stride,
        rollout_steps=args.rollout_steps or config.rollout_steps,
        hidden_dim=config.hidden_dim,
        gnn_layers=config.gnn_layers,
        patience=config.patience,
        loss_weights=config.loss_weights,
        checkpoint_path=root / "checkpoints" / config.name / f"{args.model}.pt",
        num_workers=config.num_workers,
    )
    print(f"checkpoint={root / 'checkpoints' / config.name / f'{args.model}.pt'}")
    print(f"train_seconds={payload['train_seconds']:.6g}")
    print(f"best_val_position_rmse={payload['best_val_position_rmse']:.8g}")
    print(f"trainable_parameters={payload['trainable_parameters']}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
