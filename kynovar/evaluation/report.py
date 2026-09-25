"""Markdown and CSV reports rendered from measured benchmark records."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_benchmark_csv(path: Path, models: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    horizons = _shared_horizons(models)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "horizon",
                "position_rmse",
                "velocity_rmse",
                "acceleration_rmse",
                "position_mae",
                "velocity_mae",
                "position_r2",
                "position_relative_rmse",
                "position_rmse_bounded",
                "trainable_parameters",
                "train_seconds",
            ]
        )
        for model in models:
            for horizon in horizons:
                metrics = model["horizons"][horizon]
                writer.writerow(
                    [
                        model["model"],
                        horizon,
                        _cell(metrics["position"]["rmse"]),
                        _cell(metrics["velocity"]["rmse"]),
                        _cell(metrics["acceleration"]["rmse"]),
                        _cell(metrics["position"]["mae"]),
                        _cell(metrics["velocity"]["mae"]),
                        _cell(metrics["position"]["r2"]),
                        _cell(metrics["position"]["relative_rmse"]),
                        _cell((metrics.get("position_bounded") or {}).get("rmse")),
                        model["trainable_parameters"],
                        _cell(model["train_seconds"]),
                    ]
                )


def render_benchmark_markdown(payload: dict) -> str:
    models = payload["models"]
    horizons = _shared_horizons(models)
    lines = [
        f"# {payload['title']}",
        "",
        f"Device: `{payload['device']}`",
        "",
        f"Dataset: `{payload['dataset']}`",
        "",
        "Position RMSE is the root mean square error of the x and y components at that horizon.",
        "Values are copied from the measured JSON record.",
        "",
    ]
    if payload.get("loss"):
        lines.extend([str(payload["loss"]), ""])
    lines.extend(_table(models, horizons))
    if any("position_bounded" in model["horizons"][horizon] for model in models for horizon in horizons):
        lines.extend(
            [
                "",
                "## Bounded targets",
                "",
                "Position RMSE on target states with true position norm ≤ 8, velocity norm ≤ 8, and acceleration norm ≤ 80.",
                "Frames outside that box are stiff ejections and are omitted from this table only.",
                "",
            ]
        )
        lines.extend(_table(models, horizons, channel="position_bounded"))
    comparison = payload.get("rollout_training_comparison") or []
    if comparison:
        lines.extend(["", "## Rollout-aware training", ""])
        note = payload.get("rollout_training_note")
        if note:
            lines.extend([str(note), ""])
        lines.extend(_table(comparison, _shared_horizons(comparison)))
    lines.append("")
    return "\n".join(lines)


def render_ood_markdown(payload: dict) -> str:
    lines = [
        f"# {payload['title']}",
        "",
        f"Device: `{payload['device']}`",
        "",
        "Each cell is position RMSE at the stated horizon.",
        "",
    ]
    for scenario in payload["scenarios"]:
        lines.append(f"## {scenario['name']}")
        lines.append("")
        lines.append(scenario["description"])
        lines.append("")
        horizons = _shared_horizons(scenario["models"])
        headers = ["Model", *[f"{horizon}-step" for horizon in horizons]]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] + ["---:"] * (len(horizons))) + " |")
        for model in scenario["models"]:
            cells = [model["model"]]
            for horizon in horizons:
                cells.append(_cell(model["horizons"][horizon]["position"]["rmse"]))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _table(models: list[dict], horizons: list[str], channel: str = "position") -> list[str]:
    headers = ["Model", *[f"{horizon}-step position RMSE" for horizon in horizons], "Parameters", "Train seconds"]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for model in models:
        cells = [model["model"]]
        for horizon in horizons:
            metric = model["horizons"][horizon].get(channel)
            cells.append("" if metric is None else _cell(metric["rmse"]))
        cells.append(str(model["trainable_parameters"]))
        cells.append(_cell(model["train_seconds"]))
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def _shared_horizons(models: list[dict]) -> list[str]:
    if not models:
        return []
    common = set(models[0]["horizons"])
    for model in models[1:]:
        common &= set(model["horizons"])
    return sorted(common, key=lambda item: int(item))


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)
