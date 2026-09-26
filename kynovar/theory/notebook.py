"""Structured scientific notebook.

Every entry is a typed event with a machine-readable payload. The
natural-language line is rendered from that payload by a fixed template, so
the text can never claim more than the recorded numbers.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

TEMPLATES = {
    "experiment": "Ran experiment {experiment_id}: {summary}.",
    "hypothesis_created": "Proposed {hypothesis_id}: F = {equation} (weighted NMSE {fit_error:.3g}, complexity {complexity}).",
    "hypothesis_updated": "Tested {hypothesis_id} on {experiment_id}: {verdict}; {inside:.0%} of observations inside its 95% interval (median |z| {median_z:.2f}).",
    "status_changed": "{hypothesis_id} status {old} -> {new}: {reason}.",
    "merged": "Merged {merged_id} into {hypothesis_id}: {reason}.",
    "rejected": "Rejected {hypothesis_id}: {reason}.",
    "archived": "Archived {hypothesis_id}: {reason}.",
    "reactivated": "Reactivated {hypothesis_id}: {reason}.",
    "ranking": "Ranking by penalized log predictive density (not a probability): {ranking}.",
    "challenge": "Challenge experiment {experiment_id} targeted {target}: {summary}.",
    "counterexample": "Counterexample {counterexample_id} against {hypothesis_id}: {summary}.",
    "change_detected": "Residual monitor flagged a change at experiment {experiment_index} (statistic {statistic:.3g} > threshold {threshold:.3g}).",
    "investigation": "Investigation: {summary}.",
    "theory_version": "{theory_name} v{version}: F = {equation}. {reason}.",
    "note": "{text}",
}


@dataclass
class NotebookEvent:
    index: int
    kind: str
    timestamp: float
    payload: dict
    hypotheses: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        template = TEMPLATES.get(self.kind, "{kind}")
        try:
            return template.format(kind=self.kind, **self.payload)
        except (KeyError, ValueError, IndexError):
            return f"{self.kind}: {json.dumps(self.payload, default=str)}"

    def to_dict(self) -> dict:
        return {**asdict(self), "text": self.text}


class Notebook:
    def __init__(self) -> None:
        self.events: list[NotebookEvent] = []
        self._listeners = []

    def record(self, kind: str, hypotheses: list[str] | None = None, **payload) -> NotebookEvent:
        event = NotebookEvent(len(self.events), kind, time.time(), payload, list(hypotheses or []))
        self.events.append(event)
        for listener in self._listeners:
            listener(event)
        return event

    def subscribe(self, listener) -> None:
        self._listeners.append(listener)

    def of_kind(self, kind: str) -> list[NotebookEvent]:
        return [event for event in self.events if event.kind == kind]

    def summary(self, last: int | None = None) -> str:
        chosen = self.events if last is None else self.events[-last:]
        return "\n".join(f"[{event.index:04d}] {event.text}" for event in chosen)

    def to_list(self) -> list[dict]:
        return [event.to_dict() for event in self.events]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_list(), indent=2, default=str))
