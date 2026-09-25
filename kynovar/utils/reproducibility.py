"""Metadata recorded with every scientific run."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kynovar import __version__


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def runtime_record(root: Path, seed: int | None = None) -> dict[str, object]:
    """Software and run metadata. This record contains no physical ground truth."""
    record: dict[str, object] = {
        "project": "kynovar",
        "version": __version__,
        "git_commit": _git_commit(root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if seed is not None:
        record["seed"] = int(seed)
    return record
