"""Keep path-related environment changes from leaking between tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ENV_KEYS = (
    "KYNOVAR_ROOT",
    "KYNOVAR_DATA",
    "KYNOVAR_DATA_ROOT",
    "KYNOVAR_CHECKPOINTS",
    "KYNOVAR_RESULTS",
    "KYNOVAR_LOGS",
    "HF_HOME",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
)


@pytest.fixture(autouse=True)
def restore_project_env():
    snapshot = {key: os.environ.get(key) for key in _ENV_KEYS}
    yield
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
