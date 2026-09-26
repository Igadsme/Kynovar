"""Information-boundary tests for the scientist-side packages.

Two layers:
  * a source scan: scientist code must not name hidden-state accessors or
    evaluation modules that carry ground truth;
  * a runtime trap: discovery runs end to end while `internal_state` and any
    `hidden_parameters` call outside the world's integrity check raise.
"""

from __future__ import annotations

import inspect

import pytest

from kynovar.simulator.forces import ForceLaw
from kynovar.simulator.world import World
from kynovar.utils.paths import find_repo_root

ROOT = find_repo_root()
SCIENTIST_PATHS = (
    "kynovar/discovery",
    "kynovar/theory",
    "kynovar/uncertainty",
    "kynovar/planning",
    "kynovar/falsification",
    "kynovar/perception",
)
FORBIDDEN_TOKENS = (
    "internal_state",
    "hidden_parameters",
    "ground_truth",
    "kynovar.evaluation",
    "kynovar.simulator.forces",
    "PairwisePowerLaw",
    "LinearDrag",
    "QuadraticDrag",
    "SpringForce",
    "_world",
    "_force_laws",
    ".truth",
)


def _sources():
    for relative in SCIENTIST_PATHS:
        path = ROOT / relative
        if not path.exists():
            continue
        for source in path.rglob("*.py"):
            if not source.name.startswith("._"):
                yield source


def test_scientist_sources_do_not_name_hidden_state() -> None:
    scanned = 0
    for source in _sources():
        scanned += 1
        text = source.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            assert token not in text, f"{source.relative_to(ROOT)} contains {token!r}"
    assert scanned > 0


@pytest.fixture()
def hidden_state_trap(monkeypatch):
    calls = {"internal_state": 0, "hidden_parameters": 0}

    def forbidden_internal_state(self):
        calls["internal_state"] += 1
        raise AssertionError("internal_state() called during discovery")

    originals = {cls: cls.hidden_parameters for cls in ForceLaw.__subclasses__()}

    def guarded(original):
        def wrapper(self):
            caller = inspect.stack()[1].filename
            if not caller.endswith("kynovar/simulator/world.py"):
                calls["hidden_parameters"] += 1
                raise AssertionError(f"hidden_parameters() called from {caller}")
            return original(self)

        return wrapper

    monkeypatch.setattr(World, "internal_state", forbidden_internal_state)
    for cls, original in originals.items():
        monkeypatch.setattr(cls, "hidden_parameters", guarded(original))
    return calls


def test_discovery_runs_without_hidden_state(hidden_state_trap) -> None:
    from kynovar.discovery.engine import EngineConfig
    from kynovar.discovery.lab import ExperimentClient
    from kynovar.discovery.pipeline import discover
    from kynovar.discovery.symbolic import RegressorConfig
    from kynovar.evaluation.worlds import catalog

    world = catalog()["power_plus_drag"]
    client = ExperimentClient(world.laboratory())
    config = EngineConfig(regressor=RegressorConfig(population=40, generations=3), restarts=1, methods=("symbolic", "power_sum"))
    report = discover(client, pairwise_experiments=8, single_experiments=2, engine_config=config)
    assert report.single_laws
    assert report.pairwise_laws
    assert hidden_state_trap == {"internal_state": 0, "hidden_parameters": 0}
