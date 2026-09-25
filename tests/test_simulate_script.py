"""The operator script writes observations and, separately, evaluation truth."""

from __future__ import annotations

import json

import pytest

from kynovar.simulator.boundary import assert_public_record
from kynovar.utils.paths import PathConfigError, find_repo_root
from scripts.simulate_world import main


def test_script_writes_a_public_trajectory_and_optional_truth(capsys: pytest.CaptureFixture[str]) -> None:
    root = find_repo_root()
    observations = root / "results" / "_pytest_observations.json"
    truth = root / "results" / "_pytest_truth.json"
    try:
        status = main(
            [
                "--seed",
                "4",
                "--difficulty",
                "2",
                "--duration",
                "0.05",
                "--dt",
                "0.01",
                "--output",
                str(observations),
                "--write-ground-truth",
                str(truth),
            ]
        )
        assert status == 0
        payload = json.loads(observations.read_text(encoding="utf-8"))
        assert_public_record(payload)
        assert payload["record_type"] == "trajectory"
        assert payload["run"]["project"] == "kynovar"
        assert "seed" not in payload["run"]
        assert "pairwise_power_law" not in observations.read_text(encoding="utf-8")
        hidden = json.loads(truth.read_text(encoding="utf-8"))
        assert hidden["record_type"] == "ground_truth"
        assert hidden["seed"] == 4
        assert hidden["forces"][0]["law"] == "pairwise_power_law"
        assert "k" in hidden["forces"][0]["parameters"]
        printed = capsys.readouterr().out
        assert "universe_id=" in printed
        assert "pairwise_power_law" not in printed
        assert str(hidden["forces"][0]["parameters"]["k"]) not in printed
    finally:
        observations.unlink(missing_ok=True)
        truth.unlink(missing_ok=True)


def test_script_refuses_a_path_outside_the_project() -> None:
    with pytest.raises(PathConfigError):
        main(["--seed", "1", "--output", "/tmp/kynovar-observations.json"])


def test_script_refuses_to_overwrite_observations_with_truth() -> None:
    with pytest.raises(SystemExit, match="different files"):
        main(
            [
                "--seed",
                "1",
                "--duration",
                "0.0",
                "--output",
                "results/same.json",
                "--write-ground-truth",
                "results/same.json",
            ]
        )
