"""Backend API: worlds, experiments, discovery, challenge, metrics, streaming."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend.kynovar_api import app as app_module  # noqa: E402
from backend.kynovar_api.universes import Registry  # noqa: E402
from kynovar.simulator.boundary import iter_keys  # noqa: E402

HIDDEN_KEYS = {"k", "p", "force_law", "forces", "softening", "hidden_parameters", "ground_truth"}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(app_module, "registry", Registry())
    return TestClient(app_module.app)


def _design():
    return {"masses": [1.0, 1.5], "positions": [[-1.0, 0.0, 0.1], [1.0, 0.1, -0.1]], "velocities": [[0.0, 0.2, 0.0], [0.0, -0.15, 0.0]], "duration": 0.5}


def test_world_lifecycle_and_hidden_keys(client) -> None:
    created = client.post("/worlds", json={"preset": "K-0042", "budget": 8}).json()
    assert created["universe_id"] == "K-0042" and created["governing_laws"] == "UNKNOWN"
    assert created["theory_state"] == "UNINITIALIZED" and created["experiments"] == 0
    assert not HIDDEN_KEYS & set(iter_keys(created))
    experiment = client.post("/worlds/K-0042/experiments", json=_design()).json()
    assert experiment["id"] == "E-0001" and len(experiment["positions"][0]) == 2 and len(experiment["positions"][0][0]) == 3
    assert client.get("/worlds/K-0042/experiments").json()[0]["id"] == "E-0001"
    bad = dict(_design(), positions=[[0, 0], [1, 1]])
    assert client.post("/worlds/K-0042/experiments", json=bad).status_code == 422
    too_small_dt = dict(_design(), dt=0.000001)
    assert client.post("/worlds/K-0042/experiments", json=too_small_dt).status_code == 422
    out_of_range = dict(_design(), velocities=[[101, 0, 0], [0, 0, 0]])
    assert client.post("/worlds/K-0042/experiments", json=out_of_range).status_code == 422
    assert client.get("/worlds/NOPE").status_code == 404


def test_discovery_challenge_and_metrics(client) -> None:
    client.post("/worlds", json={"preset": "K-DEMO", "budget": 10})
    universe = app_module.registry.get("K-DEMO")
    universe.session._loop()  # run synchronously for the test
    status = client.get("/worlds/K-DEMO/discovery/status").json()
    assert status["experiments"] >= 4 and status["leader"] is not None
    theories = client.get("/worlds/K-DEMO/theories").json()
    assert theories["hypotheses"] and all("score_semantics" in h for h in theories["hypotheses"])
    assert not HIDDEN_KEYS & set(iter_keys(theories))
    prediction = client.post("/worlds/K-DEMO/challenge", json={"design": _design()}).json()
    assert prediction["predicted_positions"] and prediction["challenge_id"].startswith("C-")
    outcome = client.post(f"/worlds/K-DEMO/challenge/{prediction['challenge_id']}/reveal").json()
    assert outcome["challenge"]["outcome"]["usable"] in (True, False)
    assert outcome["challenge"]["challenge_id"] == prediction["challenge_id"]
    assert client.post(f"/worlds/K-DEMO/challenge/{prediction['challenge_id']}/reveal").status_code == 404
    adversarial = client.post("/worlds/K-DEMO/challenge", json={"criterion": "extrapolation"})
    assert adversarial.status_code == 200 and adversarial.json()["criterion"] in ("extrapolation", "uncertainty", "disagreement", "weakly_observed")
    adversarial_prediction = adversarial.json()
    adversarial_outcome = client.post(f"/worlds/K-DEMO/challenge/{adversarial_prediction['challenge_id']}/reveal").json()
    assert adversarial_outcome["challenge"]["challenge_id"] == adversarial_prediction["challenge_id"]
    assert adversarial_outcome["challenge"]["criterion_score"] == pytest.approx(adversarial_prediction["criterion_score"])
    metrics = client.get("/worlds/K-DEMO/metrics").json()
    evaluation = metrics["evaluation"]
    assert evaluation["available"] and evaluation["label"].startswith("EVALUATION ONLY")
    assert evaluation["ground_truth"] == "F = 4 * m1 * m2 / r^3"
    assert "PROVEN" not in str(client.get("/worlds/K-DEMO/notebook").json())


def test_websocket_streams_real_events(client) -> None:
    client.post("/worlds", json={"preset": "K-0042", "budget": 6})
    client.post("/worlds/K-0042/experiments", json=_design())
    with client.websocket_connect("/ws/K-0042") as socket:
        first = socket.receive_json()
        assert first["index"] == 0 and first["type"] in ("notebook", "experiment", "status", "theory")


def test_background_start_stop(client) -> None:
    client.post("/worlds", json={"preset": "K-0042", "budget": 6})
    assert client.post("/worlds/K-0042/discovery/start").json()["status"] == "RUNNING"
    time.sleep(0.5)
    stopped = client.post("/worlds/K-0042/discovery/stop").json()
    assert stopped["status"] in ("STOPPED", "CONVERGED", "BUDGET_EXHAUSTED")
