"""HTTP and WebSocket API for the interactive laboratory.

Run:  .venv/bin/uvicorn backend.kynovar_api.app:app --port 8000
"""

from __future__ import annotations

import asyncio
import math
import os
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.kynovar_api.universes import Registry, evaluate
from kynovar.discovery.lab import ExperimentDesign

registry = Registry()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    registry.close()


def _cors_origins() -> list[str]:
    configured = os.environ.get("KYNOVAR_CORS_ORIGINS", "")
    if configured.strip():
        return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    return ["http://localhost:3000", "http://127.0.0.1:3000"]


app = FastAPI(title="Kynovar Laboratory API", version="0.8.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins(), allow_methods=["*"], allow_headers=["*"])


class WorldRequest(BaseModel):
    preset: str | None = "K-0042"
    seed: int = 0
    noise: float = Field(0.02, ge=0.0, le=0.5)
    budget: int = Field(16, ge=4, le=60)


class DesignRequest(BaseModel):
    masses: list[float] = Field(min_length=2, max_length=2)
    positions: list[list[float]] = Field(min_length=2, max_length=2)
    velocities: list[list[float]] = Field(min_length=2, max_length=2)
    duration: float = Field(1.0, gt=0.0, le=4.0)
    dt: float = Field(0.01, ge=0.001, le=0.1)

    def design(self) -> ExperimentDesign:
        scalars = [*self.masses, self.duration, self.dt, *(value for vector in self.positions + self.velocities for value in vector)]
        if not all(math.isfinite(value) for value in scalars):
            raise HTTPException(422, "experiment values must be finite")
        for vector in self.positions + self.velocities:
            if len(vector) != 3:
                raise HTTPException(422, "positions and velocities must be 3D vectors")
        if any(abs(value) > 100 for vector in self.positions + self.velocities for value in vector):
            raise HTTPException(422, "position and velocity components must be in [-100, 100]")
        if any(m <= 0 or m > 10 for m in self.masses):
            raise HTTPException(422, "masses must be in (0, 10]")
        steps = self.duration / self.dt
        if abs(steps - round(steps)) > 1e-9:
            raise HTTPException(422, "duration must be an integer multiple of dt")
        return ExperimentDesign(tuple(self.masses), tuple(tuple(p) for p in self.positions), tuple(tuple(v) for v in self.velocities), self.duration, self.dt, "user")


class ChallengeRequest(BaseModel):
    design: DesignRequest | None = None
    criterion: str | None = None


def _universe(universe_id: str):
    try:
        return registry.get(universe_id)
    except KeyError as error:
        raise HTTPException(404, f"unknown universe {universe_id}") from error


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/worlds")
def create_world(request: WorldRequest) -> dict:
    return registry.create(request.preset, request.seed, request.noise, request.budget).public()


@app.get("/worlds")
def list_worlds() -> list[dict]:
    return registry.list_public()


@app.get("/worlds/{universe_id}")
def get_world(universe_id: str) -> dict:
    return _universe(universe_id).public()


@app.post("/worlds/{universe_id}/experiments")
def run_experiment(universe_id: str, request: DesignRequest) -> dict:
    return _universe(universe_id).session.run_experiment(request.design(), "user-designed experiment")


@app.get("/worlds/{universe_id}/experiments")
def list_experiments(universe_id: str) -> list[dict]:
    return _universe(universe_id).session.experiments


@app.post("/worlds/{universe_id}/discovery/start")
def start_discovery(universe_id: str) -> dict:
    session = _universe(universe_id).session
    session.start()
    return session.summary()


@app.post("/worlds/{universe_id}/discovery/stop")
def stop_discovery(universe_id: str) -> dict:
    session = _universe(universe_id).session
    session.stop()
    return session.summary()


@app.get("/worlds/{universe_id}/discovery/status")
def discovery_status(universe_id: str) -> dict:
    return _universe(universe_id).session.summary()


@app.get("/worlds/{universe_id}/theories")
def theories(universe_id: str) -> dict:
    return _universe(universe_id).session.theory_state()


@app.post("/worlds/{universe_id}/challenge")
def challenge(universe_id: str, request: ChallengeRequest) -> dict:
    session = _universe(universe_id).session
    try:
        if request.design is not None:
            return session.predict(request.design.design(), "user")
        design, criterion, score = session.adversarial_design(request.criterion)
        return session.predict(design, criterion, score)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@app.post("/worlds/{universe_id}/challenge/{challenge_id}/reveal")
def reveal(universe_id: str, challenge_id: str) -> dict:
    try:
        return _universe(universe_id).session.reveal(challenge_id)
    except KeyError as error:
        raise HTTPException(404, f"unknown or already revealed challenge {challenge_id}") from error


@app.get("/worlds/{universe_id}/metrics")
def metrics(universe_id: str) -> dict:
    universe = _universe(universe_id)
    return {"ledger": universe.session.client.ledger.to_dict(), "evaluation": evaluate(universe)}


@app.get("/worlds/{universe_id}/notebook")
def notebook(universe_id: str) -> list[dict]:
    return _universe(universe_id).session.manager.notebook.to_list()


@app.websocket("/ws/{universe_id}")
async def stream(websocket: WebSocket, universe_id: str) -> None:
    await websocket.accept()
    try:
        session = registry.get(universe_id).session
    except KeyError:
        await websocket.close(code=4404)
        return
    cursor = 0
    async def wait_for_disconnect() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

    disconnect = asyncio.create_task(wait_for_disconnect())
    try:
        while not disconnect.done():
            events = session.events_since(cursor)
            for event in events:
                await websocket.send_json(event)
            cursor += len(events)
            await asyncio.sleep(0.2)
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        disconnect.cancel()
        with suppress(asyncio.CancelledError, WebSocketDisconnect):
            await disconnect
