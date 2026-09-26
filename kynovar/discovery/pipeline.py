"""End-to-end passive discovery: probe, collect evidence, search for laws.

The protocol is the same for every hidden world, so the scientist is never
told which family it faces:

1. Single-body probes. A lone body with some initial velocity. Any measured
   acceleration means a local force (for example drag) exists.
2. Two-body experiments. The signed central force along the line of centers
   is analyzed with variables m1, m2, r, and radial velocity vr.

If single-body probes show a local force, a single-body law is searched as
well. Every result comes only from laboratory observations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from kynovar.discovery.engine import DiscoveredLaw, EngineConfig, SymbolicDiscoveryEngine
from kynovar.discovery.evidence import Evidence, pairwise_evidence, single_body_evidence, state_layout
from kynovar.discovery.lab import DesignRanges, ExperimentClient, collect, random_single, random_two_body

LOCAL_FORCE_THRESHOLD = 1e-6


@dataclass
class DiscoveryReport:
    local_force_detected: bool
    local_force_scale: float
    pairwise_laws: list[DiscoveredLaw]
    single_laws: list[DiscoveredLaw]
    pairwise_evidence: Evidence | None
    single_evidence: Evidence | None
    rejected: list[dict]
    ledger: dict
    seconds: float
    notes: list[str] = field(default_factory=list)

    def best(self, kind: str) -> DiscoveredLaw | None:
        laws = self.pairwise_laws if kind == "pairwise" else self.single_laws
        return laws[0] if laws else None


def probe_local_force(client: ExperimentClient, ranges: DesignRanges, rng: np.random.Generator, probes: int = 4) -> tuple[bool, float, list[np.ndarray]]:
    designs = [random_single(rng, ranges, label=f"probe-{index}") for index in range(probes)]
    kept, _rejected = collect(client, designs)
    if not kept:
        return False, 0.0, []
    scale = float(np.median([np.linalg.norm(states[:, 0, state_layout(states)["acc"]], axis=-1).mean() for states in kept]))
    return scale > LOCAL_FORCE_THRESHOLD, scale, kept


def discover(
    client: ExperimentClient,
    pairwise_experiments: int = 24,
    single_experiments: int = 16,
    ranges: DesignRanges | None = None,
    engine_config: EngineConfig | None = None,
    max_frames: int = 3000,
    seed: int = 0,
) -> DiscoveryReport:
    started = time.perf_counter()
    ranges = ranges or DesignRanges()
    rng = np.random.default_rng(seed)
    engine = SymbolicDiscoveryEngine(engine_config or EngineConfig(seed=seed))
    notes: list[str] = []
    local, scale, probe_runs = probe_local_force(client, ranges, rng)

    single_laws: list[DiscoveredLaw] = []
    single: Evidence | None = None
    rejected: list[dict] = []
    if local:
        designs = [random_single(rng, ranges, label=f"single-{index}") for index in range(single_experiments)]
        kept, dropped = collect(client, designs)
        rejected += dropped
        single = single_body_evidence(probe_runs + kept, max_frames=max_frames, seed=seed)
        single_laws = engine.discover(single)
        notes.append(f"Single-body probes measured mean |a| = {scale:.4g}; a local force is present.")
    else:
        notes.append(f"Single-body probes measured mean |a| = {scale:.3g}; no local force detected.")

    designs = [random_two_body(rng, ranges, label=f"pair-{index}") for index in range(pairwise_experiments)]
    kept, dropped = collect(client, designs)
    rejected += dropped
    pairwise: Evidence | None = None
    pairwise_laws: list[DiscoveredLaw] = []
    if kept:
        pairwise = pairwise_evidence(kept, include_velocity=True, max_frames=max_frames, seed=seed)
        pairwise_laws = engine.discover(pairwise)
    else:
        notes.append("Every two-body experiment was rejected; no pairwise evidence.")
    return DiscoveryReport(local, scale, pairwise_laws, single_laws, pairwise, single, rejected, client.ledger.to_dict(), time.perf_counter() - started, notes)
