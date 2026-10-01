"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Equation } from "@/components/Equation";
import type { Track } from "@/components/Scene";
import { api, websocketUrl, type Design, type Experiment, type Hypothesis, type NotebookEvent, type Prediction, type Reveal, type Summary, type TheoryState, type Vec3 } from "@/lib/api";

const Scene = dynamic(() => import("@/components/Scene").then((m) => m.Scene), { ssr: false });

const UNIVERSE = "K-0042";
type Tab = "DISCOVERY" | "THEORIES" | "CHALLENGE" | "NOTEBOOK" | "EVALUATION";
const TABS: Tab[] = ["DISCOVERY", "THEORIES", "CHALLENGE", "NOTEBOOK", "EVALUATION"];

const DEFAULT_DESIGN: Design = {
  masses: [1.2, 0.9],
  positions: [
    [-1.2, 0.0, 0.1],
    [1.2, 0.1, -0.1],
  ],
  velocities: [
    [0.0, 0.2, 0.0],
    [0.0, -0.25, 0.05],
  ],
  duration: 1.0,
  dt: 0.01,
};

function fmt(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Math.abs(value) >= 1e4 || (Math.abs(value) < 1e-3 && value !== 0) ? value.toExponential(2) : value.toFixed(digits);
}

export default function Page() {
  const [phase, setPhase] = useState<"landing" | "lab">("landing");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [offline, setOffline] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const connect = useCallback(async () => {
    try {
      await api.health();
      let world: Summary;
      try {
        world = await api.getWorld(UNIVERSE);
      } catch {
        world = await api.createWorld(UNIVERSE, 16);
      }
      setSummary(world);
      setOffline(null);
      if (world.status !== "UNINITIALIZED") setPhase("lab");
    } catch (error) {
      setOffline(`Laboratory backend unreachable (${String(error)}). Start it with: make backend`);
    }
  }, []);

  useEffect(() => {
    connect();
  }, [connect]);

  const begin = async () => {
    setBusy(true);
    try {
      setSummary(await api.start(UNIVERSE));
      setPhase("lab");
    } catch (error) {
      setOffline(String(error));
    } finally {
      setBusy(false);
    }
  };

  if (phase === "landing") {
    return (
      <main className="landing">
        <div className="landing-inner">
          <div className="wordmark">KYNOVAR</div>
          <div className="tagline">AUTONOMOUS SCIENTIFIC DISCOVERY</div>
          <div className="readout">
            <span className="k">UNIVERSE</span>
            <span className="v">{summary?.universe_id ?? "—"}</span>
            <span className="k">GOVERNING LAWS</span>
            <span className="v unknown">UNKNOWN</span>
            <span className="k">EXPERIMENTS</span>
            <span className="v">{summary ? summary.experiments : "—"}</span>
            <span className="k">THEORY STATE</span>
            <span className="v">{summary?.theory_state ?? "—"}</span>
          </div>
          <button className="begin" onClick={begin} disabled={!summary || busy}>
            [ BEGIN DISCOVERY ]
          </button>
          {offline && <div className="offline">{offline}</div>}
        </div>
      </main>
    );
  }
  return <Lab initial={summary} />;
}

function Lab({ initial }: { initial: Summary | null }) {
  const [tab, setTab] = useState<Tab>("DISCOVERY");
  const [summary, setSummary] = useState<Summary | null>(initial);
  const [theory, setTheory] = useState<TheoryState | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [notebook, setNotebook] = useState<NotebookEvent[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const seenNotes = useRef<Set<number>>(new Set());
  const seenExperiments = useRef<Set<string>>(new Set());

  const refresh = useCallback(async () => {
    try {
      const [s, t] = await Promise.all([api.status(UNIVERSE), api.theories(UNIVERSE)]);
      setSummary(s);
      setTheory(t);
      setError(null);
    } catch (e) {
      // Universes are intentionally in-memory. Recover cleanly when the API
      // container restarts while this browser tab remains open.
      if (String(e).includes("404:")) {
        const world = await api.createWorld(UNIVERSE, 16);
        setSummary(world);
        setTheory(null);
        setExperiments([]);
        setNotebook([]);
        seenNotes.current.clear();
        seenExperiments.current.clear();
        setError(null);
        return;
      }
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let pollTimer: ReturnType<typeof setInterval> | null = null;
    (async () => {
      await refresh();
      const [exps, notes] = await Promise.all([api.experiments(UNIVERSE), api.notebook(UNIVERSE)]);
      setExperiments(exps);
      setNotebook(notes);
      if (closed) return;
      seenNotes.current = new Set(notes.map((n) => n.index));
      seenExperiments.current = new Set(exps.map((e) => e.id));
      const openSocket = () => {
        if (closed) return;
        socket = new WebSocket(websocketUrl(UNIVERSE));
        socket.onopen = () => setConnected(true);
        socket.onerror = () => socket?.close();
        socket.onclose = () => {
          setConnected(false);
          if (!closed) reconnectTimer = setTimeout(openSocket, 2000);
        };
        socket.onmessage = (message) => {
          let event: { type: string; payload: any };
          try {
            event = JSON.parse(message.data) as { type: string; payload: any };
          } catch {
            return;
          }
        if (event.type === "notebook") {
          const note = event.payload as NotebookEvent;
          if (seenNotes.current.has(note.index)) return;
          seenNotes.current.add(note.index);
          setNotebook((prev) => [...prev, note]);
        } else if (event.type === "experiment") {
          const experiment = event.payload as Experiment;
          if (seenExperiments.current.has(experiment.id)) return;
          seenExperiments.current.add(experiment.id);
          setExperiments((prev) => [...prev, experiment]);
          api.status(UNIVERSE).then(setSummary).catch(() => undefined);
        } else if (event.type === "theory") {
          setTheory(event.payload as TheoryState);
          api.status(UNIVERSE).then(setSummary).catch(() => undefined);
        } else if (event.type === "status") {
          setSummary((prev) => (prev ? { ...prev, status: event.payload.status } : prev));
          if (event.payload.error) setError(`Discovery loop error: ${event.payload.error}`);
        }
        };
      };
      openSocket();
      // Polling keeps state correct through proxies that do not support WebSockets.
      pollTimer = setInterval(() => refresh(), 3000);
    })().catch((e) => setError(String(e)));
    return () => {
      closed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (pollTimer) clearInterval(pollTimer);
      socket?.close();
    };
  }, [refresh]);

  const leader = theory?.hypotheses.find((h) => h.id === theory.leader) ?? null;
  const latest = experiments.length ? experiments[experiments.length - 1] : null;

  const [challengeTracks, setChallengeTracks] = useState<Track[] | null>(null);
  const discoveryTracks = useMemo<Track[]>(
    () => (latest && latest.positions.length ? [{ positions: latest.positions, masses: latest.masses, style: "observed" }] : []),
    [latest],
  );
  const tracks = tab === "CHALLENGE" && challengeTracks ? challengeTracks : discoveryTracks;
  const running = summary?.status === "RUNNING";

  const toggle = async () => {
    try {
      setSummary(running ? await api.stop(UNIVERSE) : await api.start(UNIVERSE));
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <main className="lab">
      <header className="topbar">
        <span className="brand">KYNOVAR</span>
        <span className="stat">UNIVERSE <b>{summary?.universe_id ?? UNIVERSE}</b></span>
        <span className="stat">STATUS <b>{summary?.status ?? "—"}</b></span>
        <span className="stat">EXPERIMENTS <b>{summary?.experiments ?? 0}</b>/{summary?.budget ?? "—"}</span>
        <span className="stat">THEORY <b>{summary?.theory_state ?? "—"}</b></span>
        <span className="stat">STREAM <b>{connected ? "LIVE" : "OFFLINE"}</b></span>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </nav>
      </header>
      <div className="body">
        <section className="viewport">
          <div className="viewport-label">
            {tab === "CHALLENGE" && challengeTracks
              ? "CHALLENGE — dashed: Kynovar prediction · solid: reality"
              : latest
                ? `${latest.id} · ${latest.source} · ${latest.accepted ? "usable" : `rejected: ${latest.reason}`}`
                : "NO EXPERIMENTS YET"}
          </div>
          <Scene tracks={tracks} />
          <div className="viewport-legend">
            <span><i className="swatch" style={{ background: "var(--body-a)" }} />BODY 1</span>
            <span><i className="swatch" style={{ background: "var(--body-b)" }} />BODY 2</span>
            <span>3D · grid 0.5 units</span>
          </div>
        </section>
        <aside className="side">
          {error && (
            <div className="section">
              <div className="error">{error}</div>
            </div>
          )}
          {tab === "DISCOVERY" && (
            <DiscoveryPanel summary={summary} leader={leader} theory={theory} running={running} toggle={toggle} notebook={notebook} />
          )}
          {tab === "THEORIES" && <TheoryExplorer theory={theory} selected={selected} setSelected={setSelected} />}
          {tab === "CHALLENGE" && <ChallengePanel leader={leader} criteria={summary?.criteria ?? []} setTracks={setChallengeTracks} onDone={refresh} />}
          {tab === "NOTEBOOK" && <NotebookPanel notebook={notebook} />}
          {tab === "EVALUATION" && <EvaluationPanel />}
        </aside>
      </div>
    </main>
  );
}

function StatusBadge({ status }: { status: string }) {
  return <span className={`badge ${status}`}>{status}</span>;
}

function DiscoveryPanel({ summary, leader, theory, running, toggle, notebook }: { summary: Summary | null; leader: Hypothesis | null; theory: TheoryState | null; running: boolean; toggle: () => void; notebook: NotebookEvent[] }) {
  const active = theory?.hypotheses.filter((h) => ["PROPOSED", "SUPPORTED", "CHALLENGED"].includes(h.status)) ?? [];
  return (
    <>
      <div className="section">
        <h3>DISCOVERY LOOP</h3>
        <div className="controls">
          <button className={`btn ${running ? "" : "primary"}`} onClick={toggle} disabled={summary?.status === "CONVERGED" || summary?.status === "BUDGET_EXHAUSTED"}>
            {running ? "STOP" : "START"}
          </button>
        </div>
        <div className="kv" style={{ marginTop: 12 }}>
          <span className="k">loop status</span><span>{summary?.status ?? "—"}</span>
          <span className="k">experiments run</span><span>{summary?.experiments ?? 0} (usable {summary?.usable_experiments ?? 0})</span>
          <span className="k">observations</span><span>{summary?.ledger.observations ?? 0}</span>
          <span className="k">simulator steps</span><span>{summary?.ledger.steps ?? 0}</span>
          <span className="k">active hypotheses</span><span>{summary?.active_hypotheses ?? 0}</span>
          <span className="k">counterexamples</span><span>{summary?.counterexamples ?? 0}</span>
        </div>
      </div>
      <div className="section">
        <h3>CURRENT LEADING THEORY</h3>
        {leader ? (
          <>
            <div className="equation"><Equation latex={leader.latex} /></div>
            <div className="kv" style={{ marginTop: 10 }}>
              <span className="k">id</span><span>{leader.id} <StatusBadge status={leader.status} /></span>
              <span className="k">score</span><span>{fmt(leader.score)} <span className="dim">(penalized log predictive density; not a probability)</span></span>
              <span className="k">fit error</span><span>{fmt(leader.fit_error)} weighted NMSE</span>
              <span className="k">evidence</span><span>{leader.supporting_experiments.length} supporting · {leader.contradicting_experiments.length} contradicting</span>
            </div>
            <ParameterTable hypothesis={leader} />
          </>
        ) : (
          <div className="muted mono" style={{ fontSize: 12 }}>No hypothesis yet. Kynovar needs evidence first.</div>
        )}
      </div>
      <div className="section">
        <h3>COMPETING HYPOTHESES ({active.length})</h3>
        {active.map((h) => (
          <div key={h.id} className="hyp">
            <div className="row"><span>{h.id}</span><StatusBadge status={h.status} /></div>
            <div className="eq"><Equation latex={h.latex} /></div>
          </div>
        ))}
      </div>
      <div className="section">
        <h3>LATEST NOTEBOOK ENTRIES</h3>
        <NotebookList events={notebook.slice(-8)} />
      </div>
    </>
  );
}

function ParameterTable({ hypothesis }: { hypothesis: Hypothesis }) {
  if (!hypothesis.parameters.length) return null;
  return (
    <table className="data" style={{ marginTop: 10 }}>
      <thead>
        <tr><th>PARAM</th><th>ESTIMATE</th><th>STD</th><th>95% INTERVAL</th></tr>
      </thead>
      <tbody>
        {hypothesis.parameters.map((p) => (
          <tr key={p.name}>
            <td>{p.name}</td><td>{fmt(p.value, 4)}</td><td>{fmt(p.std, 4)}</td><td>[{fmt(p.ci95[0], 4)}, {fmt(p.ci95[1], 4)}]</td>
          </tr>
        ))}
      </tbody>
      <caption className="dim" style={{ captionSide: "bottom", textAlign: "left", fontSize: 10, paddingTop: 6 }}>{hypothesis.parameter_uncertainty_method}</caption>
    </table>
  );
}

function TheoryExplorer({ theory, selected, setSelected }: { theory: TheoryState | null; selected: string | null; setSelected: (id: string) => void }) {
  if (!theory || theory.hypotheses.length === 0) return <div className="section muted mono">No hypotheses have been proposed yet.</div>;
  const current = theory.hypotheses.find((h) => h.id === selected) ?? theory.hypotheses[0];
  return (
    <>
      <div className="section">
        <h3>ALL HYPOTHESES · RANKED</h3>
        {theory.hypotheses.map((h) => (
          <div key={h.id} className={`hyp ${h.id === current.id ? "selected" : ""}`} onClick={() => setSelected(h.id)}>
            <div className="row">
              <span>{h.id}{h.id === theory.leader ? " · LEADER" : ""}</span>
              <span>score {fmt(h.score)} <StatusBadge status={h.status} /></span>
            </div>
            <div className="eq"><Equation latex={h.latex} /></div>
          </div>
        ))}
      </div>
      <div className="section">
        <h3>{current.id} · DETAIL</h3>
        <div className="kv">
          <span className="k">equation</span><span>{current.equation}</span>
          <span className="k">source</span><span>{current.source}</span>
          <span className="k">complexity</span><span>{current.complexity} nodes</span>
          <span className="k">fit error</span><span>{fmt(current.fit_error)}</span>
          <span className="k">observations</span><span>{current.observations_tested} tested</span>
        </div>
        <ParameterTable hypothesis={current} />
        <table className="data" style={{ marginTop: 14 }}>
          <thead><tr><th>EXPERIMENT</th><th>VERDICT</th><th>INSIDE 95%</th><th>MEDIAN |z|</th></tr></thead>
          <tbody>
            {current.verdicts.map((v, i) => (
              <tr key={`${v.experiment_id}-${i}`}>
                <td>{v.experiment_id}</td><td>{v.verdict}</td><td>{(100 * v.inside_95).toFixed(0)}%</td><td>{fmt(v.median_z, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="section">
        <h3>COUNTEREXAMPLES ({theory.counterexamples.length}) · PERMANENT</h3>
        <table className="data">
          <thead><tr><th>ID</th><th>AGAINST</th><th>CRITERION</th><th>INSIDE 95%</th></tr></thead>
          <tbody>
            {theory.counterexamples.map((c) => (
              <tr key={c.counterexample_id}><td>{c.counterexample_id}</td><td>{c.hypothesis_id}</td><td>{c.criterion}</td><td>{(100 * c.inside_95).toFixed(0)}%</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function VectorRow({ label, value, onChange }: { label: string; value: Vec3; onChange: (v: Vec3) => void }) {
  return (
    <>
      <span className="dim">{label}</span>
      {value.map((component, axis) => (
        <input
          key={axis}
          type="number"
          step="0.05"
          value={component}
          onChange={(e) => {
            const next = [...value] as Vec3;
            next[axis] = Number(e.target.value);
            onChange(next);
          }}
        />
      ))}
    </>
  );
}

function ChallengePanel({ leader, criteria, setTracks, onDone }: { leader: Hypothesis | null; criteria: string[]; setTracks: (t: Track[] | null) => void; onDone: () => void }) {
  const [design, setDesign] = useState<Design>(DEFAULT_DESIGN);
  const [criterion, setCriterion] = useState<string>("extrapolation");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [reveal, setReveal] = useState<Reveal | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const predictionRef = useRef<Prediction | null>(null);

  const showPrediction = (p: Prediction) => {
    predictionRef.current = p;
    setPrediction(p);
    setReveal(null);
    setTracks([{ positions: p.predicted_positions, masses: p.design.masses, style: "predicted" }]);
  };

  const run = async (action: () => Promise<Prediction>) => {
    setBusy(true);
    setError(null);
    try {
      showPrediction(await action());
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const doReveal = async () => {
    const p = predictionRef.current;
    if (!p) return;
    setBusy(true);
    try {
      const r = await api.reveal(UNIVERSE, p.challenge_id);
      setReveal(r);
      const tracks: Track[] = [{ positions: p.predicted_positions, masses: p.design.masses, style: "predicted" }];
      if (r.actual_positions.length) tracks.push({ positions: r.actual_positions, masses: p.design.masses, style: "actual" });
      setTracks(tracks);
      onDone();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const setVector = (kind: "positions" | "velocities", body: 0 | 1, v: Vec3) => {
    const next = { ...design, [kind]: [...design[kind]] } as Design;
    next[kind][body] = v;
    setDesign(next);
  };

  const outcome = reveal?.challenge.outcome;
  return (
    <>
      <div className="section">
        <h3>CHALLENGE KYNOVAR</h3>
        {leader ? (
          <div className="muted" style={{ fontSize: 12, marginBottom: 12 }}>
            Design an experiment. Kynovar predicts it with <span className="mono">{leader.id}</span> before the universe is consulted.
          </div>
        ) : (
          <div className="muted mono" style={{ fontSize: 12 }}>Kynovar has no theory yet. Start discovery first.</div>
        )}
        <div className="form-grid">
          <span className="dim">mass</span>
          <input type="number" step="0.1" value={design.masses[0]} onChange={(e) => setDesign({ ...design, masses: [Number(e.target.value), design.masses[1]] })} />
          <input type="number" step="0.1" value={design.masses[1]} onChange={(e) => setDesign({ ...design, masses: [design.masses[0], Number(e.target.value)] })} />
          <span />
          <VectorRow label="pos 1" value={design.positions[0]} onChange={(v) => setVector("positions", 0, v)} />
          <VectorRow label="pos 2" value={design.positions[1]} onChange={(v) => setVector("positions", 1, v)} />
          <VectorRow label="vel 1" value={design.velocities[0]} onChange={(v) => setVector("velocities", 0, v)} />
          <VectorRow label="vel 2" value={design.velocities[1]} onChange={(v) => setVector("velocities", 1, v)} />
        </div>
        <div className="controls" style={{ marginTop: 12 }}>
          <button className="btn primary" disabled={!leader || busy} onClick={() => run(() => api.predict(UNIVERSE, design))}>PREDICT FUTURE</button>
          <button className="btn primary" disabled={!prediction || !!reveal || busy} onClick={doReveal}>REVEAL REALITY</button>
        </div>
      </div>
      <div className="section">
        <h3>BREAK THE THEORY</h3>
        <div className="muted" style={{ fontSize: 12, marginBottom: 10 }}>Let Kynovar's challenger design the experiment most likely to expose a weakness, then reveal what happens.</div>
        <div className="controls">
          <select value={criterion} onChange={(e) => setCriterion(e.target.value)}>
            {(criteria.length ? criteria : ["extrapolation"]).map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <button className="btn" disabled={!leader || busy} onClick={() => run(() => api.breakTheory(UNIVERSE, criterion))}>DESIGN ATTACK</button>
        </div>
      </div>
      {prediction && (
        <div className="section">
          <h3>PREDICTION {prediction.challenge_id}</h3>
          <div className="equation"><Equation latex={prediction.latex} /></div>
          <div className="kv" style={{ marginTop: 10 }}>
            <span className="k">hypothesis</span><span>{prediction.hypothesis}</span>
            <span className="k">criterion</span><span>{prediction.criterion}{prediction.criterion_score !== undefined ? ` (score ${fmt(prediction.criterion_score)})` : ""}</span>
            <span className="k">initial force</span><span>{fmt(prediction.predicted_force.mean[0])} [{fmt(prediction.predicted_force.low95[0])}, {fmt(prediction.predicted_force.high95[0])}] 95%</span>
            <span className="k">masses</span><span>{prediction.design.masses.map((m) => fmt(m, 2)).join(", ")}</span>
          </div>
        </div>
      )}
      {reveal && outcome && (
        <div className="section">
          <h3>REALITY</h3>
          {outcome.usable ? (
            <div className={`verdict ${outcome.verdict}`}>
              {String(outcome.verdict).toUpperCase()} · {(100 * (outcome.inside_95 ?? 0)).toFixed(0)}% of observed forces inside the 95% interval · median |z| {fmt(outcome.median_z, 2)}
            </div>
          ) : (
            <div className="verdict inconclusive">EXPERIMENT NOT USABLE · {outcome.reason}</div>
          )}
          <div className="kv" style={{ marginTop: 10 }}>
            <span className="k">trajectory RMSE</span><span>{reveal.trajectory_rmse === null ? "not computed (outcome unusable)" : `${fmt(reveal.trajectory_rmse, 4)} units`}</span>
            <span className="k">theory status now</span><span><StatusBadge status={reveal.hypothesis_status} /></span>
            <span className="k">new counterexamples</span><span>{reveal.new_counterexamples.length ? reveal.new_counterexamples.map((c) => `${c.counterexample_id} vs ${c.hypothesis_id}`).join(", ") : "none"}</span>
          </div>
        </div>
      )}
      {error && <div className="section"><div className="error">{error}</div></div>}
    </>
  );
}

function NotebookList({ events }: { events: NotebookEvent[] }) {
  return (
    <div className="notebook">
      {events.map((e) => (
        <div key={e.index} className="entry">
          <span className="idx">{String(e.index).padStart(3, "0")}</span>
          <span className="kind">{e.kind.toUpperCase()}</span>
          {e.text}
        </div>
      ))}
      {events.length === 0 && <div className="muted">No entries yet.</div>}
    </div>
  );
}

function NotebookPanel({ notebook }: { notebook: NotebookEvent[] }) {
  const [filter, setFilter] = useState("all");
  const kinds = Array.from(new Set(notebook.map((e) => e.kind)));
  const events = filter === "all" ? notebook : notebook.filter((e) => e.kind === filter);
  return (
    <div className="section">
      <h3>RESEARCH NOTEBOOK · {notebook.length} ENTRIES</h3>
      <div className="controls" style={{ marginBottom: 10 }}>
        <select value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="all">all events</option>
          {kinds.map((k) => <option key={k} value={k}>{k}</option>)}
        </select>
      </div>
      <NotebookList events={[...events].reverse()} />
    </div>
  );
}

function EvaluationPanel() {
  const [data, setData] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const load = async () => {
    setBusy(true);
    setError(null);
    try {
      setData((await api.metrics(UNIVERSE)).evaluation);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="section">
      <h3>EVALUATION AGAINST HIDDEN GROUND TRUTH</h3>
      <div className="eval-banner">EVALUATION ONLY. The hidden law is revealed here for human assessment. The discovery session never receives it.</div>
      <button className="btn" onClick={load} disabled={busy}>{busy ? "COMPUTING…" : "COMPUTE EVALUATION"}</button>
      {error && <div className="error">{error}</div>}
      {data && !data.available && <div className="muted mono" style={{ marginTop: 12 }}>{data.reason}</div>}
      {data && data.available && (
        <>
          <div style={{ marginTop: 14 }} className="dim mono">GROUND TRUTH</div>
          <div className="equation"><Equation latex={data.ground_truth_latex} /></div>
          <div style={{ marginTop: 10 }} className="dim mono">KYNOVAR ({data.discovered_status})</div>
          <div className="equation"><Equation latex={data.discovered_latex} /></div>
          <div className="kv" style={{ marginTop: 12 }}>
            <span className="k">structural match</span><span>{String(data.structural_match)}</span>
            <span className="k">recovered (±10%)</span><span>{String(data.recovered)}</span>
            <span className="k">max coeff rel err</span><span>{fmt(data.max_coefficient_relative_error, 4)}</span>
            <span className="k">max exponent err</span><span>{fmt(data.max_exponent_abs_error, 4)}</span>
            <span className="k">held-out traj RMSE</span><span>{(data.trajectory_rmse_test_designs as (number | null)[]).map((r) => fmt(r, 4)).join(" · ")}</span>
            <span className="k">first correct leader</span><span>{data.experiments_to_first_correct_leader ?? "never"} experiments</span>
            <span className="k">experiments total</span><span>{data.experiments_total}</span>
          </div>
        </>
      )}
    </div>
  );
}
