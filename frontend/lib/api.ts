export const API = (process.env.NEXT_PUBLIC_KYNOVAR_API ?? "/api").replace(/\/$/, "");

export function websocketUrl(universeId: string): string {
  const explicit = process.env.NEXT_PUBLIC_KYNOVAR_WS;
  const base = explicit
    ? new URL(explicit, window.location.origin)
    : new URL(API, window.location.origin);
  if (!explicit) base.pathname = `${base.pathname.replace(/\/$/, "")}/ws`;
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}/${encodeURIComponent(universeId)}`;
  return base.toString();
}

export type Vec3 = [number, number, number];

export interface Design {
  masses: [number, number];
  positions: [Vec3, Vec3];
  velocities: [Vec3, Vec3];
  duration: number;
  dt?: number;
}

export interface Experiment {
  id: string;
  source: string;
  design: Design & { label: string };
  accepted: boolean;
  reason: string | null;
  dt: number;
  frame_stride: number;
  positions: Vec3[][];
  masses: number[];
}

export interface ParameterSummary {
  name: string;
  value: number;
  std: number;
  ci95: [number, number];
}

export interface Verdict {
  experiment_id: string;
  verdict: string;
  inside_95: number;
  median_z: number;
}

export interface Hypothesis {
  id: string;
  equation: string;
  latex: string;
  status: string;
  score: number | null;
  score_semantics: string;
  fit_error: number;
  complexity: number;
  source: string;
  parameters: ParameterSummary[];
  parameter_uncertainty_method: string;
  supporting_experiments: string[];
  contradicting_experiments: string[];
  verdicts: Verdict[];
  observations_tested: number;
}

export interface Counterexample {
  counterexample_id: string;
  hypothesis_id: string;
  equation: string;
  criterion: string;
  inside_95: number;
  median_z: number;
}

export interface Ledger {
  experiments: number;
  observations: number;
  steps: number;
  seconds: number;
}

export interface Summary {
  universe_id: string;
  status: string;
  experiments: number;
  usable_experiments: number;
  budget: number;
  theory_state: string;
  leader: Hypothesis | null;
  active_hypotheses: number;
  counterexamples: number;
  ledger: Ledger;
  criteria: string[];
}

export interface TheoryState {
  status: string;
  leader: string | null;
  hypotheses: Hypothesis[];
  counterexamples: Counterexample[];
  ledger: Ledger;
  theory_state: string;
}

export interface NotebookEvent {
  index: number;
  kind: string;
  timestamp: number;
  text: string;
  payload: Record<string, unknown>;
}

export interface Prediction {
  challenge_id: string;
  hypothesis: string;
  equation: string;
  latex: string;
  design: Design;
  criterion: string;
  criterion_score?: number;
  frame_stride: number;
  predicted_positions: Vec3[][];
  predicted_force: { mean: number[]; low95: number[]; high95: number[] };
}

export interface Reveal {
  challenge: { challenge_id: string; outcome: { usable: boolean; verdict?: string; inside_95?: number; median_z?: number; max_z?: number; reason?: string } };
  actual_positions: Vec3[][];
  trajectory_rmse: number | null;
  new_counterexamples: Counterexample[];
  hypothesis_status: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status}: ${text}`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<{ ok: boolean }>("/health"),
  getWorld: (id: string) => request<Summary>(`/worlds/${id}`),
  createWorld: (preset: string, budget = 16) => request<Summary>("/worlds", { method: "POST", body: JSON.stringify({ preset, budget }) }),
  start: (id: string) => request<Summary>(`/worlds/${id}/discovery/start`, { method: "POST" }),
  stop: (id: string) => request<Summary>(`/worlds/${id}/discovery/stop`, { method: "POST" }),
  status: (id: string) => request<Summary>(`/worlds/${id}/discovery/status`),
  theories: (id: string) => request<TheoryState>(`/worlds/${id}/theories`),
  experiments: (id: string) => request<Experiment[]>(`/worlds/${id}/experiments`),
  runExperiment: (id: string, design: Design) => request<Experiment>(`/worlds/${id}/experiments`, { method: "POST", body: JSON.stringify(design) }),
  predict: (id: string, design: Design) => request<Prediction>(`/worlds/${id}/challenge`, { method: "POST", body: JSON.stringify({ design }) }),
  breakTheory: (id: string, criterion: string) => request<Prediction>(`/worlds/${id}/challenge`, { method: "POST", body: JSON.stringify({ criterion }) }),
  reveal: (id: string, challengeId: string) => request<Reveal>(`/worlds/${id}/challenge/${challengeId}/reveal`, { method: "POST" }),
  metrics: (id: string) => request<{ ledger: Ledger; evaluation: Record<string, unknown> }>(`/worlds/${id}/metrics`),
  notebook: (id: string) => request<NotebookEvent[]>(`/worlds/${id}/notebook`),
};
