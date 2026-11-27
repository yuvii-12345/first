import type { ScenarioId, ScenarioOptions } from "../scenarios";

export type TamperTarget =
  | "eval-aBar"
  | "eval-zOmegaBar"
  | "eval-qMBar"
  | "eval-t1Bar"
  | "commitment-z"
  | "commitment-wZeta"
  | "public-value"
  | "none";

export interface RunRequest {
  kind: "run";
  id: number;
  scenario: ScenarioId;
  opts: ScenarioOptions;
}
export interface TamperRequest {
  kind: "tamper";
  id: number;
  runId: number;
  target: TamperTarget;
}
export type WorkerRequest = RunRequest | TamperRequest;

export interface CheckSummary {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
}

export interface VerificationSummary {
  ok: boolean;
  checks: CheckSummary[];
  challenges: { name: string; value: string }[];
  timeMs: number;
}

export interface TraceEntryView {
  label: string;
  value: string;
  kind?: string;
}
export interface TraceStepView {
  id: string;
  title: string;
  subtitle?: string;
  formula?: string;
  timeMs: number;
  entries: TraceEntryView[];
}

export interface RunResult {
  runId: number;
  scenario: ScenarioId;
  ok: boolean;
  statement: string;
  publicLabels: string[];
  publicValues: string[];
  circuit: {
    numGates: number;
    n: number;
    m: number;
    numVars: number;
    numSecrets: number;
    numPublic: number;
    srsPoints: number;
    vkBytes: number;
    proofBytes: number;
    degree: number;
  };
  audit: { violations: number; rows: number[]; totalRows: number };
  timings: { setup: number; prove: number; verify: number; total: number; steps: { id: string; title: string; timeMs: number }[] };
  verification: VerificationSummary;
  trace: TraceStepView[];
  proofHex: string;
  proofDigest: string;
  meta: {
    notes: string[];
    unsound?: boolean;
    grid?: { puzzle: number[][]; solution: number[][]; size: number };
    dslSource?: string;
    dslError?: string;
  };
  challenges: { name: string; value: string }[];
}

export type WorkerResponse =
  | { kind: "progress"; id: number; phase: string; detail?: string; stepId?: string; doneSteps?: string[] }
  | { kind: "done"; id: number; result: RunResult }
  | { kind: "tampered"; id: number; runId: number; target: TamperTarget; result: VerificationSummary }
  | { kind: "error"; id: number; message: string };
