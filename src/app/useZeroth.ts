import { useCallback, useEffect, useRef, useState } from "react";
import type { ScenarioId, ScenarioOptions } from "../scenarios";
import type { RunResult, TamperTarget, VerificationSummary, WorkerRequest, WorkerResponse } from "./protocol";

export interface ProgressItem {
  phase: string;
  detail?: string;
  stepId?: string;
}

export interface ZerothState {
  phase: "idle" | "running" | "done" | "error";
  progress: ProgressItem[];
  result: RunResult | null;
  error: string | null;
  tamper: { target: TamperTarget; result: VerificationSummary } | null;
  tampering: boolean;
}

const EMPTY: ZerothState = {
  phase: "idle",
  progress: [],
  result: null,
  error: null,
  tamper: null,
  tampering: false,
};

export function useZeroth() {
  const workerRef = useRef<Worker | null>(null);
  const [state, setState] = useState<ZerothState>(EMPTY);
  const current = useRef(0);

  useEffect(() => {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
      const msg = ev.data;
      if (msg.kind === "progress") {
        if (msg.id !== current.current) return;
        setState((s) => ({ ...s, progress: [...s.progress, { phase: msg.phase, detail: msg.detail, stepId: msg.stepId }] }));
      } else if (msg.kind === "done") {
        if (msg.id !== current.current) return;
        setState((s) => ({ ...s, phase: "done", result: msg.result }));
      } else if (msg.kind === "tampered") {
        setState((s) => ({ ...s, tampering: false, tamper: { target: msg.target, result: msg.result } }));
      } else if (msg.kind === "error") {
        if (msg.id !== current.current) return;
        setState((s) => ({ ...s, phase: "error", error: msg.message }));
      }
    };
    worker.onerror = (e) => {
      setState((s) => ({ ...s, phase: "error", error: e.message || "worker crashed" }));
    };
    return () => worker.terminate();
  }, []);

  const run = useCallback((scenario: ScenarioId, opts: ScenarioOptions) => {
    const id = ++current.current;
    setState({ ...EMPTY, phase: "running" });
    const req: WorkerRequest = { kind: "run", id, scenario, opts };
    workerRef.current?.postMessage(req);
  }, []);

  const tamper = useCallback((target: TamperTarget) => {
    const result = state.result;
    if (!result) return;
    setState((s) => ({ ...s, tampering: true }));
    const req: WorkerRequest = { kind: "tamper", id: ++current.current, runId: result.runId, target };
    workerRef.current?.postMessage(req);
  }, [state.result]);

  const reset = useCallback(() => setState(EMPTY), []);

  return { state, run, tamper, reset };
}
