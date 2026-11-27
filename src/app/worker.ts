/// <reference lib="webworker" />
/**
 * Proving happens off the main thread: a 9x9 Sudoku needs seconds of BigInt
 * arithmetic, and the UI should stay interactive (and show progress) while it
 * runs.
 */
import { deserializeProof } from "../plonk/plonk";
import { applyTamper, executeRun, type StoredRun } from "./runPipeline";
import type { WorkerRequest, WorkerResponse } from "./protocol";

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const runs = new Map<number, StoredRun>();

const post = (msg: WorkerResponse) => ctx.postMessage(msg);

ctx.onmessage = (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  try {
    if (req.kind === "run") {
      const result = executeRun(
        req.id,
        req.scenario,
        req.opts,
        (phase, detail, stepId) => post({ kind: "progress", id: req.id, phase, detail, stepId }),
        (stored) => runs.set(req.id, stored),
      );
      post({ kind: "done", id: req.id, result });
    } else {
      const stored = runs.get(req.runId);
      if (!stored) throw new Error("nothing to tamper with yet — run a proof first");
      const result = applyTamper(stored, req.target, deserializeProof);
      post({ kind: "tampered", id: req.id, runId: req.runId, target: req.target, result });
    }
  } catch (err) {
    post({ kind: "error", id: req.id, message: (err as Error).message ?? String(err) });
  }
};
