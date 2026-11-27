/** Lightweight instrumentation so the UI can show what the prover did, step by step. */

export type EntryKind = "hex" | "field" | "point" | "text" | "number";

export interface TraceEntry {
  label: string;
  value: string;
  kind?: EntryKind;
}

export interface TraceStep {
  id: string;
  title: string;
  subtitle?: string;
  formula?: string;
  timeMs: number;
  entries: TraceEntry[];
}

export class Tracer {
  readonly steps: TraceStep[] = [];

  /** Run `fn`, timing it, and record the entries it returns. */
  step<T>(
    id: string,
    title: string,
    meta: { subtitle?: string; formula?: string },
    fn: () => { result: T; entries: TraceEntry[] },
  ): T {
    const t0 = performance.now();
    const { result, entries } = fn();
    const timeMs = performance.now() - t0;
    this.steps.push({ id, title, subtitle: meta.subtitle, formula: meta.formula, timeMs, entries });
    return result;
  }

  pushRaw(step: Omit<TraceStep, "timeMs"> & { timeMs?: number }): void {
    this.steps.push({ ...step, timeMs: step.timeMs ?? 0 });
  }
}

export function shortHex(v: bigint, chars = 10): string {
  const h = v.toString(16).padStart(64, "0");
  return "0x" + h.slice(0, chars) + "…" + h.slice(-chars);
}

export function fullHex(v: bigint): string {
  return "0x" + v.toString(16).padStart(64, "0");
}
