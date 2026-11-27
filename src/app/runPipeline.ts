/**
 * The end-to-end "run": build → preprocess → prove → verify → summarise.
 * Kept out of the worker so it can be exercised directly in tests.
 */
import { buildScenario, type ScenarioId, type ScenarioOptions } from "../scenarios";
import { auditWitness } from "../plonk/circuit";
import {
  proofHex,
  prove,
  serializeProof,
  setup,
  verify,
  verifyingKey,
  type Proof,
  type ProvingKey,
  type VerifyingKey,
} from "../plonk/plonk";
import { FrField } from "../crypto/field";
import { sha256 } from "../crypto/sha256";
import { bytesToHex } from "../crypto/curve";
import type { RunResult, TamperTarget, VerificationSummary } from "./protocol";

export const hex = (v: bigint) => "0x" + v.toString(16);

export function summarise(vk: VerifyingKey, publicValues: bigint[], proof: Proof): VerificationSummary {
  const res = verify(vk, publicValues, proof);
  return {
    ok: res.ok,
    checks: res.checks,
    challenges: [
      { name: "β", value: hex(res.challenges.beta) },
      { name: "γ", value: hex(res.challenges.gamma) },
      { name: "α", value: hex(res.challenges.alpha) },
      { name: "ζ", value: hex(res.challenges.zeta) },
      { name: "v", value: hex(res.challenges.v) },
    ],
    timeMs: res.timeMs,
  };
}

export interface StoredRun {
  vk: VerifyingKey;
  proof: Proof;
  publicValues: bigint[];
}

export type ProgressFn = (phase: string, detail?: string, stepId?: string) => void;

/**
 * Preprocessing depends only on the circuit, so cache the proving key: proving
 * twice against the same circuit should not re-commit the selectors.
 */
const pkCache = new Map<string, ProvingKey>();
const MAX_CACHED_KEYS = 3;

function cacheKey(scenario: ScenarioId, opts: ScenarioOptions): string {
  return `${scenario}|${JSON.stringify(opts)}`;
}

export function clearProvingKeyCache(): void {
  pkCache.clear();
}

export function executeRun(
  runId: number,
  scenario: ScenarioId,
  opts: ScenarioOptions,
  onProgress: ProgressFn = () => {},
  onStore: (stored: StoredRun) => void = () => {},
): RunResult {
  const t0 = performance.now();
  onProgress("Arithmetization", "building the gate table and wiring");
  const built = buildScenario(scenario, opts);
  const circuit = built.circuit;
  const violations = auditWitness(circuit);

  const key = cacheKey(scenario, opts);
  const cached = pkCache.get(key);
  onProgress(
    "Preprocessing",
    cached
      ? "same circuit as last time — the proving key is cached"
      : circuit.n > 1024
        ? `building the ${circuit.n + 8} point reference string, then committing to the selectors (slow once per circuit size)`
        : "committing to the selectors and the permutation",
  );
  const setupStart = performance.now();
  let pk = cached;
  if (!pk) {
    pk = setup(circuit);
    if (pkCache.size >= MAX_CACHED_KEYS) {
      const oldest = pkCache.keys().next().value;
      if (oldest !== undefined) pkCache.delete(oldest);
    }
    pkCache.set(key, pk);
  }
  const setupMs = performance.now() - setupStart;

  const vk = verifyingKey(pk);
  const proveStart = performance.now();
  const { proof, trace } = prove(pk, {
    onStep: (stepId, title) => onProgress("Proving", title, stepId),
  });
  const proveMs = performance.now() - proveStart;

  onProgress("Verifying", "four pairings, one final exponentiation");
  const verification = summarise(vk, built.publicValues, proof);

  const bytes = serializeProof(proof);
  const proofDigest = bytesToHex(sha256(bytes)).slice(0, 32);

  onStore({ vk, proof, publicValues: built.publicValues });

  return {
    runId,
    scenario,
    ok: verification.ok,
    statement: built.meta.statement,
    publicLabels: built.meta.publicLabels,
    publicValues: built.publicValues.map((v) => v.toString()),
    circuit: {
      numGates: circuit.numGates,
      n: circuit.n,
      m: pk.coset.m,
      numVars: circuit.variables.length,
      numSecrets: circuit.variables.filter((v) => v.kind === "secret").length,
      numPublic: circuit.variables.filter((v) => v.kind === "public").length,
      srsPoints: pk.srs.g1Powers.length,
      vkBytes: 8 * 64 + 2 * 128,
      proofBytes: bytes.length,
      degree: circuit.numGates,
    },
    audit: { violations: violations.length, rows: violations.slice(0, 8).map((v) => v.row), totalRows: circuit.numGates },
    timings: {
      setup: setupMs,
      prove: proveMs,
      verify: verification.timeMs,
      total: performance.now() - t0,
      steps: trace.map((s) => ({ id: s.id, title: s.title, timeMs: s.timeMs })),
    },
    verification,
    trace: trace.map((s) => ({
      id: s.id,
      title: s.title,
      subtitle: s.subtitle,
      formula: s.formula,
      timeMs: s.timeMs,
      entries: s.entries.map((e) => ({ label: e.label, value: String(e.value), kind: e.kind })),
    })),
    proofHex: proofHex(proof),
    proofDigest,
    meta: {
      notes: built.meta.notes,
      unsound: built.meta.unsound,
      grid: built.meta.grid,
      dslSource: built.meta.dslSource,
      dslError: built.meta.dslError,
    },
    challenges: verification.challenges,
  };
}

/** Verify a deliberately corrupted copy of a stored proof. */
export function applyTamper(
  stored: StoredRun,
  target: TamperTarget,
  deserialize: (bytes: Uint8Array) => Proof,
): VerificationSummary {
  const bytes = serializeProof(stored.proof);
  const proof = deserialize(bytes);
  const publicValues = stored.publicValues.slice();

  switch (target) {
    case "eval-aBar":
      proof.evals.aBar = FrField.add(proof.evals.aBar, 1n);
      break;
    case "eval-zOmegaBar":
      proof.evals.zOmegaBar = FrField.add(proof.evals.zOmegaBar, 1n);
      break;
    case "eval-qMBar":
      proof.evals.qMBar = FrField.add(proof.evals.qMBar, 1n);
      break;
    case "eval-t1Bar":
      proof.evals.t1Bar = FrField.add(proof.evals.t1Bar, 1n);
      break;
    case "commitment-z":
      proof.z = proof.t0;
      break;
    case "commitment-wZeta":
      proof.wZeta = proof.wZetaOmega;
      break;
    case "public-value":
      if (publicValues.length > 0) publicValues[0] = FrField.add(publicValues[0], 1n);
      break;
    case "none":
      break;
  }
  return summarise(stored.vk, publicValues, proof);
}
