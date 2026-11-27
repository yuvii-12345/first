/**
 * PLONK over BN254 with KZG commitments and a Fiat–Shamir transcript.
 *
 * ── The identity ──────────────────────────────────────────────────────────
 * For every row i of the circuit table (a point x_i = w^i of the domain H):
 *
 *   gate:     qM a b + qL a + qR b + qO c + qC            = 0
 *   copy:     (a + β·x_i + γ)(b + β·k1·x_i + γ)(c + β·k2·x_i + γ)·z(x_i)
 *           = (a + β·s1(x_i) + γ)(b + β·s2(x_i) + γ)(c + β·s3(x_i) + γ)·z(w·x_i)
 *   boundary: (z(x) − 1)·L1(x) = 0
 *   public:   PI(x) = Σ_j −v_j·L_row_j(x)
 *
 * Packed with random challenges α (so a cheating prover cannot trade one
 * violation for another), the whole thing must vanish on H, i.e.
 *
 *   N(X) = gate(X) + α·perm(X) + α²·(boundary(X) + PI(X))
 *   t(X) = N(X) / Z_H(X)          with Z_H(X) = X^n − 1
 *
 * The prover commits to t (split in three pieces, each of degree ≤ n+1, so the
 * SRS only needs n+2 powers) and opens everything at a random point ζ.
 *
 * ── Why the verifier's arithmetic is enough ───────────────────────────────
 * The verifier cannot evaluate N(ζ) — it does not know the wires. So the
 * prover sends the evaluations, KZG *binds* those evaluations to the
 * commitments, and the verifier checks the scalar identity N(ζ) = Z_H(ζ)·t(ζ).
 * If N − Z_H·t were a nonzero polynomial of degree ≤ 3n+5, a random ζ would
 * catch it except with probability (3n+5)/r.
 */
import { FrField, FqField } from "../crypto/field";
import { nextPowerOfTwo } from "../crypto/ntt";
import {
  type Aff,
  G1_GEN,
  fp2Ops,
  g1Add,
  g1Mul,
  g1ToBytes,
  g1FromBytes,
  isOnCurveG1,
  bytesToHex,
  toAffine,
  fromAffine,
  add as addJac,
  mul as mulJac,
} from "../crypto/curve";
import { pairingCheck } from "../crypto/pairing";
import {
  degree,
  divideByLinear,
  evaluate,
  evaluateOnCoset,
  interpolate,
  lagrangeEval,
  mul as polyMul,
  padTo,
  rescaleByOmega,
  scaleAdd,
  sub,
  trim,
  vanishingEval,
  type Poly,
} from "../crypto/poly";
import { cosetIntt, rootOfUnity } from "../crypto/ntt";
import { commit, setup as kzgSetup, type SRS } from "../crypto/kzg";
import { sha256, concatBytes } from "../crypto/sha256";
import { Transcript } from "../crypto/transcript";
import { type CompiledCircuit, selectorsAsPolys } from "./circuit";
import { Tracer, type TraceEntry, fullHex, shortHex } from "./trace";

export interface ProvingKey {
  circuit: CompiledCircuit;
  polys: {
    qM: Poly;
    qL: Poly;
    qR: Poly;
    qO: Poly;
    qC: Poly;
    s1: Poly;
    s2: Poly;
    s3: Poly;
  };
  commits: {
    qM: Aff<bigint>;
    qL: Aff<bigint>;
    qR: Aff<bigint>;
    qO: Aff<bigint>;
    qC: Aff<bigint>;
    s1: Aff<bigint>;
    s2: Aff<bigint>;
    s3: Aff<bigint>;
  };
  srs: SRS;
  vkDigest: string;
  /** Circuit-only evaluations on the quotient coset — reused by every proof. */
  coset: {
    m: number;
    shift: bigint;
    qM: bigint[];
    qL: bigint[];
    qR: bigint[];
    qO: bigint[];
    qC: bigint[];
    s1: bigint[];
    s2: bigint[];
    s3: bigint[];
    l1: bigint[];
    zh: bigint[];
  };
}

export interface VerifyingKey {
  n: number;
  omega: bigint;
  k1: bigint;
  k2: bigint;
  commits: ProvingKey["commits"];
  g2One: Aff<import("../crypto/field").Fp2>;
  g2Tau: Aff<import("../crypto/field").Fp2>;
  publicRows: number[];
  vkDigest: string;
}

export interface Evaluations {
  aBar: bigint;
  bBar: bigint;
  cBar: bigint;
  s1Bar: bigint;
  s2Bar: bigint;
  s3Bar: bigint;
  zBar: bigint;
  zOmegaBar: bigint;
  t0Bar: bigint;
  t1Bar: bigint;
  t2Bar: bigint;
  qMBar: bigint;
  qLBar: bigint;
  qRBar: bigint;
  qOBar: bigint;
  qCBar: bigint;
}

export interface Proof {
  a: Aff<bigint>;
  b: Aff<bigint>;
  c: Aff<bigint>;
  z: Aff<bigint>;
  t0: Aff<bigint>;
  t1: Aff<bigint>;
  t2: Aff<bigint>;
  wZeta: Aff<bigint>;
  wZetaOmega: Aff<bigint>;
  evals: Evaluations;
}

export interface VerifyCheck {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
}

export interface VerifyResult {
  ok: boolean;
  checks: VerifyCheck[];
  challenges: { beta: bigint; gamma: bigint; alpha: bigint; zeta: bigint; v: bigint };
  derived: {
    zh: bigint;
    l1: bigint;
    pi: bigint;
    numerator: bigint;
    tBar: bigint;
    rhs: bigint;
  };
  timeMs: number;
}

const COSET_CANDIDATES = [5n, 6n, 8n, 9n, 10n, 11n, 12n, 14n];

function pickCosetShift(m: number): bigint {
  // The coset must avoid *all* m-th roots of unity, otherwise Z_H (or the
  // L1 denominator) vanishes at some coset point and the division blows up.
  for (const c of COSET_CANDIDATES) {
    if (FrField.pow(c, BigInt(m)) !== 1n) return c;
  }
  throw new Error("no suitable coset shift");
}

function blind(poly: Poly, count: number, n: number): Poly {
  const out = padTo(poly, n + count);
  for (let k = 0; k < count; k++) {
    const r = FrField.random();
    // add r * X^k * Z_H(X) = r*X^(n+k) - r*X^k : invisible on H
    out[n + k] = FrField.add(out[n + k], r);
    out[k] = FrField.sub(out[k], r);
  }
  return out;
}

/** g1 points from a batch of commitments, for hashing into the transcript. */
function digestPoints(points: Aff<bigint>[], extra: string): string {
  const parts = points.map((p) => g1ToBytes(p));
  const h = sha256(concatBytes(...parts, new TextEncoder().encode(extra)));
  return bytesToHex(h);
}

export function setup(circuit: CompiledCircuit): ProvingKey {
  const polys = selectorsAsPolys(circuit);
  const srs = kzgSetup(circuit.n + 8);
  const commits = {
    qM: commit(srs, polys.qM),
    qL: commit(srs, polys.qL),
    qR: commit(srs, polys.qR),
    qO: commit(srs, polys.qO),
    qC: commit(srs, polys.qC),
    s1: commit(srs, polys.s1),
    s2: commit(srs, polys.s2),
    s3: commit(srs, polys.s3),
  };
  const vkDigest = digestPoints(Object.values(commits), `n=${circuit.n}`);

  // Precompute everything about the *circuit* that the quotient needs on the
  // coset. This is circuit-only data, so it belongs to the proving key and is
  // paid once rather than once per proof.
  const m = nextPowerOfTwo(4 * circuit.n + 6);
  const shift = pickCosetShift(m);
  const wm = requireOmega(m);
  const ratio = m / circuit.n;
  const wmN = FrField.pow(wm, BigInt(circuit.n)); // primitive `ratio`-th root
  const gPowN = FrField.pow(shift, BigInt(circuit.n));

  // Z_H on the coset is periodic with period m/n, so only `ratio` values.
  const zhPeriod = new Array<bigint>(ratio);
  {
    let acc = 1n;
    for (let j = 0; j < ratio; j++) {
      zhPeriod[j] = FrField.sub(FrField.mul(gPowN, acc), 1n);
      acc = FrField.mul(acc, wmN);
    }
  }
  const zh = new Array<bigint>(m);
  const xs = new Array<bigint>(m);
  {
    let x = shift;
    for (let i = 0; i < m; i++) {
      zh[i] = zhPeriod[i % ratio];
      xs[i] = x;
      x = FrField.mul(x, wm);
    }
  }
  // L1(x) = (x^n - 1) / (n (x - 1)); batch-invert the denominators.
  const l1 = batchL1(xs, zh, circuit.n);
  const coset = {
    m,
    shift,
    qM: evaluateOnCoset(polys.qM, m, shift),
    qL: evaluateOnCoset(polys.qL, m, shift),
    qR: evaluateOnCoset(polys.qR, m, shift),
    qO: evaluateOnCoset(polys.qO, m, shift),
    qC: evaluateOnCoset(polys.qC, m, shift),
    s1: evaluateOnCoset(polys.s1, m, shift),
    s2: evaluateOnCoset(polys.s2, m, shift),
    s3: evaluateOnCoset(polys.s3, m, shift),
    l1,
    zh,
  };
  return { circuit, polys, commits, srs, vkDigest, coset };
}

export function verifyingKey(pk: ProvingKey): VerifyingKey {
  return {
    n: pk.circuit.n,
    omega: pk.circuit.omega,
    k1: pk.circuit.k1,
    k2: pk.circuit.k2,
    commits: pk.commits,
    g2One: pk.srs.g2One,
    g2Tau: pk.srs.g2Tau,
    publicRows: pk.circuit.publicInputs.map((p) => p.row),
    vkDigest: pk.vkDigest,
  };
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

export interface ProveResult {
  proof: Proof;
  trace: ReturnType<Tracer["steps"]["slice"]>;
  timings: Record<string, number>;
  domain: { n: number; m: number; cosetShift: bigint; numGates: number };
  intermediates: {
    wireValues: bigint[][];
    zValues: bigint[];
    quotient: Poly;
    tPieces: [Poly, Poly, Poly];
    publicValues: bigint[];
    challenges: { beta: bigint; gamma: bigint; alpha: bigint; zeta: bigint; v: bigint };
  };
}

export function prove(pk: ProvingKey, opts: { trace?: boolean; onStep?: (id: string, title: string) => void } = {}): ProveResult {
  const tracer = new Tracer();
  const wantTrace = opts.trace !== false;
  const n = pk.circuit.n;
  const omega = pk.circuit.omega;
  const k1 = pk.circuit.k1;
  const k2 = pk.circuit.k2;
  const values = pk.circuit.variables.map((v) => FrField.reduce(v.value));
  const publicValues = pk.circuit.publicInputs.map((p) => p.value);

  const record = (id: string, title: string, meta: { subtitle?: string; formula?: string }, fn: () => { result: any; entries: TraceEntry[] }) => {
    opts.onStep?.(id, title);
    if (!wantTrace) {
      // Still need the side effects; skip the bookkeeping.
      return fn().result;
    }
    return tracer.step(id, title, meta, fn);
  };

  const tr = new Transcript("plonk/plonk");
  tr.absorbString("vk", pk.vkDigest);
  tr.absorbFields("public", publicValues);

  // -- 1. wire polynomials ------------------------------------------------
  const aVals = pk.circuit.wireIds[0].map((id) => values[id]);
  const bVals = pk.circuit.wireIds[1].map((id) => values[id]);
  const cVals = pk.circuit.wireIds[2].map((id) => values[id]);

  const { a, b, c, aComm, bComm, cComm } = record(
    "wires",
    "Commit to the wire polynomials",
    {
      subtitle: "Interpolate a, b, c over the domain, blind with Z_H, commit.",
      formula: "a(X) = Σ a_i·L_i(X)  ⟶  [a(τ)]₁",
    },
    () => {
      const a = blind(interpolate(aVals), 2, n);
      const b = blind(interpolate(bVals), 2, n);
      const c = blind(interpolate(cVals), 2, n);
      const aComm = commit(pk.srs, a);
      const bComm = commit(pk.srs, b);
      const cComm = commit(pk.srs, c);
      tr.absorbPoint("a", g1ToBytes(aComm));
      tr.absorbPoint("b", g1ToBytes(bComm));
      tr.absorbPoint("c", g1ToBytes(cComm));
      return {
        result: { a, b, c, aComm, bComm, cComm },
        entries: [
          { label: "n (rows)", value: String(n), kind: "number" },
          { label: "distinct variables", value: String(pk.circuit.variables.length), kind: "number" },
          { label: "[a]₁", value: shortHexPoint(aComm), kind: "point" },
          { label: "[b]₁", value: shortHexPoint(bComm), kind: "point" },
          { label: "[c]₁", value: shortHexPoint(cComm), kind: "point" },
        ],
      };
    },
  );

  // -- 2. permutation challenges ------------------------------------------
  const beta = tr.challenge("beta");
  const gamma = tr.challenge("gamma");

  // -- 3. grand product ---------------------------------------------------
  const { zPoly, zComm, zValues } = record(
    "grand-product",
    "Grand product for the copy constraints",
    {
      subtitle: "z accumulates the ratio of the permutation's two sides; z(w^0)=1 and the product must wrap to 1.",
      formula: "z_{i+1} = z_i · Π(a_i+βx+γ) / Π(a_i+βσ(x)+γ)",
    },
    () => {
      const zVals = new Array<bigint>(n + 1);
      zVals[0] = 1n;
      for (let i = 0; i < n; i++) {
        const x = FrField.pow(omega, BigInt(i));
        const x1 = FrField.mul(k1, x);
        const x2 = FrField.mul(k2, x);
        const num = FrField.mul(
          FrField.mul(FrField.add(FrField.add(aVals[i], FrField.mul(beta, x)), gamma), FrField.add(FrField.add(bVals[i], FrField.mul(beta, x1)), gamma)),
          FrField.add(FrField.add(cVals[i], FrField.mul(beta, x2)), gamma),
        );
        const s1 = pk.circuit.sigma.s1[i];
        const s2 = pk.circuit.sigma.s2[i];
        const s3 = pk.circuit.sigma.s3[i];
        const den = FrField.mul(
          FrField.mul(FrField.add(FrField.add(aVals[i], FrField.mul(beta, s1)), gamma), FrField.add(FrField.add(bVals[i], FrField.mul(beta, s2)), gamma)),
          FrField.add(FrField.add(cVals[i], FrField.mul(beta, s3)), gamma),
        );
        if (den === 0n) throw new Error("permutation denominator vanished (try again)");
        zVals[i + 1] = FrField.mul(zVals[i], FrField.mul(num, FrField.inv(den)));
      }
      if (zVals[n] !== 1n) throw new Error("copy-constraint product did not close to 1 — the witness violates wiring");
      const zPoly = blind(interpolate(zVals.slice(0, n)), 3, n);
      const zComm = commit(pk.srs, zPoly);
      tr.absorbPoint("z", g1ToBytes(zComm));
      return {
        result: { zPoly, zComm, zValues: zVals },
        entries: [
          { label: "β", value: shortHex(beta), kind: "field" },
          { label: "γ", value: shortHex(gamma), kind: "field" },
          { label: "z(w⁰)", value: "1", kind: "number" },
          { label: "z(wⁿ) (must be 1)", value: String(zVals[n]), kind: "number" },
          { label: "[z]₁", value: shortHexPoint(zComm), kind: "point" },
        ],
      };
    },
  );

  const alpha = tr.challenge("alpha");

  // -- 4. quotient --------------------------------------------------------
  const m = pk.coset.m;
  const cosetShift = pk.coset.shift;

  const quotientStep = record(
    "quotient",
    "Divide by the vanishing polynomial",
    {
      subtitle: "Evaluate the whole identity on a coset of size 8n, where Z_H is invertible, then interpolate.",
      formula: "t(X) = N(X) / (Xⁿ − 1)",
    },
    () => {
      const qE = { qM: pk.coset.qM, qL: pk.coset.qL, qR: pk.coset.qR, qO: pk.coset.qO, qC: pk.coset.qC };
      const aE = evaluateOnCoset(a, m, cosetShift);
      const bE = evaluateOnCoset(b, m, cosetShift);
      const cE = evaluateOnCoset(c, m, cosetShift);
      const zE = evaluateOnCoset(zPoly, m, cosetShift);
      const zShiftE = evaluateOnCoset(rescaleByOmega(zPoly, omega), m, cosetShift);
      const s1E = pk.coset.s1;
      const s2E = pk.coset.s2;
      const s3E = pk.coset.s3;

      // PI(X) = Σ −v_j L_row_j(X)
      const piVals = new Array<bigint>(n).fill(0n);
      for (const p of pk.circuit.publicInputs) piVals[p.row] = FrField.neg(p.value);
      const piE = evaluateOnCoset(interpolate(piVals), m, cosetShift);
      const l1E = pk.coset.l1;
      const zhValues = pk.coset.zh;
      const wm = requireOmega(m);

      const tEvals = new Array<bigint>(m);
      const alpha2 = FrField.mul(alpha, alpha);
      for (let i = 0; i < m; i++) {
        const x = FrField.mul(cosetShift, FrField.pow(wm, BigInt(i)));
        const x1 = FrField.mul(k1, x);
        const x2 = FrField.mul(k2, x);
        const gate = FrField.add(
          FrField.add(FrField.mul(FrField.mul(qE.qM[i], aE[i]), bE[i]), FrField.mul(qE.qL[i], aE[i])),
          FrField.add(FrField.add(FrField.mul(qE.qR[i], bE[i]), FrField.mul(qE.qO[i], cE[i])), qE.qC[i]),
        );
        const num = FrField.mul(
          FrField.mul(FrField.add(FrField.add(aE[i], FrField.mul(beta, x)), gamma), FrField.add(FrField.add(bE[i], FrField.mul(beta, x1)), gamma)),
          FrField.add(FrField.add(cE[i], FrField.mul(beta, x2)), gamma),
        );
        const den = FrField.mul(
          FrField.mul(FrField.add(FrField.add(aE[i], FrField.mul(beta, s1E[i])), gamma), FrField.add(FrField.add(bE[i], FrField.mul(beta, s2E[i])), gamma)),
          FrField.add(FrField.add(cE[i], FrField.mul(beta, s3E[i])), gamma),
        );
        const perm = FrField.sub(FrField.mul(num, zE[i]), FrField.mul(den, zShiftE[i]));
        // Public inputs belong to the *gate* equation (qL·a + … + PI = 0).
        const boundary = FrField.mul(FrField.sub(zE[i], 1n), l1E[i]);
        const N = FrField.add(
          FrField.add(gate, piE[i]),
          FrField.add(FrField.mul(alpha, perm), FrField.mul(alpha2, boundary)),
        );
        tEvals[i] = FrField.mul(N, FrField.inv(zhValues[i]));
      }
      const tFull = cosetIntt(tEvals, cosetShift);
      // If N is not divisible by Z_H the tail will be nonzero — a loud, early
      // failure is much better than a proof that silently fails to verify.
      for (let i = 3 * n + 6; i < tFull.length; i++) {
        if (tFull[i] !== 0n) {
          throw new Error(
            `quotient has degree ${i} > 3n+5: the constraint identity does not vanish on H (row ${i}?)`,
          );
        }
      }
      const tPoly = trim(tFull.slice(0, 3 * n + 6));
      if (tPoly.length > 3 * n + 6) throw new Error("quotient too large");

      const shift0 = n + 2;
      const shift1 = 2 * n + 4;
      const t0 = padTo(tPoly.slice(0, shift0), shift0);
      const t1 = padTo(tPoly.slice(shift0, shift1), shift0);
      const t2 = padTo(tPoly.slice(shift1, 3 * n + 6), shift0);

      const t0Comm = commit(pk.srs, t0);
      const t1Comm = commit(pk.srs, t1);
      const t2Comm = commit(pk.srs, t2);
      tr.absorbPoint("t0", g1ToBytes(t0Comm));
      tr.absorbPoint("t1", g1ToBytes(t1Comm));
      tr.absorbPoint("t2", g1ToBytes(t2Comm));

      return {
        result: { tPoly, t0, t1, t2, t0Comm, t1Comm, t2Comm },
        entries: [
          { label: "coset size m", value: String(m), kind: "number" },
          { label: "coset shift g", value: String(cosetShift), kind: "number" },
          { label: "deg t", value: String(degree(tPoly)), kind: "number" },
          { label: "α", value: shortHex(alpha), kind: "field" },
          { label: "[t₀]₁", value: shortHexPoint(t0Comm), kind: "point" },
          { label: "[t₁]₁", value: shortHexPoint(t1Comm), kind: "point" },
          { label: "[t₂]₁", value: shortHexPoint(t2Comm), kind: "point" },
        ],
      };
    },
  );

  const { tPoly, t0, t1, t2, t0Comm, t1Comm, t2Comm }: any = quotientStep;

  // -- 5. evaluate at ζ ---------------------------------------------------
  const zeta = tr.challenge("zeta");

  const evalsStep = record(
    "evaluations",
    "Evaluate every polynomial at ζ",
    {
      subtitle: "These are the only numbers the verifier will see; KZG ties each one to its commitment.",
      formula: "ā = a(ζ), z̄ω = z(ζω), …",
    },
    () => {
      const evals: Evaluations = {
        aBar: evaluate(a, zeta),
        bBar: evaluate(b, zeta),
        cBar: evaluate(c, zeta),
        s1Bar: evaluate(pk.polys.s1, zeta),
        s2Bar: evaluate(pk.polys.s2, zeta),
        s3Bar: evaluate(pk.polys.s3, zeta),
        zBar: evaluate(zPoly, zeta),
        zOmegaBar: evaluate(zPoly, FrField.mul(zeta, omega)),
        t0Bar: evaluate(t0, zeta),
        t1Bar: evaluate(t1, zeta),
        t2Bar: evaluate(t2, zeta),
        qMBar: evaluate(pk.polys.qM, zeta),
        qLBar: evaluate(pk.polys.qL, zeta),
        qRBar: evaluate(pk.polys.qR, zeta),
        qOBar: evaluate(pk.polys.qO, zeta),
        qCBar: evaluate(pk.polys.qC, zeta),
      };
      tr.absorbFields("evals", [
        evals.aBar,
        evals.bBar,
        evals.cBar,
        evals.s1Bar,
        evals.s2Bar,
        evals.s3Bar,
        evals.zBar,
        evals.zOmegaBar,
        evals.t0Bar,
        evals.t1Bar,
        evals.t2Bar,
        evals.qMBar,
        evals.qLBar,
        evals.qRBar,
        evals.qOBar,
        evals.qCBar,
      ]);
      return {
        result: evals,
        entries: [
          { label: "ζ", value: shortHex(zeta), kind: "field" },
          { label: "ā = a(ζ)", value: shortHex(evals.aBar), kind: "field" },
          { label: "b̄ = b(ζ)", value: shortHex(evals.bBar), kind: "field" },
          { label: "c̄ = c(ζ)", value: shortHex(evals.cBar), kind: "field" },
          { label: "z̄ = z(ζ)", value: shortHex(evals.zBar), kind: "field" },
          { label: "z̄ω = z(ζω)", value: shortHex(evals.zOmegaBar), kind: "field" },
          { label: "t̄₀, t̄₁, t̄₂", value: `${shortHex(evals.t0Bar, 6)} ${shortHex(evals.t1Bar, 6)} ${shortHex(evals.t2Bar, 6)}`, kind: "text" },
        ],
      };
    },
  );
  const evals: Evaluations = evalsStep;

  // -- 6. opening proofs --------------------------------------------------
  const v = tr.challenge("v");

  const openingStep = record(
    "openings",
    "Produce the KZG opening proofs",
    {
      subtitle: "One batched proof at ζ for fourteen polynomials, plus one at ζω for z.",
      formula: "W_ζ = (Σ vⁱfᵢ − Σ vⁱfᵢ(ζ)) / (X − ζ)",
    },
    () => {
      const shiftedZeta = FrField.mul(zeta, omega);
      const batchPolys = [a, b, c, pk.polys.s1, pk.polys.s2, pk.polys.s3, pk.polys.qM, pk.polys.qL, pk.polys.qR, pk.polys.qO, pk.polys.qC, t0, t1, t2, zPoly];
      const batchValues = [
        evals.aBar,
        evals.bBar,
        evals.cBar,
        evals.s1Bar,
        evals.s2Bar,
        evals.s3Bar,
        evals.qMBar,
        evals.qLBar,
        evals.qRBar,
        evals.qOBar,
        evals.qCBar,
        evals.t0Bar,
        evals.t1Bar,
        evals.t2Bar,
        evals.zBar,
      ];
      let combined: Poly = [];
      let combinedValue = 0n;
      let vp = 1n;
      for (let i = 0; i < batchPolys.length; i++) {
        combined = scaleAdd(combined, batchPolys[i], vp);
        combinedValue = FrField.add(combinedValue, FrField.mul(batchValues[i], vp));
        vp = FrField.mul(vp, v);
      }
      const wZeta = commit(pk.srs, divideByLinear(sub(combined, [combinedValue]), zeta));
      tr.absorbPoint("wz", g1ToBytes(wZeta));
      const wZetaOmega = commit(pk.srs, divideByLinear(sub(zPoly, [evals.zOmegaBar]), shiftedZeta));
      tr.absorbPoint("wzw", g1ToBytes(wZetaOmega));
      return {
        result: { wZeta, wZetaOmega, combined, combinedValue },
        entries: [
          { label: "v (batch challenge)", value: shortHex(v), kind: "field" },
          { label: "[W_ζ]₁", value: shortHexPoint(wZeta), kind: "point" },
          { label: "[W_ζω]₁", value: shortHexPoint(wZetaOmega), kind: "point" },
        ],
      };
    },
  );

  const proof: Proof = {
    a: aComm,
    b: bComm,
    c: cComm,
    z: zComm,
    t0: t0Comm,
    t1: t1Comm,
    t2: t2Comm,
    wZeta: openingStep.wZeta,
    wZetaOmega: openingStep.wZetaOmega,
    evals,
  };

  const timings: Record<string, number> = {};
  for (const s of tracer.steps) timings[s.id] = s.timeMs;

  return {
    proof,
    trace: tracer.steps,
    timings,
    domain: { n, m, cosetShift, numGates: pk.circuit.numGates },
    intermediates: {
      wireValues: [aVals, bVals, cVals],
      zValues,
      quotient: tPoly,
      tPieces: [t0, t1, t2],
      publicValues,
      challenges: { beta, gamma, alpha, zeta, v },
    },
  };
}

const requireOmega = rootOfUnity;

/** L1 evaluated on the coset, with a single field inversion for the whole batch. */
function batchL1(xs: bigint[], zh: bigint[], n: number): bigint[] {
  const m = xs.length;
  const denoms = xs.map((x) => FrField.sub(x, 1n));
  const prefix = new Array<bigint>(m);
  let acc = 1n;
  for (let i = 0; i < m; i++) {
    prefix[i] = acc;
    acc = FrField.mul(acc, denoms[i]);
  }
  let invAll = FrField.inv(acc);
  const out = new Array<bigint>(m);
  const nInv = FrField.inv(BigInt(n));
  for (let i = m - 1; i >= 0; i--) {
    const denomInv = FrField.mul(invAll, prefix[i]);
    invAll = FrField.mul(invAll, denoms[i]);
    out[i] = FrField.mul(nInv, FrField.mul(zh[i], denomInv));
  }
  return out;
}

/** L1(X) = (1/n)·(X^n − 1)/(X − 1) = (1/n)·Σ X^i */
export function lagrangeFirstPoly(n: number): Poly {
  const nInv = FrField.inv(BigInt(n));
  return new Array<bigint>(n).fill(nInv);
}

function shortHexPoint(p: Aff<bigint>): string {
  if (p === null) return "O (infinity)";
  return bytesToHex(g1ToBytes(p)).slice(0, 16) + "…";
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

export function verify(vk: VerifyingKey, publicValues: bigint[], proof: Proof): VerifyResult {
  const t0 = performance.now();
  const n = vk.n;
  const omega = vk.omega;
  const checks: VerifyCheck[] = [];

  if (publicValues.length !== vk.publicRows.length) {
    return {
      ok: false,
      checks: [
        {
          id: "arity",
          label: "Statement arity",
          ok: false,
          detail: `expected ${vk.publicRows.length} public inputs, got ${publicValues.length}`,
        },
      ],
      challenges: { beta: 0n, gamma: 0n, alpha: 0n, zeta: 0n, v: 0n },
      derived: { zh: 0n, l1: 0n, pi: 0n, numerator: 0n, tBar: 0n, rhs: 0n },
      timeMs: 0,
    };
  }

  // -- transcript replay ---------------------------------------------------
  const tr = new Transcript("plonk/plonk");
  tr.absorbString("vk", vk.vkDigest);
  tr.absorbFields("public", publicValues);
  tr.absorbPoint("a", g1ToBytes(proof.a));
  tr.absorbPoint("b", g1ToBytes(proof.b));
  tr.absorbPoint("c", g1ToBytes(proof.c));
  const beta = tr.challenge("beta");
  const gamma = tr.challenge("gamma");
  tr.absorbPoint("z", g1ToBytes(proof.z));
  const alpha = tr.challenge("alpha");
  tr.absorbPoint("t0", g1ToBytes(proof.t0));
  tr.absorbPoint("t1", g1ToBytes(proof.t1));
  tr.absorbPoint("t2", g1ToBytes(proof.t2));
  const zeta = tr.challenge("zeta");
  const e = proof.evals;
  tr.absorbFields("evals", [
    e.aBar,
    e.bBar,
    e.cBar,
    e.s1Bar,
    e.s2Bar,
    e.s3Bar,
    e.zBar,
    e.zOmegaBar,
    e.t0Bar,
    e.t1Bar,
    e.t2Bar,
    e.qMBar,
    e.qLBar,
    e.qRBar,
    e.qOBar,
    e.qCBar,
  ]);
  const v = tr.challenge("v");

  // -- 0. all commitments must be on the curve ----------------------------
  const points: Array<[string, Aff<bigint>]> = [
    ["[a]₁", proof.a],
    ["[b]₁", proof.b],
    ["[c]₁", proof.c],
    ["[z]₁", proof.z],
    ["[t₀]₁", proof.t0],
    ["[t₁]₁", proof.t1],
    ["[t₂]₁", proof.t2],
    ["[W_ζ]₁", proof.wZeta],
    ["[W_ζω]₁", proof.wZetaOmega],
  ];
  const allOnCurve = points.every(([, p]) => p === null || isOnCurveG1(p));
  checks.push({
    id: "curve",
    label: "Proof elements are valid curve points",
    ok: allOnCurve,
    detail: allOnCurve ? "9 group elements on y² = x³ + 3" : "at least one point is not on the curve",
  });

  // -- 1. public input polynomial -----------------------------------------
  const zh = vanishingEval(n, zeta);
  const l1 = lagrangeEval(0, n, omega, zeta);
  let pi = 0n;
  for (let j = 0; j < publicValues.length; j++) {
    const row = vk.publicRows[j];
    pi = FrField.sub(pi, FrField.mul(publicValues[j], lagrangeEval(row, n, omega, zeta)));
  }

  // -- 2. the identity, as scalars ----------------------------------------
  const zeta1 = FrField.mul(vk.k1, zeta);
  const zeta2 = FrField.mul(vk.k2, zeta);
  const A = FrField.add(FrField.add(e.aBar, FrField.mul(beta, zeta)), gamma);
  const B = FrField.add(FrField.add(e.bBar, FrField.mul(beta, zeta1)), gamma);
  const C = FrField.add(FrField.add(e.cBar, FrField.mul(beta, zeta2)), gamma);
  const A2 = FrField.add(FrField.add(e.aBar, FrField.mul(beta, e.s1Bar)), gamma);
  const B2 = FrField.add(FrField.add(e.bBar, FrField.mul(beta, e.s2Bar)), gamma);
  const C2 = FrField.add(FrField.add(e.cBar, FrField.mul(beta, e.s3Bar)), gamma);
  const perm = FrField.sub(
    FrField.mul(FrField.mul(FrField.mul(A, B), C), e.zBar),
    FrField.mul(FrField.mul(FrField.mul(A2, B2), C2), e.zOmegaBar),
  );
  const gate = FrField.add(
    FrField.add(FrField.mul(FrField.mul(e.qMBar, e.aBar), e.bBar), FrField.mul(e.qLBar, e.aBar)),
    FrField.add(FrField.add(FrField.mul(e.qRBar, e.bBar), FrField.mul(e.qOBar, e.cBar)), e.qCBar),
  );
  const boundary = FrField.mul(FrField.sub(e.zBar, 1n), l1);
  const alpha2 = FrField.mul(alpha, alpha);
  const numerator = FrField.add(
    FrField.add(gate, pi),
    FrField.add(FrField.mul(alpha, perm), FrField.mul(alpha2, boundary)),
  );

  const zetaShift0 = FrField.pow(zeta, BigInt(n + 2));
  const zetaShift1 = FrField.pow(zeta, BigInt(2 * n + 4));
  const tBar = FrField.add(e.t0Bar, FrField.add(FrField.mul(zetaShift0, e.t1Bar), FrField.mul(zetaShift1, e.t2Bar)));
  const rhs = FrField.mul(zh, tBar);

  const identityOk = numerator === rhs;
  checks.push({
    id: "identity",
    label: "Gate + permutation + boundary identity holds at ζ",
    ok: identityOk,
    detail: identityOk
      ? "N(ζ) = Z_H(ζ)·t(ζ)"
      : "N(ζ) ≠ Z_H(ζ)·t(ζ) — a constraint, a copy, or the quotient is wrong",
  });

  // -- 3. KZG openings ----------------------------------------------------
  const batchPolys = ["a", "b", "c", "s1", "s2", "s3", "qM", "qL", "qR", "qO", "qC", "t0", "t1", "t2", "z"] as const;
  const commits: Aff<bigint>[] = [
    proof.a,
    proof.b,
    proof.c,
    vk.commits.s1,
    vk.commits.s2,
    vk.commits.s3,
    vk.commits.qM,
    vk.commits.qL,
    vk.commits.qR,
    vk.commits.qO,
    vk.commits.qC,
    proof.t0,
    proof.t1,
    proof.t2,
    proof.z,
  ];
  const vals = [e.aBar, e.bBar, e.cBar, e.s1Bar, e.s2Bar, e.s3Bar, e.qMBar, e.qLBar, e.qRBar, e.qOBar, e.qCBar, e.t0Bar, e.t1Bar, e.t2Bar, e.zBar];

  let combined: Aff<bigint> = null;
  let combinedValue = 0n;
  let vp = 1n;
  for (let i = 0; i < commits.length; i++) {
    combined = g1Add(combined, g1Mul(commits[i], vp));
    combinedValue = FrField.add(combinedValue, FrField.mul(vals[i], vp));
    vp = FrField.mul(vp, v);
  }
  const lhsPoint = g1Add(combined, g1Mul(G1_GEN, FrField.neg(combinedValue)));
  const tauMinusZeta = addG2Aff(vk.g2Tau, g2MulAff(vk.g2One, FrField.neg(zeta)));
  const tauMinusZetaOmega = addG2Aff(vk.g2Tau, g2MulAff(vk.g2One, FrField.neg(FrField.mul(zeta, omega))));
  const zShifted = g1Add(proof.z, g1Mul(G1_GEN, FrField.neg(e.zOmegaBar)));

  const openingsOk = pairingCheck([
    [lhsPoint, vk.g2One],
    [negPoint(proof.wZeta), tauMinusZeta],
    [zShifted, vk.g2One],
    [negPoint(proof.wZetaOmega), tauMinusZetaOmega],
  ]);

  checks.push({
    id: "openings",
    label: "KZG openings are consistent with the commitments",
    ok: openingsOk,
    detail: openingsOk
      ? "4 pairings = 1 multiscalar mess, all in the exponent"
      : "a commitment does not open to the claimed evaluation at ζ",
  });

  const timeMs = performance.now() - t0;
  return {
    ok: checks.every((c) => c.ok),
    checks,
    challenges: { beta, gamma, alpha, zeta, v },
    derived: { zh, l1, pi, numerator, tBar, rhs },
    timeMs,
  };
}

function addG2Aff(a: Aff<import("../crypto/field").Fp2>, b: Aff<import("../crypto/field").Fp2>) {
  return toAffine(fp2Ops, addJac(fp2Ops, fromAffine(fp2Ops, a), fromAffine(fp2Ops, b)));
}
function g2MulAff(a: Aff<import("../crypto/field").Fp2>, k: bigint) {
  return toAffine(fp2Ops, mulJac(fp2Ops, fromAffine(fp2Ops, a), k));
}
function negPoint(p: Aff<bigint>): Aff<bigint> {
  return p === null ? null : { x: p.x, y: FqField.neg(p.y) };
}

// ---------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------

/** Fixed order used for the proof bytes and for hashing the proof. */
export const PROOF_POINTS = ["a", "b", "c", "z", "t0", "t1", "t2", "wZeta", "wZetaOmega"] as const;
export const PROOF_EVALS = [
  "aBar",
  "bBar",
  "cBar",
  "s1Bar",
  "s2Bar",
  "s3Bar",
  "zBar",
  "zOmegaBar",
  "t0Bar",
  "t1Bar",
  "t2Bar",
  "qMBar",
  "qLBar",
  "qRBar",
  "qOBar",
  "qCBar",
] as const;

/** 9 uncompressed G1 points + 16 field elements = 1088 bytes. */
export function serializeProof(proof: Proof): Uint8Array {
  const out = new Uint8Array(PROOF_POINTS.length * 64 + PROOF_EVALS.length * 32);
  let off = 0;
  for (const k of PROOF_POINTS) {
    out.set(g1ToBytes(proof[k]), off);
    off += 64;
  }
  for (const k of PROOF_EVALS) {
    out.set(FrField.toBytes(proof.evals[k]), off);
    off += 32;
  }
  return out;
}

export function proofHex(proof: Proof): string {
  return bytesToHex(serializeProof(proof));
}

export function deserializeProof(bytes: Uint8Array): Proof {
  const proof: any = {};
  let off = 0;
  for (const k of PROOF_POINTS) {
    proof[k] = g1FromBytes(bytes.subarray(off, off + 64));
    off += 64;
  }
  const evals: any = {};
  for (const k of PROOF_EVALS) {
    evals[k] = FrField.fromBytes(bytes.subarray(off, off + 32));
    off += 32;
  }
  proof.evals = evals;
  return proof as Proof;
}

export function verifyingKeySizeBytes(vk: VerifyingKey): number {
  return Object.keys(vk.commits).length * 64 + 2 * 128;
}
