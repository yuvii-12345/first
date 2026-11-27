/**
 * KZG polynomial commitments: the "hash function for polynomials".
 *
 * Setup samples a secret tau and publishes
 *      [tau^0]_1, [tau^1]_1, ..., [tau^d]_1   and   [1]_2, [tau]_2
 * The secret tau ("toxic waste") must be destroyed: whoever knows it can
 * open any commitment to any value.
 *
 *   commit(p)      = [p(tau)]_1 = sum p_i [tau^i]_1
 *   open(p, z)     = pi = [(p(X) - p(z)) / (X - z)]_1
 *   verify         = e(C - [p(z)]_1, [1]_2) == e(pi, [tau - z]_2)
 *
 * Because the pairing is bilinear, a valid opening forces the committed
 * polynomial to actually pass through (z, p(z)) — with at most one opening per
 * commitment unless you know tau.
 */
import { FrField, FqField, Fp2 } from "./field";
import {
  type Aff,
  bytesToHex,
  G2_GEN,
  g1Add,
  g1Mul,
  g1ToBytes,
  g2ToBytesImpl,
  msm,
  mulWindowed,
  fqOps,
  fp2Ops,
  add as addJac,
  mul as mulJac,
  fromAffine,
  toAffine,
} from "./curve";
import { pairingCheck } from "./pairing";
import { add, degree, divideByLinear, evaluate, isZero, scale, sub, type Poly } from "./poly";

export interface SRS {
  /** [tau^i]_1 for i = 0..maxDegree */
  g1Powers: Aff<bigint>[];
  /** [tau]_2 */
  g2Tau: Aff<Fp2>;
  /** [1]_2 (the G2 generator) */
  g2One: Aff<Fp2>;
  maxDegree: number;
  /** Whether tau was generated locally (and should be destroyed). */
  generated: boolean;
}

export interface KZGProof {
  value: bigint;
  proof: Aff<bigint>;
}

const ONE_G1: Aff<bigint> = { x: 1n, y: 2n };

/**
 * A structured reference string is *universal*: one toxic-waste ceremony serves
 * every circuit up to the maximum degree. We cache it per size, exactly like a
 * real deployment would reuse a single powers-of-tau file.
 */
const srsCache = new Map<number, SRS>();

export function setup(maxDegree: number, tau?: bigint): SRS {
  if (tau === undefined) {
    const cached = srsCache.get(maxDegree);
    if (cached) return cached;
  }
  const srs = buildSRS(maxDegree, tau);
  if (tau === undefined) srsCache.set(maxDegree, srs);
  return srs;
}

export function clearSRSCache(): void {
  srsCache.clear();
}

function buildSRS(maxDegree: number, tau?: bigint): SRS {
  const secret = tau ?? FrField.random();
  const g1Powers: Aff<bigint>[] = new Array(maxDegree + 1);
  let acc = ONE_G1;
  const tauPoint = fromAffine(fqOps, ONE_G1);
  for (let i = 0; i <= maxDegree; i++) {
    g1Powers[i] = acc;
    if (i < maxDegree) acc = toAffine(fqOps, mulWindowed(fqOps, fromAffine(fqOps, acc), secret));
  }
  void tauPoint;
  return {
    g1Powers,
    g2Tau: msmG2([G2_GEN], [secret]),
    g2One: G2_GEN,
    maxDegree,
    generated: tau === undefined,
  };
}

function msmG2(points: Aff<Fp2>[], scalars: bigint[]): Aff<Fp2> {
  // Small helper: only ever used with one or two points here.
  let acc: Aff<Fp2> = null;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (p === null) continue;
    const term = scalarMulG2(p, scalars[i]);
    acc = acc === null ? term : addG2(acc, term);
  }
  return acc;
}

function addG2(a: Aff<Fp2>, b: Aff<Fp2>): Aff<Fp2> {
  return toAffine(fp2Ops, addJac(fp2Ops, fromAffine(fp2Ops, a), fromAffine(fp2Ops, b)));
}
function scalarMulG2(p: Aff<Fp2>, k: bigint): Aff<Fp2> {
  return toAffine(fp2Ops, mulJac(fp2Ops, fromAffine(fp2Ops, p), k));
}

/** Commit to a polynomial: [p(tau)]_1. */
export function commit(srs: SRS, poly: Poly): Aff<bigint> {
  const d = degree(poly);
  if (d + 1 > srs.g1Powers.length) throw new Error(`polynomial degree ${d} exceeds SRS capacity ${srs.maxDegree}`);
  const n = poly.length;
  return toAffine(fqOps, msm(fqOps, srs.g1Powers.slice(0, n), poly));
}

/** Open a polynomial at a point: returns (p(z), [(p(X)-p(z))/(X-z)]_1). */
export function open(srs: SRS, poly: Poly, z: bigint): KZGProof {
  const value = evaluate(poly, z);
  const numerator = sub(poly, [value]);
  if (!isZero([evaluate(numerator, z)])) throw new Error("open: polynomial does not pass through z");
  const quotient = divideByLinear(numerator, z);
  return { value, proof: commit(srs, quotient) };
}

/** Verify a single opening. */
export function verify(srs: SRS, commitment: Aff<bigint>, z: bigint, proof: KZGProof): boolean {
  if (commitment === null || proof.proof === null) return false;
  const lhs = g1Add(commitment, g1Mul(ONE_G1, FrField.neg(proof.value)));
  const g2Shifted = addG2(srs.g2Tau, mulG2Point(srs.g2One, FrField.neg(z)));
  return pairingCheck([
    [lhs, srs.g2One],
    [negG1(proof.proof), g2Shifted],
  ]);
}

function mulG2Point(p: Aff<Fp2>, k: bigint): Aff<Fp2> {
  return scalarMulG2(p, k);
}
function negG1(p: Aff<bigint>): Aff<bigint> {
  return p === null ? null : { x: p.x, y: FqField.neg(p.y) };
}

/**
 * Batch opening of several polynomials at a single point, combined with
 * powers of a random challenge v (Fiat–Shamir or supplied).
 *
 *   combined = sum v^i f_i,  value = combined(z),  pi = [(combined - value)/(X-z)]_1
 */
export function openBatch(srs: SRS, polys: Poly[], z: bigint, v: bigint): { value: bigint; proof: Aff<bigint>; combined: Poly } {
  let combined: Poly = [];
  let vp = 1n;
  for (const p of polys) {
    combined = add(combined, scale(p, vp));
    vp = FrField.mul(vp, v);
  }
  const value = evaluate(combined, z);
  const quotient = divideByLinear(sub(combined, [value]), z);
  return { value, proof: commit(srs, quotient), combined };
}

/** Verify a batch opening given the commitments and their claimed values. */
export function verifyBatch(
  srs: SRS,
  commitments: Aff<bigint>[],
  values: bigint[],
  z: bigint,
  proof: Aff<bigint>,
  v: bigint,
): boolean {
  let c: Aff<bigint> = null;
  let val = 0n;
  let vp = 1n;
  for (let i = 0; i < commitments.length; i++) {
    c = g1Add(c, g1Mul(commitments[i], vp));
    val = FrField.add(val, FrField.mul(values[i], vp));
    vp = FrField.mul(vp, v);
  }
  const lhs = g1Add(c, g1Mul(ONE_G1, FrField.neg(val)));
  const g2Shifted = addG2(srs.g2Tau, mulG2Point(srs.g2One, FrField.neg(z)));
  return pairingCheck([
    [lhs, srs.g2One],
    [negG1(proof), g2Shifted],
  ]);
}

// Serialisation helpers used by the UI / proof envelope ---------------------

export function srsG1Size(srs: SRS): number {
  return srs.g1Powers.length * 64;
}

export function pointHex(p: Aff<bigint>): string {
  return "0x" + bytesToHex(g1ToBytes(p));
}
export function pointHexG2(p: Aff<Fp2>): string {
  return "0x" + bytesToHex(g2ToBytesImpl(p));
}
