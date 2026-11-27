/**
 * Elliptic curves G1 (over Fq) and G2 (over Fp2) for BN254.
 *
 * Points are handled in Jacobian coordinates internally; every exported
 * operation is generic over the coordinate field, so the *same* addition
 * formulas drive both groups.
 */
import { Fp2, FqField, Q, R } from "./field";

export interface FieldOps<F> {
  zero: F;
  one: F;
  add(a: F, b: F): F;
  sub(a: F, b: F): F;
  mul(a: F, b: F): F;
  sqr(a: F): F;
  neg(a: F): F;
  inv(a: F): F;
  isZero(a: F): boolean;
  eq(a: F, b: F): boolean;
  fromBigint(v: bigint): F;
}

export const fqOps: FieldOps<bigint> = {
  zero: 0n,
  one: 1n,
  add: (a, b) => FqField.add(a, b),
  sub: (a, b) => FqField.sub(a, b),
  mul: (a, b) => FqField.mul(a, b),
  sqr: (a) => FqField.sqr(a),
  neg: (a) => FqField.neg(a),
  inv: (a) => FqField.inv(a),
  isZero: (a) => a === 0n,
  eq: (a, b) => a === b,
  fromBigint: (v) => FqField.reduce(v),
};

export const fp2Ops: FieldOps<Fp2> = {
  zero: Fp2.zero,
  one: Fp2.one,
  add: (a, b) => a.add(b),
  sub: (a, b) => a.sub(b),
  mul: (a, b) => a.mul(b),
  sqr: (a) => a.sqr(),
  neg: (a) => a.neg(),
  inv: (a) => a.inv(),
  isZero: (a) => a.isZero(),
  eq: (a, b) => a.eq(b),
  fromBigint: (v) => Fp2.from(FqField.reduce(v)),
};

/** Jacobian point. z === zero means the point at infinity. */
export interface Jac<F> {
  x: F;
  y: F;
  z: F;
}
/** Affine point; null is the point at infinity. */
export type Aff<F> = { x: F; y: F } | null;

export function infinity<F>(ops: FieldOps<F>): Jac<F> {
  return { x: ops.one, y: ops.one, z: ops.zero };
}
export function isInfinity<F>(ops: FieldOps<F>, p: Jac<F>): boolean {
  return ops.isZero(p.z);
}
export function toAffine<F>(ops: FieldOps<F>, p: Jac<F>): Aff<F> {
  if (ops.isZero(p.z)) return null;
  const zInv = ops.inv(p.z);
  const zInv2 = ops.sqr(zInv);
  return { x: ops.mul(p.x, zInv2), y: ops.mul(p.y, ops.mul(zInv2, zInv)) };
}
export function fromAffine<F>(ops: FieldOps<F>, p: Aff<F>): Jac<F> {
  if (p === null) return infinity(ops);
  return { x: p.x, y: p.y, z: ops.one };
}

/** Point doubling, specialised for curves with a = 0 (BN254 G1 and G2). */
export function dbl<F>(ops: FieldOps<F>, p: Jac<F>): Jac<F> {
  if (ops.isZero(p.z) || ops.isZero(p.y)) return infinity(ops);
  const a = ops.sqr(p.x); // A = X1^2
  const b = ops.sqr(p.y); // B = Y1^2
  const c = ops.sqr(b); // C = B^2
  const xb = ops.add(p.x, b);
  const d = ops.sub(ops.sub(ops.sqr(xb), a), c);
  const dd = ops.add(d, d); // D = 2*((X1+B)^2 - A - C)
  const e = ops.add(a, ops.add(a, a)); // E = 3*A
  const f = ops.sqr(e);
  const x3 = ops.sub(f, ops.add(dd, dd));
  const y3 = ops.sub(ops.mul(e, ops.sub(dd, x3)), ops.mul(c, ops.fromBigint(8n)));
  const z3 = ops.mul(ops.add(p.y, p.y), p.z);
  return { x: x3, y: y3, z: z3 };
}

/** Full Jacobian + Jacobian addition (add-2007-bl). */
export function add<F>(ops: FieldOps<F>, p: Jac<F>, q: Jac<F>): Jac<F> {
  if (ops.isZero(p.z)) return q;
  if (ops.isZero(q.z)) return p;
  const z1z1 = ops.sqr(p.z);
  const z2z2 = ops.sqr(q.z);
  const u1 = ops.mul(p.x, z2z2);
  const u2 = ops.mul(q.x, z1z1);
  const s1 = ops.mul(p.y, ops.mul(q.z, z2z2));
  const s2 = ops.mul(q.y, ops.mul(p.z, z1z1));
  const h = ops.sub(u2, u1);
  const r = ops.add(ops.sub(s2, s1), ops.sub(s2, s1));
  if (ops.isZero(h)) {
    if (ops.isZero(r)) return dbl(ops, p);
    return infinity(ops);
  }
  const i = ops.sqr(ops.add(h, h));
  const j = ops.mul(h, i);
  const v = ops.mul(u1, i);
  const x3 = ops.sub(ops.sub(ops.sqr(r), j), ops.add(v, v));
  const y3 = ops.sub(ops.mul(r, ops.sub(v, x3)), ops.mul(ops.add(s1, s1), j));
  const z3 = ops.mul(ops.sub(ops.sub(ops.sqr(ops.add(p.z, q.z)), z1z1), z2z2), h);
  return { x: x3, y: y3, z: z3 };
}

/** Mixed addition: Jacobian + affine (madd-2007-bl). */
export function addMixed<F>(ops: FieldOps<F>, p: Jac<F>, q: Aff<F>): Jac<F> {
  if (q === null) return p;
  if (ops.isZero(p.z)) return fromAffine(ops, q);
  const z1z1 = ops.sqr(p.z);
  const u2 = ops.mul(q.x, z1z1);
  const s2 = ops.mul(q.y, ops.mul(p.z, z1z1));
  const h = ops.sub(u2, p.x);
  const hh = ops.sqr(h);
  const i = ops.add(ops.add(hh, hh), ops.add(hh, hh));
  const j = ops.mul(h, i);
  const r = ops.add(ops.sub(s2, p.y), ops.sub(s2, p.y));
  const v = ops.mul(p.x, i);
  if (ops.isZero(h)) {
    if (ops.isZero(r)) return dbl(ops, p);
    return infinity(ops);
  }
  const x3 = ops.sub(ops.sub(ops.sqr(r), j), ops.add(v, v));
  const y3 = ops.sub(ops.mul(r, ops.sub(v, x3)), ops.mul(ops.add(p.y, p.y), j));
  const z3 = ops.sub(ops.sub(ops.sqr(ops.add(p.z, h)), z1z1), hh);
  return { x: x3, y: y3, z: z3 };
}

export function neg<F>(ops: FieldOps<F>, p: Jac<F>): Jac<F> {
  return { x: p.x, y: ops.neg(p.y), z: p.z };
}

/**
 * Windowed scalar multiplication (w = 4). ~2.5x faster than double-and-add.
 * Used for generating the structured reference string, where we multiply by the
 * same secret thousands of times.
 */
export function mulWindowed<F>(ops: FieldOps<F>, p: Jac<F>, k: bigint): Jac<F> {
  if (k === 0n || ops.isZero(p.z)) return infinity(ops);
  const negP = neg(ops, p);
  const e = k < 0n ? -k : k;
  const base = k < 0n ? negP : p;
  // table[i] = (i+1) * base
  const table: Jac<F>[] = [base];
  for (let i = 1; i < 15; i++) table.push(add(ops, table[i - 1], base));
  const digits: number[] = [];
  let rem = e;
  while (rem > 0n) {
    digits.push(Number(rem & 15n));
    rem >>= 4n;
  }
  let acc = infinity(ops);
  for (let i = digits.length - 1; i >= 0; i--) {
    for (let j = 0; j < 4; j++) acc = dbl(ops, acc);
    if (digits[i] !== 0) acc = add(ops, acc, table[digits[i] - 1]);
  }
  return acc;
}

/** Naive double-and-add scalar multiplication (used for small scalars). */
export function mul<F>(ops: FieldOps<F>, p: Jac<F>, k: bigint): Jac<F> {
  if (k === 0n || ops.isZero(p.z)) return infinity(ops);
  let acc = infinity(ops);
  let base = p;
  let e = k < 0n ? -k : k;
  if (k < 0n) base = neg(ops, p);
  while (e > 0n) {
    if (e & 1n) acc = add(ops, acc, base);
    base = dbl(ops, base);
    e >>= 1n;
  }
  return acc;
}

export const MSM_WINDOW_OVERRIDE = { value: 0 };
function msmWindowFor(n: number): number {
  if (MSM_WINDOW_OVERRIDE.value) return MSM_WINDOW_OVERRIDE.value;
  // Cost is ~ n*(254/c) mixed additions + 2^c*(254/c) full additions. The
  // second term dominates once 2^c >> n, so keep the window small.
  if (n > 1024) return 9;
  if (n > 256) return 8;
  return 7;
}

/**
 * Multi-scalar multiplication: sum_i k_i * P_i, via Pippenger's bucket method.
 * This is the workhorse behind every polynomial commitment.
 */
export function msm<F>(ops: FieldOps<F>, points: Aff<F>[], scalars: bigint[]): Jac<F> {
  if (points.length !== scalars.length) throw new Error("msm: length mismatch");
  const n = points.length;
  if (n === 0) return infinity(ops);

  // Filter zeros.
  const ps: Aff<F>[] = [];
  const ks: bigint[] = [];
  for (let i = 0; i < n; i++) {
    if (scalars[i] !== 0n && points[i] !== null) {
      ps.push(points[i]);
      ks.push(((scalars[i] % R) + R) % R);
    }
  }
  if (ps.length === 0) return infinity(ops);
  if (ps.length <= 8) {
    let acc = infinity(ops);
    for (let i = 0; i < ps.length; i++) acc = add(ops, acc, mul(ops, fromAffine(ops, ps[i]), ks[i]));
    return acc;
  }

  const bits = 254;
  const c = msmWindowFor(ps.length);
  const windows = Math.ceil(bits / c);
  const bucketsPerWindow = 1 << c;

  let result = infinity(ops);
  for (let w = windows - 1; w >= 0; w--) {
    // c doublings of the accumulator
    for (let j = 0; j < c; j++) result = dbl(ops, result);
    const buckets: Jac<F>[] = new Array(bucketsPerWindow);
    for (let b = 0; b < bucketsPerWindow; b++) buckets[b] = infinity(ops);
    const shift = BigInt(w * c);
    for (let i = 0; i < ps.length; i++) {
      const digit = Number((ks[i] >> shift) & BigInt(bucketsPerWindow - 1));
      if (digit !== 0) buckets[digit] = addMixed(ops, buckets[digit], ps[i]);
    }
    let running = infinity(ops);
    let acc = infinity(ops);
    for (let b = bucketsPerWindow - 1; b >= 1; b--) {
      running = add(ops, running, buckets[b]);
      acc = add(ops, acc, running);
    }
    result = add(ops, result, acc);
  }
  return result;
}

// ---------------------------------------------------------------------------
// The two groups
// ---------------------------------------------------------------------------

const B1 = 3n;

/** b' = 3 / (9 + u) for the sextic twist E': y^2 = x^3 + b' */
export function b2(): Fp2 {
  return new Fp2(3n, 0n).div(new Fp2(9n, 1n));
}

export const G1_GEN: Aff<bigint> = { x: 1n, y: 2n };

export const G2_GEN: Aff<Fp2> = {
  x: new Fp2(
    10857046999023057135944570762232829481370756359578518086990519993285655852781n,
    11559732032986387107991004021392285783925812861821192530917403151452391805634n,
  ),
  y: new Fp2(
    8495653923123431417604973247489272438418190587263600148770280649306958101930n,
    4082367875863433681332203403145435568316851327593401208105741076214120093531n,
  ),
};

export function isOnCurveG1(p: Aff<bigint>): boolean {
  if (p === null) return true;
  const lhs = FqField.sqr(p.y);
  const rhs = FqField.add(FqField.mul(FqField.sqr(p.x), p.x), B1);
  return lhs === rhs;
}

export function isOnCurveG2(p: Aff<Fp2>): boolean {
  if (p === null) return true;
  const lhs = p.y.sqr();
  const rhs = p.x.sqr().mul(p.x).add(b2());
  return lhs.eq(rhs);
}

/** G2 has a cofactor, so membership needs a subgroup check: [r]P == O. */
export function isInSubgroupG2(p: Aff<Fp2>): boolean {
  if (p === null) return true;
  const q = mul(fp2Ops, fromAffine(fp2Ops, p), R);
  return isInfinity(fp2Ops, q);
}

/** G1 has cofactor 1, so on-curve is enough. Kept for symmetry. */
export function isInSubgroupG1(p: Aff<bigint>): boolean {
  if (p === null) return true;
  return isInfinity(fqOps, mul(fqOps, fromAffine(fqOps, p), R));
}

// Convenience typed helpers -------------------------------------------------

export function g1Mul(p: Aff<bigint>, k: bigint): Aff<bigint> {
  return toAffine(fqOps, mul(fqOps, fromAffine(fqOps, p), k));
}
export function g1Add(a: Aff<bigint>, b: Aff<bigint>): Aff<bigint> {
  return toAffine(fqOps, add(fqOps, fromAffine(fqOps, a), fromAffine(fqOps, b)));
}
export function g2Mul(p: Aff<Fp2>, k: bigint): Aff<Fp2> {
  return toAffine(fp2Ops, mul(fp2Ops, fromAffine(fp2Ops, p), k));
}

// Serialization -------------------------------------------------------------
// Uncompressed big-endian encodings; 64 bytes for G1, 128 for G2.

export function g1ToBytes(p: Aff<bigint>): Uint8Array {
  const out = new Uint8Array(64);
  if (p === null) return out;
  out.set(FqField.toBytes(p.x), 0);
  out.set(FqField.toBytes(p.y), 32);
  return out;
}

export function g1FromBytes(bytes: Uint8Array): Aff<bigint> {
  if (bytes.length !== 64) throw new Error("g1FromBytes: expected 64 bytes");
  const x = FqField.fromBytes(bytes.subarray(0, 32));
  const y = FqField.fromBytes(bytes.subarray(32, 64));
  if (x === 0n && y === 0n) return null;
  return { x, y };
}

export function g2ToBytesImpl(p: Aff<Fp2>): Uint8Array {
  const out = new Uint8Array(128);
  if (p === null) return out;
  out.set(FqField.toBytes(p.x.c0), 0);
  out.set(FqField.toBytes(p.x.c1), 32);
  out.set(FqField.toBytes(p.y.c0), 64);
  out.set(FqField.toBytes(p.y.c1), 96);
  return out;
}

export function g2FromBytes(bytes: Uint8Array): Aff<Fp2> {
  if (bytes.length !== 128) throw new Error("g2FromBytes: expected 128 bytes");
  const x = new Fp2(FqField.fromBytes(bytes.subarray(0, 32)), FqField.fromBytes(bytes.subarray(32, 64)));
  const y = new Fp2(FqField.fromBytes(bytes.subarray(64, 96)), FqField.fromBytes(bytes.subarray(96, 128)));
  if (x.isZero() && y.isZero()) return null;
  return { x, y };
}

export function bytesToHex(b: Uint8Array): string {
  return Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
}

export { Q, R };
