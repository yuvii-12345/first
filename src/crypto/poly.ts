/**
 * Polynomials over Fr. Coefficients are stored little-endian (index = degree),
 * which is what the NTT and KZG both want.
 */
import { FrField } from "./field";
import { cosetNtt, intt, mulNtt, ntt, rootOfUnity } from "./ntt";

export type Poly = bigint[];

export function zero(): Poly {
  return [];
}
export function isZero(a: Poly): boolean {
  return a.every((c) => c === 0n);
}
export function trim(a: Poly): Poly {
  let d = a.length;
  while (d > 0 && a[d - 1] === 0n) d--;
  return a.slice(0, d);
}
export function degree(a: Poly): number {
  const t = trim(a);
  return t.length - 1;
}
export function padTo(a: Poly, n: number): Poly {
  const out = new Array<bigint>(n).fill(0n);
  for (let i = 0; i < Math.min(a.length, n); i++) out[i] = a[i];
  return out;
}
export function add(a: Poly, b: Poly): Poly {
  const n = Math.max(a.length, b.length);
  const out = new Array<bigint>(n);
  for (let i = 0; i < n; i++) out[i] = FrField.add(a[i] ?? 0n, b[i] ?? 0n);
  return out;
}
export function sub(a: Poly, b: Poly): Poly {
  const n = Math.max(a.length, b.length);
  const out = new Array<bigint>(n);
  for (let i = 0; i < n; i++) out[i] = FrField.sub(a[i] ?? 0n, b[i] ?? 0n);
  return out;
}
export function neg(a: Poly): Poly {
  return a.map((c) => FrField.neg(c));
}
export function scale(a: Poly, k: bigint): Poly {
  return a.map((c) => FrField.mul(c, k));
}
export function scaleAdd(a: Poly, b: Poly, k: bigint): Poly {
  // a + k*b
  const n = Math.max(a.length, b.length);
  const out = new Array<bigint>(n);
  for (let i = 0; i < n; i++) out[i] = FrField.add(a[i] ?? 0n, FrField.mul(b[i] ?? 0n, k));
  return out;
}
export function mul(a: Poly, b: Poly): Poly {
  if (a.length === 0 || b.length === 0) return [];
  if (a.length * b.length <= 256) {
    const out = new Array<bigint>(a.length + b.length - 1).fill(0n);
    for (let i = 0; i < a.length; i++) {
      if (a[i] === 0n) continue;
      for (let j = 0; j < b.length; j++) {
        out[i + j] = FrField.add(out[i + j], FrField.mul(a[i], b[j]));
      }
    }
    return out;
  }
  return mulNtt(a, b);
}

/** Evaluate with Horner's rule. */
export function evaluate(a: Poly, x: bigint): bigint {
  let acc = 0n;
  for (let i = a.length - 1; i >= 0; i--) acc = FrField.add(FrField.mul(acc, x), a[i]);
  return acc;
}

/** Coefficients -> evaluations on the domain {1, w, ..., w^{n-1}}. */
export function evaluateOnDomain(a: Poly, n: number): bigint[] {
  const padded = padTo(a, n);
  return ntt(padded);
}

/** Evaluations on a domain -> coefficients. */
export function interpolate(evals: bigint[]): Poly {
  return intt(evals);
}

/** Evaluate on the coset {shift, shift*w, ...}. */
export function evaluateOnCoset(a: Poly, n: number, shift: bigint): bigint[] {
  return cosetNtt(a, shift, n);
}

/**
 * Divide by (X - z) exactly (synthetic division). Returns the quotient.
 * The polynomial must vanish at z, otherwise the remainder is silently dropped;
 * callers check the remainder explicitly when it matters.
 */
export function divideByLinear(a: Poly, z: bigint): Poly {
  const n = a.length;
  if (n === 0) return [];
  const out = new Array<bigint>(n - 1).fill(0n);
  let carry = 0n;
  for (let i = n - 1; i >= 1; i--) {
    carry = FrField.add(a[i], FrField.mul(carry, z));
    out[i - 1] = carry;
  }
  return out;
}

/** Remainder of division by (X - z) = a(z). */
export function remainderByLinear(a: Poly, z: bigint): bigint {
  return evaluate(a, z);
}

/** Z_H(X) = X^n - 1 evaluated at x. */
export function vanishingEval(n: number, x: bigint): bigint {
  return FrField.sub(FrField.pow(x, BigInt(n)), 1n);
}

/** Z_H as a coefficient vector. */
export function vanishingPoly(n: number): Poly {
  const out = new Array<bigint>(n + 1).fill(0n);
  out[0] = FrField.neg(1n);
  out[n] = 1n;
  return out;
}

/**
 * The i-th Lagrange basis polynomial evaluated at x, for the domain of size n:
 *   L_i(x) = (w^i / n) * (x^n - 1) / (x - w^i)
 */
export function lagrangeEval(i: number, n: number, w: bigint, x: bigint): bigint {
  const wi = FrField.pow(w, BigInt(i));
  if (x === wi) return 1n;
  const num = FrField.mul(wi, FrField.inv(BigInt(n)));
  const zh = vanishingEval(n, x);
  return FrField.mul(FrField.mul(num, zh), FrField.inv(FrField.sub(x, wi)));
}

/** Sum_i coeffs_i * L_i(x) — evaluating a polynomial given by its domain values. */
export function interpolateEval(evals: bigint[], w: bigint, x: bigint): bigint {
  const n = evals.length;
  let acc = 0n;
  for (let i = 0; i < n; i++) {
    if (evals[i] === 0n) continue;
    acc = FrField.add(acc, FrField.mul(evals[i], lagrangeEval(i, n, w, x)));
  }
  return acc;
}

/** Multiply the polynomial by X^shift. */
export function shift(a: Poly, by: number): Poly {
  if (by <= 0) return a.slice();
  const out = new Array<bigint>(a.length + by).fill(0n);
  for (let i = 0; i < a.length; i++) out[i + by] = a[i];
  return out;
}

/**
 * Given p(X), return p(w*X): coefficient j is multiplied by w^j.
 * This is how the prover obtains the "shifted" wire polynomial in PLONK.
 */
export function rescaleByOmega(a: Poly, w: bigint): Poly {
  const out = new Array<bigint>(a.length);
  let acc = 1n;
  for (let i = 0; i < a.length; i++) {
    out[i] = FrField.mul(a[i], acc);
    acc = FrField.mul(acc, w);
  }
  return out;
}

export { rootOfUnity };
