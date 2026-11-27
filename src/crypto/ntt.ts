/**
 * Number Theoretic Transform over the BN254 scalar field.
 *
 * The NTT is the FFT with roots of unity in Fr instead of complex numbers; it
 * converts a polynomial's coefficients into its evaluations (and back) in
 * O(n log n) field operations. Every commitment, every quotient, and every
 * "divide by the vanishing polynomial" in the proof system runs through here.
 *
 * The scalar field has 2-adicity 28: r - 1 = 2^28 * 3^2 * ... so domains up to
 * 2^28 elements exist.
 */
import { FrField, R } from "./field";

export const TWO_ADICITY = 28;

const rootCache = new Map<number, bigint>();

function multiplicativeGenerator(): bigint {
  // Smallest quadratic non-residue is a generator of the 2-power subgroup.
  for (let g = 2n; g < 64n; g++) {
    if (FrField.pow(g, (R - 1n) / 2n) !== 1n) return g;
  }
  throw new Error("no generator found");
}

let GEN: bigint | null = null;

/** A primitive n-th root of unity in Fr. */
export function rootOfUnity(n: number): bigint {
  if (n <= 0 || (n & (n - 1)) !== 0) throw new Error("n must be a power of two");
  if (n > 1 << TWO_ADICITY) throw new Error("domain too large");
  const cached = rootCache.get(n);
  if (cached !== undefined) return cached;
  if (GEN === null) GEN = multiplicativeGenerator();
  const w = FrField.pow(GEN, (R - 1n) / BigInt(n));
  // sanity: w has order exactly n
  if (FrField.pow(w, BigInt(n)) !== 1n) throw new Error("root of unity of wrong order");
  rootCache.set(n, w);
  return w;
}

/** Powers of a value: [1, a, a^2, ...] — handy for coset shifts. */
export function powers(a: bigint, count: number): bigint[] {
  const out = new Array<bigint>(count);
  let acc = 1n;
  for (let i = 0; i < count; i++) {
    out[i] = acc;
    acc = FrField.mul(acc, a);
  }
  return out;
}

function bitReverse(a: bigint[]): void {
  const n = a.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
  }
}

/** In-place NTT: coefficients -> evaluations at [1, w, w^2, ...]. */
export function nttInPlace(a: bigint[], w: bigint): void {
  const n = a.length;
  bitReverse(a);
  for (let len = 2; len <= n; len <<= 1) {
    const wLen = FrField.pow(w, BigInt(n / len));
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let wk = 1n;
      for (let j = 0; j < half; j++) {
        const u = a[i + j];
        const v = FrField.mul(a[i + j + half], wk);
        a[i + j] = FrField.add(u, v);
        a[i + j + half] = FrField.sub(u, v);
        wk = FrField.mul(wk, wLen);
      }
    }
  }
}

/** In-place inverse NTT. */
export function inttInPlace(a: bigint[], w: bigint): void {
  const n = a.length;
  nttInPlace(a, FrField.inv(w));
  const nInv = FrField.inv(BigInt(n));
  for (let i = 0; i < n; i++) a[i] = FrField.mul(a[i], nInv);
}

export function ntt(coeffs: bigint[]): bigint[] {
  const out = coeffs.slice();
  nttInPlace(out, rootOfUnity(out.length));
  return out;
}

export function intt(evals: bigint[]): bigint[] {
  const out = evals.slice();
  inttInPlace(out, rootOfUnity(out.length));
  return out;
}

/**
 * Evaluate a polynomial (given by coefficients) on the coset shift * H,
 * i.e. at shift * w^i for i = 0..n-1.
 */
export function cosetNtt(coeffs: bigint[], shift: bigint, size?: number): bigint[] {
  const n = size ?? coeffs.length;
  const out = new Array<bigint>(n);
  let sp = 1n;
  for (let i = 0; i < n; i++) {
    out[i] = i < coeffs.length ? FrField.mul(coeffs[i], sp) : 0n;
    sp = FrField.mul(sp, shift);
  }
  nttInPlace(out, rootOfUnity(n));
  return out;
}

/** Inverse of cosetNtt. */
export function cosetIntt(evals: bigint[], shift: bigint): bigint[] {
  const out = evals.slice();
  inttInPlace(out, rootOfUnity(out.length));
  const shiftInv = FrField.inv(shift);
  let sp = 1n;
  for (let i = 0; i < out.length; i++) {
    out[i] = FrField.mul(out[i], sp);
    sp = FrField.mul(sp, shiftInv);
  }
  return out;
}

/** Polynomial multiplication via NTT. */
export function mulNtt(a: bigint[], b: bigint[]): bigint[] {
  if (a.length === 0 || b.length === 0) return [];
  const need = a.length + b.length - 1;
  let n = 1;
  while (n < need) n <<= 1;
  const fa = cosetNtt(a, 1n, n);
  const fb = cosetNtt(b, 1n, n);
  const fc = new Array<bigint>(n);
  for (let i = 0; i < n; i++) fc[i] = FrField.mul(fa[i], fb[i]);
  const c = intt(fc);
  return c.slice(0, need);
}

export function nextPowerOfTwo(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export function log2(n: number): number {
  return Math.round(Math.log2(n));
}
