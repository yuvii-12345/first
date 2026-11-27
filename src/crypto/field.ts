/**
 * Prime-field and extension-field arithmetic over BN254.
 *
 * Everything here is written from scratch on top of BigInt. No crypto
 * libraries, no Montgomery tricks — just `%` and schoolbook formulas, because
 * the point of this project is to be readable. (It is therefore NOT
 * constant-time. See docs/SECURITY.md.)
 */

/** BN254 base field (the field the curve coordinates live in). */
export const Q = 21888242871839275222246405745257275088696311157297823662689037894645226208583n;
/** BN254 scalar field (the field our polynomials / witness live in). */
export const R = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;

export class PrimeField {
  readonly p: bigint;
  /** Number of bytes needed to serialize an element. */
  readonly byteLength: number;

  constructor(p: bigint) {
    this.p = p;
    this.byteLength = Math.ceil(p.toString(16).length / 2);
  }

  get zero(): bigint {
    return 0n;
  }
  get one(): bigint {
    return 1n;
  }

  add(a: bigint, b: bigint): bigint {
    const s = a + b;
    return s >= this.p ? s - this.p : s;
  }
  sub(a: bigint, b: bigint): bigint {
    const d = a - b;
    return d < 0n ? d + this.p : d;
  }
  neg(a: bigint): bigint {
    return a === 0n ? 0n : this.p - a;
  }
  mul(a: bigint, b: bigint): bigint {
    return (a * b) % this.p;
  }
  sqr(a: bigint): bigint {
    return (a * a) % this.p;
  }
  /** a^-1 mod p via the extended Euclidean algorithm. */
  inv(a: bigint): bigint {
    let t = 0n;
    let newT = 1n;
    let r = this.p;
    let newR = ((a % this.p) + this.p) % this.p;
    if (newR === 0n) throw new Error("inverse does not exist (0)");
    while (newR !== 0n) {
      const q = r / newR;
      [t, newT] = [newT, t - q * newT];
      [r, newR] = [newR, r - q * newR];
    }
    if (r !== 1n) throw new Error("inverse does not exist (not coprime)");
    return ((t % this.p) + this.p) % this.p;
  }
  div(a: bigint, b: bigint): bigint {
    return this.mul(a, this.inv(b));
  }
  pow(a: bigint, e: bigint): bigint {
    if (e < 0n) return this.pow(this.inv(a), -e);
    let result = 1n;
    let base = ((a % this.p) + this.p) % this.p;
    let exp = e;
    while (exp > 0n) {
      if (exp & 1n) result = (result * base) % this.p;
      base = (base * base) % this.p;
      exp >>= 1n;
    }
    return result;
  }
  isZero(a: bigint): boolean {
    return a === 0n;
  }
  eq(a: bigint, b: bigint): boolean {
    return a === b;
  }

  /** Reduce an arbitrary bigint into [0, p). */
  reduce(v: bigint): bigint {
    const m = v % this.p;
    return m < 0n ? m + this.p : m;
  }

  /** Uniform-ish element from 32 fresh random bytes, by rejection sampling. */
  random(): bigint {
    const bytes = new Uint8Array(this.byteLength + 8);
    for (;;) {
      globalThis.crypto.getRandomValues(bytes);
      let v = 0n;
      for (const b of bytes) v = (v << 8n) | BigInt(b);
      const cand = v % this.p;
      // Reject to avoid modulo bias.
      if (v - cand + this.p <= (1n << BigInt(bytes.length * 8)) - 1n) return cand;
    }
  }

  toBytes(a: bigint): Uint8Array {
    const out = new Uint8Array(32);
    let v = a;
    for (let i = 31; i >= 0; i--) {
      out[i] = Number(v & 0xffn);
      v >>= 8n;
    }
    return out;
  }
  fromBytes(bytes: Uint8Array): bigint {
    let v = 0n;
    for (const b of bytes) v = (v << 8n) | BigInt(b);
    return this.reduce(v);
  }
  toHex(a: bigint): string {
    return "0x" + a.toString(16).padStart(64, "0");
  }
}

export const FqField = new PrimeField(Q);
export const FrField = new PrimeField(R);

// ---------------------------------------------------------------------------
// Fp2 = Fp[u] / (u^2 + 1)
// ---------------------------------------------------------------------------

export class Fp2 {
  constructor(
    readonly c0: bigint,
    readonly c1: bigint,
  ) {}

  static zero = new Fp2(0n, 0n);
  static one = new Fp2(1n, 0n);

  static from(v: bigint): Fp2 {
    return new Fp2(v, 0n);
  }

  add(o: Fp2): Fp2 {
    return new Fp2(FqField.add(this.c0, o.c0), FqField.add(this.c1, o.c1));
  }
  sub(o: Fp2): Fp2 {
    return new Fp2(FqField.sub(this.c0, o.c0), FqField.sub(this.c1, o.c1));
  }
  neg(): Fp2 {
    return new Fp2(FqField.neg(this.c0), FqField.neg(this.c1));
  }
  mul(o: Fp2): Fp2 {
    // (a+bu)(c+du) = (ac - bd) + (ad + bc)u
    const ac = FqField.mul(this.c0, o.c0);
    const bd = FqField.mul(this.c1, o.c1);
    const ad = FqField.mul(this.c0, o.c1);
    const bc = FqField.mul(this.c1, o.c0);
    return new Fp2(FqField.sub(ac, bd), FqField.add(ad, bc));
  }
  sqr(): Fp2 {
    return this.mul(this);
  }
  mulScalar(k: bigint): Fp2 {
    return new Fp2(FqField.mul(this.c0, k), FqField.mul(this.c1, k));
  }
  inv(): Fp2 {
    const norm = FqField.add(FqField.sqr(this.c0), FqField.sqr(this.c1));
    const nInv = FqField.inv(norm);
    return new Fp2(FqField.mul(this.c0, nInv), FqField.neg(FqField.mul(this.c1, nInv)));
  }
  div(o: Fp2): Fp2 {
    return this.mul(o.inv());
  }
  isZero(): boolean {
    return this.c0 === 0n && this.c1 === 0n;
  }
  eq(o: Fp2): boolean {
    return this.c0 === o.c0 && this.c1 === o.c1;
  }
  /** Conjugation = the q-power Frobenius (u^q = -u because u^2 = -1). */
  conjugate(): Fp2 {
    return new Fp2(this.c0, FqField.neg(this.c1));
  }
  toString(): string {
    return `(${this.c0.toString(16)} , ${this.c1.toString(16)})`;
  }
}

// ---------------------------------------------------------------------------
// Fp6 = Fp2[v] / (v^3 - xi),  xi = 9 + u
// ---------------------------------------------------------------------------

export const XI = new Fp2(9n, 1n);

export class Fp6 {
  constructor(
    readonly c0: Fp2,
    readonly c1: Fp2,
    readonly c2: Fp2,
  ) {}

  static zero = new Fp6(Fp2.zero, Fp2.zero, Fp2.zero);
  static one = new Fp6(Fp2.one, Fp2.zero, Fp2.zero);

  static fromFp2(v: Fp2): Fp6 {
    return new Fp6(v, Fp2.zero, Fp2.zero);
  }

  add(o: Fp6): Fp6 {
    return new Fp6(this.c0.add(o.c0), this.c1.add(o.c1), this.c2.add(o.c2));
  }
  sub(o: Fp6): Fp6 {
    return new Fp6(this.c0.sub(o.c0), this.c1.sub(o.c1), this.c2.sub(o.c2));
  }
  neg(): Fp6 {
    return new Fp6(this.c0.neg(), this.c1.neg(), this.c2.neg());
  }
  mul(o: Fp6): Fp6 {
    const a0 = this.c0;
    const a1 = this.c1;
    const a2 = this.c2;
    const b0 = o.c0;
    const b1 = o.c1;
    const b2 = o.c2;
    // c0 = a0b0 + xi(a1b2 + a2b1)
    // c1 = a0b1 + a1b0 + xi a2b2
    // c2 = a0b2 + a1b1 + a2b0
    const c0 = a0.mul(b0).add(a1.mul(b2).add(a2.mul(b1)).mul(XI));
    const c1 = a0.mul(b1).add(a1.mul(b0)).add(a2.mul(b2).mul(XI));
    const c2 = a0.mul(b2).add(a1.mul(b1)).add(a2.mul(b0));
    return new Fp6(c0, c1, c2);
  }
  mulFp2(k: Fp2): Fp6 {
    return new Fp6(this.c0.mul(k), this.c1.mul(k), this.c2.mul(k));
  }
  mulScalar(k: bigint): Fp6 {
    return new Fp6(this.c0.mulScalar(k), this.c1.mulScalar(k), this.c2.mulScalar(k));
  }
  mulByV(): Fp6 {
    // multiply by v: (c0,c1,c2) -> (xi*c2, c0, c1)
    return new Fp6(this.c2.mul(XI), this.c0, this.c1);
  }
  sqr(): Fp6 {
    return this.mul(this);
  }
  inv(): Fp6 {
    const a0 = this.c0;
    const a1 = this.c1;
    const a2 = this.c2;
    const t0 = a0.sqr().sub(a1.mul(a2).mul(XI));
    const t1 = a2.sqr().mul(XI).sub(a0.mul(a1));
    const t2 = a1.sqr().sub(a0.mul(a2));
    const denom = a0.mul(t0).add(a2.mul(t1).add(a1.mul(t2)).mul(XI));
    const dInv = denom.inv();
    return new Fp6(t0.mul(dInv), t1.mul(dInv), t2.mul(dInv));
  }
  isZero(): boolean {
    return this.c0.isZero() && this.c1.isZero() && this.c2.isZero();
  }
  eq(o: Fp6): boolean {
    return this.c0.eq(o.c0) && this.c1.eq(o.c1) && this.c2.eq(o.c2);
  }
}

// ---------------------------------------------------------------------------
// Fp12 = Fp6[w] / (w^2 - v)
// ---------------------------------------------------------------------------

export class Fp12 {
  constructor(
    readonly c0: Fp6,
    readonly c1: Fp6,
  ) {}

  static zero = new Fp12(Fp6.zero, Fp6.zero);
  static one = new Fp12(Fp6.one, Fp6.zero);

  add(o: Fp12): Fp12 {
    return new Fp12(this.c0.add(o.c0), this.c1.add(o.c1));
  }
  sub(o: Fp12): Fp12 {
    return new Fp12(this.c0.sub(o.c0), this.c1.sub(o.c1));
  }
  neg(): Fp12 {
    return new Fp12(this.c0.neg(), this.c1.neg());
  }
  mul(o: Fp12): Fp12 {
    // (a0 + a1 w)(b0 + b1 w) = (a0b0 + a1b1 v) + (a0b1 + a1b0) w
    const a0b0 = this.c0.mul(o.c0);
    const a1b1 = this.c1.mul(o.c1);
    const c0 = a0b0.add(a1b1.mulByV());
    const c1 = this.c0.mul(o.c1).add(this.c1.mul(o.c0));
    return new Fp12(c0, c1);
  }
  sqr(): Fp12 {
    return this.mul(this);
  }
  mulScalar(k: bigint): Fp12 {
    return new Fp12(this.c0.mulScalar(k), this.c1.mulScalar(k));
  }
  /** (a0 + a1 w)^-1 = (a0 - a1 w) / (a0^2 - v a1^2) */
  inverse(): Fp12 {
    const t = this.c0.sqr().sub(this.c1.sqr().mulByV());
    const tInv = t.inv();
    return new Fp12(this.c0.mul(tInv), this.c1.neg().mul(tInv));
  }
  div(o: Fp12): Fp12 {
    return this.mul(o.inverse());
  }
  pow(e: bigint): Fp12 {
    let result = Fp12.one;
    let base: Fp12 = this;
    let exp = e;
    while (exp > 0n) {
      if (exp & 1n) result = result.mul(base);
      base = base.sqr();
      exp >>= 1n;
    }
    return result;
  }
  eq(o: Fp12): boolean {
    return this.c0.eq(o.c0) && this.c1.eq(o.c1);
  }
  isOne(): boolean {
    return this.eq(Fp12.one);
  }
  /** w-power conjugation: (c0, c1) -> (c0, -c1). */
  conjugate(): Fp12 {
    return new Fp12(this.c0, this.c1.neg());
  }
  toString(): string {
    return `[${this.c0.c0}, ${this.c0.c1}, ${this.c0.c2} | ${this.c1.c0}, ${this.c1.c1}, ${this.c1.c2}]`;
  }
}

// ---------------------------------------------------------------------------
// Frobenius
// ---------------------------------------------------------------------------

/** Exponentiation in Fp2. */
export function fp2Pow(a: Fp2, e: bigint): Fp2 {
  let result = Fp2.one;
  let base = a;
  let exp = e;
  while (exp > 0n) {
    if (exp & 1n) result = result.mul(base);
    base = base.sqr();
    exp >>= 1n;
  }
  return result;
}

/**
 * Frobenius constants for Fp12 = Fp6[w] / (w^2 - v), Fp6 = Fp2[v] / (v^3 - xi).
 *
 *   v^q   = v   * xi^((q-1)/3)      =: v * G1
 *   v^2q  = v^2 * xi^(2(q-1)/3)     =: v^2 * G2
 *   w^q   = w   * xi^((q-1)/6)      =: w * Gw
 */
export const FROB_V1: Fp2 = fp2Pow(XI, (Q - 1n) / 3n);
export const FROB_V2: Fp2 = FROB_V1.sqr();
export const FROB_W: Fp2 = fp2Pow(XI, (Q - 1n) / 6n);

/** The q-power Frobenius on Fp12. */
export function frobenius(f: Fp12): Fp12 {
  const c0 = new Fp6(f.c0.c0.conjugate(), f.c0.c1.conjugate().mul(FROB_V1), f.c0.c2.conjugate().mul(FROB_V2));
  const c1 = new Fp6(f.c1.c0.conjugate(), f.c1.c1.conjugate().mul(FROB_V1), f.c1.c2.conjugate().mul(FROB_V2)).mulFp2(FROB_W);
  return new Fp12(c0, c1);
}

/** The q^k-power Frobenius. */
export function frobeniusPow(f: Fp12, k: number): Fp12 {
  let out = f;
  for (let i = 0; i < k; i++) out = frobenius(out);
  return out;
}
