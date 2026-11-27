/**
 * The optimal ate pairing on BN254, e: G1 x G2 -> GT, a subgroup of Fp12*.
 *
 * Strategy: lift both points into E(Fp12) (G1 via the identity, G2 via the
 * sextic twist), then run the textbook Miller loop with *actual lines in
 * E(Fp12)*. That is a few times slower than the sparse line formulas used in
 * production libraries, but it is much harder to get subtly wrong — and a
 * subtly wrong pairing would silently make every proof verify.
 *
 * BN254:  y^2 = x^3 + 3 over Fq,   E'(Fp2): y^2 = x^3 + 3/xi,
 *         Fp12 = Fp6[w]/(w^2 - v), Fp6 = Fp2[v]/(v^3 - xi),  xi = 9 + u,
 *         untwist  Psi(x', y') = (x' w^2, y' w^3).
 */
import { Fp2, Fp6, Fp12, Q, R, frobenius, fp2Pow, XI } from "./field";
import type { Aff } from "./curve";

/** 6u + 2 for the BN parameter u = 4965661367192848881. */
export const ATE_LOOP_COUNT = 29793968203157093288n;
const LOG_ATE_LOOP_COUNT = 63;

/** v as an element of Fp6 (v^1). */
const V6 = new Fp6(Fp2.zero, Fp2.one, Fp2.zero);
/** w^2 = v, as an Fp12 element. */
const W2 = new Fp12(V6, Fp6.zero);
/** w^3 = v*w, as an Fp12 element. */
const W3 = new Fp12(Fp6.zero, V6);

function embedFp2(a: Fp2): Fp12 {
  return new Fp12(Fp6.fromFp2(a), Fp6.zero);
}

/** Lift a G1 point into E(Fp12) (coordinates are already in Fq). */
export function castG1(P: Aff<bigint>): { x: Fp12; y: Fp12 } | null {
  if (P === null) return null;
  return { x: embedFp2(Fp2.from(P.x)), y: embedFp2(Fp2.from(P.y)) };
}

/** Lift a G2 point through the sextic twist: (x', y') -> (x' w^2, y' w^3). */
export function twist(Q2: Aff<Fp2>): { x: Fp12; y: Fp12 } | null {
  if (Q2 === null) return null;
  return {
    x: embedFp2(Q2.x).mul(W2),
    y: embedFp2(Q2.y).mul(W3),
  };
}

type PF12 = { x: Fp12; y: Fp12 };

function dblFp12(p: PF12): PF12 {
  // a = 0: lambda = 3x^2 / 2y
  const lambda = p.x.sqr().mulScalar(3n).mul(p.y.mulScalar(2n).inverse());
  const x3 = lambda.sqr().sub(p.x.mulScalar(2n));
  const y3 = lambda.mul(p.x.sub(x3)).sub(p.y);
  return { x: x3, y: y3 };
}

function addFp12(p: PF12 | null, q: PF12 | null): PF12 | null {
  if (p === null) return q;
  if (q === null) return p;
  if (p.x.eq(q.x)) {
    if (p.y.eq(q.y)) return dblFp12(p);
    return null;
  }
  const lambda = q.y.sub(p.y).mul(q.x.sub(p.x).inverse());
  const x3 = lambda.sqr().sub(p.x).sub(q.x);
  const y3 = lambda.mul(p.x.sub(x3)).sub(p.y);
  return { x: x3, y: y3 };
}

/** The line through P1, P2 evaluated at T — all in E(Fp12). */
function linefunc(p1: PF12, p2: PF12, t: PF12): Fp12 {
  if (!p1.x.eq(p2.x)) {
    const m = p2.y.sub(p1.y).mul(p2.x.sub(p1.x).inverse());
    return m.mul(t.x.sub(p1.x)).sub(t.y.sub(p1.y));
  }
  if (p1.y.eq(p2.y)) {
    const m = p1.x.sqr().mulScalar(3n).mul(p1.y.mulScalar(2n).inverse());
    return m.mul(t.x.sub(p1.x)).sub(t.y.sub(p1.y));
  }
  return t.x.sub(p1.x); // vertical
}

/** Raw Miller loop output (before the final exponentiation). */
export function millerLoopRaw(P: Aff<bigint>, Q2: Aff<Fp2>): Fp12 {
  const P12 = castG1(P);
  const Q12 = twist(Q2);
  if (P12 === null || Q12 === null) return Fp12.one;

  let Rt: PF12 = Q12;
  let f = Fp12.one;
  for (let i = LOG_ATE_LOOP_COUNT; i >= 0; i--) {
    f = f.mul(f).mul(linefunc(Rt, Rt, P12));
    Rt = dblFp12(Rt);
    if ((ATE_LOOP_COUNT >> BigInt(i)) & 1n) {
      f = f.mul(linefunc(Rt, Q12, P12));
      Rt = addFp12(Rt, Q12) as PF12;
    }
  }

  // Frobenius on E(Fp12): Q1 = pi(Q),  nQ2 = -pi^2(Q).
  const Q1: PF12 = { x: frobenius(Q12.x), y: frobenius(Q12.y) };
  const nQ2: PF12 = { x: frobenius(Q1.x), y: frobenius(Q1.y).neg() };

  f = f.mul(linefunc(Rt, Q1, P12));
  Rt = (addFp12(Rt, Q1) ?? Rt) as PF12;
  f = f.mul(linefunc(Rt, nQ2, P12));
  return f;
}

export const FINAL_EXPONENT = (Q ** 12n - 1n) / R;

export function finalExponentiation(f: Fp12): Fp12 {
  return f.pow(FINAL_EXPONENT);
}

/** The full (reduced) pairing e(P, Q). */
export function pairing(P: Aff<bigint>, Q2: Aff<Fp2>): Fp12 {
  return finalExponentiation(millerLoopRaw(P, Q2));
}

/** Check a product of pairings against one, using a single final exponentiation. */
export function pairingCheck(pairs: Array<[Aff<bigint>, Aff<Fp2>]>): boolean {
  let f = Fp12.one;
  for (const [p, q] of pairs) f = f.mul(millerLoopRaw(p, q));
  return finalExponentiation(f).isOne();
}
