import { describe, expect, it } from "vitest";
import { Fp2, Fp6, Fp12, FqField, FrField, Q, R } from "../src/crypto/field";
import {
  G1_GEN,
  G2_GEN,
  fp2Ops,
  fqOps,
  fromAffine,
  g1Mul,
  g2Mul,
  isInSubgroupG2,
  isOnCurveG1,
  isOnCurveG2,
  isInfinity,
  msm,
  toAffine,
} from "../src/crypto/curve";
import { pairing } from "../src/crypto/pairing";

describe("prime field", () => {
  it("inverts", () => {
    for (const a of [1n, 2n, 17n, R - 1n, 12345678901234567890n]) {
      expect(FrField.mul(a, FrField.inv(a))).toBe(1n);
    }
  });
  it("adds and subtracts consistently", () => {
    const a = R - 5n;
    expect(FrField.add(a, 7n)).toBe(2n);
    expect(FrField.sub(3n, 5n)).toBe(R - 2n);
    expect(FrField.neg(0n)).toBe(0n);
  });
  it("has the right characteristic", () => {
    expect(FrField.add(R - 1n, 1n)).toBe(0n);
    expect(FqField.add(Q - 1n, 1n)).toBe(0n);
  });
  it("random elements are in range", () => {
    for (let i = 0; i < 8; i++) {
      const a = FrField.random();
      expect(a >= 0n && a < R).toBe(true);
    }
  });
});

describe("extension fields", () => {
  it("Fp2 inverts", () => {
    const a = new Fp2(3n, 7n);
    expect(a.mul(a.inv()).eq(Fp2.one)).toBe(true);
  });
  it("Fp2 multiplication matches the rule u^2 = -1", () => {
    const u = new Fp2(0n, 1n);
    expect(u.mul(u).eq(new Fp2(Q - 1n, 0n))).toBe(true);
  });
  it("Fp6 inverts", () => {
    const a = new Fp6(new Fp2(1n, 2n), new Fp2(3n, 4n), new Fp2(5n, 6n));
    expect(a.mul(a.inv()).eq(Fp6.one)).toBe(true);
  });
  it("Fp6 v^3 = xi", () => {
    const v = new Fp6(Fp2.zero, Fp2.one, Fp2.zero);
    const v3 = v.mul(v).mul(v);
    expect(v3.c0.eq(new Fp2(9n, 1n))).toBe(true);
  });
  it("Fp12 inverts", () => {
    const a = new Fp12(new Fp6(new Fp2(1n, 2n), new Fp2(3n, 4n), new Fp2(5n, 6n)), new Fp6(new Fp2(7n, 8n), new Fp2(9n, 10n), new Fp2(11n, 12n)));
    expect(a.mul(a.inverse()).eq(Fp12.one)).toBe(true);
  });
});

describe("curve", () => {
  it("generators lie on their curves", () => {
    expect(isOnCurveG1(G1_GEN)).toBe(true);
    expect(isOnCurveG2(G2_GEN)).toBe(true);
  });
  it("G1 has order r", () => {
    expect(isInfinity(fqOps, fromAffine(fqOps, G1_GEN))).toBe(false);
    const rG = g1Mul(G1_GEN, R);
    expect(rG).toBeNull();
  });
  it("G2 sits in the order-r subgroup", () => {
    expect(isInSubgroupG2(G2_GEN)).toBe(true);
  });
  it("scalar multiplication is a group homomorphism", () => {
    const a = 12345n, b = 678n;
    const left = toAffine(fqOps, fromAffine(fqOps, g1Mul(G1_GEN, a + b)));
    const right = toAffine(
      fqOps,
      // add [a]G + [b]G by msm
      msm(fqOps, [G1_GEN, G1_GEN], [a, b]),
    );
    expect(left).toEqual(right);
  });
  it("msm matches naive multiplication", () => {
    const pts = [G1_GEN, g1Mul(G1_GEN, 7n), g1Mul(G1_GEN, 11n), g1Mul(G1_GEN, 19n), g1Mul(G1_GEN, 23n), g1Mul(G1_GEN, 29n), g1Mul(G1_GEN, 31n), g1Mul(G1_GEN, 37n), g1Mul(G1_GEN, 41n), g1Mul(G1_GEN, 43n)];
    const scalars = [5n, 9n, 17n, 33n, 65n, 129n, 257n, 513n, 1025n, 2049n];
    let expected = null as any;
    for (let i = 0; i < pts.length; i++) {
      const term = g1Mul(pts[i], scalars[i]);
      expected = expected === null ? term : toAffine(fqOps, msm(fqOps, [expected, term], [1n, 1n]));
    }
    const got = toAffine(fqOps, msm(fqOps, pts, scalars));
    expect(got).toEqual(expected);
  });
});

describe("pairing", () => {
  it("is bilinear", () => {
    const a = 5n;
    const b = 7n;
    const e1 = pairing(g1Mul(G1_GEN, a), G2_GEN);
    const e2 = pairing(G1_GEN, g2Mul(G2_GEN, b));
    const eAB = pairing(g1Mul(G1_GEN, a * b), G2_GEN);
    expect(eAB.eq(e1.pow(b))).toBe(true);
    expect(eAB.eq(e2.pow(a))).toBe(true);
    expect(e1.pow(b).eq(e2.pow(a))).toBe(true);
  });
  it("is non-degenerate", () => {
    const e = pairing(G1_GEN, G2_GEN);
    expect(e.eq(Fp12.one)).toBe(false);
    // e(P,Q)^r == 1
    expect(e.pow(R).eq(Fp12.one)).toBe(true);
  }, 120000);
  it("vanishes on the identity", () => {
    expect(pairing(null, G2_GEN).isOne()).toBe(true);
    expect(pairing(G1_GEN, null).isOne()).toBe(true);
  });
});
