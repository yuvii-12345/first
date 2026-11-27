import { describe, expect, it } from "vitest";
import { FrField, R } from "../src/crypto/field";
import { sha256, utf8 } from "../src/crypto/sha256";
import { Transcript } from "../src/crypto/transcript";
import { intt, ntt, rootOfUnity, cosetNtt, cosetIntt } from "../src/crypto/ntt";
import * as P from "../src/crypto/poly";
import { commit, open, setup, verify, openBatch, verifyBatch } from "../src/crypto/kzg";

describe("sha256", () => {
  it("matches known vectors", () => {
    const hex = (b: Uint8Array) => Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
    expect(hex(sha256(new Uint8Array()))).toBe("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    expect(hex(sha256(utf8("abc")))).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    expect(hex(sha256(utf8("The quick brown fox jumps over the lazy dog")))).toBe(
      "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
    );
  });
  it("hashes long inputs", () => {
    const hex = (b: Uint8Array) => Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
    expect(hex(sha256(new Uint8Array(1000).fill(0x61)))).toBe(
      "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3",
    );
  });
});

describe("transcript", () => {
  it("is deterministic", () => {
    const a = new Transcript("test");
    const b = new Transcript("test");
    a.absorbString("l", "hello").absorbField("x", 42n);
    b.absorbString("l", "hello").absorbField("x", 42n);
    expect(a.challenge("c")).toBe(b.challenge("c"));
  });
  it("separates domains", () => {
    expect(new Transcript("a").challenge("c")).not.toBe(new Transcript("b").challenge("c"));
  });
  it("gives distinct challenges in sequence", () => {
    const t = new Transcript("x");
    const c1 = t.challenge("a");
    const c2 = t.challenge("a");
    expect(c1).not.toBe(c2);
  });
});

describe("ntt", () => {
  it("produces a primitive root of unity", () => {
    for (const n of [2, 4, 8, 1024, 1 << 20]) {
      const w = rootOfUnity(n);
      expect(FrField.pow(w, BigInt(n))).toBe(1n);
      expect(FrField.pow(w, BigInt(n / 2))).toBe(R - 1n);
    }
  });
  it("round trips", () => {
    const n = 64;
    const coeffs = Array.from({ length: n }, (_, i) => FrField.reduce(BigInt(i * 7 + 3)));
    const evals = ntt(coeffs);
    expect(intt(evals)).toEqual(coeffs);
  });
  it("agrees with Horner evaluation", () => {
    const n = 32;
    const coeffs = Array.from({ length: n }, (_, i) => FrField.reduce(BigInt(i) ** 3n + 5n));
    const w = rootOfUnity(n);
    const evals = ntt(coeffs);
    for (let i = 0; i < n; i++) {
      expect(evals[i]).toBe(P.evaluate(coeffs, FrField.pow(w, BigInt(i))));
    }
  });
  it("coset transform round trips and matches evaluation", () => {
    const n = 16;
    const shift = 5n;
    const coeffs = Array.from({ length: n }, (_, i) => FrField.reduce(BigInt(i) * 11n + 1n));
    const evals = cosetNtt(coeffs, shift, n);
    const w = rootOfUnity(n);
    for (let i = 0; i < n; i++) {
      const x = FrField.mul(shift, FrField.pow(w, BigInt(i)));
      expect(evals[i]).toBe(P.evaluate(coeffs, x));
    }
    expect(cosetIntt(evals, shift)).toEqual(coeffs);
  });
});

describe("polynomials", () => {
  it("multiplies", () => {
    const a = [1n, 2n, 3n];
    const b = [4n, 5n];
    expect(P.mul(a, b)).toEqual([4n, 13n, 22n, 15n]);
  });
  it("divides by a linear factor", () => {
    const a = [1n, 2n, 3n, 4n];
    const z = 7n;
    const val = P.evaluate(a, z);
    const q = P.divideByLinear(P.sub(a, [val]), z);
    const recon = P.add(P.mul(q, [FrField.neg(z), 1n]), [val]);
    expect(P.trim(recon)).toEqual(P.trim(a));
  });
  it("lagrange evaluation agrees with interpolation", () => {
    const n = 8;
    const w = rootOfUnity(n);
    const evals = Array.from({ length: n }, (_, i) => FrField.reduce(BigInt(i * 3 + 1)));
    const x = 12345n;
    const viaLagrange = P.interpolateEval(evals, w, x);
    expect(viaLagrange).toBe(P.evaluate(P.interpolate(evals), x));
  });
});

describe("kzg", () => {
  it("commits and opens", () => {
    const srs = setup(32);
    const poly = Array.from({ length: 16 }, (_, i) => FrField.reduce(BigInt(i + 1) * 12345n));
    const c = commit(srs, poly);
    const z = 999n;
    const proof = open(srs, poly, z);
    expect(verify(srs, c, z, proof)).toBe(true);
  }, 60000);
  it("rejects a wrong value", () => {
    const srs = setup(16);
    const poly = [1n, 2n, 3n, 4n];
    const c = commit(srs, poly);
    const z = 5n;
    const proof = open(srs, poly, z);
    expect(verify(srs, c, z, { value: proof.value + 1n, proof: proof.proof })).toBe(false);
  }, 60000);
  it("batch opens", () => {
    const srs = setup(32);
    const p1 = [1n, 2n, 3n];
    const p2 = [5n, 6n];
    const z = 8n;
    const v = 1234567n;
    const { value, proof } = openBatch(srs, [p1, p2], z, v);
    const ok = verifyBatch(srs, [commit(srs, p1), commit(srs, p2)], [P.evaluate(p1, z), P.evaluate(p2, z)], z, proof, v);
    expect(ok).toBe(true);
    expect(value).toBe(FrField.add(P.evaluate(p1, z), FrField.mul(v, P.evaluate(p2, z))));
  }, 60000);
});
