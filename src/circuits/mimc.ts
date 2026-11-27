/**
 * MiMC — a hash designed to be cheap *inside* a SNARK.
 *
 *   x_{i+1} = (x_i + k + c_i)^5
 *
 * Every operation is a field multiplication or addition, so each round costs
 * four gates instead of the tens of thousands a bit-oriented hash (SHA-256)
 * would need. 110 rounds is the standard recommendation for exponent 5 over a
 * 254-bit field.
 *
 * The statement: "I know x such that MiMC(x, k) = y", with k and y public.
 */
import { FrField } from "../crypto/field";
import { sha256, utf8 } from "../crypto/sha256";
import { CircuitBuilder, type CompiledCircuit } from "../plonk/circuit";

export const MIMC_ROUNDS = 110;

let cachedConstants: bigint[] | null = null;

export function roundConstants(): bigint[] {
  if (cachedConstants) return cachedConstants;
  const out: bigint[] = [];
  for (let i = 0; i < MIMC_ROUNDS; i++) {
    const h = sha256(utf8(`zeroth/mimc/round/${i}`));
    let v = 0n;
    for (const b of h) v = (v << 8n) | BigInt(b);
    out.push(FrField.reduce(v));
  }
  cachedConstants = out;
  return out;
}

/** The plain-TS reference implementation; the circuit must agree with it. */
export function mimcHash(x: bigint, k: bigint): bigint {
  const cs = roundConstants();
  let acc = FrField.reduce(x);
  const key = FrField.reduce(k);
  for (let i = 0; i < MIMC_ROUNDS; i++) {
    const t = FrField.add(FrField.add(acc, key), cs[i]);
    const t2 = FrField.mul(t, t);
    const t4 = FrField.mul(t2, t2);
    acc = FrField.mul(t4, t);
  }
  return acc;
}

export interface MimcCircuitResult {
  circuit: CompiledCircuit;
  expected: bigint;
  outputVar: number;
}

/**
 * Build the preimage circuit. `witnessX` is what the prover claims to know;
 * `publicY` is the statement. If they disagree the proof simply will not
 * verify — that is the whole point.
 */
export function mimcPreimageCircuit(witnessX: bigint, key: bigint, publicY: bigint): MimcCircuitResult {
  const b = new CircuitBuilder();
  const k = b.publicInput("k", key);
  const x = b.secret("x", witnessX);
  const y = b.publicInput("y", publicY);

  let cur = x;
  const cs = roundConstants();
  for (let i = 0; i < MIMC_ROUNDS; i++) {
    const t = b.addConst(cur, cs[i], `r${i}_addk`); // x + k + c_i
    const tk = b.add(t, k, `r${i}_addkey`);
    const t2 = b.mul(tk, tk, `r${i}_sq`);
    const t4 = b.mul(t2, t2, `r${i}_sq2`);
    cur = b.mul(t4, tk, `r${i}_pow5`);
  }
  b.assertEqual(cur, y, "MiMC(x,k) == y");

  return { circuit: b.finalize(), expected: mimcHash(witnessX, key), outputVar: cur };
}
