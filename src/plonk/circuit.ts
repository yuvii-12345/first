/**
 * PLONKish arithmetization.
 *
 * A circuit is a table of rows. The prover knows three "wire" values per row
 * (a, b, c); the circuit fixes five selector values per row (qL, qR, qO, qM,
 * qC). The single gate equation is
 *
 *      qM·a·b + qL·a + qR·b + qO·c + qC + PI = 0
 *
 * which is enough to express multiplication, addition, constants, boolean
 * constraints, and public inputs. Wiring ("the value in row 3 wire a is the
 * same variable as row 7 wire c") is NOT expressed by selectors — that is the
 * job of the permutation argument, and skipping it is the single most common
 * way to build a forgeable circuit.
 */
import { FrField } from "../crypto/field";
import { nextPowerOfTwo, rootOfUnity } from "../crypto/ntt";
import { interpolate, type Poly } from "../crypto/poly";

export type VarId = number;

export type VarKind = "secret" | "public" | "internal" | "constant";

export interface Variable {
  id: VarId;
  name: string;
  value: bigint;
  kind: VarKind;
}

export interface Gate {
  qL: bigint;
  qR: bigint;
  qO: bigint;
  qM: bigint;
  qC: bigint;
  a: VarId;
  b: VarId;
  c: VarId;
  note?: string;
}

export interface PublicInput {
  name: string;
  row: number;
  varId: VarId;
  value: bigint;
}

const K1 = 7n;
const K2 = 13n;

export class CircuitBuilder {
  readonly variables: Variable[] = [];
  readonly gates: Gate[] = [];
  readonly publicInputs: PublicInput[] = [];
  private zeroVar: VarId;
  private oneVar: VarId;

  constructor() {
    // Variable 0 is the canonical zero. Row 0 pins it with qL = 1, qC = 0.
    this.zeroVar = this.alloc("$zero", 0n, "constant");
    this.pushGate(1n, 0n, 0n, 0n, 0n, this.zeroVar, this.zeroVar, this.zeroVar, "pin zero");
    this.oneVar = this.alloc("$one", 1n, "constant");
    this.pushGate(1n, 0n, 0n, 0n, FrField.neg(1n), this.oneVar, this.zeroVar, this.zeroVar, "pin one");
  }

  get zero(): VarId {
    return this.zeroVar;
  }

  get one(): VarId {
    return this.oneVar;
  }

  private alloc(name: string, value: bigint, kind: VarKind): VarId {
    const id = this.variables.length;
    this.variables.push({ id, name, value: FrField.reduce(value), kind });
    return id;
  }

  private pushGate(
    qL: bigint,
    qR: bigint,
    qO: bigint,
    qM: bigint,
    qC: bigint,
    a: VarId,
    b: VarId,
    c: VarId,
    note?: string,
  ): number {
    this.gates.push({
      qL: FrField.reduce(qL),
      qR: FrField.reduce(qR),
      qO: FrField.reduce(qO),
      qM: FrField.reduce(qM),
      qC: FrField.reduce(qC),
      a,
      b,
      c,
      note,
    });
    return this.gates.length - 1;
  }

  /** Allocate a witness value known only to the prover. */
  secret(name: string, value: bigint): VarId {
    return this.alloc(name, value, "secret");
  }

  /** Allocate an internal (intermediate) value. */
  internal(name: string, value: bigint): VarId {
    return this.alloc(name, value, "internal");
  }

  /** Allocate a public input and constrain it to `value`. */
  publicInput(name: string, value: bigint): VarId {
    const v = FrField.reduce(value);
    const id = this.alloc(name, v, "public");
    const row = this.pushGate(1n, 0n, 0n, 0n, 0n, id, this.zeroVar, this.zeroVar, `public ${name}`);
    this.publicInputs.push({ name, row, varId: id, value: v });
    return id;
  }

  /** Allocate a constant (prover-side) variable, pinned by a gate. */
  constant(name: string, value: bigint): VarId {
    const v = FrField.reduce(value);
    const id = this.alloc(name, v, "constant");
    this.pushGate(1n, 0n, 0n, 0n, FrField.neg(v), id, this.zeroVar, this.zeroVar, `constant ${name}`);
    return id;
  }

  /** c = a * b */
  mul(a: VarId, b: VarId, name = "mul"): VarId {
    const c = this.alloc(name, this.value(a) * this.value(b), "internal");
    this.pushGate(0n, 0n, -1n, 1n, 0n, a, b, c, `${name} = a*b`);
    return c;
  }

  /** c = a + b */
  add(a: VarId, b: VarId, name = "add"): VarId {
    const c = this.alloc(name, this.value(a) + this.value(b), "internal");
    this.pushGate(1n, 1n, -1n, 0n, 0n, a, b, c, `${name} = a+b`);
    return c;
  }

  /** c = a + k */
  addConst(a: VarId, k: bigint, name = "addc"): VarId {
    const c = this.alloc(name, this.value(a) + k, "internal");
    this.pushGate(1n, 0n, -1n, 0n, k, a, this.zeroVar, c, `${name} = a+${k}`);
    return c;
  }

  /** c = a - b */
  sub(a: VarId, b: VarId, name = "sub"): VarId {
    const c = this.alloc(name, this.value(a) - this.value(b), "internal");
    this.pushGate(1n, -1n, -1n, 0n, 0n, a, b, c, `${name} = a-b`);
    return c;
  }

  /** Enforce a == b. */
  assertEqual(a: VarId, b: VarId, note = "assert a == b"): void {
    this.pushGate(1n, -1n, 0n, 0n, 0n, a, b, this.zeroVar, note);
  }

  /** Enforce a * b == c. */
  assertMul(a: VarId, b: VarId, c: VarId, note = "assert a*b == c"): void {
    this.pushGate(0n, 0n, -1n, 1n, 0n, a, b, c, note);
  }

  /** Enforce a != 0 by providing its inverse: a * inv == 1. */
  assertNonZero(a: VarId, name = "nonzero"): VarId {
    const inv = this.alloc(name, inverseOrThrow(this.value(a)), "internal");
    this.pushGate(0n, 0n, -1n, 1n, 0n, a, inv, this.oneVar, "a * a^-1 == 1");
    return inv;
  }

  /** Enforce a ∈ {0, 1}. */
  assertBool(a: VarId, note = "bool(a)"): void {
    this.pushGate(-1n, 0n, 0n, 1n, 0n, a, a, this.zeroVar, note);
  }

  /** out = cond ? x : y. Costs three gates (no custom selectors here). */
  select(cond: VarId, x: VarId, y: VarId, name = "select"): VarId {
    const diff = this.sub(x, y, name + "_diff");
    const prod = this.mul(cond, diff, name + "_prod");
    return this.add(prod, y, name);
  }

  /** Enforce a == k for a constant k chosen by the circuit. */
  assertConstant(a: VarId, k: bigint, note = "assert a == k"): void {
    this.pushGate(1n, 0n, 0n, 0n, FrField.neg(k), a, this.zeroVar, this.zeroVar, note);
  }

  private value(id: VarId): bigint {
    return this.variables[id].value;
  }

  /** Rows currently used (before padding). */
  get numGates(): number {
    return this.gates.length;
  }

  /**
   * Fix up any intermediate variable whose value was computed with plain
   * bigints (mul/add above) so that it is reduced into the field.
   */
  finalize(): CompiledCircuit {
    for (const v of this.variables) v.value = FrField.reduce(v.value);
    return compile(this);
  }
}

function inverseOrThrow(v: bigint): bigint {
  const r = FrField.reduce(v);
  if (r === 0n) throw new Error("cannot prove knowledge of a zero inverse");
  return FrField.inv(r);
}

export interface CompiledCircuit {
  n: number;
  omega: bigint;
  k1: bigint;
  k2: bigint;
  /** Wire assignments per row, as variable ids. */
  wireIds: [VarId[], VarId[], VarId[]];
  /** Selector values on the domain (length n). */
  selectors: { qM: bigint[]; qL: bigint[]; qR: bigint[]; qO: bigint[]; qC: bigint[] };
  /** Permutation polynomial evaluations on the domain. */
  sigma: { s1: bigint[]; s2: bigint[]; s3: bigint[] };
  publicInputs: PublicInput[];
  numGates: number;
  variables: Variable[];
}

export function compile(b: CircuitBuilder): CompiledCircuit {
  const numGates = b.gates.length;
  const n = nextPowerOfTwo(Math.max(numGates, 8));
  const omega = rootOfUnityCached(n);

  const wireIds: [VarId[], VarId[], VarId[]] = [[], [], []];
  const q = { qM: [], qL: [], qR: [], qO: [], qC: [] } as Record<string, bigint[]>;
  for (const k of ["qM", "qL", "qR", "qO", "qC"]) q[k] = new Array<bigint>(n).fill(0n);

  for (let i = 0; i < n; i++) {
    const g = i < numGates ? b.gates[i] : null;
    if (g) {
      wireIds[0].push(g.a);
      wireIds[1].push(g.b);
      wireIds[2].push(g.c);
      q.qM[i] = g.qM;
      q.qL[i] = g.qL;
      q.qR[i] = g.qR;
      q.qO[i] = g.qO;
      q.qC[i] = g.qC;
    } else {
      // Padding row: no constraint, all wires on the zero variable.
      wireIds[0].push(b.zero);
      wireIds[1].push(b.zero);
      wireIds[2].push(b.zero);
    }
  }

  const sigma = buildPermutation(wireIds, n, omega);

  return {
    n,
    omega,
    k1: K1,
    k2: K2,
    wireIds,
    selectors: {
      qM: q.qM,
      qL: q.qL,
      qR: q.qR,
      qO: q.qO,
      qC: q.qC,
    },
    sigma,
    publicInputs: b.publicInputs,
    numGates,
    variables: b.variables,
  };
}

function rootOfUnityCached(n: number): bigint {
  return rootOfUnity(n);
}

/**
 * Build the three permutation polynomials from the wiring.
 *
 * We lay out 3n positions: slot* n + row. Two positions that hold the same
 * variable are linked in a cycle. s1, s2, s3 then encode, for each row, the
 * domain point of the *next* occurrence of that wire's variable.
 */
export function buildPermutation(wireIds: [VarId[], VarId[], VarId[]], n: number, omega: bigint): {
  s1: bigint[];
  s2: bigint[];
  s3: bigint[];
} {
  const occurrences = new Map<VarId, number[]>();
  for (let slot = 0; slot < 3; slot++) {
    for (let row = 0; row < n; row++) {
      const id = wireIds[slot][row];
      const pos = slot * n + row;
      const arr = occurrences.get(id);
      if (arr) arr.push(pos);
      else occurrences.set(id, [pos]);
    }
  }

  const next = new Array<number>(3 * n).fill(-1);
  for (const positions of occurrences.values()) {
    for (let k = 0; k < positions.length; k++) {
      next[positions[k]] = positions[(k + 1) % positions.length];
    }
  }

  const domainPoint = (pos: number): bigint => {
    const slot = Math.floor(pos / n);
    const row = pos % n;
    const x = FrField.pow(omega, BigInt(row));
    if (slot === 0) return x;
    if (slot === 1) return FrField.mul(K1, x);
    return FrField.mul(K2, x);
  };

  const s1 = new Array<bigint>(n);
  const s2 = new Array<bigint>(n);
  const s3 = new Array<bigint>(n);
  for (let row = 0; row < n; row++) {
    s1[row] = domainPoint(next[0 * n + row]);
    s2[row] = domainPoint(next[1 * n + row]);
    s3[row] = domainPoint(next[2 * n + row]);
  }
  return { s1, s2, s3 };
}

/** Interpolate the selector vectors into coefficient form. */
export function selectorsAsPolys(c: CompiledCircuit): {
  qM: Poly;
  qL: Poly;
  qR: Poly;
  qO: Poly;
  qC: Poly;
  s1: Poly;
  s2: Poly;
  s3: Poly;
} {
  return {
    qM: interpolate(c.selectors.qM),
    qL: interpolate(c.selectors.qL),
    qR: interpolate(c.selectors.qR),
    qO: interpolate(c.selectors.qO),
    qC: interpolate(c.selectors.qC),
    s1: interpolate(c.sigma.s1),
    s2: interpolate(c.sigma.s2),
    s3: interpolate(c.sigma.s3),
  };
}

// ---------------------------------------------------------------------------
// Witness audit
// ---------------------------------------------------------------------------

export interface RowViolation {
  row: number;
  lhs: bigint;
  note?: string;
}

/**
 * Evaluate every gate equation against the witness. A non-empty result means
 * the witness does not satisfy the circuit — the prover cannot produce a valid
 * proof, and saying *which* row is wrong is far more useful than a failed
 * quotient check deep inside the prover.
 */
export function auditWitness(c: CompiledCircuit): RowViolation[] {
  const out: RowViolation[] = [];
  const val = (id: VarId) => c.variables[id].value;
  for (let i = 0; i < c.numGates; i++) {
    const qM = c.selectors.qM[i];
    const qL = c.selectors.qL[i];
    const qR = c.selectors.qR[i];
    const qO = c.selectors.qO[i];
    const qC = c.selectors.qC[i];
    const a = val(c.wireIds[0][i]);
    const b = val(c.wireIds[1][i]);
    const cc = val(c.wireIds[2][i]);
    const lhs = (qM * a * b + qL * a + qR * b + qO * cc + qC) % FrField.p;
    const pi = c.publicInputs.filter((p) => p.row === i).reduce((acc, p) => (acc - p.value) % FrField.p, 0n);
    const total = (lhs + pi) % FrField.p;
    if (total !== 0n) out.push({ row: i, lhs: total });
  }
  return out;
}
