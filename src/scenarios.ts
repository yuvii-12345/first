/**
 * The demo library: each scenario knows how to build a circuit + statement from
 * a few knobs, without touching the cryptography.
 */
import type { CompiledCircuit } from "./plonk/circuit";
import { CircuitBuilder } from "./plonk/circuit";
import { mimcHash, mimcPreimageCircuit } from "./circuits/mimc";
import { SUDOKU_4, SUDOKU_9, sudokuCircuit } from "./circuits/sudoku";
import { compileDsl, EXAMPLE_PROGRAMS } from "./dsl/dsl";
import { FrField } from "./crypto/field";

export type ScenarioId = "multiplier" | "mimc" | "sudoku4" | "sudoku9" | "dsl";

export interface ScenarioOptions {
  // multiplier
  mx?: string;
  my?: string;
  mz?: string;
  dropConstraint?: boolean;
  // mimc
  preimage?: string;
  key?: string;
  digest?: string;
  // sudoku
  tamper?: boolean;
  omitCellSum?: boolean;
  // dsl
  source?: string;
}

export interface ScenarioMeta {
  /** Human-readable statement shown to the "verifier". */
  statement: string;
  publicLabels: string[];
  notes: string[];
  /** Extra structured info for specialised views. */
  grid?: { puzzle: number[][]; solution: number[][]; size: number };
  dslSource?: string;
  dslError?: string;
  /** Whether the circuit intentionally omits a constraint (soundness lab). */
  unsound?: boolean;
}

export interface BuiltScenario {
  id: ScenarioId;
  circuit: CompiledCircuit;
  publicValues: bigint[];
  meta: ScenarioMeta;
}

function big(s: string | undefined, fallback: bigint): bigint {
  if (s === undefined || s.trim() === "") return fallback;
  try {
    const v = BigInt(s.trim());
    return FrField.reduce(v);
  } catch {
    return fallback;
  }
}

export function buildScenario(id: ScenarioId, opts: ScenarioOptions): BuiltScenario {
  switch (id) {
    case "multiplier": {
      const x = big(opts.mx, 7n);
      const y = big(opts.my, 13n);
      const z = big(opts.mz, 91n);
      const b = new CircuitBuilder();
      const sx = b.secret("x", x);
      const sy = b.secret("y", y);
      const pz = b.publicInput("z", z);
      const prod = b.mul(sx, sy, "x*y");
      if (!opts.dropConstraint) b.assertEqual(prod, pz, "x*y == z");
      return {
        id,
        circuit: b.finalize(),
        publicValues: [z],
        meta: {
          statement: `z = ${z}`,
          publicLabels: ["z"],
          notes: opts.dropConstraint
            ? ["The constraint x·y = z has been deleted. The prover can now 'prove' statements that are false."]
            : ["x and y stay secret; only their product is revealed."],
          unsound: !!opts.dropConstraint,
        },
      };
    }
    case "mimc": {
      const x = big(opts.preimage, 424242n);
      const k = big(opts.key, 1337n);
      const y = opts.digest !== undefined && opts.digest.trim() !== "" ? big(opts.digest, 0n) : mimcHash(x, k);
      const { circuit } = mimcPreimageCircuit(x, k, y);
      return {
        id,
        circuit,
        publicValues: [k, y],
        meta: {
          statement: `MiMC(x, k) = y with k = ${k}`,
          publicLabels: ["k (key)", "y (digest)"],
          notes: [
            `MiMC-5, 110 rounds — ${circuit.numGates} gates for one hash.`,
            "A SHA-256 preimage would need ~30,000 gates: bit operations are expensive inside a SNARK.",
          ],
        },
      };
    }
    case "sudoku4":
    case "sudoku9": {
      const size = id === "sudoku4" ? 4 : 9;
      const base = size === 4 ? SUDOKU_4 : SUDOKU_9;
      const { circuit } = sudokuCircuit({
        size: size as 4 | 9,
        puzzle: base.puzzle,
        solution: base.solution,
        omitCellSum: opts.omitCellSum,
        tamper: opts.tamper ? [0, 0] : null,
      });
      const grid = {
        puzzle: base.puzzle,
        solution: opts.tamper ? tamperGrid(base.solution) : base.solution,
        size,
      };
      return {
        id,
        circuit,
        publicValues: [],
        meta: {
          statement: `a completed ${size}×${size} grid consistent with the givens`,
          publicLabels: [],
          notes: [
            "The verifier sees only the puzzle. The solution stays private.",
            "One-hot encoding: every cell owns " + size + " boolean wires.",
            opts.omitCellSum ? "Per-cell 'exactly one digit' constraints have been DELETED." : "Per-cell sum constraints included.",
          ],
          grid,
          unsound: !!opts.omitCellSum,
        },
      };
    }
    case "dsl": {
      const source = opts.source ?? EXAMPLE_PROGRAMS[0].source;
      try {
        const res = compileDsl(source);
        return {
          id,
          circuit: res.circuit,
          publicValues: res.publicValues,
          meta: {
            statement: res.publicNames.map((n, i) => `${n} = ${res.publicValues[i]}`).join(", ") || "(no public inputs)",
            publicLabels: res.publicNames,
            notes: [`Compiled ${res.statements} statements into ${res.circuit.numGates} gates.`],
            dslSource: source,
          },
        };
      } catch (err) {
        // Fall back to the smallest possible circuit so the UI stays alive.
        const b = new CircuitBuilder();
        const s = b.secret("x", 1n);
        b.publicInput("y", 1n);
        b.assertEqual(s, b.one, "fallback");
        return {
          id,
          circuit: b.finalize(),
          publicValues: [1n],
          meta: {
            statement: "compile error",
            publicLabels: ["y"],
            notes: [],
            dslSource: source,
            dslError: (err as Error).message,
          },
        };
      }
    }
  }
}

function tamperGrid(grid: number[][]): number[][] {
  const out = grid.map((r) => r.slice());
  const tmp = out[0][0];
  out[0][0] = out[0][1];
  out[0][1] = tmp;
  return out;
}

export const SCENARIO_INFO: Array<{ id: ScenarioId; title: string; subtitle: string; badge: string }> = [
  { id: "multiplier", title: "Multiplication", subtitle: "The smallest possible SNARK statement", badge: "5 gates" },
  { id: "mimc", title: "MiMC preimage", subtitle: "Prove you know the input to a hash", badge: "555 gates" },
  { id: "sudoku4", title: "Sudoku 4×4", subtitle: "A completed grid, revealed to nobody", badge: "512 rows" },
  { id: "sudoku9", title: "Sudoku 9×9", subtitle: "The classic — 3,677 constraints", badge: "4,096 rows" },
  { id: "dsl", title: "Circuit DSL", subtitle: "Write your own statement", badge: "you decide" },
];
