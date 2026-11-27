import { describe, expect, it } from "vitest";
import { auditWitness } from "../src/plonk/circuit";
import { prove, setup, verify, verifyingKey } from "../src/plonk/plonk";
import { mimcHash, mimcPreimageCircuit, MIMC_ROUNDS } from "../src/circuits/mimc";
import { SUDOKU_4, SUDOKU_9, isValidSolution, sudokuCircuit } from "../src/circuits/sudoku";
import { buildScenario } from "../src/scenarios";
import { compileDsl, EXAMPLE_PROGRAMS } from "../src/dsl/dsl";

describe("mimc", () => {
  it("is not the identity and has the right round count", () => {
    expect(mimcHash(1n, 2n)).not.toBe(1n);
    expect(MIMC_ROUNDS).toBe(110);
  });
  it("circuit agrees with the reference implementation", () => {
    const { circuit, expected } = mimcPreimageCircuit(12345n, 999n, mimcHash(12345n, 999n));
    expect(expected).toBe(mimcHash(12345n, 999n));
    expect(auditWitness(circuit)).toEqual([]);
  });
  it("proves and verifies a preimage", () => {
    const x = 424242n;
    const k = 1337n;
    const y = mimcHash(x, k);
    const { circuit } = mimcPreimageCircuit(x, k, y);
    const pk = setup(circuit);
    const { proof } = prove(pk);
    expect(verify(verifyingKey(pk), [k, y], proof).ok).toBe(true);
  }, 180000);
  it("refuses a wrong preimage", () => {
    const y = mimcHash(424242n, 1337n);
    const { circuit } = mimcPreimageCircuit(1n, 1337n, y); // wrong x
    const violations = auditWitness(circuit);
    expect(violations.length).toBeGreaterThan(0);
    expect(() => {
      const pk = setup(circuit);
      prove(pk);
    }).toThrow();
  }, 180000);
});

describe("sudoku", () => {
  it("solutions are valid grids", () => {
    expect(isValidSolution(SUDOKU_4.solution)).toBe(true);
    expect(isValidSolution(SUDOKU_9.solution)).toBe(true);
  });
  it("builds a satisfiable 4x4 circuit", () => {
    const { circuit } = sudokuCircuit({ size: 4, puzzle: SUDOKU_4.puzzle, solution: SUDOKU_4.solution });
    expect(auditWitness(circuit)).toEqual([]);
  });
  it("proves a 4x4 grid", () => {
    const { circuit } = sudokuCircuit({ size: 4, puzzle: SUDOKU_4.puzzle, solution: SUDOKU_4.solution });
    const pk = setup(circuit);
    const { proof } = prove(pk);
    expect(verify(verifyingKey(pk), [], proof).ok).toBe(true);
  }, 180000);
  it("catches a tampered grid", () => {
    const { circuit } = sudokuCircuit({ size: 4, puzzle: SUDOKU_4.puzzle, solution: SUDOKU_4.solution, tamper: [0, 0] });
    expect(auditWitness(circuit).length).toBeGreaterThan(0);
  });
  it("4x4 without per-cell sums accepts a grid with two digits in one cell", () => {
    // Drop the per-cell constraints *and* give a cell two digits; the remaining
    // constraints can still be satisfied, which is precisely the unsoundness.
    const { circuit } = sudokuCircuit({ size: 4, puzzle: SUDOKU_4.puzzle, solution: SUDOKU_4.solution, omitCellSum: true });
    expect(auditWitness(circuit)).toEqual([]);
    expect(circuit.numGates).toBeLessThan(400);
  });
});

describe("dsl", () => {
  it("compiles the multiplication example", () => {
    const res = compileDsl(EXAMPLE_PROGRAMS[0].source);
    expect(res.publicValues).toEqual([91n]);
    expect(res.circuit.numGates).toBeGreaterThan(4);
    expect(auditWitness(res.circuit)).toEqual([]);
  });
  it("compiles all examples", () => {
    for (const ex of EXAMPLE_PROGRAMS) {
      const res = compileDsl(ex.source);
      expect(auditWitness(res.circuit), ex.name).toEqual([]);
    }
  });
  it("reports a bad witness", () => {
    const res = compileDsl(`secret x = 3\npublic y = 10\nassert x * x == y`);
    expect(auditWitness(res.circuit).length).toBeGreaterThan(0);
  });
  it("rejects syntax errors with a position", () => {
    expect(() => compileDsl("secret x = ")).toThrow(/line/);
    expect(() => compileDsl("bogus 3")).toThrow(/expected a statement/);
  });
  it("proves a DSL circuit", () => {
    const res = compileDsl(EXAMPLE_PROGRAMS[3].source);
    const pk = setup(res.circuit);
    const { proof } = prove(pk);
    expect(verify(verifyingKey(pk), res.publicValues, proof).ok).toBe(true);
  }, 180000);
});

describe("scenarios", () => {
  it("builds every scenario", () => {
    for (const id of ["multiplier", "mimc", "sudoku4", "dsl"] as const) {
      const s = buildScenario(id, {});
      expect(s.circuit.numGates).toBeGreaterThan(0);
    }
  });
  it("the dropped constraint really is dropped", () => {
    const honest = buildScenario("multiplier", { mx: "3", my: "11", mz: "34" });
    expect(auditWitness(honest.circuit).length).toBeGreaterThan(0);
    const broken = buildScenario("multiplier", { mx: "3", my: "11", mz: "34", dropConstraint: true });
    expect(auditWitness(broken.circuit)).toEqual([]);
  });
});
