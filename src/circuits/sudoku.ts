/**
 * Sudoku as a zero-knowledge statement.
 *
 * Statement: "I know a completed grid that agrees with these givens."
 * Nothing about the grid is revealed — not even which cells were the hard ones.
 *
 * Encoding (one-hot): cell (r,c) owns N boolean wires, one per digit. The
 * constraints are
 *
 *   1. every wire is boolean                       v·(v−1) = 0
 *   2. every cell holds exactly one digit          Σ_d v_{r,c,d} = 1
 *   3. every row/column/box holds each digit once  Σ_{cells in group} v = 1
 *   4. givens are pinned                           v_{r,c,d} = 1
 *
 * Constraint 2 looks redundant — it is not. Without it a cell could hold two
 * digits while another holds none, and every row/column/box sum would still be
 * one. That is exactly the "single missing constraint" failure mode, and this
 * circuit is a live demonstration of it in the Soundness lab.
 */
import { FrField } from "../crypto/field";
import { CircuitBuilder, type CompiledCircuit } from "../plonk/circuit";

export interface SudokuSpec {
  /** Size 4 (2x2 boxes) or 9 (3x3 boxes). */
  size: 4 | 9;
  /** Puzzle givens, 0 = empty. */
  puzzle: number[][];
  /** The full solution the prover knows. */
  solution: number[][];
  /** Skip the per-cell "exactly one digit" constraint (soundness lab). */
  omitCellSum?: boolean;
  /** Corrupt this cell of the witness to simulate a cheating prover. */
  tamper?: [number, number] | null;
}

export interface SudokuCircuitResult {
  circuit: CompiledCircuit;
  digits: number[][];
  boxes: number;
  size: number;
}

function boxIndex(size: number, r: number, c: number): number {
  const b = size === 4 ? 2 : 3;
  return Math.floor(r / b) * b + Math.floor(c / b);
}

export function sudokuCircuit(spec: SudokuSpec): SudokuCircuitResult {
  const { size, puzzle, solution } = spec;
  const b = new CircuitBuilder();
  const N = size;
  const boxDim = size === 4 ? 2 : 3;

  const digits = solution.map((row) => row.slice());
  if (spec.tamper) {
    const [r, c] = spec.tamper;
    // swap with a neighbour so the grid is still "answer-shaped" but wrong
    const other = digits[r][(c + 1) % N];
    digits[r][(c + 1) % N] = digits[r][c];
    digits[r][c] = other;
  }

  // one-hot wires
  const cell: number[][][] = [];
  for (let r = 0; r < N; r++) {
    cell.push([]);
    for (let c = 0; c < N; c++) {
      const bits: number[] = [];
      for (let d = 0; d < N; d++) {
        const v = digits[r][c] === d + 1 ? 1n : 0n;
        bits.push(b.secret(`c${r}${c}_d${d}`, v));
      }
      cell[r].push(bits);
    }
  }

  // 1. booleans
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) for (let d = 0; d < N; d++) b.assertBool(cell[r][c][d]);

  // 2. exactly one digit per cell
  if (!spec.omitCellSum) {
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const sum = sumAll(b, cell[r][c], `cell${r}${c}_sum`);
        b.assertConstant(sum, 1n, `cell ${r},${c} holds exactly one digit`);
      }
    }
  }

  // 3. each group has each digit exactly once
  for (let d = 0; d < N; d++) {
    for (let i = 0; i < N; i++) {
      // row i
      b.assertConstant(sumAll(b, rowBits(cell, i, d, N), `row${i}_d${d}`), 1n, `row ${i} has digit ${d + 1} once`);
      // column i
      const col = Array.from({ length: N }, (_, r) => cell[r][i][d]);
      b.assertConstant(sumAll(b, col, `col${i}_d${d}`), 1n, `column ${i} has digit ${d + 1} once`);
    }
    for (let bi = 0; bi < N; bi++) {
      const group: number[] = [];
      for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          if (boxIndex(N, r, c) === bi) group.push(cell[r][c][d]);
        }
      }
      b.assertConstant(sumAll(b, group, `box${bi}_d${d}`), 1n, `box ${bi} has digit ${d + 1} once`);
    }
  }

  // 4. givens
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const g = puzzle[r][c];
      if (g > 0) b.assertConstant(cell[r][c][g - 1], 1n, `given ${r},${c} = ${g}`);
    }
  }

  return { circuit: b.finalize(), digits, boxes: boxDim, size: N };
}

function rowBits(cell: number[][][], r: number, d: number, N: number): number[] {
  return Array.from({ length: N }, (_, c) => cell[r][c][d]);
}

/** Sum a list of variables with a binary tree, using `add` gates. */
function sumAll(b: CircuitBuilder, vars: number[], name: string): number {
  if (vars.length === 0) return b.zero;
  let level = vars.slice();
  let depth = 0;
  while (level.length > 1) {
    const next: number[] = [];
    for (let i = 0; i < level.length; i += 2) {
      if (i + 1 < level.length) next.push(b.add(level[i], level[i + 1], `${name}_${depth}_${i}`));
      else next.push(level[i]);
    }
    level = next;
    depth++;
  }
  return level[0];
}

// ---------------------------------------------------------------------------
// Puzzles
// ---------------------------------------------------------------------------

export const SUDOKU_9 = {
  puzzle: [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
  ],
  solution: [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
  ],
};

export const SUDOKU_4 = {
  puzzle: [
    [1, 0, 0, 0],
    [0, 4, 0, 2],
    [0, 0, 4, 0],
    [0, 0, 0, 1],
  ],
  solution: [
    [1, 2, 3, 4],
    [3, 4, 1, 2],
    [2, 1, 4, 3],
    [4, 3, 2, 1],
  ],
};

/** Verify a solution is a legal completed grid (used for UI hints, not proofs). */
export function isValidSolution(grid: number[][]): boolean {
  const N = grid.length;
  const target = (N * (N + 1)) / 2;
  for (let i = 0; i < N; i++) {
    let rowSum = 0;
    let colSum = 0;
    for (let j = 0; j < N; j++) {
      rowSum += grid[i][j];
      colSum += grid[j][i];
    }
    if (rowSum !== target || colSum !== target) return false;
  }
  const boxDim = Math.sqrt(N);
  for (let br = 0; br < N; br += boxDim) {
    for (let bc = 0; bc < N; bc += boxDim) {
      let sum = 0;
      for (let r = 0; r < boxDim; r++) for (let c = 0; c < boxDim; c++) sum += grid[br + r][bc + c];
      if (sum !== target) return false;
    }
  }
  return true;
}

export { FrField };
