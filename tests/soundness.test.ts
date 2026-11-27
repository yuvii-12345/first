/**
 * The soundness contract of the demo UI: every mutation the Soundness tab can
 * perform on a real proof must be rejected, and the untouched proof accepted.
 */
import { describe, expect, it } from "vitest";
import { applyTamper, executeRun, type StoredRun } from "../src/app/runPipeline";
import { deserializeProof } from "../src/plonk/plonk";
import type { TamperTarget } from "../src/app/protocol";
import { buildScenario } from "../src/scenarios";
import { auditWitness } from "../src/plonk/circuit";

const TARGETS: TamperTarget[] = [
  "eval-aBar",
  "eval-zOmegaBar",
  "eval-qMBar",
  "eval-t1Bar",
  "commitment-z",
  "commitment-wZeta",
  "public-value",
];

describe("soundness of the tamper lab", () => {
  const stored = (() => {
    let captured: StoredRun | undefined;
    const run = executeRun(4100, "multiplier", { mx: "7", my: "13" }, () => {}, (s) => {
      captured = s;
    });
    if (!captured) throw new Error("run did not store its proof");
    return { run, stored: captured as StoredRun };
  })();

  it("accepts the honest proof", () => {
    expect(stored.run.ok).toBe(true);
    expect(stored.run.circuit.proofBytes).toBe(1088);
    expect(applyTamper(stored.stored, "none", deserializeProof).ok).toBe(true);
  });

  for (const target of TARGETS) {
    it(`rejects ${target}`, () => {
      const res = applyTamper(stored.stored, target, deserializeProof);
      expect(res.ok).toBe(false);
      expect(res.checks.some((c) => !c.ok)).toBe(true);
    });
  }

  it("refuses to prove a false statement", () => {
    // The prover audits the witness and then, belt and braces, checks that the
    // quotient really divides by Z_H. A violated witness fails both.
    expect(() => executeRun(4101, "multiplier", { mx: "3", my: "11", mz: "34" }, () => {})).toThrow(
      /does not vanish on H/,
    );
    const bad = buildScenario("multiplier", { mx: "3", my: "11", mz: "34" });
    expect(auditWitness(bad.circuit).length).toBeGreaterThan(0);
  });

  it("accepts a false statement when a constraint is deleted", () => {
    const forged = executeRun(4102, "multiplier", { mx: "3", my: "11", mz: "34", dropConstraint: true }, () => {});
    expect(forged.audit.violations).toBe(0);
    expect(forged.ok).toBe(true);
    expect(forged.meta.unsound).toBe(true);
  });
});
