/**
 * Terminal demo: prove a MiMC preimage, verify it, then break a circuit by
 * deleting one constraint.
 *
 *   npm run demo
 */
import { mimcHash, mimcPreimageCircuit } from "../src/circuits/mimc";
import { auditWitness } from "../src/plonk/circuit";
import { proofHex, prove, serializeProof, setup, verify, verifyingKey } from "../src/plonk/plonk";
import { buildScenario } from "../src/scenarios";
import { FrField } from "../src/crypto/field";

const fmt = (ms: number) => (ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(2)} s`);
const bar = (s: string) => `\n\x1b[38;5;191m${s}\x1b[0m\n${"─".repeat(s.length)}`;

async function main() {
  console.log(bar("1 · MiMC preimage"));
  const x = 424242n;
  const k = 1337n;
  const y = mimcHash(x, k);
  console.log(`statement : MiMC(x, ${k}) = ${y.toString(16).slice(0, 24)}…`);
  console.log(`witness   : x = ${x}  (never leaves this process)`);

  const { circuit } = mimcPreimageCircuit(x, k, y);
  console.log(`circuit   : ${circuit.numGates} gates, domain n = ${circuit.n}`);
  console.log(`audit     : ${auditWitness(circuit).length} violated constraints`);

  let t = performance.now();
  const pk = setup(circuit);
  const vk = verifyingKey(pk);
  console.log(`preprocess: ${fmt(performance.now() - t)}`);

  t = performance.now();
  const { proof } = prove(pk);
  console.log(`prove     : ${fmt(performance.now() - t)}`);

  t = performance.now();
  const res = verify(vk, [k, y], proof);
  console.log(`verify    : ${fmt(performance.now() - t)}  →  ${res.ok ? "\x1b[32mACCEPT\x1b[0m" : "\x1b[31mREJECT\x1b[0m"}`);
  for (const c of res.checks) console.log(`   ${c.ok ? "✓" : "✗"} ${c.label}`);

  const bytes = serializeProof(proof);
  console.log(`proof     : ${bytes.length} bytes`);
  console.log(`            ${proofHex(proof).slice(0, 96)}…`);

  console.log(bar("2 · a false statement"));
  const honest = buildScenario("multiplier", { mx: "3", my: "11", mz: "34" });
  console.log(`“3 × 11 = 34”: ${auditWitness(honest.circuit).length} violated constraint(s) → the prover refuses.`);

  const broken = buildScenario("multiplier", { mx: "3", my: "11", mz: "34", dropConstraint: true });
  const pk2 = setup(broken.circuit);
  const proof2 = prove(pk2).proof;
  const res2 = verify(verifyingKey(pk2), broken.publicValues, proof2);
  console.log(`delete the gate qM·a·b − c = 0 …`);
  console.log(
    `verifier  : ${res2.ok ? "\x1b[31mACCEPTED A FALSE STATEMENT\x1b[0m" : "rejected"}  (${res2.checks.filter((c) => c.ok).length}/${res2.checks.length} checks pass)`,
  );
  console.log(
    `\nsame cryptography, same verifier — the only difference is one missing constraint.\nThat is why circuit correctness is the hard part.\n`,
  );
  void FrField;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
