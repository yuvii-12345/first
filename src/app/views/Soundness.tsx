import { useState } from "react";
import type { ZerothState } from "../useZeroth";
import type { ScenarioId, ScenarioOptions } from "../../scenarios";
import type { TamperTarget } from "../protocol";
import { Badge, CheckRow, Panel, SectionHead, Spinner, Switch, fmtMs } from "../components";

export function SoundnessView({
  state,
  tamper,
  run,
}: {
  state: ZerothState;
  tamper: (t: TamperTarget) => void;
  run: (s: ScenarioId, o: ScenarioOptions) => void;
}) {
  const [drop, setDrop] = useState(false);
  const busy = state.phase === "running";
  const result = state.result;
  const mult = result?.scenario === "multiplier" ? result : null;
  const tamperedProof = state.tamper;

  const runLab1 = () => {
    run("multiplier", { mx: "3", my: "11", mz: "34", dropConstraint: drop });
  };

  return (
    <>
      <SectionHead
        kicker="04 — soundness"
        title={
          <>
            One missing constraint
            <br />
            <em>turns a proof system into a rubber stamp.</em>
          </>
        }
        lead={
          <>
            A zk-SNARK is not "cryptography that makes things private". It is a claim that{" "}
            <em>every</em> honest verifier accepts only true statements. Soundness is a property of the
            whole stack: the circuit, the wiring, the challenges, the pairing checks. Below, break each part
            on purpose.
          </>
        }
        right={<Badge tone="bad">deliberately vulnerable</Badge>}
      />

      <div className="grid-2">
        <Panel
          title="Lab 01 · delete a constraint"
          right={<Badge tone={drop ? "bad" : "ok"}>{drop ? "unsound circuit" : "sound circuit"}</Badge>}
        >
          <p style={{ fontSize: 13 }}>
            The statement is <span className="mono">3 × 11 = 34</span>. It is false. With the constraint{" "}
            <span className="mono">x·y = z</span> present, no proof exists. Delete that one gate and the
            system will happily certify it.
          </p>
          <div className="formula" style={{ marginBottom: 14 }}>
            {drop ? (
              <>
                gate: <span className="v">qM·a·b + qO·c</span> = 0 <span className="dim">// product computed…</span>
                {"\n"}assert a·b == c  <span style={{ color: "var(--red)" }}>// deleted</span>
              </>
            ) : (
              <>
                gate: qM·a·b + qO·c = 0{"\n"}assert <span className="v">a·b == c</span>{" "}
                <span className="dim">// present</span>
              </>
            )}
          </div>
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
            <Switch checked={drop} onChange={setDrop} label="delete the x·y = z constraint" />
            <button className="btn primary small" onClick={runLab1} disabled={busy}>
              {busy ? <Spinner /> : "▸"} prove & verify
            </button>
          </div>

          {mult && (
            <div className={`callout ${mult.ok ? "danger" : ""}`}>
              {mult.ok ? (
                <>
                  <b style={{ color: "var(--red)" }}>Verifier accepted.</b> The prover produced a valid proof
                  for 3 × 11 = 34 — a false statement. Nothing in the cryptography failed; the{" "}
                  <em>circuit</em> was wrong. This is the failure mode the entire field worries about.
                </>
              ) : (
                <>
                  <b>Verifier rejected / prover refused.</b>{" "}
                  {mult.audit.violations > 0
                    ? `The witness violates ${mult.audit.violations} gate equation(s) (rows ${mult.audit.rows.join(", ")}), so the prover refused to emit a proof.`
                    : "The proof did not satisfy the verifier's checks."}
                </>
              )}
            </div>
          )}
          {!mult && (
            <p className="dim mono" style={{ fontSize: 11 }}>
              run this lab to see the verdict
            </p>
          )}
        </Panel>

        <Panel
          title="Lab 02 · tamper with a proof"
          right={<Badge tone={tamperedProof ? "bad" : undefined}>{tamperedProof ? "tampered" : "untouched"}</Badge>}
        >
          <p style={{ fontSize: 13 }}>
            Take an honest proof, change one number, and watch which check catches it. Every element is
            bound by either the transcript or a pairing — there is no place to hide a fudge.
          </p>
          <div className="row wrap" style={{ gap: 6, marginBottom: 14 }}>
            <button
              className="btn small"
              onClick={() => run("multiplier", { mx: "3", my: "11", mz: "33" })}
              disabled={busy}
            >
              prove honestly first
            </button>
            {(
              [
                ["eval-aBar", "ā += 1"],
                ["eval-zOmegaBar", "z̄ω += 1"],
                ["eval-qMBar", "q̄M += 1"],
                ["eval-t1Bar", "t̄₁ += 1"],
                ["commitment-z", "[z] ← [t₀]"],
                ["public-value", "statement += 1"],
              ] as Array<[TamperTarget, string]>
            ).map(([target, label]) => (
              <button
                key={target}
                className="btn small danger"
                onClick={() => tamper(target)}
                disabled={!result || state.tampering}
              >
                {label}
              </button>
            ))}
          </div>

          {tamperedProof ? (
            <>
              <div className="callout danger" style={{ marginBottom: 10 }}>
                <b>rejected</b> — a single modified field element is enough.
              </div>
              <div style={{ border: "1px solid var(--line)", borderRadius: 3 }}>
                {tamperedProof.result.checks.map((c) => (
                  <CheckRow key={c.id} check={c} />
                ))}
              </div>
            </>
          ) : (
            <p className="dim mono" style={{ fontSize: 11 }}>
              {result ? "pick a mutation above" : "no proof in memory yet"}
            </p>
          )}
        </Panel>
      </div>

      <div className="grid-3" style={{ marginTop: 18 }}>
        <Panel title="What the checks buy">
          <ul style={{ fontSize: 12.5, color: "var(--muted)", paddingLeft: 18, margin: 0, lineHeight: 1.7 }}>
            <li>
              <b className="accent">curve</b> — every group element is on the curve, so no invalid point
              tricks the pairing.
            </li>
            <li>
              <b className="accent">identity</b> — the gate, copy and boundary equations hold at a random ζ,
              which means (with overwhelming probability) they hold everywhere on H.
            </li>
            <li>
              <b className="accent">openings</b> — the evaluations the identity used are the committed
              polynomials' true evaluations, enforced by bilinear pairings.
            </li>
          </ul>
        </Panel>

        <Panel title="Other ways to break it">
          <ul style={{ fontSize: 12.5, color: "var(--muted)", paddingLeft: 18, margin: 0, lineHeight: 1.7 }}>
            <li>
              <b>Forget to absorb a commitment</b> before squeezing a challenge → the prover can grind it.
            </li>
            <li>
              <b>Keep τ</b> ("toxic waste") → anyone can open a commitment to any value.
            </li>
            <li>
              <b>Skip the G2 subgroup check</b> → small-subgroup attacks on the pairing.
            </li>
            <li>
              <b>No blinding</b> in a, b, c, z → the proof leaks the witness.
            </li>
            <li>
              <b>Reuse ζ</b> across different circuits without care → batch-opening forgery.
            </li>
          </ul>
        </Panel>

        <Panel title={'What "zero-knowledge" means here'}>
          <p style={{ fontSize: 12.5 }}>
            a, b, c and z carry random multiples of Z_H, so their commitments are uniformly distributed and
            independent of the witness. Everything the verifier sees is either a commitment under a hiding
            assumption or a value at a random point.
          </p>
          <p style={{ fontSize: 12.5, marginBottom: 0 }}>
            The verifier learns exactly one bit: <em>the statement is true</em>. Not the witness, not the
            size of the numbers, not which gate was hard.
          </p>
        </Panel>
      </div>

      <Callout />

      <div className="row" style={{ marginTop: 16, gap: 10 }}>
        <button className="btn" onClick={() => run("sudoku9", {})} disabled={busy}>
          run the honest 9×9 Sudoku
        </button>
        <button className="btn" onClick={() => run("mimc", {})} disabled={busy}>
          run the MiMC preimage
        </button>
        <span className="dim mono" style={{ fontSize: 11 }}>
          last run: {result ? `${result.scenario} · ${fmtMs(result.timings.total)} · ${result.ok ? "accepted" : "rejected"}` : "none"}
        </span>
      </div>

    </>
  );
}

function Callout() {
  return (
    <div className="callout warn" style={{ marginTop: 18 }}>
      <b>The uncomfortable part.</b> Nothing above is a bug in the arithmetic — the field, the curve, the
      pairing and the KZG checks are all correct. Soundness is a <em>global</em> property: it fails if any
      one of the thousands of intended constraints is missing, mis-wired, or enforced at the wrong place in
      the protocol. That is why real systems ship circuit auditors, and why "the proof verifies" is not the
      same sentence as "the statement is true".
    </div>
  );
}
