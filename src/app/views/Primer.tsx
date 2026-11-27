import { useState } from "react";
import { Badge, Panel, SectionHead, Spinner } from "../components";
import { FrField, Fp12 } from "../../crypto/field";
import { G1_GEN, G2_GEN, g1Mul, g2Mul } from "../../crypto/curve";
import { pairing } from "../../crypto/pairing";

export function PrimerView() {
  return (
    <>
      <SectionHead
        kicker="05 — primer"
        title={
          <>
            From bare arithmetic
            <br />
            <em>to a proof you can hold in your hand.</em>
          </>
        }
        lead={
          <>
            Nothing in this page's bundle imports a cryptography library. Everything is built from integers
            modulo a prime. This is the ladder, one rung at a time, with the actual formulas the code uses.
          </>
        }
        right={<Badge>~11k lines · 0 dependencies</Badge>}
      />

      <div className="grid-2" style={{ gridTemplateColumns: "minmax(0,1fr) 320px" }}>
        <div className="prose">
          <h2>0 · The ladder</h2>
          <p>
            Each layer exists for exactly one reason. Reading bottom-up, every rung answers a question the
            rung below cannot.
          </p>
          <div className="formula">
            {`PLONK protocol        ── "here is a proof for this statement"
  KZG commitments     ── "a short fingerprint of a polynomial"
    NTT / FFT         ── "polynomials in O(n log n), not O(n²)"
      pairing e(P,Q)  ── "check a polynomial equation without the polynomial"
        elliptic curve ── "a group where discrete log is hard"
          finite field ── "arithmetic that wraps around, with inverses"`}
          </div>

          <h2>1 · Fields</h2>
          <p>
            A prime field <code>F_p</code> is the integers mod a prime. Addition, subtraction and
            multiplication wrap; division works because every non-zero element has an inverse (Bezout's
            identity — the code uses the extended Euclidean algorithm, not Fermat's little theorem).
          </p>
          <p>
            Two fields show up. <code>F_q</code> holds curve coordinates; <code>F_r</code> holds the witness,
            the polynomials and the challenges. Both are 254 bits.
          </p>
          <div className="formula">
            {`q = 21888242871839275222246405745257275088696311157297823662689037894645226208583
r = 21888242871839275222246405745257275088548364400416034343698204186575808495617`}
          </div>
          <InteractiveField />

          <h2>2 · The curve</h2>
          <p>
            BN254 is <code>y² = x³ + 3</code> over <code>F_q</code>. The points form a group of order{" "}
            <code>r</code> — the same prime as the scalar field, which is no accident: it is what makes the
            pairing land in the right place.
          </p>
          <p>
            Additions use Jacobian coordinates so no inversion is needed until the very end; scalar
            multiplication is a double-and-add loop. Pippenger's bucket method turns <em>n</em> separate
            multiscalar multiplications into one pass, which is what makes commitments practical.
          </p>
          <div className="formula">
            {`P + Q:  λ = (y₂−y₁)/(x₂−x₁),  x₃ = λ²−x₁−x₂,  y₃ = λ(x₁−x₃)−y₁
r·P = O  for every P  (the group order is the scalar field's modulus)`}
          </div>

          <h2>3 · The pairing</h2>
          <p>
            The pairing is a bilinear map <code>e: G₁ × G₂ → G_T</code> into the multiplicative group of{" "}
            <code>F_q¹²</code>:
          </p>
          <div className="formula">{"e(aP, bQ) = e(P, Q)^(a·b)"}</div>
          <p>
            That single equation is the engine of the whole system. It lets a verifier compare
            <em> hidden</em> scalars: if both sides of a polynomial identity are committed, the check happens
            in the exponent, where only multiplication is available — and the pairing is what turns
            multiplication back into something checkable.
          </p>
          <p>
            BN254 needs a tower of extensions (<code>F_q² ⊂ F_q⁶ ⊂ F_q¹²</code>) and a sextic twist so that
            G₂ points can live over <code>F_q²</code>. The Miller loop accumulates line functions;
            a final exponentiation by <code>(q¹²−1)/r</code> projects the result into the order-<em>r</em>
            subgroup.
          </p>
          <InteractivePairing />

          <h2>4 · NTTs</h2>
          <p>
            Polynomials are the data structure of a SNARK, and multiplying them naively is O(n²). The
            number-theoretic transform evaluates a polynomial at all <em>n</em>-th roots of unity in
            O(n log n); pointwise multiplication plus the inverse transform gives fast convolution.
          </p>
          <p>
            The scalar field has 2-adicity 28, so domains up to 2²⁸ exist. One subtlety the code leans on:
            to divide by the vanishing polynomial <code>Z_H(X) = Xⁿ − 1</code> we must evaluate
            <em>off</em> the domain, otherwise we divide by zero. A coset <code>g·H</code> with{" "}
            <code>g ∉ H</code> does the job.
          </p>
          <div className="formula">
            {`NTT: a(X) ↦ (a(1), a(ω), …, a(ω^{n−1}))    in O(n log n)
Z_H(g·ωⁱ) = gⁿ − 1 ≠ 0`}
          </div>

          <h2>5 · KZG commitments</h2>
          <p>
            A commitment is a hash that is also algebraically structured. Setup samples a secret τ; the
            public parameters are <code>[τ⁰]₁, [τ¹]₁, …</code> and <code>[1]₂, [τ]₂</code>.
          </p>
          <div className="formula">
            {`commit(p) = [p(τ)]₁
open(p, z) = [(p(X) − p(z)) / (X − z)]₁
verify: e(C − [p(z)]₁, [1]₂) = e(π, [τ − z]₂)`}
          </div>
          <p>
            The opening is only computable by someone who knows p, and the pairing check forces the
            committed polynomial to pass through <code>(z, p(z))</code>. Whoever knows τ can open anything
            to anything — τ is the "toxic waste" and must be destroyed.
          </p>

          <h2>6 · PLONK</h2>
          <p>
            The circuit is a table. Each row has three wires and five selectors, and one equation:
          </p>
          <div className="formula">q_M·a·b + q_L·a + q_R·b + q_O·c + q_C = 0</div>
          <p>
            That is enough for multiplication, addition, constants, booleans and public inputs. What it
            cannot say is "the value in row 3, wire a is the same as the value in row 7, wire c" — so the
            wires are interpolated into polynomials <code>a(X), b(X), c(X)</code> and tied together by a
            permutation σ over three cosets of the domain:
          </p>
          <div className="formula">
            {`∏ (a + βx + γ)(b + βk₁x + γ)(c + βk₂x + γ)·z(x)
  = ∏ (a + βσ₁ + γ)(b + βσ₂ + γ)(c + βσ₃ + γ)·z(ωx)`}
          </div>
          <p>
            The prover builds the running product <code>z</code>, packs everything into one polynomial that
            vanishes on the domain, and divides:
          </p>
          <div className="formula">
            {`N(X) = gate(X) + PI(X) + α·perm(X) + α²·boundary(X)
t(X) = N(X) / Z_H(X),  split into three pieces of degree ≤ n+1`}
          </div>
          <p>
            The verifier picks a random ζ, receives 16 evaluations bound by KZG, and checks a single scalar
            identity plus two opening proofs (four pairings):
          </p>
          <div className="formula">
            {`N(ζ) = Z_H(ζ)·t(ζ)

e(C_batch − [v]G, [1]) · e(−W_ζ, [τ−ζ])
  · e(C_zω − [z̄ω]G, [1]) · e(−W_ζω, [τζω]) = 1`}
          </div>
          <p>
            If <code>N − Z_H·t</code> were a non-zero polynomial, a random ζ would catch it with probability
            at least <code>1 − (3n+5)/r</code>. That is the soundness argument, and it is the reason a single
            missing constraint is fatal: with the constraint gone, the polynomial that should have been
            non-zero is zero, and every check passes honestly.
          </p>

          <h2>7 · What this is not</h2>
          <ul>
            <li>
              <strong>Not production safe.</strong> No constant-time arithmetic, no side-channel hardening,
              no compressed points.
            </li>
            <li>
              <strong>No ceremony.</strong> τ is sampled in the browser and thrown away; a real deployment
              needs a multi-party powers-of-tau ceremony, or a transparent scheme (STARKs, IPA).
            </li>
            <li>
              <strong>No recursion, no lookups, no custom gates.</strong> The gate equation is the vanilla
              five-selector one.
            </li>
            <li>
              <strong>Timings are honest.</strong> BigInt arithmetic is roughly 100× slower than a
              Montgomery-form native implementation; the numbers in the UI are what this page actually did.
            </li>
          </ul>
        </div>

        <div className="stack" style={{ alignSelf: "start", position: "sticky", top: 78 }}>
          <Panel title="Reading the code">
            <div className="mono" style={{ fontSize: 10.5, color: "var(--muted)", lineHeight: 2 }}>
              <div>
                crypto/field.ts<span className="dim"> — F_q, F_r, extension towers</span>
              </div>
              <div>crypto/curve.ts<span className="dim"> — G₁, G₂, Pippenger</span></div>
              <div>crypto/pairing.ts<span className="dim"> — Miller loop, final exp</span></div>
              <div>crypto/ntt.ts<span className="dim"> — FFT over F_r</span></div>
              <div>crypto/kzg.ts<span className="dim"> — commitments</span></div>
              <div>plonk/circuit.ts<span className="dim"> — gates, σ</span></div>
              <div>plonk/plonk.ts<span className="dim"> — prover, verifier</span></div>
            </div>
          </Panel>
          <Panel title="Asymptotics">
            <div className="mono" style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.9 }}>
              <div>prover <span className="accent">O(n log n)</span></div>
              <div>verifier <span className="accent">O(1)</span> + 4 pairings</div>
              <div>proof <span className="accent">1088 B</span></div>
              <div>vk <span className="accent">768 B</span></div>
            </div>
          </Panel>
        </div>
      </div>
    </>
  );
}

function InteractiveField() {
  const [out, setOut] = useState<string | null>(null);
  const run = () => {
    const rows: string[] = [];
    for (let i = 0; i < 3; i++) {
      const a = FrField.random();
      const inv = FrField.inv(a);
      rows.push(`a = ${a.toString(16).slice(0, 16)}…   a·a⁻¹ = ${FrField.mul(a, inv).toString()}`);
    }
    setOut(rows.join("\n"));
  };
  return (
    <div style={{ margin: "14px 0 22px" }}>
      <button className="btn small" onClick={run}>
        ▸ sample three inverses
      </button>
      {out && <div className="formula" style={{ marginTop: 10, fontSize: 11 }}>{out}</div>}
    </div>
  );
}

function InteractivePairing() {
  const [state, setState] = useState<"idle" | "running" | "done">("idle");
  const [out, setOut] = useState<string[]>([]);
  const run = async () => {
    setState("running");
    await new Promise((r) => setTimeout(r, 30));
    const t0 = performance.now();
    const lhs = pairing(g1Mul(G1_GEN, 2n), g2Mul(G2_GEN, 3n));
    const e = pairing(G1_GEN, G2_GEN);
    const rhs = e.pow(6n);
    const ok = lhs.eq(rhs);
    const nonTrivial = !e.eq(Fp12.one);
    setOut([
      `e(2P, 3Q) == e(P, Q)⁶  →  ${ok ? "true" : "false"}`,
      `e(P, Q) ≠ 1            →  ${nonTrivial ? "true" : "false"}`,
      `four Miller loops + two final exponentiations in ${(performance.now() - t0).toFixed(0)} ms`,
    ]);
    setState("done");
  };
  return (
    <div style={{ margin: "14px 0 22px" }}>
      <button className="btn small" onClick={run} disabled={state === "running"}>
        {state === "running" ? <Spinner /> : "▸"} check bilinearity now
      </button>
      {out.length > 0 && (
        <div className="formula" style={{ marginTop: 10, fontSize: 11 }}>
          {out.join("\n")}
        </div>
      )}
    </div>
  );
}
