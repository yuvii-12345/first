import { useEffect, useMemo, useState } from "react";
import type { ZerothState } from "../useZeroth";
import { Badge, KV, Panel, SectionHead, fmtMs, shortHex } from "../components";

interface Step {
  id: string;
  title: string;
  subtitle?: string;
  formula?: string;
  timeMs: number;
  entries: { label: string; value: string; kind?: string }[];
  side: "prover" | "verifier" | "both";
}

export function InspectorView({
  state,
  onProve,
  busy,
}: {
  state: ZerothState;
  onProve: () => void;
  busy: boolean;
}) {
  const result = state.result;
  const [selected, setSelected] = useState(0);

  const steps: Step[] = useMemo(() => {
    if (!result) return [];
    const arith: Step = {
      id: "arith",
      title: "Arithmetization",
      subtitle: "The statement becomes a table of gates plus a wiring pattern.",
      formula: "qM·a·b + qL·a + qR·b + qO·c + qC + PI = 0",
      timeMs: 0,
      side: "verifier",
      entries: [
        { label: "gates", value: String(result.circuit.numGates), kind: "number" },
        { label: "domain size n", value: String(result.circuit.n), kind: "number" },
        { label: "quotient coset m", value: String(result.circuit.m), kind: "number" },
        { label: "variables", value: String(result.circuit.numVars), kind: "number" },
        { label: "secret / public", value: `${result.circuit.numSecrets} / ${result.circuit.numPublic}`, kind: "text" },
      ],
    };
    const pre: Step = {
      id: "pre",
      title: "Preprocessing",
      subtitle: "Circuit-only data: selector commitments and the permutation polynomials. Same for every proof.",
      formula: "[qM]₁ … [qC]₁,  [σ₁]₁, [σ₂]₁, [σ₃]₁",
      timeMs: result.timings.setup,
      side: "both",
      entries: [
        { label: "verifier key", value: `${result.circuit.vkBytes} bytes`, kind: "text" },
        { label: "SRS powers", value: String(result.circuit.srsPoints), kind: "number" },
        { label: "setup time", value: fmtMs(result.timings.setup), kind: "text" },
      ],
    };
    return [
      arith,
      pre,
      ...result.trace.map(
        (t): Step => ({
          id: t.id,
          title: t.title,
          subtitle: t.subtitle,
          formula: t.formula,
          timeMs: t.timeMs,
          entries: t.entries,
          side: "prover",
        }),
      ),
      {
        id: "verify",
        title: "Verification",
        subtitle: "The verifier replays the transcript, checks the identity at ζ, and runs four pairings.",
        formula: "N(ζ) = Z_H(ζ)·t(ζ)   ∧   e(C − [v]G, [1]) = e(π, [τ−ζ])",
        timeMs: result.timings.verify,
        side: "verifier",
        entries: result.verification.checks.map((c) => ({
          label: c.label,
          value: c.ok ? "pass" : "FAIL",
          kind: "text",
        })),
      },
    ];
  }, [result]);

  useEffect(() => {
    setSelected(0);
  }, [result?.runId]);

  if (!result) {
    return (
      <>
        <SectionHead
          kicker="02 — inspector"
          title={<>Inside a proof</>}
          lead="Every intermediate value of the protocol, in order. Prove something first and this page fills with the actual commitments, challenges and evaluations."
        />
        <Panel>
          <div style={{ display: "grid", placeItems: "center", padding: "60px 20px", textAlign: "center" }}>
            <div>
              <p className="lead" style={{ marginBottom: 16 }}>
                No proof in memory yet.
              </p>
              <button className="btn primary" onClick={onProve} disabled={busy}>
                ▸ prove the selected statement
              </button>
            </div>
          </div>
        </Panel>
      </>
    );
  }

  const step = steps[Math.min(selected, steps.length - 1)];
  const total = steps.reduce((a, s) => a + s.timeMs, 0);

  return (
    <>
      <SectionHead
        kicker="02 — inspector"
        title={
          <>
            Inside a proof
            <br />
            <em>{result.statement}</em>
          </>
        }
        lead={
          <>
            Each stage below ran in the prover. Hover the timeline to move through the protocol; every
            number shown is the real value that went into the transcript.
          </>
        }
        right={<Badge tone={result.ok ? "ok" : "bad"}>{result.ok ? "verified" : "rejected"}</Badge>}
      />

      <div className="grid-2" style={{ gridTemplateColumns: "300px minmax(0,1fr)" }}>
        <Panel title={`Steps · ${fmtMs(total)}`} tight>
          <div className="timeline" style={{ padding: "12px 14px" }}>
            {steps.map((s, i) => (
              <div key={s.id} className="tl-item" data-active={i === selected}>
                <div className="tl-rail">
                  <span className="tl-dot" />
                  {i < steps.length - 1 && <span className="tl-line" />}
                </div>
                <div className="tl-body">
                  <div className="tl-title" onClick={() => setSelected(i)}>
                    {i + 1}. {s.title}
                  </div>
                  <div className="tl-sub">
                    {s.timeMs > 0 ? fmtMs(s.timeMs) : "—"} · {s.side}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>

        <div className="stack">
          <Panel title={step.title} right={<Badge>{step.side}</Badge>}>
            {step.subtitle && <p style={{ fontSize: 13 }}>{step.subtitle}</p>}
            {step.formula && <div className="formula">{step.formula}</div>}
            <div style={{ marginTop: 14 }}>
              <KV items={step.entries.map((e) => [e.label, <Value key={e.label} v={e.value} kind={e.kind} />])} />
            </div>
          </Panel>

          <Panel title="Transcript" right={<Badge>Fiat–Shamir</Badge>}>
            <p style={{ fontSize: 12.5 }}>
              Challenges are squeezed from a SHA-256 sponge over everything absorbed so far. Replaying the
              same sequence in the verifier reproduces them exactly — that is what makes the proof
              non-interactive.
            </p>
            <KV
              items={result.challenges.map((c) => [
                c.name,
                <span key={c.name} className="mono" style={{ fontSize: 11 }}>
                  {shortHex(c.value, 16, 10)}
                </span>,
              ])}
            />
          </Panel>
        </div>
      </div>
    </>
  );
}

function Value({ v, kind }: { v: string; kind?: string }) {
  if (kind === "field" || kind === "hex") {
    return (
      <span className="mono" title={v} style={{ fontSize: 11 }}>
        {shortHex(v, 18, 10)}
      </span>
    );
  }
  if (kind === "point") {
    return <span className="mono" style={{ fontSize: 11 }}>{v}</span>;
  }
  return <span>{v}</span>;
}
