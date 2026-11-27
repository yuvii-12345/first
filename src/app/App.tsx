import { useEffect, useMemo, useRef, useState } from "react";
import {
  buildScenario,
  SCENARIO_INFO,
  type ScenarioId,
  type ScenarioOptions,
} from "../scenarios";
import { auditWitness } from "../plonk/circuit";
import { mimcHash } from "../circuits/mimc";
import { EXAMPLE_PROGRAMS } from "../dsl/dsl";
import { FrField } from "../crypto/field";
import { useZeroth, type ZerothState } from "./useZeroth";
import {
  Badge,
  CheckRow,
  Field,
  Hex,
  KV,
  Panel,
  SectionHead,
  Spinner,
  Stat,
  Stats,
  Switch,
  fmtBytes,
  fmtMs,
  shortDec,
  shortHex,
} from "./components";
import { InspectorView } from "./views/Inspector";
import { CircuitView } from "./views/CircuitView";
import { SoundnessView } from "./views/Soundness";
import { PrimerView } from "./views/Primer";

type Tab = "prove" | "inspector" | "circuit" | "soundness" | "primer";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "prove", label: "Prove" },
  { id: "inspector", label: "Inspector" },
  { id: "circuit", label: "Circuit" },
  { id: "soundness", label: "Soundness" },
  { id: "primer", label: "Primer" },
];

const DEFAULT_OPTS: Record<ScenarioId, ScenarioOptions> = {
  multiplier: { mx: "7", my: "13", mz: "91" },
  mimc: { preimage: "424242", key: "1337" },
  sudoku4: {},
  sudoku9: {},
  dsl: { source: EXAMPLE_PROGRAMS[0].source },
};

const STAGES = [
  { id: "arith", title: "Arithmetization", detail: "statement → gate table + wiring" },
  { id: "pre", title: "Preprocessing", detail: "commit to selectors and σ" },
  { id: "wires", title: "Commit to wires", detail: "a, b, c ↦ [a]₁, [b]₁, [c]₁" },
  { id: "grand-product", title: "Grand product", detail: "z enforces that equal wires are equal" },
  { id: "quotient", title: "Quotient", detail: "t = N / Z_H, split into three" },
  { id: "evaluations", title: "Evaluations at ζ", detail: "the 16 numbers the verifier sees" },
  { id: "openings", title: "KZG openings", detail: "W_ζ and W_ζω" },
  { id: "verify", title: "Verify", detail: "identity at ζ + four pairings" },
];

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("prove");
  const [scenario, setScenario] = useState<ScenarioId>("multiplier");
  const [opts, setOpts] = useState<Record<ScenarioId, ScenarioOptions>>(DEFAULT_OPTS);
  const { state, run, tamper, reset } = useZeroth();

  const currentOpts = opts[scenario];
  const setCurrentOpts = (patch: Partial<ScenarioOptions>) =>
    setOpts((o) => ({ ...o, [scenario]: { ...o[scenario], ...patch } }));

  // Live constraint audit: rebuild the circuit on the main thread (cheap) so the
  // user can see a bad witness *before* burning seconds on a proof.
  const debouncedOpts = useDebounced(currentOpts, 260);
  const preview = useMemo(() => {
    try {
      const built = buildScenario(scenario, debouncedOpts);
      return { built, violations: auditWitness(built.circuit), error: null as string | null };
    } catch (err) {
      return { built: null, violations: [], error: (err as Error).message };
    }
  }, [scenario, debouncedOpts]);

  const busy = state.phase === "running";
  const canProve = !!preview.built && preview.violations.length === 0 && !preview.built.meta?.dslError;

  const goProve = () => {
    run(scenario, opts[scenario]);
    setTab("prove");
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="mark">
            zeroth<span>.</span>
          </span>
          <span className="sub">zk-snark from first principles</span>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.id} data-active={tab === t.id} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
        <div className="params">
          <span>
            curve <b>BN254</b>
          </span>
          <span>
            arithmetization <b>PLONK</b>
          </span>
          <span>
            commitment <b>KZG</b>
          </span>
          <span>
            field <b>254-bit</b>
          </span>
        </div>
      </header>

      <main className="page">
        {tab === "prove" && (
          <ProveView
            scenario={scenario}
            setScenario={setScenario}
            opts={currentOpts}
            setOpts={setCurrentOpts}
            preview={preview}
            state={state}
            onProve={goProve}
            canProve={canProve}
            busy={busy}
            onInspect={() => setTab("inspector")}
            reset={reset}
          />
        )}
        {tab === "inspector" && <InspectorView state={state} onProve={goProve} busy={busy} />}
        {tab === "circuit" && <CircuitView scenario={scenario} opts={currentOpts} preview={preview} setScenario={setScenario} />}
        {tab === "soundness" && <SoundnessView state={state} tamper={tamper} run={run} />}
        {tab === "primer" && <PrimerView onBreak={() => setTab("soundness")} />}
      </main>

      <footer className="footer">
        <span>zeroth · a teaching implementation</span>
        <span className="spacer" />
        <span>no trusted setup ceremony · no constant-time guarantees · do not use in production</span>
      </footer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Prove view
// ---------------------------------------------------------------------------

function ProveView({
  scenario,
  setScenario,
  opts,
  setOpts,
  preview,
  state,
  onProve,
  canProve,
  busy,
  onInspect,
  reset,
}: {
  scenario: ScenarioId;
  setScenario: (s: ScenarioId) => void;
  opts: ScenarioOptions;
  setOpts: (patch: Partial<ScenarioOptions>) => void;
  preview: { built: any; violations: any[]; error: string | null };
  state: ZerothState;
  onProve: () => void;
  canProve: boolean;
  busy: boolean;
  onInspect: () => void;
  reset: () => void;
}) {
  const result = state.result;
  const stageStates = useMemo(() => computeStageStates(state), [state.progress, state.phase, state.result]);

  return (
    <>
      <SectionHead
        kicker="01 — prove"
        title={
          <>
            Prove a statement
            <br />
            <em>without revealing why it is true.</em>
          </>
        }
        lead={
          <>
            The prover holds a secret witness. The verifier holds a handful of group elements and a
            public statement. Everything below — the field, the curve, the pairing, the commitments, the
            protocol — is implemented from scratch in this page's bundle.
          </>
        }
        right={<Badge>1088-byte proofs</Badge>}
      />

      <div className="scenarios">
        {SCENARIO_INFO.map((s) => (
          <button
            key={s.id}
            className="scenario"
            data-active={scenario === s.id}
            onClick={() => {
              setScenario(s.id);
              reset();
            }}
          >
            <div className="title">{s.title}</div>
            <div className="desc">{s.subtitle}</div>
            <span className="badge">{s.badge}</span>
          </button>
        ))}
      </div>

      <div className="grid-2">
        <div className="stack">
          <Panel
            title="Statement"
            right={<Badge>public</Badge>}
          >
            {preview.built ? (
              <>
                <div className="mono" style={{ fontSize: 13, color: "var(--text)", marginBottom: 10 }}>
                  {preview.built.meta.statement}
                </div>
                {preview.built.meta.notes.map((n: string, i: number) => (
                  <p key={i} style={{ fontSize: 12, marginBottom: 4 }}>
                    {n}
                  </p>
                ))}
              </>
            ) : (
              <p style={{ color: "var(--red)" }}>{preview.error}</p>
            )}
            <div className="divider" />
            <div className="row wrap" style={{ gap: 8 }}>
              <Badge>{preview.built ? `${preview.built.circuit.numGates} gates` : "—"}</Badge>
              <Badge>{preview.built ? `domain n = ${preview.built.circuit.n}` : "—"}</Badge>
              <Badge>{preview.built ? `${preview.built.circuit.variables.length} variables` : "—"}</Badge>
              {preview.built?.meta.unsound && <Badge tone="bad">constraint deleted</Badge>}
            </div>
          </Panel>

          <Panel title="Witness" right={preview.violations.length === 0 ? <Badge tone="ok">satisfied</Badge> : <Badge tone="bad">violated</Badge>}>
            <Editor scenario={scenario} opts={opts} setOpts={setOpts} preview={preview} />
            <div className="divider" />
            {preview.error ? (
              <div className="callout danger">{preview.error}</div>
            ) : preview.violations.length === 0 ? (
              <div className="mono" style={{ fontSize: 11.5, color: "var(--dim)" }}>
                all {preview.built?.circuit.numGates} gate equations balance
              </div>
            ) : (
              <div className="callout danger">
                <b>{preview.violations.length} constraint(s) violated</b> — rows{" "}
                {preview.violations.slice(0, 6).map((v: any) => v.row).join(", ")}
                {preview.violations.length > 6 ? "…" : ""}. The prover cannot produce a valid proof for a false
                witness; that is the whole point.
              </div>
            )}
            <div className="row" style={{ marginTop: 14 }}>
              <button className="btn primary" onClick={onProve} disabled={busy || !canProve}>
                {busy ? <Spinner /> : "▸"} {busy ? "proving" : "prove & verify"}
              </button>
              {!canProve && !busy && (
                <span className="dim mono" style={{ fontSize: 11 }}>
                  witness must satisfy the circuit
                </span>
              )}
            </div>
          </Panel>
        </div>

        <div className="stack">
          <Panel
            title="Protocol"
            tight
            right={
              state.phase === "running" ? (
                <span className="row" style={{ gap: 6 }}>
                  <Spinner />
                  <span className="mono" style={{ fontSize: 10.5, color: "var(--accent)" }}>
                    {state.progress[state.progress.length - 1]?.phase ?? "starting"}
                  </span>
                </span>
              ) : result ? (
                <Badge tone={result.ok ? "ok" : "bad"}>{result.ok ? "verified" : "rejected"}</Badge>
              ) : (
                <Badge>idle</Badge>
              )
            }
          >
            {state.phase === "running" && <div className="progressbar" />}
            <div className="pipeline">
              {STAGES.map((s, i) => (
                <div className="pstep" key={s.id} data-state={stageStates[s.id]}>
                  <span className="idx">{String(i + 1).padStart(2, "0")}</span>
                  <div>
                    <div className="t">{s.title}</div>
                    <div className="d">{s.detail}</div>
                  </div>
                  <span className="ms">
                    {stageMs(s.id, result, stageStates[s.id])}
                  </span>
                </div>
              ))}
            </div>
          </Panel>

          {state.phase === "error" && (
            <Panel title="Prover error" right={<Badge tone="bad">halted</Badge>}>
              <div className="callout danger">{state.error}</div>
              <p style={{ fontSize: 12 }}>
                A prover that cannot satisfy its own constraints must not be able to emit a proof. Here it
                refused.
              </p>
            </Panel>
          )}

          {!result && (
            <Panel title="Verification" right={<Badge>awaiting proof</Badge>}>
              <div className="stack" style={{ gap: 11 }}>
                {[
                  ["curve", "Proof elements are valid curve points"],
                  ["identity", "Gate + permutation + boundary identity holds at ζ"],
                  ["openings", "KZG openings are consistent with the commitments"],
                ].map(([id, label]) => (
                  <div key={id} className="row" style={{ gap: 10 }}>
                    <span className="mono dim" style={{ fontSize: 11, width: 12 }}>
                      ○
                    </span>
                    <span style={{ fontSize: 12.5, color: "#c3c7cf" }}>{label}</span>
                  </div>
                ))}
              </div>
              <p style={{ marginTop: 14, marginBottom: 0, fontSize: 12 }}>
                Three checks over 1088 bytes of proof. Nothing about the witness is
                among them — press <span className="accent mono">prove &amp; verify</span> to fill
                them in.
              </p>
            </Panel>
          )}

          {result && (
            <Panel
              title="Verification"
              tight
              right={<Badge tone={result.ok ? "ok" : "bad"}>{result.ok ? "accept" : "reject"}</Badge>}
            >
              {result.verification.checks.map((c) => (
                <CheckRow key={c.id} check={c} />
              ))}
              <Stats>
                <Stat k="prove" v={fmtMs(result.timings.prove)} />
                <Stat k="verify" v={fmtMs(result.timings.verify)} />
                <Stat k="proof" v={fmtBytes(result.circuit.proofBytes)} />
                <Stat k="round trip" v={fmtMs(result.timings.total)} />
              </Stats>
            </Panel>
          )}
        </div>
      </div>

      {result && (
        <div className="grid-2" style={{ marginTop: 18 }}>
          <Panel
            title="Proof envelope"
            right={
              <button className="btn small ghost" onClick={onInspect}>
                open inspector →
              </button>
            }
          >
            <KV
              items={[
                ["statement", result.statement || "—"],
                ["public inputs", result.publicValues.length ? result.publicValues.map(shortDec).join(", ") : "none"],
                ["9 G₁ elements + 16 scalars", fmtBytes(result.circuit.proofBytes)],
                ["sha-256", result.proofDigest + "…"],
                ["verifier key", fmtBytes(result.circuit.vkBytes)],
                ["SRS points", String(result.circuit.srsPoints)],
              ]}
            />
            <div style={{ marginTop: 12 }}>
              <Hex value={result.proofHex} />
            </div>
          </Panel>

          <Panel title="Challenges" right={<Badge>Fiat–Shamir</Badge>}>
            <p style={{ fontSize: 12.5 }}>
              Every challenge is derived by hashing the transcript so far. There is no verifier message in
              this protocol — the hash <em>is</em> the verifier.
            </p>
            <KV
              items={result.challenges.map((c) => [
                c.name,
                <span key={c.name} className="mono" style={{ fontSize: 11 }}>
                  {shortHex(c.value, 12, 8)}
                </span>,
              ])}
            />
            <div className="divider" />
            <div className="mono" style={{ fontSize: 11, color: "var(--dim)" }}>
              witness: {result.circuit.numSecrets} secret wires · {result.circuit.numPublic} public · {result.circuit.numVars} total
            </div>
          </Panel>
        </div>
      )}
    </>
  );
}

function stageMs(id: string, result: any, state: string): string {
  if (result) {
    if (id === "pre") return fmtMs(result.timings.setup);
    if (id === "verify") return fmtMs(result.timings.verify);
    const step = result.timings.steps.find((x: any) => x.id === id);
    if (step) return fmtMs(step.timeMs);
    return "";
  }
  return state === "active" ? "…" : "";
}

function computeStageStates(state: ZerothState): Record<string, "pending" | "active" | "done"> {
  const order = STAGES.map((s) => s.id);
  const out: Record<string, "pending" | "active" | "done"> = {};
  for (const id of order) out[id] = "pending";
  const mark = (id: string) => {
    const idx = order.indexOf(id);
    if (idx < 0) return;
    for (let i = 0; i < idx; i++) out[order[i]] = "done";
    out[id] = "active";
  };
  for (const p of state.progress) {
    if (p.phase === "Arithmetization") mark("arith");
    else if (p.phase === "Preprocessing") {
      out.arith = "done";
      mark("pre");
    } else if (p.phase === "Proving" && p.stepId) {
      out.arith = "done";
      out.pre = "done";
      mark(p.stepId);
    } else if (p.phase === "Verifying") {
      for (const id of ["arith", "pre", "wires", "grand-product", "quotient", "evaluations", "openings"]) out[id] = "done";
      mark("verify");
    }
  }
  if (state.phase === "done") for (const id of order) out[id] = "done";
  return out;
}

// ---------------------------------------------------------------------------
// Witness editors
// ---------------------------------------------------------------------------

function Editor({
  scenario,
  opts,
  setOpts,
  preview,
}: {
  scenario: ScenarioId;
  opts: ScenarioOptions;
  setOpts: (patch: Partial<ScenarioOptions>) => void;
  preview: { built: any; violations: any[]; error: string | null };
}) {
  switch (scenario) {
    case "multiplier":
      return (
        <>
          <div className="grid-3" style={{ gap: 12 }}>
            <Field label="x" hint="secret">
              <input className="text" value={opts.mx ?? "7"} onChange={(e) => setOpts({ mx: e.target.value })} />
            </Field>
            <Field label="y" hint="secret">
              <input className="text" value={opts.my ?? "13"} onChange={(e) => setOpts({ my: e.target.value })} />
            </Field>
            <Field label="z" hint="public">
              <input className="text" value={opts.mz ?? "91"} onChange={(e) => setOpts({ mz: e.target.value })} />
            </Field>
          </div>
          <p style={{ fontSize: 12, marginBottom: 0 }}>
            The circuit asserts <span className="mono">x · y = z</span>. Change z to something else to watch
            the witness audit fail.
          </p>
        </>
      );

    case "mimc": {
      const pre = safeBigInt(opts.preimage);
      const key = safeBigInt(opts.key);
      const digest = mimcHash(FrField.reduce(pre), FrField.reduce(key));
      return (
        <>
          <div className="grid-2" style={{ gap: 12 }}>
            <Field label="preimage x" hint="secret">
              <input className="text secret" value={opts.preimage ?? ""} onChange={(e) => setOpts({ preimage: e.target.value })} />
            </Field>
            <Field label="key k" hint="public">
              <input className="text" value={opts.key ?? ""} onChange={(e) => setOpts({ key: e.target.value })} />
            </Field>
          </div>
          <div className="mono" style={{ fontSize: 11, color: "var(--dim)", marginBottom: 10 }}>
            MiMC(x, k) = <span className="accent">{shortHex("0x" + digest.toString(16), 14, 8)}</span>
          </div>
          <Switch
            checked={!!opts.digest}
            onChange={(v) => setOpts({ digest: v ? (digest + 1n).toString() : undefined })}
            label="claim a wrong digest (cheating prover)"
          />
        </>
      );
    }

    case "sudoku4":
    case "sudoku9": {
      const grid = preview.built?.meta.grid;
      const size = scenario === "sudoku4" ? 4 : 9;
      return (
        <>
          <div className="row" style={{ gap: 18, alignItems: "flex-start", flexWrap: "wrap" }}>
            <div>
              <div className="mono" style={{ fontSize: 10, color: "var(--dim)", marginBottom: 6, letterSpacing: "0.1em" }}>
                PUZZLE (verifier sees)
              </div>
              <SudokuGrid puzzle={grid?.puzzle} solution={null} size={size} />
            </div>
            <div>
              <div className="mono" style={{ fontSize: 10, color: "var(--dim)", marginBottom: 6, letterSpacing: "0.1em" }}>
                WITNESS (hidden)
              </div>
              <SudokuGrid puzzle={grid?.puzzle} solution={grid?.solution} size={size} tampered={!!opts.tamper} />
            </div>
          </div>
          <div style={{ marginTop: 12 }}>
            <Switch checked={!!opts.tamper} onChange={(v) => setOpts({ tamper: v })} label="swap two cells (invalid grid)" />
          </div>
        </>
      );
    }

    case "dsl":
      return (
        <>
          <Field label="program" hint="secret / public / let / assert">
            <textarea
              className="text"
              spellCheck={false}
              value={opts.source ?? ""}
              onChange={(e) => setOpts({ source: e.target.value })}
              rows={9}
            />
          </Field>
          <div className="row wrap" style={{ gap: 6 }}>
            {EXAMPLE_PROGRAMS.map((ex) => (
              <button key={ex.name} className="btn small ghost" onClick={() => setOpts({ source: ex.source })} type="button">
                {ex.name}
              </button>
            ))}
          </div>
          {preview.built?.meta.dslError && <div className="callout danger" style={{ marginTop: 10 }}>{preview.built.meta.dslError}</div>}
        </>
      );
  }
}

function SudokuGrid({
  puzzle,
  solution,
  size,
  tampered,
}: {
  puzzle?: number[][];
  solution: number[][] | null | undefined;
  size: number;
  tampered?: boolean;
}) {
  if (!puzzle) return null;
  return (
    <div
      className="grid-board"
      style={{
        gridTemplateColumns: `repeat(${size}, 30px)`,
      }}
    >
      {puzzle.map((row, r) =>
        row.map((g, c) => {
          const value = solution ? solution[r][c] : g;
          const isGiven = g > 0;
          const bad = tampered && r === 0 && c === 1;
          return (
            <div
              key={`${r}-${c}`}
              className={`cell ${isGiven ? "given" : ""} ${bad ? "bad" : ""}`}
              style={
                size === 9
                  ? {
                      borderRight: c % 3 === 2 && c < 8 ? "1px solid var(--line-2)" : undefined,
                      borderBottom: r % 3 === 2 && r < 8 ? "1px solid var(--line-2)" : undefined,
                    }
                  : {
                      borderRight: c % 2 === 1 && c < 3 ? "1px solid var(--line-2)" : undefined,
                      borderBottom: r % 2 === 1 && r < 3 ? "1px solid var(--line-2)" : undefined,
                    }
              }
            >
              {value || ""}
            </div>
          );
        }),
      )}
    </div>
  );
}

function safeBigInt(v: string | undefined): bigint {
  if (v === undefined) return 0n;
  const t = v.trim();
  if (t === "" || t === "-") return 0n;
  try {
    return BigInt(t);
  } catch {
    return 0n;
  }
}
