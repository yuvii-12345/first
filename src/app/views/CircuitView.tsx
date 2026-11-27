import { useMemo, useState } from "react";
import type { ScenarioId, ScenarioOptions } from "../../scenarios";
import { SCENARIO_INFO } from "../../scenarios";
import { Badge, Panel, SectionHead, Switch } from "../components";
import type { CompiledCircuit } from "../../plonk/circuit";

export function CircuitView({
  scenario,
  opts,
  preview,
  setScenario,
}: {
  scenario: ScenarioId;
  opts: ScenarioOptions;
  preview: { built: any; violations: any[]; error: string | null };
  setScenario: (s: ScenarioId) => void;
}) {
  const circuit: CompiledCircuit | null = preview.built?.circuit ?? null;
  const [reveal, setReveal] = useState(false);
  const [onlyNonTrivial, setOnlyNonTrivial] = useState(true);

  const occurrences = useMemo(() => {
    if (!circuit) return [];
    const map = new Map<number, number[]>();
    for (let slot = 0; slot < 3; slot++) {
      for (let row = 0; row < circuit.n; row++) {
        const id = circuit.wireIds[slot][row];
        const pos = slot * circuit.n + row;
        const arr = map.get(id);
        if (arr) arr.push(pos);
        else map.set(id, [pos]);
      }
    }
    return Array.from(map.entries())
      .filter(([, ps]) => ps.length > 1)
      .sort((a, b) => b[1].length - a[1].length);
  }, [circuit]);

  const rows = useMemo(() => {
    if (!circuit) return [];
    const out: number[] = [];
    for (let i = 0; i < Math.min(circuit.numGates, 400); i++) {
      const trivial =
        circuit.selectors.qM[i] === 0n &&
        circuit.selectors.qL[i] === 0n &&
        circuit.selectors.qR[i] === 0n &&
        circuit.selectors.qO[i] === 0n &&
        circuit.selectors.qC[i] === 0n;
      if (onlyNonTrivial && trivial) continue;
      out.push(i);
    }
    return out;
  }, [circuit, onlyNonTrivial]);

  if (!circuit) {
    return (
      <>
        <SectionHead kicker="03 — circuit" title="The wire map" lead="Build a circuit first." />
        <Panel>
          <div className="callout danger">{preview.error ?? "no circuit"}</div>
        </Panel>
      </>
    );
  }

  const multi = occurrences.length;
  const totalSlots = occurrences.reduce((a, [, ps]) => a + ps.length, 0);

  return (
    <>
      <SectionHead
        kicker="03 — circuit"
        title={
          <>
            The wire map
            <br />
            <em>where the soundness actually lives.</em>
          </>
        }
        lead={
          <>
            The gate equation is only half of a PLONKish circuit. The other half is the wiring: which wires
            must hold the <em>same</em> value. That is enforced by a permutation argument, not by any
            selector — and it is where most hand-written circuits silently break.
          </>
        }
        right={
          <select className="text" value={scenario} onChange={(e) => setScenario(e.target.value as ScenarioId)} style={{ width: 200 }}>
            {SCENARIO_INFO.map((s) => (
              <option key={s.id} value={s.id}>
                {s.title}
              </option>
            ))}
          </select>
        }
      />

      <div className="stats" style={{ marginBottom: 18, border: "1px solid var(--line)", borderRadius: 3 }}>
        <div className="stat">
          <div className="k">rows used</div>
          <div className="v">
            {circuit.numGates}
            <small>/ {circuit.n}</small>
          </div>
        </div>
        <div className="stat">
          <div className="k">variables</div>
          <div className="v">{circuit.variables.length}</div>
        </div>
        <div className="stat">
          <div className="k">shared variables</div>
          <div className="v">{multi}</div>
        </div>
        <div className="stat">
          <div className="k">copy constraints</div>
          <div className="v">{totalSlots}</div>
        </div>
        <div className="stat">
          <div className="k">k₁, k₂</div>
          <div className="v" style={{ fontSize: 13 }}>
            {circuit.k1.toString()}, {circuit.k2.toString()}
          </div>
        </div>
      </div>

      <div className="grid-2">
        <Panel
          title="Gate table"
          tight
          right={
            <div className="row" style={{ gap: 10 }}>
              <Switch checked={onlyNonTrivial} onChange={setOnlyNonTrivial} label="hide empty rows" />
            </div>
          }
        >
          <div style={{ overflow: "auto", maxHeight: 460 }}>
            <div className="gates">
              <div className="head">row</div>
              <div className="head">qM</div>
              <div className="head">qL</div>
              <div className="head">qR</div>
              <div className="head">qO</div>
              <div className="head">qC</div>
              <div className="head">a</div>
              <div className="head">b</div>
              <div className="head">c</div>
              {rows.map((i) => (
                <Row key={i} circuit={circuit} i={i} />
              ))}
            </div>
          </div>
          <div style={{ borderTop: "1px solid var(--line)", padding: "8px 14px" }}>
            <span className="dim mono" style={{ fontSize: 10.5, lineHeight: 1.6 }}>
              {`rows 0–1 anchor the constants: qL·a + qC = 0 forces $zero = 0 and $one = 1.
unused wires point at $zero, so its copy cycle is the longest in the table.`}
            </span>
          </div>
        </Panel>

        <div className="stack">
          <Panel title="Copy constraints" right={<Badge>{multi} cycles</Badge>} tight>
            <div style={{ padding: "10px 14px", maxHeight: 240, overflow: "auto" }}>
              {occurrences.slice(0, 14).map(([id, ps]) => (
                <div key={id} className="mono" style={{ fontSize: 10.5, color: "var(--muted)", marginBottom: 7, lineHeight: 1.5 }}>
                  <span className="accent">{circuit.variables[id]?.name ?? `v${id}`}</span>{" "}
                  <span className="dim">
                    {ps.map((p) => slotLabel(p, circuit.n)).join(" → ")} → (wrap)
                  </span>
                </div>
              ))}
              {multi === 0 && <div className="dim mono" style={{ fontSize: 11 }}>no shared wires</div>}
            </div>
          </Panel>

          <Panel title="Variables" tight right={<Switch checked={reveal} onChange={setReveal} label="reveal" />}>
            <div style={{ maxHeight: 300, overflow: "auto" }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>id</th>
                    <th>name</th>
                    <th>kind</th>
                    <th style={{ textAlign: "right" }}>value</th>
                  </tr>
                </thead>
                <tbody>
                  {circuit.variables.slice(0, 80).map((v) => (
                    <tr key={v.id}>
                      <td className="num">{v.id}</td>
                      <td>{v.name}</td>
                      <td style={{ color: kindColor(v.kind) }}>{v.kind}</td>
                      <td className="num">
                        {v.kind === "secret" && !reveal ? "•••••" : truncate(v.value.toString())}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
      </div>

      <div className="callout" style={{ marginTop: 18 }}>
        <b>Why this matters.</b> The permutation σ maps every wire slot to the next slot that must hold the
        same variable, cycling. If σ is built wrong — or if the permutation argument is dropped — each gate
        can be satisfied with its <em>own</em> copy of the value, and the whole circuit stops meaning
        anything. The grand product argument makes forging such a "solution" equivalent to breaking a
        polynomial identity.
      </div>
    </>
  );
}

function Row({ circuit, i }: { circuit: CompiledCircuit; i: number }) {
  const s = circuit.selectors;
  const cell = (v: bigint) => (v === 0n ? <span className="zero">0</span> : truncate(v.toString()));
  return (
    <>
      <div className="row-idx">{i}</div>
      <div>{cell(s.qM[i])}</div>
      <div>{cell(s.qL[i])}</div>
      <div>{cell(s.qR[i])}</div>
      <div>{cell(s.qO[i])}</div>
      <div>{cell(s.qC[i])}</div>
      <div className="hl">{circuit.variables[circuit.wireIds[0][i]]?.name}</div>
      <div className="hl">{circuit.variables[circuit.wireIds[1][i]]?.name}</div>
      <div className="hl">{circuit.variables[circuit.wireIds[2][i]]?.name}</div>
    </>
  );
}

function slotLabel(pos: number, n: number): string {
  const slot = Math.floor(pos / n);
  const row = pos % n;
  return `${slot === 0 ? "a" : slot === 1 ? "b" : "c"}·r${row}`;
}

function truncate(v: string): string {
  if (v.length <= 10) return v.startsWith("-") ? "−" + v.slice(1) : v;
  const neg = v.startsWith("-");
  const body = neg ? v.slice(1) : v;
  return (neg ? "−" : "") + body.slice(0, 6) + "…";
}

function kindColor(kind: string): string {
  switch (kind) {
    case "secret":
      return "var(--accent)";
    case "public":
      return "var(--blue)";
    case "constant":
      return "var(--dim)";
    default:
      return "var(--muted)";
  }
}
