import { describe, expect, it } from "vitest";
import { renderToString } from "react-dom/server";
import App from "../src/app/App";
import { InspectorView } from "../src/app/views/Inspector";
import { SoundnessView } from "../src/app/views/Soundness";
import { PrimerView } from "../src/app/views/Primer";
import { CircuitView } from "../src/app/views/CircuitView";
import { executeRun } from "../src/app/runPipeline";
import { buildScenario } from "../src/scenarios";
import { auditWitness } from "../src/plonk/circuit";
import type { ZerothState } from "../src/app/useZeroth";

const EMPTY: ZerothState = { phase: "idle", progress: [], result: null, error: null, tamper: null, tampering: false };

describe("ui", () => {
  it("renders the app shell", () => {
    const html = renderToString(<App />);
    expect(html).toContain("zeroth");
    expect(html).toContain("Prove a statement");
  });

  it("renders the primer", () => {
    const html = renderToString(<PrimerView />);
    expect(html).toContain("From bare arithmetic");
    expect(html).toContain("PLONK");
  });

  it("renders empty states", () => {
    expect(renderToString(<InspectorView state={EMPTY} onProve={() => {}} busy={false} />)).toContain("No proof in memory");
    const preview = { built: null, violations: [], error: "nope" };
    expect(
      renderToString(<CircuitView scenario="multiplier" opts={{}} preview={preview} setScenario={() => {}} />),
    ).toContain("Build a circuit first");
  });

  it("renders a completed run in every view", () => {
    const opts = { mx: "3", my: "11", mz: "33" };
    const result = executeRun(1, "multiplier", opts);
    const state: ZerothState = { ...EMPTY, phase: "done", result };
    expect(renderToString(<InspectorView state={state} onProve={() => {}} busy={false} />)).toContain("Inside a proof");
    const built = buildScenario("multiplier", opts);
    const preview = { built, violations: auditWitness(built.circuit), error: null };
    const circuitHtml = renderToString(
      <CircuitView scenario="multiplier" opts={opts} preview={preview} setScenario={() => {}} />,
    );
    expect(circuitHtml).toContain("Gate table");
    expect(circuitHtml).toContain("Copy constraints");
    expect(
      renderToString(
        <SoundnessView state={state} tamper={() => {}} run={() => {}} />,
      ),
    ).toContain("delete a constraint");
  }, 120000);
});
