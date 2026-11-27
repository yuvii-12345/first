import React, { useCallback, useEffect, useState } from "react";
import { BarChart3, CheckCircle2, Layers, RefreshCw } from "lucide-react";
import { api } from "../api";
import type { DatasetMeta, LogitLensAnalysis } from "../types";
import { logitLensBg } from "../utils/colors";

interface LogitLensTabProps {
  defaultPrompt: string;
  datasetMeta?: DatasetMeta;
  modelStep: number;
}

export const LogitLensTab: React.FC<LogitLensTabProps> = ({
  defaultPrompt,
  datasetMeta,
  modelStep,
}) => {
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [targetToken, setTargetToken] = useState("");
  const [foilToken, setFoilToken] = useState("");
  const [colorMode, setColorMode] = useState<"top1" | "final" | "entropy">("top1");
  const [data, setData] = useState<LogitLensAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedCell, setSelectedCell] = useState<{ stageIdx: number; pos: number } | null>(null);

  useEffect(() => {
    setPrompt(defaultPrompt);
  }, [defaultPrompt]);

  const fetchLens = useCallback(
    async (p: string, tgt: string, foil: string) => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.analyzeLogitLens({
          prompt: p,
          target_token: tgt.trim() || undefined,
          foil_token: foil.trim() || undefined,
        });
        setData(res);
        setSelectedCell({
          stageIdx: Math.max(0, res.stages.length - 1),
          pos: Math.max(0, res.tokens.length - 1),
        });
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    if (prompt.trim()) {
      fetchLens(prompt, targetToken, foilToken);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelStep]);

  const activeCell =
    data && selectedCell && data.stages[selectedCell.stageIdx]?.cells[selectedCell.pos]
      ? data.stages[selectedCell.stageIdx].cells[selectedCell.pos]
      : null;

  const activeStageName =
    data && selectedCell ? data.stages[selectedCell.stageIdx]?.stage : "";

  const activePos = selectedCell?.pos ?? (data ? data.tokens.length - 1 : 0);

  // Compute max absolute DLA value for clean symmetrical bars
  const dlaComponents = data?.dla.components ?? [];
  const maxAbsDla = Math.max(0.25, ...dlaComponents.map((c) => Math.abs(c.contrib)));
  const reconDiff = data
    ? Math.abs(data.dla.actual_logit - data.dla.sum_of_contribs)
    : 0;

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="glass-panel rounded-xl p-4">
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") fetchLens(prompt, targetToken, foilToken);
            }}
            placeholder="Enter prompt for Logit Lens & Direct Logit Attribution..."
            className="min-w-[260px] flex-1 rounded-lg border border-obsidian-700 bg-obsidian-950 px-3 py-2 font-mono text-xs text-slate-100 focus:border-emerald-500 focus:outline-none"
          />
          <input
            type="text"
            value={targetToken}
            onChange={(e) => setTargetToken(e.target.value)}
            placeholder="DLA Target (auto)"
            className="w-36 rounded-lg border border-obsidian-700 bg-obsidian-950 px-2.5 py-2 font-mono text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
          />
          <input
            type="text"
            value={foilToken}
            onChange={(e) => setFoilToken(e.target.value)}
            placeholder="Foil token (opt)"
            className="w-36 rounded-lg border border-obsidian-700 bg-obsidian-950 px-2.5 py-2 font-mono text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
          />
          <button
            onClick={() => fetchLens(prompt, targetToken, foilToken)}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/50 bg-emerald-500/20 px-3.5 py-2 font-mono text-xs font-semibold text-emerald-200 transition hover:bg-emerald-500/30 disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            Project Residual Stream
          </button>
        </div>

        {datasetMeta?.sample_prompts && (
          <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
            <span className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
              Presets:
            </span>
            {datasetMeta.sample_prompts.map((sp, i) => (
              <button
                key={i}
                onClick={() => {
                  setPrompt(sp);
                  fetchLens(sp, targetToken, foilToken);
                }}
                className="max-w-[340px] truncate rounded-md border border-obsidian-700 bg-obsidian-900/80 px-2 py-0.5 font-mono text-[11px] text-slate-300 transition hover:border-emerald-500/40 hover:text-emerald-200"
              >
                {sp}
              </button>
            ))}
          </div>
        )}
        {error && <p className="mt-2 font-mono text-xs text-rose-400">{error}</p>}
      </div>

      {data && (
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
          {/* Logit Lens Matrix (8 cols) */}
          <div className="glass-panel rounded-xl p-4 xl:col-span-8">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                  <Layers className="h-4 w-4 text-emerald-400" />
                  Logit Lens — Intermediate Residual Stream Decoding
                </h2>
                <p className="text-xs text-slate-400">
                  Every intermediate hidden state projected through Final RMSNorm + Tied Unembedding W_U
                </p>
              </div>
              <div className="flex items-center gap-1 rounded-lg border border-obsidian-700 bg-obsidian-950 p-0.5 font-mono text-[11px]">
                {(
                  [
                    ["top1", "Top-1 Prob"],
                    ["final", "Final Tok Prob"],
                    ["entropy", "Sharpness"],
                  ] as const
                ).map(([m, lbl]) => (
                  <button
                    key={m}
                    onClick={() => setColorMode(m)}
                    className={`rounded-md px-2 py-1 transition ${
                      colorMode === m
                        ? "bg-emerald-500/25 text-emerald-200"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {lbl}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto pb-2">
              <table className="border-separate border-spacing-1 font-mono text-[11px]">
                <thead>
                  <tr>
                    <th className="w-20 px-2 py-1 text-left text-[10px] font-medium uppercase text-slate-500">
                      Stage
                    </th>
                    {data.tokens.map((tok) => (
                      <th
                        key={tok.pos}
                        className={`min-w-[66px] rounded px-1.5 py-1 text-center font-semibold ${
                          tok.pos === activePos
                            ? "bg-cyan-500/20 text-cyan-200"
                            : "bg-obsidian-900/90 text-slate-300"
                        }`}
                      >
                        <div className="text-[9px] text-slate-500">{tok.pos}</div>
                        <div className="max-w-[74px] truncate">{tok.display}</div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.stages.map((st, sIdx) => (
                    <tr key={st.stage}>
                      <td className="whitespace-nowrap pr-2 font-semibold text-slate-400">
                        {st.stage}
                      </td>
                      {st.cells.map((cell) => {
                        const isSel =
                          selectedCell?.stageIdx === sIdx && selectedCell?.pos === cell.pos;
                        const metricProb =
                          colorMode === "final" ? cell.final_token_prob : cell.top1_prob;
                        const bg = logitLensBg(metricProb, colorMode, cell.entropy);
                        return (
                          <td
                            key={cell.pos}
                            onClick={() => setSelectedCell({ stageIdx: sIdx, pos: cell.pos })}
                            style={{ backgroundColor: bg }}
                            className={`cursor-pointer rounded px-1.5 py-1.5 text-center transition hover:brightness-125 ${
                              isSel ? "ring-2 ring-white" : "ring-1 ring-obsidian-950/60"
                            }`}
                          >
                            <div className="max-w-[72px] truncate font-semibold text-slate-100">
                              {cell.top1_display}
                            </div>
                            <div className="text-[9px] text-slate-200/80 tabular-nums">
                              {colorMode === "entropy"
                                ? `H=${cell.entropy.toFixed(1)}`
                                : `${(metricProb * 100).toFixed(0)}%`}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Right Cell Inspector & Layer Probability Trajectory (4 cols) */}
          <div className="glass-panel flex flex-col justify-between rounded-xl p-4 xl:col-span-4">
            {activeCell && (
              <div className="space-y-4">
                <div className="border-b border-obsidian-700/80 pb-3">
                  <div className="flex items-center justify-between font-mono text-xs">
                    <span className="font-bold text-emerald-300">{activeStageName}</span>
                    <span className="text-slate-400">
                      Input [{activePos}]:{" "}
                      <span className="text-cyan-300">{data.tokens[activePos]?.display}</span>
                    </span>
                  </div>
                  <div className="mt-2 grid grid-cols-3 gap-2 font-mono text-[11px] tabular-nums">
                    <div className="rounded-lg bg-obsidian-950/80 p-2">
                      <div className="text-[9px] uppercase text-slate-500">Entropy</div>
                      <div className="font-semibold text-slate-200">
                        {activeCell.entropy.toFixed(2)}
                      </div>
                    </div>
                    <div className="rounded-lg bg-obsidian-950/80 p-2">
                      <div className="text-[9px] uppercase text-slate-500">KL → Final</div>
                      <div className="font-semibold text-amber-300">
                        {activeCell.kl_to_final.toFixed(2)}
                      </div>
                    </div>
                    <div className="rounded-lg bg-obsidian-950/80 p-2">
                      <div className="text-[9px] uppercase text-slate-500">Final Rank</div>
                      <div className="font-semibold text-emerald-300">
                        #{activeCell.final_token_rank}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Top-6 distribution at selected stage & position */}
                <div className="space-y-1.5">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
                    Top-6 Decoded Candidates at {activeStageName}
                  </div>
                  {activeCell.top6.map((cand, i) => (
                    <div key={cand.id} className="space-y-0.5 font-mono text-xs">
                      <div className="flex justify-between tabular-nums">
                        <span className={i === 0 ? "font-bold text-emerald-300" : "text-slate-200"}>
                          {cand.display}
                        </span>
                        <span className="text-slate-400">{(cand.prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-obsidian-800">
                        <div
                          className={`h-full rounded-full ${
                            i === 0 ? "bg-emerald-400" : "bg-cyan-500/70"
                          }`}
                          style={{ width: `${Math.max(2, cand.prob * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Depth trajectory of final token probability */}
                <div className="rounded-lg border border-obsidian-700/70 bg-obsidian-950/70 p-3">
                  <div className="mb-2 flex items-center justify-between font-mono text-[10px] text-slate-400">
                    <span>
                      Final Prediction Trajectory (
                      <span className="text-emerald-300">
                        {data.stages[data.stages.length - 1]?.cells[activePos]?.top1_display}
                      </span>
                      )
                    </span>
                    <span>Embed → Final</span>
                  </div>
                  <div className="flex h-20 items-end gap-1.5">
                    {data.stages.map((st, idx) => {
                      const p = st.cells[activePos]?.final_token_prob ?? 0;
                      const sel = selectedCell?.stageIdx === idx;
                      return (
                        <button
                          key={st.stage}
                          onClick={() => setSelectedCell({ stageIdx: idx, pos: activePos })}
                          className="group flex flex-1 flex-col items-center gap-1"
                        >
                          <div className="flex h-14 w-full items-end rounded bg-obsidian-900/90 p-0.5">
                            <div
                              className={`w-full rounded-sm transition-all ${
                                sel ? "bg-emerald-300" : "bg-emerald-500/70 group-hover:bg-emerald-400"
                              }`}
                              style={{ height: `${Math.max(4, p * 100)}%` }}
                            />
                          </div>
                          <span className="truncate font-mono text-[8px] text-slate-500">
                            {st.stage.replace(".attn", "a").replace(".out", "")}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Direct Logit Attribution Waterfall */}
      {data && (
        <div className="glass-panel rounded-xl p-4">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                <BarChart3 className="h-4 w-4 text-cyan-400" />
                Direct Logit Attribution (Exact Additive Residual Decomposition)
              </h2>
              <p className="text-xs text-slate-400">
                Exact logit contribution of Embedding, every Attention Head, and every SwiGLU MLP to{" "}
                <span className="font-mono font-semibold text-emerald-300">
                  {data.dla.target_display}
                </span>
                {data.dla.foil_display && (
                  <>
                    {" "}
                    minus foil{" "}
                    <span className="font-mono font-semibold text-rose-300">
                      {data.dla.foil_display}
                    </span>
                  </>
                )}{" "}
                at final position
              </p>
            </div>

            <div className="flex items-center gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 font-mono text-[11px] tabular-nums">
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
              <span className="text-slate-300">
                Actual={data.dla.actual_logit.toFixed(3)} · Σ={data.dla.sum_of_contribs.toFixed(3)}
              </span>
              <span className="text-emerald-300">(Δ={reconDiff.toExponential(1)})</span>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
            {dlaComponents.map((comp) => {
              const pct = Math.min(100, (Math.abs(comp.contrib) / maxAbsDla) * 100);
              const pos = comp.contrib >= 0;
              const kindBadge =
                comp.kind === "head"
                  ? "text-cyan-300"
                  : comp.kind === "mlp"
                    ? "text-violet-300"
                    : "text-amber-300";
              return (
                <div
                  key={comp.label}
                  className="grid grid-cols-12 items-center gap-2 rounded-lg bg-obsidian-950/70 px-3 py-1.5 font-mono text-xs"
                >
                  <div className={`col-span-3 font-semibold ${kindBadge}`}>{comp.label}</div>
                  {/* Symmetrical diverging bar */}
                  <div className="col-span-7 flex items-center">
                    <div className="flex h-3 w-1/2 justify-end border-r border-slate-600/80 pr-0.5">
                      {!pos && (
                        <div
                          className="h-full rounded-l bg-rose-500/85"
                          style={{ width: `${Math.max(2, pct)}%` }}
                        />
                      )}
                    </div>
                    <div className="flex h-3 w-1/2 justify-start pl-0.5">
                      {pos && (
                        <div
                          className="h-full rounded-r bg-emerald-400/90"
                          style={{ width: `${Math.max(2, pct)}%` }}
                        />
                      )}
                    </div>
                  </div>
                  <div
                    className={`col-span-2 text-right text-[11px] font-semibold tabular-nums ${
                      pos ? "text-emerald-300" : "text-rose-300"
                    }`}
                  >
                    {pos ? "+" : ""}
                    {comp.contrib.toFixed(3)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
