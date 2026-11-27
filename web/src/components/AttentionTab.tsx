import React, { useCallback, useEffect, useState } from "react";
import { Eye, Power, RefreshCw, Sparkles, Zap } from "lucide-react";
import { api } from "../api";
import type { AttentionAnalysis, DatasetMeta, HeadData } from "../types";
import { attentionColor, roleBadgeClass } from "../utils/colors";

interface AttentionTabProps {
  defaultPrompt: string;
  datasetMeta?: DatasetMeta;
  modelStep: number;
}

const MiniHeatmap: React.FC<{ matrix: number[][]; size?: number }> = ({ matrix, size = 76 }) => {
  const T = matrix.length;
  if (T === 0) return null;
  const cell = size / T;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="rounded bg-obsidian-950">
      {matrix.map((row, i) =>
        row.map((w, j) =>
          j > i || w < 0.01 ? null : (
            <rect
              key={`${i}-${j}`}
              x={j * cell}
              y={i * cell}
              width={Math.max(1, cell + 0.3)}
              height={Math.max(1, cell + 0.3)}
              fill={attentionColor(w)}
            />
          )
        )
      )}
    </svg>
  );
};

export const AttentionTab: React.FC<AttentionTabProps> = ({
  defaultPrompt,
  datasetMeta,
  modelStep,
}) => {
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [data, setData] = useState<AttentionAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [selectedHead, setSelectedHead] = useState<[number, number]>([1, 0]);
  const [ablatedHeads, setAblatedHeads] = useState<[number, number][]>([]);
  const [ablationMode, setAblationMode] = useState<"zero" | "mean">("zero");
  const [hoverCell, setHoverCell] = useState<{ q: number; k: number; w: number } | null>(null);
  const [activeQueryPos, setActiveQueryPos] = useState<number | null>(null);

  useEffect(() => {
    setPrompt(defaultPrompt);
  }, [defaultPrompt]);

  const fetchAttention = useCallback(
    async (
      usePrompt: string,
      useAblated: [number, number][],
      useMode: "zero" | "mean"
    ) => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.analyzeAttention({
          prompt: usePrompt,
          ablate_heads: useAblated,
          ablation_mode: useMode,
        });
        setData(res);
        setSelectedHead((prev) => {
          const [l, h] = prev;
          if (l < res.n_layers && h < res.n_heads) return prev;
          return [Math.min(1, res.n_layers - 1), 0];
        });
        setActiveQueryPos(res.tokens.length - 1);
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
      fetchAttention(prompt, ablatedHeads, ablationMode);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelStep]);

  const isAblated = (l: number, h: number) =>
    ablatedHeads.some(([al, ah]) => al === l && ah === h);

  const toggleAblate = (l: number, h: number) => {
    const next: [number, number][] = isAblated(l, h)
      ? ablatedHeads.filter(([al, ah]) => !(al === l && ah === h))
      : [...ablatedHeads, [l, h]];
    setAblatedHeads(next);
    fetchAttention(prompt, next, ablationMode);
  };

  const clearAblations = () => {
    setAblatedHeads([]);
    fetchAttention(prompt, [], ablationMode);
  };

  const activeHeadData: HeadData | null =
    data?.layers[selectedHead[0]]?.heads[selectedHead[1]] ?? null;

  const tokens = data?.tokens ?? [];
  const T = tokens.length;
  const queryIdx = activeQueryPos != null && activeQueryPos < T ? activeQueryPos : Math.max(0, T - 1);

  return (
    <div className="space-y-5">
      {/* Prompt & Controls Bar */}
      <div className="glass-panel rounded-xl p-4">
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") fetchAttention(prompt, ablatedHeads, ablationMode);
            }}
            placeholder="Enter a prompt to inspect causal attention heads..."
            className="min-w-[260px] flex-1 rounded-lg border border-obsidian-700 bg-obsidian-950 px-3 py-2 font-mono text-xs text-slate-100 focus:border-emerald-500 focus:outline-none"
          />
          <select
            value={ablationMode}
            onChange={(e) => {
              const m = e.target.value as "zero" | "mean";
              setAblationMode(m);
              if (ablatedHeads.length) fetchAttention(prompt, ablatedHeads, m);
            }}
            className="rounded-lg border border-obsidian-700 bg-obsidian-950 px-2.5 py-2 font-mono text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
          >
            <option value="zero">Zero Ablation</option>
            <option value="mean">Mean Ablation</option>
          </select>
          <button
            onClick={() => fetchAttention(prompt, ablatedHeads, ablationMode)}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/50 bg-emerald-500/20 px-3.5 py-2 font-mono text-xs font-semibold text-emerald-200 transition hover:bg-emerald-500/30 disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            Inspect Heads
          </button>
          {ablatedHeads.length > 0 && (
            <button
              onClick={clearAblations}
              className="rounded-lg border border-rose-500/40 bg-rose-500/15 px-3 py-2 font-mono text-xs text-rose-200 hover:bg-rose-500/25"
            >
              Restore {ablatedHeads.length} Ablated
            </button>
          )}
        </div>

        {/* Sample prompt chips */}
        {datasetMeta?.sample_prompts && datasetMeta.sample_prompts.length > 0 && (
          <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
            <span className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
              Presets:
            </span>
            {datasetMeta.sample_prompts.map((sp, i) => (
              <button
                key={i}
                onClick={() => {
                  setPrompt(sp);
                  fetchAttention(sp, ablatedHeads, ablationMode);
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

      {/* All-Heads Miniature Grid */}
      {data && (
        <div className="glass-panel rounded-xl p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                <Eye className="h-4 w-4 text-cyan-400" />
                Attention Head Overview ({data.n_layers} Layers × {data.n_heads} Heads)
              </h2>
              <p className="text-xs text-slate-400">
                Click any head to zoom in · Toggle the <span className="text-rose-300">Power</span> switch to causally ablate that head
              </p>
            </div>
            <div className="flex items-center gap-2 font-mono text-[11px] text-slate-400">
              <span>{T} tokens</span>
              <span>·</span>
              <span>FlashAttention-2 Exact Probs</span>
            </div>
          </div>

          <div className="space-y-3">
            {data.layers.map((layerObj) => (
              <div key={layerObj.layer} className="flex flex-wrap items-stretch gap-3">
                <div className="flex w-14 shrink-0 items-center justify-center rounded-lg border border-obsidian-700/70 bg-obsidian-950/70 font-mono text-xs font-semibold text-slate-400">
                  L{layerObj.layer}
                </div>
                <div className="grid flex-1 grid-cols-2 gap-2.5 sm:grid-cols-4">
                  {layerObj.heads.map((hd) => {
                    const selected =
                      selectedHead[0] === hd.layer && selectedHead[1] === hd.head;
                    const ablated = isAblated(hd.layer, hd.head);
                    return (
                      <div
                        key={hd.head}
                        onClick={() => setSelectedHead([hd.layer, hd.head])}
                        className={`cursor-pointer rounded-xl border p-2.5 transition ${
                          ablated
                            ? "border-rose-500/60 bg-rose-950/20 opacity-75"
                            : selected
                              ? "border-emerald-400/70 bg-emerald-500/10 shadow-[0_0_20px_rgba(16,185,129,0.14)]"
                              : "border-obsidian-700/80 bg-obsidian-950/60 hover:border-slate-500"
                        }`}
                      >
                        <div className="mb-2 flex items-center justify-between gap-1">
                          <span className="font-mono text-xs font-bold text-slate-100">
                            L{hd.layer}.H{hd.head}
                          </span>
                          <div className="flex items-center gap-1">
                            <span
                              className={`rounded border px-1.5 py-0.5 font-mono text-[9px] font-medium ${roleBadgeClass(
                                hd.role
                              )}`}
                            >
                              {hd.role}
                            </span>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleAblate(hd.layer, hd.head);
                              }}
                              title={ablated ? "Restore head" : "Ablate (knock out) this head"}
                              className={`rounded p-1 transition ${
                                ablated
                                  ? "bg-rose-500/30 text-rose-200 hover:bg-rose-500/40"
                                  : "bg-obsidian-800 text-slate-400 hover:bg-rose-500/20 hover:text-rose-300"
                              }`}
                            >
                              <Power className="h-3 w-3" />
                            </button>
                          </div>
                        </div>

                        <div className="flex items-center gap-2.5">
                          <MiniHeatmap matrix={hd.matrix} size={68} />
                          <div className="min-w-0 flex-1 space-y-1 font-mono text-[10px] tabular-nums text-slate-400">
                            <div className="flex justify-between">
                              <span>Ind</span>
                              <span
                                className={
                                  hd.induction_score > 0.22
                                    ? "font-semibold text-emerald-300"
                                    : "text-slate-300"
                                }
                              >
                                {(hd.induction_score * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span>Prev</span>
                              <span
                                className={
                                  hd.prev_token_score > 0.3
                                    ? "font-semibold text-cyan-300"
                                    : "text-slate-300"
                                }
                              >
                                {(hd.prev_token_score * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span>BOS</span>
                              <span className="text-slate-300">
                                {(hd.bos_sink_score * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span>Ent</span>
                              <span className="text-slate-300">{hd.norm_entropy.toFixed(2)}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Zoomed Head Inspector + BertViz Arcs + Ablation Impact */}
      {data && activeHeadData && (
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
          {/* Zoomed Heatmap (5 cols) */}
          <div className="glass-panel rounded-xl p-4 xl:col-span-5">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm font-bold text-emerald-300">
                  L{activeHeadData.layer}.H{activeHeadData.head}
                </span>
                <span
                  className={`rounded border px-2 py-0.5 font-mono text-[10px] ${roleBadgeClass(
                    activeHeadData.role
                  )}`}
                >
                  {activeHeadData.role}
                </span>
              </div>
              {hoverCell ? (
                <span className="font-mono text-[11px] text-cyan-300 tabular-nums">
                  {tokens[hoverCell.q]?.display} → {tokens[hoverCell.k]?.display}:{" "}
                  {(hoverCell.w * 100).toFixed(1)}%
                </span>
              ) : (
                <span className="font-mono text-[11px] text-slate-500">
                  Rows = Query · Cols = Key
                </span>
              )}
            </div>

            <div className="overflow-x-auto pb-2">
              <div className="inline-block">
                {/* Column Key labels */}
                <div className="mb-1 flex pl-16">
                  {tokens.map((tok, j) => (
                    <div
                      key={j}
                      className="flex h-12 w-6 items-end justify-center pb-1 font-mono text-[9px] text-slate-400"
                    >
                      <span className="origin-bottom-left -rotate-45 whitespace-nowrap">
                        {tok.display.slice(0, 8)}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Matrix rows */}
                {activeHeadData.matrix.map((row, i) => (
                  <div
                    key={i}
                    className={`flex items-center ${
                      queryIdx === i ? "rounded bg-emerald-500/10" : ""
                    }`}
                  >
                    <button
                      onClick={() => setActiveQueryPos(i)}
                      className={`w-16 shrink-0 truncate pr-2 text-right font-mono text-[10px] transition ${
                        queryIdx === i
                          ? "font-bold text-emerald-300"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {tokens[i]?.display}
                    </button>
                    {row.map((w, j) => (
                      <div
                        key={j}
                        onMouseEnter={() => {
                          setHoverCell({ q: i, k: j, w });
                          setActiveQueryPos(i);
                        }}
                        onMouseLeave={() => setHoverCell(null)}
                        style={{
                          backgroundColor: j > i ? "#07090e" : attentionColor(w),
                        }}
                        className="h-6 w-6 shrink-0 border border-obsidian-950/60 transition-transform hover:z-10 hover:scale-110 hover:border-white/70"
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* BertViz Token-to-Token Attention Arcs (4 cols) */}
          <div className="glass-panel rounded-xl p-4 xl:col-span-4">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="flex items-center gap-1.5 text-sm font-semibold text-slate-100">
                <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
                Attention Flow Arcs
              </h3>
              <span className="font-mono text-[11px] text-slate-400">
                Query pos {queryIdx}:{" "}
                <span className="text-emerald-300">{tokens[queryIdx]?.display}</span>
              </span>
            </div>

            <div className="max-h-[420px] overflow-y-auto pr-1">
              {T > 0 && (
                <svg
                  viewBox={`0 0 320 ${Math.max(120, T * 24 + 10)}`}
                  className="w-full select-none"
                >
                  {/* Bezier attention lines from active queryIdx on left to attended keys on right */}
                  {activeHeadData.matrix[queryIdx]?.map((w, kIdx) => {
                    if (kIdx > queryIdx || w < 0.015) return null;
                    const y1 = queryIdx * 24 + 14;
                    const y2 = kIdx * 24 + 14;
                    return (
                      <path
                        key={kIdx}
                        d={`M 98 ${y1} C 160 ${y1}, 160 ${y2}, 222 ${y2}`}
                        fill="none"
                        stroke={w > 0.35 ? "#34d399" : "#38bdf8"}
                        strokeWidth={Math.max(1, w * 7.5)}
                        strokeOpacity={Math.max(0.15, Math.min(0.95, w * 1.15))}
                      />
                    );
                  })}

                  {/* Left Query tokens & Right Key tokens */}
                  {tokens.map((tok, idx) => {
                    const y = idx * 24 + 14;
                    const isQ = idx === queryIdx;
                    const attnW = activeHeadData.matrix[queryIdx]?.[idx] ?? 0;
                    return (
                      <g key={idx}>
                        {/* Left query pill */}
                        <g
                          className="cursor-pointer"
                          onClick={() => setActiveQueryPos(idx)}
                          onMouseEnter={() => setActiveQueryPos(idx)}
                        >
                          <rect
                            x={4}
                            y={y - 9}
                            width={90}
                            height={18}
                            rx={4}
                            fill={isQ ? "rgba(16,185,129,0.25)" : "#0c1018"}
                            stroke={isQ ? "#34d399" : "#1e293b"}
                          />
                          <text
                            x={88}
                            y={y + 3.5}
                            textAnchor="end"
                            className={`font-mono text-[10px] ${
                              isQ ? "fill-emerald-200 font-bold" : "fill-slate-400"
                            }`}
                          >
                            {tok.display.slice(0, 11)}
                          </text>
                        </g>

                        {/* Right key pill */}
                        <rect
                          x={226}
                          y={y - 9}
                          width={90}
                          height={18}
                          rx={4}
                          fill={
                            idx <= queryIdx && attnW > 0.05
                              ? `rgba(16,185,129,${Math.min(0.45, attnW * 0.85)})`
                              : "#0c1018"
                          }
                          stroke={attnW > 0.25 ? "#34d399" : "#1e293b"}
                        />
                        <text
                          x={232}
                          y={y + 3.5}
                          className={`font-mono text-[10px] ${
                            attnW > 0.2 ? "fill-emerald-100 font-bold" : "fill-slate-400"
                          }`}
                        >
                          {tok.display.slice(0, 9)}
                        </text>
                        {idx <= queryIdx && attnW >= 0.02 && (
                          <text
                            x={312}
                            y={y + 3.5}
                            textAnchor="end"
                            className="fill-cyan-300 font-mono text-[9px] tabular-nums"
                          >
                            {(attnW * 100).toFixed(0)}%
                          </text>
                        )}
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>
          </div>

          {/* Live Next-Token & Ablation Impact Panel (3 cols) */}
          <div className="glass-panel rounded-xl p-4 xl:col-span-3">
            <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-slate-100">
              <Zap className="h-3.5 w-3.5 text-amber-400" />
              {data.ablation ? "Head Ablation Impact" : "Baseline Next-Token Dist"}
            </h3>

            {data.ablation ? (
              <div className="space-y-3">
                <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-2.5 font-mono text-xs">
                  <div className="flex justify-between text-rose-200">
                    <span>Ablated ({data.ablation.mode})</span>
                    <span>
                      {data.ablation.ablated_heads.map(([l, h]) => `L${l}.H${h}`).join(", ")}
                    </span>
                  </div>
                  <div className="mt-1.5 flex justify-between text-[11px] text-slate-300 tabular-nums">
                    <span>KL(base ‖ abl)</span>
                    <span className="font-bold text-amber-300">
                      {data.ablation.kl_divergence.toFixed(4)} nats
                    </span>
                  </div>
                  <div className="mt-0.5 flex justify-between text-[11px] text-slate-400 tabular-nums">
                    <span>Entropy</span>
                    <span>
                      {data.ablation.base_entropy.toFixed(2)} →{" "}
                      {data.ablation.ablated_entropy.toFixed(2)}
                    </span>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">
                    Probability &amp; Logit Shift
                  </div>
                  {data.ablation.comparisons.map((c) => (
                    <div
                      key={c.id}
                      className="rounded-lg bg-obsidian-950/70 px-2.5 py-1.5 font-mono text-[11px] tabular-nums"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-slate-100">{c.display}</span>
                        <span
                          className={
                            c.delta_prob < -0.005
                              ? "text-rose-400"
                              : c.delta_prob > 0.005
                                ? "text-emerald-400"
                                : "text-slate-400"
                          }
                        >
                          {c.delta_prob >= 0 ? "+" : ""}
                          {(c.delta_prob * 100).toFixed(1)}% ({c.delta_logit >= 0 ? "+" : ""}
                          {c.delta_logit.toFixed(2)})
                        </span>
                      </div>
                      <div className="mt-1 flex items-center gap-2 text-[10px] text-slate-400">
                        <span>Base {(c.base_prob * 100).toFixed(1)}%</span>
                        <span>→</span>
                        <span className="text-slate-200">
                          Abl {(c.ablated_prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-[11px] text-slate-400">
                  Toggle the <span className="text-rose-300">Power</span> button on any head above to measure its causal impact on these predictions.
                </p>
                <div className="rounded-lg bg-obsidian-950/70 px-2.5 py-1.5 font-mono text-[11px] text-slate-400">
                  Output Entropy:{" "}
                  <span className="text-cyan-300">{data.baseline.entropy.toFixed(3)} nats</span>
                </div>
                <div className="space-y-1.5">
                  {data.baseline.top_predictions.slice(0, 8).map((pred, idx) => (
                    <div key={pred.id} className="space-y-0.5 font-mono text-xs">
                      <div className="flex justify-between tabular-nums">
                        <span className={idx === 0 ? "font-bold text-emerald-300" : "text-slate-200"}>
                          {pred.display}
                        </span>
                        <span className="text-slate-400">{(pred.prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-obsidian-800">
                        <div
                          className={`h-full rounded-full ${
                            idx === 0 ? "bg-emerald-400" : "bg-cyan-500/70"
                          }`}
                          style={{ width: `${Math.max(2, pred.prob * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
