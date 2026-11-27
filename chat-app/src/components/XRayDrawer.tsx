import React, { useState } from "react";
import { Eye, Sparkles, X } from "lucide-react";
import type { TokenStep, XRaySnapshot } from "../types";

interface XRayDrawerProps {
  open: boolean;
  onClose: () => void;
  selectedToken: TokenStep | null;
  snapshot: XRaySnapshot | null;
}

export const XRayDrawer: React.FC<XRayDrawerProps> = ({
  open,
  onClose,
  selectedToken,
  snapshot,
}) => {
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);

  if (!open) return null;

  const safeLayer = snapshot ? Math.min(layer, snapshot.nLayers - 1) : 0;
  const safeHead = snapshot ? Math.min(head, snapshot.nHeads - 1) : 0;
  const matrix = snapshot?.attentionWeights?.[safeLayer]?.[safeHead] ?? [];
  const visibleTokens = snapshot ? snapshot.tokens.slice(-16) : [];
  const offset = snapshot ? Math.max(0, snapshot.tokens.length - 16) : 0;

  return (
    <aside className="fixed inset-y-0 right-0 z-40 flex w-80 flex-col border-l border-slate-800/90 bg-[#0b0f16] shadow-2xl sm:w-96">
      <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3.5">
        <div className="flex items-center gap-2 font-mono text-xs font-bold text-emerald-300">
          <Eye className="h-4 w-4" />
          Live Neural X-Ray
        </div>
        <button
          onClick={onClose}
          className="rounded-lg p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* Selected Token Softmax Probabilities */}
        <div className="rounded-xl border border-slate-800 bg-[#080b10] p-3.5 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[11px] font-semibold text-slate-300">
              Token Probability Breakdown
            </span>
            {selectedToken && (
              <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
                H = {selectedToken.entropy} nats
              </span>
            )}
          </div>

          {selectedToken ? (
            <div className="space-y-2">
              <div className="font-mono text-xs text-slate-400">
                Sampled token:{" "}
                <span className="rounded bg-slate-800 px-1.5 py-0.5 text-emerald-200">
                  {selectedToken.display}
                </span>{" "}
                ({(selectedToken.prob * 100).toFixed(1)}%)
              </div>
              <div className="space-y-1.5">
                {selectedToken.top_candidates.map((cand) => (
                  <div key={cand.id} className="space-y-0.5">
                    <div className="flex justify-between font-mono text-[11px]">
                      <span
                        className={
                          cand.id === selectedToken.id ? "font-bold text-emerald-300" : "text-slate-300"
                        }
                      >
                        {cand.display}
                      </span>
                      <span className="text-slate-400">{(cand.prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
                      <div
                        className={`h-full rounded-full ${
                          cand.id === selectedToken.id ? "bg-emerald-400" : "bg-cyan-500/60"
                        }`}
                        style={{ width: `${Math.max(1.5, cand.prob * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-500">
              Enable <strong>X-Ray Tokens</strong> in the top bar and click any colored word in a
              reply to inspect its exact next-token probability distribution.
            </p>
          )}
        </div>

        {/* Live GQA Attention Head Heatmap */}
        <div className="rounded-xl border border-slate-800 bg-[#080b10] p-3.5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[11px] font-semibold text-slate-300">
              GQA Attention Heatmap
            </span>
            {snapshot && (
              <span className="font-mono text-[10px] text-cyan-300">
                KV Group {Math.floor(safeHead / (snapshot.nHeads / snapshot.nKvHeads))} / {snapshot.nKvHeads}
              </span>
            )}
          </div>

          {snapshot && matrix.length > 0 ? (
            <>
              <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                <label className="space-y-1">
                  <span className="text-slate-500">Layer ({safeLayer})</span>
                  <select
                    value={safeLayer}
                    onChange={(e) => setLayer(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200"
                  >
                    {Array.from({ length: snapshot.nLayers }, (_, l) => (
                      <option key={l} value={l}>
                        Layer {l}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-1">
                  <span className="text-slate-500">Query Head ({safeHead})</span>
                  <select
                    value={safeHead}
                    onChange={(e) => setHead(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200"
                  >
                    {Array.from({ length: snapshot.nHeads }, (_, h) => (
                      <option key={h} value={h}>
                        Head {h} (KV{Math.floor(h / (snapshot.nHeads / snapshot.nKvHeads))})
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="overflow-x-auto">
                <div
                  className="grid gap-[2px]"
                  style={{
                    gridTemplateColumns: `46px repeat(${visibleTokens.length}, minmax(14px, 1fr))`,
                  }}
                >
                  <div />
                  {visibleTokens.map((t, cIdx) => (
                    <div
                      key={cIdx}
                      title={t}
                      className="truncate text-center font-mono text-[8px] text-slate-500"
                    >
                      {t.slice(0, 3)}
                    </div>
                  ))}

                  {visibleTokens.map((rowTok, rIdx) => {
                    const actualR = offset + rIdx;
                    return (
                      <React.Fragment key={rIdx}>
                        <div
                          title={rowTok}
                          className="truncate pr-1 text-right font-mono text-[9px] text-slate-400"
                        >
                          {rowTok}
                        </div>
                        {visibleTokens.map((_, cIdx) => {
                          const actualC = offset + cIdx;
                          const val = matrix[actualR]?.[actualC] ?? 0;
                          const alpha = Math.min(1, Math.pow(val, 0.7));
                          return (
                            <div
                              key={cIdx}
                              title={`${rowTok} → ${visibleTokens[cIdx]}: ${(val * 100).toFixed(1)}%`}
                              className="aspect-square rounded-[2px]"
                              style={{
                                backgroundColor:
                                  actualC > actualR
                                    ? "rgba(15, 23, 42, 0.35)"
                                    : `rgba(16, 185, 129, ${Math.max(0.06, alpha)})`,
                              }}
                            />
                          );
                        })}
                      </React.Fragment>
                    );
                  })}
                </div>
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center gap-2 py-6 text-center text-xs text-slate-500">
              <Sparkles className="h-5 w-5 text-slate-600" />
              Send a message and click <strong>Neural X-Ray</strong> on any reply to inspect its
              exact 6-layer GQA attention weights.
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};
