import React, { useCallback, useEffect, useState } from "react";
import { ArrowRight, GitBranch, Play, Sparkles } from "lucide-react";
import { api } from "../api";
import type { DatasetMeta, PatchingAnalysis } from "../types";
import { divergingPatchColor } from "../utils/colors";

interface PatchingTabProps {
  datasetMeta?: DatasetMeta;
  modelStep: number;
}

export const PatchingTab: React.FC<PatchingTabProps> = ({ datasetMeta, modelStep }) => {
  const firstPreset = datasetMeta?.patching_presets?.[0];

  const [cleanPrompt, setCleanPrompt] = useState(
    firstPreset?.clean_prompt ||
      "Echo alpha crimson beta cobalt gamma amber delta emerald alpha crimson beta"
  );
  const [corruptPrompt, setCorruptPrompt] = useState(
    firstPreset?.corrupted_prompt ||
      "Echo alpha crimson beta violet gamma amber delta emerald alpha crimson beta"
  );
  const [cleanTarget, setCleanTarget] = useState(firstPreset?.clean_target || " cobalt");
  const [corruptTarget, setCorruptTarget] = useState(firstPreset?.corrupted_target || " violet");

  const [data, setData] = useState<PatchingAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{
    stage: string;
    pos: number;
    recovery: number;
    ld: number;
  } | null>(null);

  useEffect(() => {
    if (firstPreset) {
      setCleanPrompt(firstPreset.clean_prompt);
      setCorruptPrompt(firstPreset.corrupted_prompt);
      setCleanTarget(firstPreset.clean_target);
      setCorruptTarget(firstPreset.corrupted_target);
    }
  }, [firstPreset]);

  const runPatch = useCallback(
    async (cp: string, rp: string, ct: string, rt: string) => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.runPatching({
          clean_prompt: cp,
          corrupted_prompt: rp,
          clean_target: ct || undefined,
          corrupted_target: rt || undefined,
        });
        setData(res);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    if (cleanPrompt.trim() && corruptPrompt.trim()) {
      runPatch(cleanPrompt, corruptPrompt, cleanTarget, corruptTarget);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelStep, datasetMeta?.id]);

  const totalSwing = data ? data.logit_diff_clean - data.logit_diff_corrupt : 0;

  return (
    <div className="space-y-5">
      {/* Experiment Setup */}
      <div className="glass-panel rounded-xl p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
              <GitBranch className="h-4 w-4 text-emerald-400" />
              Causal Activation Patching (Denoising Residual Stream Intervention)
            </h2>
            <p className="text-xs text-slate-400">
              Systematically patch each Clean residual vector [stage, pos] into the Corrupted forward pass to localize the exact causal circuit
            </p>
          </div>
          {datasetMeta?.patching_presets && (
            <div className="flex flex-wrap items-center gap-1.5">
              {datasetMeta.patching_presets.map((pr, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setCleanPrompt(pr.clean_prompt);
                    setCorruptPrompt(pr.corrupted_prompt);
                    setCleanTarget(pr.clean_target);
                    setCorruptTarget(pr.corrupted_target);
                    runPatch(pr.clean_prompt, pr.corrupted_prompt, pr.clean_target, pr.corrupted_target);
                  }}
                  className="inline-flex items-center gap-1 rounded-md border border-obsidian-700 bg-obsidian-900 px-2.5 py-1 font-mono text-[11px] text-slate-300 transition hover:border-emerald-500/50 hover:text-emerald-200"
                >
                  <Sparkles className="h-3 w-3 text-emerald-400" />
                  {pr.name}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-12">
          <div className="lg:col-span-5">
            <label className="mb-1 block font-mono text-[10px] uppercase tracking-wider text-emerald-400">
              Clean Prompt
            </label>
            <input
              type="text"
              value={cleanPrompt}
              onChange={(e) => setCleanPrompt(e.target.value)}
              className="w-full rounded-lg border border-emerald-500/30 bg-obsidian-950 px-3 py-2 font-mono text-xs text-slate-100 focus:border-emerald-400 focus:outline-none"
            />
          </div>
          <div className="lg:col-span-5">
            <label className="mb-1 block font-mono text-[10px] uppercase tracking-wider text-rose-400">
              Corrupted Prompt
            </label>
            <input
              type="text"
              value={corruptPrompt}
              onChange={(e) => setCorruptPrompt(e.target.value)}
              className="w-full rounded-lg border border-rose-500/30 bg-obsidian-950 px-3 py-2 font-mono text-xs text-slate-100 focus:border-rose-400 focus:outline-none"
            />
          </div>
          <div className="grid grid-cols-2 gap-2 lg:col-span-2">
            <div>
              <label className="mb-1 block font-mono text-[10px] uppercase text-emerald-300">
                Clean Ans
              </label>
              <input
                type="text"
                value={cleanTarget}
                onChange={(e) => setCleanTarget(e.target.value)}
                placeholder="auto"
                className="w-full rounded-lg border border-obsidian-700 bg-obsidian-950 px-2 py-2 font-mono text-xs text-slate-200 focus:border-emerald-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block font-mono text-[10px] uppercase text-rose-300">
                Corrupt Ans
              </label>
              <input
                type="text"
                value={corruptTarget}
                onChange={(e) => setCorruptTarget(e.target.value)}
                placeholder="auto"
                className="w-full rounded-lg border border-obsidian-700 bg-obsidian-950 px-2 py-2 font-mono text-xs text-slate-200 focus:border-rose-500 focus:outline-none"
              />
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between">
          {error ? (
            <span className="font-mono text-xs text-rose-400">{error}</span>
          ) : (
            <span className="font-mono text-[11px] text-slate-500">
              Recovery = (LD_patched − LD_corrupt) / (LD_clean − LD_corrupt) · 100% = full circuit restoration
            </span>
          )}
          <button
            onClick={() => runPatch(cleanPrompt, corruptPrompt, cleanTarget, corruptTarget)}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/50 bg-emerald-500/20 px-4 py-1.5 font-mono text-xs font-semibold text-emerald-200 transition hover:bg-emerald-500/30 disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5 fill-current" />
            {loading ? "Patching All Positions..." : "Run Causal Sweep"}
          </button>
        </div>
      </div>

      {/* Results */}
      {data && (
        <>
          {/* Clean vs Corrupted Summary */}
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <div className="glass-panel rounded-xl p-3.5 font-mono">
              <div className="text-[10px] uppercase text-emerald-400">Clean Run Prediction</div>
              <div className="mt-1 flex items-baseline justify-between tabular-nums">
                <span className="text-base font-bold text-slate-100">
                  {data.clean_target.display}
                </span>
                <span className="text-sm font-semibold text-emerald-300">
                  {(data.clean_target.clean_prob * 100).toFixed(1)}%
                </span>
              </div>
              <div className="mt-0.5 text-[11px] text-slate-400">
                LD_clean = {data.logit_diff_clean >= 0 ? "+" : ""}
                {data.logit_diff_clean.toFixed(2)}
              </div>
            </div>

            <div className="glass-panel rounded-xl p-3.5 font-mono">
              <div className="text-[10px] uppercase text-rose-400">Corrupted Run Prediction</div>
              <div className="mt-1 flex items-baseline justify-between tabular-nums">
                <span className="text-base font-bold text-slate-100">
                  {data.corrupted_target.display}
                </span>
                <span className="text-sm font-semibold text-rose-300">
                  {(data.corrupted_target.corrupt_prob * 100).toFixed(1)}%
                </span>
              </div>
              <div className="mt-0.5 text-[11px] text-slate-400">
                LD_corrupt = {data.logit_diff_corrupt >= 0 ? "+" : ""}
                {data.logit_diff_corrupt.toFixed(2)}
              </div>
            </div>

            <div className="glass-panel rounded-xl p-3.5 font-mono">
              <div className="text-[10px] uppercase text-cyan-400">Total Causal Swing</div>
              <div className="mt-1 text-xl font-bold text-cyan-200 tabular-nums">
                {totalSwing >= 0 ? "+" : ""}
                {totalSwing.toFixed(2)} logits
              </div>
              <div className="mt-0.5 text-[11px] text-slate-400">
                P({data.clean_target.display}): {(data.clean_target.corrupt_prob * 100).toFixed(1)}% →{" "}
                {(data.clean_target.clean_prob * 100).toFixed(1)}%
              </div>
            </div>

            <div className="glass-panel rounded-xl p-3.5 font-mono">
              <div className="text-[10px] uppercase text-amber-400">Peak Circuit Hotspot</div>
              {data.top_hotspots[0] && (
                <>
                  <div className="mt-1 text-base font-bold text-amber-200">
                    {data.top_hotspots[0].stage} @ [{data.top_hotspots[0].pos}]{" "}
                    {data.top_hotspots[0].clean_tok}
                  </div>
                  <div className="mt-0.5 text-[11px] text-emerald-300 tabular-nums">
                    {(data.top_hotspots[0].recovery * 100).toFixed(1)}% recovery
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Heatmap + Hotspots */}
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
            <div className="glass-panel rounded-xl p-4 xl:col-span-8">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-semibold text-slate-100">
                  Layer × Position Recovery Heatmap
                </h3>
                {hoverInfo ? (
                  <span className="font-mono text-xs text-cyan-300 tabular-nums">
                    {hoverInfo.stage} · pos {hoverInfo.pos} (
                    {data.clean_tokens[hoverInfo.pos]?.display} /{" "}
                    {data.corrupt_tokens[hoverInfo.pos]?.display}) →{" "}
                    <strong className="text-emerald-300">
                      {(hoverInfo.recovery * 100).toFixed(1)}% recovery
                    </strong>{" "}
                    (LD={hoverInfo.ld.toFixed(2)})
                  </span>
                ) : (
                  <div className="flex items-center gap-2 font-mono text-[10px] text-slate-400">
                    <span className="inline-block h-2.5 w-2.5 rounded-sm bg-rose-500" /> −100%
                    <span className="inline-block h-2.5 w-2.5 rounded-sm bg-[#101624] ring-1 ring-slate-700" />{" "}
                    0%
                    <span className="inline-block h-2.5 w-2.5 rounded-sm bg-emerald-400" /> +100%
                  </div>
                )}
              </div>

              <div className="overflow-x-auto pb-2">
                <table className="border-separate border-spacing-1 font-mono text-[11px]">
                  <thead>
                    <tr>
                      <th className="w-20 px-2 py-1 text-left text-[10px] uppercase text-slate-500">
                        Stage
                      </th>
                      {data.clean_tokens.map((ct, idx) => {
                        const rt = data.corrupt_tokens[idx];
                        const differs = ct.id !== rt?.id;
                        return (
                          <th
                            key={idx}
                            className={`min-w-[58px] rounded px-1 py-1 text-center ${
                              differs
                                ? "border border-amber-500/50 bg-amber-500/15 text-amber-200"
                                : "bg-obsidian-900 text-slate-300"
                            }`}
                          >
                            <div className="text-[9px] text-slate-500">{idx}</div>
                            <div className="max-w-[64px] truncate">{ct.display}</div>
                            {differs && rt && (
                              <div className="max-w-[64px] truncate text-[9px] text-rose-300">
                                ≠ {rt.display}
                              </div>
                            )}
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {data.rows.map((row) => (
                      <tr key={row.stage_label}>
                        <td className="whitespace-nowrap pr-2 font-semibold text-slate-300">
                          {row.stage_label}
                        </td>
                        {row.recovery.map((rec, pos) => (
                          <td
                            key={pos}
                            onMouseEnter={() =>
                              setHoverInfo({
                                stage: row.stage_label,
                                pos,
                                recovery: rec,
                                ld: row.logit_diff[pos],
                              })
                            }
                            onMouseLeave={() => setHoverInfo(null)}
                            style={{ backgroundColor: divergingPatchColor(rec) }}
                            className="h-10 cursor-crosshair rounded text-center text-[10px] font-semibold tabular-nums text-slate-100 transition hover:ring-2 hover:ring-white"
                          >
                            {Math.abs(rec) >= 0.05 ? `${(rec * 100).toFixed(0)}%` : "·"}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Top Hotspots Table */}
            <div className="glass-panel rounded-xl p-4 xl:col-span-4">
              <h3 className="mb-3 text-sm font-semibold text-slate-100">
                Top Causal Circuit Hotspots
              </h3>
              <div className="space-y-2">
                {data.top_hotspots.map((hs, i) => (
                  <div
                    key={`${hs.stage}-${hs.pos}-${i}`}
                    className="flex items-center justify-between rounded-lg bg-obsidian-950/75 px-3 py-2 font-mono text-xs tabular-nums"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] text-slate-500">#{i + 1}</span>
                        <span className="font-semibold text-cyan-300">{hs.stage}</span>
                        <span className="text-slate-500">pos {hs.pos}</span>
                      </div>
                      <div className="mt-0.5 flex items-center gap-1 text-[11px] text-slate-300">
                        <span className="text-emerald-300">{hs.clean_tok}</span>
                        {hs.clean_tok !== hs.corrupt_tok && (
                          <>
                            <ArrowRight className="h-3 w-3 text-slate-500" />
                            <span className="text-rose-300">{hs.corrupt_tok}</span>
                          </>
                        )}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-emerald-300">
                        {(hs.recovery * 100).toFixed(1)}%
                      </div>
                      <div className="text-[10px] text-slate-500">
                        LD {hs.logit_diff >= 0 ? "+" : ""}
                        {hs.logit_diff.toFixed(2)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
