import React, { useEffect, useState } from "react";
import { Binary, Play, Sparkles, Terminal } from "lucide-react";
import { api } from "../api";
import type {
  BPEInspectResponse,
  DatasetMeta,
  GeneratedTokenStep,
  GenerateResponse,
} from "../types";
import { TOKEN_CHIP_PALETTE } from "../utils/colors";

interface GenerateBpeTabProps {
  defaultPrompt: string;
  datasetMeta?: DatasetMeta;
  modelStep: number;
}

export const GenerateBpeTab: React.FC<GenerateBpeTabProps> = ({
  defaultPrompt,
  datasetMeta,
  modelStep,
}) => {
  const [genPrompt, setGenPrompt] = useState(defaultPrompt);
  const [maxTokens, setMaxTokens] = useState(32);
  const [temperature, setTemperature] = useState(0.65);
  const [topK, setTopK] = useState(16);
  const [topP, setTopP] = useState(0.92);
  const [genData, setGenData] = useState<GenerateResponse | null>(null);
  const [genLoading, setGenLoading] = useState(false);
  const [selectedStep, setSelectedStep] = useState<GeneratedTokenStep | null>(null);

  const [bpeText, setBpeText] = useState(defaultPrompt);
  const [bpeData, setBpeData] = useState<BPEInspectResponse | null>(null);
  const [mergeFilter, setMergeFilter] = useState("");

  useEffect(() => {
    setGenPrompt(defaultPrompt);
    setBpeText(defaultPrompt);
  }, [defaultPrompt]);

  const handleGenerate = async (overridePrompt?: string) => {
    const p = overridePrompt ?? genPrompt;
    setGenLoading(true);
    try {
      const res = await api.generate({
        prompt: p,
        max_new_tokens: maxTokens,
        temperature,
        top_k: topK,
        top_p: topP,
      });
      setGenData(res);
      setSelectedStep(res.tokens[0] ?? null);
    } finally {
      setGenLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      api.inspectBPE(bpeText).then(setBpeData).catch(() => {});
    }, 150);
    return () => clearTimeout(timer);
  }, [bpeText, modelStep]);

  useEffect(() => {
    if (!genData && defaultPrompt) {
      handleGenerate(defaultPrompt);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultPrompt]);

  const filteredMerges = (bpeData?.merges ?? []).filter((m) =>
    mergeFilter.trim()
      ? m.merged.toLowerCase().includes(mergeFilter.toLowerCase()) ||
        m.left.toLowerCase().includes(mergeFilter.toLowerCase()) ||
        m.right.toLowerCase().includes(mergeFilter.toLowerCase())
      : true
  );

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
      {/* Left 6 cols: Autoregressive Generation Studio */}
      <div className="space-y-4 xl:col-span-6">
        <div className="glass-panel rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
              <Terminal className="h-4 w-4 text-emerald-400" />
              Autoregressive Text Generation Studio
            </h2>
            <span className="font-mono text-[11px] text-slate-400">
              Nucleus (Top-P) + Top-K Sampler
            </span>
          </div>

          <textarea
            rows={3}
            value={genPrompt}
            onChange={(e) => setGenPrompt(e.target.value)}
            placeholder="Enter prompt to continue..."
            className="w-full rounded-lg border border-obsidian-700 bg-obsidian-950 p-3 font-mono text-xs text-slate-100 focus:border-emerald-500 focus:outline-none"
          />

          {datasetMeta?.sample_prompts && (
            <div className="flex flex-wrap gap-1.5">
              {datasetMeta.sample_prompts.map((sp, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setGenPrompt(sp);
                    handleGenerate(sp);
                  }}
                  className="max-w-[260px] truncate rounded border border-obsidian-700 bg-obsidian-900 px-2 py-0.5 font-mono text-[10px] text-slate-300 hover:border-emerald-500/40 hover:text-emerald-200"
                >
                  {sp}
                </button>
              ))}
            </div>
          )}

          {/* Sampling Sliders */}
          <div className="grid grid-cols-2 gap-3 rounded-lg border border-obsidian-700/70 bg-obsidian-950/70 p-3 font-mono text-[11px] sm:grid-cols-4">
            <div>
              <div className="flex justify-between text-slate-400">
                <span>Temp</span>
                <span className="text-emerald-300">{temperature.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="mt-1 w-full accent-emerald-400"
              />
            </div>
            <div>
              <div className="flex justify-between text-slate-400">
                <span>Top-K</span>
                <span className="text-cyan-300">{topK}</span>
              </div>
              <input
                type="range"
                min={1}
                max={64}
                step={1}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="mt-1 w-full accent-cyan-400"
              />
            </div>
            <div>
              <div className="flex justify-between text-slate-400">
                <span>Top-P</span>
                <span className="text-violet-300">{topP.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.02}
                value={topP}
                onChange={(e) => setTopP(Number(e.target.value))}
                className="mt-1 w-full accent-violet-400"
              />
            </div>
            <div>
              <div className="flex justify-between text-slate-400">
                <span>Tokens</span>
                <span className="text-amber-300">{maxTokens}</span>
              </div>
              <input
                type="range"
                min={8}
                max={96}
                step={4}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                className="mt-1 w-full accent-amber-400"
              />
            </div>
          </div>

          <button
            onClick={() => handleGenerate()}
            disabled={genLoading}
            className="inline-flex w-full items-center justify-center gap-2 rounded-lg border border-emerald-500/50 bg-emerald-500/20 px-4 py-2 font-mono text-xs font-semibold text-emerald-200 transition hover:bg-emerald-500/30 disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5 fill-current" />
            {genLoading ? "Sampling Autoregressively..." : "Generate Tokens"}
          </button>
        </div>

        {/* Interactive Token-by-Token Output */}
        {genData && (
          <div className="glass-panel rounded-xl p-4 space-y-4">
            <div>
              <div className="mb-1.5 flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-200">
                  Generated Continuation (Click any token to inspect its distribution)
                </span>
                <span className="font-mono text-[10px] text-slate-500">
                  Tint = token confidence
                </span>
              </div>
              <div className="rounded-lg border border-obsidian-700/80 bg-obsidian-950 p-3 font-mono text-xs leading-relaxed">
                <span className="text-slate-400">{genData.prompt}</span>
                {genData.tokens.map((t) => {
                  const isSel = selectedStep?.step === t.step;
                  const bg =
                    t.prob > 0.7
                      ? "bg-emerald-500/25 text-emerald-100 border-emerald-500/40"
                      : t.prob > 0.3
                        ? "bg-cyan-500/20 text-cyan-100 border-cyan-500/30"
                        : "bg-amber-500/20 text-amber-100 border-amber-500/35";
                  return (
                    <button
                      key={t.step}
                      onClick={() => setSelectedStep(t)}
                      title={`p=${(t.prob * 100).toFixed(1)}% · H=${t.entropy.toFixed(2)}`}
                      className={`mx-[1px] my-[1px] inline-block rounded border px-1 py-0.5 transition ${bg} ${
                        isSel ? "ring-2 ring-white" : "hover:brightness-125"
                      }`}
                    >
                      {t.text === "\n" ? "↵\n" : t.text}
                    </button>
                  );
                })}
              </div>
            </div>

            {selectedStep && (
              <div className="rounded-lg border border-obsidian-700/70 bg-obsidian-950/75 p-3">
                <div className="mb-2 flex items-center justify-between font-mono text-xs">
                  <span className="flex items-center gap-1.5 font-semibold text-emerald-300">
                    <Sparkles className="h-3.5 w-3.5" />
                    Step #{selectedStep.step + 1}: Chosen{" "}
                    <span className="rounded bg-emerald-500/20 px-1.5 py-0.5">
                      {selectedStep.display}
                    </span>
                  </span>
                  <span className="text-[11px] text-slate-400 tabular-nums">
                    p={(selectedStep.prob * 100).toFixed(1)}% · H={selectedStep.entropy.toFixed(2)} nats
                  </span>
                </div>
                <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
                  {selectedStep.top_candidates.map((c) => (
                    <div key={c.id} className="space-y-0.5 font-mono text-[11px]">
                      <div className="flex justify-between tabular-nums">
                        <span
                          className={
                            c.id === selectedStep.id
                              ? "font-bold text-emerald-300"
                              : "text-slate-300"
                          }
                        >
                          {c.display}
                        </span>
                        <span className="text-slate-400">{(c.prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-obsidian-800">
                        <div
                          className={`h-full rounded-full ${
                            c.id === selectedStep.id ? "bg-emerald-400" : "bg-cyan-500/60"
                          }`}
                          style={{ width: `${Math.max(2, c.prob * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right 6 cols: Byte-Level BPE Tokenizer Inspector */}
      <div className="space-y-4 xl:col-span-6">
        <div className="glass-panel rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
              <Binary className="h-4 w-4 text-cyan-400" />
              Byte-Level BPE Tokenizer Lab (Built From Scratch)
            </h2>
            {bpeData && (
              <span className="font-mono text-[11px] text-cyan-300 tabular-nums">
                Vocab {bpeData.vocab_size} (256 bytes + 3 spec + {bpeData.bpe_stats.num_merges} merges)
              </span>
            )}
          </div>

          <textarea
            rows={2}
            value={bpeText}
            onChange={(e) => setBpeText(e.target.value)}
            placeholder="Type any text, code, or Unicode (zero <UNK> tokens ever)..."
            className="w-full rounded-lg border border-obsidian-700 bg-obsidian-950 p-2.5 font-mono text-xs text-slate-100 focus:border-cyan-500 focus:outline-none"
          />

          {bpeData && (
            <>
              {/* Compression telemetry */}
              <div className="grid grid-cols-3 gap-2 font-mono text-xs tabular-nums">
                <div className="rounded-lg bg-obsidian-950/80 p-2.5">
                  <div className="text-[10px] uppercase text-slate-500">UTF-8 Bytes</div>
                  <div className="text-base font-bold text-slate-100">
                    {bpeData.inspection.num_bytes}
                  </div>
                </div>
                <div className="rounded-lg bg-obsidian-950/80 p-2.5">
                  <div className="text-[10px] uppercase text-slate-500">BPE Tokens</div>
                  <div className="text-base font-bold text-cyan-300">
                    {bpeData.inspection.num_tokens}
                  </div>
                </div>
                <div className="rounded-lg bg-obsidian-950/80 p-2.5">
                  <div className="text-[10px] uppercase text-slate-500">Compression</div>
                  <div className="text-base font-bold text-emerald-300">
                    {bpeData.inspection.compression_ratio} B/tok
                  </div>
                </div>
              </div>

              {/* Token pills */}
              <div>
                <div className="mb-1.5 font-mono text-[10px] uppercase tracking-wider text-slate-500">
                  Encoded Token Stream (hover for ID &amp; raw UTF-8 hex)
                </div>
                <div className="flex flex-wrap gap-1 rounded-lg border border-obsidian-700/70 bg-obsidian-950 p-2.5 font-mono text-xs">
                  {bpeData.inspection.tokens.map((tok, idx) => (
                    <span
                      key={idx}
                      title={`ID=${tok.id} · bytes=[${tok.bytes_hex}] · ${
                        tok.is_merged ? "Learned BPE Merge" : "Raw UTF-8 Byte"
                      }`}
                      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 ${
                        TOKEN_CHIP_PALETTE[idx % TOKEN_CHIP_PALETTE.length]
                      }`}
                    >
                      <span>{tok.display}</span>
                      <span className="text-[9px] opacity-70">{tok.id}</span>
                    </span>
                  ))}
                </div>
              </div>

              {/* Step-by-step Merge Trace */}
              {bpeData.inspection.chunk_traces.length > 0 && (
                <div>
                  <div className="mb-1.5 font-mono text-[10px] uppercase tracking-wider text-slate-500">
                    Live Merge Replay (Byte → Subword Collapse)
                  </div>
                  <div className="max-h-36 space-y-1.5 overflow-y-auto rounded-lg border border-obsidian-700/70 bg-obsidian-950/80 p-2.5 font-mono text-[11px]">
                    {bpeData.inspection.chunk_traces.slice(0, 5).map((ct, i) => (
                      <div key={i} className="space-y-0.5 border-b border-obsidian-800 pb-1.5 last:border-0">
                        <div className="text-slate-400">
                          Chunk <span className="text-cyan-300">{ct.chunk}</span>:{" "}
                          <span className="text-slate-500">[{ct.initial_bytes.join(" ")}]</span>
                        </div>
                        {ct.steps.map((st, si) => (
                          <div key={si} className="pl-3 text-[10px] text-slate-300">
                            <span className="text-slate-500">rank #{st.rank}</span>:{" "}
                            <span className="text-amber-300">{st.pair[0]}</span> +{" "}
                            <span className="text-amber-300">{st.pair[1]}</span> →{" "}
                            <span className="font-semibold text-emerald-300">{st.merged}</span>{" "}
                            <span className="text-slate-500">[{st.tokens_after.join(" ")}]</span>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Learned Merge Table */}
        {bpeData && (
          <div className="glass-panel rounded-xl p-4">
            <div className="mb-2.5 flex items-center justify-between gap-2">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                Learned BPE Merge Table (Top {bpeData.merges.length})
              </h3>
              <input
                type="text"
                value={mergeFilter}
                onChange={(e) => setMergeFilter(e.target.value)}
                placeholder="Filter merges..."
                className="w-36 rounded-md border border-obsidian-700 bg-obsidian-950 px-2 py-1 font-mono text-[11px] text-slate-200 focus:border-cyan-500 focus:outline-none"
              />
            </div>
            <div className="max-h-56 overflow-y-auto">
              <table className="w-full font-mono text-[11px] tabular-nums">
                <thead className="sticky top-0 bg-obsidian-900 text-left text-[10px] uppercase text-slate-500">
                  <tr>
                    <th className="px-2 py-1">Rank</th>
                    <th className="px-2 py-1">Merge Pair</th>
                    <th className="px-2 py-1">Result</th>
                    <th className="px-2 py-1 text-right">ID</th>
                    <th className="px-2 py-1 text-right">Freq</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-obsidian-800/70">
                  {filteredMerges.slice(0, 60).map((m) => (
                    <tr key={m.rank} className="hover:bg-obsidian-800/40">
                      <td className="px-2 py-1 text-slate-500">#{m.rank}</td>
                      <td className="px-2 py-1 text-slate-300">
                        <span className="rounded bg-obsidian-800 px-1">{m.left}</span> +{" "}
                        <span className="rounded bg-obsidian-800 px-1">{m.right}</span>
                      </td>
                      <td className="px-2 py-1 font-semibold text-emerald-300">{m.merged}</td>
                      <td className="px-2 py-1 text-right text-cyan-300">{m.new_id}</td>
                      <td className="px-2 py-1 text-right text-slate-400">{m.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
