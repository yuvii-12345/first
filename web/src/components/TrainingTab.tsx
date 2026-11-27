import React, { useMemo, useState } from "react";
import {
  Activity,
  Cpu,
  Flame,
  Gauge,
  Play,
  RefreshCw,
  Sliders,
  Sparkles,
} from "lucide-react";
import type { DatasetMeta, LabStatus, ModelPreset, StepRecord } from "../types";

interface TrainingTabProps {
  status: LabStatus;
  history: StepRecord[];
  datasets: DatasetMeta[];
  presets: ModelPreset[];
  onStartTraining: (steps: number, lr?: number) => void;
  onStopTraining: () => void;
  onResetSession: (payload: {
    dataset_id: string;
    preset_id: string;
    custom_text?: string;
    lr: number;
    batch_size: number;
    warm_steps?: number;
  }) => Promise<void>;
}

const LossChart: React.FC<{ history: StepRecord[] }> = ({ history }) => {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const dims = { w: 720, h: 250, padL: 48, padR: 20, padT: 18, padB: 32 };
  const plotW = dims.w - dims.padL - dims.padR;
  const plotH = dims.h - dims.padT - dims.padB;

  const { minLoss, maxLoss, minStep, maxStep, trainPath, valPoints } = useMemo(() => {
    if (!history.length) {
      return { minLoss: 0, maxLoss: 6, minStep: 0, maxStep: 100, trainPath: "", valPoints: [] };
    }
    const losses = history.map((h) => h.loss);
    const valLosses = history.filter((h) => h.val_loss != null).map((h) => h.val_loss as number);
    const allVals = [...losses, ...valLosses];
    const lo = Math.max(0, Math.floor((Math.min(...allVals) - 0.2) * 10) / 10);
    const hi = Math.max(lo + 0.5, Math.ceil((Math.max(...allVals) + 0.2) * 10) / 10);
    const sMin = history[0].step;
    const sMax = Math.max(sMin + 1, history[history.length - 1].step);

    const sx = (s: number) => dims.padL + ((s - sMin) / (sMax - sMin)) * plotW;
    const sy = (v: number) => dims.padT + (1 - (v - lo) / (hi - lo)) * plotH;

    const pts = history.map((h, i) => `${i === 0 ? "M" : "L"}${sx(h.step).toFixed(1)},${sy(h.loss).toFixed(1)}`);
    const vPts = history
      .filter((h) => h.val_loss != null)
      .map((h) => ({ x: sx(h.step), y: sy(h.val_loss as number), step: h.step, val: h.val_loss as number }));

    return {
      minLoss: lo,
      maxLoss: hi,
      minStep: sMin,
      maxStep: sMax,
      trainPath: pts.join(" "),
      valPoints: vPts,
    };
  }, [history, plotH, plotW]);

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => ({
    val: minLoss + (1 - t) * (maxLoss - minLoss),
    y: dims.padT + t * plotH,
  }));

  const hovered = hoverIdx != null && history[hoverIdx] ? history[hoverIdx] : null;
  const hoverX =
    hovered && maxStep > minStep
      ? dims.padL + ((hovered.step - minStep) / (maxStep - minStep)) * plotW
      : null;

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${dims.w} ${dims.h}`}
        className="h-[245px] w-full select-none"
        onMouseMove={(e) => {
          if (!history.length) return;
          const rect = e.currentTarget.getBoundingClientRect();
          const relX = ((e.clientX - rect.left) / rect.width) * dims.w - dims.padL;
          const ratio = Math.max(0, Math.min(1, relX / plotW));
          const idx = Math.round(ratio * (history.length - 1));
          setHoverIdx(idx);
        }}
        onMouseLeave={() => setHoverIdx(null)}
      >
        <defs>
          <linearGradient id="lossArea" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.28" />
            <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {/* Horizontal grid */}
        {yTicks.map((tk, i) => (
          <g key={i}>
            <line
              x1={dims.padL}
              x2={dims.w - dims.padR}
              y1={tk.y}
              y2={tk.y}
              stroke="#1e293b"
              strokeDasharray="3 4"
            />
            <text
              x={dims.padL - 8}
              y={tk.y + 4}
              textAnchor="end"
              className="fill-slate-500 font-mono text-[10px]"
            >
              {tk.val.toFixed(2)}
            </text>
          </g>
        ))}

        {/* X axis labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((t, i) => {
          const stepVal = Math.round(minStep + t * (maxStep - minStep));
          const x = dims.padL + t * plotW;
          return (
            <text
              key={i}
              x={x}
              y={dims.h - 10}
              textAnchor="middle"
              className="fill-slate-500 font-mono text-[10px]"
            >
              {stepVal}
            </text>
          );
        })}

        {/* Area fill under train loss */}
        {trainPath && (
          <path
            d={`${trainPath} L${dims.padL + plotW},${dims.padT + plotH} L${dims.padL},${dims.padT + plotH} Z`}
            fill="url(#lossArea)"
          />
        )}

        {/* Train loss curve */}
        {trainPath && (
          <path d={trainPath} fill="none" stroke="#34d399" strokeWidth={2.2} strokeLinejoin="round" />
        )}

        {/* Val loss curve & dots */}
        {valPoints.length > 1 && (
          <path
            d={valPoints.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")}
            fill="none"
            stroke="#38bdf8"
            strokeWidth={1.8}
            strokeDasharray="4 3"
          />
        )}
        {valPoints.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={3.2} fill="#0c1018" stroke="#38bdf8" strokeWidth={1.8} />
        ))}

        {/* Hover crosshair */}
        {hoverX != null && hovered && (
          <g>
            <line
              x1={hoverX}
              x2={hoverX}
              y1={dims.padT}
              y2={dims.padT + plotH}
              stroke="#94a3b8"
              strokeWidth={1}
              strokeDasharray="2 2"
            />
            <circle
              cx={hoverX}
              cy={dims.padT + (1 - (hovered.loss - minLoss) / (maxLoss - minLoss)) * plotH}
              r={4.2}
              fill="#34d399"
              stroke="#07090e"
              strokeWidth={1.5}
            />
          </g>
        )}
      </svg>

      {/* Legend & hover readout */}
      <div className="mt-1 flex flex-wrap items-center justify-between gap-2 px-2 font-mono text-[11px]">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 text-emerald-300">
            <span className="h-2 w-3 rounded-sm bg-emerald-400" /> Train Loss
          </span>
          <span className="flex items-center gap-1.5 text-sky-300">
            <span className="h-2 w-2 rounded-full border-2 border-sky-400" /> Val Loss
          </span>
        </div>
        {hovered ? (
          <div className="flex items-center gap-3 text-slate-300 tabular-nums">
            <span>Step {hovered.step}</span>
            <span className="text-emerald-300">loss={hovered.loss.toFixed(4)}</span>
            {hovered.val_loss != null && (
              <span className="text-sky-300">val={hovered.val_loss.toFixed(4)}</span>
            )}
            <span className="text-amber-300">‖g‖={hovered.grad_norm.toFixed(3)}</span>
            <span className="text-slate-400">lr={hovered.lr.toExponential(1)}</span>
          </div>
        ) : (
          <span className="text-slate-500">Hover chart to inspect step telemetry</span>
        )}
      </div>
    </div>
  );
};

export const TrainingTab: React.FC<TrainingTabProps> = ({
  status,
  history,
  datasets,
  presets,
  onStartTraining,
  onStopTraining,
  onResetSession,
}) => {
  const [selectedDataset, setSelectedDataset] = useState(
    status.dataset_id === "custom" ? "induction" : status.dataset_id
  );
  const [selectedPreset, setSelectedPreset] = useState(status.preset_id || "micro");
  const [customCorpus, setCustomCorpus] = useState("");
  const [useCustom, setUseCustom] = useState(false);
  const [lr, setLr] = useState(status.lr || 0.005);
  const [batchSize, setBatchSize] = useState(status.batch_size || 16);
  const [trainSteps, setTrainSteps] = useState(150);
  const [resetting, setResetting] = useState(false);

  const layerGrads = status.latest_layer_grads || [];
  const maxGrad = Math.max(0.001, ...layerGrads.map((g) => g.grad_norm));

  const healthColor =
    status.grad_health === "healthy"
      ? "text-emerald-300 border-emerald-500/40 bg-emerald-500/15"
      : status.grad_health === "clipped"
        ? "text-amber-300 border-amber-500/40 bg-amber-500/15"
        : status.grad_health === "vanishing" || status.grad_health === "exploding_clipped"
          ? "text-rose-300 border-rose-500/40 bg-rose-500/15"
          : "text-slate-300 border-slate-600 bg-slate-800/50";

  const handleReset = async (warmSteps: number) => {
    setResetting(true);
    try {
      await onResetSession({
        dataset_id: selectedDataset,
        preset_id: selectedPreset,
        custom_text: useCustom && customCorpus.trim().length >= 64 ? customCorpus : undefined,
        lr,
        batch_size: batchSize,
        warm_steps: warmSteps,
      });
    } finally {
      setResetting(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* Top 4 KPI Tiles */}
      <div className="grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        <div className="glass-panel rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Train / Val Loss</span>
            <Activity className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2 font-mono tabular-nums">
            <span className="text-2xl font-bold text-slate-100">
              {status.last_loss != null ? status.last_loss.toFixed(3) : "—"}
            </span>
            <span className="text-xs text-sky-400">
              val {status.last_val_loss != null ? status.last_val_loss.toFixed(3) : "—"}
            </span>
          </div>
          <div className="mt-1 font-mono text-[11px] text-slate-500">
            Step {status.step} · {(status.total_tokens_seen / 1000).toFixed(1)}K tokens seen
          </div>
        </div>

        <div className="glass-panel rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Perplexity &amp; Grad Norm</span>
            <Flame className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2 font-mono tabular-nums">
            <span className="text-2xl font-bold text-slate-100">
              {status.last_perplexity != null ? status.last_perplexity.toFixed(2) : "—"}
            </span>
            <span className="text-xs text-amber-300">
              ‖g‖ {status.last_grad_norm != null ? status.last_grad_norm.toFixed(3) : "—"}
            </span>
          </div>
          <div className="mt-1 flex items-center gap-2">
            <span className={`rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase ${healthColor}`}>
              {status.grad_health}
            </span>
            <span className="font-mono text-[11px] text-slate-500">clip={status.grad_clip}</span>
          </div>
        </div>

        <div className="glass-panel rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Training Throughput</span>
            <Gauge className="h-4 w-4 text-cyan-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2 font-mono tabular-nums">
            <span className="text-2xl font-bold text-slate-100">
              {status.tokens_per_sec > 0 ? Math.round(status.tokens_per_sec).toLocaleString() : "—"}
            </span>
            <span className="text-xs text-cyan-300">tok/s</span>
          </div>
          <div className="mt-1 font-mono text-[11px] text-slate-500">
            {status.last_step_ms > 0 ? `${status.last_step_ms.toFixed(1)} ms/step` : "Waiting for step"} · B={status.batch_size}
          </div>
        </div>

        <div className="glass-panel rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Custom Tiled GEMM</span>
            <Cpu className="h-4 w-4 text-violet-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2 font-mono tabular-nums">
            <span className="text-2xl font-bold text-slate-100">
              {status.live_gflops > 0
                ? status.live_gflops.toFixed(1)
                : status.cumulative_gflops.toFixed(1)}
            </span>
            <span className="text-xs text-violet-300">GFLOP/s</span>
          </div>
          <div className="mt-1 font-mono text-[11px] text-slate-500">
            {status.total_matmul_calls.toLocaleString()} tiled GEMM calls · C++17 AVX
          </div>
        </div>
      </div>

      {/* Main 2-column grid */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
        {/* Left 8 cols: Loss Curve + Gradient Flow */}
        <div className="space-y-5 xl:col-span-8">
          {/* Loss Curve Panel */}
          <div className="glass-panel rounded-xl p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="text-sm font-semibold text-slate-100">
                  Live Cross-Entropy Loss &amp; Convergence
                </h2>
                <p className="text-xs text-slate-400">
                  Fused log-softmax cross-entropy · AdamW (decoupled weight decay) + Cosine Warmup
                </p>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={10}
                  max={1500}
                  step={50}
                  value={trainSteps}
                  onChange={(e) => setTrainSteps(Math.max(10, Number(e.target.value)))}
                  className="w-20 rounded-lg border border-obsidian-700 bg-obsidian-950 px-2.5 py-1 font-mono text-xs text-slate-200 focus:border-emerald-500 focus:outline-none"
                />
                {status.is_training ? (
                  <button
                    onClick={onStopTraining}
                    className="rounded-lg border border-rose-500/50 bg-rose-500/15 px-3 py-1 font-mono text-xs font-semibold text-rose-200 hover:bg-rose-500/25"
                  >
                    Pause
                  </button>
                ) : (
                  <button
                    onClick={() => onStartTraining(trainSteps, lr)}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/50 bg-emerald-500/20 px-3 py-1 font-mono text-xs font-semibold text-emerald-200 hover:bg-emerald-500/30"
                  >
                    <Play className="h-3 w-3 fill-current" />
                    Run {trainSteps} Steps
                  </button>
                )}
              </div>
            </div>
            <LossChart history={history} />
          </div>

          {/* Gradient Flow & Update Ratio Panel */}
          <div className="glass-panel rounded-xl p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="text-sm font-semibold text-slate-100">
                  Per-Layer Gradient Flow &amp; Update-to-Weight Ratio
                </h2>
                <p className="text-xs text-slate-400">
                  L2 gradient magnitude per component &amp; log₁₀(‖ΔW‖ / ‖W‖) — target healthy zone ≈ −3.0
                </p>
              </div>
              <span className="font-mono text-[11px] text-slate-400">
                {layerGrads.length} parameter groups tracked
              </span>
            </div>

            {layerGrads.length === 0 ? (
              <div className="rounded-lg border border-dashed border-obsidian-700 py-10 text-center font-mono text-xs text-slate-500">
                Run at least 1 training step to visualize per-layer gradient flow.
              </div>
            ) : (
              <div className="space-y-2">
                {layerGrads.map((g) => {
                  const pct = Math.min(100, (g.grad_norm / maxGrad) * 100);
                  // Map log10 ratio [-5.5 .. -1.5] onto 0..100%
                  const ratioPos = Math.max(0, Math.min(100, ((g.log10_update_ratio + 5.5) / 4.0) * 100));
                  const barColor =
                    g.kind === "attn"
                      ? "bg-cyan-400"
                      : g.kind === "mlp"
                        ? "bg-emerald-400"
                        : g.kind === "embed"
                          ? "bg-violet-400"
                          : "bg-amber-400";
                  const ratioOk = g.log10_update_ratio >= -3.8 && g.log10_update_ratio <= -2.2;
                  return (
                    <div
                      key={g.component}
                      className="grid grid-cols-12 items-center gap-3 rounded-lg bg-obsidian-950/60 px-3 py-1.5 font-mono text-xs"
                    >
                      <div className="col-span-3 flex items-center gap-2 truncate text-slate-200">
                        <span className={`h-2 w-2 shrink-0 rounded-full ${barColor}`} />
                        {g.component}
                      </div>
                      <div className="col-span-5">
                        <div className="h-2.5 w-full overflow-hidden rounded-full bg-obsidian-800">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${barColor}`}
                            style={{ width: `${Math.max(2, pct)}%` }}
                          />
                        </div>
                      </div>
                      <div className="col-span-2 text-right text-[11px] text-slate-300 tabular-nums">
                        {g.grad_norm.toFixed(4)}
                      </div>
                      <div className="col-span-2 flex items-center gap-1.5">
                        <div className="relative h-2 flex-1 rounded-full bg-obsidian-800">
                          {/* Ideal -3.0 marker */}
                          <div
                            className="absolute top-0 bottom-0 w-0.5 bg-slate-500/70"
                            style={{ left: "62.5%" }}
                            title="Target -3.0"
                          />
                          <div
                            className={`absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ${
                              ratioOk ? "bg-emerald-400" : "bg-amber-400"
                            }`}
                            style={{ left: `${ratioPos}%` }}
                          />
                        </div>
                        <span
                          className={`w-10 text-right text-[10px] tabular-nums ${
                            ratioOk ? "text-emerald-300" : "text-amber-300"
                          }`}
                        >
                          {g.log10_update_ratio.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Live Generation Ticker */}
          {status.latest_sample && (
            <div className="glass-panel rounded-xl p-4">
              <div className="mb-1.5 flex items-center gap-2 text-xs font-semibold text-slate-200">
                <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
                Live Periodic Sample Output (Step {status.step})
              </div>
              <div className="rounded-lg border border-obsidian-700/70 bg-obsidian-950 p-3 font-mono text-xs leading-relaxed text-slate-300">
                <span className="text-slate-500">{status.default_prompt}</span>
                <span className="text-emerald-300">{status.latest_sample}</span>
              </div>
            </div>
          )}
        </div>

        {/* Right 4 cols: Corpus & Model Architecture Configurator */}
        <div className="glass-panel flex flex-col justify-between rounded-xl p-4 xl:col-span-4">
          <div className="space-y-4">
            <div className="flex items-center gap-2 border-b border-obsidian-700/80 pb-3">
              <Sliders className="h-4 w-4 text-cyan-400" />
              <div>
                <h2 className="text-sm font-semibold text-slate-100">
                  Experiment &amp; Architecture Lab
                </h2>
                <p className="text-[11px] text-slate-400">
                  Train a fresh BPE tokenizer + Transformer from scratch
                </p>
              </div>
            </div>

            {/* Dataset picker */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-300">
                Training Corpus
              </label>
              <div className="space-y-1.5">
                {datasets.map((ds) => {
                  const active = !useCustom && selectedDataset === ds.id;
                  return (
                    <button
                      key={ds.id}
                      onClick={() => {
                        setUseCustom(false);
                        setSelectedDataset(ds.id);
                      }}
                      className={`w-full rounded-lg border p-2.5 text-left transition ${
                        active
                          ? "border-emerald-500/60 bg-emerald-500/10"
                          : "border-obsidian-700/80 bg-obsidian-950/60 hover:border-slate-600"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-100">{ds.name}</span>
                        <span className="font-mono text-[10px] text-slate-400">
                          {(ds.chars / 1024).toFixed(0)} KB
                        </span>
                      </div>
                      <p className="mt-0.5 text-[11px] leading-snug text-slate-400">
                        {ds.description}
                      </p>
                    </button>
                  );
                })}
                <button
                  onClick={() => setUseCustom(true)}
                  className={`w-full rounded-lg border p-2 text-left text-xs font-medium transition ${
                    useCustom
                      ? "border-cyan-500/60 bg-cyan-500/10 text-cyan-200"
                      : "border-obsidian-700/80 bg-obsidian-950/60 text-slate-300 hover:border-slate-600"
                  }`}
                >
                  + Paste Custom Text Corpus
                </button>
              </div>

              {useCustom && (
                <textarea
                  rows={4}
                  value={customCorpus}
                  onChange={(e) => setCustomCorpus(e.target.value)}
                  placeholder="Paste any text or code corpus (>= 64 chars) to train BPE + Transformer from scratch..."
                  className="mt-2 w-full rounded-lg border border-obsidian-700 bg-obsidian-950 p-2.5 font-mono text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
                />
              )}
            </div>

            {/* Architecture Preset */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-300">
                Transformer Architecture (RoPE · RMSNorm · FlashAttn · SwiGLU)
              </label>
              <select
                value={selectedPreset}
                onChange={(e) => setSelectedPreset(e.target.value)}
                className="w-full rounded-lg border border-obsidian-700 bg-obsidian-950 px-3 py-2 font-mono text-xs text-slate-200 focus:border-emerald-500 focus:outline-none"
              >
                {presets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-[11px] text-slate-400">
                {presets.find((p) => p.id === selectedPreset)?.description}
              </p>
            </div>

            {/* Hyperparameters */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="mb-1 block font-mono text-[11px] text-slate-400">
                  Max Learning Rate
                </label>
                <input
                  type="number"
                  step={0.0005}
                  min={0.0001}
                  max={0.02}
                  value={lr}
                  onChange={(e) => setLr(Number(e.target.value))}
                  className="w-full rounded-lg border border-obsidian-700 bg-obsidian-950 px-2.5 py-1.5 font-mono text-xs text-slate-200 focus:border-emerald-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="mb-1 block font-mono text-[11px] text-slate-400">
                  Batch Size
                </label>
                <input
                  type="number"
                  min={2}
                  max={32}
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="w-full rounded-lg border border-obsidian-700 bg-obsidian-950 px-2.5 py-1.5 font-mono text-xs text-slate-200 focus:border-emerald-500 focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Action buttons */}
          <div className="mt-5 space-y-2 border-t border-obsidian-700/80 pt-4">
            <button
              disabled={resetting}
              onClick={() => handleReset(0)}
              className="flex w-full items-center justify-center gap-2 rounded-lg border border-cyan-500/50 bg-cyan-500/15 px-3 py-2 font-mono text-xs font-semibold text-cyan-200 transition hover:bg-cyan-500/25 disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${resetting ? "animate-spin" : ""}`} />
              {resetting ? "Rebuilding BPE & Model..." : "Reset & Train from Step 0"}
            </button>
            <button
              disabled={resetting}
              onClick={async () => {
                await handleReset(0);
                onStartTraining(240, lr);
              }}
              className="flex w-full items-center justify-center gap-2 rounded-lg border border-emerald-500/50 bg-emerald-500/20 px-3 py-2 font-mono text-xs font-semibold text-emerald-200 transition hover:bg-emerald-500/30 disabled:opacity-50"
            >
              <Play className="h-3.5 w-3.5 fill-current" />
              Rebuild &amp; Stream 240 Steps Live
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
