import React, { useRef, useState } from "react";
import {
  Activity,
  Check,
  Copy,
  Flame,
  GraduationCap,
  Play,
  Plus,
  RotateCcw,
  Sparkles,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { LESSON_PACKS } from "../data/prompts";
import { liveEngine } from "../engine/glassboxLiveEngine";
import type { BasicsBundle, Proof100M, QAPair, TrainProgressStep } from "../types";

interface TrainStudioModalProps {
  open: boolean;
  onClose: () => void;
  bundle: BasicsBundle | null;
  proof100m: Proof100M | null;
  customPairs: QAPair[];
  onChangeCustomPairs: (pairs: QAPair[]) => void;
  onTrainingFinished: (taughtQuestions: string[]) => void;
  onResetBase: () => void;
  customStepsTrained: number;
}

function MiniLossChart({ points }: { points: { step: number; loss: number }[] }) {
  if (points.length < 2) {
    return (
      <div className="flex h-28 items-center justify-center rounded-xl border border-slate-800 bg-[#070a0f] font-mono text-xs text-slate-500">
        Live Float32 backprop loss curve will plot here as steps run…
      </div>
    );
  }
  const W = 480;
  const H = 112;
  const pad = 12;
  const maxL = Math.max(...points.map((p) => p.loss), 0.5);
  const minL = 0;
  const maxS = Math.max(...points.map((p) => p.step), 1);
  const coords = points.map((p) => {
    const x = pad + (p.step / maxS) * (W - 2 * pad);
    const y = H - pad - ((p.loss - minL) / (maxL - minL || 1)) * (H - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  return (
    <div className="rounded-xl border border-slate-800 bg-[#070a0f] p-2.5">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-28 w-full">
        <polyline
          fill="none"
          stroke="#10b981"
          strokeWidth="2.5"
          strokeLinejoin="round"
          strokeLinecap="round"
          points={coords.join(" ")}
        />
        {points.map((p, idx) => {
          if (idx !== 0 && idx !== points.length - 1) return null;
          const x = pad + (p.step / maxS) * (W - 2 * pad);
          const y = H - pad - ((p.loss - minL) / (maxL - minL || 1)) * (H - 2 * pad);
          return <circle key={idx} cx={x} cy={y} r="3.5" fill="#34d399" />;
        })}
      </svg>
    </div>
  );
}

export const TrainStudioModal: React.FC<TrainStudioModalProps> = ({
  open,
  onClose,
  bundle,
  proof100m,
  customPairs,
  onChangeCustomPairs,
  onTrainingFinished,
  onResetBase,
  customStepsTrained,
}) => {
  const [tab, setTab] = useState<"teach" | "pretrained">("teach");
  const [steps, setSteps] = useState(20);
  const [lr, setLr] = useState(0.0018);
  const [keepBasicsReplay, setKeepBasicsReplay] = useState(true);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState<TrainProgressStep[]>([]);
  const [justFinished, setJustFinished] = useState(false);
  const [copiedCmd, setCopiedCmd] = useState(false);
  const stopRef = useRef(false);

  if (!open) return null;

  const addPair = () => {
    onChangeCustomPairs([
      ...customPairs,
      { id: crypto.randomUUID?.() ?? String(Date.now()), question: "", answer: "" },
    ]);
  };

  const updatePair = (id: string, field: "question" | "answer", val: string) => {
    onChangeCustomPairs(customPairs.map((p) => (p.id === id ? { ...p, [field]: val } : p)));
  };

  const removePair = (id: string) => {
    onChangeCustomPairs(customPairs.filter((p) => p.id !== id));
  };

  const loadPack = (packId: string) => {
    const pack = LESSON_PACKS.find((p) => p.id === packId);
    if (!pack) return;
    onChangeCustomPairs(
      pack.pairs.map((p, i) => ({
        id: `${packId}-${i}-${Date.now()}`,
        question: p.question,
        answer: p.answer,
      }))
    );
    setJustFinished(false);
  };

  const handleStartTrain = async () => {
    const valid = customPairs.filter((p) => p.question.trim() && p.answer.trim());
    if (valid.length === 0 || isTraining) return;
    setIsTraining(true);
    setJustFinished(false);
    setProgress([]);
    stopRef.current = false;

    await liveEngine.trainOnPairs(valid, {
      steps,
      lr,
      keepBasicsReplay,
      shouldStop: () => stopRef.current,
      onStep: (st) => {
        setProgress((prev) => [...prev, st]);
      },
    });

    setIsTraining(false);
    setJustFinished(true);
    onTrainingFinished(valid.map((p) => p.question.trim()));
  };

  const latest = progress[progress.length - 1];
  const gpuCmd =
    "git clone https://github.com/yuvii-12345/first.git && cd first && git checkout arena/803a1d9c-first && bash cuda/launch_gpu.sh 100m";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-xs">
      <div className="flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-slate-700/80 bg-[#0d1118] shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-emerald-500/40 bg-emerald-500/15">
              <GraduationCap className="h-5 w-5 text-emerald-400" />
            </div>
            <div>
              <h2 className="font-mono text-sm font-bold text-slate-100">
                Train & Teach Glassbox Live
              </h2>
              <p className="text-xs text-slate-400">
                Real 6-layer GQA Float32 backpropagation + AdamW right inside your browser
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={isTraining}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200 disabled:opacity-40"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Sub-Tabs */}
        <div className="flex items-center gap-2 border-b border-slate-800 bg-[#0a0d13] px-5 py-2.5">
          <button
            onClick={() => setTab("teach")}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 font-mono text-xs transition ${
              tab === "teach"
                ? "bg-emerald-500/20 font-semibold text-emerald-200 border border-emerald-500/35"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Sparkles className="h-3.5 w-3.5" />
            Teach Live (In-Browser Backprop)
          </button>
          <button
            onClick={() => setTab("pretrained")}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 font-mono text-xs transition ${
              tab === "pretrained"
                ? "bg-cyan-500/20 font-semibold text-cyan-200 border border-cyan-500/35"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Flame className="h-3.5 w-3.5" />
            Pretrained Basics & 100M CUDA
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {tab === "teach" ? (
            <>
              {/* 1-Click Lesson Packs */}
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    1-Click Lesson Packs
                  </span>
                  {customStepsTrained > 0 && (
                    <button
                      onClick={() => {
                        onResetBase();
                        setProgress([]);
                        setJustFinished(false);
                      }}
                      disabled={isTraining}
                      className="inline-flex items-center gap-1 rounded-lg border border-slate-700 px-2.5 py-1 font-mono text-[11px] text-slate-300 hover:border-rose-500/40 hover:text-rose-300"
                    >
                      <RotateCcw className="h-3 w-3" />
                      Reset to Pretrained ({customStepsTrained} live steps)
                    </button>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {LESSON_PACKS.map((pack) => (
                    <button
                      key={pack.id}
                      onClick={() => loadPack(pack.id)}
                      disabled={isTraining}
                      className="rounded-xl border border-slate-800 bg-slate-900/70 p-2.5 text-left transition hover:border-emerald-500/45 hover:bg-slate-900 disabled:opacity-50"
                    >
                      <div className="font-mono text-xs font-semibold text-emerald-300">
                        {pack.name}
                      </div>
                      <div className="mt-0.5 line-clamp-2 text-[11px] leading-snug text-slate-400">
                        {pack.description}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Editable Q&A Pairs */}
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    Q&A Pairs to Train On ({customPairs.length})
                  </span>
                  <button
                    onClick={addPair}
                    disabled={isTraining}
                    className="inline-flex items-center gap-1 rounded-lg border border-emerald-500/35 bg-emerald-500/10 px-2.5 py-1 font-mono text-[11px] text-emerald-200 hover:bg-emerald-500/20 disabled:opacity-40"
                  >
                    <Plus className="h-3.5 w-3.5" />
                    Add Q&A Pair
                  </button>
                </div>

                {customPairs.map((pair, idx) => (
                  <div
                    key={pair.id}
                    className="rounded-xl border border-slate-800 bg-[#0a0d13] p-3 space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[10px] text-slate-500">Lesson #{idx + 1}</span>
                      {customPairs.length > 1 && (
                        <button
                          onClick={() => removePair(pair.id)}
                          disabled={isTraining}
                          className="text-slate-500 hover:text-rose-400"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      )}
                    </div>
                    <input
                      type="text"
                      value={pair.question}
                      disabled={isTraining}
                      onChange={(e) => updatePair(pair.id, "question", e.target.value)}
                      placeholder="User asks: e.g. What is my name?"
                      className="w-full rounded-lg border border-slate-700/80 bg-slate-950 px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:border-emerald-500/60 focus:outline-none"
                    />
                    <textarea
                      rows={2}
                      value={pair.answer}
                      disabled={isTraining}
                      onChange={(e) => updatePair(pair.id, "answer", e.target.value)}
                      placeholder="Glassbox answers: e.g. Your name is Yuvraj!"
                      className="w-full rounded-lg border border-slate-700/80 bg-slate-950 px-3 py-2 text-xs text-emerald-100 placeholder:text-slate-500 focus:border-emerald-500/60 focus:outline-none"
                    />
                  </div>
                ))}
              </div>

              {/* Controls */}
              <div className="grid grid-cols-1 gap-3 rounded-xl border border-slate-800 bg-[#0a0d13] p-3.5 sm:grid-cols-3">
                <div>
                  <div className="flex justify-between font-mono text-[11px] text-slate-400">
                    <span>Gradient Steps</span>
                    <span className="text-emerald-300">{steps}</span>
                  </div>
                  <input
                    type="range"
                    min={10}
                    max={40}
                    step={2}
                    disabled={isTraining}
                    value={steps}
                    onChange={(e) => setSteps(Number(e.target.value))}
                    className="mt-1 w-full accent-emerald-400"
                  />
                </div>
                <div>
                  <div className="flex justify-between font-mono text-[11px] text-slate-400">
                    <span>AdamW LR</span>
                    <span className="text-cyan-300">{lr.toFixed(4)}</span>
                  </div>
                  <input
                    type="range"
                    min={0.0006}
                    max={0.003}
                    step={0.0002}
                    disabled={isTraining}
                    value={lr}
                    onChange={(e) => setLr(Number(e.target.value))}
                    className="mt-1 w-full accent-cyan-400"
                  />
                </div>
                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={keepBasicsReplay}
                    disabled={isTraining}
                    onChange={(e) => setKeepBasicsReplay(e.target.checked)}
                    className="h-4 w-4 accent-emerald-500"
                  />
                  <span>Protect pretrained basics (Replay Anchor)</span>
                </label>
              </div>

              {/* Live Chart & Telemetry */}
              <div className="space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-xs">
                  <span className="flex items-center gap-1.5 text-slate-300">
                    <Activity className="h-3.5 w-3.5 text-emerald-400" />
                    Live Backpropagation Telemetry
                  </span>
                  {latest && (
                    <div className="flex flex-wrap items-center gap-3 text-[11px]">
                      <span className="text-slate-400">
                        Step <strong className="text-slate-100">{latest.step}/{latest.totalSteps}</strong>
                      </span>
                      <span className="text-slate-400">
                        Loss <strong className="text-emerald-300">{latest.loss.toFixed(4)}</strong>
                      </span>
                      <span className="text-slate-400">
                        ‖∇‖ <strong className="text-cyan-300">{latest.gradNorm}</strong>
                      </span>
                      <span className="text-slate-400">
                        <strong className="text-amber-300">{latest.tokensPerSec}</strong> tok/s
                      </span>
                    </div>
                  )}
                </div>
                <MiniLossChart points={progress.map((p) => ({ step: p.step, loss: p.loss }))} />
              </div>

              {justFinished && (
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-emerald-500/40 bg-emerald-500/10 px-4 py-3 text-xs text-emerald-200">
                  <span>
                    Training complete! Weights updated live in RAM (final loss{" "}
                    <strong className="font-mono">{latest?.loss.toFixed(4)}</strong>). Ask your
                    question in chat right now!
                  </span>
                  <button
                    onClick={onClose}
                    className="rounded-lg bg-emerald-500 px-3 py-1.5 font-mono text-xs font-semibold text-slate-950 hover:bg-emerald-400"
                  >
                    Test in Chat →
                  </button>
                </div>
              )}
            </>
          ) : (
            <div className="space-y-4">
              {/* Pretrained Basics Summary */}
              {bundle && (
                <div className="rounded-xl border border-slate-800 bg-[#0a0d13] p-4 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h3 className="font-mono text-xs font-bold text-emerald-300">
                        {bundle.model_name} (Pretrained from Scratch in C++17)
                      </h3>
                      <p className="text-xs text-slate-400">
                        Trained on {bundle.topics_count} foundational basics topics ({bundle.qa_variants_count} Q&A variants) covering greetings, math, science, geography, Python coding, jokes, poems & AI.
                      </p>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 font-mono text-xs">
                    <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
                      <div className="text-[10px] text-slate-500">Parameters</div>
                      <div className="font-bold text-slate-100">{bundle.num_parameters.toLocaleString()}</div>
                    </div>
                    <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
                      <div className="text-[10px] text-slate-500">Architecture</div>
                      <div className="font-bold text-emerald-300">
                        {bundle.config.n_layers}L · {bundle.config.n_heads}Q/{bundle.config.n_kv_heads}KV GQA
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
                      <div className="text-[10px] text-slate-500">C++17 Steps / Tokens</div>
                      <div className="font-bold text-cyan-300">
                        {bundle.step} / {(bundle.tokens_seen / 1000).toFixed(0)}K
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
                      <div className="text-[10px] text-slate-500">Train → Val Loss</div>
                      <div className="font-bold text-amber-300">
                        {bundle.final_loss.toFixed(3)} / {bundle.final_val_loss.toFixed(3)}
                      </div>
                    </div>
                  </div>
                  <MiniLossChart
                    points={bundle.history.map((h) => ({ step: h.step, loss: h.loss }))}
                  />
                </div>
              )}

              {/* Glassbox-100M Pure CUDA C++17 Scale-Up */}
              {proof100m && (
                <div className="rounded-xl border border-amber-500/30 bg-amber-950/15 p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-mono text-xs font-bold text-amber-300">
                        Scale Up to Glassbox-100M ({proof100m.exact_params.toLocaleString()} Params · Pure CUDA C++17)
                      </h3>
                      <p className="text-xs text-slate-300">
                        Same exact GQA + SwiGLU + RoPE + RMSNorm architecture scaled to 12L · 768d · 12Q/4KV · 32K BPE · 2048 ctx. Verified live 3-step loss drop{" "}
                        <span className="font-mono text-emerald-300">
                          {proof100m.live_100m_cpu_smoke_steps[0].loss} → {proof100m.post_step_python_verified_loss}
                        </span>
                        . Train on 100M–10B tokens on any NVIDIA GPU with zero PyTorch:
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center justify-between gap-2 rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 font-mono text-[11px] text-emerald-300">
                    <code className="overflow-x-auto">{gpuCmd}</code>
                    <button
                      onClick={() => {
                        navigator.clipboard?.writeText(gpuCmd);
                        setCopiedCmd(true);
                        setTimeout(() => setCopiedCmd(false), 1500);
                      }}
                      className="shrink-0 rounded-md border border-slate-700 px-2 py-1 text-[10px] text-slate-300 hover:border-emerald-500/50 hover:text-emerald-200"
                    >
                      {copiedCmd ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        {tab === "teach" && (
          <div className="flex items-center justify-between border-t border-slate-800 bg-[#0a0d13] px-5 py-3.5">
            <span className="font-mono text-[11px] text-slate-500">
              100% private — gradients run locally in your browser tab
            </span>
            <div className="flex items-center gap-2">
              {isTraining ? (
                <button
                  onClick={() => {
                    stopRef.current = true;
                  }}
                  className="inline-flex items-center gap-1.5 rounded-xl border border-rose-500/40 bg-rose-500/15 px-4 py-2 font-mono text-xs font-semibold text-rose-200 hover:bg-rose-500/25"
                >
                  <Square className="h-3.5 w-3.5" />
                  Stop
                </button>
              ) : (
                <button
                  onClick={handleStartTrain}
                  disabled={!customPairs.some((p) => p.question.trim() && p.answer.trim())}
                  className="inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-4 py-2 font-mono text-xs font-bold text-slate-950 shadow-[0_0_24px_rgba(16,185,129,0.3)] transition hover:bg-emerald-400 disabled:opacity-40"
                >
                  <Play className="h-3.5 w-3.5 fill-current" />
                  Train Live in Browser ({steps} steps)
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
