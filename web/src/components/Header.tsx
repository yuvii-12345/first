import React from "react";
import {
  Activity,
  Cpu,
  Eye,
  Flame,
  GitBranch,
  Layers,
  MessageSquare,
  Pause,
  Play,
  SkipForward,
  Sparkles,
  Terminal,
} from "lucide-react";
import type { LabStatus } from "../types";

export type TabId =
  | "chat"
  | "llm-100m"
  | "attention"
  | "logit-lens"
  | "patching"
  | "training"
  | "generate"
  | "engine";

interface HeaderProps {
  status: LabStatus | null;
  activeTab: TabId;
  runtimeMode: "server" | "browser";
  onSelectTab: (tab: TabId) => void;
  onStartTraining: (steps: number) => void;
  onStopTraining: () => void;
  onStepOne: () => void;
  onSwitchModel: (modelId: "chat" | "induction") => void;
  sseConnected: boolean;
}

const TABS: { id: TabId; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: "chat", label: "Chat & X-Ray", icon: MessageSquare },
  { id: "llm-100m", label: "100M LLM & CUDA", icon: Flame },
  { id: "attention", label: "Attention & Ablation", icon: Eye },
  { id: "logit-lens", label: "Logit Lens & DLA", icon: Layers },
  { id: "patching", label: "Causal Patching", icon: GitBranch },
  { id: "training", label: "Training & Gradients", icon: Activity },
  { id: "generate", label: "Generate & BPE", icon: Terminal },
  { id: "engine", label: "Engine Proof & DAG", icon: Cpu },
];

export const Header: React.FC<HeaderProps> = ({
  status,
  activeTab,
  runtimeMode,
  onSelectTab,
  onStartTraining,
  onStopTraining,
  onStepOne,
  onSwitchModel,
  sseConnected,
}) => {
  const cfg = status?.config;
  const paramStr = status
    ? status.num_parameters >= 1_000_000
      ? `${(status.num_parameters / 1_000_000).toFixed(2)}M`
      : `${(status.num_parameters / 1_000).toFixed(1)}K`
    : "—";

  const isChatModel = status?.dataset_id === "chat";

  return (
    <header className="sticky top-0 z-30 border-b border-obsidian-700/80 bg-obsidian-950/90 backdrop-blur-md">
      <div className="mx-auto max-w-[1600px] px-4 sm:px-6">
        {/* Top bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 py-3">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-emerald-500/40 bg-emerald-500/10 shadow-[0_0_20px_rgba(16,185,129,0.2)]">
              <Sparkles className="h-5 w-5 text-emerald-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-mono text-base font-bold tracking-wider text-slate-100">
                  GLASSBOX
                </h1>
                <span className="rounded border border-emerald-500/30 bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wider text-emerald-300">
                  Zero PyTorch · From Scratch
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Custom GPT Transformer Chat &amp; Mechanistic Interpretability Observatory
              </p>
            </div>
          </div>

          {/* Live Model Spec & Telemetry */}
          {status && cfg && (
            <div className="hidden items-center gap-2 lg:flex">
              <div className="flex items-center gap-3 rounded-lg border border-obsidian-700/80 bg-obsidian-900/90 px-3 py-1.5 font-mono text-xs tabular-nums">
                <span className="text-slate-400">
                  {cfg.n_layers}L ·{" "}
                  {cfg.n_kv_heads && cfg.n_kv_heads !== cfg.n_heads
                    ? `${cfg.n_heads}Q/${cfg.n_kv_heads}KV GQA`
                    : `${cfg.n_heads}H`}{" "}
                  · {cfg.d_model}d
                </span>
                <span className="text-obsidian-700">|</span>
                <span className="text-cyan-300">{paramStr} params</span>
                <span className="text-obsidian-700">|</span>
                <span className="text-slate-300">Step {status.step}</span>
                <span className="text-obsidian-700">|</span>
                <span className="text-emerald-300">
                  Loss {status.last_loss != null ? status.last_loss.toFixed(3) : "—"}
                </span>
              </div>

              <div
                title={
                  runtimeMode === "browser"
                    ? "Running real Float32 Transformer inference natively in your browser"
                    : sseConnected
                      ? "Live C++17 server SSE telemetry connected"
                      : "Reconnecting..."
                }
                className="flex items-center gap-1.5 rounded-md border border-obsidian-700/70 bg-obsidian-900 px-2 py-1 font-mono text-[11px]"
              >
                <span
                  className={`h-2 w-2 rounded-full ${
                    status.is_training
                      ? "animate-pulse bg-emerald-400 shadow-[0_0_8px_#34d399]"
                      : runtimeMode === "browser" || sseConnected
                        ? "bg-emerald-400"
                        : "bg-amber-500"
                  }`}
                />
                <span className="text-slate-300">
                  {status.is_training
                    ? "TRAINING"
                    : runtimeMode === "browser"
                      ? "BROWSER FLOAT32"
                      : "C++17 LIVE"}
                </span>
              </div>
            </div>
          )}

          {/* Global Controls: Training in server mode, Model switcher in browser mode */}
          <div className="flex items-center gap-2">
            {runtimeMode === "server" ? (
              status?.is_training ? (
                <button
                  onClick={onStopTraining}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-rose-500/50 bg-rose-500/15 px-3 py-1.5 font-mono text-xs font-semibold text-rose-200 transition hover:bg-rose-500/25"
                >
                  <Pause className="h-3.5 w-3.5" />
                  Stop ({status.step}/{status.target_steps})
                </button>
              ) : (
                <>
                  <button
                    onClick={onStepOne}
                    title="Run exactly 1 forward + backward + AdamW step"
                    className="inline-flex items-center gap-1 rounded-lg border border-obsidian-700 bg-obsidian-850 px-2.5 py-1.5 font-mono text-xs font-medium text-slate-200 transition hover:border-slate-500 hover:bg-obsidian-800"
                  >
                    <SkipForward className="h-3.5 w-3.5 text-cyan-400" />
                    +1 Step
                  </button>
                  <button
                    onClick={() => onStartTraining(120)}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/50 bg-emerald-500/20 px-3 py-1.5 font-mono text-xs font-semibold text-emerald-200 shadow-[0_0_16px_rgba(16,185,129,0.18)] transition hover:bg-emerald-500/30"
                  >
                    <Play className="h-3.5 w-3.5 fill-current" />
                    Train +120
                  </button>
                </>
              )
            ) : (
              <div className="flex items-center gap-1 rounded-lg border border-obsidian-700 bg-obsidian-900 p-0.5 font-mono text-[11px]">
                <button
                  onClick={() => onSwitchModel("chat")}
                  className={`rounded-md px-2.5 py-1 transition ${
                    isChatModel
                      ? "bg-emerald-500/25 font-semibold text-emerald-200"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Chat 4L
                </button>
                <button
                  onClick={() => onSwitchModel("induction")}
                  className={`rounded-md px-2.5 py-1 transition ${
                    !isChatModel
                      ? "bg-cyan-500/25 font-semibold text-cyan-200"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Induction 3L
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Navigation tabs */}
        <nav className="-mb-px flex items-center gap-1 overflow-x-auto">
          {TABS.map((t) => {
            const Icon = t.icon;
            const active = activeTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => onSelectTab(t.id)}
                className={`group relative flex shrink-0 items-center gap-2 border-b-2 px-3.5 py-2.5 text-xs font-medium transition ${
                  active
                    ? "border-emerald-400 text-emerald-300"
                    : "border-transparent text-slate-400 hover:border-slate-600 hover:text-slate-200"
                }`}
              >
                <Icon
                  className={`h-3.5 w-3.5 ${
                    active ? "text-emerald-400" : "text-slate-500 group-hover:text-slate-300"
                  }`}
                />
                {t.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
