import React, { useCallback, useEffect, useState } from "react";
import { api, type RuntimeMode } from "./api";
import { AttentionTab } from "./components/AttentionTab";
import { ChatTab } from "./components/ChatTab";
import { EngineProofTab } from "./components/EngineProofTab";
import { GenerateBpeTab } from "./components/GenerateBpeTab";
import { Header, type TabId } from "./components/Header";
import { LogitLensTab } from "./components/LogitLensTab";
import { PatchingTab } from "./components/PatchingTab";
import { TrainingTab } from "./components/TrainingTab";
import type { DatasetMeta, LabStatus, ModelPreset, StepRecord } from "./types";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>("chat");
  const [runtimeMode, setRuntimeMode] = useState<RuntimeMode>("browser");
  const [status, setStatus] = useState<LabStatus | null>(null);
  const [history, setHistory] = useState<StepRecord[]>([]);
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [presets, setPresets] = useState<ModelPreset[]>([]);
  const [sseConnected, setSseConnected] = useState(false);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [interpRefreshKey, setInterpRefreshKey] = useState(0);
  const [promptOverride, setPromptOverride] = useState<string | null>(null);

  const refreshStatus = useCallback(async () => {
    try {
      const mode = await api.detectMode();
      setRuntimeMode(mode);
      const res = await api.getStatus();
      setStatus(res.status);
      setHistory(res.history);
      setDatasets(res.datasets);
      setPresets(res.presets);
      setGlobalError(null);
      return res;
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : String(e));
      return null;
    }
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  // Live SSE Telemetry subscription (only in local C++17 server mode — zero 404s on Vercel!)
  useEffect(() => {
    if (runtimeMode !== "server") return;
    const es = new EventSource("/api/events");
    es.onopen = () => setSseConnected(true);
    es.onerror = () => setSseConnected(false);

    es.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as {
          type: string;
          status?: LabStatus;
          data?: StepRecord;
          message?: string;
        };
        if (msg.type === "connected" && msg.status) {
          setStatus(msg.status);
          setSseConnected(true);
        } else if (msg.type === "step" && msg.data) {
          const rec = msg.data;
          setHistory((prev) => [...prev, rec].slice(-800));
          setStatus((prev) =>
            prev
              ? {
                  ...prev,
                  step: rec.step,
                  total_tokens_seen: rec.total_tokens,
                  last_loss: rec.loss,
                  last_val_loss: rec.val_loss ?? prev.last_val_loss,
                  last_perplexity: rec.perplexity,
                  last_grad_norm: rec.grad_norm,
                  grad_health: rec.grad_health,
                  last_step_ms: rec.step_ms,
                  tokens_per_sec: rec.tokens_per_sec,
                  live_gflops: rec.gflops,
                  latest_sample: rec.latest_sample || prev.latest_sample,
                  latest_layer_grads: rec.layer_grads.length
                    ? rec.layer_grads
                    : prev.latest_layer_grads,
                }
              : prev
          );
        } else if (msg.type === "training_started" && msg.status) {
          setStatus(msg.status);
        } else if (msg.type === "training_done" && msg.status) {
          setStatus(msg.status);
          setInterpRefreshKey((k) => k + 1);
        } else if (msg.type === "reset" && msg.status) {
          refreshStatus();
          setInterpRefreshKey((k) => k + 1);
        } else if (msg.type === "error" && msg.message) {
          setGlobalError(msg.message);
        }
      } catch {
        // ignore malformed frames
      }
    };

    return () => {
      es.close();
    };
  }, [runtimeMode, refreshStatus]);

  const handleStepOne = async () => {
    try {
      const res = await api.trainOneStep();
      setStatus(res.status);
      setHistory((prev) => [...prev, res.step_record].slice(-800));
      setInterpRefreshKey((k) => k + 1);
      setGlobalError(null);
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleStartTraining = async (steps: number, lr?: number) => {
    try {
      const res = await api.startTraining(steps, lr);
      setStatus(res.status);
      setGlobalError(null);
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleStopTraining = async () => {
    try {
      const res = await api.stopTraining();
      setStatus(res.status);
      setInterpRefreshKey((k) => k + 1);
      setGlobalError(null);
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleSwitchModel = async (modelId: "chat" | "induction") => {
    try {
      const res = await api.switchModel(modelId);
      setStatus(res.status);
      setHistory(res.history);
      setPromptOverride(null);
      setInterpRefreshKey((k) => k + 1);
      setGlobalError(null);
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleResetSession = async (payload: {
    dataset_id: string;
    preset_id: string;
    custom_text?: string;
    lr: number;
    batch_size: number;
    warm_steps?: number;
  }) => {
    try {
      const res = await api.resetSession(payload);
      setStatus(res.status);
      setHistory(res.history);
      setPromptOverride(null);
      setInterpRefreshKey((k) => k + 1);
      setGlobalError(null);
    } catch (e: unknown) {
      setGlobalError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleJumpToTab = (tab: TabId, prompt?: string) => {
    if (prompt) setPromptOverride(prompt);
    setActiveTab(tab);
  };

  const activeDatasetMeta = datasets.find((d) => d.id === status?.dataset_id);
  const defaultPrompt =
    promptOverride ||
    status?.default_prompt ||
    activeDatasetMeta?.default_prompt ||
    "User: Who are you?\nGlassbox:";

  return (
    <div className="min-h-screen bg-obsidian-950 sci-grid text-slate-100">
      <Header
        status={status}
        activeTab={activeTab}
        runtimeMode={runtimeMode}
        onSelectTab={setActiveTab}
        onStartTraining={(steps) => handleStartTraining(steps)}
        onStopTraining={handleStopTraining}
        onStepOne={handleStepOne}
        onSwitchModel={handleSwitchModel}
        sseConnected={sseConnected}
      />

      <main className="mx-auto max-w-[1600px] px-4 py-5 sm:px-6">
        {globalError && (
          <div className="mb-4 flex items-center justify-between rounded-xl border border-amber-500/40 bg-amber-950/30 px-4 py-2.5 font-mono text-xs text-amber-200">
            <span>{globalError}</span>
            <button
              onClick={() => setGlobalError(null)}
              className="rounded border border-amber-500/40 px-2 py-0.5 hover:bg-amber-500/20"
            >
              Dismiss
            </button>
          </div>
        )}

        {!status ? (
          <div className="glass-panel flex h-80 flex-col items-center justify-center gap-3 rounded-2xl">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-emerald-400 border-t-transparent" />
            <div className="font-mono text-sm text-slate-300">
              Loading Glassbox Transformer Weights &amp; BPE Vocabulary...
            </div>
          </div>
        ) : (
          <>
            {activeTab === "chat" && (
              <ChatTab
                status={status}
                runtimeMode={runtimeMode}
                onSwitchModel={handleSwitchModel}
                onJumpToTab={handleJumpToTab}
              />
            )}
            {activeTab === "training" && (
              <TrainingTab
                status={status}
                history={history}
                datasets={datasets}
                presets={presets}
                onStartTraining={handleStartTraining}
                onStopTraining={handleStopTraining}
                onResetSession={handleResetSession}
              />
            )}
            {activeTab === "attention" && (
              <AttentionTab
                defaultPrompt={defaultPrompt}
                datasetMeta={activeDatasetMeta}
                modelStep={interpRefreshKey}
              />
            )}
            {activeTab === "logit-lens" && (
              <LogitLensTab
                defaultPrompt={defaultPrompt}
                datasetMeta={activeDatasetMeta}
                modelStep={interpRefreshKey}
              />
            )}
            {activeTab === "patching" && (
              <PatchingTab
                datasetMeta={activeDatasetMeta}
                modelStep={interpRefreshKey}
              />
            )}
            {activeTab === "generate" && (
              <GenerateBpeTab
                defaultPrompt={defaultPrompt}
                datasetMeta={activeDatasetMeta}
                modelStep={interpRefreshKey}
              />
            )}
            {activeTab === "engine" && <EngineProofTab />}
          </>
        )}
      </main>
    </div>
  );
};

export default App;
