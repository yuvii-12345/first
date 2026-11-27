import React, { useEffect, useRef, useState } from "react";
import {
  Activity,
  Bot,
  Eye,
  Layers,
  MessageSquare,
  Power,
  RotateCcw,
  Send,
  Sliders,
  Sparkles,
  Square,
  Trash2,
  User,
  Zap,
} from "lucide-react";
import { api, type ChatMessage } from "../api";
import type {
  AttentionAnalysis,
  GeneratedTokenStep,
  LabStatus,
  LogitLensAnalysis,
} from "../types";
import { roleBadgeClass } from "../utils/colors";
import type { TabId } from "./Header";

interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  content: string;
  promptUsed?: string;
  tokens?: GeneratedTokenStep[];
  ablatedHeads?: [number, number][];
  streaming?: boolean;
  ms?: number;
}

interface ChatTabProps {
  status: LabStatus;
  runtimeMode: "server" | "browser";
  onSwitchModel: (modelId: "chat" | "induction") => Promise<void>;
  onJumpToTab: (tab: TabId, prompt?: string) => void;
}

const CHAT_STARTERS = [
  { label: "Who are you?", tag: "Identity" },
  { label: "How does FlashAttention work?", tag: "Attention" },
  { label: "What is RoPE?", tag: "Embeddings" },
  { label: "What is Logit Lens?", tag: "Interp" },
  { label: "What is Direct Logit Attribution?", tag: "Interp" },
  { label: "Write a fibonacci function", tag: "Python" },
  { label: "Write a haiku", tag: "Creative" },
  { label: "Tell me a joke", tag: "Fun" },
];

const INDUCTION_STARTERS = [
  {
    label: "Echo alpha crimson beta cobalt gamma amber delta emerald alpha crimson beta",
    tag: "Copy → cobalt",
  },
  {
    label: "Echo omega silver sigma golden kappa scarlet theta violet omega silver",
    tag: "Copy → sigma",
  },
  {
    label: "Echo cobalt amber emerald crimson violet silver golden scarlet cobalt amber emerald",
    tag: "Copy → crimson",
  },
];

export const ChatTab: React.FC<ChatTabProps> = ({
  status,
  runtimeMode,
  onSwitchModel,
  onJumpToTab,
}) => {
  const isChatModel = status.dataset_id === "chat";
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const stopRef = useRef(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Sampling & X-ray controls
  const [showSettings, setShowSettings] = useState(false);
  const [temperature, setTemperature] = useState(0.25);
  const [topK, setTopK] = useState(12);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(110);
  const [xrayColorTokens, setXrayColorTokens] = useState(true);
  const [ablatedHeads, setAblatedHeads] = useState<[number, number][]>([]);
  const [switchingModel, setSwitchingModel] = useState(false);

  // Live Neural X-Ray state for the inspected assistant turn
  const [inspectedTurnId, setInspectedTurnId] = useState<string | null>(null);
  const [selectedTokStep, setSelectedTokStep] = useState<GeneratedTokenStep | null>(null);
  const [attnData, setAttnData] = useState<AttentionAnalysis | null>(null);
  const [lensData, setLensData] = useState<LogitLensAnalysis | null>(null);
  const [xrayHead, setXrayHead] = useState<[number, number]>([Math.min(1, status.config.n_layers - 1), 0]);
  const [xrayLoading, setXrayLoading] = useState(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  // Reset ablated heads & clear turns when switching model family
  useEffect(() => {
    setAblatedHeads([]);
    setXrayHead([Math.min(1, status.config.n_layers - 1), 0]);
  }, [status.dataset_id, status.config.n_layers]);

  const inspectedTurn =
    turns.find((t) => t.id === inspectedTurnId) ??
    [...turns].reverse().find((t) => t.role === "assistant") ??
    null;

  // Automatically run Attention & Logit Lens X-Ray whenever an assistant turn finishes or is clicked
  useEffect(() => {
    if (!inspectedTurn || inspectedTurn.streaming || !inspectedTurn.promptUsed) return;
    let cancelled = false;
    setXrayLoading(true);
    Promise.all([
      api.analyzeAttention({
        prompt: inspectedTurn.promptUsed,
        ablate_heads: ablatedHeads,
        ablation_mode: "zero",
      }),
      api.analyzeLogitLens({ prompt: inspectedTurn.promptUsed }),
    ])
      .then(([att, ll]) => {
        if (cancelled) return;
        setAttnData(att);
        setLensData(ll);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setXrayLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [inspectedTurn?.id, inspectedTurn?.streaming, inspectedTurn?.promptUsed, ablatedHeads]);

  const sendMessage = async (textOverride?: string, historyOverride?: ChatTurn[]) => {
    const content = (textOverride ?? input).trim();
    if (!content || isStreaming) return;

    const baseTurns = historyOverride ?? turns;
    const userTurn: ChatTurn = {
      id: `u-${Date.now()}`,
      role: "user",
      content,
    };
    const asstId = `a-${Date.now() + 1}`;
    const asstPlaceholder: ChatTurn = {
      id: asstId,
      role: "assistant",
      content: "",
      tokens: [],
      ablatedHeads: [...ablatedHeads],
      streaming: true,
    };

    const nextTurns = [...baseTurns, userTurn, asstPlaceholder];
    setTurns(nextTurns);
    setInspectedTurnId(asstId);
    setSelectedTokStep(null);
    if (!textOverride) setInput("");

    setIsStreaming(true);
    stopRef.current = false;
    const t0 = performance.now();

    const apiMessages: ChatMessage[] = [...baseTurns, userTurn].map((t) => ({
      role: t.role,
      content: t.content,
    }));

    try {
      const gen = api.chatStream({
        messages: apiMessages,
        max_new_tokens: isChatModel ? maxTokens : 28,
        temperature,
        top_k: topK,
        top_p: topP,
        ablate_heads: ablatedHeads,
        ablation_mode: "zero",
      });

      const collectedTokens: GeneratedTokenStep[] = [];
      let usedPrompt = "";
      let latestReply = "";

      while (true) {
        if (stopRef.current) break;
        const step = await gen.next();
        if (step.done) {
          usedPrompt = step.value.prompt;
          latestReply = step.value.reply;
          break;
        }
        usedPrompt = step.value.prompt;
        latestReply = step.value.partialReply;
        collectedTokens.push(step.value.token);
        const snapTokens = [...collectedTokens];

        setTurns((prev) =>
          prev.map((t) =>
            t.id === asstId
              ? { ...t, content: latestReply, promptUsed: usedPrompt, tokens: snapTokens }
              : t
          )
        );
      }

      const elapsed = Math.round(performance.now() - t0);
      setTurns((prev) =>
        prev.map((t) =>
          t.id === asstId
            ? {
                ...t,
                content: latestReply || "(empty reply)",
                promptUsed: usedPrompt,
                tokens: collectedTokens,
                streaming: false,
                ms: elapsed,
              }
            : t
        )
      );
      if (collectedTokens.length > 0) {
        setSelectedTokStep(collectedTokens[0]);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setTurns((prev) =>
        prev.map((t) =>
          t.id === asstId
            ? { ...t, content: `Engine error: ${msg}`, streaming: false }
            : t
        )
      );
    } finally {
      setIsStreaming(false);
    }
  };

  const regenerateLast = () => {
    if (isStreaming || turns.length < 2) return;
    const lastUserIdx = [...turns].map((t) => t.role).lastIndexOf("user");
    if (lastUserIdx === -1) return;
    const lastUserMsg = turns[lastUserIdx].content;
    const before = turns.slice(0, lastUserIdx);
    sendMessage(lastUserMsg, before);
  };

  const toggleAblate = (l: number, h: number) => {
    setAblatedHeads((prev) => {
      const exists = prev.some(([al, ah]) => al === l && ah === h);
      return exists
        ? prev.filter(([al, ah]) => !(al === l && ah === h))
        : [...prev, [l, h]];
    });
  };

  const handleModelSwitch = async (target: "chat" | "induction") => {
    setSwitchingModel(true);
    try {
      await onSwitchModel(target);
      setTurns([]);
      setAttnData(null);
      setLensData(null);
      setSelectedTokStep(null);
    } finally {
      setSwitchingModel(false);
    }
  };

  const activeHeadObj = attnData?.layers[xrayHead[0]]?.heads[xrayHead[1]] ?? null;
  const lastQueryRow =
    activeHeadObj && activeHeadObj.matrix.length > 0
      ? activeHeadObj.matrix[activeHeadObj.matrix.length - 1]
      : [];

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
      {/* Main Chat Column (8 cols) */}
      <div className="glass-panel flex h-[calc(100vh-148px)] min-h-[580px] flex-col overflow-hidden rounded-2xl xl:col-span-8">
        {/* Chat Top Bar */}
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-obsidian-700/80 bg-obsidian-950/60 px-4 py-3">
          <div className="flex items-center gap-2">
            <div className="flex rounded-lg border border-obsidian-700 bg-obsidian-900 p-0.5 font-mono text-[11px]">
              <button
                onClick={() => handleModelSwitch("chat")}
                disabled={switchingModel || isStreaming}
                className={`rounded-md px-2.5 py-1 transition ${
                  isChatModel
                    ? "bg-emerald-500/25 font-semibold text-emerald-200"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Glassbox-Chat (4L)
              </button>
              <button
                onClick={() => handleModelSwitch("induction")}
                disabled={switchingModel || isStreaming}
                className={`rounded-md px-2.5 py-1 transition ${
                  !isChatModel
                    ? "bg-cyan-500/25 font-semibold text-cyan-200"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Induction Circuit (3L)
              </button>
            </div>
            <span className="hidden rounded border border-obsidian-700 bg-obsidian-900/90 px-2 py-1 font-mono text-[10px] text-slate-400 sm:inline">
              {runtimeMode === "server" ? "C++17 Native Server" : "In-Browser Float32 Engine"}
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setXrayColorTokens((v) => !v)}
              title="Color tokens by model probability & make every token clickable"
              className={`inline-flex items-center gap-1 rounded-lg border px-2.5 py-1 font-mono text-[11px] transition ${
                xrayColorTokens
                  ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-200"
                  : "border-obsidian-700 bg-obsidian-900 text-slate-400 hover:text-slate-200"
              }`}
            >
              <Sparkles className="h-3 w-3" />
              X-Ray Tokens
            </button>

            <button
              onClick={() => setShowSettings((v) => !v)}
              className={`inline-flex items-center gap-1 rounded-lg border px-2.5 py-1 font-mono text-[11px] transition ${
                showSettings
                  ? "border-cyan-500/40 bg-cyan-500/15 text-cyan-200"
                  : "border-obsidian-700 bg-obsidian-900 text-slate-300 hover:border-slate-600"
              }`}
            >
              <Sliders className="h-3 w-3" />
              T={temperature.toFixed(2)}
            </button>

            {turns.length > 0 && (
              <button
                onClick={() => {
                  setTurns([]);
                  setAttnData(null);
                  setLensData(null);
                  setSelectedTokStep(null);
                }}
                disabled={isStreaming}
                title="Clear conversation"
                className="rounded-lg border border-obsidian-700 bg-obsidian-900 p-1.5 text-slate-400 transition hover:border-rose-500/40 hover:text-rose-300 disabled:opacity-40"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* Collapsible Sampling Sliders & Active Ablation Banner */}
        {(showSettings || ablatedHeads.length > 0) && (
          <div className="space-y-2 border-b border-obsidian-700/70 bg-obsidian-950/80 px-4 py-2.5">
            {showSettings && (
              <div className="grid grid-cols-2 gap-3 font-mono text-[11px] sm:grid-cols-4">
                <div>
                  <div className="flex justify-between text-slate-400">
                    <span>Temp</span>
                    <span className="text-emerald-300">{temperature.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={1.2}
                    step={0.05}
                    value={temperature}
                    onChange={(e) => setTemperature(Number(e.target.value))}
                    className="w-full accent-emerald-400"
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
                    max={40}
                    step={1}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="w-full accent-cyan-400"
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
                    step={0.05}
                    value={topP}
                    onChange={(e) => setTopP(Number(e.target.value))}
                    className="w-full accent-violet-400"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-slate-400">
                    <span>Max Toks</span>
                    <span className="text-amber-300">{maxTokens}</span>
                  </div>
                  <input
                    type="range"
                    min={24}
                    max={130}
                    step={8}
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(Number(e.target.value))}
                    className="w-full accent-amber-400"
                  />
                </div>
              </div>
            )}

            {ablatedHeads.length > 0 && (
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-rose-500/40 bg-rose-950/30 px-3 py-1.5 font-mono text-[11px] text-rose-200">
                <span>
                  <Zap className="mr-1 inline h-3.5 w-3.5 text-rose-400" />
                  Live Head Ablation Active:{" "}
                  <strong>{ablatedHeads.map(([l, h]) => `L${l}.H${h}`).join(", ")}</strong> zeroed
                  out during generation!
                </span>
                <div className="flex items-center gap-2">
                  {turns.length >= 2 && (
                    <button
                      onClick={regenerateLast}
                      disabled={isStreaming}
                      className="rounded border border-amber-500/40 bg-amber-500/15 px-2 py-0.5 text-amber-200 hover:bg-amber-500/25"
                    >
                      Regenerate with Ablation
                    </button>
                  )}
                  <button
                    onClick={() => setAblatedHeads([])}
                    className="rounded border border-rose-500/40 px-2 py-0.5 hover:bg-rose-500/20"
                  >
                    Restore All
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Conversation Feed */}
        <div className="flex-1 overflow-y-auto px-4 py-5 space-y-4">
          {turns.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-emerald-500/40 bg-emerald-500/10 shadow-[0_0_35px_rgba(16,185,129,0.2)]">
                <MessageSquare className="h-7 w-7 text-emerald-400" />
              </div>
              <h2 className="font-mono text-xl font-bold tracking-tight text-slate-100">
                Chat with {isChatModel ? "Glassbox-Chat (4L)" : "Glassbox-Induction (3L)"}
              </h2>
              <p className="mt-1.5 max-w-lg text-xs leading-relaxed text-slate-400">
                {isChatModel
                  ? `A real ${(status.num_parameters / 1000).toFixed(0)}K-parameter GPT Transformer trained from scratch with zero PyTorch. Every reply streams token-by-token while exposing its live attention heads, token probabilities, and Logit Lens.`
                  : "Our 3-layer Induction Circuit Transformer (97.9% recall, 100% induction heads). Send an Echo sequence to watch its induction heads copy tokens in real time!"}
              </p>

              <div className="mt-6 grid w-full max-w-2xl grid-cols-1 gap-2 sm:grid-cols-2">
                {(isChatModel ? CHAT_STARTERS : INDUCTION_STARTERS).map((st, idx) => (
                  <button
                    key={idx}
                    onClick={() => sendMessage(st.label)}
                    className="group flex items-center justify-between gap-2 rounded-xl border border-obsidian-700/80 bg-obsidian-950/75 px-3.5 py-2.5 text-left transition hover:border-emerald-500/50 hover:bg-emerald-500/5"
                  >
                    <span className="truncate font-mono text-xs text-slate-200 group-hover:text-emerald-200">
                      {st.label}
                    </span>
                    <span className="shrink-0 rounded border border-obsidian-700 bg-obsidian-900 px-1.5 py-0.5 font-mono text-[9px] uppercase text-slate-400 group-hover:border-emerald-500/40 group-hover:text-emerald-300">
                      {st.tag}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            turns.map((turn, idx) => {
              const isUser = turn.role === "user";
              const isInspected = inspectedTurn?.id === turn.id;
              return (
                <div
                  key={turn.id}
                  className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}
                >
                  {!isUser && (
                    <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-emerald-500/40 bg-emerald-500/15">
                      <Bot className="h-4 w-4 text-emerald-300" />
                    </div>
                  )}

                  <div
                    onClick={() => {
                      if (!isUser) setInspectedTurnId(turn.id);
                    }}
                    className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                      isUser
                        ? "border border-cyan-500/30 bg-cyan-500/10 text-slate-100"
                        : `cursor-pointer border transition ${
                            isInspected
                              ? "border-emerald-500/50 bg-obsidian-900/95 shadow-[0_0_24px_rgba(16,185,129,0.08)]"
                              : "border-obsidian-700/80 bg-obsidian-950/80 hover:border-slate-600"
                          }`
                    }`}
                  >
                    {isUser ? (
                      <div className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-slate-100">
                        {turn.content}
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {/* Render either clickable X-Ray confidence tokens or clean prose */}
                        {xrayColorTokens && turn.tokens && turn.tokens.length > 0 ? (
                          <div className="whitespace-pre-wrap font-mono text-xs leading-relaxed">
                            {turn.tokens.map((tok, ti) => {
                              const isSel =
                                isInspected && selectedTokStep?.step === tok.step && selectedTokStep?.id === tok.id;
                              const tint =
                                tok.prob >= 0.75
                                  ? "bg-emerald-500/15 text-emerald-100 hover:bg-emerald-500/30"
                                  : tok.prob >= 0.35
                                    ? "bg-cyan-500/15 text-cyan-100 hover:bg-cyan-500/30"
                                    : "bg-amber-500/20 text-amber-100 hover:bg-amber-500/35";
                              return (
                                <span
                                  key={ti}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setInspectedTurnId(turn.id);
                                    setSelectedTokStep(tok);
                                  }}
                                  title={`p=${(tok.prob * 100).toFixed(1)}% · H=${tok.entropy.toFixed(2)} nats (click to inspect)`}
                                  className={`inline cursor-pointer rounded px-[1.5px] py-[0.5px] transition ${tint} ${
                                    isSel ? "ring-1 ring-white" : ""
                                  }`}
                                >
                                  {tok.text}
                                </span>
                              );
                            })}
                            {turn.streaming && (
                              <span className="ml-0.5 inline-block h-3.5 w-2 animate-pulse bg-emerald-400 align-middle" />
                            )}
                          </div>
                        ) : (
                          <div className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-slate-100">
                            {turn.content}
                            {turn.streaming && (
                              <span className="ml-1 inline-block h-3.5 w-2 animate-pulse bg-emerald-400 align-middle" />
                            )}
                          </div>
                        )}

                        {/* Turn footer telemetry */}
                        {!turn.streaming && (
                          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-obsidian-800/80 pt-2 font-mono text-[10px] text-slate-500">
                            <div className="flex items-center gap-2 tabular-nums">
                              <span>{turn.tokens?.length ?? 0} tokens</span>
                              {turn.ms != null && (
                                <>
                                  <span>·</span>
                                  <span>{turn.ms}ms</span>
                                </>
                              )}
                              {turn.ablatedHeads && turn.ablatedHeads.length > 0 && (
                                <span className="rounded bg-rose-500/20 px-1.5 py-0.5 text-rose-300">
                                  Ablated: {turn.ablatedHeads.map(([l, h]) => `L${l}.H${h}`).join(",")}
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-1.5">
                              {idx === turns.length - 1 && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    regenerateLast();
                                  }}
                                  className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-slate-400 hover:bg-obsidian-800 hover:text-slate-200"
                                >
                                  <RotateCcw className="h-3 w-3" />
                                  Retry
                                </button>
                              )}
                              <span className={isInspected ? "text-emerald-400" : "text-slate-500"}>
                                {isInspected ? "● X-Ray Active" : "Click to X-Ray"}
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {isUser && (
                    <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-cyan-500/40 bg-cyan-500/15">
                      <User className="h-4 w-4 text-cyan-300" />
                    </div>
                  )}
                </div>
              );
            })
          )}
          <div ref={bottomRef} />
        </div>

        {/* Composer Input Bar */}
        <div className="border-t border-obsidian-700/80 bg-obsidian-950/75 p-3">
          <div className="flex items-end gap-2">
            <textarea
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder={
                isChatModel
                  ? "Ask Glassbox anything — e.g. 'How does FlashAttention work?', 'Write a haiku', 'Tell me a joke'..."
                  : "Enter an Echo prompt — e.g. 'Echo alpha crimson beta cobalt gamma amber delta emerald alpha crimson beta'"
              }
              className="max-h-28 min-h-[42px] flex-1 resize-none rounded-xl border border-obsidian-700 bg-obsidian-900 px-3.5 py-2.5 font-mono text-xs text-slate-100 placeholder-slate-500 focus:border-emerald-500 focus:outline-none"
            />
            {isStreaming ? (
              <button
                onClick={() => {
                  stopRef.current = true;
                }}
                className="inline-flex h-[42px] items-center gap-1.5 rounded-xl border border-rose-500/50 bg-rose-500/20 px-4 font-mono text-xs font-semibold text-rose-200 hover:bg-rose-500/30"
              >
                <Square className="h-3.5 w-3.5 fill-current" />
                Stop
              </button>
            ) : (
              <button
                onClick={() => sendMessage()}
                disabled={!input.trim()}
                className="inline-flex h-[42px] items-center gap-1.5 rounded-xl border border-emerald-500/50 bg-emerald-500/20 px-4 font-mono text-xs font-semibold text-emerald-200 transition hover:bg-emerald-500/30 disabled:opacity-40"
              >
                <Send className="h-3.5 w-3.5" />
                Send
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Right Column (4 cols): Live Neural X-Ray Sidebar */}
      <div className="flex h-[calc(100vh-148px)] min-h-[580px] flex-col gap-4 overflow-y-auto xl:col-span-4">
        {/* 1. Token Probability Distribution */}
        <div className="glass-panel rounded-2xl p-4">
          <div className="mb-2.5 flex items-center justify-between">
            <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-200">
              <Activity className="h-3.5 w-3.5 text-emerald-400" />
              Token Probability X-Ray
            </h3>
            {selectedTokStep && (
              <span className="font-mono text-[10px] text-cyan-300 tabular-nums">
                H = {selectedTokStep.entropy.toFixed(2)} nats
              </span>
            )}
          </div>

          {selectedTokStep ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between rounded-lg bg-obsidian-950/80 px-2.5 py-1.5 font-mono text-xs">
                <span className="text-slate-400">
                  Step #{selectedTokStep.step + 1} Chosen:
                </span>
                <span className="rounded bg-emerald-500/20 px-2 py-0.5 font-bold text-emerald-200">
                  {selectedTokStep.display} ({(selectedTokStep.prob * 100).toFixed(1)}%)
                </span>
              </div>
              <div className="space-y-1.5">
                {selectedTokStep.top_candidates.slice(0, 6).map((c) => (
                  <div key={c.id} className="space-y-0.5 font-mono text-[11px]">
                    <div className="flex justify-between tabular-nums">
                      <span
                        className={
                          c.id === selectedTokStep.id
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
                          c.id === selectedTokStep.id ? "bg-emerald-400" : "bg-cyan-500/65"
                        }`}
                        style={{ width: `${Math.max(2, c.prob * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="font-mono text-[11px] leading-relaxed text-slate-500">
              Send a message and click any colored token in Glassbox&apos;s reply to inspect its exact
              next-token Softmax probabilities and Shannon entropy.
            </p>
          )}
        </div>

        {/* 2. Live Attention Spotlight + Interactive Head Ablation */}
        <div className="glass-panel rounded-2xl p-4">
          <div className="mb-2.5 flex items-center justify-between">
            <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-200">
              <Eye className="h-3.5 w-3.5 text-cyan-400" />
              Live Attention &amp; Head Knockout
            </h3>
            {xrayLoading && (
              <span className="font-mono text-[10px] text-cyan-300 animate-pulse">Tracing...</span>
            )}
          </div>

          {attnData && activeHeadObj ? (
            <div className="space-y-3">
              {/* Compact Head Selector Grid with Power/Ablation Toggle */}
              <div className="grid grid-cols-4 gap-1.5">
                {attnData.layers.flatMap((l) =>
                  l.heads.map((h) => {
                    const sel = xrayHead[0] === h.layer && xrayHead[1] === h.head;
                    const abl = ablatedHeads.some(([al, ah]) => al === h.layer && ah === h.head);
                    return (
                      <div
                        key={`${h.layer}-${h.head}`}
                        onClick={() => setXrayHead([h.layer, h.head])}
                        className={`flex cursor-pointer items-center justify-between rounded-md border px-1.5 py-1 font-mono text-[10px] transition ${
                          abl
                            ? "border-rose-500/60 bg-rose-950/40 text-rose-200"
                            : sel
                              ? "border-emerald-400/70 bg-emerald-500/15 text-emerald-200"
                              : "border-obsidian-700 bg-obsidian-950/70 text-slate-400 hover:border-slate-500"
                        }`}
                      >
                        <span>
                          L{h.layer}.H{h.head}
                        </span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleAblate(h.layer, h.head);
                          }}
                          title={abl ? "Restore head" : "Zero-ablate head in chat"}
                          className={`rounded p-0.5 ${
                            abl
                              ? "bg-rose-500/30 text-rose-100"
                              : "hover:bg-rose-500/20 hover:text-rose-300"
                          }`}
                        >
                          <Power className="h-2.5 w-2.5" />
                        </button>
                      </div>
                    );
                  })
                )}
              </div>

              <div className="flex items-center justify-between font-mono text-[10px]">
                <span className="text-slate-400">
                  Head <strong className="text-emerald-300">L{activeHeadObj.layer}.H{activeHeadObj.head}</strong>{" "}
                  attending from final prompt token:
                </span>
                <span
                  className={`rounded border px-1.5 py-0.5 text-[9px] ${roleBadgeClass(
                    activeHeadObj.role
                  )}`}
                >
                  {activeHeadObj.role}
                </span>
              </div>

              {/* Prompt Tokens Illuminated by Attention Weight */}
              <div className="flex flex-wrap gap-1 rounded-lg border border-obsidian-700/70 bg-obsidian-950 p-2 font-mono text-[11px]">
                {attnData.tokens.map((tok, i) => {
                  const w = lastQueryRow[i] ?? 0;
                  const alpha = Math.min(0.85, Math.pow(w, 0.55) * 0.9);
                  return (
                    <span
                      key={i}
                      title={`${tok.display}: ${(w * 100).toFixed(1)}% attention`}
                      style={{
                        backgroundColor:
                          w > 0.02 ? `rgba(16, 185, 129, ${alpha})` : "rgba(15, 23, 42, 0.7)",
                      }}
                      className="rounded border border-obsidian-800 px-1.5 py-0.5 text-slate-100"
                    >
                      {tok.display}
                      {w >= 0.08 && (
                        <span className="ml-1 text-[9px] text-emerald-100/90">
                          {(w * 100).toFixed(0)}%
                        </span>
                      )}
                    </span>
                  );
                })}
              </div>

              <button
                onClick={() => onJumpToTab("attention", inspectedTurn?.promptUsed)}
                className="w-full rounded-lg border border-obsidian-700 bg-obsidian-900 py-1.5 font-mono text-[11px] text-cyan-300 transition hover:border-cyan-500/40 hover:bg-cyan-500/10"
              >
                Open Full Attention &amp; Arc Explorer →
              </button>
            </div>
          ) : (
            <p className="font-mono text-[11px] leading-relaxed text-slate-500">
              Live attention weights and interactive per-head knockout switches appear here as soon as
              Glassbox answers.
            </p>
          )}
        </div>

        {/* 3. Layer-by-Layer Logit Lens Snapshot */}
        <div className="glass-panel rounded-2xl p-4">
          <div className="mb-2.5 flex items-center justify-between">
            <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-200">
              <Layers className="h-3.5 w-3.5 text-violet-400" />
              Logit Lens (Layer Emergence)
            </h3>
            {lensData && (
              <button
                onClick={() => onJumpToTab("logit-lens", inspectedTurn?.promptUsed)}
                className="font-mono text-[10px] text-emerald-300 hover:underline"
              >
                Full Lens →
              </button>
            )}
          </div>

          {lensData ? (
            <div className="space-y-1.5 font-mono text-[11px] tabular-nums">
              {lensData.stages.map((st) => {
                const cell = st.cells[st.cells.length - 1];
                if (!cell) return null;
                return (
                  <div
                    key={st.stage}
                    className="flex items-center justify-between rounded-lg bg-obsidian-950/75 px-2.5 py-1"
                  >
                    <span className="w-16 text-slate-500">{st.stage}</span>
                    <span className="flex-1 truncate font-semibold text-slate-100">
                      {cell.top1_display}
                    </span>
                    <div className="flex w-24 items-center gap-1.5">
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-obsidian-800">
                        <div
                          className="h-full rounded-full bg-emerald-400"
                          style={{ width: `${Math.max(3, cell.top1_prob * 100)}%` }}
                        />
                      </div>
                      <span className="w-9 text-right text-[10px] text-emerald-300">
                        {(cell.top1_prob * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="font-mono text-[11px] text-slate-500">
              Watch Glassbox&apos;s first predicted token crystallize from Embed → Final Layer.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};
