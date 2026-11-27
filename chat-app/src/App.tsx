import React, { useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Cpu,
  Eye,
  GraduationCap,
  Menu,
  SlidersHorizontal,
  Sparkles,
  Square,
} from "lucide-react";
import { ChatMessage } from "./components/ChatMessage";
import { Sidebar } from "./components/Sidebar";
import { TrainStudioModal } from "./components/TrainStudioModal";
import { XRayDrawer } from "./components/XRayDrawer";
import { HERO_SUGGESTIONS, LESSON_PACKS } from "./data/prompts";
import { liveEngine } from "./engine/glassboxLiveEngine";
import type {
  BasicsBundle,
  ChatMessageItem,
  Conversation,
  Proof100M,
  QAPair,
  TokenStep,
  XRaySnapshot,
} from "./types";

function makeConversation(): Conversation {
  return {
    id: crypto.randomUUID?.() ?? `chat-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    title: "New Chat",
    messages: [],
    createdAt: Date.now(),
  };
}

export function App() {
  const [bundle, setBundle] = useState<BasicsBundle | null>(null);
  const [proof100m, setProof100m] = useState<Proof100M | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [conversations, setConversations] = useState<Conversation[]>(() => [makeConversation()]);
  const [activeId, setActiveId] = useState<string>(() => "");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [temperature, setTemperature] = useState(0.15);
  const [showTempPopover, setShowTempPopover] = useState(false);
  const [xrayMode, setXrayMode] = useState(false);
  const [xrayOpen, setXrayOpen] = useState(false);
  const [selectedToken, setSelectedToken] = useState<TokenStep | null>(null);
  const [xraySnapshot, setXraySnapshot] = useState<XRaySnapshot | null>(null);

  const [trainOpen, setTrainOpen] = useState(false);
  const [customStepsTrained, setCustomStepsTrained] = useState(0);
  const [taughtSuggestions, setTaughtSuggestions] = useState<string[]>([]);
  const [customPairs, setCustomPairs] = useState<QAPair[]>(() =>
    LESSON_PACKS[0].pairs.map((p, idx) => ({
      id: `init-${idx}`,
      question: p.question,
      answer: p.answer,
    }))
  );

  const stopRef = useRef(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Ensure activeId always points to an existing conversation
  const activeConv =
    conversations.find((c) => c.id === activeId) ?? conversations[0];

  useEffect(() => {
    if (!activeId && conversations[0]) {
      setActiveId(conversations[0].id);
    }
  }, [activeId, conversations]);

  // Boot the Pure-TypeScript Float32 GQA Engine
  useEffect(() => {
    let mounted = true;
    liveEngine
      .init("/models/glassbox-basics.json")
      .then((b) => {
        if (mounted) setBundle(b);
      })
      .catch((err) => {
        if (mounted) setLoadError(err?.message ?? "Failed to load Glassbox neural weights.");
      });

    fetch("/models/glassbox-100m-proof.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (mounted && d) setProof100m(d);
      })
      .catch(() => {});

    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConv?.messages.length, isStreaming]);

  const updateActiveMessages = (updater: (prev: ChatMessageItem[]) => ChatMessageItem[], newTitle?: string) => {
    const targetId = activeConv.id;
    setConversations((prev) =>
      prev.map((c) =>
        c.id === targetId
          ? {
              ...c,
              title: newTitle && c.title === "New Chat" ? newTitle.slice(0, 36) : c.title,
              messages: updater(c.messages),
            }
          : c
      )
    );
  };

  const sendMessage = async (textOverride?: string) => {
    const text = (textOverride ?? input).trim();
    if (!text || isStreaming || !bundle) return;

    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    stopRef.current = false;

    const userMsg: ChatMessageItem = {
      id: `u-${Date.now()}`,
      role: "user",
      content: text,
      timestamp: Date.now(),
    };
    const asstId = `a-${Date.now() + 1}`;
    const asstPlaceholder: ChatMessageItem = {
      id: asstId,
      role: "assistant",
      content: "",
      tokens: [],
      timestamp: Date.now(),
    };

    const historyForModel = [
      ...activeConv.messages.map((m) => ({ role: m.role, content: m.content })),
      { role: "user" as const, content: text },
    ];

    updateActiveMessages((prev) => [...prev, userMsg, asstPlaceholder], text);
    setIsStreaming(true);

    try {
      const gen = liveEngine.chatStream({
        messages: historyForModel,
        temperature,
        topK: 12,
        topP: 0.9,
        maxNewTokens: 128,
      });

      const collected: TokenStep[] = [];
      let streamedText = "";

      while (true) {
        if (stopRef.current) break;
        const step = await gen.next();
        if (step.done) {
          const final = step.value;
          updateActiveMessages((prev) =>
            prev.map((m) =>
              m.id === asstId
                ? {
                    ...m,
                    content: final.reply || streamedText.trim(),
                    tokens: final.tokens,
                    promptUsed: final.promptUsed,
                    tokPerSec: final.tokPerSec,
                    latencyMs: final.latencyMs,
                  }
                : m
            )
          );
          break;
        } else {
          collected.push(step.value);
          streamedText += step.value.text;
          const clean = streamedText.split("<EOS>")[0].split("\nUser:")[0];
          const snapTokens = [...collected];
          updateActiveMessages((prev) =>
            prev.map((m) =>
              m.id === asstId ? { ...m, content: clean, tokens: snapTokens } : m
            )
          );
        }
      }
    } finally {
      setIsStreaming(false);
    }
  };

  const handleRegenerate = async () => {
    if (isStreaming || activeConv.messages.length < 2) return;
    const msgs = activeConv.messages;
    const lastUser = [...msgs].reverse().find((m) => m.role === "user");
    if (!lastUser) return;
    // Remove last assistant turn and re-send
    const trimmed = msgs.slice(0, msgs.length - 1);
    const targetId = activeConv.id;
    setConversations((prev) =>
      prev.map((c) => (c.id === targetId ? { ...c, messages: trimmed.slice(0, -1) } : c))
    );
    await sendMessage(lastUser.content);
  };

  const handleInspectMessage = (msg: ChatMessageItem) => {
    const inspectText = msg.promptUsed ? `${msg.promptUsed} ${msg.content}` : msg.content;
    const snap = liveEngine.inspectXRay(inspectText);
    setXraySnapshot(snap);
    if (msg.tokens && msg.tokens.length > 0 && !selectedToken) {
      setSelectedToken(msg.tokens[0]);
    }
    setXrayMode(true);
    setXrayOpen(true);
  };

  const handleSelectToken = (tok: TokenStep, msg: ChatMessageItem) => {
    setSelectedToken(tok);
    const inspectText = msg.promptUsed ? `${msg.promptUsed} ${msg.content}` : msg.content;
    setXraySnapshot(liveEngine.inspectXRay(inspectText));
    setXrayOpen(true);
  };

  const handleNewChat = () => {
    const c = makeConversation();
    setConversations((prev) => [c, ...prev]);
    setActiveId(c.id);
  };

  const handleDeleteConversation = (id: string) => {
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (next.length === 0) {
        const fresh = makeConversation();
        setActiveId(fresh.id);
        return [fresh];
      }
      if (id === activeId) setActiveId(next[0].id);
      return next;
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Loading state
  if (!bundle && !loadError) {
    return (
      <div className="flex h-screen flex-col items-center justify-center bg-[#090c11] p-6 text-center">
        <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-emerald-500/40 bg-emerald-500/15 shadow-[0_0_40px_rgba(16,185,129,0.25)]">
          <Sparkles className="h-7 w-7 animate-pulse text-emerald-400" />
        </div>
        <h1 className="font-mono text-lg font-bold text-slate-100">Loading Glassbox AI…</h1>
        <p className="mt-1 font-mono text-xs text-slate-400">
          Unpacking 1.18M Int8 GQA neural weights directly into your browser
        </p>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="flex h-screen flex-col items-center justify-center bg-[#090c11] p-6 text-center">
        <p className="font-mono text-sm text-rose-400">{loadError}</p>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[#090c11] text-slate-100">
      {/* Collapsible Left Sidebar */}
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        conversations={conversations}
        activeId={activeConv.id}
        onSelectConversation={setActiveId}
        onNewChat={handleNewChat}
        onDeleteConversation={handleDeleteConversation}
        onQuickPrompt={(p) => sendMessage(p)}
        onOpenTrainStudio={() => setTrainOpen(true)}
        bundle={bundle}
        customStepsTrained={customStepsTrained}
      />

      {/* Main Chat Column */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Clean Top Navigation Bar */}
        <header className="flex items-center justify-between gap-2 border-b border-slate-800/80 bg-[#0b0f16]/90 px-4 py-3 backdrop-blur-sm">
          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setSidebarOpen(true)}
              className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-100 lg:hidden"
              aria-label="Open menu"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div className="flex items-center gap-2">
              <span className="font-mono text-sm font-bold text-slate-100">Glassbox AI</span>
              <span className="hidden rounded-full border border-emerald-500/35 bg-emerald-500/12 px-2 py-0.5 font-mono text-[10px] text-emerald-300 sm:inline-flex items-center gap-1">
                <Cpu className="h-3 w-3" />
                1.2M 6L GQA · In-Browser
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Sampling Temp Popover */}
            <div className="relative">
              <button
                onClick={() => setShowTempPopover((v) => !v)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700/80 bg-slate-900/90 px-2.5 py-1.5 font-mono text-[11px] text-slate-300 hover:border-slate-600 hover:text-slate-100"
                title="Adjust response creativity (temperature)"
              >
                <SlidersHorizontal className="h-3.5 w-3.5 text-cyan-400" />
                <span className="hidden sm:inline">Temp</span> {temperature.toFixed(2)}
              </button>
              {showTempPopover && (
                <div className="absolute right-0 top-10 z-30 w-56 rounded-xl border border-slate-700 bg-[#0e131b] p-3 shadow-xl">
                  <div className="mb-1.5 flex justify-between font-mono text-[11px] text-slate-300">
                    <span>Temperature</span>
                    <span className="text-emerald-300">{temperature.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={0.9}
                    step={0.05}
                    value={temperature}
                    onChange={(e) => setTemperature(Number(e.target.value))}
                    className="w-full accent-emerald-400"
                  />
                  <div className="mt-2 flex justify-between gap-1 font-mono text-[10px]">
                    <button
                      onClick={() => setTemperature(0)}
                      className="flex-1 rounded bg-slate-800 py-1 text-slate-300 hover:bg-slate-700"
                    >
                      Exact (0.0)
                    </button>
                    <button
                      onClick={() => setTemperature(0.2)}
                      className="flex-1 rounded bg-slate-800 py-1 text-slate-300 hover:bg-slate-700"
                    >
                      Balanced
                    </button>
                    <button
                      onClick={() => setTemperature(0.65)}
                      className="flex-1 rounded bg-slate-800 py-1 text-slate-300 hover:bg-slate-700"
                    >
                      Creative
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* X-Ray Toggle */}
            <button
              onClick={() => {
                setXrayMode((v) => !v);
                if (!xrayOpen && activeConv.messages.some((m) => m.role === "assistant")) {
                  const lastAsst = [...activeConv.messages].reverse().find((m) => m.role === "assistant");
                  if (lastAsst) handleInspectMessage(lastAsst);
                }
              }}
              className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-mono text-[11px] transition ${
                xrayMode
                  ? "border-cyan-500/45 bg-cyan-500/15 text-cyan-200"
                  : "border-slate-700/80 bg-slate-900/90 text-slate-300 hover:text-slate-100"
              }`}
            >
              <Eye className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">X-Ray</span>
            </button>

            {/* Train & Teach CTA */}
            <button
              onClick={() => setTrainOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-500 px-3 py-1.5 font-mono text-xs font-semibold text-slate-950 shadow-[0_0_20px_rgba(16,185,129,0.28)] transition hover:bg-emerald-400"
            >
              <GraduationCap className="h-4 w-4" />
              Train & Teach
              {customStepsTrained > 0 && (
                <span className="rounded bg-slate-950/20 px-1 text-[10px]">+{customStepsTrained}</span>
              )}
            </button>
          </div>
        </header>

        {/* Scrollable Chat Feed */}
        <main className="flex-1 overflow-y-auto px-4 py-6">
          <div className="mx-auto max-w-3xl space-y-6">
            {activeConv.messages.length === 0 ? (
              <div className="flex flex-col items-center pt-6 pb-8 text-center sm:pt-12">
                <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-emerald-500/40 bg-emerald-500/12 shadow-[0_0_45px_rgba(16,185,129,0.2)]">
                  <Sparkles className="h-8 w-8 text-emerald-400" />
                </div>
                <h1 className="text-2xl font-bold tracking-tight text-slate-100 sm:text-3xl">
                  Chat with Glassbox AI
                </h1>
                <p className="mt-2 max-w-xl text-sm leading-relaxed text-slate-400">
                  A real <strong className="text-slate-200">1.18M-parameter 6-layer GQA neural network</strong>{" "}
                  trained from scratch with zero PyTorch on 106 everyday basics — running privately
                  right inside your browser. You can even{" "}
                  <button
                    onClick={() => setTrainOpen(true)}
                    className="font-medium text-emerald-400 underline decoration-emerald-500/50 underline-offset-4 hover:text-emerald-300"
                  >
                    teach & train it live
                  </button>{" "}
                  on your own Q&A!
                </p>

                {/* Newly taught quick-test chips if user just trained */}
                {taughtSuggestions.length > 0 && (
                  <div className="mt-5 flex flex-wrap items-center justify-center gap-2 rounded-xl border border-emerald-500/35 bg-emerald-500/10 px-4 py-2.5">
                    <span className="font-mono text-[11px] font-semibold text-emerald-300">
                      Test your newly trained facts:
                    </span>
                    {taughtSuggestions.map((q) => (
                      <button
                        key={q}
                        onClick={() => sendMessage(q)}
                        className="rounded-lg bg-emerald-500/20 px-2.5 py-1 font-mono text-xs text-emerald-100 hover:bg-emerald-500/35"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                )}

                {/* 6 Hero Starter Cards */}
                <div className="mt-7 grid w-full grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
                  {HERO_SUGGESTIONS.map((item) => (
                    <button
                      key={item.prompt}
                      onClick={() => sendMessage(item.prompt)}
                      className="group rounded-xl border border-slate-800 bg-[#0d1118] p-3.5 text-left transition hover:border-emerald-500/45 hover:bg-slate-900/90"
                    >
                      <div className="flex items-center justify-between">
                        <span className="rounded bg-slate-800/90 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
                          {item.badge}
                        </span>
                        <ArrowUp className="h-3.5 w-3.5 rotate-45 text-slate-600 transition group-hover:text-emerald-400" />
                      </div>
                      <div className="mt-2 text-xs font-medium text-slate-200">{item.title}</div>
                      <div className="mt-0.5 truncate font-mono text-[11px] text-slate-500">
                        “{item.prompt}”
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <>
                {taughtSuggestions.length > 0 && (
                  <div className="flex flex-wrap items-center gap-2 rounded-xl border border-emerald-500/30 bg-emerald-500/8 px-3.5 py-2 text-xs">
                    <span className="font-mono text-[11px] text-emerald-300">Newly trained:</span>
                    {taughtSuggestions.map((q) => (
                      <button
                        key={q}
                        onClick={() => sendMessage(q)}
                        disabled={isStreaming}
                        className="rounded-md bg-emerald-500/20 px-2 py-0.5 font-mono text-[11px] text-emerald-100 hover:bg-emerald-500/35"
                      >
                        Ask “{q}”
                      </button>
                    ))}
                  </div>
                )}

                {activeConv.messages.map((msg, idx) => (
                  <ChatMessage
                    key={msg.id}
                    message={msg}
                    isStreaming={isStreaming && idx === activeConv.messages.length - 1 && msg.role === "assistant"}
                    xrayMode={xrayMode}
                    selectedTokenIdx={selectedToken?.step ?? null}
                    onSelectToken={handleSelectToken}
                    onInspectMessage={handleInspectMessage}
                    onRegenerate={
                      idx === activeConv.messages.length - 1 && msg.role === "assistant"
                        ? handleRegenerate
                        : undefined
                    }
                  />
                ))}
              </>
            )}
            <div ref={bottomRef} />
          </div>
        </main>

        {/* Bottom Composer */}
        <footer className="border-t border-slate-800/80 bg-[#0b0f16]/90 px-4 py-3.5">
          <div className="mx-auto max-w-3xl">
            <div className="flex items-end gap-2 rounded-2xl border border-slate-700/80 bg-[#0f141d] px-3.5 py-2.5 shadow-lg focus-within:border-emerald-500/60">
              <textarea
                ref={textareaRef}
                rows={1}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  e.target.style.height = "auto";
                  e.target.style.height = `${Math.min(140, e.target.scrollHeight)}px`;
                }}
                onKeyDown={handleKeyDown}
                placeholder="Ask Glassbox anything — greetings, math, science, geography, Python, or your custom taught facts…"
                className="max-h-36 flex-1 resize-none bg-transparent py-1 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none"
              />
              {isStreaming ? (
                <button
                  onClick={() => {
                    stopRef.current = true;
                  }}
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-rose-500/20 text-rose-300 border border-rose-500/40 hover:bg-rose-500/30"
                  title="Stop generating"
                >
                  <Square className="h-4 w-4 fill-current" />
                </button>
              ) : (
                <button
                  onClick={() => sendMessage()}
                  disabled={!input.trim()}
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-emerald-500 text-slate-950 transition hover:bg-emerald-400 disabled:bg-slate-800 disabled:text-slate-600"
                  title="Send message"
                >
                  <ArrowUp className="h-4 w-4 stroke-[2.5]" />
                </button>
              )}
            </div>
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 px-1 font-mono text-[10px] text-slate-500">
              <span>
                100% local Float32 GQA Transformer (Zero PyTorch · Zero external LLM APIs)
              </span>
              <button
                onClick={() => setTrainOpen(true)}
                className="text-emerald-400/90 hover:text-emerald-300"
              >
                Want it to learn something new? Click Train & Teach →
              </button>
            </div>
          </div>
        </footer>
      </div>

      {/* Optional Slide-Over Neural X-Ray Drawer */}
      <XRayDrawer
        open={xrayOpen}
        onClose={() => setXrayOpen(false)}
        selectedToken={selectedToken}
        snapshot={xraySnapshot}
      />

      {/* Interactive Live In-Browser Training & Basics Studio Modal */}
      <TrainStudioModal
        open={trainOpen}
        onClose={() => setTrainOpen(false)}
        bundle={bundle}
        proof100m={proof100m}
        customPairs={customPairs}
        onChangeCustomPairs={setCustomPairs}
        customStepsTrained={customStepsTrained}
        onTrainingFinished={(qs) => {
          setCustomStepsTrained(liveEngine.customStepsTrained);
          setTaughtSuggestions((prev) => Array.from(new Set([...qs, ...prev])).slice(0, 6));
        }}
        onResetBase={() => {
          liveEngine.resetToBase();
          setCustomStepsTrained(0);
          setTaughtSuggestions([]);
        }}
      />
    </div>
  );
}

export default App;
