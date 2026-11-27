import React, { useState } from "react";
import {
  Cpu,
  GraduationCap,
  MessageSquare,
  Plus,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { STARTER_CATEGORIES } from "../data/prompts";
import type { BasicsBundle, Conversation } from "../types";

interface SidebarProps {
  open: boolean;
  onClose: () => void;
  conversations: Conversation[];
  activeId: string;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onDeleteConversation: (id: string) => void;
  onQuickPrompt: (prompt: string) => void;
  onOpenTrainStudio: () => void;
  bundle: BasicsBundle | null;
  customStepsTrained: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  open,
  onClose,
  conversations,
  activeId,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  onQuickPrompt,
  onOpenTrainStudio,
  bundle,
  customStepsTrained,
}) => {
  const [activeCategory, setActiveCategory] = useState(STARTER_CATEGORIES[0].id);
  const currentCat =
    STARTER_CATEGORIES.find((c) => c.id === activeCategory) ?? STARTER_CATEGORIES[0];

  return (
    <>
      {open && (
        <div
          onClick={onClose}
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-xs lg:hidden"
        />
      )}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-slate-800/80 bg-[#0c1017] transition-transform duration-200 lg:static lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Top Brand + New Chat */}
        <div className="flex items-center justify-between border-b border-slate-800/80 px-4 py-3.5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl border border-emerald-500/40 bg-emerald-500/15 shadow-[0_0_20px_rgba(16,185,129,0.2)]">
              <Sparkles className="h-4 w-4 text-emerald-400" />
            </div>
            <div>
              <div className="flex items-center gap-1.5 font-mono text-sm font-bold tracking-tight text-slate-100">
                Glassbox AI
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              </div>
              <p className="font-mono text-[10px] text-slate-400">Zero-PyTorch Neural Chat</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200 lg:hidden"
            aria-label="Close sidebar"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-3 space-y-2">
          <button
            onClick={() => {
              onNewChat();
              onClose();
            }}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-700/80 bg-slate-900/90 px-3 py-2.5 text-xs font-medium text-slate-100 transition hover:border-emerald-500/50 hover:bg-slate-800/90"
          >
            <Plus className="h-4 w-4 text-emerald-400" />
            New Chat
          </button>

          <button
            onClick={() => {
              onOpenTrainStudio();
              onClose();
            }}
            className="flex w-full items-center justify-between rounded-xl border border-emerald-500/35 bg-emerald-500/10 px-3 py-2 text-xs font-medium text-emerald-200 transition hover:bg-emerald-500/20"
          >
            <span className="flex items-center gap-2">
              <GraduationCap className="h-4 w-4 text-emerald-400" />
              Train & Teach Live
            </span>
            {customStepsTrained > 0 && (
              <span className="rounded-full bg-emerald-500/25 px-1.5 py-0.5 font-mono text-[10px] text-emerald-200">
                +{customStepsTrained}
              </span>
            )}
          </button>
        </div>

        {/* Conversations History */}
        <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-4">
          <div>
            <div className="mb-1.5 px-1 font-mono text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Chats
            </div>
            <div className="space-y-1">
              {conversations.map((conv) => {
                const active = conv.id === activeId;
                return (
                  <div
                    key={conv.id}
                    className={`group flex items-center justify-between rounded-lg px-2.5 py-2 text-xs transition ${
                      active
                        ? "bg-slate-800/90 text-slate-100"
                        : "text-slate-400 hover:bg-slate-900/80 hover:text-slate-200"
                    }`}
                  >
                    <button
                      onClick={() => {
                        onSelectConversation(conv.id);
                        onClose();
                      }}
                      className="flex min-w-0 flex-1 items-center gap-2 text-left"
                    >
                      <MessageSquare className="h-3.5 w-3.5 shrink-0 text-slate-500" />
                      <span className="truncate">{conv.title}</span>
                    </button>
                    {conversations.length > 1 && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteConversation(conv.id);
                        }}
                        className="ml-1 hidden rounded p-0.5 text-slate-500 hover:bg-slate-700 hover:text-rose-400 group-hover:block"
                        title="Delete chat"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Quick Basics Explorer */}
          <div className="border-t border-slate-800/80 pt-3">
            <div className="mb-2 px-1 font-mono text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Try Trained Basics
            </div>
            <div className="mb-2 flex flex-wrap gap-1">
              {STARTER_CATEGORIES.map((cat) => (
                <button
                  key={cat.id}
                  onClick={() => setActiveCategory(cat.id)}
                  className={`rounded-md px-2 py-1 font-mono text-[10px] transition ${
                    activeCategory === cat.id
                      ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/35"
                      : "bg-slate-900/80 text-slate-400 border border-slate-800 hover:text-slate-200"
                  }`}
                >
                  {cat.label}
                </button>
              ))}
            </div>
            <div className="space-y-1">
              {currentCat.prompts.map((p) => (
                <button
                  key={p}
                  onClick={() => {
                    onQuickPrompt(p);
                    onClose();
                  }}
                  className="block w-full truncate rounded-lg border border-transparent px-2.5 py-1.5 text-left text-xs text-slate-300 transition hover:border-slate-800 hover:bg-slate-900/90 hover:text-emerald-200"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom Compact Model Info */}
        <div className="border-t border-slate-800/80 bg-[#090c11]/70 p-3">
          <div className="flex items-center justify-between text-[11px] font-mono text-slate-300">
            <span className="flex items-center gap-1.5">
              <Cpu className="h-3.5 w-3.5 text-emerald-400" />
              Glassbox-Basics 1.2M
            </span>
            <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300">
              LIVE
            </span>
          </div>
          {bundle && (
            <div className="mt-1.5 grid grid-cols-2 gap-1 font-mono text-[10px] text-slate-500">
              <span>
                {bundle.config.n_layers}L · {bundle.config.n_heads}Q/{bundle.config.n_kv_heads}KV GQA
              </span>
              <span className="text-right">{(bundle.num_parameters / 1e6).toFixed(2)}M params</span>
              <span>{bundle.topics_count} basics topics</span>
              <span className="text-right">
                {customStepsTrained > 0 ? `+${customStepsTrained} live steps` : "O(1) KV-Cache"}
              </span>
            </div>
          )}
        </div>
      </aside>
    </>
  );
};
