import React, { useState } from "react";
import { Check, Copy, Eye, RefreshCw, Sparkles, User } from "lucide-react";
import type { ChatMessageItem, TokenStep } from "../types";

interface ChatMessageProps {
  message: ChatMessageItem;
  isStreaming?: boolean;
  xrayMode: boolean;
  selectedTokenIdx: number | null;
  onSelectToken: (tok: TokenStep, msg: ChatMessageItem) => void;
  onInspectMessage: (msg: ChatMessageItem) => void;
  onRegenerate?: () => void;
}

/** Lightweight markdown code-fence & inline-code renderer */
function renderFormattedContent(text: string): React.ReactNode {
  const parts = text.split(/(```[\s\S]*?```)/g);
  return parts.map((part, i) => {
    if (part.startsWith("```") && part.endsWith("```")) {
      const inner = part.slice(3, -3);
      const nl = inner.indexOf("\n");
      const lang = nl > -1 ? inner.slice(0, nl).trim() : "";
      const code = nl > -1 ? inner.slice(nl + 1) : inner;
      return (
        <div
          key={i}
          className="my-2.5 overflow-hidden rounded-xl border border-slate-700/80 bg-[#070a0f]"
        >
          {lang && (
            <div className="border-b border-slate-800 bg-slate-900/70 px-3 py-1 font-mono text-[10px] uppercase tracking-wider text-slate-400">
              {lang}
            </div>
          )}
          <pre className="overflow-x-auto p-3.5 font-mono text-xs leading-relaxed text-emerald-200">
            <code>{code}</code>
          </pre>
        </div>
      );
    }
    return (
      <span key={i} className="whitespace-pre-wrap">
        {part}
      </span>
    );
  });
}

function probToClass(prob: number, selected: boolean): string {
  const ring = selected ? "ring-1 ring-white/90" : "";
  if (prob >= 0.8) return `bg-emerald-500/20 text-emerald-100 hover:bg-emerald-500/30 ${ring}`;
  if (prob >= 0.45) return `bg-cyan-500/20 text-cyan-100 hover:bg-cyan-500/30 ${ring}`;
  if (prob >= 0.18) return `bg-amber-500/20 text-amber-100 hover:bg-amber-500/30 ${ring}`;
  return `bg-rose-500/25 text-rose-100 hover:bg-rose-500/35 ${ring}`;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  isStreaming,
  xrayMode,
  selectedTokenIdx,
  onSelectToken,
  onInspectMessage,
  onRegenerate,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard?.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="flex max-w-[82%] items-start gap-2.5">
          <div className="rounded-2xl rounded-tr-sm border border-emerald-500/25 bg-emerald-500/12 px-4 py-2.5 text-sm leading-relaxed text-slate-100 shadow-sm">
            {message.content}
          </div>
          <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-slate-700 bg-slate-800/90 text-slate-300">
            <User className="h-3.5 w-3.5" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="group flex items-start gap-3">
      <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-emerald-500/40 bg-emerald-500/12 text-emerald-400">
        <Sparkles className="h-4 w-4" />
      </div>

      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="flex items-center gap-2 font-mono text-[11px] text-slate-400">
          <span className="font-semibold text-slate-200">Glassbox AI</span>
          {message.tokPerSec && (
            <span className="rounded bg-slate-800/80 px-1.5 py-0.5 text-[10px] text-emerald-300">
              {message.tokPerSec} tok/s
            </span>
          )}
          {isStreaming && (
            <span className="text-[10px] text-emerald-400 animate-pulse">generating…</span>
          )}
        </div>

        <div className="text-sm leading-relaxed text-slate-200">
          {xrayMode && message.tokens && message.tokens.length > 0 ? (
            <div className="flex flex-wrap items-baseline gap-y-1 leading-7">
              {message.tokens.map((tok) => (
                <button
                  key={tok.step}
                  onClick={() => onSelectToken(tok, message)}
                  title={`Token: "${tok.display}" | Prob: ${(tok.prob * 100).toFixed(1)}% | Entropy: ${tok.entropy} nats`}
                  className={`rounded px-0.5 font-mono text-[13px] transition cursor-pointer ${probToClass(
                    tok.prob,
                    selectedTokenIdx === tok.step
                  )}`}
                >
                  {tok.text === "\n" ? "↵\n" : tok.text}
                </button>
              ))}
              {isStreaming && (
                <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse rounded-xs bg-emerald-400 align-middle" />
              )}
            </div>
          ) : (
            <div>
              {renderFormattedContent(message.content)}
              {isStreaming && (
                <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse rounded-xs bg-emerald-400 align-middle" />
              )}
            </div>
          )}
        </div>

        {/* Action Bar */}
        {!isStreaming && message.content && (
          <div className="flex flex-wrap items-center gap-1 pt-0.5 opacity-75 transition group-hover:opacity-100">
            <button
              onClick={handleCopy}
              className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-slate-400 hover:bg-slate-800/80 hover:text-slate-200"
            >
              {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
              {copied ? "Copied" : "Copy"}
            </button>

            {onRegenerate && (
              <button
                onClick={onRegenerate}
                className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-slate-400 hover:bg-slate-800/80 hover:text-slate-200"
              >
                <RefreshCw className="h-3 w-3" />
                Regenerate
              </button>
            )}

            <button
              onClick={() => onInspectMessage(message)}
              className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-slate-400 hover:bg-emerald-500/15 hover:text-emerald-300"
            >
              <Eye className="h-3 w-3" />
              Neural X-Ray
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
