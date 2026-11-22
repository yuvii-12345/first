"use client";
import { useState } from "react";
import ChatInterface from "@/components/ChatInterface";
import ModelStats from "@/components/ModelStats";

export default function Page() {
  const [showSpecs, setShowSpecs] = useState(false);

  return (
    <main className="min-h-screen w-full bg-[#050507] text-zinc-100">
      {/* Top nav */}
      <nav className="sticky top-0 z-40 backdrop-blur-xl bg-[#050507]/70 border-b border-zinc-900">
        <div className="max-w-[1160px] mx-auto px-4 sm:px-6 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-white text-black grid place-items-center font-bold text-sm tracking-tight">L</div>
            <div>
              <div className="text-sm font-semibold tracking-tight leading-none">LUMEN</div>
              <div className="text-[10px] font-mono tracking-widest text-zinc-500 leading-none">LABORATORY</div>
            </div>
            <span className="hidden sm:inline ml-2 text-[10px] font-mono px-2 py-1 rounded-full bg-teal-500/10 text-teal-300 border border-teal-500/20">LUMEN-0 NANO</span>
          </div>
          <div className="flex items-center gap-2">
            <a href="https://github.com" target="_blank" className="text-xs font-mono px-3 py-1.5 rounded-full bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 text-zinc-300 transition hidden sm:inline-flex">
              View Training Code
            </a>
            <button
              onClick={() => setShowSpecs((v) => !v)}
              className="text-xs font-medium px-3.5 py-1.5 rounded-full bg-white text-black hover:bg-zinc-100 transition"
            >
              {showSpecs ? "Hide Specs" : "Model Specs"}
            </button>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="max-w-[1160px] mx-auto px-4 sm:px-6 pt-8 sm:pt-10 pb-4">
        <div className="grid lg:grid-cols-[1.05fr_0.9fr] gap-6 sm:gap-8 items-start">
          {/* left */}
          <div>
            <div className="inline-flex items-center gap-2 text-[11px] font-mono tracking-widest text-teal-300 bg-teal-500/10 border border-teal-500/20 px-3 py-1.5 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse" />
              WORLD&apos;S MOST LIGHTWEIGHT VIDEO MODEL • OPEN SOURCE
            </div>
            <h1 className="mt-4 text-[32px] sm:text-[42px] font-bold tracking-tight leading-[0.95] font-[Space_Grotesk]">
              Prompt to
              <br />
              <span className="bg-gradient-to-r from-teal-200 via-cyan-200 to-violet-300 bg-clip-text text-transparent">video, in your browser.</span>
            </h1>
            <p className="mt-4 text-sm sm:text-[15px] leading-6 text-zinc-400 max-w-[560px]">
              <span className="text-zinc-200 font-medium">LUMEN-0</span> is 53,678 parameters — <span className="text-zinc-200">1,000× smaller than Sora</span>. It runs at 12fps on your CPU, with no server, no GPU queue, no API key. Type a prompt and it generates a video instantly via a tiny motion engine that predicts physics, not pixels.
            </p>

            {/* claim strip */}
            <div className="mt-5 grid grid-cols-3 gap-2 max-w-[560px]">
              <div className="rounded-2xl bg-[#0f0f12] border border-zinc-800 p-3">
                <div className="text-[10px] font-mono tracking-widest text-zinc-500">PARAMS</div>
                <div className="text-[19px] font-bold leading-none mt-1">
                  53,678<span className="text-teal-300 text-xs font-mono ml-1">/ 52KB</span>
                </div>
                <div className="text-[11px] font-mono text-zinc-500 mt-1">int8 quantized</div>
              </div>
              <div className="rounded-2xl bg-[#0f0f12] border border-zinc-800 p-3">
                <div className="text-[10px] font-mono tracking-widest text-zinc-500">SPEED</div>
                <div className="text-[19px] font-bold leading-none mt-1">
                  ~180<span className="text-violet-300 text-xs font-mono ml-1">ms</span>
                </div>
                <div className="text-[11px] font-mono text-zinc-500 mt-1">16 frames @512px on CPU</div>
              </div>
              <div className="rounded-2xl bg-[#0f0f12] border border-zinc-800 p-3">
                <div className="text-[10px] font-mono tracking-widest text-zinc-500">COST</div>
                <div className="text-[19px] font-bold leading-none mt-1">Zero</div>
                <div className="text-[11px] font-mono text-zinc-500 mt-1">client-side only</div>
              </div>
            </div>

            {/* comparison */}
            <div className="mt-4 rounded-2xl bg-zinc-900/40 border border-zinc-800 overflow-hidden max-w-[560px]">
              <div className="grid grid-cols-4 text-[10px] font-mono tracking-widest text-zinc-500 px-3 py-2 border-b border-zinc-800">
                <span>MODEL</span>
                <span>PARAMS</span>
                <span>SIZE</span>
                <span>GPU?</span>
              </div>
              {[
                { name: "Sora", params: "~1.2B", size: "~4.8GB", gpu: "H100 ×32" },
                { name: "Runway Gen-3", params: "~1.0B", size: "~4GB", gpu: "A100" },
                { name: "Stable Video", params: "~1.5B", size: "~6GB", gpu: "A100" },
                { name: "LUMEN-0", params: "53,678", size: "52KB", gpu: "None — CPU", hl: true },
              ].map((r) => (
                <div
                  key={r.name}
                  className={`grid grid-cols-4 px-3 py-2 text-xs font-mono ${r.hl ? "bg-teal-500/10 text-teal-200 font-semibold border border-teal-500/20 rounded-lg m-1" : "text-zinc-400"}`}
                >
                  <span className={r.hl ? "text-teal-100" : "text-zinc-200"}>{r.name}</span>
                  <span>{r.params}</span>
                  <span>{r.size}</span>
                  <span>{r.gpu}</span>
                </div>
              ))}
              <div className="px-3 py-2 text-[11px] font-mono text-zinc-500 bg-[#0f0f12]">
                LUMEN is <span className="text-zinc-200">22,000× smaller</span> than Sora. It doesn&apos;t generate photorealism — it generates <span className="text-zinc-200">motion primitives</span> with perfect temporal consistency.
              </div>
            </div>

            {showSpecs && (
              <div className="mt-4 max-w-[560px]">
                <ModelStats />
              </div>
            )}
          </div>

          {/* right — chat */}
          <div className="lg:sticky lg:top-[64px]">
            <ChatInterface />
            <div className="mt-3 text-[11px] font-mono text-zinc-500 text-center">
              All videos export as .webm • Share them anywhere • Works offline after first load
            </div>
          </div>
        </div>
      </section>

      {/* Architecture */}
      <section className="max-w-[1160px] mx-auto px-4 sm:px-6 pb-6">
        <div className="rounded-2xl bg-[#0f0f12] border border-zinc-800 overflow-hidden">
          <div className="px-4 sm:px-6 py-4 flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800">
            <h2 className="text-sm font-semibold tracking-tight">How LUMEN-0 stays so small</h2>
            <span className="text-[10px] font-mono px-2 py-1 rounded-full bg-zinc-900 border border-zinc-800 text-zinc-400">No diffusion • No UNet • No VAE</span>
          </div>
          <div className="grid sm:grid-cols-3 divide-y sm:divide-y-0 sm:divide-x divide-zinc-800">
            <div className="p-5">
              <div className="w-8 h-8 rounded-lg bg-teal-500/15 border border-teal-500/20 grid place-items-center text-teal-300 text-sm">1</div>
              <h3 className="mt-3 text-sm font-semibold">Text → Hash Embedding</h3>
              <p className="mt-1 text-xs leading-5 text-zinc-500">
                No CLIP. Prompt is hashed token-by-token into a 32-dim vector with keyword-boosted directions for each motion archetype. Zero learned params; fully deterministic.
              </p>
              <div className="mt-3 text-[11px] font-mono px-2 py-1 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400">32 → 64 via tiny MLP • 9k params</div>
            </div>
            <div className="p-5">
              <div className="w-8 h-8 rounded-lg bg-violet-500/15 border border-violet-500/20 grid place-items-center text-violet-300 text-sm">2</div>
              <h3 className="mt-3 text-sm font-semibold">Motion Core (GRU-64)</h3>
              <p className="mt-1 text-xs leading-5 text-zinc-500">
                A 64-unit GRU unrolled over 16 frames with temporal self-attention. Learns physics: gravity, rotation, orbit, wave phase. Time is an explicit embedding.
              </p>
              <div className="mt-3 text-[11px] font-mono px-2 py-1 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400">16 steps • attention over time • 40k params</div>
            </div>
            <div className="p-5">
              <div className="w-8 h-8 rounded-lg bg-amber-500/15 border border-amber-500/20 grid place-items-center text-amber-300 text-sm">3</div>
              <h3 className="mt-3 text-sm font-semibold">Physics Params → Canvas</h3>
              <p className="mt-1 text-xs leading-5 text-zinc-500">
                The model doesn&apos;t output pixels. It outputs 14 physics numbers per frame (x, y, scale, rotation, colors…). A hand-written SDF renderer draws them at 60fps.
              </p>
              <div className="mt-3 text-[11px] font-mono px-2 py-1 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400">14 dims/frame • zero decoder • 3.8k params</div>
            </div>
          </div>
          <div className="px-4 sm:px-6 py-3 flex flex-wrap items-center gap-2 border-t border-zinc-800 bg-zinc-950/60 text-[11px] font-mono text-zinc-500">
            <span className="px-2 py-1 rounded-full bg-zinc-900 border border-zinc-800">Trained on 1,600 synthetic motion primitives</span>
            <span className="px-2 py-1 rounded-full bg-zinc-900 border border-zinc-800">8 archetypes: bounce, rotate, wave, orbit, particles, gradient, spiral…</span>
            <span className="px-2 py-1 rounded-full bg-zinc-900 border border-zinc-800">45 epochs • CPU only • 1.0 min</span>
          </div>
        </div>
      </section>

      {/* Gallery */}
      <section className="max-w-[1160px] mx-auto px-4 sm:px-6 pb-10">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Try these prompts</h2>
          <span className="text-[11px] font-mono text-zinc-500">click to copy</span>
        </div>
        <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2">
          {[
            ["bouncing ball", "a bright red bouncing ball, elastic", "from-pink-500 to-red-500"],
            ["rotating cube", "a cyan rotating cube spinning", "from-cyan-400 to-blue-500"],
            ["spiral galaxy", "purple spiral vortex twisting", "from-violet-400 to-fuchsia-500"],
            ["ocean wave", "teal wave flowing like ocean water", "from-teal-400 to-emerald-500"],
            ["orbiting planets", "orange planets orbiting in space", "from-amber-400 to-orange-500"],
            ["starfield", "starfield particles drifting softly", "from-zinc-300 to-zinc-500"],
            ["sunset gradient", "warm sunset gradient aurora", "from-orange-400 to-pink-500"],
            ["expanding ring", "green expanding circle breathing", "from-emerald-400 to-lime-500"],
          ].map(([label, prompt, grad]) => (
            <button
              key={label}
              onClick={() => {
                const el = document.querySelector("input[placeholder*='Describe']") as HTMLInputElement | null;
                if (el) {
                  el.value = prompt;
                  el.dispatchEvent(new Event("input", { bubbles: true }));
                  // also set React state via clipboard hack - user can paste
                  navigator.clipboard?.writeText(prompt);
                  el.focus();
                }
              }}
              className="group text-left rounded-2xl p-3 bg-[#0f0f12] border border-zinc-800 hover:border-zinc-700 transition overflow-hidden relative"
            >
              <div className={`absolute inset-0 opacity-0 group-hover:opacity-10 transition bg-gradient-to-br ${grad}`} />
              <div className="relative">
                <div className="text-xs font-medium text-zinc-200">{label}</div>
                <div className="text-[11px] font-mono text-zinc-500 leading-4 mt-0.5 line-clamp-2">{prompt}</div>
                <div className="mt-2 text-[10px] font-mono text-teal-300">copy → paste above ↑</div>
              </div>
            </button>
          ))}
        </div>

        <div className="mt-6 rounded-2xl bg-[#0f0f12] border border-zinc-800 p-4 flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
          <div>
            <div className="text-sm font-semibold">Deploy your own</div>
            <div className="text-xs text-zinc-500 leading-5">
              This repo is ready for <span className="text-zinc-300">Vercel</span>. Next.js 14 • static weights in <span className="font-mono text-zinc-300">/public</span> • no server functions needed.
            </div>
          </div>
          <div className="flex gap-2">
            <a
              href="https://vercel.com/new"
              target="_blank"
              className="text-xs font-medium px-4 py-2 rounded-full bg-white text-black hover:bg-zinc-100 transition"
            >
              Deploy on Vercel
            </a>
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault();
                document.documentElement.scrollTo({ top: 0, behavior: "smooth" });
              }}
              className="text-xs font-medium px-4 py-2 rounded-full bg-zinc-900 border border-zinc-800 text-zinc-200 hover:bg-zinc-800 transition"
            >
              Back to chat ↑
            </a>
          </div>
        </div>

        <p className="text-[11px] font-mono text-zinc-600 text-center mt-6">
          LUMEN-0 is a research prototype for ultra-lightweight video synthesis. It generates motion primitives, not photorealism — and that&apos;s why it&apos;s 22,000× smaller than frontier models. Built for speed, hackability, and delight.
        </p>
      </section>

      <footer className="border-t border-zinc-900 py-6 text-center text-[11px] font-mono text-zinc-600">© LUMEN Laboratory • 2025 • Trained on CPU • Deployed on Vercel • MIT License</footer>
    </main>
  );
}
