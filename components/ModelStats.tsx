"use client";
import { useEffect, useState } from "react";

export default function ModelStats() {
  const [meta, setMeta] = useState<any>(null);
  useEffect(() => {
    fetch("/model-meta.json")
      .then((r) => r.json())
      .then(setMeta)
      .catch(() => {});
  }, []);

  if (!meta) return null;

  const history = meta.training.history;
  const train = history.train_loss;
  const val = history.val_loss;

  const max = Math.max(...train, ...val);
  const min = Math.min(...train, ...val);

  const toPath = (arr: number[], color: string) => {
    const w = 320;
    const h = 80;
    const pad = 8;
    const points = arr.map((v, i) => {
      const x = pad + (i / (arr.length - 1)) * (w - pad * 2);
      const y = h - pad - ((v - min) / (max - min + 0.01)) * (h - pad * 2);
      return `${x},${y}`;
    });
    return (
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points.join(" ")}
      />
    );
  };

  return (
    <div className="rounded-2xl bg-[#0f0f12] border border-zinc-800 p-4 sm:p-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold tracking-tight">Training Log — LUMEN-0</h3>
        <span className="text-[10px] font-mono px-2 py-1 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
          45 epochs • {meta.training.num_train} samples • {meta.training.time_minutes.toFixed(1)} min
        </span>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-2">
        <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-3">
          <div className="text-[10px] font-mono tracking-widest text-zinc-500">PARAMS</div>
          <div className="text-lg font-bold tracking-tight">{meta.parameters.toLocaleString()}</div>
          <div className="text-[11px] font-mono text-zinc-500">{meta.params_fp32_kb.toFixed(1)} KB fp32 • {meta.params_int8_kb.toFixed(1)} KB int8</div>
        </div>
        <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-3">
          <div className="text-[10px] font-mono tracking-widest text-zinc-500">ARCHETYPES</div>
          <div className="text-lg font-bold tracking-tight">{meta.archetypes.length}</div>
          <div className="text-[11px] font-mono text-zinc-500">bouncing • rotating • wave…</div>
        </div>
        <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-3">
          <div className="text-[10px] font-mono tracking-widest text-zinc-500">BEST VAL LOSS</div>
          <div className="text-lg font-bold tracking-tight">{meta.training.best_val.toFixed(4)}</div>
          <div className="text-[11px] font-mono text-zinc-500">MSE weighted + smoothness</div>
        </div>
      </div>

      <div className="mt-4">
        <div className="text-[11px] font-mono text-zinc-500 mb-2 flex gap-3">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-0.5 bg-teal-400" />
            train
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-0.5 bg-violet-400" />
            val
          </span>
          <span className="ml-auto">epoch 1 → {meta.training.epochs}</span>
        </div>
        <div className="w-full overflow-x-auto">
          <svg viewBox="0 0 320 80" className="w-full h-[80px] bg-zinc-950 rounded-xl border border-zinc-800">
            {/* grid */}
            <line x1="8" y1="20" x2="312" y2="20" stroke="#27272a" strokeWidth="0.5" strokeDasharray="2 3" />
            <line x1="8" y1="40" x2="312" y2="40" stroke="#27272a" strokeWidth="0.5" strokeDasharray="2 3" />
            <line x1="8" y1="60" x2="312" y2="60" stroke="#27272a" strokeWidth="0.5" strokeDasharray="2 3" />
            {toPath(train, "#2dd4bf")}
            {toPath(val, "#a78bfa")}
          </svg>
        </div>
        <div className="mt-2 text-[10px] font-mono text-zinc-600">
          Synthetic motion primitives • text → physics params • procedural SDF renderer • trained on CPU in {meta.training.time_minutes.toFixed(0)}m — no GPU needed.
        </div>
      </div>
    </div>
  );
}
