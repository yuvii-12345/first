/** Perceptually tuned scientific color scales for Glassbox Mechanistic Interpretability Lab */

export function attentionColor(w: number): string {
  // Deep obsidian -> indigo -> teal -> luminous emerald/cyan
  const v = Math.max(0, Math.min(1, w));
  const p = Math.pow(v, 0.72); // subtle gamma lift so 0.05..0.3 patterns remain crisp
  if (p < 0.02) return "#0b0f18";
  if (p < 0.35) {
    const t = p / 0.35;
    const r = Math.round(11 + t * 18);
    const g = Math.round(15 + t * 65);
    const b = Math.round(24 + t * 105);
    return `rgb(${r},${g},${b})`;
  }
  if (p < 0.72) {
    const t = (p - 0.35) / 0.37;
    const r = Math.round(29 - t * 13);
    const g = Math.round(80 + t * 105);
    const b = Math.round(129 + t * 25);
    return `rgb(${r},${g},${b})`;
  }
  const t = (p - 0.72) / 0.28;
  const r = Math.round(16 + t * 165);
  const g = Math.round(185 + t * 65);
  const b = Math.round(154 + t * 60);
  return `rgb(${r},${g},${b})`;
}

export function logitLensBg(prob: number, mode: "top1" | "final" | "entropy", entropyVal = 0): string {
  if (mode === "entropy") {
    // Low entropy (sharp/confident) = emerald; high entropy = deep slate/amber
    const norm = Math.max(0, Math.min(1, entropyVal / 5.5));
    const conf = 1 - norm;
    const r = Math.round(15 + (1 - conf) * 45);
    const g = Math.round(23 + conf * 135);
    const b = Math.round(42 + conf * 90);
    return `rgba(${r},${g},${b},0.82)`;
  }
  const p = Math.max(0, Math.min(1, prob));
  const lift = Math.pow(p, 0.65);
  const r = Math.round(14 + lift * 12);
  const g = Math.round(20 + lift * 142);
  const b = Math.round(35 + lift * 102);
  return `rgba(${r},${g},${b},0.88)`;
}

export function divergingPatchColor(val: number): string {
  // -1 (coral/rose) -> 0 (deep obsidian slate) -> +1 (luminous emerald/mint)
  const v = Math.max(-1, Math.min(1.25, val));
  if (Math.abs(v) < 0.03) return "#101624";
  if (v > 0) {
    const t = Math.min(1, Math.pow(v, 0.75));
    const r = Math.round(16 + t * 20);
    const g = Math.round(24 + t * 176);
    const b = Math.round(38 + t * 108);
    return `rgb(${r},${g},${b})`;
  }
  const t = Math.min(1, Math.pow(-v, 0.75));
  const r = Math.round(16 + t * 218);
  const g = Math.round(24 + t * 38);
  const b = Math.round(38 + t * 62);
  return `rgb(${r},${g},${b})`;
}

export function roleBadgeClass(role: string): string {
  switch (role) {
    case "Induction":
      return "bg-emerald-500/20 text-emerald-300 border-emerald-500/40";
    case "Prev-Token":
      return "bg-cyan-500/20 text-cyan-300 border-cyan-500/40";
    case "BOS Sink":
      return "bg-amber-500/20 text-amber-300 border-amber-500/40";
    case "Self-Attend":
      return "bg-indigo-500/20 text-indigo-300 border-indigo-500/40";
    case "Sharp Focus":
      return "bg-fuchsia-500/20 text-fuchsia-300 border-fuchsia-500/40";
    default:
      return "bg-slate-700/40 text-slate-300 border-slate-600/50";
  }
}

export const TOKEN_CHIP_PALETTE = [
  "bg-emerald-500/15 border-emerald-500/35 text-emerald-200",
  "bg-sky-500/15 border-sky-500/35 text-sky-200",
  "bg-violet-500/15 border-violet-500/35 text-violet-200",
  "bg-amber-500/15 border-amber-500/35 text-amber-200",
  "bg-rose-500/15 border-rose-500/35 text-rose-200",
  "bg-teal-500/15 border-teal-500/35 text-teal-200",
];
