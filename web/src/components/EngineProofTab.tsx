import React, { useEffect, useMemo, useState } from "react";
import {
  CheckCircle2,
  Cpu,
  Gauge,
  GitFork,
  ShieldCheck,
  Sparkles,
  XCircle,
} from "lucide-react";
import { api } from "../api";
import type {
  AutogradDagResponse,
  AutogradNode,
  EngineVerifyResponse,
  MatmulBenchRow,
} from "../types";

export const EngineProofTab: React.FC = () => {
  const [verifyData, setVerifyData] = useState<EngineVerifyResponse | null>(null);
  const [verifyLoading, setVerifyLoading] = useState(false);

  const [benchRows, setBenchRows] = useState<MatmulBenchRow[] | null>(null);
  const [benchLoading, setBenchLoading] = useState(false);

  const [dagData, setDagData] = useState<AutogradDagResponse | null>(null);
  const [dagLoading, setDagLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState<AutogradNode | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runVerify = async () => {
    setVerifyLoading(true);
    setError(null);
    try {
      setVerifyData(await api.runEngineVerify());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setVerifyLoading(false);
    }
  };

  const runBenchmark = async () => {
    setBenchLoading(true);
    setError(null);
    try {
      const res = await api.runMatmulBenchmark([64, 128, 192, 256]);
      setBenchRows(res.results);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBenchLoading(false);
    }
  };

  const fetchDag = async () => {
    setDagLoading(true);
    setError(null);
    try {
      const d = await api.captureAutogradDag();
      setDagData(d);
      const nodes = d.graph.nodes;
      setSelectedNode(nodes[nodes.length - 1] ?? null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDagLoading(false);
    }
  };

  useEffect(() => {
    fetchDag();
  }, []);

  // Layout DAG nodes by topological depth (left-to-right layered graph)
  const dagLayout = useMemo(() => {
    if (!dagData) return null;
    const nodes = dagData.graph.nodes;
    const edges = dagData.graph.edges;

    const parentsMap = new Map<number, number[]>();
    nodes.forEach((n) => parentsMap.set(n.id, []));
    edges.forEach((e) => {
      parentsMap.get(e.target)?.push(e.source);
    });

    const depthMap = new Map<number, number>();
    const getDepth = (id: number, visiting = new Set<number>()): number => {
      if (depthMap.has(id)) return depthMap.get(id)!;
      if (visiting.has(id)) return 0;
      visiting.add(id);
      const ps = parentsMap.get(id) ?? [];
      const d = ps.length === 0 ? 0 : 1 + Math.max(...ps.map((p) => getDepth(p, visiting)));
      depthMap.set(id, d);
      return d;
    };
    nodes.forEach((n) => getDepth(n.id));

    const cols = new Map<number, AutogradNode[]>();
    nodes.forEach((n) => {
      const d = depthMap.get(n.id) ?? 0;
      if (!cols.has(d)) cols.set(d, []);
      cols.get(d)!.push(n);
    });

    const maxDepth = Math.max(0, ...Array.from(cols.keys()));
    const colW = 148;
    const rowH = 52;
    const maxRows = Math.max(1, ...Array.from(cols.values()).map((c) => c.length));

    const positions = new Map<number, { x: number; y: number }>();
    cols.forEach((colNodes, d) => {
      const totalH = colNodes.length * rowH;
      const startY = (maxRows * rowH - totalH) / 2 + 28;
      colNodes.forEach((n, idx) => {
        positions.set(n.id, {
          x: d * colW + 24,
          y: startY + idx * rowH,
        });
      });
    });

    return {
      width: (maxDepth + 1) * colW + 60,
      height: Math.max(280, maxRows * rowH + 60),
      positions,
    };
  }, [dagData]);

  return (
    <div className="space-y-5">
      {/* Top Banner */}
      <div className="glass-panel rounded-xl p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
              <ShieldCheck className="h-4 w-4 text-emerald-400" />
              Zero-Framework Verification Suite (Pure C++17 Kernels + Python Stdlib Autograd)
            </h2>
            <p className="text-xs text-slate-400">
              No PyTorch · No TensorFlow · No JAX · No NumPy · Every differentiable op verified in float64 against central finite differences
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={runVerify}
              disabled={verifyLoading}
              className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/50 bg-emerald-500/20 px-3.5 py-1.5 font-mono text-xs font-semibold text-emerald-200 transition hover:bg-emerald-500/30 disabled:opacity-50"
            >
              <CheckCircle2 className="h-3.5 w-3.5" />
              {verifyLoading
                ? "Verifying float64 + FlashAttn..."
                : "Run Gradcheck + FlashAttn Proof"}
            </button>
            <button
              onClick={runBenchmark}
              disabled={benchLoading}
              className="inline-flex items-center gap-1.5 rounded-lg border border-amber-500/50 bg-amber-500/20 px-3.5 py-1.5 font-mono text-xs font-semibold text-amber-200 transition hover:bg-amber-500/30 disabled:opacity-50"
            >
              <Gauge className="h-3.5 w-3.5" />
              {benchLoading ? "Benchmarking SGEMM..." : "Benchmark C++17 GEMM"}
            </button>
          </div>
        </div>
        {error && <p className="mt-2 font-mono text-xs text-rose-400">{error}</p>}
      </div>

      {/* 3 Proof Panels */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        {/* 1. Float64 Numerical Gradcheck */}
        <div className="glass-panel rounded-xl p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="flex items-center gap-1.5 text-sm font-semibold text-slate-100">
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
              Float64 Numerical Gradcheck
            </h3>
            {verifyData && (
              <span
                className={`rounded px-2 py-0.5 font-mono text-[10px] font-bold ${
                  verifyData.all_passed
                    ? "bg-emerald-500/20 text-emerald-300"
                    : "bg-rose-500/20 text-rose-300"
                }`}
              >
                {verifyData.all_passed ? "ALL PASSED" : "FAILED"} · {verifyData.duration_ms}ms
              </span>
            )}
          </div>

          {!verifyData ? (
            <div className="flex h-48 flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-obsidian-700 bg-obsidian-950/50 p-4 text-center">
              <p className="font-mono text-xs text-slate-400">
                Verify analytical backprop against central differences{" "}
                <span className="text-emerald-300">(f(x+ε)−f(x−ε))/2ε</span> across all 9 ops + an end-to-end 2-layer GPT Transformer.
              </p>
              <button
                onClick={runVerify}
                disabled={verifyLoading}
                className="rounded-md border border-emerald-500/40 bg-emerald-500/15 px-3 py-1 font-mono text-xs text-emerald-200 hover:bg-emerald-500/25"
              >
                {verifyLoading ? "Computing..." : "Run Verification"}
              </button>
            </div>
          ) : (
            <div className="max-h-72 space-y-1.5 overflow-y-auto pr-1 font-mono text-[11px] tabular-nums">
              {verifyData.gradchecks.map((r) => (
                <div
                  key={r.op}
                  className="flex items-center justify-between rounded-lg bg-obsidian-950/75 px-2.5 py-1.5"
                >
                  <div className="min-w-0 flex-1 pr-2">
                    <div className="flex items-center gap-1.5">
                      {r.passed ? (
                        <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
                      ) : (
                        <XCircle className="h-3.5 w-3.5 shrink-0 text-rose-400" />
                      )}
                      <span className="truncate font-semibold text-slate-200">{r.op}</span>
                    </div>
                    <div className="pl-5 text-[10px] text-slate-500">
                      cos={r.cosine_similarity.toFixed(12)} · {r.checked_elements} elems
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-emerald-300">{r.max_abs_error.toExponential(1)}</div>
                    <div className="text-[9px] text-slate-500">max abs</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 2. FlashAttention-2 Exact Equivalence */}
        <div className="glass-panel rounded-xl p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="flex items-center gap-1.5 text-sm font-semibold text-slate-100">
              <Sparkles className="h-4 w-4 text-cyan-400" />
              FlashAttention-2 Exactness
            </h3>
            {verifyData && (
              <span className="rounded bg-cyan-500/20 px-2 py-0.5 font-mono text-[10px] font-bold text-cyan-300">
                {verifyData.flash_equivalence.every((c) => c.passed)
                  ? "MACHINE EPSILON"
                  : "MISMATCH"}
              </span>
            )}
          </div>

          {!verifyData ? (
            <div className="flex h-48 flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-obsidian-700 bg-obsidian-950/50 p-4 text-center">
              <p className="font-mono text-xs text-slate-400">
                Compare O(T)-memory online-softmax tiled FlashAttention-2 forward &amp; recomputing backward against materialized naive O(T²) attention in float64.
              </p>
              <button
                onClick={runVerify}
                disabled={verifyLoading}
                className="rounded-md border border-cyan-500/40 bg-cyan-500/15 px-3 py-1 font-mono text-xs text-cyan-200 hover:bg-cyan-500/25"
              >
                {verifyLoading ? "Verifying..." : "Verify FlashAttention-2"}
              </button>
            </div>
          ) : (
            <div className="space-y-2.5 font-mono text-[11px] tabular-nums">
              {verifyData.flash_equivalence.map((c, idx) => (
                <div key={idx} className="rounded-lg bg-obsidian-950/75 p-3">
                  <div className="mb-1.5 flex items-center justify-between">
                    <span className="font-semibold text-cyan-300">
                      [{c.shape.join("×")}] {c.causal ? "Causal" : "Bidirectional"} ({c.dtype})
                    </span>
                    <span className="text-[10px] font-bold text-emerald-400">
                      {c.passed ? "PASS" : "FAIL"}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-1.5 text-[10px] text-slate-400">
                    <div>
                      ΔOut: <span className="text-slate-200">{c.max_diff_output.toExponential(2)}</span>
                    </div>
                    <div>
                      ΔdQ: <span className="text-slate-200">{c.max_diff_dQ.toExponential(2)}</span>
                    </div>
                    <div>
                      ΔdK: <span className="text-slate-200">{c.max_diff_dK.toExponential(2)}</span>
                    </div>
                    <div>
                      ΔdV: <span className="text-slate-200">{c.max_diff_dV.toExponential(2)}</span>
                    </div>
                  </div>
                </div>
              ))}
              <p className="pt-1 text-[10px] leading-relaxed text-slate-500">
                Tested on non-block-aligned lengths (T=45, T=41 vs Br=16, Bc=16) proving exact online softmax rescaling &amp; log-sum-exp backward recomputation.
              </p>
            </div>
          )}
        </div>

        {/* 3. Custom C++17 Cache-Tiled GEMM Benchmark */}
        <div className="glass-panel rounded-xl p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="flex items-center gap-1.5 text-sm font-semibold text-slate-100">
              <Cpu className="h-4 w-4 text-amber-400" />
              C++17 Cache-Tiled SGEMM
            </h3>
            {benchRows && (
              <span className="rounded bg-amber-500/20 px-2 py-0.5 font-mono text-[10px] font-bold text-amber-300">
                Peak {Math.max(...benchRows.map((r) => r.tiled_gflops)).toFixed(1)} GF/s
              </span>
            )}
          </div>

          {!benchRows ? (
            <div className="flex h-48 flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-obsidian-700 bg-obsidian-950/50 p-4 text-center">
              <p className="font-mono text-xs text-slate-400">
                Race our cache-blocked 64×64 + 4×8 AVX/FMA register-tiled C++17 kernel against a naive triple-loop GEMM live on this CPU.
              </p>
              <button
                onClick={runBenchmark}
                disabled={benchLoading}
                className="rounded-md border border-amber-500/40 bg-amber-500/15 px-3 py-1 font-mono text-xs text-amber-200 hover:bg-amber-500/25"
              >
                {benchLoading ? "Running SGEMM..." : "Run Live Benchmark"}
              </button>
            </div>
          ) : (
            <div className="space-y-2.5 font-mono text-[11px] tabular-nums">
              {benchRows.map((r) => {
                const maxGf = Math.max(1, ...benchRows.map((x) => x.tiled_gflops));
                return (
                  <div key={r.size} className="rounded-lg bg-obsidian-950/75 p-2.5">
                    <div className="mb-1 flex items-center justify-between">
                      <span className="font-semibold text-slate-200">
                        {r.size}×{r.size} FP32
                      </span>
                      <span className="font-bold text-emerald-300">{r.speedup}× faster</span>
                    </div>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="w-10 text-[10px] text-amber-300">Tiled</span>
                        <div className="h-2 flex-1 overflow-hidden rounded-full bg-obsidian-800">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-amber-400 to-emerald-400"
                            style={{ width: `${(r.tiled_gflops / maxGf) * 100}%` }}
                          />
                        </div>
                        <span className="w-20 text-right text-[10px] text-slate-200">
                          {r.tiled_gflops} GF/s
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-10 text-[10px] text-slate-500">Naive</span>
                        <div className="h-2 flex-1 overflow-hidden rounded-full bg-obsidian-800">
                          <div
                            className="h-full rounded-full bg-slate-500"
                            style={{ width: `${Math.max(2, (r.naive_gflops / maxGf) * 100)}%` }}
                          />
                        </div>
                        <span className="w-20 text-right text-[10px] text-slate-400">
                          {r.naive_gflops} GF/s
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Interactive Reverse-Mode Autograd DAG Explorer */}
      <div className="glass-panel rounded-xl p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-100">
              <GitFork className="h-4 w-4 text-cyan-400" />
              Live Reverse-Mode Autograd Computation DAG
            </h2>
            <p className="text-xs text-slate-400">
              Actual dynamic forward+backward graph captured from{" "}
              <code className="font-mono text-cyan-300">loss.backward()</code> · Click any node to inspect shape, data norm &amp; gradient norm
            </p>
          </div>
          <div className="flex items-center gap-3 font-mono text-xs">
            {dagData && (
              <span className="text-slate-400 tabular-nums">
                {dagData.num_nodes} nodes · {dagData.num_edges} edges · loss={dagData.loss.toFixed(4)}
              </span>
            )}
            <button
              onClick={fetchDag}
              disabled={dagLoading}
              className="rounded-lg border border-obsidian-700 bg-obsidian-800 px-3 py-1 text-slate-200 hover:border-cyan-500/40"
            >
              {dagLoading ? "Capturing..." : "Recapture DAG"}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
          {/* Scrollable SVG DAG Canvas */}
          <div
            className="overflow-auto rounded-lg border border-obsidian-700/80 bg-obsidian-950/90 p-2 xl:col-span-9"
            style={{ maxHeight: 460 }}
          >
            {dagData && dagLayout && (
              <svg width={dagLayout.width} height={dagLayout.height} className="select-none">
                {/* Edges */}
                {dagData.graph.edges.map((e, i) => {
                  const p1 = dagLayout.positions.get(e.source);
                  const p2 = dagLayout.positions.get(e.target);
                  if (!p1 || !p2) return null;
                  const x1 = p1.x + 114;
                  const y1 = p1.y + 18;
                  const x2 = p2.x;
                  const y2 = p2.y + 18;
                  const midX = (x1 + x2) / 2;
                  const active =
                    selectedNode?.id === e.source || selectedNode?.id === e.target;
                  return (
                    <path
                      key={i}
                      d={`M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`}
                      fill="none"
                      stroke={active ? "#34d399" : "#2b374f"}
                      strokeWidth={active ? 1.8 : 1.0}
                      strokeOpacity={active ? 0.95 : 0.7}
                    />
                  );
                })}

                {/* Nodes */}
                {dagData.graph.nodes.map((n) => {
                  const pos = dagLayout.positions.get(n.id);
                  if (!pos) return null;
                  const isSel = selectedNode?.id === n.id;
                  const isLoss = n.op === "cross_entropy";
                  const fill = isLoss
                    ? "rgba(244,63,94,0.22)"
                    : n.is_leaf && n.requires_grad
                      ? "rgba(16,185,129,0.16)"
                      : n.op === "flash_attention"
                        ? "rgba(56,189,248,0.22)"
                        : "#101624";
                  const stroke = isSel
                    ? "#ffffff"
                    : isLoss
                      ? "#fb7185"
                      : n.is_leaf && n.requires_grad
                        ? "#10b981"
                        : n.op === "flash_attention"
                          ? "#38bdf8"
                          : "#334155";
                  return (
                    <g
                      key={n.id}
                      transform={`translate(${pos.x}, ${pos.y})`}
                      onClick={() => setSelectedNode(n)}
                      className="cursor-pointer"
                    >
                      <rect
                        width={114}
                        height={36}
                        rx={7}
                        fill={fill}
                        stroke={stroke}
                        strokeWidth={isSel ? 2 : 1}
                      />
                      <text
                        x={8}
                        y={14}
                        className="fill-slate-100 font-mono text-[10px] font-semibold"
                      >
                        {(n.name || n.op).slice(0, 15)}
                      </text>
                      <text x={8} y={28} className="fill-slate-400 font-mono text-[8.5px]">
                        [{n.shape.join("×")}]
                      </text>
                    </g>
                  );
                })}
              </svg>
            )}
          </div>

          {/* Node Inspector Sidebar */}
          <div className="rounded-lg border border-obsidian-700/80 bg-obsidian-950/75 p-3.5 font-mono text-xs xl:col-span-3">
            <div className="mb-2 text-[10px] uppercase tracking-wider text-slate-500">
              Selected Tensor Node
            </div>
            {selectedNode ? (
              <div className="space-y-2.5 tabular-nums">
                <div>
                  <div className="text-[10px] text-slate-500">Name / Op</div>
                  <div className="text-sm font-bold text-emerald-300">
                    {selectedNode.name || selectedNode.op}
                  </div>
                  <div className="text-[11px] text-cyan-300">op: {selectedNode.op}</div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded bg-obsidian-900 p-2">
                    <div className="text-[9px] text-slate-500">Shape</div>
                    <div className="text-slate-200">[{selectedNode.shape.join(", ")}]</div>
                  </div>
                  <div className="rounded bg-obsidian-900 p-2">
                    <div className="text-[9px] text-slate-500">Elements</div>
                    <div className="text-slate-200">{selectedNode.numel.toLocaleString()}</div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded bg-obsidian-900 p-2">
                    <div className="text-[9px] text-slate-500">‖Tensor‖₂</div>
                    <div className="text-cyan-300">
                      {selectedNode.data_norm != null ? selectedNode.data_norm.toFixed(4) : "—"}
                    </div>
                  </div>
                  <div className="rounded bg-obsidian-900 p-2">
                    <div className="text-[9px] text-slate-500">‖Grad‖₂</div>
                    <div className="text-amber-300">
                      {selectedNode.grad_norm != null ? selectedNode.grad_norm.toFixed(4) : "none"}
                    </div>
                  </div>
                </div>
                <div className="rounded bg-obsidian-900 p-2 text-[11px] text-slate-400">
                  <div>Leaf: {selectedNode.is_leaf ? "Yes (Parameter/Input)" : "No (Op Output)"}</div>
                  <div>Requires Grad: {selectedNode.requires_grad ? "True" : "False"}</div>
                </div>

                {/* Color Legend */}
                <div className="space-y-1 border-t border-obsidian-800 pt-2.5 text-[10px] text-slate-400">
                  <div className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-sm border border-emerald-500 bg-emerald-500/20" />
                    Learnable Parameter Leaf
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-sm border border-sky-400 bg-sky-400/25" />
                    FlashAttention-2 Op
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-sm border border-rose-400 bg-rose-500/25" />
                    Cross-Entropy Scalar Loss
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-slate-500">Click any node in the graph to inspect it.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
