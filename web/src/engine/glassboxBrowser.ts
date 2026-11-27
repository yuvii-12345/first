/**
 * Glassbox Browser Engine — Zero-Dependency Pure-TypeScript Float32 Transformer & Interpretability Engine.
 *
 * Mirrors `glassbox/csrc/kernels.cpp`, `glassbox/bpe.py`, `glassbox/transformer.py`, and `glassbox/interp.py`
 * bit-for-bit so the deployed Vercel app runs REAL neural inference, live token-by-token streaming chat,
 * attention head ablation, Logit Lens, Direct Logit Attribution, and Causal Activation Patching directly
 * in the user's browser with zero external ML libraries or servers.
 */

import type {
  AblationComparison,
  AttentionAnalysis,
  AutogradDagResponse,
  BPEInspectResponse,
  BPEStats,
  DatasetMeta,
  DLAComponent,
  EngineVerifyResponse,
  GeneratedTokenStep,
  GenerateResponse,
  GPTConfig,
  HeadData,
  LabStatus,
  LogitLensAnalysis,
  LogitLensCell,
  MatmulBenchRow,
  ModelPreset,
  PatchingAnalysis,
  StatusResponse,
  StepRecord,
  TokenInfo,
  TokenPrediction,
} from "../types";

const BOS_ID = 256;
const EOS_ID = 257;
const PAD_ID = 258;
const PRETOK_RE = /'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+/g;

interface PackedTensor {
  shape: number[];
  quant: "int8_row" | "f32";
  scales_b64?: string;
  data_b64: string;
}

export interface BrowserModelBundle {
  format: string;
  model_label: string;
  status: LabStatus;
  history: StepRecord[];
  datasets: DatasetMeta[];
  presets: ModelPreset[];
  cfg: GPTConfig;
  tokenizer: {
    merges: [number, number, number][];
    merge_counts: Record<string, number>;
  };
  weights: Record<string, PackedTensor>;
  verify_snapshot: EngineVerifyResponse;
  benchmark_snapshot: MatmulBenchRow[];
  autograd_dag_snapshot: AutogradDagResponse;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  prompt: string;
  reply: string;
  tokens: GeneratedTokenStep[];
}

// ---- Base64 + Quantization Unpack ----

function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function unpackTensor(p: PackedTensor): Float32Array {
  if (p.quant === "f32") {
    const bytes = b64ToBytes(p.data_b64);
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }
  // Symmetric per-row Int8 + Float32 scales
  const [rows, cols] = p.shape;
  const sBytes = b64ToBytes(p.scales_b64!);
  const scales = new Float32Array(sBytes.buffer, sBytes.byteOffset, rows);
  const qBytes = b64ToBytes(p.data_b64);
  const qInt8 = new Int8Array(qBytes.buffer, qBytes.byteOffset, rows * cols);
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const sc = scales[r];
    const off = r * cols;
    for (let c = 0; c < cols; c++) {
      out[off + c] = qInt8[off + c] * sc;
    }
  }
  return out;
}

// ---- Exact Byte-Level BPE Tokenizer ----

const UTF8_ENC = new TextEncoder();
const UTF8_DEC = new TextDecoder("utf-8", { fatal: false });

function formatDisplay(bytes: Uint8Array): string {
  const s = UTF8_DEC.decode(bytes);
  let out = "";
  for (const ch of s) {
    if (ch === " ") out += "·";
    else if (ch === "\n") out += "↵";
    else if (ch === "\t") out += "⇥";
    else if (ch === "\r") out += "␍";
    else out += ch;
  }
  return out || "∅";
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join(" ");
}

export class BrowserBPETokenizer {
  vocabBytes: Map<number, Uint8Array> = new Map();
  merges: [number, number, number][] = [];
  mergeRank: Map<string, [number, number]> = new Map(); // "a,b" -> [rank, newId]
  mergeCounts: Record<string, number> = {};
  vocabSize: number = 259;

  constructor(data: { merges: [number, number, number][]; merge_counts: Record<string, number> }) {
    for (let b = 0; b < 256; b++) {
      this.vocabBytes.set(b, new Uint8Array([b]));
    }
    this.merges = data.merges;
    this.mergeCounts = data.merge_counts || {};
    for (let rank = 0; rank < this.merges.length; rank++) {
      const [a, b, newId] = this.merges[rank];
      this.mergeRank.set(`${a},${b}`, [rank, newId]);
      const ba = this.vocabBytes.get(a) ?? new Uint8Array();
      const bb = this.vocabBytes.get(b) ?? new Uint8Array();
      const merged = new Uint8Array(ba.length + bb.length);
      merged.set(ba, 0);
      merged.set(bb, ba.length);
      this.vocabBytes.set(newId, merged);
    }
    this.vocabSize = 259 + this.merges.length;
  }

  tokenText(tid: number): string {
    if (tid === BOS_ID) return "<BOS>";
    if (tid === EOS_ID) return "<EOS>";
    if (tid === PAD_ID) return "<PAD>";
    const b = this.vocabBytes.get(tid);
    return b ? UTF8_DEC.decode(b) : "";
  }

  tokenDisplay(tid: number): string {
    if (tid === BOS_ID) return "<BOS>";
    if (tid === EOS_ID) return "<EOS>";
    if (tid === PAD_ID) return "<PAD>";
    const b = this.vocabBytes.get(tid);
    return b ? formatDisplay(b) : `?${tid}`;
  }

  private encodeChunk(chunk: string): number[] {
    const raw = UTF8_ENC.encode(chunk);
    let ids = Array.from(raw);
    while (ids.length >= 2) {
      let bestRank = Infinity;
      let bestPairA = -1;
      let bestPairB = -1;
      let bestNewId = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const hit = this.mergeRank.get(`${ids[i]},${ids[i + 1]}`);
        if (hit && hit[0] < bestRank) {
          bestRank = hit[0];
          bestPairA = ids[i];
          bestPairB = ids[i + 1];
          bestNewId = hit[1];
        }
      }
      if (bestNewId === -1) break;
      const next: number[] = [];
      let i = 0;
      while (i < ids.length) {
        if (i < ids.length - 1 && ids[i] === bestPairA && ids[i + 1] === bestPairB) {
          next.push(bestNewId);
          i += 2;
        } else {
          next.push(ids[i]);
          i += 1;
        }
      }
      ids = next;
    }
    return ids;
  }

  encode(text: string, addBos = true, addEos = false): number[] {
    const out: number[] = [];
    if (addBos) out.push(BOS_ID);
    const chunks = text.match(PRETOK_RE) || [];
    for (const ch of chunks) {
      const cIds = this.encodeChunk(ch);
      for (const id of cIds) out.push(id);
    }
    if (addEos) out.push(EOS_ID);
    return out;
  }

  decode(ids: number[]): string {
    const parts: Uint8Array[] = [];
    let total = 0;
    for (const id of ids) {
      if (id === BOS_ID || id === EOS_ID || id === PAD_ID) continue;
      const b = this.vocabBytes.get(id);
      if (b) {
        parts.push(b);
        total += b.length;
      }
    }
    const merged = new Uint8Array(total);
    let off = 0;
    for (const p of parts) {
      merged.set(p, off);
      off += p.length;
    }
    return UTF8_DEC.decode(merged);
  }

  inspectEncode(text: string): BPEInspectResponse["inspection"] {
    const rawBytes = UTF8_ENC.encode(text);
    const chunks = text.match(PRETOK_RE) || [];
    const tokens: BPEInspectResponse["inspection"]["tokens"] = [];
    const chunk_traces: BPEInspectResponse["inspection"]["chunk_traces"] = [];

    for (const ch of chunks) {
      const chBytes = UTF8_ENC.encode(ch);
      let ids = Array.from(chBytes);
      const initial_bytes = ids.map((b) => this.tokenDisplay(b));
      const steps: BPEInspectResponse["inspection"]["chunk_traces"][number]["steps"] = [];

      while (ids.length >= 2) {
        let bestRank = Infinity;
        let pairA = -1;
        let pairB = -1;
        let newId = -1;
        for (let i = 0; i < ids.length - 1; i++) {
          const hit = this.mergeRank.get(`${ids[i]},${ids[i + 1]}`);
          if (hit && hit[0] < bestRank) {
            bestRank = hit[0];
            pairA = ids[i];
            pairB = ids[i + 1];
            newId = hit[1];
          }
        }
        if (newId === -1) break;
        const next: number[] = [];
        let i = 0;
        while (i < ids.length) {
          if (i < ids.length - 1 && ids[i] === pairA && ids[i + 1] === pairB) {
            next.push(newId);
            i += 2;
          } else {
            next.push(ids[i]);
            i += 1;
          }
        }
        ids = next;
        steps.push({
          rank: bestRank,
          pair: [this.tokenDisplay(pairA), this.tokenDisplay(pairB)],
          merged: this.tokenDisplay(newId),
          new_id: newId,
          tokens_after: ids.map((x) => this.tokenDisplay(x)),
        });
      }

      if (steps.length > 0 && chunk_traces.length < 16) {
        chunk_traces.push({ chunk: formatDisplay(chBytes), initial_bytes, steps });
      }
      for (const tid of ids) {
        const b = this.vocabBytes.get(tid) ?? new Uint8Array();
        tokens.push({
          id: tid,
          text: this.tokenText(tid),
          display: this.tokenDisplay(tid),
          bytes_hex: bytesToHex(b),
          byte_len: b.length,
          is_merged: tid >= 259,
        });
      }
    }

    const num_bytes = rawBytes.length;
    const num_tokens = tokens.length;
    return {
      text,
      num_bytes,
      num_tokens,
      compression_ratio: num_tokens > 0 ? Math.round((num_bytes / num_tokens) * 100) / 100 : 1,
      tokens,
      chunk_traces,
    };
  }

  getMergeTable(limit = 100): BPEInspectResponse["merges"] {
    return this.merges.slice(0, limit).map(([a, b, newId], rank) => ({
      rank,
      new_id: newId,
      left_id: a,
      right_id: b,
      left: this.tokenDisplay(a),
      right: this.tokenDisplay(b),
      merged: this.tokenDisplay(newId),
      count: this.mergeCounts[String(newId)] ?? 0,
    }));
  }
}

// ---- Pure Float32 Transformer Forward & Interpretability Kernels ----

interface BlockWeights {
  norm1: Float32Array;
  wQ: Float32Array;
  wK: Float32Array;
  wV: Float32Array;
  wO: Float32Array;
  norm2: Float32Array;
  wGate: Float32Array;
  wUp: Float32Array;
  wDown: Float32Array;
}

interface ForwardTrace {
  residStages: { label: string; data: Float32Array }[]; // each [T * D]
  embedOut: Float32Array; // [T * D]
  attnProbs: Float32Array[]; // per layer: [H * T * T]
  perHeadLastOut: Float32Array[][]; // [layer][head] -> [D] at final pos
  mlpLastOut: Float32Array[]; // [layer] -> [D] at final pos
  finalResidLast: Float32Array; // [D] before final_norm
}

interface PatchSpec {
  stageIdx: number; // 0 = embed, 1..L = after block (stageIdx-1)
  pos: number;
  vector: Float32Array; // [D]
}

function matmulNT(x: Float32Array, w: Float32Array, T: number, K: number, N: number): Float32Array {
  // y[t, n] = sum_k x[t, k] * w[n, k]
  const y = new Float32Array(T * N);
  for (let t = 0; t < T; t++) {
    const xOff = t * K;
    const yOff = t * N;
    for (let n = 0; n < N; n++) {
      const wOff = n * K;
      let acc = 0.0;
      for (let k = 0; k < K; k++) {
        acc += x[xOff + k] * w[wOff + k];
      }
      y[yOff + n] = acc;
    }
  }
  return y;
}

function rmsNorm(x: Float32Array, w: Float32Array, T: number, D: number, eps: number): Float32Array {
  const y = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const off = t * D;
    let ss = 0.0;
    for (let d = 0; d < D; d++) {
      const v = x[off + d];
      ss += v * v;
    }
    const inv = 1.0 / Math.sqrt(ss / D + eps);
    for (let d = 0; d < D; d++) {
      y[off + d] = x[off + d] * inv * w[d];
    }
  }
  return y;
}

function applyRoPEInPlace(
  x: Float32Array, // [H, T, Dh]
  H: number,
  T: number,
  Dh: number,
  base: number
): void {
  const half = Dh >> 1;
  const cosTab = new Float32Array(T * half);
  const sinTab = new Float32Array(T * half);
  for (let i = 0; i < half; i++) {
    const invFreq = Math.pow(base, (-2.0 * i) / Dh);
    for (let t = 0; t < T; t++) {
      const ang = t * invFreq;
      cosTab[t * half + i] = Math.cos(ang);
      sinTab[t * half + i] = Math.sin(ang);
    }
  }
  const hStride = T * Dh;
  for (let h = 0; h < H; h++) {
    const hOff = h * hStride;
    for (let t = 0; t < T; t++) {
      const row = hOff + t * Dh;
      const tOff = t * half;
      for (let i = 0; i < half; i++) {
        const x0 = x[row + 2 * i];
        const x1 = x[row + 2 * i + 1];
        const c = cosTab[tOff + i];
        const s = sinTab[tOff + i];
        x[row + 2 * i] = x0 * c - x1 * s;
        x[row + 2 * i + 1] = x0 * s + x1 * c;
      }
    }
  }
}

function softmaxVec(logits: Float32Array | number[], temp = 1.0): { probs: Float32Array; entropy: number } {
  const V = logits.length;
  const invT = 1.0 / Math.max(temp, 1e-5);
  let mx = -Infinity;
  for (let i = 0; i < V; i++) {
    const s = logits[i] * invT;
    if (s > mx) mx = s;
  }
  const probs = new Float32Array(V);
  let sum = 0.0;
  for (let i = 0; i < V; i++) {
    const e = Math.exp(logits[i] * invT - mx);
    probs[i] = e;
    sum += e;
  }
  const invS = 1.0 / (sum || 1.0);
  let ent = 0.0;
  for (let i = 0; i < V; i++) {
    const p = probs[i] * invS;
    probs[i] = p;
    if (p > 1e-12) ent -= p * Math.log(p);
  }
  return { probs, entropy: ent };
}

export class BrowserGPTModel {
  cfg: GPTConfig;
  tokEmb: Float32Array;
  blocks: BlockWeights[];
  finalNorm: Float32Array;

  constructor(cfg: GPTConfig, packedWeights: Record<string, PackedTensor>) {
    this.cfg = cfg;
    this.tokEmb = unpackTensor(packedWeights["tok_emb.weight"]);
    this.finalNorm = unpackTensor(packedWeights["final_norm.weight"]);
    this.blocks = [];
    for (let l = 0; l < cfg.n_layers; l++) {
      const p = `blocks.${l}`;
      this.blocks.push({
        norm1: unpackTensor(packedWeights[`${p}.norm1.weight`]),
        wQ: unpackTensor(packedWeights[`${p}.attn.w_q.weight`]),
        wK: unpackTensor(packedWeights[`${p}.attn.w_k.weight`]),
        wV: unpackTensor(packedWeights[`${p}.attn.w_v.weight`]),
        wO: unpackTensor(packedWeights[`${p}.attn.w_o.weight`]),
        norm2: unpackTensor(packedWeights[`${p}.norm2.weight`]),
        wGate: unpackTensor(packedWeights[`${p}.mlp.w_gate.weight`]),
        wUp: unpackTensor(packedWeights[`${p}.mlp.w_up.weight`]),
        wDown: unpackTensor(packedWeights[`${p}.mlp.w_down.weight`]),
      });
    }
  }

  projectToLogits(resid: Float32Array, T: number): Float32Array {
    const { d_model: D, vocab_size: V, rmsnorm_eps: eps } = this.cfg;
    const normed = rmsNorm(resid, this.finalNorm, T, D, eps);
    return matmulNT(normed, this.tokEmb, T, D, V);
  }

  forward(
    tokenIds: number[],
    opts?: {
      captureTrace?: boolean;
      ablateHeads?: Map<number, Set<number>>;
      ablationMode?: "zero" | "mean";
      patch?: PatchSpec;
    }
  ): { logits: Float32Array; trace?: ForwardTrace } {
    const { d_model: D, n_layers: L, n_heads: H, d_ff: FF, rope_base: ropeBase, rmsnorm_eps: eps } =
      this.cfg;
    const Dh = D / H;
    const scale = 1.0 / Math.sqrt(Dh);
    const T = tokenIds.length;
    const capture = Boolean(opts?.captureTrace);
    const ablateHeads = opts?.ablateHeads;
    const ablationMode = opts?.ablationMode || "zero";
    const patch = opts?.patch;

    // 1. Token Embedding lookup
    let x = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const tid = Math.max(0, Math.min(this.cfg.vocab_size - 1, tokenIds[t]));
      x.set(this.tokEmb.subarray(tid * D, tid * D + D), t * D);
    }

    if (patch && patch.stageIdx === 0 && patch.pos < T) {
      x.set(patch.vector, patch.pos * D);
    }

    const trace: ForwardTrace | undefined = capture
      ? {
          residStages: [{ label: "Embed", data: new Float32Array(x) }],
          embedOut: new Float32Array(x),
          attnProbs: [],
          perHeadLastOut: [],
          mlpLastOut: [],
          finalResidLast: new Float32Array(D),
        }
      : undefined;

    for (let l = 0; l < L; l++) {
      const blk = this.blocks[l];

      // ---- Attention Sub-layer ----
      const xNorm1 = rmsNorm(x, blk.norm1, T, D, eps);
      const qFlat = matmulNT(xNorm1, blk.wQ, T, D, D);
      const kFlat = matmulNT(xNorm1, blk.wK, T, D, D);
      const vFlat = matmulNT(xNorm1, blk.wV, T, D, D);

      // Reshape [T, H, Dh] -> [H, T, Dh]
      const qH = new Float32Array(H * T * Dh);
      const kH = new Float32Array(H * T * Dh);
      const vH = new Float32Array(H * T * Dh);
      for (let t = 0; t < T; t++) {
        for (let h = 0; h < H; h++) {
          const src = t * D + h * Dh;
          const dst = (h * T + t) * Dh;
          for (let d = 0; d < Dh; d++) {
            qH[dst + d] = qFlat[src + d];
            kH[dst + d] = kFlat[src + d];
            vH[dst + d] = vFlat[src + d];
          }
        }
      }

      applyRoPEInPlace(qH, H, T, Dh, ropeBase);
      applyRoPEInPlace(kH, H, T, Dh, ropeBase);

      // Causal scaled dot-product attention per head
      const ctxH = new Float32Array(H * T * Dh);
      const layerProbs = capture ? new Float32Array(H * T * T) : null;
      const scores = new Float32Array(T);
      const ablateSet = ablateHeads?.get(l);

      for (let h = 0; h < H; h++) {
        const hOff = h * T * Dh;
        for (let qi = 0; qi < T; qi++) {
          const qRow = hOff + qi * Dh;
          let mx = -Infinity;
          for (let kj = 0; kj <= qi; kj++) {
            const kRow = hOff + kj * Dh;
            let dot = 0.0;
            for (let d = 0; d < Dh; d++) dot += qH[qRow + d] * kH[kRow + d];
            const sc = dot * scale;
            scores[kj] = sc;
            if (sc > mx) mx = sc;
          }
          let sum = 0.0;
          for (let kj = 0; kj <= qi; kj++) {
            const e = Math.exp(scores[kj] - mx);
            scores[kj] = e;
            sum += e;
          }
          const invSum = 1.0 / (sum || 1.0);
          const outRow = hOff + qi * Dh;
          for (let kj = 0; kj <= qi; kj++) {
            const w = scores[kj] * invSum;
            if (layerProbs) layerProbs[(h * T + qi) * T + kj] = w;
            const vRow = hOff + kj * Dh;
            for (let d = 0; d < Dh; d++) {
              ctxH[outRow + d] += w * vH[vRow + d];
            }
          }
        }

        // Apply live head ablation if requested
        if (ablateSet && ablateSet.has(h)) {
          if (ablationMode === "mean") {
            const meanVec = new Float32Array(Dh);
            for (let t = 0; t < T; t++) {
              const row = hOff + t * Dh;
              for (let d = 0; d < Dh; d++) meanVec[d] += ctxH[row + d];
            }
            for (let d = 0; d < Dh; d++) meanVec[d] /= T;
            for (let t = 0; t < T; t++) {
              ctxH.set(meanVec, hOff + t * Dh);
            }
          } else {
            ctxH.fill(0, hOff, hOff + T * Dh);
          }
        }
      }

      // Merge heads [H, T, Dh] -> [T, D]
      const merged = new Float32Array(T * D);
      for (let t = 0; t < T; t++) {
        for (let h = 0; h < H; h++) {
          const src = (h * T + t) * Dh;
          const dst = t * D + h * Dh;
          for (let d = 0; d < Dh; d++) merged[dst + d] = ctxH[src + d];
        }
      }

      // Capture exact per-head W_o slice contributions at last position for DLA
      if (trace) {
        trace.attnProbs.push(layerProbs!);
        const headsLast: Float32Array[] = [];
        const lastT = T - 1;
        for (let h = 0; h < H; h++) {
          const cOff = (h * T + lastT) * Dh;
          const hOut = new Float32Array(D);
          const colStart = h * Dh;
          for (let o = 0; o < D; o++) {
            const wRow = o * D + colStart;
            let acc = 0.0;
            for (let d = 0; d < Dh; d++) acc += ctxH[cOff + d] * blk.wO[wRow + d];
            hOut[o] = acc;
          }
          headsLast.push(hOut);
        }
        trace.perHeadLastOut.push(headsLast);
      }

      const attnOut = matmulNT(merged, blk.wO, T, D, D);
      const xPostAttn = new Float32Array(T * D);
      for (let i = 0; i < T * D; i++) xPostAttn[i] = x[i] + attnOut[i];

      if (trace) {
        trace.residStages.push({ label: `L${l}.attn`, data: new Float32Array(xPostAttn) });
      }

      // ---- SwiGLU MLP Sub-layer ----
      const xNorm2 = rmsNorm(xPostAttn, blk.norm2, T, D, eps);
      const gate = matmulNT(xNorm2, blk.wGate, T, D, FF);
      const up = matmulNT(xNorm2, blk.wUp, T, D, FF);
      const hidden = new Float32Array(T * FF);
      for (let i = 0; i < T * FF; i++) {
        const g = gate[i];
        const silu = g / (1.0 + Math.exp(-g));
        hidden[i] = silu * up[i];
      }
      const mlpOut = matmulNT(hidden, blk.wDown, T, FF, D);

      const xOut = new Float32Array(T * D);
      for (let i = 0; i < T * D; i++) xOut[i] = xPostAttn[i] + mlpOut[i];

      if (patch && patch.stageIdx === l + 1 && patch.pos < T) {
        xOut.set(patch.vector, patch.pos * D);
      }

      if (trace) {
        const lastOff = (T - 1) * D;
        trace.mlpLastOut.push(new Float32Array(mlpOut.subarray(lastOff, lastOff + D)));
        trace.residStages.push({ label: `L${l}.out`, data: new Float32Array(xOut) });
      }

      x = xOut;
    }

    if (trace) {
      trace.finalResidLast = new Float32Array(x.subarray((T - 1) * D, T * D));
    }

    const logits = this.projectToLogits(x, T);
    return { logits, trace };
  }
}

// ---- Full Glassbox Browser Runtime (Drop-In Replacement for Server API on Vercel) ----

export class GlassboxBrowserRuntime {
  private bundleCache: Map<string, BrowserModelBundle> = new Map();
  private currentModelId: "chat" | "induction" = "chat";
  private bundle!: BrowserModelBundle;
  private tokenizer!: BrowserBPETokenizer;
  private model!: BrowserGPTModel;
  private ready = false;

  async init(modelId: "chat" | "induction" = "chat"): Promise<void> {
    if (this.ready && this.currentModelId === modelId) return;
    let b = this.bundleCache.get(modelId);
    if (!b) {
      const url = modelId === "chat" ? "/models/glassbox-chat.json" : "/models/glassbox-induction.json";
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Failed to load model weights (${url}): HTTP ${res.status}`);
      b = (await res.json()) as BrowserModelBundle;
      this.bundleCache.set(modelId, b);
    }
    this.currentModelId = modelId;
    this.bundle = b;
    this.tokenizer = new BrowserBPETokenizer(b.tokenizer);
    this.model = new BrowserGPTModel(b.cfg, b.weights);
    this.ready = true;
  }

  getActiveModelId(): "chat" | "induction" {
    return this.currentModelId;
  }

  getStatus(): StatusResponse {
    return {
      status: { ...this.bundle.status, is_training: false },
      history: this.bundle.history,
      datasets: this.bundle.datasets,
      presets: this.bundle.presets,
    };
  }

  async switchModel(modelId: "chat" | "induction"): Promise<{ status: LabStatus; history: StepRecord[] }> {
    await this.init(modelId);
    return { status: { ...this.bundle.status, is_training: false }, history: this.bundle.history };
  }

  private sampleNextToken(
    rawLogits: Float32Array,
    temperature: number,
    topK: number,
    topP: number
  ): { nextId: number; prob: number; entropy: number; topCandidates: { id: number; display: string; prob: number }[] } {
    const { probs, entropy } = softmaxVec(rawLogits, temperature);
    const V = probs.length;
    const ranked: [number, number][] = [];
    for (let i = 0; i < V; i++) ranked.push([i, probs[i]]);
    ranked.sort((a, b) => b[1] - a[1]);

    const limited = topK > 0 ? ranked.slice(0, topK) : ranked;
    const filtered: [number, number][] = [];
    let cum = 0.0;
    for (const item of limited) {
      filtered.push(item);
      cum += item[1];
      if (cum >= topP && filtered.length >= 1) break;
    }

    let nextId: number;
    if (temperature <= 1e-4) {
      nextId = filtered[0][0];
    } else {
      const total = filtered.reduce((acc, x) => acc + x[1], 0);
      let r = Math.random() * total;
      nextId = filtered[filtered.length - 1][0];
      for (const [tid, p] of filtered) {
        r -= p;
        if (r <= 0) {
          nextId = tid;
          break;
        }
      }
    }

    return {
      nextId,
      prob: Math.round(probs[nextId] * 100000) / 100000,
      entropy: Math.round(entropy * 10000) / 10000,
      topCandidates: ranked.slice(0, 8).map(([tid, p]) => ({
        id: tid,
        display: this.tokenizer.tokenDisplay(tid),
        prob: Math.round(p * 100000) / 100000,
      })),
    };
  }

  /** Live streaming chat generator — yields each token cleanly to React for butter-smooth UI streaming! */
  async *chatStream(payload: {
    messages: ChatMessage[];
    max_new_tokens?: number;
    temperature?: number;
    top_k?: number;
    top_p?: number;
    ablate_heads?: [number, number][];
    ablation_mode?: "zero" | "mean";
  }): AsyncGenerator<{ prompt: string; token: GeneratedTokenStep; partialReply: string }, ChatResponse, void> {
    const isChat = this.bundle.status.dataset_id === "chat";
    let prompt: string;
    if (isChat) {
      const recent = payload.messages.slice(-4);
      prompt =
        recent
          .map((m) => `${m.role === "user" ? "User" : "Glassbox"}: ${m.content.trim()}`)
          .join("\n") + "\nGlassbox:";
    } else {
      const lastUser = [...payload.messages].reverse().find((m) => m.role === "user");
      prompt = lastUser?.content.trim() || this.bundle.status.default_prompt;
    }

    const ablateMap = new Map<number, Set<number>>();
    for (const [l, h] of payload.ablate_heads || []) {
      if (!ablateMap.has(l)) ablateMap.set(l, new Set());
      ablateMap.get(l)!.add(h);
    }

    const maxNew = Math.max(1, Math.min(140, payload.max_new_tokens ?? 100));
    const temperature = payload.temperature ?? 0.3;
    const topK = payload.top_k ?? 16;
    const topP = payload.top_p ?? 0.9;
    const seq = this.tokenizer.encode(prompt, true, false);
    const genIds: number[] = [];
    const stepsOut: GeneratedTokenStep[] = [];
    const V = this.model.cfg.vocab_size;

    for (let stepIdx = 0; stepIdx < maxNew; stepIdx++) {
      const ctx = seq.slice(-this.model.cfg.context_len);
      const { logits } = this.model.forward(ctx, {
        ablateHeads: ablateMap.size > 0 ? ablateMap : undefined,
        ablationMode: payload.ablation_mode || "zero",
      });
      const lastLogits = logits.subarray((ctx.length - 1) * V, ctx.length * V);
      const sampled = this.sampleNextToken(lastLogits, temperature, topK, topP);

      if (sampled.nextId === EOS_ID) break;
      seq.push(sampled.nextId);
      genIds.push(sampled.nextId);

      const stepObj: GeneratedTokenStep = {
        step: stepIdx,
        id: sampled.nextId,
        text: this.tokenizer.tokenText(sampled.nextId),
        display: this.tokenizer.tokenDisplay(sampled.nextId),
        prob: sampled.prob,
        entropy: sampled.entropy,
        top_candidates: sampled.topCandidates,
      };
      stepsOut.push(stepObj);

      const decoded = this.tokenizer.decode(genIds);
      if (isChat && decoded.includes("\nUser:")) {
        break;
      }

      yield { prompt, token: stepObj, partialReply: decoded.trimStart() };
      // Yield a micro-frame so browser paints smoothly at 60fps
      await new Promise((r) => setTimeout(r, 8));
    }

    let reply = this.tokenizer.decode(genIds);
    if (isChat && reply.includes("\nUser:")) {
      reply = reply.split("\nUser:", 1)[0];
    }
    return { prompt, reply: reply.trim(), tokens: stepsOut };
  }

  generate(payload: {
    prompt: string;
    max_new_tokens: number;
    temperature: number;
    top_k: number;
    top_p: number;
  }): GenerateResponse {
    const seq = this.tokenizer.encode(payload.prompt, true, false);
    const genIds: number[] = [];
    const tokens: GeneratedTokenStep[] = [];
    const V = this.model.cfg.vocab_size;

    for (let stepIdx = 0; stepIdx < Math.min(140, payload.max_new_tokens); stepIdx++) {
      const ctx = seq.slice(-this.model.cfg.context_len);
      const { logits } = this.model.forward(ctx);
      const lastLogits = logits.subarray((ctx.length - 1) * V, ctx.length * V);
      const s = this.sampleNextToken(lastLogits, payload.temperature, payload.top_k, payload.top_p);
      seq.push(s.nextId);
      genIds.push(s.nextId);
      tokens.push({
        step: stepIdx,
        id: s.nextId,
        text: this.tokenizer.tokenText(s.nextId),
        display: this.tokenizer.tokenDisplay(s.nextId),
        prob: s.prob,
        entropy: s.entropy,
        top_candidates: s.topCandidates,
      });
      if (s.nextId === EOS_ID) break;
    }

    const generated_text = this.tokenizer.decode(genIds);
    return {
      prompt: payload.prompt,
      generated_text,
      full_text: payload.prompt + generated_text,
      tokens,
    };
  }

  analyzeAttention(payload: {
    prompt: string;
    ablate_heads?: [number, number][];
    ablation_mode?: string;
  }): AttentionAnalysis {
    const ids = this.tokenizer.encode(payload.prompt, true, false).slice(0, this.model.cfg.context_len);
    const T = ids.length;
    const { n_layers: L, n_heads: H, vocab_size: V } = this.model.cfg;

    const { logits: baseLogits, trace } = this.model.forward(ids, { captureTrace: true });
    const lastBase = baseLogits.subarray((T - 1) * V, T * V);
    const { probs: baseProbs, entropy: baseEnt } = softmaxVec(lastBase, 1.0);

    const rankedBase = Array.from(baseProbs)
      .map((p, id) => ({ id, p }))
      .sort((a, b) => b.p - a.p);

    const topPreds: TokenPrediction[] = rankedBase.slice(0, 10).map(({ id, p }) => ({
      id,
      display: this.tokenizer.tokenDisplay(id),
      text: this.tokenizer.tokenText(id),
      prob: Math.round(p * 100000) / 100000,
      logit: Math.round(lastBase[id] * 10000) / 10000,
    }));

    const tokens: TokenInfo[] = ids.map((tid, pos) => ({
      pos,
      id: tid,
      display: this.tokenizer.tokenDisplay(tid),
      text: this.tokenizer.tokenText(tid),
    }));

    const layers: { layer: number; heads: HeadData[] }[] = [];
    for (let l = 0; l < L; l++) {
      const probsL = trace!.attnProbs[l];
      const heads: HeadData[] = [];
      for (let h = 0; h < H; h++) {
        const mat: number[][] = [];
        let prevSum = 0;
        let prevCnt = 0;
        let selfSum = 0;
        let bosSum = 0;
        let indSum = 0;
        let indOpp = 0;
        let normEntSum = 0;
        let normEntCnt = 0;

        for (let i = 0; i < T; i++) {
          const row: number[] = [];
          let rowEnt = 0;
          for (let j = 0; j < T; j++) {
            const w = j <= i ? probsL[(h * T + i) * T + j] : 0;
            row.push(Math.round(w * 10000) / 10000);
            if (j <= i && w > 1e-12) rowEnt -= w * Math.log(w);
          }
          mat.push(row);
          selfSum += row[i];
          if (i >= 1) {
            prevSum += row[i - 1];
            prevCnt++;
            bosSum += row[0];
            normEntSum += rowEnt / Math.log(i + 1);
            normEntCnt++;
          }
          if (i >= 2) {
            const cur = ids[i];
            let mass = 0;
            let hasMatch = false;
            for (let k = 0; k < i - 1; k++) {
              if (ids[k] === cur) {
                mass += row[k + 1];
                hasMatch = true;
              }
            }
            if (hasMatch) {
              indSum += mass;
              indOpp++;
            }
          }
        }

        const prevScore = prevCnt ? prevSum / prevCnt : 0;
        const selfScore = T ? selfSum / T : 0;
        const bosScore = prevCnt ? bosSum / prevCnt : 0;
        const indScore = indOpp ? indSum / indOpp : 0;
        const normEnt = normEntCnt ? normEntSum / normEntCnt : 0;

        let role = "Contextual";
        if (indOpp > 0 && indScore >= 0.28) role = "Induction";
        else if (prevScore >= 0.38) role = "Prev-Token";
        else if (bosScore >= 0.45) role = "BOS Sink";
        else if (selfScore >= 0.45) role = "Self-Attend";
        else if (normEnt <= 0.35) role = "Sharp Focus";
        else if (normEnt >= 0.82) role = "Broad Mix";

        heads.push({
          layer: l,
          head: h,
          matrix: mat,
          prev_token_score: Math.round(prevScore * 10000) / 10000,
          self_score: Math.round(selfScore * 10000) / 10000,
          bos_sink_score: Math.round(bosScore * 10000) / 10000,
          induction_score: Math.round(indScore * 10000) / 10000,
          induction_opportunities: indOpp,
          norm_entropy: Math.round(normEnt * 10000) / 10000,
          role,
        });
      }
      layers.push({ layer: l, heads });
    }

    let ablation: AttentionAnalysis["ablation"] = null;
    const ablateList = payload.ablate_heads || [];
    if (ablateList.length > 0) {
      const ablateMap = new Map<number, Set<number>>();
      for (const [l, h] of ablateList) {
        if (!ablateMap.has(l)) ablateMap.set(l, new Set());
        ablateMap.get(l)!.add(h);
      }
      const mode = payload.ablation_mode === "mean" ? "mean" : "zero";
      const { logits: ablLogits } = this.model.forward(ids, {
        ablateHeads: ablateMap,
        ablationMode: mode,
      });
      const lastAbl = ablLogits.subarray((T - 1) * V, T * V);
      const { probs: ablProbs, entropy: ablEnt } = softmaxVec(lastAbl, 1.0);

      let kl = 0;
      for (let i = 0; i < V; i++) {
        if (baseProbs[i] > 1e-12) {
          kl += baseProbs[i] * Math.log(baseProbs[i] / Math.max(ablProbs[i], 1e-12));
        }
      }

      const rankedAbl = Array.from(ablProbs)
        .map((p, id) => ({ id, p }))
        .sort((a, b) => b.p - a.p);

      const compIds: number[] = [];
      for (const x of [...rankedBase.slice(0, 6), ...rankedAbl.slice(0, 6)]) {
        if (!compIds.includes(x.id)) compIds.push(x.id);
      }

      const comparisons: AblationComparison[] = compIds.slice(0, 8).map((tid) => ({
        id: tid,
        display: this.tokenizer.tokenDisplay(tid),
        base_prob: Math.round(baseProbs[tid] * 100000) / 100000,
        ablated_prob: Math.round(ablProbs[tid] * 100000) / 100000,
        delta_prob: Math.round((ablProbs[tid] - baseProbs[tid]) * 100000) / 100000,
        delta_logit: Math.round((lastAbl[tid] - lastBase[tid]) * 10000) / 10000,
      }));

      ablation = {
        ablated_heads: ablateList,
        mode,
        kl_divergence: Math.round(kl * 100000) / 100000,
        base_entropy: Math.round(baseEnt * 10000) / 10000,
        ablated_entropy: Math.round(ablEnt * 10000) / 10000,
        top_predictions: rankedAbl.slice(0, 10).map(({ id, p }) => ({
          id,
          display: this.tokenizer.tokenDisplay(id),
          prob: Math.round(p * 100000) / 100000,
        })),
        comparisons,
      };
    }

    return {
      prompt: payload.prompt,
      tokens,
      n_layers: L,
      n_heads: H,
      baseline: { entropy: Math.round(baseEnt * 10000) / 10000, top_predictions: topPreds },
      layers,
      ablation,
    };
  }

  private resolveTokenId(text?: string): number | null {
    if (!text || !text.trim()) return null;
    const ids = this.tokenizer.encode(text, false, false);
    if (ids.length > 0) return ids[0];
    return null;
  }

  analyzeLogitLens(payload: {
    prompt: string;
    target_token?: string;
    foil_token?: string;
  }): LogitLensAnalysis {
    const ids = this.tokenizer.encode(payload.prompt, true, false).slice(0, this.model.cfg.context_len);
    const T = ids.length;
    const { d_model: D, n_layers: L, n_heads: H, vocab_size: V, rmsnorm_eps: eps } = this.model.cfg;

    const { logits: finalLogits, trace } = this.model.forward(ids, { captureTrace: true });

    // Precompute final distribution per position
    const finalProbsPerPos: Float32Array[] = [];
    const finalTop1PerPos: number[] = [];
    for (let t = 0; t < T; t++) {
      const { probs } = softmaxVec(finalLogits.subarray(t * V, (t + 1) * V), 1.0);
      finalProbsPerPos.push(probs);
      let best = 0;
      for (let v = 1; v < V; v++) if (probs[v] > probs[best]) best = v;
      finalTop1PerPos.push(best);
    }

    const stages: { stage: string; cells: LogitLensCell[] }[] = [];
    for (const st of trace!.residStages) {
      const stageLogits = this.model.projectToLogits(st.data, T);
      const cells: LogitLensCell[] = [];
      for (let t = 0; t < T; t++) {
        const { probs, entropy } = softmaxVec(stageLogits.subarray(t * V, (t + 1) * V), 1.0);
        const ranked = Array.from(probs)
          .map((p, id) => ({ id, p }))
          .sort((a, b) => b.p - a.p);
        const ft = finalTop1PerPos[t];
        const ftProb = probs[ft];
        let rank = 1;
        for (let i = 0; i < ranked.length; i++) {
          if (ranked[i].id === ft) {
            rank = i + 1;
            break;
          }
        }
        const fProbs = finalProbsPerPos[t];
        let kl = 0;
        for (let v = 0; v < V; v++) {
          if (fProbs[v] > 1e-12) kl += fProbs[v] * Math.log(fProbs[v] / Math.max(probs[v], 1e-12));
        }

        cells.push({
          pos: t,
          top1_id: ranked[0].id,
          top1_display: this.tokenizer.tokenDisplay(ranked[0].id),
          top1_prob: Math.round(ranked[0].p * 10000) / 10000,
          entropy: Math.round(entropy * 1000) / 1000,
          final_token_prob: Math.round(ftProb * 10000) / 10000,
          final_token_rank: rank,
          kl_to_final: Math.round(kl * 1000) / 1000,
          top6: ranked.slice(0, 6).map(({ id, p }) => ({
            id,
            display: this.tokenizer.tokenDisplay(id),
            prob: Math.round(p * 10000) / 10000,
          })),
        });
      }
      stages.push({ stage: st.label, cells });
    }

    // Exact Direct Logit Attribution at final position
    const lastLogits = finalLogits.subarray((T - 1) * V, T * V);
    const lastProbs = finalProbsPerPos[T - 1];
    const rankedFinal = Array.from(lastProbs)
      .map((p, id) => ({ id, p }))
      .sort((a, b) => b.p - a.p);

    const targetId = this.resolveTokenId(payload.target_token) ?? rankedFinal[0].id;
    const foilId = this.resolveTokenId(payload.foil_token);

    const residLast = trace!.finalResidLast;
    let ss = 0;
    for (let d = 0; d < D; d++) ss += residLast[d] * residLast[d];
    const invRms = 1.0 / Math.sqrt(ss / D + eps);

    // Effective unembedding vector: invRms * finalNorm * (W_U[target] - W_U[foil])
    const effDir = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      const wu =
        this.model.tokEmb[targetId * D + d] -
        (foilId !== null ? this.model.tokEmb[foilId * D + d] : 0);
      effDir[d] = wu * this.model.finalNorm[d] * invRms;
    }

    const dotEff = (vec: Float32Array): number => {
      let s = 0;
      for (let d = 0; d < D; d++) s += vec[d] * effDir[d];
      return s;
    };

    const components: DLAComponent[] = [];
    const embLast = trace!.embedOut.subarray((T - 1) * D, T * D);
    components.push({
      label: "Embed",
      kind: "embed",
      layer: -1,
      contrib: Math.round(dotEff(embLast) * 100000) / 100000,
    });

    for (let l = 0; l < L; l++) {
      for (let h = 0; h < H; h++) {
        components.push({
          label: `L${l}.H${h}`,
          kind: "head",
          layer: l,
          head: h,
          contrib: Math.round(dotEff(trace!.perHeadLastOut[l][h]) * 100000) / 100000,
        });
      }
      components.push({
        label: `L${l}.MLP`,
        kind: "mlp",
        layer: l,
        contrib: Math.round(dotEff(trace!.mlpLastOut[l]) * 100000) / 100000,
      });
    }

    const actualLogit = lastLogits[targetId] - (foilId !== null ? lastLogits[foilId] : 0);
    const sumContribs = components.reduce((a, c) => a + c.contrib, 0);

    return {
      prompt: payload.prompt,
      tokens: ids.map((tid, pos) => ({
        pos,
        id: tid,
        display: this.tokenizer.tokenDisplay(tid),
        text: this.tokenizer.tokenText(tid),
      })),
      stages,
      final_top_predictions: rankedFinal.slice(0, 8).map(({ id, p }) => ({
        id,
        display: this.tokenizer.tokenDisplay(id),
        prob: Math.round(p * 100000) / 100000,
        logit: Math.round(lastLogits[id] * 10000) / 10000,
      })),
      dla: {
        target_id: targetId,
        target_display: this.tokenizer.tokenDisplay(targetId),
        target_prob: Math.round(lastProbs[targetId] * 100000) / 100000,
        foil_id: foilId,
        foil_display: foilId !== null ? this.tokenizer.tokenDisplay(foilId) : null,
        actual_logit: Math.round(actualLogit * 100000) / 100000,
        sum_of_contribs: Math.round(sumContribs * 100000) / 100000,
        components,
      },
    };
  }

  runPatching(payload: {
    clean_prompt: string;
    corrupted_prompt: string;
    clean_target?: string;
    corrupted_target?: string;
  }): PatchingAnalysis {
    let cleanIds = this.tokenizer.encode(payload.clean_prompt, true, false).slice(0, this.model.cfg.context_len);
    let corrIds = this.tokenizer.encode(payload.corrupted_prompt, true, false).slice(0, this.model.cfg.context_len);
    const aligned = cleanIds.length === corrIds.length;
    const T = Math.min(cleanIds.length, corrIds.length);
    cleanIds = cleanIds.slice(0, T);
    corrIds = corrIds.slice(0, T);

    const { d_model: D, n_layers: L, vocab_size: V } = this.model.cfg;

    const { logits: cleanLogits, trace: cleanTrace } = this.model.forward(cleanIds, { captureTrace: true });
    const { logits: corrLogits } = this.model.forward(corrIds);

    const lastClean = cleanLogits.subarray((T - 1) * V, T * V);
    const lastCorr = corrLogits.subarray((T - 1) * V, T * V);
    const { probs: cleanProbs } = softmaxVec(lastClean, 1.0);
    const { probs: corrProbs } = softmaxVec(lastCorr, 1.0);

    let bestClean = 0;
    let bestCorr = 0;
    for (let v = 1; v < V; v++) {
      if (cleanProbs[v] > cleanProbs[bestClean]) bestClean = v;
      if (corrProbs[v] > corrProbs[bestCorr]) bestCorr = v;
    }
    const cleanTgt = this.resolveTokenId(payload.clean_target) ?? bestClean;
    let corrTgt = this.resolveTokenId(payload.corrupted_target) ?? bestCorr;
    if (corrTgt === cleanTgt) {
      let second = cleanTgt === 0 ? 1 : 0;
      for (let v = 0; v < V; v++) {
        if (v !== cleanTgt && corrProbs[v] > corrProbs[second]) second = v;
      }
      corrTgt = second;
    }

    const ldClean = lastClean[cleanTgt] - lastClean[corrTgt];
    const ldCorr = lastCorr[cleanTgt] - lastCorr[corrTgt];
    const denom = Math.abs(ldClean - ldCorr) > 1e-6 ? ldClean - ldCorr : 1.0;

    // Collect Clean stage vectors: stage 0 = Embed, stage 1..L = L{l}.out
    const cleanStageData: { label: string; data: Float32Array }[] = [
      { label: "Embed", data: cleanTrace!.embedOut },
    ];
    for (let l = 0; l < L; l++) {
      const found = cleanTrace!.residStages.find((s) => s.label === `L${l}.out`);
      cleanStageData.push({ label: `Block ${l}`, data: found!.data });
    }

    const rows: PatchingAnalysis["rows"] = [];
    const hotspots: PatchingAnalysis["top_hotspots"] = [];

    for (let sIdx = 0; sIdx <= L; sIdx++) {
      const stageLabel = cleanStageData[sIdx].label;
      const stageMat = cleanStageData[sIdx].data;
      const recRow: number[] = [];
      const ldRow: number[] = [];

      for (let pos = 0; pos < T; pos++) {
        const vec = stageMat.subarray(pos * D, (pos + 1) * D);
        const { logits: pLogits } = this.model.forward(corrIds, {
          patch: { stageIdx: sIdx, pos, vector: vec },
        });
        const pLast = pLogits.subarray((T - 1) * V, T * V);
        const ldPatched = pLast[cleanTgt] - pLast[corrTgt];
        const rec = Math.max(-1.5, Math.min(1.5, (ldPatched - ldCorr) / denom));
        recRow.push(Math.round(rec * 10000) / 10000);
        ldRow.push(Math.round(ldPatched * 10000) / 10000);
        hotspots.push({
          stage: stageLabel,
          pos,
          clean_tok: this.tokenizer.tokenDisplay(cleanIds[pos]),
          corrupt_tok: this.tokenizer.tokenDisplay(corrIds[pos]),
          recovery: Math.round(rec * 10000) / 10000,
          logit_diff: Math.round(ldPatched * 10000) / 10000,
        });
      }
      rows.push({ stage_idx: sIdx, stage_label: stageLabel, recovery: recRow, logit_diff: ldRow });
    }

    hotspots.sort((a, b) => b.recovery - a.recovery);

    return {
      length_aligned: aligned,
      clean_tokens: cleanIds.map((id, pos) => ({ pos, id, display: this.tokenizer.tokenDisplay(id) })),
      corrupt_tokens: corrIds.map((id, pos) => ({ pos, id, display: this.tokenizer.tokenDisplay(id) })),
      clean_target: {
        id: cleanTgt,
        display: this.tokenizer.tokenDisplay(cleanTgt),
        clean_prob: Math.round(cleanProbs[cleanTgt] * 100000) / 100000,
        corrupt_prob: Math.round(corrProbs[cleanTgt] * 100000) / 100000,
      },
      corrupted_target: {
        id: corrTgt,
        display: this.tokenizer.tokenDisplay(corrTgt),
        clean_prob: Math.round(cleanProbs[corrTgt] * 100000) / 100000,
        corrupt_prob: Math.round(corrProbs[corrTgt] * 100000) / 100000,
      },
      logit_diff_clean: Math.round(ldClean * 10000) / 10000,
      logit_diff_corrupt: Math.round(ldCorr * 10000) / 10000,
      rows,
      top_hotspots: hotspots.slice(0, 10),
    };
  }

  inspectBPE(text: string): BPEInspectResponse {
    return {
      vocab_size: this.tokenizer.vocabSize,
      bpe_stats: this.bundle.status.bpe_stats as BPEStats,
      inspection: this.tokenizer.inspectEncode(text),
      merges: this.tokenizer.getMergeTable(100),
    };
  }

  getVerifySnapshot(): EngineVerifyResponse {
    return this.bundle.verify_snapshot;
  }

  getAutogradDagSnapshot(): AutogradDagResponse {
    return this.bundle.autograd_dag_snapshot;
  }

  /** Run a live browser JS cache-tiled vs naive Float32 GEMM benchmark right on the user's device! */
  runBrowserMatmulBenchmark(sizes: number[] = [64, 128, 192]): { results: MatmulBenchRow[] } {
    const results: MatmulBenchRow[] = [];
    for (const N of sizes) {
      const A = new Float32Array(N * N);
      const B = new Float32Array(N * N);
      for (let i = 0; i < N * N; i++) {
        A[i] = ((i * 2654435761) & 255) / 255 - 0.5;
        B[i] = (((i + 17) * 1597334677) & 255) / 255 - 0.5;
      }
      const Cnaive = new Float32Array(N * N);
      const Ctiled = new Float32Array(N * N);

      const t0 = performance.now();
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          let s = 0;
          for (let k = 0; k < N; k++) s += A[i * N + k] * B[k * N + j];
          Cnaive[i * N + j] = s;
        }
      }
      const naiveMs = Math.max(0.05, performance.now() - t0);

      // IKJ cache-friendly blocked GEMM
      const BS = 32;
      const t1 = performance.now();
      for (let ii = 0; ii < N; ii += BS) {
        const iEnd = Math.min(N, ii + BS);
        for (let kk = 0; kk < N; kk += BS) {
          const kEnd = Math.min(N, kk + BS);
          for (let jj = 0; jj < N; jj += BS) {
            const jEnd = Math.min(N, jj + BS);
            for (let i = ii; i < iEnd; i++) {
              const cRow = i * N;
              const aRow = i * N;
              for (let k = kk; k < kEnd; k++) {
                const aik = A[aRow + k];
                const bRow = k * N;
                for (let j = jj; j < jEnd; j++) {
                  Ctiled[cRow + j] += aik * B[bRow + j];
                }
              }
            }
          }
        }
      }
      const tiledMs = Math.max(0.05, performance.now() - t1);

      let maxErr = 0;
      for (let i = 0; i < N * N; i++) {
        const d = Math.abs(Cnaive[i] - Ctiled[i]);
        if (d > maxErr) maxErr = d;
      }
      const flops = 2 * N * N * N;
      results.push({
        size: N,
        naive_ms: Math.round(naiveMs * 100) / 100,
        tiled_ms: Math.round(tiledMs * 100) / 100,
        naive_gflops: Math.round((flops / (naiveMs * 1e6)) * 100) / 100,
        tiled_gflops: Math.round((flops / (tiledMs * 1e6)) * 100) / 100,
        speedup: Math.round((naiveMs / tiledMs) * 100) / 100,
        max_abs_error: Number(maxErr.toExponential(2)),
      });
    }
    return { results };
  }
}

export const browserEngine = new GlassboxBrowserRuntime();
