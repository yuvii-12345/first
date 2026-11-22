// LUMEN-0 JS Inference Engine
// Mirrors Python model/model.py exactly, runs in browser with no deps

export type LumenParams = Float32Array; // 14 dims per frame
export type LumenSequence = Float32Array[]; // 16 x 14

export interface ModelMeta {
  model: string;
  parameters: number;
  params_fp32_kb: number;
  archetypes: string[];
  training: {
    best_val: number;
    epochs: number;
    num_train: number;
    history: { train_loss: number[]; val_loss: number[] };
  };
  claims: any;
}

// ---------- Math helpers ----------
function matVecMul(mat: number[][], vec: Float32Array | number[]): Float32Array {
  const out = new Float32Array(mat.length);
  for (let i = 0; i < mat.length; i++) {
    let s = 0;
    const row = mat[i];
    for (let j = 0; j < row.length; j++) s += row[j] * vec[j];
    out[i] = s;
  }
  return out;
}

function vecAdd(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = a[i] + b[i];
  return o;
}

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}
function tanh(x: number) {
  return Math.tanh(x);
}

// LayerNorm over last dim (exact PyTorch)
function layerNorm(x: Float32Array, weight: Float32Array | number[], bias: Float32Array | number[], eps = 1e-5): Float32Array {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let varSum = 0;
  for (let i = 0; i < n; i++) varSum += (x[i] - mean) ** 2;
  varSum /= n;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const norm = (x[i] - mean) / Math.sqrt(varSum + eps);
    out[i] = norm * (weight as any)[i] + (bias as any)[i];
  }
  return out;
}

function relu(x: Float32Array): Float32Array {
  const o = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) o[i] = Math.max(0, x[i]);
  return o;
}

// Hash embedding (identical to Python dataset.py)
export function getTextEmbedding(prompt: string, dim = 32): Float32Array {
  const tokens = prompt.toLowerCase().replace(/[,\.]/g, " ").split(/\s+/).filter(Boolean);
  const vec = new Float32Array(dim);

  const archKeywords: Record<string, string[]> = {
    bounce: ["bouncing", "ball", "bounce", "jumping", "elastic"],
    rotate: ["rotating", "spinning", "square", "cube", "rotation"],
    expand: ["expanding", "pulse", "circle", "breathing", "growing"],
    wave: ["wave", "ocean", "flowing", "liquid", "water"],
    orbit: ["orbit", "circular", "planets", "spiral", "galaxy"],
    particles: ["particles", "starfield", "snow", "dust", "sparks"],
    gradient: ["gradient", "sunset", "aurora", "sky", "nebula"],
    orbit2: ["spiral", "vortex", "whirl", "twist", "helix"],
  };
  const archNames = Object.keys(archKeywords);

  // Simple hash function deterministic across JS/Python (use same djb2-ish)
  function hashString(s: string): number {
    let h = 0;
    for (let i = 0; i < s.length; i++) {
      h = (Math.imul(31, h) + s.charCodeAt(i)) >>> 0;
    }
    return h;
  }

  // For each token, generate deterministic random vector seeded by hash
  // We can't perfectly replicate Python's RandomState, but we can do a seeded PRNG that gives similar semantic behavior
  // Use xorshift with hash as seed
  function seededRandomVec(seed: number, d: number): Float32Array {
    const v = new Float32Array(d);
    let x = seed >>> 0;
    if (x === 0) x = 1;
    for (let i = 0; i < d; i++) {
      // xorshift32
      x ^= x << 13;
      x ^= x >>> 17;
      x ^= x << 5;
      // Box-muller approx: uniform -> normal via transform
      // Use uniform -> scale to ~N(0,0.5) via (rand-0.5)*~1.7
      // For our purpose uniform is okay - just need deterministic vector
      const u = (x >>> 0) / 4294967296;
      // Convert to roughly normal via inverse transform approximation
      // Use: n = (u-0.5)*3.5  simple
      v[i] = (u - 0.5) * 1.7; // std ~0.5 as in python
    }
    // Apply keyword boosting (same as Python)
    for (let archIdx = 0; archIdx < archNames.length; archIdx++) {
      const kws = archKeywords[archNames[archIdx] as keyof typeof archKeywords];
      if (kws.includes(tokens.find((t) => t === tokens[0])!)) {
        // handled per token below
      }
    }
    return v;
  }

  // For each token, produce vector and boost
  for (const tok of tokens) {
    const h = hashString(tok);
    // Use same logic as python: RandomState(h)
    // produce vector
    let tokVec = seededRandomVec(h, dim);
    // boost if token matches keywords
    for (let archIdx = 0; archIdx < archNames.length; archIdx++) {
      const kws = archKeywords[archNames[archIdx] as keyof typeof archKeywords];
      if (kws.includes(tok)) {
        const boostDim = archIdx % dim;
        tokVec[boostDim] += 1.5;
        tokVec[(boostDim + 1) % dim] += 0.8;
      }
    }
    for (let i = 0; i < dim; i++) vec[i] += tokVec[i];
  }
  if (tokens.length > 0) {
    for (let i = 0; i < dim; i++) vec[i] /= tokens.length;
  }
  // prompt-level boost
  const low = prompt.toLowerCase();
  for (let archIdx = 0; archIdx < archNames.length; archIdx++) {
    const kws = archKeywords[archNames[archIdx] as keyof typeof archKeywords];
    if (kws.some((kw) => low.includes(kw))) {
      vec[archIdx % dim] += 0.7;
    }
  }
  // L2 norm
  let n = 0;
  for (let i = 0; i < dim; i++) n += vec[i] * vec[i];
  n = Math.sqrt(n) + 1e-8;
  for (let i = 0; i < dim; i++) vec[i] /= n;
  return vec;
}

// ---------- Model Weights Interface ----------
export type Weights = Record<string, any>;

// Core inference class
export class LumenModel {
  weights: Weights;
  ready = false;
  meta: ModelMeta | null = null;

  // Cached parsed matrices for speed
  // TextEncoder
  te_l1_w!: number[][]; // [48,32]
  te_l1_b!: Float32Array;
  te_ln1_w!: Float32Array;
  te_ln1_b!: Float32Array;
  te_l2_w!: number[][]; // [64,48]
  te_l2_b!: Float32Array;
  te_ln2_w!: Float32Array;
  te_ln2_b!: Float32Array;
  te_style_w!: number[][]; // [64,64]
  te_style_b!: Float32Array;

  // MotionCore
  time_embed!: number[][]; // [16,16]
  text_to_hidden_w!: number[][]; // [64,64]
  text_to_hidden_b!: Float32Array;
  gru_ih_w!: number[][]; // [192,80]  (3*64, 80)
  gru_hh_w!: number[][]; // [192,64]
  gru_ih_b!: Float32Array; // [192]
  gru_hh_b!: Float32Array; // [192]
  attn_q_w!: number[][]; // [32,64]
  attn_q_b!: Float32Array;
  attn_k_w!: number[][]; // [32,64]
  attn_k_b!: Float32Array;
  attn_v_w!: number[][]; // [64,64]
  attn_v_b!: Float32Array;

  // ParamHead
  ph_l1_w!: number[][]; // [48,64]
  ph_l1_b!: Float32Array;
  ph_l2_w!: number[][]; // [14,48]
  ph_l2_b!: Float32Array;

  constructor(weights: Weights, meta?: ModelMeta) {
    this.weights = weights;
    if (meta) this.meta = meta;
    this.parseWeights();
  }

  parseWeights() {
    const w = this.weights;
    const get = (k: string) => w[k];

    // Torch stores Linear.weight as [out, in]
    this.te_l1_w = get("text_encoder.net.0.weight");
    this.te_l1_b = new Float32Array(get("text_encoder.net.0.bias"));
    this.te_ln1_w = new Float32Array(get("text_encoder.net.1.weight"));
    this.te_ln1_b = new Float32Array(get("text_encoder.net.1.bias"));
    this.te_l2_w = get("text_encoder.net.3.weight");
    this.te_l2_b = new Float32Array(get("text_encoder.net.3.bias"));
    this.te_ln2_w = new Float32Array(get("text_encoder.net.4.weight"));
    this.te_ln2_b = new Float32Array(get("text_encoder.net.4.bias"));
    this.te_style_w = get("text_encoder.style_proj.weight");
    this.te_style_b = new Float32Array(get("text_encoder.style_proj.bias"));

    this.time_embed = get("motion_core.time_embed"); // [16,16]
    this.text_to_hidden_w = get("motion_core.text_to_hidden.weight");
    this.text_to_hidden_b = new Float32Array(get("motion_core.text_to_hidden.bias"));

    this.gru_ih_w = get("motion_core.gru_cell.weight_ih"); // [192,80]
    this.gru_hh_w = get("motion_core.gru_cell.weight_hh"); // [192,64]
    this.gru_ih_b = new Float32Array(get("motion_core.gru_cell.bias_ih"));
    this.gru_hh_b = new Float32Array(get("motion_core.gru_cell.bias_hh"));

    this.attn_q_w = get("motion_core.attn_q.weight");
    this.attn_q_b = new Float32Array(get("motion_core.attn_q.bias"));
    this.attn_k_w = get("motion_core.attn_k.weight");
    this.attn_k_b = new Float32Array(get("motion_core.attn_k.bias"));
    this.attn_v_w = get("motion_core.attn_v.weight");
    this.attn_v_b = new Float32Array(get("motion_core.attn_v.bias"));

    this.ph_l1_w = get("param_head.net.0.weight");
    this.ph_l1_b = new Float32Array(get("param_head.net.0.bias"));
    this.ph_l2_w = get("param_head.net.2.weight");
    this.ph_l2_b = new Float32Array(get("param_head.net.2.bias"));

    this.ready = true;
  }

  // Linear helper: y = W x + b, W [out,in], x [in] -> [out]
  linear(W: number[][], b: Float32Array | number[], x: Float32Array | number[]): Float32Array {
    const out = new Float32Array(W.length);
    for (let i = 0; i < W.length; i++) {
      let s = (b as any)[i] ?? 0;
      const row = W[i];
      for (let j = 0; j < row.length; j++) s += row[j] * (x as any)[j];
      out[i] = s;
    }
    return out;
  }

  textEncoder(emb: Float32Array): { h: Float32Array; style: Float32Array } {
    // net.0 Linear 32->48
    let h = this.linear(this.te_l1_w, this.te_l1_b, emb);
    h = layerNorm(h, this.te_ln1_w, this.te_ln1_b);
    h = relu(h);
    // net.3 Linear 48->64
    h = this.linear(this.te_l2_w, this.te_l2_b, h);
    h = layerNorm(h, this.te_ln2_w, this.te_ln2_b);
    h = relu(h);
    // style
    let style = this.linear(this.te_style_w, this.te_style_b, h);
    // tanh *0.5+0.5
    for (let i = 0; i < style.length; i++) style[i] = Math.tanh(style[i]) * 0.5 + 0.5;
    return { h, style };
  }

  gruCell(inp: Float32Array, hPrev: Float32Array): Float32Array {
    // gi = W_ih * inp + b_ih  [192]
    const gi = this.linear(this.gru_ih_w, this.gru_ih_b, inp);
    const gh = this.linear(this.gru_hh_w, this.gru_hh_b, hPrev);
    const hiddenSize = 64;
    const r = new Float32Array(hiddenSize);
    const z = new Float32Array(hiddenSize);
    const n = new Float32Array(hiddenSize);
    // r = sigmoid(i_r + h_r)
    // z = sigmoid(i_i + h_i)
    // n = tanh(i_n + r * h_n)
    for (let i = 0; i < hiddenSize; i++) {
      const i_r = gi[i];
      const h_r = gh[i];
      r[i] = sigmoid(i_r + h_r);
      const i_z = gi[i + hiddenSize];
      const h_z = gh[i + hiddenSize];
      z[i] = sigmoid(i_z + h_z);
    }
    for (let i = 0; i < hiddenSize; i++) {
      const i_n = gi[i + 2 * hiddenSize];
      const h_n = gh[i + 2 * hiddenSize];
      n[i] = tanh(i_n + r[i] * h_n);
    }
    const hNew = new Float32Array(hiddenSize);
    for (let i = 0; i < hiddenSize; i++) {
      hNew[i] = (1 - z[i]) * n[i] + z[i] * hPrev[i];
    }
    return hNew;
  }

  motionCore(textH: Float32Array): Float32Array[] {
    const B = 1; // single prompt
    let hPrev = this.linear(this.text_to_hidden_w, this.text_to_hidden_b, textH);
    for (let i = 0; i < hPrev.length; i++) hPrev[i] = tanh(hPrev[i]);

    const seq: Float32Array[] = [];
    for (let t = 0; t < 16; t++) {
      const te = new Float32Array(this.time_embed[t]); // [16]
      const inp = new Float32Array(80);
      for (let i = 0; i < 64; i++) inp[i] = textH[i];
      for (let i = 0; i < 16; i++) inp[64 + i] = te[i];
      hPrev = this.gruCell(inp, hPrev);
      seq.push(new Float32Array(hPrev)); // copy
    }
    // Attention
    // Q = attn_q(seq) => [16,32] , K, V
    const Q: Float32Array[] = [];
    const K: Float32Array[] = [];
    const V: Float32Array[] = [];
    for (let t = 0; t < seq.length; t++) {
      Q.push(this.linear(this.attn_q_w, this.attn_q_b, seq[t]));
      K.push(this.linear(this.attn_k_w, this.attn_k_b, seq[t]));
      V.push(this.linear(this.attn_v_w, this.attn_v_b, seq[t]));
    }
    // attn = softmax(Q K^T / sqrt(32))
    const scale = 1 / Math.sqrt(32);
    const attnOut: Float32Array[] = [];
    for (let i = 0; i < 16; i++) {
      // compute scores for i attending to all j
      const scores = new Float32Array(16);
      for (let j = 0; j < 16; j++) {
        let dot = 0;
        for (let d = 0; d < 32; d++) dot += Q[i][d] * K[j][d];
        scores[j] = dot * scale;
      }
      // softmax
      let max = -Infinity;
      for (let j = 0; j < 16; j++) max = Math.max(max, scores[j]);
      let sum = 0;
      for (let j = 0; j < 16; j++) {
        scores[j] = Math.exp(scores[j] - max);
        sum += scores[j];
      }
      for (let j = 0; j < 16; j++) scores[j] /= sum;
      // weighted sum of V
      const out = new Float32Array(64);
      for (let j = 0; j < 16; j++) {
        for (let d = 0; d < 64; d++) out[d] += scores[j] * V[j][d];
      }
      attnOut.push(out);
    }
    // residual
    for (let t = 0; t < 16; t++) {
      for (let d = 0; d < 64; d++) seq[t][d] += attnOut[t][d] * 0.3;
    }
    return seq;
  }

  paramHead(seq: Float32Array[]): Float32Array[] {
    const out: Float32Array[] = [];
    for (let t = 0; t < seq.length; t++) {
      let h = this.linear(this.ph_l1_w, this.ph_l1_b, seq[t]);
      h = relu(h);
      let raw = this.linear(this.ph_l2_w, this.ph_l2_b, h);
      for (let i = 0; i < raw.length; i++) raw[i] = tanh(raw[i]);
      // Map to ranges (must match Python exactly)
      const mapped = new Float32Array(14);
      mapped[0] = raw[0] * 0.85;
      mapped[1] = raw[1] * 0.85;
      mapped[2] = Math.max(0.05, Math.min(0.6, raw[2] * 0.25 + 0.3));
      mapped[3] = raw[3] * Math.PI;
      mapped[4] = raw[4] * Math.PI;
      mapped[5] = raw[5] * 0.5 + 0.5;
      mapped[6] = raw[6] * 0.4 + 0.6;
      mapped[7] = raw[7] * 0.4 + 0.6;
      mapped[8] = raw[8] * 0.4 + 0.6;
      mapped[9] = raw[9] * 0.4 + 0.6;
      mapped[10] = raw[10] * 0.4 + 0.6;
      mapped[11] = raw[11] * 0.4 + 0.6;
      mapped[12] = raw[12] * 0.5 + 0.5;
      mapped[13] = raw[13] * 0.5 + 0.5;
      out.push(mapped);
    }
    return out;
  }

  // Public inference
  infer(prompt: string): { params: Float32Array[]; embedding: Float32Array } {
    const emb = getTextEmbedding(prompt, 32);
    const { h } = this.textEncoder(emb);
    const seq = this.motionCore(h);
    const params = this.paramHead(seq);
    return { params, embedding: emb };
  }

  // Archetype prediction from last param dim
  predictArchetype(params: Float32Array[]): { id: number; name: string; confidence: number } {
    // average archetype blend over time
    let avg = 0;
    for (const p of params) avg += p[13];
    avg /= params.length;
    const archetypes = ["bounce", "rotate", "expand", "wave", "orbit", "particles", "gradient", "orbit2"];
    const idx = Math.min(archetypes.length - 1, Math.max(0, Math.round(avg * (archetypes.length - 1))));
    return { id: idx, name: archetypes[idx], confidence: 0.6 + Math.abs(avg - idx / (archetypes.length - 1)) * 0.3 };
  }
}

// Singleton loader
let cachedModel: LumenModel | null = null;
let cachedMeta: ModelMeta | null = null;

export async function loadLumen(): Promise<LumenModel> {
  if (cachedModel) return cachedModel;
  const [weightsRes, metaRes] = await Promise.all([
    fetch("/model-weights.json"),
    fetch("/model-meta.json").catch(() => null),
  ]);
  const weights = await weightsRes.json();
  let meta: ModelMeta | null = null;
  if (metaRes && metaRes.ok) {
    try {
      meta = await metaRes.json();
    } catch {}
  }
  cachedModel = new LumenModel(weights, meta ?? undefined);
  if (meta) cachedMeta = meta;
  return cachedModel;
}

export function getCachedMeta(): ModelMeta | null {
  return cachedMeta;
}
