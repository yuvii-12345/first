export interface GPTConfig {
  vocab_size: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  n_kv_heads?: number | null;
  d_ff: number;
  context_len: number;
  rope_base: number;
  rmsnorm_eps: number;
  tie_weights: boolean;
  dtype: string;
}

export interface BPEStats {
  vocab_size: number;
  num_merges: number;
  corpus_bytes: number;
  corpus_tokens: number;
  compression_ratio: number;
}

export interface LayerGradStat {
  component: string;
  kind: string;
  grad_norm: number;
  param_norm: number;
  update_ratio: number;
  log10_update_ratio: number;
}

export interface StepRecord {
  step: number;
  loss: number;
  val_loss: number | null;
  perplexity: number;
  lr: number;
  grad_norm: number;
  was_clipped: boolean;
  grad_health: string;
  step_ms: number;
  tokens_per_sec: number;
  gflops: number;
  total_tokens: number;
  layer_grads: LayerGradStat[];
  latest_sample: string;
}

export interface LabStatus {
  is_training: boolean;
  step: number;
  target_steps: number;
  dataset_id: string;
  preset_id: string;
  config: GPTConfig;
  num_parameters: number;
  bpe_stats: BPEStats;
  train_tokens: number;
  val_tokens: number;
  total_tokens_seen: number;
  lr: number;
  batch_size: number;
  grad_clip: number;
  grad_health: string;
  last_loss: number | null;
  last_val_loss: number | null;
  last_perplexity: number | null;
  last_grad_norm: number | null;
  last_step_ms: number;
  tokens_per_sec: number;
  live_gflops: number;
  cumulative_gflops: number;
  total_matmul_calls: number;
  latest_sample: string;
  latest_layer_grads: LayerGradStat[];
  default_prompt: string;
}

export interface PatchingPreset {
  name: string;
  clean_prompt: string;
  corrupted_prompt: string;
  clean_target: string;
  corrupted_target: string;
}

export interface DatasetMeta {
  id: string;
  name: string;
  description: string;
  chars: number;
  default_prompt: string;
  sample_prompts: string[];
  patching_presets: PatchingPreset[];
}

export interface ModelPreset {
  id: string;
  name: string;
  description: string;
  n_layers: number;
  n_heads: number;
  n_kv_heads?: number | null;
  d_model: number;
  d_ff: number;
  context_len: number;
  vocab_size: number;
}

export interface StatusResponse {
  status: LabStatus;
  history: StepRecord[];
  datasets: DatasetMeta[];
  presets: ModelPreset[];
}

export interface TokenInfo {
  pos: number;
  id: number;
  display: string;
  text: string;
}

export interface TokenPrediction {
  id: number;
  display: string;
  text?: string;
  prob: number;
  logit?: number;
}

export interface HeadData {
  layer: number;
  head: number;
  matrix: number[][];
  prev_token_score: number;
  self_score: number;
  bos_sink_score: number;
  induction_score: number;
  induction_opportunities: number;
  norm_entropy: number;
  role: string;
}

export interface AblationComparison {
  id: number;
  display: string;
  base_prob: number;
  ablated_prob: number;
  delta_prob: number;
  delta_logit: number;
}

export interface AttentionAnalysis {
  prompt: string;
  tokens: TokenInfo[];
  n_layers: number;
  n_heads: number;
  baseline: {
    entropy: number;
    top_predictions: TokenPrediction[];
  };
  layers: { layer: number; heads: HeadData[] }[];
  ablation: {
    ablated_heads: [number, number][];
    mode: string;
    kl_divergence: number;
    base_entropy: number;
    ablated_entropy: number;
    top_predictions: TokenPrediction[];
    comparisons: AblationComparison[];
  } | null;
}

export interface LogitLensCell {
  pos: number;
  top1_id: number;
  top1_display: string;
  top1_prob: number;
  entropy: number;
  final_token_prob: number;
  final_token_rank: number;
  kl_to_final: number;
  top6: { id: number; display: string; prob: number }[];
}

export interface DLAComponent {
  label: string;
  kind: "embed" | "head" | "mlp";
  layer: number;
  head?: number;
  contrib: number;
}

export interface LogitLensAnalysis {
  prompt: string;
  tokens: TokenInfo[];
  stages: { stage: string; cells: LogitLensCell[] }[];
  final_top_predictions: TokenPrediction[];
  dla: {
    target_id: number;
    target_display: string;
    target_prob: number;
    foil_id: number | null;
    foil_display: string | null;
    actual_logit: number;
    sum_of_contribs: number;
    components: DLAComponent[];
  };
}

export interface PatchingAnalysis {
  length_aligned: boolean;
  clean_tokens: { pos: number; id: number; display: string }[];
  corrupt_tokens: { pos: number; id: number; display: string }[];
  clean_target: { id: number; display: string; clean_prob: number; corrupt_prob: number };
  corrupted_target: { id: number; display: string; clean_prob: number; corrupt_prob: number };
  logit_diff_clean: number;
  logit_diff_corrupt: number;
  rows: { stage_idx: number; stage_label: string; recovery: number[]; logit_diff: number[] }[];
  top_hotspots: {
    stage: string;
    pos: number;
    clean_tok: string;
    corrupt_tok: string;
    recovery: number;
    logit_diff: number;
  }[];
}

export interface GeneratedTokenStep {
  step: number;
  id: number;
  text: string;
  display: string;
  prob: number;
  entropy: number;
  top_candidates: { id: number; display: string; prob: number }[];
}

export interface GenerateResponse {
  prompt: string;
  generated_text: string;
  full_text: string;
  tokens: GeneratedTokenStep[];
}

export interface BPEInspectResponse {
  vocab_size: number;
  bpe_stats: BPEStats;
  inspection: {
    text: string;
    num_bytes: number;
    num_tokens: number;
    compression_ratio: number;
    tokens: {
      id: number;
      text: string;
      display: string;
      bytes_hex: string;
      byte_len: number;
      is_merged: boolean;
    }[];
    chunk_traces: {
      chunk: string;
      initial_bytes: string[];
      steps: {
        rank: number;
        pair: [string, string];
        merged: string;
        new_id: number;
        tokens_after: string[];
      }[];
    }[];
  };
  merges: {
    rank: number;
    new_id: number;
    left_id: number;
    right_id: number;
    left: string;
    right: string;
    merged: string;
    count: number;
  }[];
}

export interface AutogradNode {
  id: number;
  op: string;
  name: string;
  shape: number[];
  numel: number;
  requires_grad: boolean;
  is_leaf: boolean;
  data_norm: number | null;
  grad_norm: number | null;
}

export interface AutogradDagResponse {
  loss: number;
  seq_len: number;
  num_nodes: number;
  num_edges: number;
  graph: {
    root_id: number;
    nodes: AutogradNode[];
    edges: { source: number; target: number }[];
  };
}

export interface EngineVerifyResponse {
  all_passed: boolean;
  duration_ms: number;
  gradchecks: {
    op: string;
    passed: boolean;
    checked_elements: number;
    max_abs_error: number;
    max_rel_error: number;
    cosine_similarity: number;
  }[];
  flash_equivalence: {
    passed: boolean;
    shape: number[];
    causal: boolean;
    dtype: string;
    max_diff_output: number;
    max_diff_probs: number;
    max_diff_dQ: number;
    max_diff_dK: number;
    max_diff_dV: number;
  }[];
}

export interface MatmulBenchRow {
  size: number;
  naive_ms: number;
  tiled_ms: number;
  naive_gflops: number;
  tiled_gflops: number;
  speedup: number;
  max_abs_error: number;
}

export interface KVCacheBenchResult {
  prompt_tokens: number;
  decode_tokens: number;
  full_ms: number;
  cached_ms: number;
  speedup: number;
  full_tok_s: number;
  cached_tok_s: number;
  max_abs_diff: number;
  n_heads: number;
  n_kv_heads: number;
  kv_compression: number;
}

export interface LLM100MProof {
  model_name: string;
  exact_params: number;
  config: {
    vocab_size: number;
    d_model: number;
    n_layers: number;
    n_heads: number;
    n_kv_heads: number;
    head_dim: number;
    d_ff: number;
    context_len: number;
    tie_weights: boolean;
    rope_base: number;
  };
  param_breakdown: {
    tok_emb_tied: number;
    attn_per_layer: number;
    swiglu_per_layer: number;
    norms_total: number;
    all_12_layers: number;
  };
  verified_proofs: {
    gqa_flash_attn_gradcheck: {
      passed: boolean;
      max_abs_error: number;
      cosine_similarity: number;
      checked_elements: number;
    };
    gqa_2l_transformer_gradcheck: {
      passed: boolean;
      max_abs_error: number;
      cosine_similarity: number;
      checked_elements: number;
    };
    kv_cache_max_diff: number;
    cpp17_vs_python_loss_diff: number;
    cpp17_vs_python_gnorm_diff: number;
    gbx_402mb_python_load_sec: number;
    gbx_checkpoint_bytes: number;
  };
  live_100m_cpu_smoke_steps: {
    step: number;
    loss: number;
    gnorm: number;
    step_sec: number;
    gflops: number;
  }[];
  post_step_python_verified_loss: number;
  random_init_baseline_loss: number;
}
