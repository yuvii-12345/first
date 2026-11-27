export interface ModelConfig {
  vocab_size: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  d_ff: number;
  context_len: number;
  tie_weights: boolean;
  rope_base: number;
  rmsnorm_eps: number;
}

export interface HistoryPoint {
  step: number;
  loss: number;
  val_loss: number | null;
  gflops?: number;
}

export interface PackedTensor {
  shape: number[];
  scale: number;
  q_b64: string;
}

export interface BasicsBundle {
  format: string;
  model_name: string;
  dataset_id: string;
  preset_id: string;
  step: number;
  tokens_seen: number;
  num_parameters: number;
  final_loss: number;
  final_val_loss: number;
  topics_count: number;
  qa_variants_count: number;
  config: ModelConfig;
  tokenizer: {
    merges: [number, number, number][];
    merge_counts?: number[];
  };
  history: HistoryPoint[];
  weights_int8: Record<string, PackedTensor>;
}

export interface TokenCandidate {
  id: number;
  display: string;
  prob: number;
}

export interface TokenStep {
  step: number;
  id: number;
  text: string;
  display: string;
  prob: number;
  entropy: number;
  top_candidates: TokenCandidate[];
}

export interface ChatMessageItem {
  id: string;
  role: "user" | "assistant";
  content: string;
  tokens?: TokenStep[];
  promptUsed?: string;
  tokPerSec?: number;
  latencyMs?: number;
  timestamp: number;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessageItem[];
  createdAt: number;
}

export interface QAPair {
  id: string;
  question: string;
  answer: string;
}

export interface TrainProgressStep {
  step: number;
  totalSteps: number;
  loss: number;
  gradNorm: number;
  stepMs: number;
  tokensPerSec: number;
}

export interface XRaySnapshot {
  tokens: string[];
  nLayers: number;
  nHeads: number;
  nKvHeads: number;
  attentionWeights: number[][][][]; // [layer][head][T][T]
  topNextTokens: TokenCandidate[];
}

export interface Proof100M {
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
  verified_proofs: {
    gqa_flash_attn_gradcheck: { passed: boolean; max_abs_error: number; cosine_similarity: number };
    gqa_2l_transformer_gradcheck: { passed: boolean; max_abs_error: number; cosine_similarity: number };
    kv_cache_max_diff: number;
    cpp17_vs_python_loss_diff: number;
    cpp17_vs_python_gnorm_diff: number;
    gbx_402mb_python_load_sec: number;
    gbx_checkpoint_bytes: number;
  };
  live_100m_cpu_smoke_steps: { step: number; loss: number; gnorm: number; step_sec: number; gflops: number }[];
  post_step_python_verified_loss: number;
  random_init_baseline_loss: number;
}
