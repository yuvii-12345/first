// ============================================================================
// Glassbox-100M Pure From-Scratch CUDA C++17 Pretrainer (Zero PyTorch!)
// Trains Glassbox-100M (100,682,496 params: 12L, 768d, 12Q/4KV GQA, SwiGLU 2048,
// RoPE, RMSNorm, 32K BPE, 2048 ctx) on 100M to 10B+ tokens from .gbxtok shards
// and writes .gbx checkpoints directly loadable by Python Glassbox & Web Lab.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_89 -DGBX_USE_CUBLAS \
//        train_glassbox_100m.cu kernels.cu -lcublas -o glassbox_cuda_train
//   (or omit -DGBX_USE_CUBLAS -lcublas for 100% pure custom CUDA GEMM!)
// ============================================================================

#include "glassbox_cuda.cuh"
#include "glassbox_format.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace glassbox;
using namespace glassbox_cuda;

struct TrainArgs {
    std::string preset = "100m";
    std::string data_dir = "data/shards";
    std::string val_prefix = "val_";
    std::string train_prefix = "train_";
    std::string out_path = "checkpoints/glassbox_100m.gbx";
    std::string log_path = "checkpoints/glassbox_100m_metrics.jsonl";
    std::string resume_path = "";

    int batch_size = 8;
    int seq_len = 1024;
    int grad_accum = 8;                // 8 * 1024 * 8 = 65,536 tokens/step
    uint64_t total_tokens = 100000000ULL; // Default 100M tokens (set 10000000000 for 10B!)
    int warmup_steps = 100;
    float max_lr = 6e-4f;
    float min_lr = 6e-5f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float eps = 1e-8f;
    float weight_decay = 0.1f;
    float grad_clip = 1.0f;
    int log_every = 10;
    int eval_every = 100;
    int eval_batches = 8;
    int save_every = 250;
    uint64_t seed = 42;
};

static TrainArgs parse_args(int argc, char** argv) {
    TrainArgs a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", k.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (k == "--preset") a.preset = need();
        else if (k == "--data-dir") a.data_dir = need();
        else if (k == "--train-prefix") a.train_prefix = need();
        else if (k == "--val-prefix") a.val_prefix = need();
        else if (k == "--out") a.out_path = need();
        else if (k == "--log") a.log_path = need();
        else if (k == "--resume") a.resume_path = need();
        else if (k == "--batch-size") a.batch_size = std::stoi(need());
        else if (k == "--seq-len") a.seq_len = std::stoi(need());
        else if (k == "--grad-accum") a.grad_accum = std::stoi(need());
        else if (k == "--total-tokens") a.total_tokens = std::stoull(need());
        else if (k == "--warmup-steps") a.warmup_steps = std::stoi(need());
        else if (k == "--lr") a.max_lr = std::stof(need());
        else if (k == "--min-lr") a.min_lr = std::stof(need());
        else if (k == "--weight-decay") a.weight_decay = std::stof(need());
        else if (k == "--grad-clip") a.grad_clip = std::stof(need());
        else if (k == "--log-every") a.log_every = std::stoi(need());
        else if (k == "--eval-every") a.eval_every = std::stoi(need());
        else if (k == "--save-every") a.save_every = std::stoi(need());
        else if (k == "--seed") a.seed = std::stoull(need());
    }
    return a;
}

static float cosine_lr(uint64_t step, uint64_t warmup, uint64_t total, float max_lr, float min_lr) {
    if (step < warmup) return max_lr * static_cast<float>(step + 1) / static_cast<float>(std::max<uint64_t>(1, warmup));
    if (step >= total) return min_lr;
    double progress = static_cast<double>(step - warmup) / static_cast<double>(std::max<uint64_t>(1, total - warmup));
    double coeff = 0.5 * (1.0 + std::cos(3.14159265358979323846 * progress));
    return min_lr + static_cast<float>(coeff) * (max_lr - min_lr);
}

// Per-layer saved forward activations on GPU
struct LayerActivations {
    float* ln1 = nullptr;      // [BT, D]
    float* inv_rms1 = nullptr; // [BT]
    float* q_rot = nullptr;    // [B, Hq, T, Dh]
    float* k_rot = nullptr;    // [B, Hkv, T, Dh]
    float* v_p = nullptr;      // [B, Hkv, T, Dh]
    float* attn_o = nullptr;   // [B, Hq, T, Dh]
    float* attn_l = nullptr;   // [B, Hq, T]
    float* attn_m = nullptr;   // [BT, D]
    float* mid = nullptr;      // [BT, D]
    float* ln2 = nullptr;      // [BT, D]
    float* inv_rms2 = nullptr; // [BT]
    float* gate = nullptr;     // [BT, FF]
    float* up = nullptr;       // [BT, FF]
    float* act = nullptr;      // [BT, FF]
};

static float* dev_alloc(size_t count) {
    float* ptr = nullptr;
    GBX_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
    return ptr;
}

struct GpuTrainer {
    ModelConfig cfg;
    ParamLayout lay;
    CudaContext ctx;
    int B, T, BT;

    // Single contiguous parameter, gradient, and AdamW state buffers on GPU
    float *d_params = nullptr, *d_grads = nullptr, *d_m = nullptr, *d_v = nullptr;
    int32_t *d_x = nullptr, *d_y = nullptr;

    // Forward activation cache across layers
    std::vector<float*> resid; // L+1 buffers of [BT, D]
    std::vector<LayerActivations> acts;
    float *d_final_ln = nullptr, *d_inv_rms_final = nullptr;
    float *d_logits = nullptr, *d_dlogits = nullptr;

    // Reusable backward scratch buffers
    float *d_tmp_q = nullptr, *d_tmp_k = nullptr, *d_tmp_v = nullptr;
    float *d_tmp_qp = nullptr, *d_tmp_kp = nullptr;
    float *d_tmp_d = nullptr, *d_tmp_d2 = nullptr, *d_tmp_d3 = nullptr;
    float *d_tmp_ff = nullptr, *d_tmp_ff2 = nullptr, *d_tmp_ff3 = nullptr;
    float *d_tmp_bhtd = nullptr, *d_tmp_kv_bhtd = nullptr, *d_tmp_kv_bhtd2 = nullptr;
    float *d_scalar = nullptr;

    GpuTrainer(const ModelConfig& c, int batch, int seq)
        : cfg(c), lay(ParamLayout::build(c)), B(batch), T(seq), BT(batch * seq) {
        ctx.init();
        const size_t P = lay.total_params;
        const size_t D = cfg.d_model, KV = cfg.kv_dim(), FF = cfg.d_ff, V = cfg.vocab_size;

        d_params = dev_alloc(P);
        d_grads  = dev_alloc(P);
        d_m      = dev_alloc(P);
        d_v      = dev_alloc(P);
        GBX_CUDA_CHECK(cudaMemset(d_grads, 0, P * sizeof(float)));
        GBX_CUDA_CHECK(cudaMemset(d_m, 0, P * sizeof(float)));
        GBX_CUDA_CHECK(cudaMemset(d_v, 0, P * sizeof(float)));

        GBX_CUDA_CHECK(cudaMalloc(&d_x, BT * sizeof(int32_t)));
        GBX_CUDA_CHECK(cudaMalloc(&d_y, BT * sizeof(int32_t)));

        resid.resize(cfg.n_layers + 1);
        for (uint32_t l = 0; l <= cfg.n_layers; ++l) resid[l] = dev_alloc(static_cast<size_t>(BT) * D);

        acts.resize(cfg.n_layers);
        for (uint32_t l = 0; l < cfg.n_layers; ++l) {
            auto& a = acts[l];
            a.ln1      = dev_alloc(static_cast<size_t>(BT) * D);
            a.inv_rms1 = dev_alloc(BT);
            a.q_rot    = dev_alloc(static_cast<size_t>(BT) * D);
            a.k_rot    = dev_alloc(static_cast<size_t>(BT) * KV);
            a.v_p      = dev_alloc(static_cast<size_t>(BT) * KV);
            a.attn_o   = dev_alloc(static_cast<size_t>(BT) * D);
            a.attn_l   = dev_alloc(static_cast<size_t>(B) * cfg.n_heads * T);
            a.attn_m   = dev_alloc(static_cast<size_t>(BT) * D);
            a.mid      = dev_alloc(static_cast<size_t>(BT) * D);
            a.ln2      = dev_alloc(static_cast<size_t>(BT) * D);
            a.inv_rms2 = dev_alloc(BT);
            a.gate     = dev_alloc(static_cast<size_t>(BT) * FF);
            a.up       = dev_alloc(static_cast<size_t>(BT) * FF);
            a.act      = dev_alloc(static_cast<size_t>(BT) * FF);
        }

        d_final_ln      = dev_alloc(static_cast<size_t>(BT) * D);
        d_inv_rms_final = dev_alloc(BT);
        d_logits        = dev_alloc(static_cast<size_t>(BT) * V);
        d_dlogits       = dev_alloc(static_cast<size_t>(BT) * V);

        d_tmp_q       = dev_alloc(static_cast<size_t>(BT) * D);
        d_tmp_k       = dev_alloc(static_cast<size_t>(BT) * KV);
        d_tmp_v       = dev_alloc(static_cast<size_t>(BT) * KV);
        d_tmp_qp      = dev_alloc(static_cast<size_t>(BT) * D);
        d_tmp_kp      = dev_alloc(static_cast<size_t>(BT) * KV);
        d_tmp_d       = dev_alloc(static_cast<size_t>(BT) * D);
        d_tmp_d2      = dev_alloc(static_cast<size_t>(BT) * D);
        d_tmp_d3      = dev_alloc(static_cast<size_t>(BT) * D);
        d_tmp_ff      = dev_alloc(static_cast<size_t>(BT) * FF);
        d_tmp_ff2     = dev_alloc(static_cast<size_t>(BT) * FF);
        d_tmp_ff3     = dev_alloc(static_cast<size_t>(BT) * FF);
        d_tmp_bhtd    = dev_alloc(static_cast<size_t>(BT) * D);
        d_tmp_kv_bhtd = dev_alloc(static_cast<size_t>(BT) * KV);
        d_tmp_kv_bhtd2= dev_alloc(static_cast<size_t>(BT) * KV);
        d_scalar      = dev_alloc(1);
    }

    ~GpuTrainer() {
        cudaFree(d_params); cudaFree(d_grads); cudaFree(d_m); cudaFree(d_v);
        cudaFree(d_x); cudaFree(d_y);
        for (auto* p : resid) cudaFree(p);
        for (auto& a : acts) {
            cudaFree(a.ln1); cudaFree(a.inv_rms1); cudaFree(a.q_rot); cudaFree(a.k_rot);
            cudaFree(a.v_p); cudaFree(a.attn_o); cudaFree(a.attn_l); cudaFree(a.attn_m);
            cudaFree(a.mid); cudaFree(a.ln2); cudaFree(a.inv_rms2);
            cudaFree(a.gate); cudaFree(a.up); cudaFree(a.act);
        }
        cudaFree(d_final_ln); cudaFree(d_inv_rms_final);
        cudaFree(d_logits); cudaFree(d_dlogits);
        cudaFree(d_tmp_q); cudaFree(d_tmp_k); cudaFree(d_tmp_v);
        cudaFree(d_tmp_qp); cudaFree(d_tmp_kp);
        cudaFree(d_tmp_d); cudaFree(d_tmp_d2); cudaFree(d_tmp_d3);
        cudaFree(d_tmp_ff); cudaFree(d_tmp_ff2); cudaFree(d_tmp_ff3);
        cudaFree(d_tmp_bhtd); cudaFree(d_tmp_kv_bhtd); cudaFree(d_tmp_kv_bhtd2);
        cudaFree(d_scalar);
        ctx.destroy();
    }

    void upload_weights(const std::vector<float>& host_w) {
        GBX_CUDA_CHECK(cudaMemcpy(d_params, host_w.data(), lay.total_params * sizeof(float), cudaMemcpyHostToDevice));
    }

    void download_weights(std::vector<float>& host_w) {
        host_w.resize(lay.total_params);
        GBX_CUDA_CHECK(cudaMemcpy(host_w.data(), d_params, lay.total_params * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void zero_grads() {
        GBX_CUDA_CHECK(cudaMemsetAsync(d_grads, 0, lay.total_params * sizeof(float), ctx.stream));
    }

    float forward_backward(const int32_t* h_x, const int32_t* h_y, float grad_scale, bool do_backward) {
        const int D = cfg.d_model, Hq = cfg.n_heads, Hkv = cfg.n_kv_heads;
        const int Dh = cfg.head_dim(), KV = cfg.kv_dim(), FF = cfg.d_ff, V = cfg.vocab_size;
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(Dh));

        GBX_CUDA_CHECK(cudaMemcpyAsync(d_x, h_x, BT * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream));
        GBX_CUDA_CHECK(cudaMemcpyAsync(d_y, h_y, BT * sizeof(int32_t), cudaMemcpyHostToDevice, ctx.stream));

        // 1. Token Embedding
        embedding_fwd(ctx, d_params + lay.tok_emb, d_x, resid[0], BT, D);

        // 2. Transformer Blocks (Pre-Norm RMSNorm -> GQA RoPE FlashAttn2 -> SwiGLU)
        for (uint32_t l = 0; l < cfg.n_layers; ++l) {
            const auto& lo = lay.layers[l];
            auto& a = acts[l];

            rmsnorm_fwd(ctx, resid[l], d_params + lo.norm1, a.ln1, a.inv_rms1, BT, D, cfg.rmsnorm_eps);

            gemm(ctx, a.ln1, d_params + lo.w_q, d_tmp_q, BT, D, D, false, true, false);
            gemm(ctx, a.ln1, d_params + lo.w_k, d_tmp_k, BT, D, KV, false, true, false);
            gemm(ctx, a.ln1, d_params + lo.w_v, d_tmp_v, BT, D, KV, false, true, false);

            permute_bthd_to_bhtd(ctx, d_tmp_q, d_tmp_qp, B, T, Hq, Dh);
            permute_bthd_to_bhtd(ctx, d_tmp_k, d_tmp_kp, B, T, Hkv, Dh);
            permute_bthd_to_bhtd(ctx, d_tmp_v, a.v_p,    B, T, Hkv, Dh);

            rope_apply(ctx, d_tmp_qp, a.q_rot, B * Hq,  T, Dh, 0, cfg.rope_base, false);
            rope_apply(ctx, d_tmp_kp, a.k_rot, B * Hkv, T, Dh, 0, cfg.rope_base, false);

            flash_attn_gqa_fwd(ctx, a.q_rot, a.k_rot, a.v_p, a.attn_o, a.attn_l, B, Hq, Hkv, T, Dh, attn_scale);
            permute_bhtd_to_bthd(ctx, a.attn_o, a.attn_m, B, Hq, T, Dh);

            gemm(ctx, a.attn_m, d_params + lo.w_o, d_tmp_d, BT, D, D, false, true, false);
            add_vec(ctx, resid[l], d_tmp_d, a.mid, BT * D);

            rmsnorm_fwd(ctx, a.mid, d_params + lo.norm2, a.ln2, a.inv_rms2, BT, D, cfg.rmsnorm_eps);
            gemm(ctx, a.ln2, d_params + lo.w_gate, a.gate, BT, D, FF, false, true, false);
            gemm(ctx, a.ln2, d_params + lo.w_up,   a.up,   BT, D, FF, false, true, false);
            swiglu_fwd(ctx, a.gate, a.up, a.act, BT * FF);
            gemm(ctx, a.act, d_params + lo.w_down, d_tmp_d, BT, FF, D, false, true, false);
            add_vec(ctx, a.mid, d_tmp_d, resid[l + 1], BT * D);
        }

        // 3. Final RMSNorm + Weight-Tied LM Head + Fused Online Cross-Entropy
        rmsnorm_fwd(ctx, resid[cfg.n_layers], d_params + lay.final_norm, d_final_ln, d_inv_rms_final, BT, D, cfg.rmsnorm_eps);
        gemm(ctx, d_final_ln, d_params + lay.lm_head, d_logits, BT, D, V, false, true, false);

        float loss = fused_cross_entropy_fwd_bwd(
            ctx, d_logits, d_y, do_backward ? d_dlogits : nullptr, d_scalar, BT, V, grad_scale);
        if (!do_backward) return loss;

        // 4. Backward: LM Head & Final RMSNorm
        gemm(ctx, d_dlogits, d_final_ln, d_grads + lay.lm_head, V, BT, D, true, false, true);
        gemm(ctx, d_dlogits, d_params + lay.lm_head, d_tmp_d, BT, V, D, false, false, false);
        // d_tmp_d2 holds d_resid flowing backward through layers
        rmsnorm_bwd(ctx, resid[cfg.n_layers], d_params + lay.final_norm, d_inv_rms_final,
                    d_tmp_d, d_tmp_d2, d_grads + lay.final_norm, BT, D);

        // 5. Backward through Transformer Blocks (L-1 down to 0)
        for (int l = static_cast<int>(cfg.n_layers) - 1; l >= 0; --l) {
            const auto& lo = lay.layers[l];
            const auto& a = acts[l];

            // --- MLP Backward (d_tmp_d2 = d_resid_{l+1}) ---
            gemm(ctx, d_tmp_d2, a.act, d_grads + lo.w_down, D, BT, FF, true, false, true);
            gemm(ctx, d_tmp_d2, d_params + lo.w_down, d_tmp_ff, BT, D, FF, false, false, false);
            swiglu_bwd(ctx, a.gate, a.up, d_tmp_ff, d_tmp_ff2, d_tmp_ff3, BT * FF);

            gemm(ctx, d_tmp_ff2, a.ln2, d_grads + lo.w_gate, FF, BT, D, true, false, true);
            gemm(ctx, d_tmp_ff3, a.ln2, d_grads + lo.w_up,   FF, BT, D, true, false, true);

            gemm(ctx, d_tmp_ff2, d_params + lo.w_gate, d_tmp_d, BT, FF, D, false, false, false);
            gemm(ctx, d_tmp_ff3, d_params + lo.w_up,   d_tmp_d, BT, FF, D, false, false, true);

            rmsnorm_bwd(ctx, a.mid, d_params + lo.norm2, a.inv_rms2,
                        d_tmp_d, d_tmp_d3, d_grads + lo.norm2, BT, D);
            // d_mid = d_resid_{l+1} + d_mid_from_mlp  (accumulated into d_tmp_d2)
            add_inplace(ctx, d_tmp_d2, d_tmp_d3, BT * D);

            // --- Attention Backward (d_tmp_d2 = d_mid) ---
            gemm(ctx, d_tmp_d2, a.attn_m, d_grads + lo.w_o, D, BT, D, true, false, true);
            gemm(ctx, d_tmp_d2, d_params + lo.w_o, d_tmp_d, BT, D, D, false, false, false);

            permute_bthd_to_bhtd(ctx, d_tmp_d, d_tmp_bhtd, B, T, Hq, Dh);
            flash_attn_gqa_bwd(ctx, a.q_rot, a.k_rot, a.v_p, a.attn_o, a.attn_l, d_tmp_bhtd,
                               d_tmp_qp, d_tmp_kv_bhtd, d_tmp_kv_bhtd2, B, Hq, Hkv, T, Dh, attn_scale);

            rope_apply(ctx, d_tmp_qp,      d_tmp_bhtd, B * Hq,  T, Dh, 0, cfg.rope_base, true);
            rope_apply(ctx, d_tmp_kv_bhtd, d_tmp_kp,   B * Hkv, T, Dh, 0, cfg.rope_base, true);

            permute_bhtd_to_bthd(ctx, d_tmp_bhtd,     d_tmp_q, B, Hq,  T, Dh);
            permute_bhtd_to_bthd(ctx, d_tmp_kp,       d_tmp_k, B, Hkv, T, Dh);
            permute_bhtd_to_bthd(ctx, d_tmp_kv_bhtd2, d_tmp_v, B, Hkv, T, Dh);

            gemm(ctx, d_tmp_q, a.ln1, d_grads + lo.w_q, D,  BT, D, true, false, true);
            gemm(ctx, d_tmp_k, a.ln1, d_grads + lo.w_k, KV, BT, D, true, false, true);
            gemm(ctx, d_tmp_v, a.ln1, d_grads + lo.w_v, KV, BT, D, true, false, true);

            gemm(ctx, d_tmp_q, d_params + lo.w_q, d_tmp_d, BT, D,  D, false, false, false);
            gemm(ctx, d_tmp_k, d_params + lo.w_k, d_tmp_d, BT, KV, D, false, false, true);
            gemm(ctx, d_tmp_v, d_params + lo.w_v, d_tmp_d, BT, KV, D, false, false, true);

            rmsnorm_bwd(ctx, resid[l], d_params + lo.norm1, a.inv_rms1,
                        d_tmp_d, d_tmp_d3, d_grads + lo.norm1, BT, D);
            // d_resid_l = d_mid + d_x_from_attn
            add_inplace(ctx, d_tmp_d2, d_tmp_d3, BT * D);
        }

        // 6. Embedding Backward (accumulates cleanly onto weight-tied tok_emb grad!)
        embedding_bwd(ctx, d_tmp_d2, d_x, d_grads + lay.tok_emb, BT, D);
        return loss;
    }
};

int main(int argc, char** argv) {
    TrainArgs args = parse_args(argc, argv);

    int dev_count = 0;
    GBX_CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    cudaDeviceProp prop{};
    GBX_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    ModelConfig cfg = (args.preset == "micro") ? ModelConfig::preset_micro() : ModelConfig::preset_100m();
    ParamLayout lay = ParamLayout::build(cfg);
    std::vector<float> host_weights;
    uint64_t start_step = 0;
    uint64_t tokens_seen = 0;

    if (!args.resume_path.empty()) {
        GBXHeader hdr = load_gbx(args.resume_path, cfg, lay, host_weights);
        start_step = hdr.step;
        tokens_seen = hdr.tokens_seen;
        std::printf("[Glassbox CUDA] Resumed %s at step %llu (%llu tokens)\n",
                    args.resume_path.c_str(), (unsigned long long)start_step, (unsigned long long)tokens_seen);
    } else {
        host_weights.resize(lay.total_params);
        init_model_weights(host_weights.data(), cfg, lay, args.seed);
    }

    ShardStream train_shards(args.data_dir, args.train_prefix);
    std::unique_ptr<ShardStream> val_shards;
    try {
        val_shards = std::make_unique<ShardStream>(args.data_dir, args.val_prefix);
    } catch (...) {
        val_shards.reset();
    }

    const uint64_t tokens_per_step = static_cast<uint64_t>(args.batch_size) * args.seq_len * args.grad_accum;
    const uint64_t total_steps = (args.total_tokens + tokens_per_step - 1) / tokens_per_step;

    std::printf("===========================================================================\n");
    std::printf(" Glassbox-100M Pure CUDA C++17 Pretrainer (Zero PyTorch / Zero cuDNN)\n");
    std::printf(" GPU: %s (SM %d.%d, %.1f GB VRAM)\n",
                prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1e9);
    std::printf(" Arch: %uL | %ud | %uQ/%uKV GQA | ff=%u | vocab=%u | ctx=%u\n",
                cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                cfg.d_ff, cfg.vocab_size, cfg.context_len);
    std::printf(" Exact Parameters: %zu (%.2fM)\n", lay.total_params, lay.total_params / 1e6);
    std::printf(" Train Shards: %zu (%llu tokens) | Target: %llu tokens (%llu steps)\n",
                train_shards.num_shards(), (unsigned long long)train_shards.total_tokens,
                (unsigned long long)args.total_tokens, (unsigned long long)total_steps);
    std::printf(" Batch: %d x %d x accum %d = %llu tokens/step\n",
                args.batch_size, args.seq_len, args.grad_accum, (unsigned long long)tokens_per_step);
    std::printf("===========================================================================\n");

    GpuTrainer trainer(cfg, args.batch_size, args.seq_len);
    trainer.upload_weights(host_weights);

    std::vector<int32_t> h_x(static_cast<size_t>(args.batch_size) * args.seq_len);
    std::vector<int32_t> h_y(static_cast<size_t>(args.batch_size) * args.seq_len);
    std::ofstream log_file(args.log_path, std::ios::app);

    for (uint64_t step = start_step; step < total_steps; ++step) {
        auto t0 = std::chrono::steady_clock::now();
        trainer.zero_grads();

        double loss_acc = 0.0;
        const float g_scale = 1.0f / static_cast<float>(args.grad_accum);
        for (int micro = 0; micro < args.grad_accum; ++micro) {
            train_shards.next_batch(args.batch_size, args.seq_len, h_x.data(), h_y.data());
            loss_acc += static_cast<double>(trainer.forward_backward(h_x.data(), h_y.data(), g_scale, true));
        }
        double train_loss = loss_acc / static_cast<double>(args.grad_accum);

        float gnorm = global_grad_norm(trainer.ctx, trainer.d_grads, trainer.d_scalar, lay.total_params);
        float clip_scale = (gnorm > args.grad_clip && gnorm > 0.0f) ? (args.grad_clip / gnorm) : 1.0f;
        float lr = cosine_lr(step, args.warmup_steps, total_steps, args.max_lr, args.min_lr);

        adamw_step(trainer.ctx, trainer.d_params, trainer.d_grads, trainer.d_m, trainer.d_v,
                   lay.total_params, lr, args.beta1, args.beta2, args.eps,
                   args.weight_decay, clip_scale, static_cast<int>(step + 1));
        GBX_CUDA_CHECK(cudaStreamSynchronize(trainer.ctx.stream));

        tokens_seen += tokens_per_step;
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        double tok_s = static_cast<double>(tokens_per_step) / std::max(dt, 1e-6);
        double tflops = (6.0 * static_cast<double>(lay.total_params) * static_cast<double>(tokens_per_step)) / (dt * 1e12);
        double pct = 100.0 * static_cast<double>(tokens_seen) / static_cast<double>(args.total_tokens);
        double eta_min = (static_cast<double>(total_steps - step - 1) * dt) / 60.0;

        if ((step + 1) % args.log_every == 0 || step == start_step || step + 1 == total_steps) {
            double val_loss = -1.0;
            if (val_shards && ((step + 1) % args.eval_every == 0 || step + 1 == total_steps)) {
                val_shards->reset();
                double v_acc = 0.0;
                for (int eb = 0; eb < args.eval_batches; ++eb) {
                    val_shards->next_batch(args.batch_size, args.seq_len, h_x.data(), h_y.data());
                    v_acc += static_cast<double>(trainer.forward_backward(h_x.data(), h_y.data(), 0.0f, false));
                }
                val_loss = v_acc / args.eval_batches;
            }
            std::printf("step %6llu/%llu (%5.1f%%) | loss %.4f | val %.4f | lr %.2e | gnorm %.3f | %.0f tok/s | %.1f TF/s | ETA %.1fm\n",
                        (unsigned long long)(step + 1), (unsigned long long)total_steps, pct,
                        train_loss, val_loss, lr, gnorm, tok_s, tflops, eta_min);
            std::fflush(stdout);

            if (log_file) {
                log_file << "{\"step\":" << (step + 1)
                         << ",\"tokens\":" << tokens_seen
                         << ",\"loss\":" << train_loss
                         << ",\"val_loss\":" << val_loss
                         << ",\"lr\":" << lr
                         << ",\"gnorm\":" << gnorm
                         << ",\"tok_s\":" << tok_s
                         << ",\"tflops\":" << tflops << "}\n";
                log_file.flush();
            }
        }

        if ((step + 1) % args.save_every == 0 || step + 1 == total_steps) {
            trainer.download_weights(host_weights);
            std::string meta = "{\"preset\":\"" + args.preset + "\",\"loss\":" + std::to_string(train_loss) + "}";
            save_gbx(args.out_path, cfg, lay, host_weights.data(), step + 1, tokens_seen, meta);
            std::printf("[Checkpoint] Saved %s (step=%llu, tokens=%llu)\n",
                        args.out_path.c_str(), (unsigned long long)(step + 1), (unsigned long long)tokens_seen);
        }
    }
    return 0;
}
