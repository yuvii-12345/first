// ============================================================================
// Glassbox-100M Pure Native C++17 CPU Reference Trainer (Zero External Libs!)
// Shares the EXACT same glassbox_format.hpp, ParamLayout, .gbxtok mmap shards,
// .gbx binary checkpoints, and forward/backward/AdamW algorithm as
// cuda/train_glassbox_100m.cu — runnable & verifiable on any x86_64/ARM64 CPU.
//
// Build:
//   g++ -O3 -march=native -fno-math-errno -std=c++17 \
//       cuda/train_glassbox_cpu.cpp glassbox/csrc/kernels.cpp -o cuda/glassbox_cpu_train
// ============================================================================

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

// Link directly to Glassbox's AVX2/FMA C++17 kernels in glassbox/csrc/kernels.cpp
extern "C" {
void gb_matmul_tiled(const void* A, const void* B, void* C,
                     int batch, int batch_a, int batch_b,
                     int M, int K, int N,
                     int trans_a, int trans_b, int accumulate, int dtype);
void gb_flash_attn_gqa_fwd(const void* Q, const void* K, const void* V,
                           void* O, void* L, void* out_probs,
                           int B, int H_q, int H_kv, int T_q, int T_k, int D,
                           double scale, int causal, int q_offset, int dtype);
void gb_flash_attn_gqa_bwd(const void* Q, const void* K, const void* V,
                           const void* O, const void* L, const void* dO,
                           void* dQ, void* dK, void* dV,
                           int B, int H_q, int H_kv, int T_q, int T_k, int D,
                           double scale, int causal, int q_offset, int dtype);
void gb_rmsnorm_fwd(const void* x, const void* w, void* y, void* inv_rms,
                    int N, int D, double eps, int dtype);
void gb_rmsnorm_bwd(const void* x, const void* w, const void* inv_rms, const void* dy,
                    void* dx, void* dw, int N, int D, int dtype);
void gb_rope(const void* x, void* y, int BH, int T_len, int D,
             int pos_offset, double base, int inverse, int dtype);
void gb_embedding_fwd(const void* table, const int32_t* idx, void* out,
                      int n_idx, int D, int dtype);
void gb_embedding_bwd(const void* dout, const int32_t* idx, void* dtable,
                      int n_idx, int vocab, int D, int dtype);
void gb_adamw_step(void* p, const void* g, void* m, void* v,
                   int n, double lr, double beta1, double beta2,
                   double eps, double weight_decay, int step,
                   double* out_update_sq, double* out_param_sq, int dtype);
double gb_l2_norm_sq(const void* x, int n, int dtype);
void gb_scale_inplace(void* x, int n, double factor, int dtype);
}

static inline void cpu_gemm(const float* A, const float* B, float* C,
                            int M, int K, int N, bool ta, bool tb, bool accum) {
    gb_matmul_tiled(A, B, C, 1, 1, 1, M, K, N, ta ? 1 : 0, tb ? 1 : 0, accum ? 1 : 0, 0);
}

static void permute_bthd_to_bhtd(const float* src, float* dst, int B, int T, int H, int Dh) {
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h)
            for (int t = 0; t < T; ++t)
                std::memcpy(dst + ((static_cast<size_t>(b) * H + h) * T + t) * Dh,
                            src + ((static_cast<size_t>(b) * T + t) * H + h) * Dh,
                            static_cast<size_t>(Dh) * sizeof(float));
}

static void permute_bhtd_to_bthd(const float* src, float* dst, int B, int H, int T, int Dh) {
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < H; ++h)
                std::memcpy(dst + ((static_cast<size_t>(b) * T + t) * H + h) * Dh,
                            src + ((static_cast<size_t>(b) * H + h) * T + t) * Dh,
                            static_cast<size_t>(Dh) * sizeof(float));
}

static void swiglu_fwd_cpu(const float* gate, const float* up, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + std::exp(-g));
        out[i] = (g * sig) * up[i];
    }
}

static void swiglu_bwd_cpu(const float* gate, const float* up, const float* dout,
                           float* dgate, float* dup, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float g = gate[i];
        float u = up[i];
        float dy = dout[i];
        float sig = 1.0f / (1.0f + std::exp(-g));
        float silu = g * sig;
        float dsilu = sig * (1.0f + g * (1.0f - sig));
        dup[i] = dy * silu;
        dgate[i] = dy * u * dsilu;
    }
}

static float fused_ce_fwd_bwd_cpu(const float* logits, const int32_t* targets,
                                  float* dlogits, int N, int V, float grad_scale) {
    double loss_sum = 0.0;
    const float scale = grad_scale / static_cast<float>(N);
    for (int i = 0; i < N; ++i) {
        const float* row = logits + static_cast<size_t>(i) * V;
        int tgt = targets[i];
        float mx = row[0];
        for (int v = 1; v < V; ++v) if (row[v] > mx) mx = row[v];
        double se = 0.0;
        for (int v = 0; v < V; ++v) se += std::exp(static_cast<double>(row[v] - mx));
        double log_z = static_cast<double>(mx) + std::log(se);
        loss_sum += log_z - static_cast<double>(row[tgt]);
        if (dlogits != nullptr) {
            float* drow = dlogits + static_cast<size_t>(i) * V;
            double inv_s = 1.0 / se;
            for (int v = 0; v < V; ++v) {
                double p = std::exp(static_cast<double>(row[v] - mx)) * inv_s;
                drow[v] = static_cast<float>((p - (v == tgt ? 1.0 : 0.0)) * scale);
            }
        }
    }
    return static_cast<float>(loss_sum / static_cast<double>(N));
}

struct CpuLayerActs {
    std::vector<float> ln1, inv_rms1, q_rot, k_rot, v_p, attn_o, attn_l, attn_m;
    std::vector<float> mid, ln2, inv_rms2, gate, up, act;
};

struct CpuTrainer {
    ModelConfig cfg;
    ParamLayout lay;
    int B, T, BT;

    std::vector<float> params, grads, adam_m, adam_v;
    std::vector<std::vector<float>> resid;
    std::vector<CpuLayerActs> acts;
    std::vector<float> final_ln, inv_rms_final, logits, dlogits;
    std::vector<float> tmp_q, tmp_k, tmp_v, tmp_qp, tmp_kp;
    std::vector<float> tmp_d, tmp_d2, tmp_d3, tmp_ff, tmp_ff2, tmp_ff3;
    std::vector<float> tmp_bhtd, tmp_kv_bhtd, tmp_kv_bhtd2, tmp_dw;

    CpuTrainer(const ModelConfig& c, int batch, int seq, bool alloc_optim = true)
        : cfg(c), lay(ParamLayout::build(c)), B(batch), T(seq), BT(batch * seq) {
        const size_t P = lay.total_params;
        const size_t D = cfg.d_model, KV = cfg.kv_dim(), FF = cfg.d_ff, V = cfg.vocab_size;

        params.resize(P);
        grads.assign(P, 0.0f);
        if (alloc_optim) {
            adam_m.assign(P, 0.0f);
            adam_v.assign(P, 0.0f);
        }

        resid.resize(cfg.n_layers + 1);
        for (uint32_t l = 0; l <= cfg.n_layers; ++l) resid[l].resize(static_cast<size_t>(BT) * D);

        acts.resize(cfg.n_layers);
        for (uint32_t l = 0; l < cfg.n_layers; ++l) {
            auto& a = acts[l];
            a.ln1.resize(static_cast<size_t>(BT) * D);
            a.inv_rms1.resize(BT);
            a.q_rot.resize(static_cast<size_t>(BT) * D);
            a.k_rot.resize(static_cast<size_t>(BT) * KV);
            a.v_p.resize(static_cast<size_t>(BT) * KV);
            a.attn_o.resize(static_cast<size_t>(BT) * D);
            a.attn_l.resize(static_cast<size_t>(B) * cfg.n_heads * T);
            a.attn_m.resize(static_cast<size_t>(BT) * D);
            a.mid.resize(static_cast<size_t>(BT) * D);
            a.ln2.resize(static_cast<size_t>(BT) * D);
            a.inv_rms2.resize(BT);
            a.gate.resize(static_cast<size_t>(BT) * FF);
            a.up.resize(static_cast<size_t>(BT) * FF);
            a.act.resize(static_cast<size_t>(BT) * FF);
        }

        final_ln.resize(static_cast<size_t>(BT) * D);
        inv_rms_final.resize(BT);
        logits.resize(static_cast<size_t>(BT) * V);
        dlogits.resize(static_cast<size_t>(BT) * V);

        tmp_q.resize(static_cast<size_t>(BT) * D);
        tmp_k.resize(static_cast<size_t>(BT) * KV);
        tmp_v.resize(static_cast<size_t>(BT) * KV);
        tmp_qp.resize(static_cast<size_t>(BT) * D);
        tmp_kp.resize(static_cast<size_t>(BT) * KV);
        tmp_d.resize(static_cast<size_t>(BT) * D);
        tmp_d2.resize(static_cast<size_t>(BT) * D);
        tmp_d3.resize(static_cast<size_t>(BT) * D);
        tmp_ff.resize(static_cast<size_t>(BT) * FF);
        tmp_ff2.resize(static_cast<size_t>(BT) * FF);
        tmp_ff3.resize(static_cast<size_t>(BT) * FF);
        tmp_bhtd.resize(static_cast<size_t>(BT) * D);
        tmp_kv_bhtd.resize(static_cast<size_t>(BT) * KV);
        tmp_kv_bhtd2.resize(static_cast<size_t>(BT) * KV);
        tmp_dw.resize(D);
    }

    void zero_grads() { std::fill(grads.begin(), grads.end(), 0.0f); }

    float forward_backward(const int32_t* x, const int32_t* y, float grad_scale, bool do_backward) {
        const int D = cfg.d_model, Hq = cfg.n_heads, Hkv = cfg.n_kv_heads;
        const int Dh = cfg.head_dim(), KV = cfg.kv_dim(), FF = cfg.d_ff, V = cfg.vocab_size;
        const double attn_scale = 1.0 / std::sqrt(static_cast<double>(Dh));

        gb_embedding_fwd(params.data() + lay.tok_emb, x, resid[0].data(), BT, D, 0);

        for (uint32_t l = 0; l < cfg.n_layers; ++l) {
            const auto& lo = lay.layers[l];
            auto& a = acts[l];

            gb_rmsnorm_fwd(resid[l].data(), params.data() + lo.norm1, a.ln1.data(), a.inv_rms1.data(), BT, D, cfg.rmsnorm_eps, 0);

            cpu_gemm(a.ln1.data(), params.data() + lo.w_q, tmp_q.data(), BT, D, D, false, true, false);
            cpu_gemm(a.ln1.data(), params.data() + lo.w_k, tmp_k.data(), BT, D, KV, false, true, false);
            cpu_gemm(a.ln1.data(), params.data() + lo.w_v, tmp_v.data(), BT, D, KV, false, true, false);

            permute_bthd_to_bhtd(tmp_q.data(), tmp_qp.data(), B, T, Hq, Dh);
            permute_bthd_to_bhtd(tmp_k.data(), tmp_kp.data(), B, T, Hkv, Dh);
            permute_bthd_to_bhtd(tmp_v.data(), a.v_p.data(),  B, T, Hkv, Dh);

            gb_rope(tmp_qp.data(), a.q_rot.data(), B * Hq,  T, Dh, 0, cfg.rope_base, 0, 0);
            gb_rope(tmp_kp.data(), a.k_rot.data(), B * Hkv, T, Dh, 0, cfg.rope_base, 0, 0);

            gb_flash_attn_gqa_fwd(a.q_rot.data(), a.k_rot.data(), a.v_p.data(),
                                  a.attn_o.data(), a.attn_l.data(), nullptr,
                                  B, Hq, Hkv, T, T, Dh, attn_scale, 1, 0, 0);
            permute_bhtd_to_bthd(a.attn_o.data(), a.attn_m.data(), B, Hq, T, Dh);

            cpu_gemm(a.attn_m.data(), params.data() + lo.w_o, tmp_d.data(), BT, D, D, false, true, false);
            for (size_t i = 0; i < static_cast<size_t>(BT) * D; ++i) a.mid[i] = resid[l][i] + tmp_d[i];

            gb_rmsnorm_fwd(a.mid.data(), params.data() + lo.norm2, a.ln2.data(), a.inv_rms2.data(), BT, D, cfg.rmsnorm_eps, 0);
            cpu_gemm(a.ln2.data(), params.data() + lo.w_gate, a.gate.data(), BT, D, FF, false, true, false);
            cpu_gemm(a.ln2.data(), params.data() + lo.w_up,   a.up.data(),   BT, D, FF, false, true, false);
            swiglu_fwd_cpu(a.gate.data(), a.up.data(), a.act.data(), static_cast<size_t>(BT) * FF);
            cpu_gemm(a.act.data(), params.data() + lo.w_down, tmp_d.data(), BT, FF, D, false, true, false);
            for (size_t i = 0; i < static_cast<size_t>(BT) * D; ++i) resid[l + 1][i] = a.mid[i] + tmp_d[i];
        }

        gb_rmsnorm_fwd(resid[cfg.n_layers].data(), params.data() + lay.final_norm,
                       final_ln.data(), inv_rms_final.data(), BT, D, cfg.rmsnorm_eps, 0);
        cpu_gemm(final_ln.data(), params.data() + lay.lm_head, logits.data(), BT, D, V, false, true, false);

        float loss = fused_ce_fwd_bwd_cpu(logits.data(), y, do_backward ? dlogits.data() : nullptr, BT, V, grad_scale);
        if (!do_backward) return loss;

        // Backward: LM head & Final RMSNorm
        cpu_gemm(dlogits.data(), final_ln.data(), grads.data() + lay.lm_head, V, BT, D, true, false, true);
        cpu_gemm(dlogits.data(), params.data() + lay.lm_head, tmp_d.data(), BT, V, D, false, false, false);
        gb_rmsnorm_bwd(resid[cfg.n_layers].data(), params.data() + lay.final_norm, inv_rms_final.data(),
                       tmp_d.data(), tmp_d2.data(), tmp_dw.data(), BT, D, 0);
        for (int d = 0; d < D; ++d) grads[lay.final_norm + d] += tmp_dw[d];

        for (int l = static_cast<int>(cfg.n_layers) - 1; l >= 0; --l) {
            const auto& lo = lay.layers[l];
            const auto& a = acts[l];

            // MLP backward
            cpu_gemm(tmp_d2.data(), a.act.data(), grads.data() + lo.w_down, D, BT, FF, true, false, true);
            cpu_gemm(tmp_d2.data(), params.data() + lo.w_down, tmp_ff.data(), BT, D, FF, false, false, false);
            swiglu_bwd_cpu(a.gate.data(), a.up.data(), tmp_ff.data(), tmp_ff2.data(), tmp_ff3.data(), static_cast<size_t>(BT) * FF);

            cpu_gemm(tmp_ff2.data(), a.ln2.data(), grads.data() + lo.w_gate, FF, BT, D, true, false, true);
            cpu_gemm(tmp_ff3.data(), a.ln2.data(), grads.data() + lo.w_up,   FF, BT, D, true, false, true);

            cpu_gemm(tmp_ff2.data(), params.data() + lo.w_gate, tmp_d.data(), BT, FF, D, false, false, false);
            cpu_gemm(tmp_ff3.data(), params.data() + lo.w_up,   tmp_d.data(), BT, FF, D, false, false, true);

            gb_rmsnorm_bwd(a.mid.data(), params.data() + lo.norm2, a.inv_rms2.data(),
                           tmp_d.data(), tmp_d3.data(), tmp_dw.data(), BT, D, 0);
            for (int d = 0; d < D; ++d) grads[lo.norm2 + d] += tmp_dw[d];
            for (size_t i = 0; i < static_cast<size_t>(BT) * D; ++i) tmp_d2[i] += tmp_d3[i];

            // Attention backward
            cpu_gemm(tmp_d2.data(), a.attn_m.data(), grads.data() + lo.w_o, D, BT, D, true, false, true);
            cpu_gemm(tmp_d2.data(), params.data() + lo.w_o, tmp_d.data(), BT, D, D, false, false, false);

            permute_bthd_to_bhtd(tmp_d.data(), tmp_bhtd.data(), B, T, Hq, Dh);
            gb_flash_attn_gqa_bwd(a.q_rot.data(), a.k_rot.data(), a.v_p.data(),
                                  a.attn_o.data(), a.attn_l.data(), tmp_bhtd.data(),
                                  tmp_qp.data(), tmp_kv_bhtd.data(), tmp_kv_bhtd2.data(),
                                  B, Hq, Hkv, T, T, Dh, attn_scale, 1, 0, 0);

            gb_rope(tmp_qp.data(),      tmp_bhtd.data(), B * Hq,  T, Dh, 0, cfg.rope_base, 1, 0);
            gb_rope(tmp_kv_bhtd.data(), tmp_kp.data(),   B * Hkv, T, Dh, 0, cfg.rope_base, 1, 0);

            permute_bhtd_to_bthd(tmp_bhtd.data(),     tmp_q.data(), B, Hq,  T, Dh);
            permute_bhtd_to_bthd(tmp_kp.data(),       tmp_k.data(), B, Hkv, T, Dh);
            permute_bhtd_to_bthd(tmp_kv_bhtd2.data(), tmp_v.data(), B, Hkv, T, Dh);

            cpu_gemm(tmp_q.data(), a.ln1.data(), grads.data() + lo.w_q, D,  BT, D, true, false, true);
            cpu_gemm(tmp_k.data(), a.ln1.data(), grads.data() + lo.w_k, KV, BT, D, true, false, true);
            cpu_gemm(tmp_v.data(), a.ln1.data(), grads.data() + lo.w_v, KV, BT, D, true, false, true);

            cpu_gemm(tmp_q.data(), params.data() + lo.w_q, tmp_d.data(), BT, D,  D, false, false, false);
            cpu_gemm(tmp_k.data(), params.data() + lo.w_k, tmp_d.data(), BT, KV, D, false, false, true);
            cpu_gemm(tmp_v.data(), params.data() + lo.w_v, tmp_d.data(), BT, KV, D, false, false, true);

            gb_rmsnorm_bwd(resid[l].data(), params.data() + lo.norm1, a.inv_rms1.data(),
                           tmp_d.data(), tmp_d3.data(), tmp_dw.data(), BT, D, 0);
            for (int d = 0; d < D; ++d) grads[lo.norm1 + d] += tmp_dw[d];
            for (size_t i = 0; i < static_cast<size_t>(BT) * D; ++i) tmp_d2[i] += tmp_d3[i];
        }

        // Embedding backward (accumulates onto weight-tied tok_emb grad!)
        for (int i = 0; i < BT; ++i) {
            float* dst = grads.data() + lay.tok_emb + static_cast<size_t>(x[i]) * D;
            const float* src = tmp_d2.data() + static_cast<size_t>(i) * D;
            for (int d = 0; d < D; ++d) dst[d] += src[d];
        }
        return loss;
    }
};

int main(int argc, char** argv) {
    std::string preset = "100m";
    std::string data_path = "";
    std::string out_path = "";
    std::string load_path = "";
    int batch_size = 1;
    int seq_len = 16;
    int steps = 3;
    float lr = 3e-4f;
    float grad_clip = 1.0f;
    uint64_t seed = 42;
    bool use_sgd = false; // Optional memory-lightweight SGD mode for 3.8GB RAM boxes on 100M params

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&]() -> std::string { return argv[++i]; };
        if (k == "--preset") preset = need();
        else if (k == "--data") data_path = need();
        else if (k == "--out") out_path = need();
        else if (k == "--load") load_path = need();
        else if (k == "--batch-size") batch_size = std::stoi(need());
        else if (k == "--seq-len") seq_len = std::stoi(need());
        else if (k == "--steps") steps = std::stoi(need());
        else if (k == "--lr") lr = std::stof(need());
        else if (k == "--seed") seed = std::stoull(need());
        else if (k == "--sgd") use_sgd = true;
    }

    ModelConfig cfg = (preset == "micro") ? ModelConfig::preset_micro() : ModelConfig::preset_100m();
    ParamLayout lay = ParamLayout::build(cfg);
    std::vector<float> init_w;
    if (!load_path.empty()) {
        load_gbx(load_path, cfg, lay, init_w);
    }

    CpuTrainer trainer(cfg, batch_size, seq_len, !use_sgd);
    if (!init_w.empty()) {
        trainer.params = std::move(init_w);
    } else {
        init_model_weights(trainer.params.data(), cfg, lay, seed);
    }

    std::unique_ptr<ShardStream> shards;
    if (!data_path.empty()) {
        shards = std::make_unique<ShardStream>(data_path);
    }

    std::printf("===========================================================================\n");
    std::printf(" Glassbox Pure C++17 Native Trainer (%s | %s)\n",
                preset.c_str(), use_sgd ? "SGD-Lite" : "AdamW");
    std::printf(" Arch: %uL | %ud | %uQ/%uKV GQA | ff=%u | vocab=%u | ctx=%u\n",
                cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                cfg.d_ff, cfg.vocab_size, cfg.context_len);
    std::printf(" Exact Parameters: %zu (%.2fM)\n", lay.total_params, lay.total_params / 1e6);
    std::printf("===========================================================================\n");

    std::vector<int32_t> x(static_cast<size_t>(batch_size) * seq_len);
    std::vector<int32_t> y(static_cast<size_t>(batch_size) * seq_len);
    uint64_t tokens_seen = 0;

    for (int s = 0; s < steps; ++s) {
        auto t0 = std::chrono::steady_clock::now();
        if (shards) {
            shards->next_batch(batch_size, seq_len, x.data(), y.data());
        } else {
            // Deterministic repeating pattern to verify real loss minimization
            for (int b = 0; b < batch_size; ++b) {
                for (int t = 0; t < seq_len; ++t) {
                    x[b * seq_len + t] = static_cast<int32_t>((100 + t * 17 + b * 7) % cfg.vocab_size);
                    y[b * seq_len + t] = static_cast<int32_t>((100 + (t + 1) * 17 + b * 7) % cfg.vocab_size);
                }
            }
        }
        trainer.zero_grads();
        float loss = trainer.forward_backward(x.data(), y.data(), 1.0f, true);
        double gnorm = std::sqrt(gb_l2_norm_sq(trainer.grads.data(), static_cast<int>(lay.total_params), 0));
        if (gnorm > grad_clip && gnorm > 0.0) {
            gb_scale_inplace(trainer.grads.data(), static_cast<int>(lay.total_params), grad_clip / gnorm, 0);
        }

        if (use_sgd) {
            for (size_t i = 0; i < lay.total_params; ++i) {
                trainer.params[i] -= lr * trainer.grads[i];
            }
        } else {
            double u_sq = 0.0, p_sq = 0.0;
            gb_adamw_step(trainer.params.data(), trainer.grads.data(),
                          trainer.adam_m.data(), trainer.adam_v.data(),
                          static_cast<int>(lay.total_params),
                          lr, 0.9, 0.95, 1e-8, 0.1, s + 1, &u_sq, &p_sq, 0);
        }

        tokens_seen += static_cast<uint64_t>(batch_size) * seq_len;
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        double gflops = (6.0 * static_cast<double>(lay.total_params) * batch_size * seq_len) / (dt * 1e9);
        std::printf("step %4d/%d | loss %.4f | gnorm %.4f | %.3fs | %.1f tok/s | %.2f GF/s\n",
                    s + 1, steps, loss, gnorm, dt, (batch_size * seq_len) / dt, gflops);
        std::fflush(stdout);
    }

    if (!out_path.empty()) {
        save_gbx(out_path, cfg, lay, trainer.params.data(), steps, tokens_seen, "{\"engine\":\"glassbox_cpp17\"}");
        std::printf("[Checkpoint] Saved %s\n", out_path.c_str());
    }
    return 0;
}
