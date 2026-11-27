#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

// Global performance telemetry counters
std::atomic<uint64_t> g_matmul_flops{0};
std::atomic<uint64_t> g_matmul_calls{0};
std::atomic<uint64_t> g_matmul_ns{0};

// ---------------------------------------------------------------------------
// Cache-Blocked & Register-Tiled GEMM (zero external BLAS)
// Computes: C = (accumulate ? C : 0) + op(A) @ op(B)
//   op(A) has shape [M, K], op(B) has shape [K, N], C has shape [M, N]
// ---------------------------------------------------------------------------

template <typename T>
void gemm_naive_2d(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                   int M, int K, int N, int trans_a, int trans_b, int accumulate) {
    if (!accumulate) {
        std::memset(C, 0, static_cast<size_t>(M) * N * sizeof(T));
    }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = T(0);
            for (int k = 0; k < K; ++k) {
                T a = trans_a ? A[k * M + i] : A[i * K + k];
                T b = trans_b ? B[j * K + k] : B[k * N + j];
                sum += a * b;
            }
            C[i * N + j] += sum;
        }
    }
}

template <typename T>
void gemm_tiled_2d(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                   int M, int K, int N, int trans_a, int trans_b, int accumulate) {
    if (!accumulate) {
        std::memset(C, 0, static_cast<size_t>(M) * N * sizeof(T));
    }

    // Tile sizes tuned for 48KB L1d / 1.3MB L2 cache
    constexpr int MC = 64;
    constexpr int KC = 64;
    constexpr int NC = 64;

    alignas(64) T A_pack[MC * KC];
    alignas(64) T B_pack[KC * NC];

    for (int i0 = 0; i0 < M; i0 += MC) {
        const int mc = std::min(MC, M - i0);
        for (int k0 = 0; k0 < K; k0 += KC) {
            const int kc = std::min(KC, K - k0);

            // Pack tile of op(A)[i0:i0+mc, k0:k0+kc] into contiguous row-major A_pack[mc x kc]
            if (!trans_a) {
                for (int i = 0; i < mc; ++i) {
                    const T* src = A + static_cast<size_t>(i0 + i) * K + k0;
                    T* dst = A_pack + static_cast<size_t>(i) * kc;
                    std::memcpy(dst, src, static_cast<size_t>(kc) * sizeof(T));
                }
            } else {
                for (int i = 0; i < mc; ++i) {
                    T* dst = A_pack + static_cast<size_t>(i) * kc;
                    for (int k = 0; k < kc; ++k) {
                        dst[k] = A[static_cast<size_t>(k0 + k) * M + (i0 + i)];
                    }
                }
            }

            for (int j0 = 0; j0 < N; j0 += NC) {
                const int nc = std::min(NC, N - j0);

                // Pack tile of op(B)[k0:k0+kc, j0:j0+nc] into contiguous row-major B_pack[kc x nc]
                if (!trans_b) {
                    for (int k = 0; k < kc; ++k) {
                        const T* src = B + static_cast<size_t>(k0 + k) * N + j0;
                        T* dst = B_pack + static_cast<size_t>(k) * nc;
                        std::memcpy(dst, src, static_cast<size_t>(nc) * sizeof(T));
                    }
                } else {
                    for (int k = 0; k < kc; ++k) {
                        T* dst = B_pack + static_cast<size_t>(k) * nc;
                        for (int j = 0; j < nc; ++j) {
                            dst[j] = B[static_cast<size_t>(j0 + j) * K + (k0 + k)];
                        }
                    }
                }

                // Register-blocked micro-kernel: 4 rows of A x contiguous row of B (auto-vectorizes to AVX2/AVX-512 FMA)
                int i = 0;
                for (; i + 3 < mc; i += 4) {
                    T* __restrict__ c0 = C + static_cast<size_t>(i0 + i + 0) * N + j0;
                    T* __restrict__ c1 = C + static_cast<size_t>(i0 + i + 1) * N + j0;
                    T* __restrict__ c2 = C + static_cast<size_t>(i0 + i + 2) * N + j0;
                    T* __restrict__ c3 = C + static_cast<size_t>(i0 + i + 3) * N + j0;

                    const T* __restrict__ a0_row = A_pack + static_cast<size_t>(i + 0) * kc;
                    const T* __restrict__ a1_row = A_pack + static_cast<size_t>(i + 1) * kc;
                    const T* __restrict__ a2_row = A_pack + static_cast<size_t>(i + 2) * kc;
                    const T* __restrict__ a3_row = A_pack + static_cast<size_t>(i + 3) * kc;

                    for (int k = 0; k < kc; ++k) {
                        const T a0 = a0_row[k];
                        const T a1 = a1_row[k];
                        const T a2 = a2_row[k];
                        const T a3 = a3_row[k];
                        const T* __restrict__ b_row = B_pack + static_cast<size_t>(k) * nc;
                        for (int j = 0; j < nc; ++j) {
                            const T bv = b_row[j];
                            c0[j] += a0 * bv;
                            c1[j] += a1 * bv;
                            c2[j] += a2 * bv;
                            c3[j] += a3 * bv;
                        }
                    }
                }
                for (; i < mc; ++i) {
                    T* __restrict__ ci = C + static_cast<size_t>(i0 + i) * N + j0;
                    const T* __restrict__ ai_row = A_pack + static_cast<size_t>(i) * kc;
                    for (int k = 0; k < kc; ++k) {
                        const T a = ai_row[k];
                        const T* __restrict__ b_row = B_pack + static_cast<size_t>(k) * nc;
                        for (int j = 0; j < nc; ++j) {
                            ci[j] += a * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void batched_gemm_tiled(const T* A, const T* B, T* C,
                        int batch, int batch_a, int batch_b,
                        int M, int K, int N,
                        int trans_a, int trans_b, int accumulate) {
    auto t0 = std::chrono::steady_clock::now();
    const size_t stride_a = static_cast<size_t>(M) * K;
    const size_t stride_b = static_cast<size_t>(K) * N;
    const size_t stride_c = static_cast<size_t>(M) * N;

    for (int b = 0; b < batch; ++b) {
        const int ba = (batch_a == 1) ? 0 : b;
        const int bb = (batch_b == 1) ? 0 : b;
        gemm_tiled_2d<T>(A + ba * stride_a, B + bb * stride_b, C + b * stride_c,
                         M, K, N, trans_a, trans_b, accumulate);
    }

    auto t1 = std::chrono::steady_clock::now();
    uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    uint64_t flops = 2ULL * static_cast<uint64_t>(batch) * M * K * N;
    g_matmul_flops.fetch_add(flops, std::memory_order_relaxed);
    g_matmul_calls.fetch_add(1, std::memory_order_relaxed);
    g_matmul_ns.fetch_add(ns, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// FlashAttention-2 Style Tiled Online-Softmax Attention (Forward & Backward)
// Q, K, V, O: [BH, T, D], L: [BH, T]
// Never materializes [T, T] during training (out_probs == nullptr)
// ---------------------------------------------------------------------------

template <typename T>
void flash_attn_fwd_impl(const T* __restrict__ Q, const T* __restrict__ K,
                         const T* __restrict__ V, T* __restrict__ O,
                         T* __restrict__ L, T* __restrict__ out_probs,
                         int BH, int T_len, int D, T scale, int causal) {
    constexpr int BR = 32;
    constexpr int BC = 32;
    const T neg_inf = -std::numeric_limits<T>::infinity();

    std::vector<T> S_tile(BR * BC);
    std::vector<T> O_tile(BR * D);
    T m_vec[BR];
    T l_vec[BR];

    const size_t mat_stride = static_cast<size_t>(T_len) * D;
    const size_t l_stride = static_cast<size_t>(T_len);
    const size_t prob_stride = static_cast<size_t>(T_len) * T_len;

    for (int bh = 0; bh < BH; ++bh) {
        const T* Qb = Q + bh * mat_stride;
        const T* Kb = K + bh * mat_stride;
        const T* Vb = V + bh * mat_stride;
        T* Ob = O + bh * mat_stride;
        T* Lb = L + bh * l_stride;

        for (int i0 = 0; i0 < T_len; i0 += BR) {
            const int br = std::min(BR, T_len - i0);
            for (int r = 0; r < br; ++r) {
                m_vec[r] = neg_inf;
                l_vec[r] = T(0);
            }
            std::fill(O_tile.begin(), O_tile.begin() + br * D, T(0));

            const int j_limit = causal ? std::min(T_len, i0 + br) : T_len;

            for (int j0 = 0; j0 < j_limit; j0 += BC) {
                const int bc = std::min(BC, j_limit - j0);

                // Compute tile attention scores S_ij = scale * Q_i @ K_j^T
                for (int r = 0; r < br; ++r) {
                    const int qi = i0 + r;
                    const T* q_row = Qb + static_cast<size_t>(qi) * D;
                    T* s_row = S_tile.data() + r * bc;

                    T m_tile = neg_inf;
                    for (int c = 0; c < bc; ++c) {
                        const int kj = j0 + c;
                        if (causal && kj > qi) {
                            s_row[c] = neg_inf;
                        } else {
                            const T* k_row = Kb + static_cast<size_t>(kj) * D;
                            T dot = T(0);
                            for (int d = 0; d < D; ++d) {
                                dot += q_row[d] * k_row[d];
                            }
                            const T val = dot * scale;
                            s_row[c] = val;
                            if (val > m_tile) m_tile = val;
                        }
                    }

                    if (m_tile == neg_inf) continue;

                    const T m_old = m_vec[r];
                    const T m_new = (m_old > m_tile) ? m_old : m_tile;
                    const T alpha = (m_old == neg_inf) ? T(0) : std::exp(m_old - m_new);

                    T* o_row = O_tile.data() + r * D;
                    if (alpha != T(1)) {
                        for (int d = 0; d < D; ++d) {
                            o_row[d] *= alpha;
                        }
                        l_vec[r] *= alpha;
                    }

                    T l_tile = T(0);
                    for (int c = 0; c < bc; ++c) {
                        const T sv = s_row[c];
                        if (sv == neg_inf) continue;
                        const T p = std::exp(sv - m_new);
                        l_tile += p;
                        const T* v_row = Vb + static_cast<size_t>(j0 + c) * D;
                        for (int d = 0; d < D; ++d) {
                            o_row[d] += p * v_row[d];
                        }
                    }
                    l_vec[r] += l_tile;
                    m_vec[r] = m_new;
                }
            }

            // Finalize O and L for this query tile
            for (int r = 0; r < br; ++r) {
                const int qi = i0 + r;
                const T inv_l = (l_vec[r] > T(0)) ? (T(1) / l_vec[r]) : T(0);
                const T* o_src = O_tile.data() + r * D;
                T* o_dst = Ob + static_cast<size_t>(qi) * D;
                for (int d = 0; d < D; ++d) {
                    o_dst[d] = o_src[d] * inv_l;
                }
                Lb[qi] = m_vec[r] + std::log(l_vec[r]);
            }
        }

        // Optional interpretability probe: materialize exact attention probabilities only when requested
        if (out_probs != nullptr) {
            T* Pb = out_probs + bh * prob_stride;
            for (int i = 0; i < T_len; ++i) {
                const T* q_row = Qb + static_cast<size_t>(i) * D;
                const T li = Lb[i];
                T* p_row = Pb + static_cast<size_t>(i) * T_len;
                for (int j = 0; j < T_len; ++j) {
                    if (causal && j > i) {
                        p_row[j] = T(0);
                    } else {
                        const T* k_row = Kb + static_cast<size_t>(j) * D;
                        T dot = T(0);
                        for (int d = 0; d < D; ++d) dot += q_row[d] * k_row[d];
                        p_row[j] = std::exp(dot * scale - li);
                    }
                }
            }
        }
    }
}

template <typename T>
void flash_attn_bwd_impl(const T* __restrict__ Q, const T* __restrict__ K,
                         const T* __restrict__ V, const T* __restrict__ O,
                         const T* __restrict__ L, const T* __restrict__ dO,
                         T* __restrict__ dQ, T* __restrict__ dK, T* __restrict__ dV,
                         int BH, int T_len, int D, T scale, int causal) {
    constexpr int BR = 32;
    constexpr int BC = 32;

    const size_t total = static_cast<size_t>(BH) * T_len * D;
    std::memset(dQ, 0, total * sizeof(T));
    std::memset(dK, 0, total * sizeof(T));
    std::memset(dV, 0, total * sizeof(T));

    std::vector<T> D_vec(T_len);
    std::vector<T> P_tile(BR * BC);
    std::vector<T> dS_tile(BR * BC);
    std::vector<T> dK_tile(BC * D);
    std::vector<T> dV_tile(BC * D);

    const size_t mat_stride = static_cast<size_t>(T_len) * D;
    const size_t l_stride = static_cast<size_t>(T_len);

    for (int bh = 0; bh < BH; ++bh) {
        const T* Qb = Q + bh * mat_stride;
        const T* Kb = K + bh * mat_stride;
        const T* Vb = V + bh * mat_stride;
        const T* Ob = O + bh * mat_stride;
        const T* Lb = L + bh * l_stride;
        const T* dOb = dO + bh * mat_stride;

        T* dQb = dQ + bh * mat_stride;
        T* dKb = dK + bh * mat_stride;
        T* dVb = dV + bh * mat_stride;

        // Precompute D_i = dot(dO_i, O_i)
        for (int i = 0; i < T_len; ++i) {
            const T* do_row = dOb + static_cast<size_t>(i) * D;
            const T* o_row = Ob + static_cast<size_t>(i) * D;
            T dot = T(0);
            for (int d = 0; d < D; ++d) dot += do_row[d] * o_row[d];
            D_vec[i] = dot;
        }

        // Outer loop over KV blocks
        for (int j0 = 0; j0 < T_len; j0 += BC) {
            const int bc = std::min(BC, T_len - j0);
            std::fill(dK_tile.begin(), dK_tile.begin() + bc * D, T(0));
            std::fill(dV_tile.begin(), dV_tile.begin() + bc * D, T(0));

            const int i_start = causal ? (j0 / BR) * BR : 0;

            for (int i0 = i_start; i0 < T_len; i0 += BR) {
                const int br = std::min(BR, T_len - i0);

                // Recompute P_ij and dS_ij on the fly inside SRAM tile
                for (int r = 0; r < br; ++r) {
                    const int qi = i0 + r;
                    const T* q_row = Qb + static_cast<size_t>(qi) * D;
                    const T* do_row = dOb + static_cast<size_t>(qi) * D;
                    const T li = Lb[qi];
                    const T di = D_vec[qi];
                    T* p_row = P_tile.data() + r * bc;
                    T* ds_row = dS_tile.data() + r * bc;

                    for (int c = 0; c < bc; ++c) {
                        const int kj = j0 + c;
                        if (causal && kj > qi) {
                            p_row[c] = T(0);
                            ds_row[c] = T(0);
                        } else {
                            const T* k_row = Kb + static_cast<size_t>(kj) * D;
                            const T* v_row = Vb + static_cast<size_t>(kj) * D;
                            T s_dot = T(0);
                            T dp_dot = T(0);
                            for (int d = 0; d < D; ++d) {
                                s_dot += q_row[d] * k_row[d];
                                dp_dot += do_row[d] * v_row[d];
                            }
                            const T p = std::exp(s_dot * scale - li);
                            p_row[c] = p;
                            ds_row[c] = p * (dp_dot - di);
                        }
                    }
                }

                // Accumulate dV_tile += P^T @ dO_i and dK_tile += scale * dS^T @ Q_i
                for (int c = 0; c < bc; ++c) {
                    T* dv_row = dV_tile.data() + c * D;
                    T* dk_row = dK_tile.data() + c * D;
                    for (int r = 0; r < br; ++r) {
                        const int qi = i0 + r;
                        const T p = P_tile[r * bc + c];
                        const T ds = dS_tile[r * bc + c] * scale;
                        if (p == T(0) && ds == T(0)) continue;
                        const T* do_row = dOb + static_cast<size_t>(qi) * D;
                        const T* q_row = Qb + static_cast<size_t>(qi) * D;
                        for (int d = 0; d < D; ++d) {
                            dv_row[d] += p * do_row[d];
                            dk_row[d] += ds * q_row[d];
                        }
                    }
                }

                // Accumulate dQ_i += scale * dS @ K_j
                for (int r = 0; r < br; ++r) {
                    const int qi = i0 + r;
                    T* dq_row = dQb + static_cast<size_t>(qi) * D;
                    const T* ds_row = dS_tile.data() + r * bc;
                    for (int c = 0; c < bc; ++c) {
                        const T ds = ds_row[c] * scale;
                        if (ds == T(0)) continue;
                        const T* k_row = Kb + static_cast<size_t>(j0 + c) * D;
                        for (int d = 0; d < D; ++d) {
                            dq_row[d] += ds * k_row[d];
                        }
                    }
                }
            }

            // Store completed dK and dV tile
            for (int c = 0; c < bc; ++c) {
                std::memcpy(dKb + static_cast<size_t>(j0 + c) * D,
                            dK_tile.data() + c * D, static_cast<size_t>(D) * sizeof(T));
                std::memcpy(dVb + static_cast<size_t>(j0 + c) * D,
                            dV_tile.data() + c * D, static_cast<size_t>(D) * sizeof(T));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Grouped-Query FlashAttention-2 (GQA + KV-Cache q_offset support)
// Q, O: [B, H_q, T_q, D], K, V: [B, H_kv, T_k, D], L: [B, H_q, T_q]
// Optional out_probs: [B, H_q, T_q, T_k]
// ---------------------------------------------------------------------------

template <typename T>
void flash_attn_gqa_fwd_impl(const T* __restrict__ Q, const T* __restrict__ K,
                             const T* __restrict__ V, T* __restrict__ O,
                             T* __restrict__ L, T* __restrict__ out_probs,
                             int B, int H_q, int H_kv, int T_q, int T_k, int D,
                             T scale, int causal, int q_offset) {
    constexpr int BR = 32;
    constexpr int BC = 32;
    const T neg_inf = -std::numeric_limits<T>::infinity();
    const int group_size = H_q / H_kv;

    std::vector<T> S_tile(BR * BC);
    std::vector<T> O_tile(BR * D);
    T m_vec[BR];
    T l_vec[BR];

    const size_t q_head_stride = static_cast<size_t>(T_q) * D;
    const size_t kv_head_stride = static_cast<size_t>(T_k) * D;
    const size_t l_head_stride = static_cast<size_t>(T_q);
    const size_t prob_head_stride = static_cast<size_t>(T_q) * T_k;

    for (int b = 0; b < B; ++b) {
        for (int hq = 0; hq < H_q; ++hq) {
            const int hkv = hq / group_size;
            const size_t q_idx = static_cast<size_t>(b) * H_q + hq;
            const size_t kv_idx = static_cast<size_t>(b) * H_kv + hkv;

            const T* Qb = Q + q_idx * q_head_stride;
            const T* Kb = K + kv_idx * kv_head_stride;
            const T* Vb = V + kv_idx * kv_head_stride;
            T* Ob = O + q_idx * q_head_stride;
            T* Lb = L + q_idx * l_head_stride;

            for (int i0 = 0; i0 < T_q; i0 += BR) {
                const int br = std::min(BR, T_q - i0);
                for (int r = 0; r < br; ++r) {
                    m_vec[r] = neg_inf;
                    l_vec[r] = T(0);
                }
                std::fill(O_tile.begin(), O_tile.begin() + br * D, T(0));

                const int max_q_pos = q_offset + i0 + br - 1;
                const int j_limit = causal ? std::min(T_k, max_q_pos + 1) : T_k;

                for (int j0 = 0; j0 < j_limit; j0 += BC) {
                    const int bc = std::min(BC, j_limit - j0);

                    for (int r = 0; r < br; ++r) {
                        const int qi = i0 + r;
                        const int abs_q = q_offset + qi;
                        const T* q_row = Qb + static_cast<size_t>(qi) * D;
                        T* s_row = S_tile.data() + r * bc;

                        T m_tile = neg_inf;
                        for (int c = 0; c < bc; ++c) {
                            const int kj = j0 + c;
                            if (causal && kj > abs_q) {
                                s_row[c] = neg_inf;
                            } else {
                                const T* k_row = Kb + static_cast<size_t>(kj) * D;
                                T dot = T(0);
                                for (int d = 0; d < D; ++d) {
                                    dot += q_row[d] * k_row[d];
                                }
                                const T val = dot * scale;
                                s_row[c] = val;
                                if (val > m_tile) m_tile = val;
                            }
                        }

                        if (m_tile == neg_inf) continue;

                        const T m_old = m_vec[r];
                        const T m_new = (m_old > m_tile) ? m_old : m_tile;
                        const T alpha = (m_old == neg_inf) ? T(0) : std::exp(m_old - m_new);

                        T* o_row = O_tile.data() + r * D;
                        if (alpha != T(1)) {
                            for (int d = 0; d < D; ++d) o_row[d] *= alpha;
                            l_vec[r] *= alpha;
                        }

                        T l_tile = T(0);
                        for (int c = 0; c < bc; ++c) {
                            const T sv = s_row[c];
                            if (sv == neg_inf) continue;
                            const T p = std::exp(sv - m_new);
                            l_tile += p;
                            const T* v_row = Vb + static_cast<size_t>(j0 + c) * D;
                            for (int d = 0; d < D; ++d) {
                                o_row[d] += p * v_row[d];
                            }
                        }
                        l_vec[r] += l_tile;
                        m_vec[r] = m_new;
                    }
                }

                for (int r = 0; r < br; ++r) {
                    const int qi = i0 + r;
                    const T inv_l = (l_vec[r] > T(0)) ? (T(1) / l_vec[r]) : T(0);
                    const T* o_src = O_tile.data() + r * D;
                    T* o_dst = Ob + static_cast<size_t>(qi) * D;
                    for (int d = 0; d < D; ++d) o_dst[d] = o_src[d] * inv_l;
                    Lb[qi] = m_vec[r] + std::log(l_vec[r]);
                }
            }

            if (out_probs != nullptr) {
                T* Pb = out_probs + q_idx * prob_head_stride;
                for (int i = 0; i < T_q; ++i) {
                    const int abs_q = q_offset + i;
                    const T* q_row = Qb + static_cast<size_t>(i) * D;
                    const T li = Lb[i];
                    T* p_row = Pb + static_cast<size_t>(i) * T_k;
                    for (int j = 0; j < T_k; ++j) {
                        if (causal && j > abs_q) {
                            p_row[j] = T(0);
                        } else {
                            const T* k_row = Kb + static_cast<size_t>(j) * D;
                            T dot = T(0);
                            for (int d = 0; d < D; ++d) dot += q_row[d] * k_row[d];
                            p_row[j] = std::exp(dot * scale - li);
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void flash_attn_gqa_bwd_impl(const T* __restrict__ Q, const T* __restrict__ K,
                             const T* __restrict__ V, const T* __restrict__ O,
                             const T* __restrict__ L, const T* __restrict__ dO,
                             T* __restrict__ dQ, T* __restrict__ dK, T* __restrict__ dV,
                             int B, int H_q, int H_kv, int T_q, int T_k, int D,
                             T scale, int causal, int q_offset) {
    constexpr int BR = 32;
    constexpr int BC = 32;
    const int group_size = H_q / H_kv;

    std::memset(dQ, 0, static_cast<size_t>(B) * H_q * T_q * D * sizeof(T));
    std::memset(dK, 0, static_cast<size_t>(B) * H_kv * T_k * D * sizeof(T));
    std::memset(dV, 0, static_cast<size_t>(B) * H_kv * T_k * D * sizeof(T));

    std::vector<T> D_vec(T_q);
    std::vector<T> P_tile(BR * BC);
    std::vector<T> dS_tile(BR * BC);
    std::vector<T> dK_tile(BC * D);
    std::vector<T> dV_tile(BC * D);

    const size_t q_head_stride = static_cast<size_t>(T_q) * D;
    const size_t kv_head_stride = static_cast<size_t>(T_k) * D;
    const size_t l_head_stride = static_cast<size_t>(T_q);

    for (int b = 0; b < B; ++b) {
        for (int hq = 0; hq < H_q; ++hq) {
            const int hkv = hq / group_size;
            const size_t q_idx = static_cast<size_t>(b) * H_q + hq;
            const size_t kv_idx = static_cast<size_t>(b) * H_kv + hkv;

            const T* Qb = Q + q_idx * q_head_stride;
            const T* Kb = K + kv_idx * kv_head_stride;
            const T* Vb = V + kv_idx * kv_head_stride;
            const T* Ob = O + q_idx * q_head_stride;
            const T* Lb = L + q_idx * l_head_stride;
            const T* dOb = dO + q_idx * q_head_stride;

            T* dQb = dQ + q_idx * q_head_stride;
            T* dKb = dK + kv_idx * kv_head_stride;
            T* dVb = dV + kv_idx * kv_head_stride;

            for (int i = 0; i < T_q; ++i) {
                const T* do_row = dOb + static_cast<size_t>(i) * D;
                const T* o_row = Ob + static_cast<size_t>(i) * D;
                T dot = T(0);
                for (int d = 0; d < D; ++d) dot += do_row[d] * o_row[d];
                D_vec[i] = dot;
            }

            for (int j0 = 0; j0 < T_k; j0 += BC) {
                const int bc = std::min(BC, T_k - j0);
                std::fill(dK_tile.begin(), dK_tile.begin() + bc * D, T(0));
                std::fill(dV_tile.begin(), dV_tile.begin() + bc * D, T(0));

                const int min_qi = causal ? std::max(0, j0 - q_offset) : 0;
                const int i_start = (min_qi / BR) * BR;

                for (int i0 = i_start; i0 < T_q; i0 += BR) {
                    const int br = std::min(BR, T_q - i0);

                    for (int r = 0; r < br; ++r) {
                        const int qi = i0 + r;
                        const int abs_q = q_offset + qi;
                        const T* q_row = Qb + static_cast<size_t>(qi) * D;
                        const T* do_row = dOb + static_cast<size_t>(qi) * D;
                        const T li = Lb[qi];
                        const T di = D_vec[qi];
                        T* p_row = P_tile.data() + r * bc;
                        T* ds_row = dS_tile.data() + r * bc;

                        for (int c = 0; c < bc; ++c) {
                            const int kj = j0 + c;
                            if (causal && kj > abs_q) {
                                p_row[c] = T(0);
                                ds_row[c] = T(0);
                            } else {
                                const T* k_row = Kb + static_cast<size_t>(kj) * D;
                                const T* v_row = Vb + static_cast<size_t>(kj) * D;
                                T s_dot = T(0);
                                T dp_dot = T(0);
                                for (int d = 0; d < D; ++d) {
                                    s_dot += q_row[d] * k_row[d];
                                    dp_dot += do_row[d] * v_row[d];
                                }
                                const T p = std::exp(s_dot * scale - li);
                                p_row[c] = p;
                                ds_row[c] = p * (dp_dot - di);
                            }
                        }
                    }

                    for (int c = 0; c < bc; ++c) {
                        T* dv_row = dV_tile.data() + c * D;
                        T* dk_row = dK_tile.data() + c * D;
                        for (int r = 0; r < br; ++r) {
                            const int qi = i0 + r;
                            const T p = P_tile[r * bc + c];
                            const T ds = dS_tile[r * bc + c] * scale;
                            if (p == T(0) && ds == T(0)) continue;
                            const T* do_row = dOb + static_cast<size_t>(qi) * D;
                            const T* q_row = Qb + static_cast<size_t>(qi) * D;
                            for (int d = 0; d < D; ++d) {
                                dv_row[d] += p * do_row[d];
                                dk_row[d] += ds * q_row[d];
                            }
                        }
                    }

                    for (int r = 0; r < br; ++r) {
                        const int qi = i0 + r;
                        T* dq_row = dQb + static_cast<size_t>(qi) * D;
                        const T* ds_row = dS_tile.data() + r * bc;
                        for (int c = 0; c < bc; ++c) {
                            const T ds = ds_row[c] * scale;
                            if (ds == T(0)) continue;
                            const T* k_row = Kb + static_cast<size_t>(j0 + c) * D;
                            for (int d = 0; d < D; ++d) {
                                dq_row[d] += ds * k_row[d];
                            }
                        }
                    }
                }

                // Accumulate into shared KV head gradient (crucial for GQA!)
                for (int c = 0; c < bc; ++c) {
                    T* dk_dst = dKb + static_cast<size_t>(j0 + c) * D;
                    T* dv_dst = dVb + static_cast<size_t>(j0 + c) * D;
                    const T* dk_src = dK_tile.data() + c * D;
                    const T* dv_src = dV_tile.data() + c * D;
                    for (int d = 0; d < D; ++d) {
                        dk_dst[d] += dk_src[d];
                        dv_dst[d] += dv_src[d];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fast C++17 SplitMix64 + Box-Muller Gaussian Initializer (100M params in ~0.3s)
// ---------------------------------------------------------------------------

static inline uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

template <typename T>
void fill_randn_impl(T* __restrict__ out, int n, double std_dev, uint64_t seed) {
    uint64_t st = seed ^ 0x243f6a8885a308d3ULL;
    constexpr double TWO_PI = 6.28318530717958647692;
    constexpr double INV_53 = 1.0 / 9007199254740992.0; // 2^-53
    int i = 0;
    while (i < n) {
        double u1 = static_cast<double>(splitmix64_next(st) >> 11) * INV_53;
        double u2 = static_cast<double>(splitmix64_next(st) >> 11) * INV_53;
        if (u1 < 1e-15) u1 = 1e-15;
        const double mag = std_dev * std::sqrt(-2.0 * std::log(u1));
        const double ang = TWO_PI * u2;
        out[i++] = static_cast<T>(mag * std::cos(ang));
        if (i < n) {
            out[i++] = static_cast<T>(mag * std::sin(ang));
        }
    }
}

// ---------------------------------------------------------------------------
// RMSNorm (Forward & Backward)
// x: [N, D], weight: [D], y: [N, D], inv_rms: [N]
// ---------------------------------------------------------------------------

template <typename T>
void rmsnorm_fwd_impl(const T* __restrict__ x, const T* __restrict__ w,
                      T* __restrict__ y, T* __restrict__ inv_rms_out,
                      int N, int D, T eps) {
    const T inv_d = T(1) / static_cast<T>(D);
    for (int i = 0; i < N; ++i) {
        const T* xr = x + static_cast<size_t>(i) * D;
        T* yr = y + static_cast<size_t>(i) * D;
        T ss = T(0);
        for (int j = 0; j < D; ++j) ss += xr[j] * xr[j];
        const T inv_rms = T(1) / std::sqrt(ss * inv_d + eps);
        inv_rms_out[i] = inv_rms;
        for (int j = 0; j < D; ++j) {
            yr[j] = xr[j] * inv_rms * w[j];
        }
    }
}

template <typename T>
void rmsnorm_bwd_impl(const T* __restrict__ x, const T* __restrict__ w,
                      const T* __restrict__ inv_rms_in, const T* __restrict__ dy,
                      T* __restrict__ dx, T* __restrict__ dw, int N, int D) {
    std::memset(dw, 0, static_cast<size_t>(D) * sizeof(T));
    const T inv_d = T(1) / static_cast<T>(D);
    for (int i = 0; i < N; ++i) {
        const T* xr = x + static_cast<size_t>(i) * D;
        const T* dyr = dy + static_cast<size_t>(i) * D;
        T* dxr = dx + static_cast<size_t>(i) * D;
        const T ir = inv_rms_in[i];
        const T ir3_inv_d = ir * ir * ir * inv_d;

        T dot = T(0);
        for (int j = 0; j < D; ++j) {
            const T dy_w = dyr[j] * w[j];
            dot += dy_w * xr[j];
            dw[j] += dyr[j] * xr[j] * ir;
        }
        const T coeff = dot * ir3_inv_d;
        for (int j = 0; j < D; ++j) {
            dxr[j] = dyr[j] * w[j] * ir - xr[j] * coeff;
        }
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embeddings (RoPE) Forward & Backward
// x, y: [B, H, T, D] (D must be even)
// ---------------------------------------------------------------------------

template <typename T>
void rope_impl(const T* __restrict__ x, T* __restrict__ y,
               int BH, int T_len, int D, int pos_offset, T base, int inverse) {
    const int half = D / 2;
    std::vector<T> cos_tab(static_cast<size_t>(T_len) * half);
    std::vector<T> sin_tab(static_cast<size_t>(T_len) * half);

    for (int i = 0; i < half; ++i) {
        const double inv_freq = std::pow(static_cast<double>(base), -2.0 * i / static_cast<double>(D));
        for (int t = 0; t < T_len; ++t) {
            const double angle = static_cast<double>(t + pos_offset) * inv_freq;
            cos_tab[static_cast<size_t>(t) * half + i] = static_cast<T>(std::cos(angle));
            sin_tab[static_cast<size_t>(t) * half + i] = static_cast<T>(std::sin(angle));
        }
    }

    const size_t head_stride = static_cast<size_t>(T_len) * D;
    for (int bh = 0; bh < BH; ++bh) {
        const T* xb = x + bh * head_stride;
        T* yb = y + bh * head_stride;
        for (int t = 0; t < T_len; ++t) {
            const T* xr = xb + static_cast<size_t>(t) * D;
            T* yr = yb + static_cast<size_t>(t) * D;
            const T* ct = cos_tab.data() + static_cast<size_t>(t) * half;
            const T* st = sin_tab.data() + static_cast<size_t>(t) * half;
            if (!inverse) {
                for (int i = 0; i < half; ++i) {
                    const T x0 = xr[2 * i];
                    const T x1 = xr[2 * i + 1];
                    const T c = ct[i];
                    const T s = st[i];
                    yr[2 * i]     = x0 * c - x1 * s;
                    yr[2 * i + 1] = x0 * s + x1 * c;
                }
            } else {
                for (int i = 0; i < half; ++i) {
                    const T y0 = xr[2 * i];
                    const T y1 = xr[2 * i + 1];
                    const T c = ct[i];
                    const T s = st[i];
                    yr[2 * i]     =  y0 * c + y1 * s;
                    yr[2 * i + 1] = -y0 * s + y1 * c;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Cross-Entropy Loss (Numerically Stable Log-Softmax)
// logits: [N, V], targets: int32[N]
// ---------------------------------------------------------------------------

template <typename T>
T cross_entropy_fwd_impl(const T* __restrict__ logits, const int32_t* __restrict__ targets,
                         T* __restrict__ probs_out, int N, int V, int ignore_index, int* valid_out) {
    T total_loss = T(0);
    int valid = 0;
    for (int i = 0; i < N; ++i) {
        const int tgt = targets[i];
        const T* row = logits + static_cast<size_t>(i) * V;
        T* prow = probs_out + static_cast<size_t>(i) * V;
        if (tgt == ignore_index) {
            std::memset(prow, 0, static_cast<size_t>(V) * sizeof(T));
            continue;
        }
        T max_v = row[0];
        for (int v = 1; v < V; ++v) {
            if (row[v] > max_v) max_v = row[v];
        }
        T sum_exp = T(0);
        for (int v = 0; v < V; ++v) {
            const T e = std::exp(row[v] - max_v);
            prow[v] = e;
            sum_exp += e;
        }
        const T inv_sum = T(1) / sum_exp;
        for (int v = 0; v < V; ++v) {
            prow[v] *= inv_sum;
        }
        const T log_prob = (row[tgt] - max_v) - std::log(sum_exp);
        total_loss -= log_prob;
        ++valid;
    }
    *valid_out = valid;
    return (valid > 0) ? (total_loss / static_cast<T>(valid)) : T(0);
}

template <typename T>
void cross_entropy_bwd_impl(const T* __restrict__ probs, const int32_t* __restrict__ targets,
                            T* __restrict__ dlogits, T grad_out,
                            int N, int V, int ignore_index, int valid) {
    if (valid == 0) {
        std::memset(dlogits, 0, static_cast<size_t>(N) * V * sizeof(T));
        return;
    }
    const T scale = grad_out / static_cast<T>(valid);
    for (int i = 0; i < N; ++i) {
        const int tgt = targets[i];
        const T* prow = probs + static_cast<size_t>(i) * V;
        T* drow = dlogits + static_cast<size_t>(i) * V;
        if (tgt == ignore_index) {
            std::memset(drow, 0, static_cast<size_t>(V) * sizeof(T));
            continue;
        }
        for (int v = 0; v < V; ++v) {
            drow[v] = prow[v] * scale;
        }
        drow[tgt] -= scale;
    }
}

// ---------------------------------------------------------------------------
// Broadcasted Binary Ops & Un-Broadcast Reduction
// ---------------------------------------------------------------------------

template <typename T>
void binary_bcast_impl(const T* a, const T* b, T* out,
                       const int* shape_out, const int* stride_a, const int* stride_b,
                       int ndim, int total, int op) {
    // Fast contiguous path for identical shapes
    int fast_same = 1;
    int expected = 1;
    for (int d = ndim - 1; d >= 0; --d) {
        if (stride_a[d] != expected || stride_b[d] != expected) {
            fast_same = 0;
            break;
        }
        expected *= shape_out[d];
    }
    if (fast_same) {
        if (op == 0) {
            for (int i = 0; i < total; ++i) out[i] = a[i] + b[i];
        } else if (op == 1) {
            for (int i = 0; i < total; ++i) out[i] = a[i] - b[i];
        } else if (op == 2) {
            for (int i = 0; i < total; ++i) out[i] = a[i] * b[i];
        } else {
            for (int i = 0; i < total; ++i) out[i] = a[i] / b[i];
        }
        return;
    }

    int coords[8] = {0};
    for (int idx = 0; idx < total; ++idx) {
        int off_a = 0, off_b = 0;
        for (int d = 0; d < ndim; ++d) {
            off_a += coords[d] * stride_a[d];
            off_b += coords[d] * stride_b[d];
        }
        const T va = a[off_a];
        const T vb = b[off_b];
        if (op == 0) out[idx] = va + vb;
        else if (op == 1) out[idx] = va - vb;
        else if (op == 2) out[idx] = va * vb;
        else out[idx] = va / vb;

        for (int d = ndim - 1; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < shape_out[d]) break;
            coords[d] = 0;
        }
    }
}

template <typename T>
void reduce_bcast_impl(const T* grad_in, T* grad_out,
                       const int* shape_in, const int* stride_out,
                       int ndim, int total_in, int total_out) {
    std::memset(grad_out, 0, static_cast<size_t>(total_out) * sizeof(T));
    int coords[8] = {0};
    for (int idx = 0; idx < total_in; ++idx) {
        int off = 0;
        for (int d = 0; d < ndim; ++d) {
            off += coords[d] * stride_out[d];
        }
        grad_out[off] += grad_in[idx];
        for (int d = ndim - 1; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < shape_in[d]) break;
            coords[d] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Activations (SiLU, GELU, ReLU, Tanh, Exp, Log, Softmax)
// ---------------------------------------------------------------------------

template <typename T>
void unary_op_impl(const T* x, T* y, int n, int op) {
    if (op == 0) { // SiLU: x * sigmoid(x)
        for (int i = 0; i < n; ++i) {
            const T v = x[i];
            const T sig = T(1) / (T(1) + std::exp(-v));
            y[i] = v * sig;
        }
    } else if (op == 1) { // ReLU
        for (int i = 0; i < n; ++i) y[i] = (x[i] > T(0)) ? x[i] : T(0);
    } else if (op == 2) { // Tanh
        for (int i = 0; i < n; ++i) y[i] = std::tanh(x[i]);
    } else if (op == 3) { // Exp
        for (int i = 0; i < n; ++i) y[i] = std::exp(x[i]);
    } else if (op == 4) { // Log
        for (int i = 0; i < n; ++i) y[i] = std::log(x[i]);
    } else if (op == 5) { // Neg
        for (int i = 0; i < n; ++i) y[i] = -x[i];
    } else if (op == 6) { // GELU (tanh approximation)
        constexpr double c = 0.7978845608028654; // sqrt(2/pi)
        for (int i = 0; i < n; ++i) {
            const double v = static_cast<double>(x[i]);
            const double inner = c * (v + 0.044715 * v * v * v);
            y[i] = static_cast<T>(0.5 * v * (1.0 + std::tanh(inner)));
        }
    }
}

template <typename T>
void unary_bwd_impl(const T* x, const T* dy, T* dx, int n, int op) {
    if (op == 0) { // SiLU backward: sig(x) * (1 + x * (1 - sig(x)))
        for (int i = 0; i < n; ++i) {
            const T v = x[i];
            const T sig = T(1) / (T(1) + std::exp(-v));
            dx[i] = dy[i] * sig * (T(1) + v * (T(1) - sig));
        }
    } else if (op == 1) { // ReLU backward
        for (int i = 0; i < n; ++i) dx[i] = (x[i] > T(0)) ? dy[i] : T(0);
    } else if (op == 2) { // Tanh backward
        for (int i = 0; i < n; ++i) {
            const T t = std::tanh(x[i]);
            dx[i] = dy[i] * (T(1) - t * t);
        }
    } else if (op == 3) { // Exp backward
        for (int i = 0; i < n; ++i) dx[i] = dy[i] * std::exp(x[i]);
    } else if (op == 4) { // Log backward
        for (int i = 0; i < n; ++i) dx[i] = dy[i] / x[i];
    } else if (op == 5) { // Neg backward
        for (int i = 0; i < n; ++i) dx[i] = -dy[i];
    } else if (op == 6) { // GELU backward
        constexpr double c = 0.7978845608028654;
        for (int i = 0; i < n; ++i) {
            const double v = static_cast<double>(x[i]);
            const double inner = c * (v + 0.044715 * v * v * v);
            const double th = std::tanh(inner);
            const double sech2 = 1.0 - th * th;
            const double d_inner = c * (1.0 + 3.0 * 0.044715 * v * v);
            const double grad = 0.5 * (1.0 + th) + 0.5 * v * sech2 * d_inner;
            dx[i] = static_cast<T>(static_cast<double>(dy[i]) * grad);
        }
    }
}

template <typename T>
void softmax_fwd_impl(const T* x, T* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const T* xr = x + static_cast<size_t>(r) * cols;
        T* yr = y + static_cast<size_t>(r) * cols;
        T mx = xr[0];
        for (int c = 1; c < cols; ++c) if (xr[c] > mx) mx = xr[c];
        T sum = T(0);
        for (int c = 0; c < cols; ++c) {
            const T e = std::exp(xr[c] - mx);
            yr[c] = e;
            sum += e;
        }
        const T inv = T(1) / sum;
        for (int c = 0; c < cols; ++c) yr[c] *= inv;
    }
}

template <typename T>
void softmax_bwd_impl(const T* y, const T* dy, T* dx, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const T* yr = y + static_cast<size_t>(r) * cols;
        const T* dyr = dy + static_cast<size_t>(r) * cols;
        T* dxr = dx + static_cast<size_t>(r) * cols;
        T dot = T(0);
        for (int c = 0; c < cols; ++c) dot += yr[c] * dyr[c];
        for (int c = 0; c < cols; ++c) dxr[c] = yr[c] * (dyr[c] - dot);
    }
}

// ---------------------------------------------------------------------------
// Embedding Forward / Backward
// ---------------------------------------------------------------------------

template <typename T>
void embedding_fwd_impl(const T* table, const int32_t* idx, T* out, int n_idx, int D) {
    for (int i = 0; i < n_idx; ++i) {
        const int id = idx[i];
        std::memcpy(out + static_cast<size_t>(i) * D,
                    table + static_cast<size_t>(id) * D,
                    static_cast<size_t>(D) * sizeof(T));
    }
}

template <typename T>
void embedding_bwd_impl(const T* dout, const int32_t* idx, T* dtable,
                        int n_idx, int vocab, int D) {
    std::memset(dtable, 0, static_cast<size_t>(vocab) * D * sizeof(T));
    for (int i = 0; i < n_idx; ++i) {
        const int id = idx[i];
        const T* src = dout + static_cast<size_t>(i) * D;
        T* dst = dtable + static_cast<size_t>(id) * D;
        for (int d = 0; d < D; ++d) dst[d] += src[d];
    }
}

// ---------------------------------------------------------------------------
// General N-D Permute / Transpose
// ---------------------------------------------------------------------------

template <typename T>
void permute_impl(const T* src, T* dst, const int* in_strides,
                  const int* out_shape, const int* perm, int ndim, int total) {
    int mapped_strides[8];
    for (int d = 0; d < ndim; ++d) mapped_strides[d] = in_strides[perm[d]];
    int coords[8] = {0};
    for (int idx = 0; idx < total; ++idx) {
        int in_off = 0;
        for (int d = 0; d < ndim; ++d) in_off += coords[d] * mapped_strides[d];
        dst[idx] = src[in_off];
        for (int d = ndim - 1; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < out_shape[d]) break;
            coords[d] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Fused AdamW Optimizer Step (returns update L2 norm squared and param L2 norm squared)
// ---------------------------------------------------------------------------

template <typename T>
void adamw_step_impl(T* __restrict__ p, const T* __restrict__ g,
                     T* __restrict__ m, T* __restrict__ v,
                     int n, double lr, double beta1, double beta2,
                     double eps, double weight_decay, int step,
                     double* out_update_sq, double* out_param_sq) {
    const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(step));
    const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(step));
    const double step_size = lr / bc1;
    const double inv_sqrt_bc2 = 1.0 / std::sqrt(bc2);
    const double decay_factor = 1.0 - lr * weight_decay;

    double upd_sq = 0.0;
    double prm_sq = 0.0;

    for (int i = 0; i < n; ++i) {
        const double gi = static_cast<double>(g[i]);
        const double mi = beta1 * static_cast<double>(m[i]) + (1.0 - beta1) * gi;
        const double vi = beta2 * static_cast<double>(v[i]) + (1.0 - beta2) * gi * gi;
        m[i] = static_cast<T>(mi);
        v[i] = static_cast<T>(vi);

        const double old_p = static_cast<double>(p[i]);
        const double decayed = old_p * decay_factor;
        const double new_p = decayed - step_size * mi / (std::sqrt(vi) * inv_sqrt_bc2 + eps);
        p[i] = static_cast<T>(new_p);

        const double delta = new_p - old_p;
        upd_sq += delta * delta;
        prm_sq += new_p * new_p;
    }
    *out_update_sq = upd_sq;
    *out_param_sq = prm_sq;
}

template <typename T>
double l2_norm_sq_impl(const T* x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = static_cast<double>(x[i]);
        s += v * v;
    }
    return s;
}

template <typename T>
void scale_inplace_impl(T* x, int n, double factor) {
    const T f = static_cast<T>(factor);
    for (int i = 0; i < n; ++i) x[i] *= f;
}

template <typename T>
void add_inplace_impl(T* dst, const T* src, int n, double alpha) {
    const T a = static_cast<T>(alpha);
    for (int i = 0; i < n; ++i) dst[i] += a * src[i];
}

} // namespace

// ===========================================================================
// Extern "C" ABI (dtype: 0 = float32, 1 = float64)
// ===========================================================================

extern "C" {

void gb_reset_stats() {
    g_matmul_flops.store(0);
    g_matmul_calls.store(0);
    g_matmul_ns.store(0);
}

void gb_get_stats(uint64_t* flops, uint64_t* calls, uint64_t* ns) {
    *flops = g_matmul_flops.load();
    *calls = g_matmul_calls.load();
    *ns = g_matmul_ns.load();
}

void gb_matmul_tiled(const void* A, const void* B, void* C,
                     int batch, int batch_a, int batch_b,
                     int M, int K, int N,
                     int trans_a, int trans_b, int accumulate, int dtype) {
    if (dtype == 0) {
        batched_gemm_tiled<float>(
            static_cast<const float*>(A), static_cast<const float*>(B),
            static_cast<float*>(C), batch, batch_a, batch_b,
            M, K, N, trans_a, trans_b, accumulate);
    } else {
        batched_gemm_tiled<double>(
            static_cast<const double*>(A), static_cast<const double*>(B),
            static_cast<double*>(C), batch, batch_a, batch_b,
            M, K, N, trans_a, trans_b, accumulate);
    }
}

void gb_matmul_naive(const void* A, const void* B, void* C,
                     int M, int K, int N,
                     int trans_a, int trans_b, int accumulate, int dtype) {
    if (dtype == 0) {
        gemm_naive_2d<float>(
            static_cast<const float*>(A), static_cast<const float*>(B),
            static_cast<float*>(C), M, K, N, trans_a, trans_b, accumulate);
    } else {
        gemm_naive_2d<double>(
            static_cast<const double*>(A), static_cast<const double*>(B),
            static_cast<double*>(C), M, K, N, trans_a, trans_b, accumulate);
    }
}

void gb_flash_attn_fwd(const void* Q, const void* K, const void* V,
                       void* O, void* L, void* out_probs,
                       int BH, int T_len, int D, double scale, int causal, int dtype) {
    if (dtype == 0) {
        flash_attn_fwd_impl<float>(
            static_cast<const float*>(Q), static_cast<const float*>(K),
            static_cast<const float*>(V), static_cast<float*>(O),
            static_cast<float*>(L), static_cast<float*>(out_probs),
            BH, T_len, D, static_cast<float>(scale), causal);
    } else {
        flash_attn_fwd_impl<double>(
            static_cast<const double*>(Q), static_cast<const double*>(K),
            static_cast<const double*>(V), static_cast<double*>(O),
            static_cast<double*>(L), static_cast<double*>(out_probs),
            BH, T_len, D, scale, causal);
    }
}

void gb_flash_attn_bwd(const void* Q, const void* K, const void* V,
                       const void* O, const void* L, const void* dO,
                       void* dQ, void* dK, void* dV,
                       int BH, int T_len, int D, double scale, int causal, int dtype) {
    if (dtype == 0) {
        flash_attn_bwd_impl<float>(
            static_cast<const float*>(Q), static_cast<const float*>(K),
            static_cast<const float*>(V), static_cast<const float*>(O),
            static_cast<const float*>(L), static_cast<const float*>(dO),
            static_cast<float*>(dQ), static_cast<float*>(dK), static_cast<float*>(dV),
            BH, T_len, D, static_cast<float>(scale), causal);
    } else {
        flash_attn_bwd_impl<double>(
            static_cast<const double*>(Q), static_cast<const double*>(K),
            static_cast<const double*>(V), static_cast<const double*>(O),
            static_cast<const double*>(L), static_cast<const double*>(dO),
            static_cast<double*>(dQ), static_cast<double*>(dK), static_cast<double*>(dV),
            BH, T_len, D, scale, causal);
    }
}

void gb_rmsnorm_fwd(const void* x, const void* w, void* y, void* inv_rms,
                    int N, int D, double eps, int dtype) {
    if (dtype == 0) {
        rmsnorm_fwd_impl<float>(
            static_cast<const float*>(x), static_cast<const float*>(w),
            static_cast<float*>(y), static_cast<float*>(inv_rms),
            N, D, static_cast<float>(eps));
    } else {
        rmsnorm_fwd_impl<double>(
            static_cast<const double*>(x), static_cast<const double*>(w),
            static_cast<double*>(y), static_cast<double*>(inv_rms),
            N, D, eps);
    }
}

void gb_rmsnorm_bwd(const void* x, const void* w, const void* inv_rms, const void* dy,
                    void* dx, void* dw, int N, int D, int dtype) {
    if (dtype == 0) {
        rmsnorm_bwd_impl<float>(
            static_cast<const float*>(x), static_cast<const float*>(w),
            static_cast<const float*>(inv_rms), static_cast<const float*>(dy),
            static_cast<float*>(dx), static_cast<float*>(dw), N, D);
    } else {
        rmsnorm_bwd_impl<double>(
            static_cast<const double*>(x), static_cast<const double*>(w),
            static_cast<const double*>(inv_rms), static_cast<const double*>(dy),
            static_cast<double*>(dx), static_cast<double*>(dw), N, D);
    }
}

void gb_rope(const void* x, void* y, int BH, int T_len, int D,
             int pos_offset, double base, int inverse, int dtype) {
    if (dtype == 0) {
        rope_impl<float>(static_cast<const float*>(x), static_cast<float*>(y),
                         BH, T_len, D, pos_offset, static_cast<float>(base), inverse);
    } else {
        rope_impl<double>(static_cast<const double*>(x), static_cast<double*>(y),
                          BH, T_len, D, pos_offset, base, inverse);
    }
}

double gb_cross_entropy_fwd(const void* logits, const int32_t* targets, void* probs,
                            int N, int V, int ignore_index, int* valid_out, int dtype) {
    if (dtype == 0) {
        return static_cast<double>(cross_entropy_fwd_impl<float>(
            static_cast<const float*>(logits), targets, static_cast<float*>(probs),
            N, V, ignore_index, valid_out));
    } else {
        return cross_entropy_fwd_impl<double>(
            static_cast<const double*>(logits), targets, static_cast<double*>(probs),
            N, V, ignore_index, valid_out);
    }
}

void gb_cross_entropy_bwd(const void* probs, const int32_t* targets, void* dlogits,
                          double grad_out, int N, int V, int ignore_index, int valid, int dtype) {
    if (dtype == 0) {
        cross_entropy_bwd_impl<float>(
            static_cast<const float*>(probs), targets, static_cast<float*>(dlogits),
            static_cast<float>(grad_out), N, V, ignore_index, valid);
    } else {
        cross_entropy_bwd_impl<double>(
            static_cast<const double*>(probs), targets, static_cast<double*>(dlogits),
            grad_out, N, V, ignore_index, valid);
    }
}

void gb_binary_bcast(const void* a, const void* b, void* out,
                     const int* shape_out, const int* stride_a, const int* stride_b,
                     int ndim, int total, int op, int dtype) {
    if (dtype == 0) {
        binary_bcast_impl<float>(
            static_cast<const float*>(a), static_cast<const float*>(b),
            static_cast<float*>(out), shape_out, stride_a, stride_b, ndim, total, op);
    } else {
        binary_bcast_impl<double>(
            static_cast<const double*>(a), static_cast<const double*>(b),
            static_cast<double*>(out), shape_out, stride_a, stride_b, ndim, total, op);
    }
}

void gb_reduce_bcast(const void* grad_in, void* grad_out,
                     const int* shape_in, const int* stride_out,
                     int ndim, int total_in, int total_out, int dtype) {
    if (dtype == 0) {
        reduce_bcast_impl<float>(
            static_cast<const float*>(grad_in), static_cast<float*>(grad_out),
            shape_in, stride_out, ndim, total_in, total_out);
    } else {
        reduce_bcast_impl<double>(
            static_cast<const double*>(grad_in), static_cast<double*>(grad_out),
            shape_in, stride_out, ndim, total_in, total_out);
    }
}

void gb_unary_fwd(const void* x, void* y, int n, int op, int dtype) {
    if (dtype == 0) unary_op_impl<float>(static_cast<const float*>(x), static_cast<float*>(y), n, op);
    else unary_op_impl<double>(static_cast<const double*>(x), static_cast<double*>(y), n, op);
}

void gb_unary_bwd(const void* x, const void* dy, void* dx, int n, int op, int dtype) {
    if (dtype == 0) {
        unary_bwd_impl<float>(static_cast<const float*>(x), static_cast<const float*>(dy),
                              static_cast<float*>(dx), n, op);
    } else {
        unary_bwd_impl<double>(static_cast<const double*>(x), static_cast<const double*>(dy),
                               static_cast<double*>(dx), n, op);
    }
}

void gb_softmax_fwd(const void* x, void* y, int rows, int cols, int dtype) {
    if (dtype == 0) softmax_fwd_impl<float>(static_cast<const float*>(x), static_cast<float*>(y), rows, cols);
    else softmax_fwd_impl<double>(static_cast<const double*>(x), static_cast<double*>(y), rows, cols);
}

void gb_softmax_bwd(const void* y, const void* dy, void* dx, int rows, int cols, int dtype) {
    if (dtype == 0) {
        softmax_bwd_impl<float>(static_cast<const float*>(y), static_cast<const float*>(dy),
                                static_cast<float*>(dx), rows, cols);
    } else {
        softmax_bwd_impl<double>(static_cast<const double*>(y), static_cast<const double*>(dy),
                                 static_cast<double*>(dx), rows, cols);
    }
}

void gb_embedding_fwd(const void* table, const int32_t* idx, void* out,
                      int n_idx, int D, int dtype) {
    if (dtype == 0) {
        embedding_fwd_impl<float>(static_cast<const float*>(table), idx,
                                  static_cast<float*>(out), n_idx, D);
    } else {
        embedding_fwd_impl<double>(static_cast<const double*>(table), idx,
                                   static_cast<double*>(out), n_idx, D);
    }
}

void gb_embedding_bwd(const void* dout, const int32_t* idx, void* dtable,
                      int n_idx, int vocab, int D, int dtype) {
    if (dtype == 0) {
        embedding_bwd_impl<float>(static_cast<const float*>(dout), idx,
                                  static_cast<float*>(dtable), n_idx, vocab, D);
    } else {
        embedding_bwd_impl<double>(static_cast<const double*>(dout), idx,
                                   static_cast<double*>(dtable), n_idx, vocab, D);
    }
}

void gb_permute(const void* src, void* dst, const int* in_strides,
                const int* out_shape, const int* perm, int ndim, int total, int dtype) {
    if (dtype == 0) {
        permute_impl<float>(static_cast<const float*>(src), static_cast<float*>(dst),
                            in_strides, out_shape, perm, ndim, total);
    } else {
        permute_impl<double>(static_cast<const double*>(src), static_cast<double*>(dst),
                             in_strides, out_shape, perm, ndim, total);
    }
}

void gb_adamw_step(void* p, const void* g, void* m, void* v,
                   int n, double lr, double beta1, double beta2,
                   double eps, double weight_decay, int step,
                   double* out_update_sq, double* out_param_sq, int dtype) {
    if (dtype == 0) {
        adamw_step_impl<float>(static_cast<float*>(p), static_cast<const float*>(g),
                               static_cast<float*>(m), static_cast<float*>(v),
                               n, lr, beta1, beta2, eps, weight_decay, step,
                               out_update_sq, out_param_sq);
    } else {
        adamw_step_impl<double>(static_cast<double*>(p), static_cast<const double*>(g),
                                static_cast<double*>(m), static_cast<double*>(v),
                                n, lr, beta1, beta2, eps, weight_decay, step,
                                out_update_sq, out_param_sq);
    }
}

double gb_l2_norm_sq(const void* x, int n, int dtype) {
    if (dtype == 0) return l2_norm_sq_impl<float>(static_cast<const float*>(x), n);
    return l2_norm_sq_impl<double>(static_cast<const double*>(x), n);
}

void gb_scale_inplace(void* x, int n, double factor, int dtype) {
    if (dtype == 0) scale_inplace_impl<float>(static_cast<float*>(x), n, factor);
    else scale_inplace_impl<double>(static_cast<double*>(x), n, factor);
}

void gb_add_inplace(void* dst, const void* src, int n, double alpha, int dtype) {
    if (dtype == 0) add_inplace_impl<float>(static_cast<float*>(dst), static_cast<const float*>(src), n, alpha);
    else add_inplace_impl<double>(static_cast<double*>(dst), static_cast<const double*>(src), n, alpha);
}

void gb_flash_attn_gqa_fwd(const void* Q, const void* K, const void* V,
                           void* O, void* L, void* out_probs,
                           int B, int H_q, int H_kv, int T_q, int T_k, int D,
                           double scale, int causal, int q_offset, int dtype) {
    if (dtype == 0) {
        flash_attn_gqa_fwd_impl<float>(
            static_cast<const float*>(Q), static_cast<const float*>(K),
            static_cast<const float*>(V), static_cast<float*>(O),
            static_cast<float*>(L), static_cast<float*>(out_probs),
            B, H_q, H_kv, T_q, T_k, D, static_cast<float>(scale), causal, q_offset);
    } else {
        flash_attn_gqa_fwd_impl<double>(
            static_cast<const double*>(Q), static_cast<const double*>(K),
            static_cast<const double*>(V), static_cast<double*>(O),
            static_cast<double*>(L), static_cast<double*>(out_probs),
            B, H_q, H_kv, T_q, T_k, D, scale, causal, q_offset);
    }
}

void gb_flash_attn_gqa_bwd(const void* Q, const void* K, const void* V,
                           const void* O, const void* L, const void* dO,
                           void* dQ, void* dK, void* dV,
                           int B, int H_q, int H_kv, int T_q, int T_k, int D,
                           double scale, int causal, int q_offset, int dtype) {
    if (dtype == 0) {
        flash_attn_gqa_bwd_impl<float>(
            static_cast<const float*>(Q), static_cast<const float*>(K),
            static_cast<const float*>(V), static_cast<const float*>(O),
            static_cast<const float*>(L), static_cast<const float*>(dO),
            static_cast<float*>(dQ), static_cast<float*>(dK), static_cast<float*>(dV),
            B, H_q, H_kv, T_q, T_k, D, static_cast<float>(scale), causal, q_offset);
    } else {
        flash_attn_gqa_bwd_impl<double>(
            static_cast<const double*>(Q), static_cast<const double*>(K),
            static_cast<const double*>(V), static_cast<const double*>(O),
            static_cast<const double*>(L), static_cast<const double*>(dO),
            static_cast<double*>(dQ), static_cast<double*>(dK), static_cast<double*>(dV),
            B, H_q, H_kv, T_q, T_k, D, scale, causal, q_offset);
    }
}

void gb_fill_randn(void* out, int n, double std_dev, uint64_t seed, int dtype) {
    if (dtype == 0) {
        fill_randn_impl<float>(static_cast<float*>(out), n, std_dev, seed);
    } else {
        fill_randn_impl<double>(static_cast<double*>(out), n, std_dev, seed);
    }
}

} // extern "C"
