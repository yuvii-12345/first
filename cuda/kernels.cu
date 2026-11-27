// ============================================================================
// Glassbox-100M Pure From-Scratch CUDA C++17 Compute Kernels
// Zero PyTorch, Zero cuDNN. Compiles standalone with nvcc -O3 -std=c++17.
// Optional -DGBX_USE_CUBLAS enables cuBLAS TF32/Tensor-Core GEMM on Ampere/Hopper.
// ============================================================================

#include "glassbox_cuda.cuh"
#include <cmath>

namespace glassbox_cuda {

void CudaContext::init() {
    GBX_CUDA_CHECK(cudaStreamCreate(&stream));
#ifdef GBX_USE_CUBLAS
    if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[Glassbox CUDA] cublasCreate failed\n");
        std::exit(1);
    }
    cublasSetStream(cublas, stream);
    cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH);
#endif
}

void CudaContext::destroy() {
#ifdef GBX_USE_CUBLAS
    if (cublas) cublasDestroy(cublas);
#endif
    if (stream) cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// 1. 2D Shared-Memory + 4x4 Register-Blocked Tiled SGEMM (64x64x16 tiles)
//    C[M, N] = (accum ? C : 0) + op(A)[M, K] @ op(B)[K, N]
// ---------------------------------------------------------------------------

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int K, int N,
                                  int trans_a, int trans_b, int accum) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;  // 0..15
    const int ty = threadIdx.y;  // 0..15
    const int tid = ty * 16 + tx; // 0..255

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float reg_c[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            reg_c[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load 4 elements per thread into shared tiles As[BK][BM] and Bs[BK][BN]
        #pragma unroll
        for (int load_idx = 0; load_idx < 4; ++load_idx) {
            const int idx = tid + load_idx * 256; // 0..1023
            // As: [BK=16][BM=64]
            const int a_k = idx / BM;
            const int a_m = idx % BM;
            const int gm = block_row + a_m;
            const int gk_a = k0 + a_k;
            float va = 0.0f;
            if (gm < M && gk_a < K) {
                va = trans_a ? A[static_cast<size_t>(gk_a) * M + gm]
                             : A[static_cast<size_t>(gm) * K + gk_a];
            }
            As[a_k][a_m] = va;

            // Bs: [BK=16][BN=64]
            const int b_k = idx / BN;
            const int b_n = idx % BN;
            const int gk_b = k0 + b_k;
            const int gn = block_col + b_n;
            float vb = 0.0f;
            if (gk_b < K && gn < N) {
                vb = trans_b ? B[static_cast<size_t>(gn) * K + gk_b]
                             : B[static_cast<size_t>(gk_b) * N + gn];
            }
            Bs[b_k][b_n] = vb;
        }
        __syncthreads();

        // Outer-product accumulation in registers (4x4 per thread)
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float reg_a[TM];
            float reg_b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) reg_a[i] = As[kk][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) reg_b[j] = Bs[kk][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    reg_c[i][j] = fmaf(reg_a[i], reg_b[j], reg_c[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Write back 4x4 tile
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gm = block_row + ty * TM + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gn = block_col + tx * TN + j;
            if (gn >= N) continue;
            const size_t off = static_cast<size_t>(gm) * N + gn;
            if (accum) C[off] += reg_c[i][j];
            else C[off] = reg_c[i][j];
        }
    }
}

void gemm(CudaContext& ctx,
          const float* A, const float* B, float* C,
          int M, int K, int N,
          bool trans_a, bool trans_b, bool accumulate) {
#ifdef GBX_USE_CUBLAS
    // Row-major C[M,N] = op(A)[M,K] @ op(B)[K,N]  <==>  Col-major C^T[N,M] = op(B)^T @ op(A)^T
    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldb = trans_b ? K : N;
    int lda = trans_a ? M : K;
    cublasSgemm(ctx.cublas, op_b, op_a, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
#else
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_tiled_kernel<<<grid, block, 0, ctx.stream>>>(
        A, B, C, M, K, N, trans_a ? 1 : 0, trans_b ? 1 : 0, accumulate ? 1 : 0);
#endif
}

// ---------------------------------------------------------------------------
// 2. Embedding Lookup & Tied Weight Backward (atomicAdd)
// ---------------------------------------------------------------------------

__global__ void embedding_fwd_kernel(const float* __restrict__ table,
                                     const int32_t* __restrict__ ids,
                                     float* __restrict__ out,
                                     int N, int D) {
    const int n = blockIdx.x;
    const int tok = ids[n];
    const float* src = table + static_cast<size_t>(tok) * D;
    float* dst = out + static_cast<size_t>(n) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        dst[d] = src[d];
    }
}

__global__ void embedding_bwd_kernel(const float* __restrict__ dout,
                                     const int32_t* __restrict__ ids,
                                     float* __restrict__ dtable,
                                     int N, int D) {
    const int n = blockIdx.x;
    const int tok = ids[n];
    const float* src = dout + static_cast<size_t>(n) * D;
    float* dst = dtable + static_cast<size_t>(tok) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        atomicAdd(&dst[d], src[d]);
    }
}

void embedding_fwd(CudaContext& ctx, const float* table, const int32_t* ids,
                   float* out, int N, int D) {
    int threads = (D < 256) ? D : 256;
    embedding_fwd_kernel<<<N, threads, 0, ctx.stream>>>(table, ids, out, N, D);
}

void embedding_bwd(CudaContext& ctx, const float* dout, const int32_t* ids,
                   float* dtable, int N, int D) {
    int threads = (D < 256) ? D : 256;
    embedding_bwd_kernel<<<N, threads, 0, ctx.stream>>>(dout, ids, dtable, N, D);
}

// ---------------------------------------------------------------------------
// 3. Fused RMSNorm Forward & Backward (Warp-Shuffle Reduction)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < nwarps) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    __shared__ float total;
    if (threadIdx.x == 0) total = val;
    __syncthreads();
    return total;
}

__global__ void rmsnorm_fwd_kernel(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   float* __restrict__ y,
                                   float* __restrict__ inv_rms_out,
                                   int N, int D, float eps) {
    const int row = blockIdx.x;
    const float* xr = x + static_cast<size_t>(row) * D;
    float* yr = y + static_cast<size_t>(row) * D;

    float ss = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = xr[d];
        ss = fmaf(v, v, ss);
    }
    ss = block_reduce_sum(ss);
    float inv_rms = rsqrtf(ss / static_cast<float>(D) + eps);
    if (threadIdx.x == 0) inv_rms_out[row] = inv_rms;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        yr[d] = xr[d] * inv_rms * w[d];
    }
}

__global__ void rmsnorm_bwd_kernel(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   const float* __restrict__ inv_rms_buf,
                                   const float* __restrict__ dy,
                                   float* __restrict__ dx,
                                   float* __restrict__ dw,
                                   int N, int D) {
    const int row = blockIdx.x;
    const float* xr = x + static_cast<size_t>(row) * D;
    const float* dyr = dy + static_cast<size_t>(row) * D;
    float* dxr = dx + static_cast<size_t>(row) * D;
    const float inv_rms = inv_rms_buf[row];

    float dot = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        dot = fmaf(dyr[d] * w[d], xr[d], dot);
    }
    dot = block_reduce_sum(dot);
    const float c = dot * inv_rms * inv_rms / static_cast<float>(D);

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float xval = xr[d];
        const float dyval = dyr[d];
        dxr[d] = inv_rms * (dyval * w[d] - xval * c);
        atomicAdd(&dw[d], dyval * xval * inv_rms);
    }
}

void rmsnorm_fwd(CudaContext& ctx, const float* x, const float* w,
                 float* y, float* inv_rms, int N, int D, float eps) {
    rmsnorm_fwd_kernel<<<N, 256, 0, ctx.stream>>>(x, w, y, inv_rms, N, D, eps);
}

void rmsnorm_bwd(CudaContext& ctx, const float* x, const float* w,
                 const float* inv_rms, const float* dy,
                 float* dx, float* dw, int N, int D) {
    rmsnorm_bwd_kernel<<<N, 256, 0, ctx.stream>>>(x, w, inv_rms, dy, dx, dw, N, D);
}

// ---------------------------------------------------------------------------
// 4. Head Layout Permute & Interleaved RoPE (2i, 2i+1)
// ---------------------------------------------------------------------------

__global__ void permute_bthd_to_bhtd_kernel(const float* __restrict__ src,
                                            float* __restrict__ dst,
                                            int B, int T, int H, int Dh) {
    const size_t total = static_cast<size_t>(B) * T * H * Dh;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = static_cast<int>(idx % Dh);
    size_t rem = idx / Dh;
    int t = static_cast<int>(rem % T);
    rem /= T;
    int h = static_cast<int>(rem % H);
    int b = static_cast<int>(rem / H);
    size_t src_off = ((static_cast<size_t>(b) * T + t) * H + h) * Dh + d;
    dst[idx] = src[src_off];
}

__global__ void permute_bhtd_to_bthd_kernel(const float* __restrict__ src,
                                            float* __restrict__ dst,
                                            int B, int H, int T, int Dh) {
    const size_t total = static_cast<size_t>(B) * H * T * Dh;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = static_cast<int>(idx % Dh);
    size_t rem = idx / Dh;
    int h = static_cast<int>(rem % H);
    rem /= H;
    int t = static_cast<int>(rem % T);
    int b = static_cast<int>(rem / T);
    size_t src_off = ((static_cast<size_t>(b) * H + h) * T + t) * Dh + d;
    dst[idx] = src[src_off];
}

void permute_bthd_to_bhtd(CudaContext& ctx, const float* src, float* dst,
                          int B, int T, int H, int Dh) {
    size_t total = static_cast<size_t>(B) * T * H * Dh;
    int blocks = static_cast<int>((total + 255) / 256);
    permute_bthd_to_bhtd_kernel<<<blocks, 256, 0, ctx.stream>>>(src, dst, B, T, H, Dh);
}

void permute_bhtd_to_bthd(CudaContext& ctx, const float* src, float* dst,
                          int B, int H, int T, int Dh) {
    size_t total = static_cast<size_t>(B) * H * T * Dh;
    int blocks = static_cast<int>((total + 255) / 256);
    permute_bhtd_to_bthd_kernel<<<blocks, 256, 0, ctx.stream>>>(src, dst, B, H, T, Dh);
}

__global__ void rope_kernel(const float* __restrict__ x, float* __restrict__ y,
                            int BH, int T, int Dh, int pos_offset,
                            float base, int inverse) {
    const int half = Dh / 2;
    const size_t total_pairs = static_cast<size_t>(BH) * T * half;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int pair = static_cast<int>(idx % half);
    size_t rem = idx / half;
    int t = static_cast<int>(rem % T);
    int bh = static_cast<int>(rem / T);

    float pos = static_cast<float>(pos_offset + t);
    float inv_freq = powf(base, -2.0f * static_cast<float>(pair) / static_cast<float>(Dh));
    float angle = pos * inv_freq;
    float c = cosf(angle);
    float s = inverse ? -sinf(angle) : sinf(angle);

    size_t off = (static_cast<size_t>(bh) * T + t) * Dh + 2 * pair;
    float x0 = x[off];
    float x1 = x[off + 1];
    y[off]     = x0 * c - x1 * s;
    y[off + 1] = x0 * s + x1 * c;
}

void rope_apply(CudaContext& ctx, const float* x, float* y,
                int BH, int T, int Dh, int pos_offset, float base, bool inverse) {
    size_t total_pairs = static_cast<size_t>(BH) * T * (Dh / 2);
    int blocks = static_cast<int>((total_pairs + 255) / 256);
    rope_kernel<<<blocks, 256, 0, ctx.stream>>>(
        x, y, BH, T, Dh, pos_offset, base, inverse ? 1 : 0);
}

// ---------------------------------------------------------------------------
// 5. FlashAttention-2 Grouped-Query Attention (GQA) Forward & Backward
//    Zero O(T^2) HBM allocation — online softmax tiled in SRAM!
// ---------------------------------------------------------------------------

constexpr int FA_BR = 32;
constexpr int FA_BC = 32;

__global__ void flash_attn_gqa_fwd_kernel(const float* __restrict__ Q,
                                          const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          float* __restrict__ O,
                                          float* __restrict__ L,
                                          int B, int H_q, int H_kv,
                                          int T, int Dh, float scale) {
    // One thread per query row inside a BR=32 tile (warp-sized block)
    const int q_tile = blockIdx.x;
    const int bh_q = blockIdx.y;
    const int b = bh_q / H_q;
    const int hq = bh_q % H_q;
    const int hkv = hq / (H_q / H_kv);
    const int bh_kv = b * H_kv + hkv;

    const int qi = q_tile * FA_BR + threadIdx.x;
    const size_t head_stride = static_cast<size_t>(T) * Dh;
    const float* Qb = Q + bh_q * head_stride;
    const float* Kb = K + bh_kv * head_stride;
    const float* Vb = V + bh_kv * head_stride;
    float* Ob = O + bh_q * head_stride;
    float* Lb = L + static_cast<size_t>(bh_q) * T;

    extern __shared__ float smem[];
    float* K_s = smem;                     // [FA_BC * Dh]
    float* V_s = smem + FA_BC * Dh;        // [FA_BC * Dh]

    float q_reg[128];
    float o_reg[128];
    const bool valid_q = (qi < T);
    if (valid_q) {
        const float* q_ptr = Qb + static_cast<size_t>(qi) * Dh;
        for (int d = 0; d < Dh; ++d) {
            q_reg[d] = q_ptr[d];
            o_reg[d] = 0.0f;
        }
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    const int max_kv = min(T, (q_tile + 1) * FA_BR);
    for (int j0 = 0; j0 < max_kv; j0 += FA_BC) {
        const int bc = min(FA_BC, max_kv - j0);
        // Cooperatively load K_tile and V_tile into shared SRAM
        for (int idx = threadIdx.x; idx < bc * Dh; idx += blockDim.x) {
            int c = idx / Dh;
            int d = idx % Dh;
            K_s[idx] = Kb[static_cast<size_t>(j0 + c) * Dh + d];
            V_s[idx] = Vb[static_cast<size_t>(j0 + c) * Dh + d];
        }
        __syncthreads();

        if (valid_q) {
            float s_tile[FA_BC];
            float m_tile = -INFINITY;
            for (int c = 0; c < bc; ++c) {
                int kj = j0 + c;
                if (kj > qi) {
                    s_tile[c] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    const float* k_row = K_s + c * Dh;
                    for (int d = 0; d < Dh; ++d) dot = fmaf(q_reg[d], k_row[d], dot);
                    float sv = dot * scale;
                    s_tile[c] = sv;
                    if (sv > m_tile) m_tile = sv;
                }
            }

            if (m_tile > -INFINITY) {
                float m_new = fmaxf(m_i, m_tile);
                float alpha = (m_i == -INFINITY) ? 0.0f : expf(m_i - m_new);
                if (alpha != 1.0f) {
                    for (int d = 0; d < Dh; ++d) o_reg[d] *= alpha;
                    l_i *= alpha;
                }
                float l_tile = 0.0f;
                for (int c = 0; c < bc; ++c) {
                    if (s_tile[c] == -INFINITY) continue;
                    float p = expf(s_tile[c] - m_new);
                    l_tile += p;
                    const float* v_row = V_s + c * Dh;
                    for (int d = 0; d < Dh; ++d) o_reg[d] = fmaf(p, v_row[d], o_reg[d]);
                }
                l_i += l_tile;
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (valid_q) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        float* o_ptr = Ob + static_cast<size_t>(qi) * Dh;
        for (int d = 0; d < Dh; ++d) o_ptr[d] = o_reg[d] * inv_l;
        Lb[qi] = m_i + logf(l_i);
    }
}

__global__ void flash_attn_gqa_bwd_kernel(const float* __restrict__ Q,
                                          const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          const float* __restrict__ O,
                                          const float* __restrict__ L,
                                          const float* __restrict__ dO,
                                          float* __restrict__ dQ,
                                          float* __restrict__ dK,
                                          float* __restrict__ dV,
                                          int B, int H_q, int H_kv,
                                          int T, int Dh, float scale) {
    // One thread per KV position inside a BC=32 tile; loops over causal Q tiles
    const int kv_tile = blockIdx.x;
    const int bh_q = blockIdx.y;
    const int b = bh_q / H_q;
    const int hq = bh_q % H_q;
    const int hkv = hq / (H_q / H_kv);
    const int bh_kv = b * H_kv + hkv;

    const int kj = kv_tile * FA_BC + threadIdx.x;
    const size_t head_stride = static_cast<size_t>(T) * Dh;
    const float* Qb  = Q  + bh_q  * head_stride;
    const float* Kb  = K  + bh_kv * head_stride;
    const float* Vb  = V  + bh_kv * head_stride;
    const float* Ob  = O  + bh_q  * head_stride;
    const float* Lb  = L  + static_cast<size_t>(bh_q) * T;
    const float* dOb = dO + bh_q  * head_stride;
    float* dQb = dQ + bh_q  * head_stride;
    float* dKb = dK + bh_kv * head_stride;
    float* dVb = dV + bh_kv * head_stride;

    float k_reg[128], v_reg[128], dk_reg[128], dv_reg[128];
    const bool valid_k = (kj < T);
    if (valid_k) {
        const float* k_ptr = Kb + static_cast<size_t>(kj) * Dh;
        const float* v_ptr = Vb + static_cast<size_t>(kj) * Dh;
        for (int d = 0; d < Dh; ++d) {
            k_reg[d] = k_ptr[d];
            v_reg[d] = v_ptr[d];
            dk_reg[d] = 0.0f;
            dv_reg[d] = 0.0f;
        }
    }

    const int i_start = kv_tile * FA_BC;
    for (int qi = i_start; qi < T; ++qi) {
        const float* q_row = Qb + static_cast<size_t>(qi) * Dh;
        const float* o_row = Ob + static_cast<size_t>(qi) * Dh;
        const float* do_row = dOb + static_cast<size_t>(qi) * Dh;

        // Compute D_i = dot(dO_i, O_i) via thread 0 broadcast
        __shared__ float s_Di;
        if (threadIdx.x == 0) {
            float ddot = 0.0f;
            for (int d = 0; d < Dh; ++d) ddot = fmaf(do_row[d], o_row[d], ddot);
            s_Di = ddot;
        }
        __syncthreads();

        float ds_scaled = 0.0f;
        if (valid_k && kj <= qi) {
            float s_dot = 0.0f;
            float dp_dot = 0.0f;
            for (int d = 0; d < Dh; ++d) {
                s_dot  = fmaf(q_row[d], k_reg[d], s_dot);
                dp_dot = fmaf(do_row[d], v_reg[d], dp_dot);
            }
            float p = expf(s_dot * scale - Lb[qi]);
            float ds = p * (dp_dot - s_Di);
            ds_scaled = ds * scale;
            for (int d = 0; d < Dh; ++d) {
                dv_reg[d] = fmaf(p, do_row[d], dv_reg[d]);
                dk_reg[d] = fmaf(ds_scaled, q_row[d], dk_reg[d]);
            }
        }

        // Accumulate dQ[qi, d] += sum_{kj in tile} ds_scaled * k_reg[d] via warp reduction
        float* dq_row = dQb + static_cast<size_t>(qi) * Dh;
        for (int d = 0; d < Dh; ++d) {
            float contrib = (valid_k && kj <= qi) ? (ds_scaled * k_reg[d]) : 0.0f;
            contrib = warp_reduce_sum(contrib);
            if (threadIdx.x == 0 && contrib != 0.0f) {
                atomicAdd(&dq_row[d], contrib);
            }
        }
        __syncthreads();
    }

    // Commit GQA-shared dK and dV across query head group via atomicAdd
    if (valid_k) {
        float* dk_ptr = dKb + static_cast<size_t>(kj) * Dh;
        float* dv_ptr = dVb + static_cast<size_t>(kj) * Dh;
        for (int d = 0; d < Dh; ++d) {
            atomicAdd(&dk_ptr[d], dk_reg[d]);
            atomicAdd(&dv_ptr[d], dv_reg[d]);
        }
    }
}

void flash_attn_gqa_fwd(CudaContext& ctx,
                        const float* Q, const float* K, const float* V,
                        float* O, float* L,
                        int B, int H_q, int H_kv, int T, int Dh, float scale) {
    dim3 grid((T + FA_BR - 1) / FA_BR, B * H_q);
    size_t smem_bytes = 2ULL * FA_BC * Dh * sizeof(float);
    flash_attn_gqa_fwd_kernel<<<grid, FA_BR, smem_bytes, ctx.stream>>>(
        Q, K, V, O, L, B, H_q, H_kv, T, Dh, scale);
}

void flash_attn_gqa_bwd(CudaContext& ctx,
                        const float* Q, const float* K, const float* V,
                        const float* O, const float* L, const float* dO,
                        float* dQ, float* dK, float* dV,
                        int B, int H_q, int H_kv, int T, int Dh, float scale) {
    GBX_CUDA_CHECK(cudaMemsetAsync(dQ, 0, static_cast<size_t>(B) * H_q * T * Dh * sizeof(float), ctx.stream));
    GBX_CUDA_CHECK(cudaMemsetAsync(dK, 0, static_cast<size_t>(B) * H_kv * T * Dh * sizeof(float), ctx.stream));
    GBX_CUDA_CHECK(cudaMemsetAsync(dV, 0, static_cast<size_t>(B) * H_kv * T * Dh * sizeof(float), ctx.stream));
    dim3 grid((T + FA_BC - 1) / FA_BC, B * H_q);
    flash_attn_gqa_bwd_kernel<<<grid, FA_BC, 0, ctx.stream>>>(
        Q, K, V, O, L, dO, dQ, dK, dV, B, H_q, H_kv, T, Dh, scale);
}

// ---------------------------------------------------------------------------
// 6. Fused SwiGLU & Residual Ops
// ---------------------------------------------------------------------------

__global__ void swiglu_fwd_kernel(const float* __restrict__ gate,
                                  const float* __restrict__ up,
                                  float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float sig = 1.0f / (1.0f + expf(-g));
    out[i] = (g * sig) * up[i];
}

__global__ void swiglu_bwd_kernel(const float* __restrict__ gate,
                                  const float* __restrict__ up,
                                  const float* __restrict__ dout,
                                  float* __restrict__ dgate,
                                  float* __restrict__ dup, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    float dy = dout[i];
    float sig = 1.0f / (1.0f + expf(-g));
    float silu = g * sig;
    float dsilu = sig * (1.0f + g * (1.0f - sig));
    dup[i] = dy * silu;
    dgate[i] = dy * u * dsilu;
}

void swiglu_fwd(CudaContext& ctx, const float* gate, const float* up, float* out, int n) {
    swiglu_fwd_kernel<<<(n + 255) / 256, 256, 0, ctx.stream>>>(gate, up, out, n);
}

void swiglu_bwd(CudaContext& ctx, const float* gate, const float* up, const float* dout,
                float* dgate, float* dup, int n) {
    swiglu_bwd_kernel<<<(n + 255) / 256, 256, 0, ctx.stream>>>(gate, up, dout, dgate, dup, n);
}

__global__ void add_vec_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void add_inplace_kernel(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

void add_vec(CudaContext& ctx, const float* a, const float* b, float* out, int n) {
    add_vec_kernel<<<(n + 255) / 256, 256, 0, ctx.stream>>>(a, b, out, n);
}

void add_inplace(CudaContext& ctx, float* dst, const float* src, int n) {
    add_inplace_kernel<<<(n + 255) / 256, 256, 0, ctx.stream>>>(dst, src, n);
}

// ---------------------------------------------------------------------------
// 7. Fused Online LogSumExp Cross-Entropy Forward + Backward (Zero Softmax Buffer!)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < nwarps) ? shared[lane] : -INFINITY;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    __shared__ float total;
    if (threadIdx.x == 0) total = val;
    __syncthreads();
    return total;
}

__global__ void fused_ce_fwd_bwd_kernel(const float* __restrict__ logits,
                                        const int32_t* __restrict__ targets,
                                        float* __restrict__ dlogits,
                                        float* __restrict__ loss_sum,
                                        int N, int V, float grad_scale) {
    const int row = blockIdx.x;
    const int tgt = targets[row];
    const float* row_in = logits + static_cast<size_t>(row) * V;
    float* row_out = dlogits + static_cast<size_t>(row) * V;

    float mx = -INFINITY;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        mx = fmaxf(mx, row_in[v]);
    }
    mx = block_reduce_max(mx);

    float sum_e = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        sum_e += expf(row_in[v] - mx);
    }
    sum_e = block_reduce_sum(sum_e);

    const float log_z = mx + logf(sum_e);
    const float inv_sum = 1.0f / sum_e;
    if (threadIdx.x == 0 && tgt >= 0 && tgt < V) {
        float loss = log_z - row_in[tgt];
        atomicAdd(loss_sum, loss);
    }

    if (dlogits != nullptr) {
        const float scale = (tgt >= 0 && tgt < V) ? (grad_scale / static_cast<float>(N)) : 0.0f;
        for (int v = threadIdx.x; v < V; v += blockDim.x) {
            float prob = expf(row_in[v] - mx) * inv_sum;
            float indicator = (v == tgt) ? 1.0f : 0.0f;
            row_out[v] = (prob - indicator) * scale;
        }
    }
}

float fused_cross_entropy_fwd_bwd(CudaContext& ctx,
                                  const float* logits, const int32_t* targets,
                                  float* dlogits, float* d_loss_buf,
                                  int N, int V, float grad_scale) {
    GBX_CUDA_CHECK(cudaMemsetAsync(d_loss_buf, 0, sizeof(float), ctx.stream));
    fused_ce_fwd_bwd_kernel<<<N, 256, 0, ctx.stream>>>(
        logits, targets, dlogits, d_loss_buf, N, V, grad_scale);
    float h_loss = 0.0f;
    GBX_CUDA_CHECK(cudaMemcpyAsync(&h_loss, d_loss_buf, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream));
    GBX_CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    return h_loss / static_cast<float>(N);
}

// ---------------------------------------------------------------------------
// 8. Global Gradient Norm & Fused Clipped Decoupled AdamW
// ---------------------------------------------------------------------------

__global__ void l2_norm_sq_kernel(const float* __restrict__ g, float* __restrict__ out, size_t n) {
    float acc = 0.0f;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        float v = g[i];
        acc = fmaf(v, v, acc);
    }
    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0) atomicAdd(out, acc);
}

float global_grad_norm(CudaContext& ctx, const float* grad, float* d_norm_buf, size_t n) {
    GBX_CUDA_CHECK(cudaMemsetAsync(d_norm_buf, 0, sizeof(float), ctx.stream));
    int blocks = static_cast<int>(min<size_t>(1024, (n + 255) / 256));
    l2_norm_sq_kernel<<<blocks, 256, 0, ctx.stream>>>(grad, d_norm_buf, n);
    float h_sq = 0.0f;
    GBX_CUDA_CHECK(cudaMemcpyAsync(&h_sq, d_norm_buf, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream));
    GBX_CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    return std::sqrt(h_sq);
}

__global__ void adamw_step_kernel(float* __restrict__ p,
                                  const float* __restrict__ g,
                                  float* __restrict__ m,
                                  float* __restrict__ v,
                                  size_t n, float lr, float beta1, float beta2,
                                  float eps, float weight_decay,
                                  float grad_scale, float inv_bc1, float inv_bc2) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float gv = g[idx] * grad_scale;
    float mi = beta1 * m[idx] + (1.0f - beta1) * gv;
    float vi = beta2 * v[idx] + (1.0f - beta2) * gv * gv;
    m[idx] = mi;
    v[idx] = vi;
    float m_hat = mi * inv_bc1;
    float v_hat = vi * inv_bc2;
    float pv = p[idx];
    pv -= lr * weight_decay * pv;
    pv -= lr * (m_hat / (sqrtf(v_hat) + eps));
    p[idx] = pv;
}

void adamw_step(CudaContext& ctx,
                float* params, const float* grads, float* m, float* v,
                size_t n, float lr, float beta1, float beta2,
                float eps, float weight_decay, float grad_scale, int step) {
    float inv_bc1 = 1.0f / (1.0f - std::pow(beta1, static_cast<float>(step)));
    float inv_bc2 = 1.0f / (1.0f - std::pow(beta2, static_cast<float>(step)));
    int blocks = static_cast<int>((n + 255) / 256);
    adamw_step_kernel<<<blocks, 256, 0, ctx.stream>>>(
        params, grads, m, v, n, lr, beta1, beta2, eps, weight_decay, grad_scale, inv_bc1, inv_bc2);
}

}  // namespace glassbox_cuda
