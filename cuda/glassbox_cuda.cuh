#pragma once
// ============================================================================
// Glassbox-100M Pure From-Scratch CUDA C++17 Compute Kernels Header
// Zero PyTorch. Zero cuDNN. Optional -DGBX_USE_CUBLAS for Tensor Core TF32 boost.
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifdef GBX_USE_CUBLAS
#include <cublas_v2.h>
#endif

#define GBX_CUDA_CHECK(call)                                                       \
    do {                                                                           \
        cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                \
            std::fprintf(stderr, "[Glassbox CUDA Error] %s:%d: %s\n",              \
                         __FILE__, __LINE__, cudaGetErrorString(err__));           \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

namespace glassbox_cuda {

struct CudaContext {
    cudaStream_t stream = nullptr;
#ifdef GBX_USE_CUBLAS
    cublasHandle_t cublas = nullptr;
#endif
    void init();
    void destroy();
};

// C = (accumulate ? C : 0) + op(A) @ op(B)
// op(A) is [M, K], op(B) is [K, N], C is [M, N], all row-major float32
void gemm(CudaContext& ctx,
          const float* A, const float* B, float* C,
          int M, int K, int N,
          bool trans_a, bool trans_b, bool accumulate);

void embedding_fwd(CudaContext& ctx, const float* table, const int32_t* ids,
                   float* out, int N, int D);
void embedding_bwd(CudaContext& ctx, const float* dout, const int32_t* ids,
                   float* dtable, int N, int D);

void rmsnorm_fwd(CudaContext& ctx, const float* x, const float* w,
                 float* y, float* inv_rms, int N, int D, float eps);
void rmsnorm_bwd(CudaContext& ctx, const float* x, const float* w,
                 const float* inv_rms, const float* dy,
                 float* dx, float* dw, int N, int D);

void permute_bthd_to_bhtd(CudaContext& ctx, const float* src, float* dst,
                          int B, int T, int H, int Dh);
void permute_bhtd_to_bthd(CudaContext& ctx, const float* src, float* dst,
                          int B, int H, int T, int Dh);

void rope_apply(CudaContext& ctx, const float* x, float* y,
                int BH, int T, int Dh, int pos_offset, float base, bool inverse);

// Grouped-Query FlashAttention-2 (Q,O: [B,H_q,T,Dh]; K,V: [B,H_kv,T,Dh]; L: [B,H_q,T])
void flash_attn_gqa_fwd(CudaContext& ctx,
                        const float* Q, const float* K, const float* V,
                        float* O, float* L,
                        int B, int H_q, int H_kv, int T, int Dh, float scale);

void flash_attn_gqa_bwd(CudaContext& ctx,
                        const float* Q, const float* K, const float* V,
                        const float* O, const float* L, const float* dO,
                        float* dQ, float* dK, float* dV,
                        int B, int H_q, int H_kv, int T, int Dh, float scale);

// Fused SwiGLU: out = SiLU(gate) * up
void swiglu_fwd(CudaContext& ctx, const float* gate, const float* up, float* out, int n);
void swiglu_bwd(CudaContext& ctx, const float* gate, const float* up, const float* dout,
                float* dgate, float* dup, int n);

void add_vec(CudaContext& ctx, const float* a, const float* b, float* out, int n);
void add_inplace(CudaContext& ctx, float* dst, const float* src, int n);

// Fused row-wise online logsumexp Cross-Entropy forward + backward directly into dlogits
// Returns scalar average loss on host (via pinned/temp device scalar)
float fused_cross_entropy_fwd_bwd(CudaContext& ctx,
                                  const float* logits, const int32_t* targets,
                                  float* dlogits, float* d_loss_buf,
                                  int N, int V, float grad_scale);

// Global L2 norm squared of gradient vector + fused clipped AdamW parameter step
float global_grad_norm(CudaContext& ctx, const float* grad, float* d_norm_buf, size_t n);

void adamw_step(CudaContext& ctx,
                float* params, const float* grads, float* m, float* v,
                size_t n, float lr, float beta1, float beta2,
                float eps, float weight_decay, float grad_scale, int step);

}  // namespace glassbox_cuda
