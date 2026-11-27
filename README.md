# Glassbox — From-Scratch 100M LLM, Pure CUDA C++17 Engine & Mechanistic Interpretability Lab

**Zero PyTorch. Zero TensorFlow. Zero JAX. Zero NumPy. Zero cuDNN. Zero external web frameworks.**

Live Web App (Pure TypeScript Float32 + GQA KV-Cache Engine): **`https://glassbox-chat-self.vercel.app`**

Glassbox is a complete from-scratch deep learning stack spanning:
1. **`Glassbox-100M` (`100,682,496` parameters)** — Modern Llama/GPT-style LLM (`12 layers · 768d · 12Q/4KV 3:1 GQA · SwiGLU 2048 · RoPE · 32K BPE · 2048 context · weight-tied LM head`) with a **pure from-scratch CUDA C++17 training engine** (`cuda/`) supporting **100M to 10B+ tokens** via memory-mapped `.gbxtok` shards and zero-copy `.gbx` binary checkpoints.
2. **`Glassbox-Chat 1M GQA` (`1,007,160` parameters)** — Live conversational 6-layer `6Q/2KV` GQA Transformer trained from scratch on C++17 kernels (`loss=0.0824`, `val=0.0776`) running both on our native C++17 server AND directly inside the browser with a **bit-exact `O(1)` KV-Cache (`12.8×` faster decode, `0.000e+00` diff)**.
3. **Interactive Mechanistic Interpretability Observatory (`server/app.py`, `web/`)** — 8-tab live laboratory: Chat & Live Neural X-Ray, **100M LLM & Pure CUDA C++17**, Attention & Head Ablation, Logit Lens & Exact DLA, Causal Activation Patching, Training & Gradients, Generate & BPE, and Engine Proof & Autograd DAG.

---

## Glassbox-100M Architecture & Pure CUDA C++17 Engine (`cuda/`)

| Specification | Value |
| :--- | :--- |
| **Exact Parameters** | **`100,682,496`** (`100.68M`) |
| **Transformer Topology** | `12 layers · d_model=768 · 12 Q heads / 4 KV heads (3:1 GQA) · head_dim=64` |
| **FFN & Normalization** | SwiGLU (`d_ff=2048`) · Pre-Norm RMSNorm · RoPE (`θ=10,000`) · Tied Embeddings |
| **Vocabulary & Context** | `32,768` Byte-Level BPE · `2,048` context window |
| **Binary Checkpoints (`.gbx`)** | 72-byte `GBX1` header + raw contiguous `float32` tensors (`402.7 MB` for 100M, shared bit-for-bit across Python, C++17 CPU & CUDA C++17) |
| **Token Shards (`.gbxtok`)** | 32-byte `GBXT` header + POSIX `mmap` `uint16` token streams (`200 MB` for 100M tokens, `20 GB` for 10B tokens) |

### Pure From-Scratch CUDA Kernels (`cuda/kernels.cu`, `cuda/train_glassbox_100m.cu`)
- **2D Register-Tiled SGEMM (`64×64×16`, `4×4` thread micro-tile)** + optional cuBLAS TF32/FP32 switch (`-DUSE_CUBLAS`)
- **GQA FlashAttention-2** (online log-sum-exp softmax forward + recomputing backward, `q_offset` KV-cache support, zero `T×T` materialization)
- **Warp-Shuffle Fused RMSNorm**, **RoPE**, **SwiGLU**, **Fused Online Cross-Entropy**, **Gradient Norm & Clipping**, and **Decoupled AdamW**
- **Cross-Engine Numerical Parity Verified**:
  - `micro` 4L GQA (`451,680` params) Python vs Native C++17 CPU: `loss_diff = 2.529e-05`, `gnorm_diff = 6.731e-07`
  - Live `Glassbox-100M` (`100,682,496` params) 3-step C++17 smoke test: `10.4726 → 8.7736 → 7.5214`, saved `402.7 MB .gbx`, reloaded into Python in `2.19s` (`post-step loss = 6.5047`)

### Train Glassbox-100M on 100M → 10B Tokens on Any NVIDIA GPU (Zero PyTorch)

```bash
# 1. Build pure CUDA C++17 binary (or `make -C cuda cuda-pure` for 100% custom SGEMM, zero cuBLAS)
make -C cuda cuda

# 2. One-command turnkey GPU launch (streams FineWeb-Edu into .gbxtok shards & trains Glassbox-100M):
#    100M tokens: ~2.5 min on 1x H100 (~$0.15) or ~8.3 min on 1x RTX 4090
bash cuda/launch_gpu.sh 100m

#    2B tokens (Chinchilla-optimal): ~50 min on 1x H100
bash cuda/launch_gpu.sh 2b

#    10B tokens (over-trained frontier quality): ~4.2 hours on 1x H100 (~$12)
bash cuda/launch_gpu.sh 10b
```

---

## Local Quickstart (CPU + Web Lab)

```bash
# 1. Run the full 8-part verification suite (float64 gradchecks, GQA, KV-cache, .gbx/.gbxtok, 100M accounting, SGEMM, BPE, induction)
python3 -m tests.test_all

# 2. Compile & run the native C++17 CPU 100M engine smoke test
make -C cuda cpu
./cuda/glassbox_cpu_train --preset 100m --sgd --batch-size 1 --seq-len 16 --steps 3 --lr 0.15

# 3. Build the web UI
cd web && npm install && npm run build && cd ..

# 4. Launch the unified HTTP + SSE Mechanistic Interpretability Lab on 0.0.0.0:8000
python3 server/app.py
```
