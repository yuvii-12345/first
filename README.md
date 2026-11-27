# Glassbox — From-Scratch Deep Learning Framework, GPT Transformer & Mechanistic Interpretability Lab

Zero PyTorch. Zero TensorFlow. Zero JAX. Zero NumPy. Zero external BLAS or web frameworks.

Glassbox is a complete deep learning framework, modern GPT-style Transformer, byte-level BPE tokenizer, and interactive **Mechanistic Interpretability Observatory** built entirely from scratch using **C++17 compute kernels** (`libglassbox.so`), **Python 3.11 standard library** autograd/HTTP+SSE backend, and a **React + TypeScript + Tailwind** live scientific frontend.

## Architecture

- **Custom C++17 Compute Kernels (`glassbox/csrc/kernels.cpp`)** — Templated `float32` & `float64` kernels compiled on-the-fly with `-O3 -march=native`:
  - Cache-blocked (`64×64`) & AVX/FMA register-tiled (`4×8`) batched GEMM (`~14–22×` faster than naive triple-loop GEMM, up to `~40 GFLOP/s` on 2 vCPUs)
  - **FlashAttention-2** tiled online-softmax causal & bidirectional forward + log-sum-exp recomputing backward (never materializes `T×T` during training; matches naive exact attention to `~1e-16` in `float64`)
  - Fused **RMSNorm**, **Rotary Position Embeddings (RoPE)**, **SwiGLU** (`SiLU` + gating), **Cross-Entropy Loss**, **AdamW**, and broadcast/reduction/permute/embedding kernels
- **Pure-Python Reverse-Mode Autograd (`glassbox/tensor.py`)** — Dynamic computation DAG, iterative topological-sort `backward()`, `no_grad()`, and full DAG serialization (`export_graph()`).
- **Numerical Verification (`glassbox/gradcheck.py`)** — Central finite-difference `float64` gradient verification across all differentiable primitives AND an end-to-end 2-layer GPT Transformer (`max_abs_error ~ 1e-10`, `cosine_similarity = 1.000000000000`).
- **Byte-Level BPE Tokenizer (`glassbox/bpe.py`)** — Lossless UTF-8 byte vocabulary (`0..255`), special tokens (`<BOS>`, `<EOS>`, `<PAD>`), GPT-2 pre-tokenization, step-by-step merge replay, and JSON serialization.
- **Modern GPT Transformer (`glassbox/transformer.py`, `glassbox/nn.py`, `glassbox/optim.py`)** — Pre-Norm RMSNorm, RoPE, FlashAttention-2 with live per-head zero/mean ablation and exact `W_o`-slice Direct Logit Attribution capture, SwiGLU FFN, weight-tied LM head, causal residual patching hooks, cosine warmup scheduler, global gradient clipping, and per-layer Karpathy update-to-weight ratio telemetry.
- **Interactive Mechanistic Interpretability Lab (`server/app.py`, `web/`)**:
  1. **Training & Gradient Flow** — Live SSE-streamed Loss/Val-Loss curves, tokens/sec, custom C++ kernel GFLOP/s, per-layer gradient norms & `log10(‖ΔW‖/‖W‖)` health indicators, and dataset/model configurator.
  2. **Attention & Ablation Explorer** — All-head miniature grid with automatic mechanistic role badges (`Induction`, `Prev-Token`, `BOS Sink`, `Self-Attend`), zoomed causal heatmap, BertViz-style token connection arcs, and live interactive head zero/mean ablation with KL divergence.
  3. **Logit Lens & Direct Logit Attribution** — Layer × Token intermediate residual decoding matrix + exact algebraic additive logit decomposition (`Actual Logit == Σ Components`).
  4. **Causal Activation Patching** — Clean vs Corrupted denoising residual stream sweep localizing information flow across layers and positions.
  5. **Generation Studio & BPE Lab** — Temperature / Top-K / Top-P autoregressive sampler with per-token probability inspection + live BPE tokenization & merge-step replay.
  6. **Engine Proof & Autograd DAG** — 1-click float64 gradcheck, FlashAttention-2 equivalence proof, live tiled-vs-naive SGEMM benchmark, and interactive reverse-mode computation DAG explorer.

## Quickstart

```bash
# 1. Run the 7-part verification suite (float64 gradcheck, FlashAttention equivalence, BPE, SGEMM, induction circuit)
python3 -m tests.test_all

# 2. Build the web UI (if not already built)
cd web && npm install && npm run build && cd ..

# 3. Start the unified HTTP + SSE Mechanistic Interpretability Lab on 0.0.0.0:8000
python3 server/app.py
```
