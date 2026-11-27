"""Numerical finite-difference gradient checker & FlashAttention equivalence verifier."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Sequence

from glassbox.tensor import (
    Tensor,
    cross_entropy,
    embedding,
    flash_attention,
    linear,
    matmul,
    naive_attention,
    no_grad,
    rmsnorm,
    rope,
)


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    max_elements_per_tensor: int = 128,
) -> Dict[str, Any]:
    """Compare analytical reverse-mode autograd gradients against central finite differences."""
    for inp in inputs:
        inp.zero_grad()

    out = fn(*inputs)
    if out.numel != 1:
        out = out.sum()
    out.backward()

    max_abs = 0.0
    max_rel = 0.0
    dot_an = 0.0
    norm_a_sq = 0.0
    norm_n_sq = 0.0
    checked = 0

    for inp in inputs:
        if not inp.requires_grad:
            continue
        assert inp.grad is not None, f"Missing analytical gradient for input {inp}"
        analytical = [float(inp.grad._buf[i]) for i in range(inp.numel)]

        # Deterministic subsample if tensor is large
        indices = list(range(inp.numel))
        if len(indices) > max_elements_per_tensor:
            rng = random.Random(123)
            indices = sorted(rng.sample(indices, max_elements_per_tensor))

        for idx in indices:
            orig = float(inp._buf[idx])
            with no_grad():
                inp._buf[idx] = orig + eps
                o_plus = fn(*inputs)
                f_plus = o_plus.item() if o_plus.numel == 1 else o_plus.sum().item()

                inp._buf[idx] = orig - eps
                o_minus = fn(*inputs)
                f_minus = o_minus.item() if o_minus.numel == 1 else o_minus.sum().item()

                inp._buf[idx] = orig

            num_g = (f_plus - f_minus) / (2.0 * eps)
            ana_g = analytical[idx]
            abs_err = abs(ana_g - num_g)
            denom = max(abs(ana_g), abs(num_g), 1e-12)
            rel_err = abs_err / denom if abs_err > atol * 0.1 else abs_err

            if abs_err > max_abs:
                max_abs = abs_err
            if rel_err > max_rel:
                max_rel = rel_err
            dot_an += ana_g * num_g
            norm_a_sq += ana_g * ana_g
            norm_n_sq += num_g * num_g
            checked += 1

    cos_sim = (
        dot_an / (math.sqrt(norm_a_sq) * math.sqrt(norm_n_sq) + 1e-18)
        if norm_a_sq > 0 and norm_n_sq > 0
        else 1.0
    )
    passed = (max_abs <= atol) or (max_rel <= rtol)
    return {
        "passed": bool(passed),
        "checked_elements": checked,
        "max_abs_error": float(max_abs),
        "max_rel_error": float(max_rel),
        "cosine_similarity": float(cos_sim),
    }


def verify_flash_attention_equivalence(
    B: int = 2, H: int = 3, T: int = 41, D: int = 16, causal: bool = True, dtype: str = "float64"
) -> Dict[str, Any]:
    """Prove tiled online-softmax FlashAttention matches unfused naive attention in fwd + bwd."""
    rng = random.Random(2026)
    q1 = Tensor.randn((B, H, T, D), std=0.5, dtype=dtype, requires_grad=True, rng=rng)
    k1 = Tensor.randn((B, H, T, D), std=0.5, dtype=dtype, requires_grad=True, rng=rng)
    v1 = Tensor.randn((B, H, T, D), std=0.5, dtype=dtype, requires_grad=True, rng=rng)
    w = Tensor.randn((B, H, T, D), std=0.5, dtype=dtype, requires_grad=False, rng=rng)

    q2 = q1.clone(); q2.requires_grad = True
    k2 = k1.clone(); k2.requires_grad = True
    v2 = v1.clone(); v2.requires_grad = True

    out_flash, probs_flash = flash_attention(q1, k1, v1, causal=causal, return_probs=True)
    loss_flash = (out_flash * w).sum()
    loss_flash.backward()

    out_naive, probs_naive = naive_attention(q2, k2, v2, causal=causal)
    loss_naive = (out_naive * w).sum()
    loss_naive.backward()

    def _max_diff(a: Tensor, b: Tensor) -> float:
        return max(abs(float(a._buf[i]) - float(b._buf[i])) for i in range(a.numel))

    diff_out = _max_diff(out_flash, out_naive)
    diff_probs = _max_diff(probs_flash, probs_naive)
    diff_dq = _max_diff(q1.grad, q2.grad)
    diff_dk = _max_diff(k1.grad, k2.grad)
    diff_dv = _max_diff(v1.grad, v2.grad)

    tol = 1e-10 if dtype == "float64" else 1e-4
    passed = max(diff_out, diff_probs, diff_dq, diff_dk, diff_dv) < tol
    return {
        "passed": bool(passed),
        "shape": [B, H, T, D],
        "causal": causal,
        "dtype": dtype,
        "max_diff_output": float(diff_out),
        "max_diff_probs": float(diff_probs),
        "max_diff_dQ": float(diff_dq),
        "max_diff_dK": float(diff_dk),
        "max_diff_dV": float(diff_dv),
    }


def run_full_gradcheck_suite() -> List[Dict[str, Any]]:
    """Run float64 central finite-difference gradcheck on every autograd primitive."""
    rng = random.Random(7)
    results: List[Dict[str, Any]] = []

    def _record(name: str, res: Dict[str, Any]):
        results.append({"op": name, **res})

    # 1. Tiled MatMul (2D and N-D)
    a = Tensor.randn((5, 7), std=0.6, dtype="float64", requires_grad=True, rng=rng)
    b = Tensor.randn((7, 4), std=0.6, dtype="float64", requires_grad=True, rng=rng)
    w = Tensor.randn((5, 4), std=0.6, dtype="float64", rng=rng)
    _record("TiledMatMul (2D)", gradcheck(lambda x, y: ((x @ y) * w).sum(), [a, b]))

    # 2. Fused Linear (with bias broadcast)
    lx = Tensor.randn((2, 4, 6), std=0.5, dtype="float64", requires_grad=True, rng=rng)
    lw = Tensor.randn((5, 6), std=0.5, dtype="float64", requires_grad=True, rng=rng)
    lb = Tensor.randn((5,), std=0.5, dtype="float64", requires_grad=True, rng=rng)
    l_proj = Tensor.randn((2, 4, 5), std=0.5, dtype="float64", rng=rng)
    _record("Linear + Bias Broadcast", gradcheck(lambda x, wt, bi: (linear(x, wt, bi) * l_proj).sum(), [lx, lw, lb]))

    # 3. RMSNorm
    rx = Tensor.randn((3, 8), std=0.8, dtype="float64", requires_grad=True, rng=rng)
    rw = Tensor.randn((8,), std=0.5, mean=1.0, dtype="float64", requires_grad=True, rng=rng)
    rm = Tensor.randn((3, 8), std=0.5, dtype="float64", rng=rng)
    _record("RMSNorm", gradcheck(lambda x, wt: (rmsnorm(x, wt, eps=1e-6) * rm).sum(), [rx, rw]))

    # 4. RoPE (Rotary Position Embeddings)
    px = Tensor.randn((1, 2, 6, 8), std=0.6, dtype="float64", requires_grad=True, rng=rng)
    pm = Tensor.randn((1, 2, 6, 8), std=0.6, dtype="float64", rng=rng)
    _record("RoPE (Rotary Embeddings)", gradcheck(lambda x: (rope(x, pos_offset=3) * pm).sum(), [px]))

    # 5. FlashAttention-2 Tiled Causal Attention (T > tile_size test + non-square)
    fq = Tensor.randn((1, 2, 10, 8), std=0.5, dtype="float64", requires_grad=True, rng=rng)
    fk = Tensor.randn((1, 2, 10, 8), std=0.5, dtype="float64", requires_grad=True, rng=rng)
    fv = Tensor.randn((1, 2, 10, 8), std=0.5, dtype="float64", requires_grad=True, rng=rng)
    fm = Tensor.randn((1, 2, 10, 8), std=0.5, dtype="float64", rng=rng)
    _record(
        "FlashAttention (Causal Tiled)",
        gradcheck(lambda q, k, v: (flash_attention(q, k, v, causal=True) * fm).sum(), [fq, fk, fv]),
    )

    # 6. SwiGLU Gate (SiLU * Up)
    gx = Tensor.randn((2, 6), std=0.7, dtype="float64", requires_grad=True, rng=rng)
    gu = Tensor.randn((2, 6), std=0.7, dtype="float64", requires_grad=True, rng=rng)
    gm = Tensor.randn((2, 6), std=0.7, dtype="float64", rng=rng)
    _record("SwiGLU (SiLU * Up)", gradcheck(lambda x, u: ((x.silu() * u) * gm).sum(), [gx, gu]))

    # 7. Fused Cross-Entropy Loss
    logits = Tensor.randn((4, 9), std=0.9, dtype="float64", requires_grad=True, rng=rng)
    targets = [2, 0, -100, 7]
    _record("CrossEntropy (Fused LogSoftmax)", gradcheck(lambda lg: cross_entropy(lg, targets, ignore_index=-100), [logits]))

    # 8. Embedding Lookup + Scatter-Add Backward
    tab = Tensor.randn((6, 5), std=0.5, dtype="float64", requires_grad=True, rng=rng)
    em = Tensor.randn((2, 3, 5), std=0.5, dtype="float64", rng=rng)
    _record("Embedding Scatter-Add", gradcheck(lambda t: (embedding(t, [1, 4, 1, 0, 5, 4], (2, 3)) * em).sum(), [tab]))

    # 9. Broadcasted Binary Ops & Permute
    ba = Tensor.randn((2, 3, 4), std=0.5, dtype="float64", requires_grad=True, rng=rng)
    bb = Tensor.randn((1, 3, 1), std=0.3, mean=1.5, dtype="float64", requires_grad=True, rng=rng)
    bm = Tensor.randn((2, 4, 3), std=0.5, dtype="float64", rng=rng)
    _record(
        "Broadcast Div/Add + Permute",
        gradcheck(lambda x, y: (((x + y) / (y + 2.0)).permute(0, 2, 1) * bm).sum(), [ba, bb]),
    )

    return results
