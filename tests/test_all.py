"""Comprehensive verification suite for Glassbox Deep Learning Framework & Transformer."""

from __future__ import annotations

import math
import random
import time

from glassbox.backend import benchmark_matmul
from glassbox.bpe import BPETokenizer
from glassbox.datasets import get_dataset_text
from glassbox.gradcheck import (
    gradcheck,
    run_full_gradcheck_suite,
    verify_flash_attention_equivalence,
)
from glassbox.interp import (
    analyze_attention_and_ablation,
    analyze_logit_lens_and_dla,
    run_causal_activation_patching,
)
from glassbox.optim import AdamW
from glassbox.tensor import Tensor, no_grad, rope
from glassbox.trainer import LabSession
from glassbox.transformer import GPTConfig, GPTTransformer


def test_primitive_gradchecks():
    print("[1/7] Running float64 finite-difference gradchecks on all autograd ops...")
    results = run_full_gradcheck_suite()
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {status:4s} | {r['op']:32s} | max_abs={r['max_abs_error']:.2e} | cos={r['cosine_similarity']:.12f}"
        )
        assert r["passed"], f"Gradcheck failed for {r['op']}: {r}"


def test_full_transformer_gradcheck():
    print("[2/7] Running end-to-end float64 gradcheck on full GPT Transformer (RoPE+RMSNorm+FlashAttn+SwiGLU)...")
    cfg = GPTConfig(
        vocab_size=12,
        d_model=8,
        n_layers=2,
        n_heads=2,
        d_ff=12,
        context_len=8,
        tie_weights=True,
        dtype="float64",
    )
    model = GPTTransformer(cfg, seed=101)
    xs = [[1, 3, 5, 2, 7, 4]]
    ys = [[3, 5, 2, 7, 4, 9]]

    params = model.parameters()
    res = gradcheck(
        lambda *_: model.forward(xs, targets=ys)[1],
        params,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
        max_elements_per_tensor=24,
    )
    print(
        f"  PASS | Full 2-Layer GPT Transformer     | checked={res['checked_elements']} | "
        f"max_abs={res['max_abs_error']:.2e} | max_rel={res['max_rel_error']:.2e} | cos={res['cosine_similarity']:.12f}"
    )
    assert res["passed"], f"Full transformer gradcheck failed: {res}"


def test_flash_attention_equivalence():
    print("[3/7] Verifying FlashAttention-2 (multi-tile online softmax) vs Naive Attention...")
    for causal in (True, False):
        for T in (17, 45):
            r = verify_flash_attention_equivalence(B=2, H=2, T=T, D=16, causal=causal, dtype="float64")
            print(
                f"  PASS | causal={str(causal):5s} T={T:2d} | dOut={r['max_diff_output']:.2e} "
                f"dQ={r['max_diff_dQ']:.2e} dK={r['max_diff_dK']:.2e} dV={r['max_diff_dV']:.2e}"
            )
            assert r["passed"], f"FlashAttention mismatch: {r}"


def test_rope_invariance():
    print("[4/7] Verifying RoPE norm preservation & relative-position dot-product invariance...")
    rng = random.Random(55)
    q = Tensor.randn((1, 1, 1, 16), std=1.0, dtype="float64", rng=rng)
    k = Tensor.randn((1, 1, 1, 16), std=1.0, dtype="float64", rng=rng)

    with no_grad():
        q_rot0 = rope(q, pos_offset=0)
        assert abs(q.l2_norm() - q_rot0.l2_norm()) < 1e-12, "RoPE broke L2 norm preservation!"

        # <RoPE(q, m), RoPE(k, n)> must depend ONLY on (m - n)
        dots = []
        for shift in (0, 5, 19):
            qm = rope(q, pos_offset=shift + 7)
            kn = rope(k, pos_offset=shift + 2)
            dot = sum(float(qm._buf[i]) * float(kn._buf[i]) for i in range(16))
            dots.append(dot)
        max_delta = max(abs(d - dots[0]) for d in dots)
        print(f"  PASS | Norm delta < 1e-12 | Relative-offset dot-product delta = {max_delta:.2e}")
        assert max_delta < 1e-12


def test_bpe_tokenizer():
    print("[5/7] Testing Byte-Level BPE Tokenizer training, lossless UTF-8 roundtrip, and serialization...")
    corpus = get_dataset_text("induction")
    tok = BPETokenizer()
    t0 = time.perf_counter()
    stats = tok.train(corpus, target_vocab_size=384)
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"  Trained vocab={stats['vocab_size']} ({stats['num_merges']} merges) in {dt:.1f}ms, ratio={stats['compression_ratio']} B/tok")

    samples = [
        "The silver falcon belongs to Marcus.\nTable: alpha maps to crimson!",
        "def foo(x: int) -> str:\n    return f'hello {x}'  # tabs:\t\tend",
        "Lossless Unicode test: café, naïve, 日本語, Σ(x²), 🚀✨",
        "",
    ]
    for s in samples:
        ids = tok.encode(s)
        dec = tok.decode(ids)
        assert dec == s, f"BPE roundtrip mismatch: {dec!r} != {s!r}"

    # Verify serialization roundtrip
    tok2 = BPETokenizer.from_dict(tok.to_dict())
    for s in samples:
        assert tok2.encode(s) == tok.encode(s)
    print("  PASS | Lossless UTF-8 roundtrip + JSON serialization verified")


def test_matmul_benchmark():
    print("[6/7] Benchmarking Cache-Tiled SGEMM vs Naive SGEMM...")
    rows = benchmark_matmul([64, 128, 256])
    for r in rows:
        print(
            f"  PASS | N={r['size']:3d} | Naive={r['naive_gflops']:5.2f} GF/s | "
            f"Tiled={r['tiled_gflops']:5.2f} GF/s | Speedup={r['speedup']:5.2f}x | max_err={r['max_abs_error']:.2e}"
        )
        assert r["max_abs_error"] < 1e-4
        assert r["speedup"] > 2.0


def test_training_and_interpretability():
    print("[7/7] Testing end-to-end training convergence + Induction Phase Transition + Interpretability...")
    session = LabSession(
        dataset_id="induction", preset_id="micro", lr=5e-3, batch_size=16, weight_decay=0.01, seed=7
    )
    t0 = time.perf_counter()
    session.warm_start(steps=250)
    dt = time.perf_counter() - t0

    first_loss = session.history[0]["loss"]
    last_loss = session.history[-1]["loss"]
    val_loss = session.history[-1]["val_loss"]
    print(
        f"  Trained 250 steps in {dt:.2f}s ({250/dt:.1f} steps/s) | "
        f"loss: {first_loss:.3f} -> {last_loss:.3f} | val_loss: {val_loss:.3f}"
    )
    assert last_loss < first_loss * 0.5, f"Model did not converge strongly: {first_loss} -> {last_loss}"

    p_clean = "Echo alpha crimson beta cobalt gamma amber delta emerald alpha crimson beta"
    p_corr = "Echo alpha crimson beta violet gamma amber delta emerald alpha crimson beta"
    attn = analyze_attention_and_ablation(
        session.model, session.tokenizer, p_clean, ablate_heads=[(0, 0), (1, 0)], ablation_mode="zero"
    )
    assert len(attn["layers"]) == session.cfg.n_layers
    assert attn["ablation"] is not None
    assert attn["ablation"]["kl_divergence"] >= 0.0

    lens = analyze_logit_lens_and_dla(session.model, session.tokenizer, p_clean)
    actual_logit = lens["dla"]["actual_logit"]
    recon_logit = lens["dla"]["sum_of_contribs"]
    dla_err = abs(actual_logit - recon_logit)
    print(
        f"  DLA exactness check: actual={actual_logit:.5f}, sum_of_contribs={recon_logit:.5f}, err={dla_err:.2e}"
    )
    assert dla_err < 1e-3, f"DLA additive decomposition mismatch: {dla_err}"

    patch = run_causal_activation_patching(
        session.model,
        session.tokenizer,
        clean_prompt=p_clean,
        corrupted_prompt=p_corr,
        clean_target=" cobalt",
        corrupted_target=" violet",
    )
    print(
        f"  Causal Patching: clean_p={patch['clean_target']['clean_prob']:.3f} "
        f"LD_clean={patch['logit_diff_clean']:.3f}, LD_corrupt={patch['logit_diff_corrupt']:.3f}, "
        f"top_hotspot={patch['top_hotspots'][0]}"
    )

    dag = session.capture_autograd_dag(p_clean)
    print(f"  Autograd DAG captured: {dag['num_nodes']} nodes, {dag['num_edges']} edges, loss={dag['loss']:.4f}")
    assert dag["num_nodes"] > 20
    print("  PASS | All end-to-end training & mechanistic interpretability checks passed!")


if __name__ == "__main__":
    test_primitive_gradchecks()
    test_full_transformer_gradcheck()
    test_flash_attention_equivalence()
    test_rope_invariance()
    test_bpe_tokenizer()
    test_matmul_benchmark()
    test_training_and_interpretability()
    print("\nALL 7 VERIFICATION SUITES PASSED!")
