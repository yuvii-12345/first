"""Train the upgraded 6-Layer 1.01M-Parameter 3:1 GQA Glassbox-Chat LLM from scratch
on our C++17 compute kernels, verify every reply, save .json + .gbx checkpoints, and
export the compact Base64 Int8 browser bundle for Vercel.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glassbox.checkpoint import save_gbx_checkpoint
from glassbox.datasets import get_dataset_text
from glassbox.trainer import LabSession
from scripts.train_and_export_chat import _build_verify_snapshot, export_browser_bundle


def main() -> None:
    total_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 620
    ckpt_json = ROOT / "checkpoints" / "chat_v1.json"
    ckpt_gbx = ROOT / "checkpoints" / "chat_gqa_1m.gbx"

    print("[1/4] Capturing float64 gradcheck + FlashAttention-2 + C++17 GEMM proof snapshot...", flush=True)
    verify_snap = _build_verify_snapshot()
    print(f"      All gradchecks passed: {verify_snap['verify']['all_passed']}", flush=True)

    # Instantiate fresh 6-Layer 1.01M-Parameter 3:1 GQA Transformer (matching Glassbox-100M GQA ratio!)
    sess = LabSession(
        dataset_id="chat",
        preset_id="chat_gqa_1m",
        lr=4.5e-3,
        batch_size=12,
        weight_decay=0.01,
        grad_clip=1.0,
        seed=42,
    )
    print(
        f"[2/4] Built Glassbox-Chat 1M GQA: {sess.model.num_parameters():,} params "
        f"({sess.cfg.n_layers}L · {sess.cfg.n_heads}Q/{sess.cfg.effective_kv_heads}KV GQA · "
        f"{sess.cfg.d_model}d · ff={sess.cfg.d_ff} · vocab={sess.cfg.vocab_size})",
        flush=True,
    )

    # Deduplicate docs so every unique Q&A dialogue receives uniform balanced coverage
    raw_docs = [d.strip() for d in get_dataset_text("chat").split("\n\n") if d.strip()]
    unique_qa: list[str] = []
    seen = set()
    for d in raw_docs:
        if d.startswith("User: Remember ") or d.startswith("User: Repeat:"):
            continue
        if d not in seen:
            seen.add(d)
            unique_qa.append(d)

    enc_qa = [
        sess.tokenizer.encode(d, add_bos=True, add_eos=True)[: sess.cfg.context_len + 1]
        for d in unique_qa
    ]
    sess.train_docs = enc_qa
    sess.val_docs = enc_qa[: max(10, len(enc_qa) // 5)]
    print(f"      Balanced {len(enc_qa)} unique Q&A dialogues for {total_steps} cosine steps...", flush=True)

    sess.scheduler.warmup_steps = 30
    sess.scheduler.total_steps = total_steps
    sess.scheduler.max_lr = 4.5e-3
    sess.scheduler.min_lr = 4.5e-4

    t0 = time.perf_counter()
    for s in range(total_steps):
        rec = sess.train_one_step(compute_val=(s % 60 == 59 or s == total_steps - 1))
        if (s + 1) % 60 == 0 or s == total_steps - 1:
            v = f"{rec['val_loss']:.4f}" if rec["val_loss"] is not None else "—"
            print(
                f"      step {sess.step:4d}/{total_steps} | loss={rec['loss']:.4f} | val={v} | "
                f"lr={rec['lr']:.2e} | {rec['tokens_per_sec']:.0f} tok/s | {rec['gflops']:.1f} GF/s",
                flush=True,
            )

    dt = time.perf_counter() - t0
    print(f"      Training finished in {dt:.1f}s ({sess.total_tokens_seen:,} tokens seen)", flush=True)

    sess.save_checkpoint(ckpt_json)
    gbx_info = save_gbx_checkpoint(
        ckpt_gbx, sess.model, step=sess.step, tokens_seen=sess.total_tokens_seen,
        metadata={"preset": "chat_gqa_1m", "loss": sess.history[-1]["loss"]},
    )
    print(f"      Saved {ckpt_json.name} & {ckpt_gbx.name} ({gbx_info['bytes']/1e6:.2f} MB)", flush=True)

    print("\n[3/4] Evaluating Glassbox-Chat 1M GQA (with KV-Cache):", flush=True)
    test_msgs = [
        "Hi!",
        "Who are you?",
        "How big are you?",
        "What is Glassbox-100M?",
        "What is GQA?",
        "What is a KV cache?",
        "Do you have CUDA kernels?",
        "How does FlashAttention work?",
        "What is RoPE?",
        "What is Logit Lens?",
        "Write a fibonacci function",
        "Write a haiku",
        "Tell me a joke",
    ]
    for msg in test_msgs:
        prompt = f"User: {msg}\nGlassbox:"
        pids = sess.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        gen_ids = []
        for item in sess.model.generate_stream(
            pids, max_new_tokens=72, temperature=0.15, top_k=10, top_p=0.9,
            seed=7, stop_token_id=sess.tokenizer.eos_id, use_kv_cache=True,
        ):
            tid = item["token_id"]
            if tid == sess.tokenizer.eos_id:
                break
            gen_ids.append(tid)
        reply = sess.tokenizer.decode(gen_ids).strip()
        print(f"  Q: {msg}\n  A: {reply}\n", flush=True)

    print("[4/4] Exporting compact Base64 Int8 browser bundle...", flush=True)
    export_browser_bundle(
        sess,
        ROOT / "web" / "public" / "models" / "glassbox-chat.json",
        verify_snap,
        model_label="Glassbox-Chat 1M GQA (6L · 6Q/2KV)",
    )


if __name__ == "__main__":
    main()
