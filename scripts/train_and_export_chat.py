"""Train / resume the from-scratch Glassbox-Chat 4L GPT Transformer and export Int8 browser bundles."""

from __future__ import annotations

import base64
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glassbox.backend import benchmark_matmul
from glassbox.datasets import MODEL_PRESETS, list_datasets_meta
from glassbox.gradcheck import gradcheck, run_full_gradcheck_suite, verify_flash_attention_equivalence
from glassbox.interp import run_causal_activation_patching
from glassbox.trainer import LabSession
from glassbox.transformer import GPTConfig, GPTTransformer


def _pack_model_int8(model: GPTTransformer) -> Dict[str, Any]:
    """Pack 2D matrices as symmetric per-row Int8 + Float32 scales, and 1D vectors as raw Float32."""
    packed: Dict[str, Any] = {}
    for name, param in model.named_parameters():
        shape = list(param.shape)
        vals = [float(param._buf[i]) for i in range(param.numel)]
        if len(shape) == 2:
            rows, cols = shape
            q_bytes = bytearray(rows * cols)
            scales: List[float] = []
            for r in range(rows):
                off = r * cols
                row_slice = vals[off : off + cols]
                mx = max((abs(v) for v in row_slice), default=0.0)
                scale = (mx / 127.0) if mx > 1e-12 else 1.0
                scales.append(scale)
                inv = 1.0 / scale
                for c, v in enumerate(row_slice):
                    q = int(round(v * inv))
                    if q > 127:
                        q = 127
                    elif q < -127:
                        q = -127
                    q_bytes[off + c] = q & 0xFF
            packed[name] = {
                "shape": shape,
                "quant": "int8_row",
                "scales_b64": base64.b64encode(struct.pack(f"<{rows}f", *scales)).decode("ascii"),
                "data_b64": base64.b64encode(bytes(q_bytes)).decode("ascii"),
            }
        else:
            packed[name] = {
                "shape": shape,
                "quant": "f32",
                "data_b64": base64.b64encode(struct.pack(f"<{len(vals)}f", *vals)).decode("ascii"),
            }
    return packed


def _build_verify_snapshot() -> Dict[str, Any]:
    t0 = time.perf_counter()
    gradchecks = run_full_gradcheck_suite()
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
    res = gradcheck(
        lambda *_: model.forward(xs, targets=ys)[1],
        model.parameters(),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
        max_elements_per_tensor=20,
    )
    gradchecks.append({"op": "Full 2-Layer GPT Transformer (End-to-End)", **res})
    flash_causal = verify_flash_attention_equivalence(B=2, H=3, T=45, D=16, causal=True, dtype="float64")
    flash_noncausal = verify_flash_attention_equivalence(B=2, H=2, T=41, D=16, causal=False, dtype="float64")
    dt_ms = (time.perf_counter() - t0) * 1000.0
    bench = benchmark_matmul([64, 128, 256])
    return {
        "verify": {
            "all_passed": all(g["passed"] for g in gradchecks)
            and flash_causal["passed"]
            and flash_noncausal["passed"],
            "duration_ms": round(dt_ms, 1),
            "gradchecks": gradchecks,
            "flash_equivalence": [flash_causal, flash_noncausal],
        },
        "benchmark": bench,
    }


def export_browser_bundle(
    sess: LabSession,
    out_path: Path,
    verify_snapshot: Dict[str, Any],
    model_label: str,
) -> None:
    st = sess.get_status()
    dag = sess.capture_autograd_dag()
    bundle = {
        "format": "glassbox_browser_v1",
        "model_label": model_label,
        "status": st,
        "history": sess.history[-300:],
        "datasets": list_datasets_meta(),
        "presets": list(MODEL_PRESETS.values()),
        "cfg": sess.cfg.to_dict(),
        "tokenizer": sess.tokenizer.to_dict(),
        "weights": _pack_model_int8(sess.model),
        "verify_snapshot": verify_snapshot["verify"],
        "benchmark_snapshot": verify_snapshot["benchmark"],
        "autograd_dag_snapshot": dag,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024.0
    print(f"[Export] Wrote {out_path} ({size_kb:.1f} KiB)", flush=True)


def main() -> None:
    extra_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 260
    ckpt_chat = ROOT / "checkpoints" / "chat_v1.json"

    print("[1/4] Capturing float64 gradcheck, FlashAttention-2 & C++17 GEMM proof snapshot...", flush=True)
    verify_snap = _build_verify_snapshot()
    print(f"      All gradchecks passed: {verify_snap['verify']['all_passed']}", flush=True)

    sess = LabSession(
        dataset_id="chat",
        preset_id="chat_4l",
        lr=4.5e-3,
        batch_size=16,
        weight_decay=0.01,
        grad_clip=1.0,
        seed=42,
    )
    if ckpt_chat.exists():
        print(f"[2/4] Resuming {ckpt_chat} for +{extra_steps} steps to sharpen working-memory induction heads...", flush=True)
        sess.load_checkpoint(ckpt_chat)
        # Keep steady mid-LR so induction heads cross the phase transition cleanly
        sess.base_lr = 3.0e-3
        sess.scheduler.max_lr = 3.0e-3
        sess.scheduler.min_lr = 1.2e-3
        sess.scheduler.warmup_steps = 10
        sess.scheduler.total_steps = sess.step + extra_steps
    else:
        print(f"[2/4] Training fresh Glassbox-Chat for {extra_steps} steps...", flush=True)
        sess.scheduler.warmup_steps = 25
        sess.scheduler.total_steps = extra_steps
        sess.scheduler.min_lr = sess.base_lr * 0.25

    t0 = time.perf_counter()
    for s in range(extra_steps):
        rec = sess.train_one_step(compute_val=(s % 40 == 39 or s == extra_steps - 1))
        if (s + 1) % 65 == 0 or s == extra_steps - 1:
            v = f"{rec['val_loss']:.4f}" if rec["val_loss"] is not None else "—"
            print(
                f"      step {sess.step:4d} | loss={rec['loss']:.4f} | val={v} | "
                f"{rec['tokens_per_sec']:.0f} tok/s | {rec['gflops']:.1f} GF/s",
                flush=True,
            )

    dt = time.perf_counter() - t0
    print(f"      Done in {dt:.1f}s | total step={sess.step}", flush=True)
    sess.save_checkpoint(ckpt_chat)

    print("\n[3/4] Evaluating Glassbox-Chat replies:", flush=True)
    test_msgs = [
        "Hi!",
        "Who are you?",
        "How does FlashAttention work?",
        "What is Logit Lens?",
        "Tell me a joke",
        "Write a haiku",
        "Write a fibonacci function",
        "Remember my color is cobalt. What is my color?",
        "Remember my code is scarlet. What is my code?",
        "Remember alpha is crimson and beta is emerald. What is beta?",
        "Repeat: silver amber crimson violet",
    ]
    for msg in test_msgs:
        prompt = f"User: {msg}\nGlassbox:"
        pids = sess.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        gen_ids = []
        for item in sess.model.generate_stream(
            pids,
            max_new_tokens=96,
            temperature=0.05,
            top_k=5,
            top_p=0.9,
            seed=7,
            stop_token_id=sess.tokenizer.eos_id,
        ):
            if item["token_id"] == sess.tokenizer.eos_id:
                break
            gen_ids.append(item["token_id"])
        reply = sess.tokenizer.decode(gen_ids).strip()
        print(f"  User:     {msg}")
        print(f"  Glassbox: {reply}\n")

    patch = run_causal_activation_patching(
        sess.model,
        sess.tokenizer,
        clean_prompt="User: Remember my color is cobalt. What is my color?\nGlassbox: Your color is",
        corrupted_prompt="User: Remember my color is violet. What is my color?\nGlassbox: Your color is",
        clean_target=" cobalt",
        corrupted_target=" violet",
    )
    print(
        f"  [Memory Causal Patching] LD_clean={patch['logit_diff_clean']:+.2f}, "
        f"LD_corr={patch['logit_diff_corrupt']:+.2f}, "
        f"top_hotspot={patch['top_hotspots'][0] if patch['top_hotspots'] else None}",
        flush=True,
    )

    print("\n[4/4] Exporting compact Int8 browser bundles to web/public/models/...", flush=True)
    export_browser_bundle(
        sess,
        ROOT / "web" / "public" / "models" / "glassbox-chat.json",
        verify_snap,
        model_label="Glassbox-Chat (4L Conversational + Memory)",
    )

    ind_ckpt = ROOT / "checkpoints" / "startup_v1.json"
    if ind_ckpt.exists():
        ind_sess = LabSession(dataset_id="induction", preset_id="micro", seed=7)
        ind_sess.load_checkpoint(ind_ckpt)
        export_browser_bundle(
            ind_sess,
            ROOT / "web" / "public" / "models" / "glassbox-induction.json",
            verify_snap,
            model_label="Glassbox-Induction (3L 100%-Recovery Circuit)",
        )


if __name__ == "__main__":
    main()
