"""Glassbox Backend Server — pure Python stdlib HTTP + SSE server (zero external web frameworks).

Serves both the built React Mechanistic Interpretability Lab (`web/dist`) and all JSON/SSE APIs.
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

# Ensure repository root is importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from glassbox.backend import benchmark_matmul
from glassbox.datasets import MODEL_PRESETS, list_datasets_meta
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
from glassbox.trainer import LabSession
from glassbox.transformer import GPTConfig, GPTTransformer

WEB_DIST = ROOT_DIR / "web" / "dist"
CHAT_CHECKPOINT_FILE = ROOT_DIR / "checkpoints" / "chat_v1.json"
INDUCTION_CHECKPOINT_FILE = ROOT_DIR / "checkpoints" / "startup_v1.json"

SESSION: LabSession | None = None


def init_session() -> LabSession:
    global SESSION
    SESSION = LabSession(
        dataset_id="chat" if CHAT_CHECKPOINT_FILE.exists() else "induction",
        preset_id="chat_4l" if CHAT_CHECKPOINT_FILE.exists() else "micro",
        lr=4.5e-3 if CHAT_CHECKPOINT_FILE.exists() else 5e-3,
        batch_size=16,
        weight_decay=0.01,
        grad_clip=1.0,
        seed=42,
    )
    if CHAT_CHECKPOINT_FILE.exists():
        print(f"[Glassbox] Loading Chat 4L checkpoint from {CHAT_CHECKPOINT_FILE}...", flush=True)
        SESSION.load_checkpoint(CHAT_CHECKPOINT_FILE)
        print(f"[Glassbox] Loaded step={SESSION.step}, loss={SESSION.history[-1]['loss'] if SESSION.history else 'N/A'}", flush=True)
    elif INDUCTION_CHECKPOINT_FILE.exists():
        print(f"[Glassbox] Loading startup checkpoint from {INDUCTION_CHECKPOINT_FILE}...", flush=True)
        SESSION.load_checkpoint(INDUCTION_CHECKPOINT_FILE)
        print(f"[Glassbox] Loaded step={SESSION.step}, loss={SESSION.history[-1]['loss'] if SESSION.history else 'N/A'}", flush=True)
    else:
        print("[Glassbox] Warm-starting default 3L Transformer (360 steps)...", flush=True)
        t0 = time.perf_counter()
        SESSION.warm_start(steps=360)
        SESSION.save_checkpoint(INDUCTION_CHECKPOINT_FILE)
        print(
            f"[Glassbox] Warm-start complete in {time.perf_counter() - t0:.2f}s | "
            f"step={SESSION.step} | loss={SESSION.history[-1]['loss']:.3f}",
            flush=True,
        )
    return SESSION


def _run_transformer_gradcheck() -> Dict[str, Any]:
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
    return {"op": "Full 2-Layer GPT Transformer (End-to-End)", **res}


class GlassboxHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Quiet routine asset polling; print errors cleanly
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write(f"[HTTP] {self.address_string()} - {format % args}\n")

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_error_json(self, msg: str, status: int = 400) -> None:
        self._send_json({"error": msg}, status=status)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(min(length, 5 * 1024 * 1024))
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _serve_static(self, req_path: str) -> None:
        if not WEB_DIST.exists():
            body = b"Glassbox frontend not built yet. Run `cd web && npm run build`."
            self.send_response(503)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        rel = req_path.lstrip("/")
        if not rel or rel.endswith("/"):
            rel = "index.html"
        target = (WEB_DIST / rel).resolve()
        # Prevent path traversal outside WEB_DIST
        try:
            target.relative_to(WEB_DIST.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return

        if not target.exists() or not target.is_file():
            target = WEB_DIST / "index.html"

        data = target.read_bytes()
        mime, _ = mimetypes.guess_type(str(target))
        if mime is None:
            mime = "application/octet-stream"
        if mime.startswith("text/") or mime in ("application/javascript", "application/json"):
            mime = f"{mime}; charset=utf-8"

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        if rel.startswith("assets/"):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _handle_sse_events(self) -> None:
        assert SESSION is not None
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        q = SESSION.subscribe()
        try:
            # Send initial handshake status
            init_payload = json.dumps({"type": "connected", "status": SESSION.get_status()})
            self.wfile.write(f"data: {init_payload}\n\n".encode("utf-8"))
            self.wfile.flush()

            while True:
                try:
                    ev = q.get(timeout=15.0)
                    msg = f"data: {json.dumps(ev)}\n\n".encode("utf-8")
                    self.wfile.write(msg)
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            SESSION.unsubscribe(q)

    # ---- GET Routes ----

    def do_GET(self) -> None:
        assert SESSION is not None
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._send_json({"ok": True, "step": SESSION.step})
            return

        if path == "/api/status":
            with SESSION.lock:
                status = SESSION.get_status()
                history = list(SESSION.history)
            self._send_json(
                {
                    "status": status,
                    "history": history,
                    "datasets": list_datasets_meta(),
                    "presets": list(MODEL_PRESETS.values()),
                }
            )
            return

        if path == "/api/events":
            self._handle_sse_events()
            return

        if path.startswith("/api/"):
            self._send_error_json(f"Unknown API endpoint: {path}", status=404)
            return

        self._serve_static(path)

    # ---- POST Routes ----

    def do_POST(self) -> None:
        assert SESSION is not None
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_error_json(f"Invalid JSON payload: {exc}", status=400)
            return

        try:
            if path == "/api/train/start":
                steps = int(body.get("steps", 150))
                lr = body.get("lr")
                st = SESSION.start_training(num_steps=max(1, min(2000, steps)), lr=float(lr) if lr else None)
                self._send_json({"status": st})
                return

            if path == "/api/train/stop":
                st = SESSION.stop_training()
                self._send_json({"status": st})
                return

            if path == "/api/train/step":
                rec = SESSION.train_one_step(compute_val=True)
                self._send_json({"step_record": rec, "status": SESSION.get_status()})
                return

            if path == "/api/session/reset":
                dataset_id = str(body.get("dataset_id", SESSION.dataset_id if SESSION.dataset_id != "custom" else "induction"))
                preset_id = str(body.get("preset_id", SESSION.preset_id))
                custom_text = body.get("custom_text")
                config_overrides = body.get("config_overrides")
                lr = float(body.get("lr", 4e-3))
                batch_size = int(body.get("batch_size", 12))
                weight_decay = float(body.get("weight_decay", 0.01))
                grad_clip = float(body.get("grad_clip", 1.0))
                seed = int(body.get("seed", 42))
                warm_steps = int(body.get("warm_steps", 0))

                st = SESSION.reset(
                    dataset_id=dataset_id,
                    preset_id=preset_id,
                    custom_text=custom_text,
                    config_overrides=config_overrides,
                    lr=lr,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    grad_clip=grad_clip,
                    seed=seed,
                )
                if warm_steps > 0:
                    SESSION.warm_start(steps=min(400, warm_steps))
                    st = SESSION.get_status()
                with SESSION.lock:
                    hist = list(SESSION.history)
                self._send_json({"status": st, "history": hist})
                return

            if path == "/api/models/switch":
                model_id = str(body.get("model_id", "chat"))
                ckpt = CHAT_CHECKPOINT_FILE if model_id == "chat" else INDUCTION_CHECKPOINT_FILE
                if ckpt.exists():
                    SESSION.stop_training()
                    SESSION.load_checkpoint(ckpt)
                    with SESSION.lock:
                        st = SESSION.get_status()
                        hist = list(SESSION.history)
                    SESSION._broadcast({"type": "reset", "status": st})
                    self._send_json({"status": st, "history": hist})
                else:
                    self._send_error_json(f"Checkpoint for model '{model_id}' not found", status=404)
                return

            if path == "/api/chat":
                import math
                import random as _random
                from glassbox.tensor import no_grad

                messages = body.get("messages") or []
                max_new = max(1, min(140, int(body.get("max_new_tokens", 100))))
                temperature = float(body.get("temperature", 0.35))
                top_k = int(body.get("top_k", 16))
                top_p = float(body.get("top_p", 0.9))
                ablate_list = body.get("ablate_heads") or []
                ablation_mode = str(body.get("ablation_mode", "zero"))

                # Build formatted prompt respecting active dataset format
                with SESSION.lock:
                    is_chat_ds = SESSION.dataset_id == "chat"
                    if is_chat_ds:
                        # Keep last 2 turns so prompt fits comfortably inside context_len=128
                        recent = messages[-4:]
                        parts = []
                        for m in recent:
                            role = "User" if m.get("role") == "user" else "Glassbox"
                            parts.append(f"{role}: {str(m.get('content', '')).strip()}")
                        prompt = "\n".join(parts) + "\nGlassbox:"
                    else:
                        last_user = next(
                            (str(m.get("content", "")).strip() for m in reversed(messages) if m.get("role") == "user"),
                            SESSION.default_prompt(),
                        )
                        prompt = last_user

                    from glassbox.transformer import InterpretabilityTrace, KVCache

                    ablate_set = {
                        (int(pair[0]), int(pair[1]))
                        for pair in ablate_list
                        if isinstance(pair, (list, tuple)) and len(pair) == 2
                    }

                    p_ids = SESSION.tokenizer.encode(prompt, add_bos=True, add_eos=False)
                    seq = list(p_ids)
                    steps_out = []
                    gen_ids = []
                    rng = _random.Random()
                    eos_id = SESSION.tokenizer.eos_id
                    V = SESSION.cfg.vocab_size
                    cache = (
                        KVCache(SESSION.cfg, batch_size=1)
                        if (ablation_mode != "mean" and len(seq) <= SESSION.cfg.context_len)
                        else None
                    )

                    with no_grad():
                        for step_idx in range(max_new):
                            trace = (
                                InterpretabilityTrace(ablate_heads=ablate_set, ablation_mode=ablation_mode)
                                if ablate_set
                                else None
                            )
                            if cache is not None and cache.seq_len < SESSION.cfg.context_len:
                                if step_idx == 0:
                                    ctx = seq[-SESSION.cfg.context_len :]
                                    logits, _ = SESSION.model.forward([ctx], trace=trace, kv_cache=cache)
                                    last_off = (len(ctx) - 1) * V
                                else:
                                    logits, _ = SESSION.model.forward([[seq[-1]]], trace=trace, kv_cache=cache)
                                    last_off = 0
                            else:
                                cache = None
                                ctx = seq[-SESSION.cfg.context_len :]
                                logits, _ = SESSION.model.forward([ctx], trace=trace)
                                last_off = (len(ctx) - 1) * V
                            raw_logits = [float(logits._buf[last_off + v]) for v in range(V)]
                            temp = max(temperature, 1e-5)
                            scaled = [lg / temp for lg in raw_logits]
                            mx = max(scaled)
                            exps = [math.exp(v - mx) for v in scaled]
                            s_exp = sum(exps)
                            probs = [e / s_exp for e in exps]
                            entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs if p > 1e-12)

                            ranked = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)
                            top_ranked = ranked[:top_k] if top_k > 0 else ranked
                            filtered = []
                            cum = 0.0
                            for tid, p in top_ranked:
                                filtered.append((tid, p))
                                cum += p
                                if cum >= top_p and len(filtered) >= 1:
                                    break
                            if temperature <= 1e-4:
                                next_id = filtered[0][0]
                            else:
                                tot_p = sum(p for _, p in filtered)
                                r = rng.random() * tot_p
                                acc = 0.0
                                next_id = filtered[-1][0]
                                for tid, p in filtered:
                                    acc += p
                                    if r <= acc:
                                        next_id = tid
                                        break

                            if next_id == eos_id:
                                break

                            seq.append(next_id)
                            gen_ids.append(next_id)
                            steps_out.append(
                                {
                                    "step": step_idx,
                                    "id": next_id,
                                    "text": SESSION.tokenizer.token_text(next_id),
                                    "display": SESSION.tokenizer.token_display(next_id),
                                    "prob": round(probs[next_id], 5),
                                    "entropy": round(entropy, 4),
                                    "top_candidates": [
                                        {
                                            "id": tid,
                                            "display": SESSION.tokenizer.token_display(tid),
                                            "prob": round(p, 5),
                                        }
                                        for tid, p in ranked[:8]
                                    ],
                                }
                            )
                            # Stop cleanly if the model starts a new User: turn
                            decoded_so_far = SESSION.tokenizer.decode(gen_ids)
                            if is_chat_ds and "\nUser:" in decoded_so_far:
                                break

                    reply = SESSION.tokenizer.decode(gen_ids)
                    if is_chat_ds and "\nUser:" in reply:
                        reply = reply.split("\nUser:", 1)[0]
                    reply = reply.strip()

                self._send_json(
                    {
                        "prompt": prompt,
                        "reply": reply,
                        "tokens": steps_out,
                    }
                )
                return

            if path == "/api/generate":
                prompt = str(body.get("prompt", SESSION.default_prompt()))
                max_new = max(1, min(160, int(body.get("max_new_tokens", 48))))
                temperature = float(body.get("temperature", 0.7))
                top_k = int(body.get("top_k", 20))
                top_p = float(body.get("top_p", 0.92))
                seed = body.get("seed")

                with SESSION.lock:
                    p_ids = SESSION.tokenizer.encode(prompt, add_bos=True)
                    steps_out = []
                    gen_ids = []
                    for item in SESSION.model.generate_stream(
                        p_ids,
                        max_new_tokens=max_new,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=int(seed) if seed is not None else None,
                    ):
                        tid = item["token_id"]
                        gen_ids.append(tid)
                        steps_out.append(
                            {
                                "step": item["step"],
                                "id": tid,
                                "text": SESSION.tokenizer.token_text(tid),
                                "display": SESSION.tokenizer.token_display(tid),
                                "prob": item["prob"],
                                "entropy": item["entropy"],
                                "top_candidates": [
                                    {
                                        "id": c["id"],
                                        "display": SESSION.tokenizer.token_display(c["id"]),
                                        "prob": c["prob"],
                                    }
                                    for c in item["top_candidates"]
                                ],
                            }
                        )
                    generated_text = SESSION.tokenizer.decode(gen_ids)
                self._send_json(
                    {
                        "prompt": prompt,
                        "generated_text": generated_text,
                        "full_text": prompt + generated_text,
                        "tokens": steps_out,
                    }
                )
                return

            if path == "/api/interp/attention":
                prompt = str(body.get("prompt", SESSION.default_prompt()))
                ablate_heads = body.get("ablate_heads", [])
                ablation_mode = str(body.get("ablation_mode", "zero"))
                with SESSION.lock:
                    res = analyze_attention_and_ablation(
                        SESSION.model,
                        SESSION.tokenizer,
                        prompt,
                        ablate_heads=ablate_heads,
                        ablation_mode=ablation_mode,
                    )
                self._send_json(res)
                return

            if path == "/api/interp/logit-lens":
                prompt = str(body.get("prompt", SESSION.default_prompt()))
                target_token = body.get("target_token") or None
                foil_token = body.get("foil_token") or None
                with SESSION.lock:
                    res = analyze_logit_lens_and_dla(
                        SESSION.model,
                        SESSION.tokenizer,
                        prompt,
                        target_token=target_token,
                        foil_token=foil_token,
                    )
                self._send_json(res)
                return

            if path == "/api/interp/patching":
                clean_prompt = str(body.get("clean_prompt", ""))
                corrupted_prompt = str(body.get("corrupted_prompt", ""))
                clean_target = body.get("clean_target") or None
                corrupted_target = body.get("corrupted_target") or None
                if not clean_prompt or not corrupted_prompt:
                    self._send_error_json("Both clean_prompt and corrupted_prompt are required", status=400)
                    return
                with SESSION.lock:
                    res = run_causal_activation_patching(
                        SESSION.model,
                        SESSION.tokenizer,
                        clean_prompt=clean_prompt,
                        corrupted_prompt=corrupted_prompt,
                        clean_target=clean_target,
                        corrupted_target=corrupted_target,
                    )
                self._send_json(res)
                return

            if path == "/api/bpe/inspect":
                text = str(body.get("text", SESSION.default_prompt()))
                with SESSION.lock:
                    inspection = SESSION.tokenizer.inspect_encode(text)
                    merges = SESSION.tokenizer.get_merge_table(limit=100)
                    vocab_size = SESSION.tokenizer.vocab_size
                    bpe_stats = SESSION.bpe_stats
                self._send_json(
                    {
                        "vocab_size": vocab_size,
                        "bpe_stats": bpe_stats,
                        "inspection": inspection,
                        "merges": merges,
                    }
                )
                return

            if path == "/api/engine/autograd-dag":
                prompt = body.get("prompt")
                res = SESSION.capture_autograd_dag(prompt=prompt)
                self._send_json(res)
                return

            if path == "/api/engine/verify":
                t0 = time.perf_counter()
                gradchecks = run_full_gradcheck_suite()
                gradchecks.append(_run_transformer_gradcheck())
                flash_causal = verify_flash_attention_equivalence(B=2, H=3, T=45, D=16, causal=True, dtype="float64")
                flash_noncausal = verify_flash_attention_equivalence(B=2, H=2, T=41, D=16, causal=False, dtype="float64")
                dt_ms = (time.perf_counter() - t0) * 1000.0
                self._send_json(
                    {
                        "all_passed": all(g["passed"] for g in gradchecks)
                        and flash_causal["passed"]
                        and flash_noncausal["passed"],
                        "duration_ms": round(dt_ms, 1),
                        "gradchecks": gradchecks,
                        "flash_equivalence": [flash_causal, flash_noncausal],
                    }
                )
                return

            if path == "/api/engine/benchmark":
                sizes = body.get("sizes") or [64, 128, 256, 384]
                sizes = [max(16, min(512, int(s))) for s in sizes]
                rows = benchmark_matmul(sizes)
                self._send_json({"results": rows})
                return

            self._send_error_json(f"Unknown API endpoint: {path}", status=404)
        except Exception as exc:
            self._send_error_json(str(exc), status=500)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    init_session()
    server = ReusableThreadingHTTPServer((host, port), GlassboxHandler)
    print(f"[Glassbox] Server listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Glassbox] Shutting down...", flush=True)
    finally:
        if SESSION is not None:
            SESSION.stop_training()
        server.server_close()


if __name__ == "__main__":
    main()
