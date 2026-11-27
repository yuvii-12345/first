"""Live Transformer Trainer & Lab Session Manager with thread-safe telemetry & SSE streaming."""

from __future__ import annotations

import json
import math
import queue
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from glassbox.backend import get_kernel_stats, reset_kernel_stats
from glassbox.bpe import BPETokenizer
from glassbox.datasets import DATASETS, MODEL_PRESETS, get_dataset_text
from glassbox.optim import AdamW, CosineWarmupScheduler, clip_grad_norm
from glassbox.tensor import no_grad
from glassbox.transformer import GPTConfig, GPTTransformer


class LabSession:
    """Owns the active BPE tokenizer, GPTTransformer, AdamW optimizer, dataset, and telemetry."""

    def __init__(
        self,
        dataset_id: str = "induction",
        preset_id: str = "balanced",
        custom_text: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        lr: float = 4e-3,
        batch_size: int = 8,
        weight_decay: float = 0.05,
        grad_clip: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.lock = threading.RLock()
        self._stop_event = threading.Event()
        self._train_thread: Optional[threading.Thread] = None
        self._subscribers: List[queue.Queue] = []
        self._sub_lock = threading.Lock()

        self.is_training = False
        self.target_steps = 0
        self.step = 0
        self.total_tokens_seen = 0
        self.history: List[Dict[str, Any]] = []
        self.latest_layer_grads: List[Dict[str, Any]] = []
        self.latest_sample: str = ""
        self.grad_health: str = "idle"
        self.last_step_ms: float = 0.0
        self.tokens_per_sec: float = 0.0
        self.live_gflops: float = 0.0

        self.reset(
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

    def reset(
        self,
        dataset_id: str = "induction",
        preset_id: str = "balanced",
        custom_text: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        lr: float = 4e-3,
        batch_size: int = 8,
        weight_decay: float = 0.05,
        grad_clip: float = 1.0,
        seed: int = 42,
    ) -> Dict[str, Any]:
        self._halt_thread()
        with self.lock:
            self.dataset_id = "custom" if custom_text else dataset_id
            self.preset_id = preset_id
            self.base_lr = float(lr)
            self.batch_size = max(1, min(32, int(batch_size)))
            self.weight_decay = float(weight_decay)
            self.grad_clip = float(grad_clip)
            self.seed = int(seed)
            self.rng = random.Random(self.seed)

            preset = MODEL_PRESETS.get(preset_id, MODEL_PRESETS["balanced"])
            cfg_dict = {
                "n_layers": preset["n_layers"],
                "n_heads": preset["n_heads"],
                "d_model": preset["d_model"],
                "d_ff": preset["d_ff"],
                "context_len": preset["context_len"],
                "vocab_size": preset["vocab_size"],
            }
            if config_overrides:
                for k in ("n_layers", "n_heads", "d_model", "d_ff", "context_len", "vocab_size"):
                    if k in config_overrides and config_overrides[k] is not None:
                        cfg_dict[k] = int(config_overrides[k])
            # Guarantee valid head divisibility & even head_dim for RoPE
            cfg_dict["n_heads"] = max(1, min(8, cfg_dict["n_heads"]))
            if cfg_dict["d_model"] % (cfg_dict["n_heads"] * 2) != 0:
                unit = cfg_dict["n_heads"] * 2
                cfg_dict["d_model"] = max(unit * 4, (cfg_dict["d_model"] // unit) * unit)

            corpus = custom_text if (custom_text and len(custom_text.strip()) >= 64) else get_dataset_text(dataset_id)
            self.corpus_text = corpus
            ds_meta = DATASETS.get(dataset_id, {}) if not custom_text else {}
            doc_delim = ds_meta.get("doc_delimiter") or ("\n" if ds_meta.get("line_delimited", False) else None)
            add_eos = bool(ds_meta.get("add_eos", False))
            self.line_delimited = doc_delim is not None

            # 1. Train Byte-Level BPE Tokenizer from scratch on corpus
            self.tokenizer = BPETokenizer()
            self.bpe_stats = self.tokenizer.train(corpus, target_vocab_size=cfg_dict["vocab_size"], min_frequency=2)

            # Encode dataset: document-aligned sequences for delimited corpora (chat/induction), stream for prose/code
            self.train_docs: List[List[int]] = []
            self.val_docs: List[List[int]] = []
            if doc_delim is not None:
                raw_docs = [d.strip() for d in corpus.split(doc_delim) if d.strip()]
                enc_docs = [self.tokenizer.encode(d, add_bos=True, add_eos=add_eos) for d in raw_docs]
                enc_docs = [d[: cfg_dict["context_len"] + 1] for d in enc_docs if len(d) >= 4]
                split_doc = max(1, int(len(enc_docs) * 0.9))
                self.train_docs = enc_docs[:split_doc]
                self.val_docs = enc_docs[split_doc:] or enc_docs[-max(1, len(enc_docs) // 10) :]
                self.train_ids = [tid for d in self.train_docs for tid in d]
                self.val_ids = [tid for d in self.val_docs for tid in d]
            else:
                all_ids = self.tokenizer.encode(corpus, add_bos=True, add_eos=True)
                split_idx = max(cfg_dict["context_len"] + 2, int(len(all_ids) * 0.9))
                self.train_ids = all_ids[:split_idx]
                self.val_ids = all_ids[split_idx:]
                if len(self.val_ids) < cfg_dict["context_len"] + 2:
                    self.val_ids = all_ids[-min(len(all_ids), cfg_dict["context_len"] * 4) :]

            # 2. Build GPT Transformer from scratch
            self.cfg = GPTConfig(
                vocab_size=self.tokenizer.vocab_size,
                d_model=cfg_dict["d_model"],
                n_layers=cfg_dict["n_layers"],
                n_heads=cfg_dict["n_heads"],
                d_ff=cfg_dict["d_ff"],
                context_len=cfg_dict["context_len"],
                tie_weights=True,
                dtype="float32",
            )
            self.model = GPTTransformer(self.cfg, seed=self.seed)
            self.optimizer = AdamW(
                self.model.named_parameters(),
                lr=self.base_lr,
                betas=(0.9, 0.95),
                weight_decay=self.weight_decay,
            )
            self.scheduler = CosineWarmupScheduler(
                max_lr=self.base_lr,
                min_lr=self.base_lr * 0.08,
                warmup_steps=20,
                total_steps=600,
            )

            self.step = 0
            self.total_tokens_seen = 0
            self.history = []
            self.latest_layer_grads = []
            self.latest_sample = ""
            self.grad_health = "ready"
            self.last_step_ms = 0.0
            self.tokens_per_sec = 0.0
            self.live_gflops = 0.0
            reset_kernel_stats()

        self._broadcast({"type": "reset", "status": self.get_status()})
        return self.get_status()

    # ---- Batch Sampling & Evaluation ----

    def _sample_batch(
        self, stream: Sequence[int], batch_size: int, docs: Optional[List[List[int]]] = None
    ) -> Tuple[List[List[int]], List[List[int]]]:
        if docs:
            chosen = [self.rng.choice(docs) for _ in range(batch_size)]
            T = min(self.cfg.context_len, max(len(d) for d in chosen) - 1)
            xs: List[List[int]] = []
            ys: List[List[int]] = []
            pad = self.tokenizer.pad_id
            for d in chosen:
                x = d[:T]
                y = d[1 : T + 1]
                if len(x) < T:
                    x = x + [pad] * (T - len(x))
                if len(y) < T:
                    y = y + [-100] * (T - len(y))
                xs.append(x)
                ys.append(y)
            return xs, ys

        T = min(self.cfg.context_len, len(stream) - 2)
        max_start = max(0, len(stream) - T - 1)
        xs = []
        ys = []
        for _ in range(batch_size):
            st = self.rng.randint(0, max_start) if max_start > 0 else 0
            xs.append(list(stream[st : st + T]))
            ys.append(list(stream[st + 1 : st + T + 1]))
        return xs, ys

    def eval_val_loss(self, num_batches: int = 3) -> float:
        with self.lock, no_grad():
            total = 0.0
            for _ in range(num_batches):
                xs, ys = self._sample_batch(
                    self.val_ids, min(self.batch_size, 8), docs=self.val_docs if self.line_delimited else None
                )
                _, loss = self.model.forward(xs, targets=ys)
                total += loss.item()
            return total / num_batches

    # ---- Single Training Step ----

    def _summarize_layer_grads(self, param_telemetry: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []

        def _bucket(name: str) -> Tuple[str, str]:
            if name.startswith("tok_emb"):
                return "tok_emb", "embed"
            if name.startswith("final_norm"):
                return "final_norm", "norm"
            if name.startswith("blocks."):
                parts = name.split(".")
                l_idx = parts[1]
                sub = parts[2]
                if sub == "attn":
                    return f"L{l_idx}.attn", "attn"
                if sub == "mlp":
                    return f"L{l_idx}.mlp", "mlp"
                return f"L{l_idx}.norm", "norm"
            return name, "other"

        for pname, stats in param_telemetry.items():
            key, kind = _bucket(pname)
            if key not in groups:
                groups[key] = {
                    "component": key,
                    "kind": kind,
                    "grad_sq": 0.0,
                    "upd_sq": 0.0,
                    "prm_sq": 0.0,
                }
                order.append(key)
            g = groups[key]
            g["grad_sq"] += stats["grad_norm"] ** 2
            g["upd_sq"] += stats["update_norm"] ** 2
            g["prm_sq"] += stats["param_norm"] ** 2

        out = []
        for key in order:
            g = groups[key]
            gn = math.sqrt(g["grad_sq"])
            un = math.sqrt(g["upd_sq"])
            pn = math.sqrt(g["prm_sq"])
            ratio = un / max(pn, 1e-12)
            out.append(
                {
                    "component": g["component"],
                    "kind": g["kind"],
                    "grad_norm": round(gn, 5),
                    "param_norm": round(pn, 4),
                    "update_ratio": ratio,
                    "log10_update_ratio": round(math.log10(max(ratio, 1e-12)), 3),
                }
            )
        return out

    def train_one_step(self, compute_val: bool = False) -> Dict[str, Any]:
        with self.lock:
            stats_before = get_kernel_stats()
            t0 = time.perf_counter()

            xs, ys = self._sample_batch(
                self.train_ids, self.batch_size, docs=self.train_docs if self.line_delimited else None
            )
            self.optimizer.zero_grad()
            _, loss = self.model.forward(xs, targets=ys)
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                self.grad_health = "nan_detected"
                raise RuntimeError("NaN/Inf loss detected")

            loss.backward()
            grad_norm, was_clipped = clip_grad_norm(self.model.named_parameters(), max_norm=self.grad_clip)

            lr = self.scheduler.get_lr(self.step)
            param_telem = self.optimizer.step(lr=lr)

            dt = time.perf_counter() - t0
            stats_after = get_kernel_stats()
            d_flops = max(0.0, stats_after["total_flops"] - stats_before["total_flops"])
            d_ns_sec = max(1e-9, stats_after["matmul_seconds"] - stats_before["matmul_seconds"])
            kernel_gflops = (d_flops / 1e9) / d_ns_sec

            self.step += 1
            tokens_this_step = len(xs) * len(xs[0])
            self.total_tokens_seen += tokens_this_step
            self.last_step_ms = dt * 1000.0
            self.tokens_per_sec = tokens_this_step / max(dt, 1e-6)
            self.live_gflops = kernel_gflops

            layer_grads = self._summarize_layer_grads(param_telem)
            self.latest_layer_grads = layer_grads

            # Diagnose gradient flow health
            if grad_norm < 1e-4:
                self.grad_health = "vanishing"
            elif grad_norm > self.grad_clip * 4.0:
                self.grad_health = "exploding_clipped"
            elif was_clipped:
                self.grad_health = "clipped"
            else:
                self.grad_health = "healthy"

            val_loss = None
            if compute_val or self.step == 1 or self.step % 20 == 0:
                val_loss = self.eval_val_loss(num_batches=2)

            if self.step == 1 or self.step % 40 == 0:
                prompt_text = self.default_prompt()
                p_ids = self.tokenizer.encode(prompt_text, add_bos=True)
                gen_ids = []
                for tok in self.model.generate_stream(p_ids, max_new_tokens=20, temperature=0.7, top_k=15, seed=self.step):
                    gen_ids.append(tok["token_id"])
                self.latest_sample = self.tokenizer.decode(gen_ids)

            ppl = math.exp(min(loss_val, 20.0))
            rec = {
                "step": self.step,
                "loss": round(loss_val, 4),
                "val_loss": round(val_loss, 4) if val_loss is not None else None,
                "perplexity": round(ppl, 2),
                "lr": round(lr, 6),
                "grad_norm": round(grad_norm, 4),
                "was_clipped": was_clipped,
                "grad_health": self.grad_health,
                "step_ms": round(self.last_step_ms, 2),
                "tokens_per_sec": round(self.tokens_per_sec, 1),
                "gflops": round(self.live_gflops, 2),
                "total_tokens": self.total_tokens_seen,
                "layer_grads": layer_grads,
                "latest_sample": self.latest_sample,
            }
            self.history.append(rec)
            if len(self.history) > 1200:
                self.history = self.history[-1200:]
            return rec

    # ---- Background Live Training Loop ----

    def start_training(self, num_steps: int = 150, lr: Optional[float] = None) -> Dict[str, Any]:
        if self.is_training:
            self.target_steps += int(num_steps)
            return self.get_status()

        with self.lock:
            if lr is not None and lr > 0:
                self.base_lr = float(lr)
            self.target_steps = self.step + int(num_steps)
            self.scheduler = CosineWarmupScheduler(
                max_lr=self.base_lr,
                min_lr=self.base_lr * 0.08,
                warmup_steps=min(20, max(5, int(num_steps * 0.1))),
                total_steps=max(self.target_steps + 20, 400),
            )
            self._stop_event.clear()
            self.is_training = True

        def _worker():
            try:
                while not self._stop_event.is_set() and self.step < self.target_steps:
                    rec = self.train_one_step(compute_val=(self.step + 1 == self.target_steps))
                    self._broadcast({"type": "step", "data": rec})
                    # Tiny yield so interactive HTTP interpretability calls acquire lock immediately
                    time.sleep(0.002)
            except Exception as exc:
                self._broadcast({"type": "error", "message": str(exc)})
            finally:
                with self.lock:
                    self.is_training = False
                self._broadcast({"type": "training_done", "status": self.get_status()})

        self._train_thread = threading.Thread(target=_worker, daemon=True, name="glassbox-trainer")
        self._train_thread.start()
        self._broadcast({"type": "training_started", "status": self.get_status()})
        return self.get_status()

    def _halt_thread(self) -> None:
        self._stop_event.set()
        t = self._train_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=3.0)
        with self.lock:
            self.is_training = False

    def stop_training(self) -> Dict[str, Any]:
        self._halt_thread()
        return self.get_status()

    def warm_start(self, steps: int = 320) -> None:
        """Synchronously pre-train initial steps so the Lab opens with real learned circuits."""
        self.scheduler = CosineWarmupScheduler(
            max_lr=self.base_lr,
            min_lr=self.base_lr * 0.65,
            warmup_steps=15,
            total_steps=max(steps + 250, 500),
        )
        for i in range(steps):
            self.train_one_step(compute_val=(i == 0 or i == steps - 1 or (i + 1) % 30 == 0))

    # ---- Autograd Computation Graph Capture ----

    def capture_autograd_dag(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Run a compact forward+backward pass on a shadow copy of gradients and export the live DAG."""
        with self.lock:
            text = prompt or self.default_prompt()
            ids = self.tokenizer.encode(text, add_bos=True)[:10]
            if len(ids) < 4:
                ids = self.train_ids[:8]
            xs = [ids[:-1]]
            ys = [ids[1:]]

            # Snapshot existing grads cleanly, run one backward pass, export graph, then clear
            self.model.zero_grad()
            _, loss = self.model.forward(xs, targets=ys)
            loss.backward()
            dag = loss.export_graph(max_nodes=140)
            loss_val = loss.item()
            self.model.zero_grad()

            return {
                "loss": round(loss_val, 5),
                "seq_len": len(xs[0]),
                "num_nodes": len(dag["nodes"]),
                "num_edges": len(dag["edges"]),
                "graph": dag,
            }

    # ---- SSE Pub/Sub ----

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast(self, event: Dict[str, Any]) -> None:
        with self._sub_lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait(event)
                    except Exception:
                        dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    # ---- Status & Metadata ----

    def default_prompt(self) -> str:
        if self.dataset_id in DATASETS:
            return DATASETS[self.dataset_id]["default_prompt"]
        return self.corpus_text[:120].strip()

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            last_rec = self.history[-1] if self.history else None
            kstats = get_kernel_stats()
            return {
                "is_training": self.is_training,
                "step": self.step,
                "target_steps": self.target_steps,
                "dataset_id": self.dataset_id,
                "preset_id": self.preset_id,
                "config": self.cfg.to_dict(),
                "num_parameters": self.model.num_parameters(),
                "bpe_stats": self.bpe_stats,
                "train_tokens": len(self.train_ids),
                "val_tokens": len(self.val_ids),
                "total_tokens_seen": self.total_tokens_seen,
                "lr": self.base_lr,
                "batch_size": self.batch_size,
                "grad_clip": self.grad_clip,
                "grad_health": self.grad_health,
                "last_loss": last_rec["loss"] if last_rec else None,
                "last_val_loss": next(
                    (h["val_loss"] for h in reversed(self.history) if h.get("val_loss") is not None),
                    None,
                ),
                "last_perplexity": last_rec["perplexity"] if last_rec else None,
                "last_grad_norm": last_rec["grad_norm"] if last_rec else None,
                "last_step_ms": round(self.last_step_ms, 2),
                "tokens_per_sec": round(self.tokens_per_sec, 1),
                "live_gflops": round(self.live_gflops, 2),
                "cumulative_gflops": round(kstats["gflops"], 2),
                "total_matmul_calls": kstats["matmul_calls"],
                "latest_sample": self.latest_sample,
                "latest_layer_grads": self.latest_layer_grads,
                "default_prompt": self.default_prompt(),
            }

    def save_checkpoint(self, path: str | Path) -> None:
        with self.lock:
            payload = {
                "dataset_id": self.dataset_id,
                "preset_id": self.preset_id,
                "step": self.step,
                "total_tokens_seen": self.total_tokens_seen,
                "base_lr": self.base_lr,
                "batch_size": self.batch_size,
                "cfg": self.cfg.to_dict(),
                "tokenizer": self.tokenizer.to_dict(),
                "bpe_stats": self.bpe_stats,
                "history": self.history[-400:],
                "latest_layer_grads": self.latest_layer_grads,
                "latest_sample": self.latest_sample,
                "state_dict": self.model.state_dict(),
            }
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

    def load_checkpoint(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        with self.lock:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.dataset_id = data["dataset_id"]
            self.preset_id = data["preset_id"]
            self.base_lr = data["base_lr"]
            self.batch_size = data["batch_size"]
            self.corpus_text = get_dataset_text(self.dataset_id)
            ds_meta = DATASETS.get(self.dataset_id, {})
            doc_delim = ds_meta.get("doc_delimiter") or ("\n" if ds_meta.get("line_delimited", False) else None)
            add_eos = bool(ds_meta.get("add_eos", False))
            self.line_delimited = doc_delim is not None
            self.tokenizer = BPETokenizer.from_dict(data["tokenizer"])
            self.bpe_stats = data["bpe_stats"]
            self.cfg = GPTConfig.from_dict(data["cfg"])

            self.train_docs = []
            self.val_docs = []
            if doc_delim is not None:
                raw_docs = [d.strip() for d in self.corpus_text.split(doc_delim) if d.strip()]
                enc_docs = [self.tokenizer.encode(d, add_bos=True, add_eos=add_eos) for d in raw_docs]
                enc_docs = [d[: self.cfg.context_len + 1] for d in enc_docs if len(d) >= 4]
                split_doc = max(1, int(len(enc_docs) * 0.9))
                self.train_docs = enc_docs[:split_doc]
                self.val_docs = enc_docs[split_doc:] or enc_docs[-max(1, len(enc_docs) // 10) :]
                self.train_ids = [tid for d in self.train_docs for tid in d]
                self.val_ids = [tid for d in self.val_docs for tid in d]
            else:
                all_ids = self.tokenizer.encode(self.corpus_text, add_bos=True, add_eos=True)
                split_idx = int(len(all_ids) * 0.9)
                self.train_ids = all_ids[:split_idx]
                self.val_ids = all_ids[split_idx:]

            self.model = GPTTransformer(self.cfg, seed=self.seed)
            self.model.load_state_dict(data["state_dict"])
            self.optimizer = AdamW(
                self.model.named_parameters(),
                lr=self.base_lr,
                betas=(0.9, 0.95),
                weight_decay=self.weight_decay,
            )
            self.step = data["step"]
            self.total_tokens_seen = data["total_tokens_seen"]
            self.history = data["history"]
            self.latest_layer_grads = data.get("latest_layer_grads", [])
            self.latest_sample = data.get("latest_sample", "")
            self.grad_health = "healthy"
            return True
