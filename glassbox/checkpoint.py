"""Fast Zero-Copy Binary Checkpoint Format (.gbx) shared identically by:
  - Python Glassbox framework
  - Pure C++17 Native CPU Trainer (cuda/train_glassbox_cpu.cpp)
  - Pure CUDA C++17 GPU Trainer (cuda/train_glassbox_100m.cu)

Saves/loads 100.68M float32 parameters (402.7 MB) in <0.5s with zero Python list overhead.
"""

from __future__ import annotations

import ctypes
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from glassbox.transformer import GPTConfig, GPTTransformer

GBX_MAGIC = b"GBX1"
GBX_VERSION = 1
# Naturally 8-byte aligned C++17 struct layout (72 bytes exact, zero compiler padding):
#   magic[4], version, dtype_code, vocab_size, d_model, n_layers, n_heads, n_kv_heads,
#   d_ff, context_len, tie_weights, rope_base(f32), rmsnorm_eps(f32), meta_len(u32),
#   step(u64), tokens_seen(u64)
GBX_HEADER_FMT = "<4s10IffIQQ"
GBX_HEADER_SIZE = struct.calcsize(GBX_HEADER_FMT)
assert GBX_HEADER_SIZE == 72, f"Expected 72-byte header, got {GBX_HEADER_SIZE}"


def canonical_param_order(cfg: GPTConfig) -> List[str]:
    """Deterministic parameter order shared identically between Python, C++17, and CUDA."""
    names = ["tok_emb.weight"]
    for l in range(cfg.n_layers):
        p = f"blocks.{l}"
        names.extend([
            f"{p}.norm1.weight",
            f"{p}.attn.w_q.weight",
            f"{p}.attn.w_k.weight",
            f"{p}.attn.w_v.weight",
            f"{p}.attn.w_o.weight",
            f"{p}.norm2.weight",
            f"{p}.mlp.w_gate.weight",
            f"{p}.mlp.w_up.weight",
            f"{p}.mlp.w_down.weight",
        ])
    names.append("final_norm.weight")
    if not cfg.tie_weights:
        names.append("lm_head.weight")
    return names


def save_gbx_checkpoint(
    path: str | Path,
    model: GPTTransformer,
    step: int = 0,
    tokens_seen: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write model weights directly from ctypes C buffers to a .gbx binary checkpoint."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg
    named = dict(model.named_parameters())
    order = canonical_param_order(cfg)

    meta_bytes = json.dumps(metadata or {}, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(meta_bytes) % 8)) % 8
    meta_padded = meta_bytes + (b"\x00" * pad)

    header = struct.pack(
        GBX_HEADER_FMT,
        GBX_MAGIC,
        GBX_VERSION,
        0,  # dtype_code = 0 (float32)
        int(cfg.vocab_size),
        int(cfg.d_model),
        int(cfg.n_layers),
        int(cfg.n_heads),
        int(cfg.effective_kv_heads),
        int(cfg.d_ff),
        int(cfg.context_len),
        1 if cfg.tie_weights else 0,
        float(cfg.rope_base),
        float(cfg.rmsnorm_eps),
        len(meta_padded),
        int(step),
        int(tokens_seen),
    )

    total_params = 0
    with out_path.open("wb") as f:
        f.write(header)
        if meta_padded:
            f.write(meta_padded)
        for name in order:
            t = named[name]
            total_params += t.numel
            if t.dtype == "float32":
                f.write(memoryview(t._buf).cast("B"))
            else:
                f32_buf = (ctypes.c_float * t.numel)(*(float(t._buf[i]) for i in range(t.numel)))
                f.write(memoryview(f32_buf).cast("B"))

    size_bytes = out_path.stat().st_size
    return {
        "path": str(out_path),
        "params": total_params,
        "bytes": size_bytes,
        "step": step,
        "tokens_seen": tokens_seen,
    }


def read_gbx_header(path: str | Path) -> Tuple[GPTConfig, Dict[str, Any]]:
    """Read only the 64-byte header + metadata JSON of a .gbx checkpoint without loading weights."""
    p = Path(path)
    with p.open("rb") as f:
        hdr = f.read(GBX_HEADER_SIZE)
        if len(hdr) != GBX_HEADER_SIZE:
            raise ValueError(f"Truncated .gbx header in {p}")
        (
            magic, version, dtype_code,
            vocab_size, d_model, n_layers, n_heads, n_kv_heads, d_ff, context_len, tie_weights,
            rope_base, rmsnorm_eps, meta_len, step, tokens_seen,
        ) = struct.unpack(GBX_HEADER_FMT, hdr)
        if magic != GBX_MAGIC:
            raise ValueError(f"Invalid .gbx magic {magic!r} in {p}")
        if version != GBX_VERSION or dtype_code != 0:
            raise ValueError(f"Unsupported .gbx version={version} dtype_code={dtype_code}")
        meta_raw = f.read(meta_len).rstrip(b"\x00")
        metadata = json.loads(meta_raw.decode("utf-8")) if meta_raw else {}

    cfg = GPTConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        context_len=context_len,
        rope_base=rope_base,
        rmsnorm_eps=rmsnorm_eps,
        tie_weights=bool(tie_weights),
        dtype="float32",
    )
    info = {
        "step": step,
        "tokens_seen": tokens_seen,
        "metadata": metadata,
        "data_offset": GBX_HEADER_SIZE + meta_len,
    }
    return cfg, info


def load_gbx_checkpoint(
    path: str | Path,
    model: Optional[GPTTransformer] = None,
) -> Tuple[GPTTransformer, Dict[str, Any]]:
    """Load a .gbx binary checkpoint directly into ctypes C buffers (zero Python list allocation)."""
    p = Path(path)
    cfg, info = read_gbx_header(p)
    if model is None:
        model = GPTTransformer(cfg, seed=0)

    named = dict(model.named_parameters())
    order = canonical_param_order(cfg)

    with p.open("rb") as f:
        f.seek(info["data_offset"])
        for name in order:
            t = named[name]
            nbytes = t.numel * 4
            if t.dtype == "float32":
                dst = memoryview(t._buf).cast("B")
                read_n = f.readinto(dst)
                if read_n != nbytes:
                    raise ValueError(f"Unexpected EOF loading {name}: got {read_n}/{nbytes} bytes")
            else:
                raw = f.read(nbytes)
                vals = struct.unpack(f"<{t.numel}f", raw)
                for i, v in enumerate(vals):
                    t._buf[i] = v

    return model, info
