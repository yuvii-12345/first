"""Zero-PyTorch Parallel Token Shard Builder (.gbxtok) for Glassbox-100M.

Prepares compact uint16 binary .gbxtok shards (2 bytes/token: 100M tokens = 200 MB,
10B tokens = 20 GB) readable directly by:
  - cuda/train_glassbox_100m.cu   (Pure CUDA C++17 GPU Trainer)
  - cuda/train_glassbox_cpu.cpp   (Native C++17 CPU Trainer)
  - glassbox/shards.py            (Python Glassbox mmap TokenShardDataset)

Supports:
  --dataset fineweb-edu   (streams HuggingFaceFW/fineweb-edu sample-10BT via datasets lib)
  --dataset tinystories   (streams roneneldan/TinyStories)
  --dataset local         (tokenizes any local .txt/.jsonl file or directory, zero deps!)
  --dataset builtin       (tokenizes Glassbox built-in corpora for instant offline tests)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from glassbox.bpe import BPETokenizer
from glassbox.datasets import DATASETS, get_dataset_text
from glassbox.shards import write_token_shard


def iter_texts(dataset: str, input_path: str | None) -> Iterator[str]:
    if dataset == "builtin":
        for did in ("chat", "python_code", "shakespeare", "fables", "induction"):
            if did in DATASETS:
                yield get_dataset_text(did)
        return

    if dataset == "local":
        if not input_path:
            raise ValueError("--input-path is required when --dataset local")
        p = Path(input_path)
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in files:
            if not f.is_file():
                continue
            if f.suffix == ".jsonl":
                with f.open("r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        txt = obj.get("text") or obj.get("content") or ""
                        if txt:
                            yield txt
            else:
                yield f.read_text(encoding="utf-8", errors="ignore")
        return

    # Streaming public corpora (FineWeb-Edu 10BT / TinyStories) on GPU machines with HF access
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Streaming HuggingFace datasets requires `pip install datasets`.\n"
            "For zero-dependency offline sharding, use `--dataset local --input-path <file>` or `--dataset builtin`."
        ) from exc

    if dataset == "fineweb-edu":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    elif dataset == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    else:
        ds = load_dataset(dataset, split="train", streaming=True)

    for row in ds:
        txt = row.get("text") or ""
        if txt:
            yield txt


def main() -> None:
    ap = argparse.ArgumentParser(description="Build .gbxtok binary shards for Glassbox-100M")
    ap.add_argument("--dataset", default="fineweb-edu", help="fineweb-edu | tinystories | local | builtin")
    ap.add_argument("--input-path", default=None, help="Local text/jsonl path for --dataset local")
    ap.add_argument("--target-tokens", type=int, default=100_000_000, help="Total tokens to shard (e.g. 100000000 or 10000000000)")
    ap.add_argument("--shard-size", type=int, default=10_000_000, help="Tokens per train shard (default 10M = 20MB)")
    ap.add_argument("--val-tokens", type=int, default=1_000_000, help="Tokens reserved for val_000.gbxtok")
    ap.add_argument("--vocab-size", type=int, default=32768, help="Target BPE vocabulary size (default 32768)")
    ap.add_argument("--tokenizer", default=None, help="Optional existing tokenizer JSON path")
    ap.add_argument("--out-dir", default="data/shards", help="Output directory for .gbxtok shards")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.json"

    # Optional fast tiktoken/tokenizers acceleration when targeting 32K vocab at 100M-10B scale,
    # falling back cleanly to Glassbox's pure-Python Byte-Level BPE tokenizer!
    use_bpe = True
    tok: BPETokenizer
    if args.tokenizer and Path(args.tokenizer).exists():
        tok = BPETokenizer.from_dict(json.loads(Path(args.tokenizer).read_text()))
        print(f"[Tokenizer] Loaded {args.tokenizer} (vocab={tok.vocab_size})")
    else:
        print(f"[Tokenizer] Training Glassbox Byte-Level BPE (target_vocab={args.vocab_size})...")
        sample_chunks: List[str] = []
        sample_chars = 0
        for txt in iter_texts(args.dataset, args.input_path):
            sample_chunks.append(txt)
            sample_chars += len(txt)
            if sample_chars >= 2_000_000:
                break
        tok = BPETokenizer()
        tok.train("\n".join(sample_chunks), target_vocab_size=min(args.vocab_size, 4096 if sample_chars < 500_000 else args.vocab_size))
        tok_path.write_text(json.dumps(tok.to_dict()))
        print(f"[Tokenizer] Saved {tok_path} (vocab={tok.vocab_size})")

    t0 = time.perf_counter()
    buf: List[int] = []
    val_written = False
    shard_idx = 0
    total_written = 0
    val_budget = min(args.val_tokens, max(4096, args.target_tokens // 20))

    def flush(prefix: str, idx: int, data: List[int]) -> None:
        nonlocal total_written
        p = out_dir / f"{prefix}_{idx:04d}.gbxtok"
        info = write_token_shard(p, data, vocab_size=max(tok.vocab_size, args.vocab_size))
        total_written += info["n_tokens"]
        dt = time.perf_counter() - t0
        print(f"[Shard] {p.name} | {info['n_tokens']:,} tokens ({info['bytes']/1e6:.2f} MB) | total={total_written:,} | {total_written/max(dt,1e-3):,.0f} tok/s")

    for txt in iter_texts(args.dataset, args.input_path):
        ids = tok.encode(txt, add_bos=True, add_eos=True)
        buf.extend(ids)
        if not val_written and len(buf) >= val_budget:
            flush("val", 0, buf[:val_budget])
            buf = buf[val_budget:]
            val_written = True
        while val_written and len(buf) >= args.shard_size and total_written < args.target_tokens:
            take = min(args.shard_size, args.target_tokens - total_written)
            flush("train", shard_idx, buf[:take])
            buf = buf[take:]
            shard_idx += 1
        if total_written >= args.target_tokens:
            break

    if buf and total_written < args.target_tokens:
        if not val_written:
            split = max(256, len(buf) // 10)
            flush("val", 0, buf[:split])
            if len(buf) > split:
                flush("train", shard_idx, buf[split:])
        else:
            flush("train", shard_idx, buf[: args.target_tokens - total_written])

    print(f"[Done] Wrote {total_written:,} tokens into {out_dir}")


if __name__ == "__main__":
    main()
