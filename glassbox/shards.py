"""Zero-RAM Memory-Mapped Binary Token Shards (.gbxtok) for 100M–10B Token Pretraining.

Exact binary layout shared by Python Glassbox, C++17 CPU trainer, and pure CUDA C++17 trainer:
  - 32-byte little-endian header:
      magic[4]    = "GBXT"
      version     = 1 (uint32)
      vocab_size  = e.g. 32768 (uint32)
      dtype_bytes = 2 (uint16) or 4 (uint32)
      n_tokens    = uint64
      reserved    = uint64 (0)
  - Contiguous little-endian uint16 (or uint32) token IDs:
      100M tokens = 200 MB on disk
      10B tokens  = 20 GB on disk
"""

from __future__ import annotations

import array
import mmap
import random
import struct
from pathlib import Path
from typing import List, Sequence, Tuple

GBXT_MAGIC = b"GBXT"
GBXT_VERSION = 1
GBXT_HEADER_FMT = "<4sIIIQQ"  # 32 bytes exact
GBXT_HEADER_SIZE = struct.calcsize(GBXT_HEADER_FMT)
assert GBXT_HEADER_SIZE == 32


def write_token_shard(
    path: str | Path,
    tokens: Sequence[int],
    vocab_size: int,
) -> DictInfo:
    """Write token IDs into a compact .gbxtok binary shard (uint16 if vocab <= 65535)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype_bytes = 2 if vocab_size <= 65535 else 4
    n_tokens = len(tokens)
    header = struct.pack(
        GBXT_HEADER_FMT,
        GBXT_MAGIC,
        GBXT_VERSION,
        int(vocab_size),
        dtype_bytes,
        int(n_tokens),
        0,
    )
    arr_code = "H" if dtype_bytes == 2 else "I"
    arr = array.array(arr_code, tokens)
    with out_path.open("wb") as f:
        f.write(header)
        arr.tofile(f)
    return {
        "path": str(out_path),
        "n_tokens": n_tokens,
        "vocab_size": vocab_size,
        "dtype_bytes": dtype_bytes,
        "bytes": out_path.stat().st_size,
    }


DictInfo = dict


def read_shard_header(path: str | Path) -> DictInfo:
    p = Path(path)
    with p.open("rb") as f:
        hdr = f.read(GBXT_HEADER_SIZE)
    if len(hdr) != GBXT_HEADER_SIZE:
        raise ValueError(f"Truncated .gbxtok header in {p}")
    magic, version, vocab_size, dtype_bytes, n_tokens, _ = struct.unpack(GBXT_HEADER_FMT, hdr)
    if magic != GBXT_MAGIC:
        raise ValueError(f"Invalid .gbxtok magic {magic!r} in {p}")
    if version != GBXT_VERSION or dtype_bytes not in (2, 4):
        raise ValueError(f"Unsupported .gbxtok version={version} dtype_bytes={dtype_bytes}")
    return {
        "path": str(p),
        "vocab_size": vocab_size,
        "dtype_bytes": dtype_bytes,
        "n_tokens": n_tokens,
    }


class TokenShardDataset:
    """Memory-mapped multi-shard dataset capable of streaming 100M–10B tokens with ~0 MB RAM."""

    def __init__(self, shard_paths: Sequence[str | Path], seq_len: int) -> None:
        self.seq_len = int(seq_len)
        self.shards: List[dict] = []
        self.total_tokens = 0
        self.vocab_size = 0

        for sp in sorted(Path(p) for p in shard_paths):
            info = read_shard_header(sp)
            f = sp.open("rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            if self.vocab_size == 0:
                self.vocab_size = info["vocab_size"]
            self.shards.append({
                "path": str(sp),
                "file": f,
                "mmap": mm,
                "n_tokens": info["n_tokens"],
                "dtype_bytes": info["dtype_bytes"],
                "fmt": "H" if info["dtype_bytes"] == 2 else "I",
                "offset_start": self.total_tokens,
            })
            self.total_tokens += info["n_tokens"]

        if not self.shards:
            raise ValueError("No .gbxtok shards provided")
        self._cursor_shard = 0
        self._cursor_pos = 0

    def close(self) -> None:
        for s in self.shards:
            s["mmap"].close()
            s["file"].close()

    def _read_slice(self, shard: dict, pos: int, count: int) -> List[int]:
        esz = shard["dtype_bytes"]
        byte_off = GBXT_HEADER_SIZE + pos * esz
        return list(struct.unpack_from(f"<{count}{shard['fmt']}", shard["mmap"], byte_off))

    def next_batch(self, batch_size: int) -> Tuple[List[List[int]], List[List[int]]]:
        """Sequential deterministic batching across shards (identical to CUDA DataLoader)."""
        xs: List[List[int]] = []
        ys: List[List[int]] = []
        T = self.seq_len
        for _ in range(batch_size):
            shard = self.shards[self._cursor_shard]
            if self._cursor_pos + T + 1 > shard["n_tokens"]:
                self._cursor_shard = (self._cursor_shard + 1) % len(self.shards)
                self._cursor_pos = 0
                shard = self.shards[self._cursor_shard]
            chunk = self._read_slice(shard, self._cursor_pos, T + 1)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            self._cursor_pos += T
        return xs, ys

    def sample_batch(self, batch_size: int, rng: random.Random) -> Tuple[List[List[int]], List[List[int]]]:
        """Random-access O(1) batch sampling from mmap'd shards."""
        xs: List[List[int]] = []
        ys: List[List[int]] = []
        T = self.seq_len
        for _ in range(batch_size):
            shard = rng.choice(self.shards)
            max_start = max(0, shard["n_tokens"] - T - 1)
            pos = rng.randint(0, max_start) if max_start > 0 else 0
            chunk = self._read_slice(shard, pos, T + 1)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return xs, ys
