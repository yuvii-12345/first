"""Byte-Level BPE Tokenizer built from scratch (zero external ML/tokenizer libraries).

Features:
- Lossless UTF-8 byte-level base alphabet (IDs 0..255) — zero <UNK> tokens ever.
- GPT-2 style pre-tokenization preserving exact byte fidelity.
- Fast frequency-deduplicated training loop.
- Full merge history & interactive step-by-step merge trace for interpretability.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
SPECIAL_TOKENS = {
    "<BOS>": BOS_ID,
    "<EOS>": EOS_ID,
    "<PAD>": PAD_ID,
}
FIRST_MERGE_ID = 259

# GPT-2 inspired pre-tokenization pattern (pure Python stdlib re, 100% lossless)
_PRETOK_RE = re.compile(
    r"""'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


def _format_display(raw: bytes) -> str:
    """Human-readable token label for heatmaps and Logit Lens cells."""
    s = raw.decode("utf-8", errors="replace")
    s = s.replace("\n", "↵").replace("\t", "⇥").replace("\r", "␍")
    if s == " ":
        return "␣"
    if s.startswith(" "):
        return "·" + s[1:]
    return s


class BPETokenizer:
    """Byte-level Byte-Pair Encoding tokenizer with full merge inspection."""

    def __init__(self) -> None:
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_id_to_str: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.merges: List[Tuple[int, int, int]] = []  # (a, b, new_id)
        self.merge_ranks: Dict[Tuple[int, int], int] = {}
        self.merge_to_id: Dict[Tuple[int, int], int] = {}
        self.merge_counts: List[int] = []

    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE_ID + len(self.merges)

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID

    def _pretokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return _PRETOK_RE.findall(text)

    def train(self, text: str, target_vocab_size: int = 384, min_frequency: int = 2) -> Dict[str, Any]:
        """Train byte-level BPE merges up to target_vocab_size."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.merge_ranks = {}
        self.merge_to_id = {}
        self.merge_counts = []

        num_merges = max(0, target_vocab_size - FIRST_MERGE_ID)
        chunks = self._pretokenize(text)
        word_freqs: Counter[Tuple[int, ...]] = Counter()
        for ch in chunks:
            b = tuple(ch.encode("utf-8"))
            if b:
                word_freqs[b] += 1

        # Iterative pair merging on deduplicated word table
        words: List[Tuple[List[int], int]] = [(list(w), freq) for w, freq in word_freqs.items()]

        for step in range(num_merges):
            pair_counts: Counter[Tuple[int, int]] = Counter()
            for seq, freq in words:
                if len(seq) < 2:
                    continue
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += freq

            if not pair_counts:
                break

            # Deterministic tie-breaking by (-count, pair)
            best_pair, best_count = min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if best_count < min_frequency:
                break

            a, b = best_pair
            new_id = FIRST_MERGE_ID + step
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            self.merges.append((a, b, new_id))
            self.merge_ranks[best_pair] = step
            self.merge_to_id[best_pair] = new_id
            self.merge_counts.append(best_count)

            # Apply merge to deduplicated words
            new_words: List[Tuple[List[int], int]] = []
            for seq, freq in words:
                if len(seq) < 2:
                    new_words.append((seq, freq))
                    continue
                merged: List[int] = []
                i = 0
                n = len(seq)
                while i < n:
                    if i + 1 < n and seq[i] == a and seq[i + 1] == b:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(seq[i])
                        i += 1
                new_words.append((merged, freq))
            words = new_words

        total_bytes = len(text.encode("utf-8"))
        total_tokens = sum(len(seq) * freq for seq, freq in words)
        return {
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "corpus_bytes": total_bytes,
            "corpus_tokens": total_tokens,
            "compression_ratio": round(total_bytes / max(1, total_tokens), 3),
        }

    def _encode_chunk(self, raw_bytes: bytes, trace: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        ids = list(raw_bytes)
        if len(ids) < 2 or not self.merge_ranks:
            return ids

        while len(ids) >= 2:
            best_rank = None
            best_pair = None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            a, b = best_pair
            new_id = self.merge_to_id[best_pair]
            new_ids: List[int] = []
            i = 0
            n = len(ids)
            while i < n:
                if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            if trace is not None:
                trace.append(
                    {
                        "rank": best_rank,
                        "pair": [self.token_display(a), self.token_display(b)],
                        "merged": self.token_display(new_id),
                        "new_id": new_id,
                        "tokens_after": [self.token_display(t) for t in new_ids],
                    }
                )
            ids = new_ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        out: List[int] = []
        if add_bos:
            out.append(BOS_ID)
        for ch in self._pretokenize(text):
            out.extend(self._encode_chunk(ch.encode("utf-8")))
        if add_eos:
            out.append(EOS_ID)
        return out

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        buf = bytearray()
        for tid in ids:
            if tid in self.special_id_to_str:
                if not skip_special:
                    buf.extend(self.special_id_to_str[tid].encode("utf-8"))
            elif tid in self.vocab:
                buf.extend(self.vocab[tid])
        return buf.decode("utf-8", errors="replace")

    def token_text(self, tid: int) -> str:
        if tid in self.special_id_to_str:
            return self.special_id_to_str[tid]
        return self.vocab.get(tid, b"").decode("utf-8", errors="replace")

    def token_display(self, tid: int) -> str:
        if tid in self.special_id_to_str:
            return self.special_id_to_str[tid]
        return _format_display(self.vocab.get(tid, b""))

    def inspect_encode(self, text: str) -> Dict[str, Any]:
        """Rich tokenization inspection including byte lengths and step-by-step merge trace."""
        tokens = []
        chunk_traces = []
        raw_byte_len = len(text.encode("utf-8"))
        for idx, ch in enumerate(self._pretokenize(text)):
            raw = ch.encode("utf-8")
            trace: List[Dict[str, Any]] = []
            ids = self._encode_chunk(raw, trace=trace if idx < 12 else None)
            if idx < 12 and trace:
                chunk_traces.append(
                    {
                        "chunk": _format_display(raw),
                        "initial_bytes": [_format_display(bytes([b])) for b in raw],
                        "steps": trace,
                    }
                )
            for tid in ids:
                raw_tok = self.vocab.get(tid, b"")
                tokens.append(
                    {
                        "id": tid,
                        "text": self.token_text(tid),
                        "display": self.token_display(tid),
                        "bytes_hex": raw_tok.hex(" "),
                        "byte_len": len(raw_tok),
                        "is_merged": tid >= FIRST_MERGE_ID,
                    }
                )
        return {
            "text": text,
            "num_bytes": raw_byte_len,
            "num_tokens": len(tokens),
            "compression_ratio": round(raw_byte_len / max(1, len(tokens)), 3),
            "tokens": tokens,
            "chunk_traces": chunk_traces,
        }

    def get_merge_table(self, limit: int = 80) -> List[Dict[str, Any]]:
        rows = []
        for rank, (a, b, new_id) in enumerate(self.merges[:limit]):
            rows.append(
                {
                    "rank": rank,
                    "new_id": new_id,
                    "left_id": a,
                    "right_id": b,
                    "left": self.token_display(a),
                    "right": self.token_display(b),
                    "merged": self.token_display(new_id),
                    "count": self.merge_counts[rank] if rank < len(self.merge_counts) else 0,
                }
            )
        return rows

    def to_dict(self) -> Dict[str, Any]:
        return {
            "merges": self.merges,
            "merge_counts": self.merge_counts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BPETokenizer":
        tok = cls()
        tok.merges = [tuple(m) for m in data.get("merges", [])]
        tok.merge_counts = list(data.get("merge_counts", []))
        for rank, (a, b, new_id) in enumerate(tok.merges):
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
            tok.merge_ranks[(a, b)] = rank
            tok.merge_to_id[(a, b)] = new_id
        return tok

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
