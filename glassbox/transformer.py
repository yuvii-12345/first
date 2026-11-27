"""Full GPT-Style Transformer from scratch with RoPE, RMSNorm, SwiGLU, FlashAttention,
and built-in Mechanistic Interpretability hooks (Logit Lens, Head Ablation, DLA, Patching).
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple

from glassbox.nn import Embedding, Linear, Module, RMSNorm, SwiGLU
from glassbox.tensor import (
    Tensor,
    cross_entropy,
    flash_attention,
    linear,
    no_grad,
    rope,
)


@dataclass
class GPTConfig:
    vocab_size: int = 384
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 160
    context_len: int = 64
    rope_base: float = 10000.0
    rmsnorm_eps: float = 1e-6
    tie_weights: bool = True
    dtype: str = "float32"

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GPTConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class InterpretabilityTrace:
    """Hooks & captured activations for mechanistic interpretability experiments."""

    capture_attn: bool = False
    capture_residuals: bool = False
    capture_dla: bool = False

    # Interventions
    ablate_heads: Set[Tuple[int, int]] = field(default_factory=set)
    ablation_mode: str = "zero"  # "zero" or "mean"
    # Map (stage_idx, pos) -> vector[d_model]; stage -1 = after embed, stage l = after block l
    patch_residual: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)

    # Captured outputs (batch index 0)
    attn_probs: Dict[int, List[List[List[float]]]] = field(default_factory=dict)  # layer -> [H][T][T]
    residuals: List[Tuple[str, Tensor]] = field(default_factory=list)  # (stage_label, [1, T, D])
    embed_contrib: Optional[Tensor] = None  # [T, D]
    head_contribs: Dict[Tuple[int, int], Tensor] = field(default_factory=dict)  # (l, h) -> [T, D]
    mlp_contribs: Dict[int, Tensor] = field(default_factory=dict)  # l -> [T, D]
    final_pre_norm: Optional[Tensor] = None  # [1, T, D]


def _apply_residual_patch(x: Tensor, stage_idx: int, patch_map: Dict[Tuple[int, int], List[float]]) -> Tensor:
    """Overwrite residual vectors at specified (stage_idx, pos) positions for causal patching."""
    matching = [(pos, vec) for (st, pos), vec in patch_map.items() if st == stage_idx]
    if not matching:
        return x
    patched = x.clone()
    B, T_len, D = patched.shape
    for pos, vec in matching:
        if 0 <= pos < T_len:
            base = pos * D
            for d in range(D):
                patched._buf[base + d] = vec[d]
    return patched


class FlashCausalSelfAttention(Module):
    """Multi-Head Causal Self-Attention with RoPE and FlashAttention-2 tiled kernel."""

    def __init__(
        self,
        cfg: GPTConfig,
        layer_idx: int,
        rng: random.Random,
    ) -> None:
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model

        in_std = 0.02
        resid_std = 0.02 / math.sqrt(2.0 * cfg.n_layers)
        pfx = f"blocks.{layer_idx}.attn"

        self.w_q = Linear(cfg.d_model, cfg.d_model, bias=False, std=in_std, dtype=cfg.dtype, rng=rng, name=f"{pfx}.w_q")
        self.w_k = Linear(cfg.d_model, cfg.d_model, bias=False, std=in_std, dtype=cfg.dtype, rng=rng, name=f"{pfx}.w_k")
        self.w_v = Linear(cfg.d_model, cfg.d_model, bias=False, std=in_std, dtype=cfg.dtype, rng=rng, name=f"{pfx}.w_v")
        self.w_o = Linear(cfg.d_model, cfg.d_model, bias=False, std=resid_std, dtype=cfg.dtype, rng=rng, name=f"{pfx}.w_o")

    def __call__(self, x: Tensor, trace: Optional[InterpretabilityTrace] = None) -> Tensor:
        B, T_len, D = x.shape
        H = self.n_heads
        Dh = self.head_dim

        q = self.w_q(x).reshape(B, T_len, H, Dh).permute(0, 2, 1, 3)
        k = self.w_k(x).reshape(B, T_len, H, Dh).permute(0, 2, 1, 3)
        v = self.w_v(x).reshape(B, T_len, H, Dh).permute(0, 2, 1, 3)

        # Rotary Position Embeddings on Q and K
        q = rope(q, pos_offset=0, base=self.cfg.rope_base)
        k = rope(k, pos_offset=0, base=self.cfg.rope_base)

        need_probs = trace is not None and trace.capture_attn
        if need_probs:
            attn_out, probs = flash_attention(q, k, v, causal=True, return_probs=True)
            # Store batch-0 attention tensor [H, T, T]
            probs_list = probs.tolist()[0]
            trace.attn_probs[self.layer_idx] = probs_list
        else:
            attn_out = flash_attention(q, k, v, causal=True, return_probs=False)

        # Optional Head Ablation intervention (inference / mechanistic analysis)
        if trace is not None and trace.ablate_heads:
            heads_here = [h for (l, h) in trace.ablate_heads if l == self.layer_idx and 0 <= h < H]
            if heads_here:
                attn_out = attn_out.clone()
                for h in heads_here:
                    if trace.ablation_mode == "mean":
                        # Mean-ablate across sequence positions per batch
                        for b in range(B):
                            base_bh = (b * H + h) * T_len * Dh
                            means = [0.0] * Dh
                            for t in range(T_len):
                                row_off = base_bh + t * Dh
                                for d in range(Dh):
                                    means[d] += float(attn_out._buf[row_off + d])
                            inv_t = 1.0 / max(1, T_len)
                            for d in range(Dh):
                                means[d] *= inv_t
                            for t in range(T_len):
                                row_off = base_bh + t * Dh
                                for d in range(Dh):
                                    attn_out._buf[row_off + d] = means[d]
                    else:
                        # Zero-ablate head h
                        for b in range(B):
                            base_bh = (b * H + h) * T_len * Dh
                            for idx in range(T_len * Dh):
                                attn_out._buf[base_bh + idx] = 0.0

        # Optional per-head Direct Logit Attribution (exact head_h @ W_o_h^T decomposition)
        if trace is not None and trace.capture_dla and B == 1:
            # W_o.weight has shape [D, H*Dh]; slice columns [h*Dh : (h+1)*Dh] per head
            w_o_vals = self.w_o.weight._buf
            for h in range(H):
                # Extract head h output [T, Dh]
                h_vals = [0.0] * (T_len * Dh)
                base_h = h * T_len * Dh
                for i in range(T_len * Dh):
                    h_vals[i] = float(attn_out._buf[base_h + i])
                h_tensor = Tensor(h_vals, shape=(T_len, Dh), dtype=x.dtype, requires_grad=False)

                # Extract W_o slice [D, Dh]
                w_slice_vals = [0.0] * (D * Dh)
                col_off = h * Dh
                for row in range(D):
                    src_off = row * D + col_off
                    dst_off = row * Dh
                    for c in range(Dh):
                        w_slice_vals[dst_off + c] = float(w_o_vals[src_off + c])
                w_slice = Tensor(w_slice_vals, shape=(D, Dh), dtype=x.dtype, requires_grad=False)
                trace.head_contribs[(self.layer_idx, h)] = linear(h_tensor, w_slice)

        # Merge heads: [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, D]
        merged = attn_out.permute(0, 2, 1, 3).reshape(B, T_len, D)
        return self.w_o(merged)


class TransformerBlock(Module):
    """Pre-Norm Transformer Block: RMSNorm -> FlashAttn -> Residual -> RMSNorm -> SwiGLU -> Residual."""

    def __init__(self, cfg: GPTConfig, layer_idx: int, rng: random.Random) -> None:
        self.layer_idx = layer_idx
        pfx = f"blocks.{layer_idx}"
        resid_std = 0.02 / math.sqrt(2.0 * cfg.n_layers)

        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps, dtype=cfg.dtype, name=f"{pfx}.norm1")
        self.attn = FlashCausalSelfAttention(cfg, layer_idx, rng)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps, dtype=cfg.dtype, name=f"{pfx}.norm2")
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff, resid_std=resid_std, dtype=cfg.dtype, rng=rng, name=f"{pfx}.mlp")

    def __call__(self, x: Tensor, trace: Optional[InterpretabilityTrace] = None) -> Tensor:
        attn_delta = self.attn(self.norm1(x), trace=trace)
        x = x + attn_delta
        if trace is not None and trace.capture_residuals:
            trace.residuals.append((f"L{self.layer_idx}.attn", x.detach().clone()))

        mlp_delta = self.mlp(self.norm2(x))
        if trace is not None and trace.capture_dla and x.shape[0] == 1:
            trace.mlp_contribs[self.layer_idx] = mlp_delta.detach().reshape(x.shape[1], x.shape[2]).clone()

        x = x + mlp_delta
        if trace is not None and trace.patch_residual:
            x = _apply_residual_patch(x, self.layer_idx, trace.patch_residual)
        if trace is not None and trace.capture_residuals:
            trace.residuals.append((f"L{self.layer_idx}.out", x.detach().clone()))
        return x


class GPTTransformer(Module):
    """End-to-end GPT-style autoregressive Transformer built from scratch."""

    def __init__(self, cfg: GPTConfig, seed: int = 42) -> None:
        self.cfg = cfg
        rng = random.Random(seed)
        self.tok_emb = Embedding(cfg.vocab_size, cfg.d_model, std=0.02, dtype=cfg.dtype, rng=rng, name="tok_emb")
        self.blocks = [TransformerBlock(cfg, i, rng) for i in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps, dtype=cfg.dtype, name="final_norm")
        if cfg.tie_weights:
            self.lm_head_weight: Optional[Tensor] = None
        else:
            self.lm_head_weight = Tensor.randn(
                (cfg.vocab_size, cfg.d_model),
                std=0.02,
                dtype=cfg.dtype,
                requires_grad=True,
                rng=rng,
                name="lm_head.weight",
            )

    def unembed_weight(self) -> Tensor:
        return self.tok_emb.weight if self.cfg.tie_weights else self.lm_head_weight

    def project_to_logits(self, hidden: Tensor) -> Tensor:
        """Apply final RMSNorm and unembedding matrix W_U (used by forward & Logit Lens)."""
        normed = self.final_norm(hidden)
        return linear(normed, self.unembed_weight())

    def forward(
        self,
        input_ids: Sequence[Sequence[int]],
        targets: Optional[Sequence[Sequence[int]]] = None,
        trace: Optional[InterpretabilityTrace] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B = len(input_ids)
        T_len = len(input_ids[0])
        if T_len > self.cfg.context_len:
            raise ValueError(f"Sequence length {T_len} exceeds context_len {self.cfg.context_len}")

        flat_ids: List[int] = []
        for row in input_ids:
            flat_ids.extend(row)

        x = self.tok_emb(flat_ids, (B, T_len))
        if trace is not None and trace.patch_residual:
            x = _apply_residual_patch(x, -1, trace.patch_residual)
        if trace is not None and trace.capture_residuals:
            trace.residuals.append(("Embed", x.detach().clone()))
        if trace is not None and trace.capture_dla and B == 1:
            trace.embed_contrib = x.detach().reshape(T_len, self.cfg.d_model).clone()

        for block in self.blocks:
            x = block(x, trace=trace)

        if trace is not None:
            trace.final_pre_norm = x.detach().clone()

        logits = self.project_to_logits(x)

        loss = None
        if targets is not None:
            flat_targets: List[int] = []
            for row in targets:
                flat_targets.extend(row)
            loss = cross_entropy(logits, flat_targets, ignore_index=-100)
        return logits, loss

    def __call__(
        self,
        input_ids: Sequence[Sequence[int]],
        targets: Optional[Sequence[Sequence[int]]] = None,
        trace: Optional[InterpretabilityTrace] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self.forward(input_ids, targets=targets, trace=trace)

    def generate_stream(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 20,
        top_p: float = 0.92,
        seed: Optional[int] = None,
        stop_token_id: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Autoregressive sampling yielding each token + next-token probability distribution."""
        rng = random.Random(seed) if seed is not None else random.Random()
        seq = list(prompt_ids)
        if not seq:
            seq = [256]  # <BOS>

        with no_grad():
            for step_idx in range(max_new_tokens):
                ctx = seq[-self.cfg.context_len :]
                logits, _ = self.forward([ctx])
                V = self.cfg.vocab_size
                last_off = (len(ctx) - 1) * V
                raw_logits = [float(logits._buf[last_off + v]) for v in range(V)]

                # Compute full un-filtered distribution for entropy & telemetry
                temp = max(temperature, 1e-5)
                scaled = [lg / temp for lg in raw_logits]
                mx = max(scaled)
                exps = [math.exp(v - mx) for v in scaled]
                s_exp = sum(exps)
                probs = [e / s_exp for e in exps]

                entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs if p > 1e-12)

                # Top-k + Top-p nucleus filtering
                ranked = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)
                if top_k > 0:
                    ranked = ranked[:top_k]

                filtered: List[Tuple[int, float]] = []
                cum = 0.0
                for tid, p in ranked:
                    filtered.append((tid, p))
                    cum += p
                    if cum >= top_p and len(filtered) >= 1:
                        break

                if temperature <= 1e-4:
                    next_id = filtered[0][0]
                else:
                    total_p = sum(p for _, p in filtered)
                    r = rng.random() * total_p
                    acc = 0.0
                    next_id = filtered[-1][0]
                    for tid, p in filtered:
                        acc += p
                        if r <= acc:
                            next_id = tid
                            break

                top8 = [{"id": tid, "prob": round(p, 5)} for tid, p in ranked[:8]]
                chosen_prob = probs[next_id]
                seq.append(next_id)

                yield {
                    "step": step_idx,
                    "token_id": next_id,
                    "prob": round(chosen_prob, 5),
                    "entropy": round(entropy, 4),
                    "top_candidates": top8,
                }

                if stop_token_id is not None and next_id == stop_token_id:
                    break
