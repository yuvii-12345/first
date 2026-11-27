"""Neural network layers built directly on Glassbox's autograd Tensor engine."""

from __future__ import annotations

import math
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from glassbox.tensor import Tensor, embedding, linear, rmsnorm


class Module:
    """Base neural network module with recursive parameter discovery."""

    def named_parameters(self, prefix: str = "") -> List[Tuple[str, Tensor]]:
        params: List[Tuple[str, Tensor]] = []
        seen = set()

        def _visit(obj, pfx: str):
            if isinstance(obj, Tensor):
                if obj.requires_grad and obj.node_id not in seen:
                    seen.add(obj.node_id)
                    if obj.name is None:
                        obj.name = pfx
                    params.append((pfx, obj))
            elif isinstance(obj, Module):
                for k, v in obj.__dict__.items():
                    if k.startswith("_"):
                        continue
                    child_pfx = f"{pfx}.{k}" if pfx else k
                    _visit(v, child_pfx)
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    child_pfx = f"{pfx}.{i}" if pfx else str(i)
                    _visit(v, child_pfx)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    child_pfx = f"{pfx}.{k}" if pfx else str(k)
                    _visit(v, child_pfx)

        _visit(self, prefix)
        return params

    def parameters(self) -> List[Tensor]:
        return [p for _, p in self.named_parameters()]

    def num_parameters(self) -> int:
        return sum(p.numel for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self) -> Dict[str, List[float]]:
        return {name: p.flat_list() for name, p in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, List[float]]) -> None:
        named = dict(self.named_parameters())
        for name, vals in state.items():
            if name not in named:
                continue
            p = named[name]
            if len(vals) != p.numel:
                raise ValueError(f"Parameter {name} size mismatch: {len(vals)} != {p.numel}")
            for i, v in enumerate(vals):
                p._buf[i] = v


class Linear(Module):
    """Affine transformation y = x @ W^T (+ b) using zero-transpose tiled GEMM."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        std: Optional[float] = None,
        dtype: str = "float32",
        rng: Optional[random.Random] = None,
        name: str = "linear",
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        init_std = std if std is not None else (1.0 / math.sqrt(in_features))
        self.weight = Tensor.randn(
            (out_features, in_features),
            std=init_std,
            dtype=dtype,
            requires_grad=True,
            rng=rng,
            name=f"{name}.weight",
        )
        self.bias: Optional[Tensor] = (
            Tensor.zeros((out_features,), dtype=dtype, requires_grad=True, name=f"{name}.bias")
            if bias
            else None
        )

    def __call__(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)


class Embedding(Module):
    """Token embedding lookup table [vocab_size, d_model]."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        std: float = 0.02,
        dtype: str = "float32",
        rng: Optional[random.Random] = None,
        name: str = "tok_emb",
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = Tensor.randn(
            (vocab_size, d_model),
            std=std,
            dtype=dtype,
            requires_grad=True,
            rng=rng,
            name=f"{name}.weight",
        )

    def __call__(self, token_ids: Sequence[int], out_shape_prefix: Tuple[int, ...]) -> Tensor:
        return embedding(self.weight, token_ids, out_shape_prefix)


class RMSNorm(Module):
    """Root Mean Square Layer Normalization (Llama/GPT-modern style)."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        dtype: str = "float32",
        name: str = "rmsnorm",
    ) -> None:
        self.d_model = d_model
        self.eps = eps
        self.weight = Tensor.ones((d_model,), dtype=dtype, requires_grad=True, name=f"{name}.weight")

    def __call__(self, x: Tensor) -> Tensor:
        return rmsnorm(x, self.weight, eps=self.eps)


class SwiGLU(Module):
    """Gated Feed-Forward Network: (SiLU(x W_gate) * (x W_up)) W_down."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        resid_std: float,
        dtype: str = "float32",
        rng: Optional[random.Random] = None,
        name: str = "mlp",
    ) -> None:
        in_std = 0.02
        self.w_gate = Linear(d_model, d_ff, bias=False, std=in_std, dtype=dtype, rng=rng, name=f"{name}.w_gate")
        self.w_up = Linear(d_model, d_ff, bias=False, std=in_std, dtype=dtype, rng=rng, name=f"{name}.w_up")
        self.w_down = Linear(d_ff, d_model, bias=False, std=resid_std, dtype=dtype, rng=rng, name=f"{name}.w_down")

    def __call__(self, x: Tensor) -> Tensor:
        return self.w_down(self.w_gate(x).silu() * self.w_up(x))
