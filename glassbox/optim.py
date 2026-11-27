"""AdamW optimizer (fused C++ kernel), cosine warmup scheduler, and gradient telemetry."""

from __future__ import annotations

import ctypes
import math
from typing import Dict, List, Tuple

from glassbox.backend import LIB
from glassbox.tensor import Tensor, _ctype_for


def clip_grad_norm(named_params: List[Tuple[str, Tensor]], max_norm: float = 1.0) -> Tuple[float, bool]:
    """Clip gradients globally by L2 norm. Returns (total_grad_norm, was_clipped)."""
    total_sq = 0.0
    for _, p in named_params:
        if p.grad is not None:
            total_sq += p.grad.l2_norm_sq()
    total_norm = math.sqrt(max(0.0, total_sq))
    if max_norm > 0 and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for _, p in named_params:
            if p.grad is not None:
                LIB.gb_scale_inplace(p.grad._buf, p.grad.numel, scale, p.grad._dtype_code)
        return total_norm, True
    return total_norm, False


class CosineWarmupScheduler:
    """Linear warmup followed by cosine decay down to min_lr."""

    def __init__(self, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_lr * float(step + 1) / float(self.warmup_steps)
        if step >= self.total_steps:
            return self.min_lr
        progress = float(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


class AdamW:
    """Decoupled Weight Decay AdamW backed by Glassbox's fused C++ kernel."""

    def __init__(
        self,
        named_params: List[Tuple[str, Tensor]],
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.05,
    ) -> None:
        self.named_params = list(named_params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_num = 0

        # Allocate first & second moment buffers per parameter
        self._m: Dict[int, object] = {}
        self._v: Dict[int, object] = {}
        for _, p in self.named_params:
            ctype = _ctype_for(p.dtype)
            self._m[p.node_id] = (ctype * max(1, p.numel))()
            self._v[p.node_id] = (ctype * max(1, p.numel))()

    def zero_grad(self) -> None:
        for _, p in self.named_params:
            p.zero_grad()

    def step(self, lr: float | None = None) -> Dict[str, Dict[str, float]]:
        """Apply fused AdamW update and return per-parameter grad/update telemetry."""
        self.step_num += 1
        use_lr = self.lr if lr is None else lr
        telemetry: Dict[str, Dict[str, float]] = {}

        upd_sq_c = ctypes.c_double(0.0)
        prm_sq_c = ctypes.c_double(0.0)

        for name, p in self.named_params:
            if p.grad is None:
                continue
            grad_norm = p.grad.l2_norm()
            # Apply decoupled weight decay only to >=2D weights (not RMSNorm scales or biases)
            wd = self.weight_decay if p.ndim >= 2 else 0.0
            LIB.gb_adamw_step(
                p._buf,
                p.grad._buf,
                self._m[p.node_id],
                self._v[p.node_id],
                p.numel,
                use_lr,
                self.beta1,
                self.beta2,
                self.eps,
                wd,
                self.step_num,
                ctypes.byref(upd_sq_c),
                ctypes.byref(prm_sq_c),
                p._dtype_code,
            )
            upd_norm = math.sqrt(max(0.0, upd_sq_c.value))
            prm_norm = math.sqrt(max(0.0, prm_sq_c.value))
            ratio = upd_norm / max(prm_norm, 1e-12)
            telemetry[name] = {
                "grad_norm": grad_norm,
                "param_norm": prm_norm,
                "update_norm": upd_norm,
                "update_ratio": ratio,
                "log10_update_ratio": math.log10(max(ratio, 1e-12)),
            }
        return telemetry
