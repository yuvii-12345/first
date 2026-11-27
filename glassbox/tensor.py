"""Pure-Python reverse-mode autograd engine backed by Glassbox C++17 kernels.

Zero NumPy, zero PyTorch, zero TensorFlow — just raw ctypes buffers,
dynamic computation graphs, and exact analytical backward passes.
"""

from __future__ import annotations

import ctypes
import math
import random
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from glassbox.backend import DTYPE_F32, DTYPE_F64, LIB

# Thread-local autograd toggle (safe when training & serving concurrently)
_TLS = threading.local()
_NODE_COUNTER_LOCK = threading.Lock()
_NODE_COUNTER = 0


def _next_node_id() -> int:
    global _NODE_COUNTER
    with _NODE_COUNTER_LOCK:
        _NODE_COUNTER += 1
        return _NODE_COUNTER


def is_grad_enabled() -> bool:
    return getattr(_TLS, "grad_enabled", True)


@contextmanager
def no_grad():
    prev = is_grad_enabled()
    _TLS.grad_enabled = False
    try:
        yield
    finally:
        _TLS.grad_enabled = prev


def _dtype_code(dtype: str) -> int:
    if dtype == "float32":
        return DTYPE_F32
    if dtype == "float64":
        return DTYPE_F64
    raise ValueError(f"Unsupported dtype: {dtype!r}")


def _ctype_for(dtype: str):
    return ctypes.c_float if dtype == "float32" else ctypes.c_double


def _prod(shape: Sequence[int]) -> int:
    p = 1
    for s in shape:
        p *= s
    return p


def _contiguous_strides(shape: Sequence[int]) -> List[int]:
    strides = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return strides


def _broadcast_shapes(s1: Tuple[int, ...], s2: Tuple[int, ...]) -> Tuple[int, ...]:
    n1, n2 = len(s1), len(s2)
    n = max(n1, n2)
    out = []
    for i in range(n):
        d1 = s1[i - (n - n1)] if i >= n - n1 else 1
        d2 = s2[i - (n - n2)] if i >= n - n2 else 1
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise ValueError(f"Incompatible shapes for broadcast: {s1} and {s2}")
        out.append(max(d1, d2))
    return tuple(out)


def _broadcast_strides(orig_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> List[int]:
    orig_strides = _contiguous_strides(orig_shape)
    pad = len(target_shape) - len(orig_shape)
    out = []
    for i in range(len(target_shape)):
        if i < pad:
            out.append(0)
        else:
            oi = i - pad
            out.append(0 if orig_shape[oi] == 1 else orig_strides[oi])
    return out


def _c_int_arr(vals: Sequence[int]):
    return (ctypes.c_int * len(vals))(*vals)


def _flatten_nested(data: Any) -> Tuple[List[float], Tuple[int, ...]]:
    if isinstance(data, (int, float)):
        return [float(data)], ()
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Cannot convert {type(data)} to Tensor")
    if len(data) == 0:
        return [], (0,)
    first_flat, first_sub = _flatten_nested(data[0])
    flat = list(first_flat)
    for item in data[1:]:
        sub_flat, sub_shape = _flatten_nested(item)
        if sub_shape != first_sub:
            raise ValueError("Ragged nested sequence")
        flat.extend(sub_flat)
    return flat, (len(data),) + first_sub


def _unflatten(flat: Sequence[float], shape: Tuple[int, ...], offset: int = 0) -> Any:
    if len(shape) == 0:
        return float(flat[offset])
    if len(shape) == 1:
        return [float(flat[offset + i]) for i in range(shape[0])]
    step = _prod(shape[1:])
    return [_unflatten(flat, shape[1:], offset + i * step) for i in range(shape[0])]


class Tensor:
    """N-dimensional contiguous tensor with dynamic reverse-mode autograd."""

    __slots__ = (
        "_buf",
        "shape",
        "numel",
        "dtype",
        "_dtype_code",
        "requires_grad",
        "grad",
        "_parents",
        "_backward",
        "_op",
        "name",
        "node_id",
    )

    def __init__(
        self,
        data: Any = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: str = "float32",
        requires_grad: bool = False,
        _buf: Any = None,
        _op: str = "Leaf",
        _parents: Tuple["Tensor", ...] = (),
        _backward: Optional[Callable[[], None]] = None,
        name: Optional[str] = None,
    ):
        self.dtype = dtype
        self._dtype_code = _dtype_code(dtype)
        ctype = _ctype_for(dtype)

        if _buf is not None:
            assert shape is not None
            self.shape = tuple(shape)
            self.numel = _prod(self.shape)
            self._buf = _buf
        else:
            if isinstance(data, Tensor):
                self.shape = data.shape
                self.numel = data.numel
                buf = (ctype * max(1, self.numel))()
                if data.dtype == dtype:
                    ctypes.memmove(buf, data._buf, self.numel * ctypes.sizeof(ctype))
                else:
                    for i in range(self.numel):
                        buf[i] = data._buf[i]
                self._buf = buf
            else:
                flat, inferred_shape = _flatten_nested(data)
                self.shape = tuple(shape) if shape is not None else inferred_shape
                self.numel = _prod(self.shape)
                if len(flat) != self.numel:
                    raise ValueError(f"Data length {len(flat)} does not match shape {self.shape}")
                buf = (ctype * max(1, self.numel))(*flat)
                self._buf = buf

        self.requires_grad = bool(requires_grad)
        self.grad: Optional[Tensor] = None
        self._parents: Tuple[Tensor, ...] = _parents if (self.requires_grad and is_grad_enabled()) else ()
        self._backward: Optional[Callable[[], None]] = _backward if (self.requires_grad and is_grad_enabled()) else None
        self._op = _op
        self.name = name
        self.node_id = _next_node_id()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    # ---- Constructors ----

    @classmethod
    def zeros(cls, shape: Sequence[int], dtype: str = "float32", requires_grad: bool = False, name: Optional[str] = None) -> "Tensor":
        sh = tuple(shape)
        n = _prod(sh)
        ctype = _ctype_for(dtype)
        buf = (ctype * max(1, n))()
        return cls(_buf=buf, shape=sh, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def ones(cls, shape: Sequence[int], dtype: str = "float32", requires_grad: bool = False, name: Optional[str] = None) -> "Tensor":
        return cls.full(shape, 1.0, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def full(cls, shape: Sequence[int], value: float, dtype: str = "float32", requires_grad: bool = False, name: Optional[str] = None) -> "Tensor":
        sh = tuple(shape)
        n = _prod(sh)
        ctype = _ctype_for(dtype)
        buf = (ctype * max(1, n))(*([value] * n))
        return cls(_buf=buf, shape=sh, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def randn(
        cls,
        shape: Sequence[int],
        std: float = 1.0,
        mean: float = 0.0,
        dtype: str = "float32",
        requires_grad: bool = False,
        rng: Optional[random.Random] = None,
        name: Optional[str] = None,
    ) -> "Tensor":
        sh = tuple(shape)
        n = _prod(sh)
        r = rng if rng is not None else random
        ctype = _ctype_for(dtype)
        if n >= 65536 and mean == 0.0:
            buf = (ctype * max(1, n))()
            seed_u64 = r.getrandbits(64)
            LIB.gb_fill_randn(buf, n, float(std), seed_u64, _dtype_code(dtype))
            return cls(_buf=buf, shape=sh, dtype=dtype, requires_grad=requires_grad, name=name)
        vals = [r.gauss(mean, std) for _ in range(n)]
        buf = (ctype * max(1, n))(*vals)
        return cls(_buf=buf, shape=sh, dtype=dtype, requires_grad=requires_grad, name=name)

    # ---- Utilities ----

    def item(self) -> float:
        if self.numel != 1:
            raise ValueError(f"item() requires 1-element tensor, got shape {self.shape}")
        return float(self._buf[0])

    def tolist(self) -> Any:
        return _unflatten(self._buf, self.shape, 0)

    def flat_list(self) -> List[float]:
        return [float(self._buf[i]) for i in range(self.numel)]

    def detach(self) -> "Tensor":
        return Tensor(_buf=self._buf, shape=self.shape, dtype=self.dtype, requires_grad=False, name=self.name)

    def clone(self) -> "Tensor":
        ctype = _ctype_for(self.dtype)
        buf = (ctype * max(1, self.numel))()
        ctypes.memmove(buf, self._buf, self.numel * ctypes.sizeof(ctype))
        return Tensor(_buf=buf, shape=self.shape, dtype=self.dtype, requires_grad=False, name=self.name)

    def astype(self, dtype: str) -> "Tensor":
        if dtype == self.dtype:
            return self.clone()
        ctype = _ctype_for(dtype)
        buf = (ctype * max(1, self.numel))(*(float(self._buf[i]) for i in range(self.numel)))
        return Tensor(_buf=buf, shape=self.shape, dtype=dtype, requires_grad=self.requires_grad, name=self.name)

    def l2_norm_sq(self) -> float:
        return float(LIB.gb_l2_norm_sq(self._buf, self.numel, self._dtype_code))

    def l2_norm(self) -> float:
        return math.sqrt(max(0.0, self.l2_norm_sq()))

    def zero_grad(self) -> None:
        self.grad = None

    def _accum_grad(self, grad_buf: Any, shape: Optional[Tuple[int, ...]] = None) -> None:
        if not self.requires_grad:
            return
        if self.grad is None:
            ctype = _ctype_for(self.dtype)
            buf = (ctype * max(1, self.numel))()
            ctypes.memmove(buf, grad_buf, self.numel * ctypes.sizeof(ctype))
            self.grad = Tensor(_buf=buf, shape=self.shape, dtype=self.dtype, requires_grad=False)
        else:
            LIB.gb_add_inplace(self.grad._buf, grad_buf, self.numel, 1.0, self._dtype_code)

    # ---- Autograd Engine ----

    def backward(self, grad: Optional["Tensor"] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor with requires_grad=False")

        if grad is None:
            if self.numel != 1:
                raise RuntimeError("grad must be specified for non-scalar tensor backward()")
            grad = Tensor.ones(self.shape, dtype=self.dtype)

        # Iterative topological sort (avoids Python recursion limits)
        topo: List[Tensor] = []
        visited = set()
        stack: List[Tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if node.node_id in visited:
                continue
            visited.add(node.node_id)
            stack.append((node, True))
            for p in node._parents:
                if p.node_id not in visited:
                    stack.append((p, False))

        self._accum_grad(grad._buf)
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward()

    def export_graph(self, max_nodes: int = 120) -> Dict[str, Any]:
        """Export the dynamic autograd DAG with tensor shapes and gradient norms."""
        nodes = []
        edges = []
        visited = set()
        queue: List[Tensor] = [self]
        while queue and len(visited) < max_nodes:
            node = queue.pop(0)
            if node.node_id in visited:
                continue
            visited.add(node.node_id)
            gn = node.grad.l2_norm() if node.grad is not None else None
            dn = node.l2_norm()
            nodes.append(
                {
                    "id": node.node_id,
                    "op": node._op,
                    "name": node.name or (node._op if node._parents else "Tensor"),
                    "shape": list(node.shape),
                    "numel": node.numel,
                    "requires_grad": node.requires_grad,
                    "is_leaf": len(node._parents) == 0,
                    "data_norm": round(dn, 5) if math.isfinite(dn) else None,
                    "grad_norm": round(gn, 6) if (gn is not None and math.isfinite(gn)) else None,
                }
            )
            for p in node._parents:
                edges.append({"source": p.node_id, "target": node.node_id})
                if p.node_id not in visited:
                    queue.append(p)
        valid_ids = {n["id"] for n in nodes}
        edges = [e for e in edges if e["source"] in valid_ids and e["target"] in valid_ids]
        return {"root_id": self.node_id, "nodes": nodes, "edges": edges}

    # ---- Broadcasted Binary Ops ----

    def _ensure_tensor(self, other: Union["Tensor", float, int]) -> "Tensor":
        if isinstance(other, Tensor):
            if other.dtype != self.dtype:
                raise TypeError(f"Dtype mismatch: {self.dtype} vs {other.dtype}")
            return other
        return Tensor.full((), float(other), dtype=self.dtype, requires_grad=False)

    def _binary_op(self, other: Union["Tensor", float, int], op_code: int, op_name: str) -> "Tensor":
        b = self._ensure_tensor(other)
        out_shape = _broadcast_shapes(self.shape, b.shape)
        ndim = len(out_shape)
        total = _prod(out_shape)
        sa = _broadcast_strides(self.shape, out_shape)
        sb = _broadcast_strides(b.shape, out_shape)

        ctype = _ctype_for(self.dtype)
        out_buf = (ctype * max(1, total))()
        if ndim == 0:
            va, vb = self._buf[0], b._buf[0]
            if op_code == 0:
                out_buf[0] = va + vb
            elif op_code == 1:
                out_buf[0] = va - vb
            elif op_code == 2:
                out_buf[0] = va * vb
            else:
                out_buf[0] = va / vb
        else:
            LIB.gb_binary_bcast(
                self._buf,
                b._buf,
                out_buf,
                _c_int_arr(out_shape),
                _c_int_arr(sa),
                _c_int_arr(sb),
                ndim,
                total,
                op_code,
                self._dtype_code,
            )

        req = is_grad_enabled() and (self.requires_grad or b.requires_grad)
        out = Tensor(
            _buf=out_buf,
            shape=out_shape,
            dtype=self.dtype,
            requires_grad=req,
            _op=op_name,
            _parents=(self, b) if req else (),
        )

        if req:
            def _reduce_and_accum(target: Tensor, raw_grad_buf: Any, target_strides: List[int]):
                if not target.requires_grad:
                    return
                if target.shape == out_shape:
                    target._accum_grad(raw_grad_buf)
                elif len(target.shape) == 0:
                    s = sum(float(raw_grad_buf[i]) for i in range(total))
                    rb = (ctype * 1)(s)
                    target._accum_grad(rb)
                else:
                    rb = (ctype * max(1, target.numel))()
                    LIB.gb_reduce_bcast(
                        raw_grad_buf,
                        rb,
                        _c_int_arr(out_shape),
                        _c_int_arr(target_strides),
                        ndim,
                        total,
                        target.numel,
                        self._dtype_code,
                    )
                    target._accum_grad(rb)

            def _backward():
                g = out.grad._buf
                if op_code == 0:  # Add
                    _reduce_and_accum(self, g, sa)
                    _reduce_and_accum(b, g, sb)
                elif op_code == 1:  # Sub
                    _reduce_and_accum(self, g, sa)
                    if b.requires_grad:
                        neg_g = (ctype * max(1, total))()
                        LIB.gb_unary_fwd(g, neg_g, total, 5, self._dtype_code)
                        _reduce_and_accum(b, neg_g, sb)
                elif op_code == 2:  # Mul
                    if self.requires_grad:
                        ga = (ctype * max(1, total))()
                        if ndim == 0:
                            ga[0] = g[0] * b._buf[0]
                        else:
                            LIB.gb_binary_bcast(
                                g, b._buf, ga,
                                _c_int_arr(out_shape),
                                _c_int_arr(_contiguous_strides(out_shape)),
                                _c_int_arr(sb),
                                ndim, total, 2, self._dtype_code,
                            )
                        _reduce_and_accum(self, ga, sa)
                    if b.requires_grad:
                        gb = (ctype * max(1, total))()
                        if ndim == 0:
                            gb[0] = g[0] * self._buf[0]
                        else:
                            LIB.gb_binary_bcast(
                                g, self._buf, gb,
                                _c_int_arr(out_shape),
                                _c_int_arr(_contiguous_strides(out_shape)),
                                _c_int_arr(sa),
                                ndim, total, 2, self._dtype_code,
                            )
                        _reduce_and_accum(b, gb, sb)
                else:  # Div: a / b
                    if self.requires_grad:
                        ga = (ctype * max(1, total))()
                        if ndim == 0:
                            ga[0] = g[0] / b._buf[0]
                        else:
                            LIB.gb_binary_bcast(
                                g, b._buf, ga,
                                _c_int_arr(out_shape),
                                _c_int_arr(_contiguous_strides(out_shape)),
                                _c_int_arr(sb),
                                ndim, total, 3, self._dtype_code,
                            )
                        _reduce_and_accum(self, ga, sa)
                    if b.requires_grad:
                        # -g * a / (b * b)
                        num = (ctype * max(1, total))()
                        den = (ctype * max(1, total))()
                        gb = (ctype * max(1, total))()
                        if ndim == 0:
                            gb[0] = -g[0] * self._buf[0] / (b._buf[0] * b._buf[0])
                        else:
                            out_st = _c_int_arr(_contiguous_strides(out_shape))
                            sh_arr = _c_int_arr(out_shape)
                            sb_arr = _c_int_arr(sb)
                            LIB.gb_binary_bcast(g, self._buf, num, sh_arr, out_st, _c_int_arr(sa), ndim, total, 2, self._dtype_code)
                            LIB.gb_binary_bcast(b._buf, b._buf, den, sh_arr, sb_arr, sb_arr, ndim, total, 2, self._dtype_code)
                            LIB.gb_binary_bcast(num, den, gb, sh_arr, out_st, out_st, ndim, total, 3, self._dtype_code)
                            LIB.gb_scale_inplace(gb, total, -1.0, self._dtype_code)
                        _reduce_and_accum(b, gb, sb)

            out._backward = _backward
        return out

    def __add__(self, other): return self._binary_op(other, 0, "Add")
    def __radd__(self, other): return self._ensure_tensor(other)._binary_op(self, 0, "Add")
    def __sub__(self, other): return self._binary_op(other, 1, "Sub")
    def __rsub__(self, other): return self._ensure_tensor(other)._binary_op(self, 1, "Sub")
    def __mul__(self, other): return self._binary_op(other, 2, "Mul")
    def __rmul__(self, other): return self._ensure_tensor(other)._binary_op(self, 2, "Mul")
    def __truediv__(self, other): return self._binary_op(other, 3, "Div")
    def __rtruediv__(self, other): return self._ensure_tensor(other)._binary_op(self, 3, "Div")

    # ---- Unary Ops ----

    def _unary_op(self, op_code: int, op_name: str) -> "Tensor":
        ctype = _ctype_for(self.dtype)
        out_buf = (ctype * max(1, self.numel))()
        LIB.gb_unary_fwd(self._buf, out_buf, self.numel, op_code, self._dtype_code)
        req = is_grad_enabled() and self.requires_grad
        out = Tensor(
            _buf=out_buf,
            shape=self.shape,
            dtype=self.dtype,
            requires_grad=req,
            _op=op_name,
            _parents=(self,) if req else (),
        )
        if req:
            def _backward():
                dx = (ctype * max(1, self.numel))()
                LIB.gb_unary_bwd(self._buf, out.grad._buf, dx, self.numel, op_code, self._dtype_code)
                self._accum_grad(dx)
            out._backward = _backward
        return out

    def silu(self) -> "Tensor": return self._unary_op(0, "SiLU")
    def relu(self) -> "Tensor": return self._unary_op(1, "ReLU")
    def tanh(self) -> "Tensor": return self._unary_op(2, "Tanh")
    def exp(self) -> "Tensor": return self._unary_op(3, "Exp")
    def log(self) -> "Tensor": return self._unary_op(4, "Log")
    def __neg__(self) -> "Tensor": return self._unary_op(5, "Neg")
    def gelu(self) -> "Tensor": return self._unary_op(6, "GELU")

    # ---- Shape Ops ----

    def reshape(self, *shape: Any) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = list(shape[0])
        else:
            new_shape = list(shape)
        neg_idx = -1
        known = 1
        for i, s in enumerate(new_shape):
            if s == -1:
                if neg_idx != -1:
                    raise ValueError("Only one -1 allowed in reshape")
                neg_idx = i
            else:
                known *= s
        if neg_idx != -1:
            if self.numel % known != 0:
                raise ValueError(f"Cannot reshape {self.shape} to {new_shape}")
            new_shape[neg_idx] = self.numel // known
        sh = tuple(new_shape)
        if _prod(sh) != self.numel:
            raise ValueError(f"Cannot reshape {self.shape} to {sh}")

        req = is_grad_enabled() and self.requires_grad
        out = Tensor(
            _buf=self._buf,
            shape=sh,
            dtype=self.dtype,
            requires_grad=req,
            _op="Reshape",
            _parents=(self,) if req else (),
        )
        if req:
            def _backward():
                self._accum_grad(out.grad._buf)
            out._backward = _backward
        return out

    def permute(self, *dims: int) -> "Tensor":
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            perm = tuple(dims[0])
        else:
            perm = tuple(dims)
        if len(perm) != self.ndim:
            raise ValueError(f"permute dims {perm} must match ndim {self.ndim}")

        out_shape = tuple(self.shape[d] for d in perm)
        in_strides = _contiguous_strides(self.shape)
        ctype = _ctype_for(self.dtype)
        out_buf = (ctype * max(1, self.numel))()
        LIB.gb_permute(
            self._buf,
            out_buf,
            _c_int_arr(in_strides),
            _c_int_arr(out_shape),
            _c_int_arr(perm),
            self.ndim,
            self.numel,
            self._dtype_code,
        )
        req = is_grad_enabled() and self.requires_grad
        out = Tensor(
            _buf=out_buf,
            shape=out_shape,
            dtype=self.dtype,
            requires_grad=req,
            _op="Permute",
            _parents=(self,) if req else (),
        )
        if req:
            inv_perm = [0] * self.ndim
            for i, d in enumerate(perm):
                inv_perm[d] = i
            out_strides = _contiguous_strides(out_shape)

            def _backward():
                dx = (ctype * max(1, self.numel))()
                LIB.gb_permute(
                    out.grad._buf,
                    dx,
                    _c_int_arr(out_strides),
                    _c_int_arr(self.shape),
                    _c_int_arr(inv_perm),
                    self.ndim,
                    self.numel,
                    self._dtype_code,
                )
                self._accum_grad(dx)

            out._backward = _backward
        return out

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        dims = list(range(self.ndim))
        d0 = dim0 if dim0 >= 0 else self.ndim + dim0
        d1 = dim1 if dim1 >= 0 else self.ndim + dim1
        dims[d0], dims[d1] = dims[d1], dims[d0]
        return self.permute(*dims)

    # ---- Reductions & Softmax ----

    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        ctype = _ctype_for(self.dtype)
        if dim is None:
            total_sum = sum(float(self._buf[i]) for i in range(self.numel))
            out_shape = tuple(1 for _ in self.shape) if keepdim else ()
            out_buf = (ctype * 1)(total_sum)
            req = is_grad_enabled() and self.requires_grad
            out = Tensor(
                _buf=out_buf,
                shape=out_shape,
                dtype=self.dtype,
                requires_grad=req,
                _op="Sum",
                _parents=(self,) if req else (),
            )
            if req:
                def _backward():
                    val = float(out.grad._buf[0])
                    gb = (ctype * max(1, self.numel))(*([val] * self.numel))
                    self._accum_grad(gb)
                out._backward = _backward
            return out

        d = dim if dim >= 0 else self.ndim + dim
        out_shape_keep = tuple(1 if i == d else s for i, s in enumerate(self.shape))
        out_shape = out_shape_keep if keepdim else tuple(s for i, s in enumerate(self.shape) if i != d)
        out_numel = _prod(out_shape)
        out_strides_keep = _broadcast_strides(out_shape_keep, self.shape)
        out_buf = (ctype * max(1, out_numel))()
        LIB.gb_reduce_bcast(
            self._buf,
            out_buf,
            _c_int_arr(self.shape),
            _c_int_arr(out_strides_keep),
            self.ndim,
            self.numel,
            out_numel,
            self._dtype_code,
        )
        req = is_grad_enabled() and self.requires_grad
        out = Tensor(
            _buf=out_buf,
            shape=out_shape,
            dtype=self.dtype,
            requires_grad=req,
            _op="Sum",
            _parents=(self,) if req else (),
        )
        if req:
            def _backward():
                zero_buf = (ctype * 1)(0.0)
                gb = (ctype * max(1, self.numel))()
                LIB.gb_binary_bcast(
                    out.grad._buf,
                    zero_buf,
                    gb,
                    _c_int_arr(self.shape),
                    _c_int_arr(out_strides_keep),
                    _c_int_arr([0] * self.ndim),
                    self.ndim,
                    self.numel,
                    0,
                    self._dtype_code,
                )
                self._accum_grad(gb)
            out._backward = _backward
        return out

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        s = self.sum(dim=dim, keepdim=keepdim)
        denom = float(self.numel if dim is None else self.shape[dim if dim >= 0 else self.ndim + dim])
        return s * (1.0 / denom)

    def softmax(self, dim: int = -1) -> "Tensor":
        d = dim if dim >= 0 else self.ndim + dim
        if d != self.ndim - 1:
            raise NotImplementedError("softmax currently expects dim=-1")
        cols = self.shape[-1]
        rows = self.numel // cols
        ctype = _ctype_for(self.dtype)
        out_buf = (ctype * max(1, self.numel))()
        LIB.gb_softmax_fwd(self._buf, out_buf, rows, cols, self._dtype_code)
        req = is_grad_enabled() and self.requires_grad
        out = Tensor(
            _buf=out_buf,
            shape=self.shape,
            dtype=self.dtype,
            requires_grad=req,
            _op="Softmax",
            _parents=(self,) if req else (),
        )
        if req:
            def _backward():
                dx = (ctype * max(1, self.numel))()
                LIB.gb_softmax_bwd(out._buf, out.grad._buf, dx, rows, cols, self._dtype_code)
                self._accum_grad(dx)
            out._backward = _backward
        return out

    # ---- Matrix Multiplication (Cache-Tiled GEMM) ----

    def matmul(self, other: "Tensor") -> "Tensor":
        return matmul(self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return matmul(self, other)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad}, op={self._op})"


# ===========================================================================
# High-Performance Differentiable Ops
# ===========================================================================

def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Cache-tiled matrix multiplication supporting 2D, N-D x 2D, and batched N-D."""
    if a.dtype != b.dtype:
        raise TypeError(f"matmul dtype mismatch: {a.dtype} vs {b.dtype}")
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(f"matmul requires >=2D tensors, got {a.shape} and {b.shape}")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"matmul inner dim mismatch: {a.shape} @ {b.shape}")

    ctype = _ctype_for(a.dtype)
    K = a.shape[-1]
    N = b.shape[-1]

    # Fast path: [..., M, K] @ [K, N] -> fold leading dims into single 2D tiled GEMM!
    if b.ndim == 2:
        M_total = a.numel // K
        out_shape = a.shape[:-1] + (N,)
        out_buf = (ctype * max(1, M_total * N))()
        LIB.gb_matmul_tiled(
            a._buf, b._buf, out_buf,
            1, 1, 1, M_total, K, N, 0, 0, 0, a._dtype_code,
        )
        req = is_grad_enabled() and (a.requires_grad or b.requires_grad)
        out = Tensor(
            _buf=out_buf,
            shape=out_shape,
            dtype=a.dtype,
            requires_grad=req,
            _op="TiledMatMul",
            _parents=(a, b) if req else (),
        )
        if req:
            def _backward():
                g = out.grad._buf
                if a.requires_grad:
                    da = (ctype * max(1, a.numel))()
                    LIB.gb_matmul_tiled(g, b._buf, da, 1, 1, 1, M_total, N, K, 0, 1, 0, a._dtype_code)
                    a._accum_grad(da)
                if b.requires_grad:
                    db = (ctype * max(1, b.numel))()
                    LIB.gb_matmul_tiled(a._buf, g, db, 1, 1, 1, K, M_total, N, 1, 0, 0, a._dtype_code)
                    b._accum_grad(db)
            out._backward = _backward
        return out

    # Batched N-D matmul: [B_dims, M, K] @ [B_dims, K, N]
    M = a.shape[-2]
    ba_shape = a.shape[:-2]
    bb_shape = b.shape[:-2]
    if ba_shape != bb_shape and _prod(ba_shape) != 1 and _prod(bb_shape) != 1:
        raise ValueError(f"Unsupported batch broadcast in matmul: {a.shape} @ {b.shape}")
    out_batch_shape = ba_shape if _prod(ba_shape) >= _prod(bb_shape) else bb_shape
    batch = _prod(out_batch_shape)
    batch_a = _prod(ba_shape)
    batch_b = _prod(bb_shape)

    out_shape = out_batch_shape + (M, N)
    out_buf = (ctype * max(1, batch * M * N))()
    LIB.gb_matmul_tiled(
        a._buf, b._buf, out_buf,
        batch, batch_a, batch_b, M, K, N, 0, 0, 0, a._dtype_code,
    )
    req = is_grad_enabled() and (a.requires_grad or b.requires_grad)
    out = Tensor(
        _buf=out_buf,
        shape=out_shape,
        dtype=a.dtype,
        requires_grad=req,
        _op="BatchedMatMul",
        _parents=(a, b) if req else (),
    )
    if req:
        def _backward():
            g = out.grad._buf
            if a.requires_grad:
                if batch_a == batch:
                    da = (ctype * max(1, a.numel))()
                    LIB.gb_matmul_tiled(g, b._buf, da, batch, batch, batch_b, M, N, K, 0, 1, 0, a._dtype_code)
                    a._accum_grad(da)
                else:
                    # batch_a == 1: accumulate across batches
                    da_all = (ctype * max(1, batch * M * K))()
                    LIB.gb_matmul_tiled(g, b._buf, da_all, batch, batch, batch_b, M, N, K, 0, 1, 0, a._dtype_code)
                    da = (ctype * max(1, M * K))()
                    stride = M * K
                    for bi in range(batch):
                        for idx in range(stride):
                            da[idx] += da_all[bi * stride + idx]
                    a._accum_grad(da)
            if b.requires_grad:
                if batch_b == batch:
                    db = (ctype * max(1, b.numel))()
                    LIB.gb_matmul_tiled(a._buf, g, db, batch, batch_a, batch, K, M, N, 1, 0, 0, a._dtype_code)
                    b._accum_grad(db)
                else:
                    db_all = (ctype * max(1, batch * K * N))()
                    LIB.gb_matmul_tiled(a._buf, g, db_all, batch, batch_a, batch, K, M, N, 1, 0, 0, a._dtype_code)
                    db = (ctype * max(1, K * N))()
                    stride = K * N
                    for bi in range(batch):
                        for idx in range(stride):
                            db[idx] += db_all[bi * stride + idx]
                    b._accum_grad(db)
        out._backward = _backward
    return out


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Fused Linear: y = x @ weight^T (+ bias) using zero-transpose tiled GEMM."""
    if x.dtype != weight.dtype:
        raise TypeError("linear dtype mismatch")
    out_f, in_f = weight.shape
    if x.shape[-1] != in_f:
        raise ValueError(f"linear shape mismatch: x={x.shape}, weight={weight.shape}")

    rows = x.numel // in_f
    out_shape = x.shape[:-1] + (out_f,)
    ctype = _ctype_for(x.dtype)
    out_buf = (ctype * max(1, rows * out_f))()
    LIB.gb_matmul_tiled(
        x._buf, weight._buf, out_buf,
        1, 1, 1, rows, in_f, out_f, 0, 1, 0, x._dtype_code,
    )
    req = is_grad_enabled() and (x.requires_grad or weight.requires_grad)
    out = Tensor(
        _buf=out_buf,
        shape=out_shape,
        dtype=x.dtype,
        requires_grad=req,
        _op="Linear",
        _parents=(x, weight) if req else (),
    )
    if req:
        def _backward():
            g = out.grad._buf
            if x.requires_grad:
                dx = (ctype * max(1, x.numel))()
                LIB.gb_matmul_tiled(g, weight._buf, dx, 1, 1, 1, rows, out_f, in_f, 0, 0, 0, x._dtype_code)
                x._accum_grad(dx)
            if weight.requires_grad:
                dw = (ctype * max(1, weight.numel))()
                LIB.gb_matmul_tiled(g, x._buf, dw, 1, 1, 1, out_f, rows, in_f, 1, 0, 0, x._dtype_code)
                weight._accum_grad(dw)
        out._backward = _backward

    if bias is not None:
        out = out + bias
    return out


def embedding(table: Tensor, indices: Sequence[int], out_shape_prefix: Tuple[int, ...]) -> Tensor:
    """Lookup rows from [V, D] embedding table into (*out_shape_prefix, D)."""
    V, D = table.shape
    n_idx = len(indices)
    idx_arr = (ctypes.c_int32 * max(1, n_idx))(*indices)
    ctype = _ctype_for(table.dtype)
    out_buf = (ctype * max(1, n_idx * D))()
    LIB.gb_embedding_fwd(table._buf, idx_arr, out_buf, n_idx, D, table._dtype_code)

    req = is_grad_enabled() and table.requires_grad
    out = Tensor(
        _buf=out_buf,
        shape=out_shape_prefix + (D,),
        dtype=table.dtype,
        requires_grad=req,
        _op="Embedding",
        _parents=(table,) if req else (),
    )
    if req:
        def _backward():
            dtab = (ctype * max(1, V * D))()
            LIB.gb_embedding_bwd(out.grad._buf, idx_arr, dtab, n_idx, V, D, table._dtype_code)
            table._accum_grad(dtab)
        out._backward = _backward
    return out


def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Root Mean Square Layer Normalization (fused C++ forward + backward)."""
    D = x.shape[-1]
    if weight.shape != (D,):
        raise ValueError(f"rmsnorm weight shape {weight.shape} != ({D},)")
    N = x.numel // D
    ctype = _ctype_for(x.dtype)
    out_buf = (ctype * max(1, x.numel))()
    inv_rms_buf = (ctype * max(1, N))()
    LIB.gb_rmsnorm_fwd(x._buf, weight._buf, out_buf, inv_rms_buf, N, D, float(eps), x._dtype_code)

    req = is_grad_enabled() and (x.requires_grad or weight.requires_grad)
    out = Tensor(
        _buf=out_buf,
        shape=x.shape,
        dtype=x.dtype,
        requires_grad=req,
        _op="RMSNorm",
        _parents=(x, weight) if req else (),
    )
    if req:
        def _backward():
            dx = (ctype * max(1, x.numel))()
            dw = (ctype * max(1, D))()
            LIB.gb_rmsnorm_bwd(
                x._buf, weight._buf, inv_rms_buf, out.grad._buf,
                dx, dw, N, D, x._dtype_code,
            )
            if x.requires_grad:
                x._accum_grad(dx)
            if weight.requires_grad:
                weight._accum_grad(dw)
        out._backward = _backward
    return out


def rope(x: Tensor, pos_offset: int = 0, base: float = 10000.0) -> Tensor:
    """Rotary Position Embedding (RoPE) on shape [B, H, T, D]."""
    if x.ndim != 4:
        raise ValueError(f"rope expects 4D tensor [B, H, T, D], got {x.shape}")
    B, H, T_len, D = x.shape
    if D % 2 != 0:
        raise ValueError(f"rope head_dim must be even, got {D}")
    ctype = _ctype_for(x.dtype)
    out_buf = (ctype * max(1, x.numel))()
    LIB.gb_rope(x._buf, out_buf, B * H, T_len, D, int(pos_offset), float(base), 0, x._dtype_code)

    req = is_grad_enabled() and x.requires_grad
    out = Tensor(
        _buf=out_buf,
        shape=x.shape,
        dtype=x.dtype,
        requires_grad=req,
        _op="RoPE",
        _parents=(x,) if req else (),
    )
    if req:
        def _backward():
            dx = (ctype * max(1, x.numel))()
            LIB.gb_rope(out.grad._buf, dx, B * H, T_len, D, int(pos_offset), float(base), 1, x._dtype_code)
            x._accum_grad(dx)
        out._backward = _backward
    return out


def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool = True,
    return_probs: bool = False,
    q_offset: int = 0,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """FlashAttention-2 style tiled online-softmax attention (O(T) memory during training).
    Supports standard MHA, Grouped-Query Attention (H_q % H_kv == 0), and KV-cache decoding (T_q <= T_k, q_offset).
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise ValueError(f"flash_attention expects 4D [B,H_q,T_q,D] & [B,H_kv,T_k,D], got {q.shape}, {k.shape}, {v.shape}")
    B, H_q, T_q, D = q.shape
    Bk, H_kv, T_k, Dk = k.shape
    if B != Bk or D != Dk or H_q % H_kv != 0:
        raise ValueError(f"Incompatible GQA shapes: Q={q.shape}, K={k.shape}")
    scale = 1.0 / math.sqrt(D)
    ctype = _ctype_for(q.dtype)

    o_buf = (ctype * max(1, q.numel))()
    l_buf = (ctype * max(1, B * H_q * T_q))()
    probs_buf = (ctype * max(1, B * H_q * T_q * T_k))() if return_probs else None

    use_gqa = (H_q != H_kv) or (T_q != T_k) or (q_offset != 0)
    if use_gqa:
        LIB.gb_flash_attn_gqa_fwd(
            q._buf, k._buf, v._buf,
            o_buf, l_buf, probs_buf,
            B, H_q, H_kv, T_q, T_k, D,
            scale, 1 if causal else 0, int(q_offset), q._dtype_code,
        )
    else:
        LIB.gb_flash_attn_fwd(
            q._buf, k._buf, v._buf,
            o_buf, l_buf, probs_buf,
            B * H_q, T_q, D, scale, 1 if causal else 0, q._dtype_code,
        )

    req = is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    out = Tensor(
        _buf=o_buf,
        shape=q.shape,
        dtype=q.dtype,
        requires_grad=req,
        _op="FlashAttnGQA" if H_q != H_kv else "FlashAttn",
        _parents=(q, k, v) if req else (),
    )
    if req:
        def _backward():
            dq = (ctype * max(1, q.numel))()
            dk = (ctype * max(1, k.numel))()
            dv = (ctype * max(1, v.numel))()
            if use_gqa:
                LIB.gb_flash_attn_gqa_bwd(
                    q._buf, k._buf, v._buf,
                    o_buf, l_buf, out.grad._buf,
                    dq, dk, dv,
                    B, H_q, H_kv, T_q, T_k, D,
                    scale, 1 if causal else 0, int(q_offset), q._dtype_code,
                )
            else:
                LIB.gb_flash_attn_bwd(
                    q._buf, k._buf, v._buf,
                    o_buf, l_buf, out.grad._buf,
                    dq, dk, dv,
                    B * H_q, T_q, D, scale, 1 if causal else 0, q._dtype_code,
                )
            if q.requires_grad:
                q._accum_grad(dq)
            if k.requires_grad:
                k._accum_grad(dk)
            if v.requires_grad:
                v._accum_grad(dv)
        out._backward = _backward

    if return_probs:
        probs_tensor = Tensor(_buf=probs_buf, shape=(B, H_q, T_q, T_k), dtype=q.dtype, requires_grad=False)
        return out, probs_tensor
    return out


def naive_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = True) -> Tuple[Tensor, Tensor]:
    """Reference unfused O(T^2) attention for equivalence verification against FlashAttention."""
    B, H, T_len, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        mask_vals = [0.0 if j <= i else -1e9 for i in range(T_len) for j in range(T_len)]
        mask = Tensor(mask_vals, shape=(1, 1, T_len, T_len), dtype=q.dtype, requires_grad=False)
        scores = scores + mask
    probs = scores.softmax(dim=-1)
    out = probs @ v
    return out, probs


def cross_entropy(logits: Tensor, targets: Sequence[int], ignore_index: int = -100) -> Tensor:
    """Numerically stable fused log-softmax cross-entropy loss."""
    V = logits.shape[-1]
    N = logits.numel // V
    if len(targets) != N:
        raise ValueError(f"targets length {len(targets)} != batch*seq {N}")

    tgt_arr = (ctypes.c_int32 * max(1, N))(*targets)
    ctype = _ctype_for(logits.dtype)
    probs_buf = (ctype * max(1, N * V))()
    valid_c = ctypes.c_int(0)

    loss_val = LIB.gb_cross_entropy_fwd(
        logits._buf, tgt_arr, probs_buf, N, V, int(ignore_index), ctypes.byref(valid_c), logits._dtype_code,
    )
    out_buf = (ctype * 1)(loss_val)

    req = is_grad_enabled() and logits.requires_grad
    out = Tensor(
        _buf=out_buf,
        shape=(),
        dtype=logits.dtype,
        requires_grad=req,
        _op="CrossEntropy",
        _parents=(logits,) if req else (),
    )
    if req:
        valid_count = valid_c.value

        def _backward():
            dlogits = (ctype * max(1, N * V))()
            grad_out = float(out.grad._buf[0])
            LIB.gb_cross_entropy_bwd(
                probs_buf, tgt_arr, dlogits, grad_out, N, V, int(ignore_index), valid_count, logits._dtype_code,
            )
            logits._accum_grad(dlogits)

        out._backward = _backward
    return out
