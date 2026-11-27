"""Zero-dependency ctypes bindings to Glassbox's custom C++17 compute kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

CSRC_DIR = Path(__file__).resolve().parent / "csrc"
SRC_FILE = CSRC_DIR / "kernels.cpp"
LIB_FILE = CSRC_DIR / "libglassbox.so"

DTYPE_F32 = 0
DTYPE_F64 = 1


def compile_kernels(force: bool = False) -> Path:
    """Compile C++17 SIMD/cache-tiled kernels with g++ (zero external libraries)."""
    if (
        not force
        and LIB_FILE.exists()
        and LIB_FILE.stat().st_mtime >= SRC_FILE.stat().st_mtime
    ):
        return LIB_FILE

    cmd = [
        "g++",
        "-O3",
        "-march=native",
        "-fno-math-errno",
        "-fno-trapping-math",
        "-shared",
        "-fPIC",
        "-std=c++17",
        str(SRC_FILE),
        "-o",
        str(LIB_FILE),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to compile Glassbox C++ kernels:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return LIB_FILE


def _load_library() -> ctypes.CDLL:
    lib_path = compile_kernels(force=False)
    lib = ctypes.CDLL(str(lib_path))

    c_vp = ctypes.c_void_p
    c_i32p = ctypes.POINTER(ctypes.c_int32)
    c_ip = ctypes.POINTER(ctypes.c_int)
    c_u64p = ctypes.POINTER(ctypes.c_uint64)
    c_dp = ctypes.POINTER(ctypes.c_double)
    c_int = ctypes.c_int
    c_dbl = ctypes.c_double

    lib.gb_reset_stats.argtypes = []
    lib.gb_reset_stats.restype = None

    lib.gb_get_stats.argtypes = [c_u64p, c_u64p, c_u64p]
    lib.gb_get_stats.restype = None

    lib.gb_matmul_tiled.argtypes = [
        c_vp, c_vp, c_vp,
        c_int, c_int, c_int,
        c_int, c_int, c_int,
        c_int, c_int, c_int, c_int,
    ]
    lib.gb_matmul_tiled.restype = None

    lib.gb_matmul_naive.argtypes = [
        c_vp, c_vp, c_vp,
        c_int, c_int, c_int,
        c_int, c_int, c_int, c_int,
    ]
    lib.gb_matmul_naive.restype = None

    lib.gb_flash_attn_fwd.argtypes = [
        c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
        c_int, c_int, c_int, c_dbl, c_int, c_int,
    ]
    lib.gb_flash_attn_fwd.restype = None

    lib.gb_flash_attn_bwd.argtypes = [
        c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
        c_vp, c_vp, c_vp,
        c_int, c_int, c_int, c_dbl, c_int, c_int,
    ]
    lib.gb_flash_attn_bwd.restype = None

    lib.gb_rmsnorm_fwd.argtypes = [c_vp, c_vp, c_vp, c_vp, c_int, c_int, c_dbl, c_int]
    lib.gb_rmsnorm_fwd.restype = None

    lib.gb_rmsnorm_bwd.argtypes = [c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_int, c_int, c_int]
    lib.gb_rmsnorm_bwd.restype = None

    lib.gb_rope.argtypes = [c_vp, c_vp, c_int, c_int, c_int, c_int, c_dbl, c_int, c_int]
    lib.gb_rope.restype = None

    lib.gb_cross_entropy_fwd.argtypes = [c_vp, c_i32p, c_vp, c_int, c_int, c_int, c_ip, c_int]
    lib.gb_cross_entropy_fwd.restype = c_dbl

    lib.gb_cross_entropy_bwd.argtypes = [c_vp, c_i32p, c_vp, c_dbl, c_int, c_int, c_int, c_int, c_int]
    lib.gb_cross_entropy_bwd.restype = None

    lib.gb_binary_bcast.argtypes = [c_vp, c_vp, c_vp, c_ip, c_ip, c_ip, c_int, c_int, c_int, c_int]
    lib.gb_binary_bcast.restype = None

    lib.gb_reduce_bcast.argtypes = [c_vp, c_vp, c_ip, c_ip, c_int, c_int, c_int, c_int]
    lib.gb_reduce_bcast.restype = None

    lib.gb_unary_fwd.argtypes = [c_vp, c_vp, c_int, c_int, c_int]
    lib.gb_unary_fwd.restype = None

    lib.gb_unary_bwd.argtypes = [c_vp, c_vp, c_vp, c_int, c_int, c_int]
    lib.gb_unary_bwd.restype = None

    lib.gb_softmax_fwd.argtypes = [c_vp, c_vp, c_int, c_int, c_int]
    lib.gb_softmax_fwd.restype = None

    lib.gb_softmax_bwd.argtypes = [c_vp, c_vp, c_vp, c_int, c_int, c_int]
    lib.gb_softmax_bwd.restype = None

    lib.gb_embedding_fwd.argtypes = [c_vp, c_i32p, c_vp, c_int, c_int, c_int]
    lib.gb_embedding_fwd.restype = None

    lib.gb_embedding_bwd.argtypes = [c_vp, c_i32p, c_vp, c_int, c_int, c_int, c_int]
    lib.gb_embedding_bwd.restype = None

    lib.gb_permute.argtypes = [c_vp, c_vp, c_ip, c_ip, c_ip, c_int, c_int, c_int]
    lib.gb_permute.restype = None

    lib.gb_adamw_step.argtypes = [
        c_vp, c_vp, c_vp, c_vp,
        c_int, c_dbl, c_dbl, c_dbl, c_dbl, c_dbl, c_int,
        c_dp, c_dp, c_int,
    ]
    lib.gb_adamw_step.restype = None

    lib.gb_l2_norm_sq.argtypes = [c_vp, c_int, c_int]
    lib.gb_l2_norm_sq.restype = c_dbl

    lib.gb_scale_inplace.argtypes = [c_vp, c_int, c_dbl, c_int]
    lib.gb_scale_inplace.restype = None

    lib.gb_add_inplace.argtypes = [c_vp, c_vp, c_int, c_dbl, c_int]
    lib.gb_add_inplace.restype = None

    return lib


LIB = _load_library()


def reset_kernel_stats() -> None:
    LIB.gb_reset_stats()


def get_kernel_stats() -> Dict[str, float]:
    flops = ctypes.c_uint64(0)
    calls = ctypes.c_uint64(0)
    ns = ctypes.c_uint64(0)
    LIB.gb_get_stats(ctypes.byref(flops), ctypes.byref(calls), ctypes.byref(ns))
    secs = ns.value / 1e9 if ns.value > 0 else 0.0
    gflops = (flops.value / 1e9) / secs if secs > 0 else 0.0
    return {
        "total_flops": float(flops.value),
        "matmul_calls": int(calls.value),
        "matmul_seconds": secs,
        "gflops": gflops,
    }


def benchmark_matmul(sizes: List[int] | None = None) -> List[Dict[str, float]]:
    """Benchmark custom cache-tiled AVX/FMA SGEMM vs naive i-j-k SGEMM."""
    import random

    if sizes is None:
        sizes = [64, 128, 256, 384]

    rng = random.Random(42)
    results = []
    for n in sizes:
        buf_a = (ctypes.c_float * (n * n))(*(rng.uniform(-1.0, 1.0) for _ in range(n * n)))
        buf_b = (ctypes.c_float * (n * n))(*(rng.uniform(-1.0, 1.0) for _ in range(n * n)))
        buf_c_naive = (ctypes.c_float * (n * n))()
        buf_c_tiled = (ctypes.c_float * (n * n))()

        flops = 2.0 * n * n * n
        iters_tiled = 20 if n <= 128 else (6 if n <= 256 else 3)
        iters_naive = 5 if n <= 128 else (2 if n <= 256 else 1)

        # Warmup tiled
        LIB.gb_matmul_tiled(buf_a, buf_b, buf_c_tiled, 1, 1, 1, n, n, n, 0, 0, 0, DTYPE_F32)

        t0 = time.perf_counter()
        for _ in range(iters_naive):
            LIB.gb_matmul_naive(buf_a, buf_b, buf_c_naive, n, n, n, 0, 0, 0, DTYPE_F32)
        naive_ms = ((time.perf_counter() - t0) / iters_naive) * 1000.0

        t0 = time.perf_counter()
        for _ in range(iters_tiled):
            LIB.gb_matmul_tiled(buf_a, buf_b, buf_c_tiled, 1, 1, 1, n, n, n, 0, 0, 0, DTYPE_F32)
        tiled_ms = ((time.perf_counter() - t0) / iters_tiled) * 1000.0

        # Verify numerical equivalence
        max_err = 0.0
        step = max(1, (n * n) // 512)
        for i in range(0, n * n, step):
            err = abs(buf_c_naive[i] - buf_c_tiled[i])
            if err > max_err:
                max_err = err

        naive_gflops = (flops / 1e9) / (naive_ms / 1000.0)
        tiled_gflops = (flops / 1e9) / (tiled_ms / 1000.0)
        results.append(
            {
                "size": n,
                "naive_ms": round(naive_ms, 3),
                "tiled_ms": round(tiled_ms, 3),
                "naive_gflops": round(naive_gflops, 2),
                "tiled_gflops": round(tiled_gflops, 2),
                "speedup": round(naive_ms / max(tiled_ms, 1e-9), 2),
                "max_abs_error": float(max_err),
            }
        )
    return results
