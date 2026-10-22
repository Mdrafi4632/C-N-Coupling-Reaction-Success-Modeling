# models/_tanimoto.py — ctypes wrapper for tanimoto.c
# Auto-compiles tanimoto.c on first import (or when source is newer than .so).
# Exposes tanimoto_matrix() and tanimoto_distance_matrix() as numpy functions.
# Falls back to a vectorised numpy implementation if gcc is unavailable.

import ctypes
import subprocess
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_SRC  = _HERE / "tanimoto.c"
_LIB  = _HERE / "tanimoto.so"


def _compile() -> bool:
    flags = ["-O3", "-march=native", "-shared", "-fPIC"]
    # Try with OpenMP first (parallel outer loop)
    try:
        subprocess.run(
            ["gcc", *flags, "-fopenmp", "-o", str(_LIB), str(_SRC)],
            check=True, capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    # Retry without OpenMP
    try:
        subprocess.run(
            ["gcc", *flags, "-o", str(_LIB), str(_SRC)],
            check=True, capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _load():
    needs_compile = (
        not _LIB.exists()
        or (_SRC.exists() and _SRC.stat().st_mtime > _LIB.stat().st_mtime)
    )
    if needs_compile and not _compile():
        return None
    try:
        lib = ctypes.CDLL(str(_LIB))
        _fp = ctypes.POINTER(ctypes.c_float)
        for name in ("tanimoto_matrix", "tanimoto_distance_matrix"):
            fn = getattr(lib, name)
            fn.argtypes = [_fp, _fp, _fp, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            fn.restype  = None
        return lib
    except Exception:
        return None


_lib = _load()


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _numpy_tanimoto(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    dot  = (A @ B.T).astype(np.float32)
    sa   = A.sum(1, keepdims=True).astype(np.float32)
    sb   = B.sum(1, keepdims=True).T.astype(np.float32)
    denom = sa + sb - dot
    out   = np.zeros_like(dot)
    mask  = denom > 1e-9
    out[mask] = dot[mask] / denom[mask]
    return out


def tanimoto_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return n_a × n_b float32 Tanimoto similarity matrix."""
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[1]
    n_a, n_b, d = A.shape[0], B.shape[0], A.shape[1]
    out = np.empty((n_a, n_b), dtype=np.float32)
    if _lib is None:
        return _numpy_tanimoto(A, B)
    _lib.tanimoto_matrix(_ptr(A), _ptr(B), _ptr(out), n_a, n_b, d)
    return out


def tanimoto_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return n_a × n_b float32 Tanimoto distance matrix (1 − similarity)."""
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[1]
    n_a, n_b, d = A.shape[0], B.shape[0], A.shape[1]
    out = np.empty((n_a, n_b), dtype=np.float32)
    if _lib is None:
        return 1.0 - _numpy_tanimoto(A, B)
    _lib.tanimoto_distance_matrix(_ptr(A), _ptr(B), _ptr(out), n_a, n_b, d)
    return out
