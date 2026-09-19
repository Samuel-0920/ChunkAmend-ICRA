"""Pinned NumPy dispatch specialization; public fallback outside proven shapes/types.

This reuses the exact NumPy 1.26.4 gufunc, signature and error callback.
It does not change LAPACK, clipping order, bisection, tolerances or float dtype.
"""
import numpy as np
from contextvars import ContextVar
from contextlib import contextmanager

_ENABLED = ContextVar("numeric_dispatch_scoped_v2_enabled", default=False)

@contextmanager
def enabled():
    token = _ENABLED.set(True)
    try:
        yield
    finally:
        _ENABLED.reset(token)

_F64 = np.dtype(np.float64)
_SOLVE1 = _CLIP = _EXT = None
if np.__version__ == '1.26.4':
    try:
        from numpy.linalg import _umath_linalg, linalg as _linalg
        from numpy.core import umath as _umath
        _SOLVE1 = _umath_linalg.solve1
        _EXT = _linalg.get_linalg_error_extobj(_linalg._raise_linalgerror_singular)
        _CLIP = _umath.clip
    except (ImportError, AttributeError):
        _SOLVE1 = _CLIP = _EXT = None

def solve3_f64(a, b):
    if (_ENABLED.get() and _SOLVE1 is not None and type(a) is np.ndarray and type(b) is np.ndarray
            and a.dtype == _F64 and b.dtype == _F64 and a.shape == (3, 3) and b.shape == (3,)):
        return _SOLVE1(a, b, signature='dd->d', extobj=_EXT)
    return np.linalg.solve(a, b)

def clip_scalar_f64(value, lower, upper):
    if (_ENABLED.get() and _CLIP is not None and type(value) is np.float64
            and type(lower) is np.float64 and type(upper) is np.float64):
        return _CLIP(value, lower, upper)
    return np.clip(value, lower, upper)

def implementation_receipt():
    return dict(identity='NUMERIC_DISPATCH_SCOPED_V2', enabled_in_context=_ENABLED.get(), numpy_version=np.__version__,
        specialized_entry_points_available=_SOLVE1 is not None and _CLIP is not None,
        unsupported_runtime_or_input='PUBLIC_NUMPY_FALLBACK', mathematical_operations_changed=False)
