
"""Opt-in compiled execution of the existing coordinate-descent loop."""
from contextvars import ContextVar
from contextlib import contextmanager
import numpy as np
_WORK=ContextVar('compiled_bridge_work',default=None)
_KERNEL=None
_NUMBA_VERSION=None
if np.__version__=='1.26.4':
    try:
        import numba
        _NUMBA_VERSION=numba.__version__
        if _NUMBA_VERSION=='0.61.2':
            @numba.njit(cache=True,fastmath=False,error_model='numpy')
            def _kernel(hessian,gradient,lower,upper):
                n=len(gradient)
                solution=np.zeros(n,dtype=np.float64)
                for cycle in range(10000):
                    previous=solution.copy()
                    for index in range(n):
                        partial=gradient[index]+np.dot(hessian[index],solution)-hessian[index,index]*solution[index]
                        value=-partial/hessian[index,index]
                        solution[index]=np.minimum(np.maximum(value,lower[index]),upper[index])
                    if np.max(np.abs(solution-previous))<=1e-12:
                        return solution,cycle+1
                return solution,-1
            _KERNEL=_kernel
    except ImportError:
        pass
@contextmanager
def enabled():
    token=_WORK.set(dict(calls=0,cycles=[],fallbacks=0))
    try:yield
    finally:_WORK.reset(token)
def solve(hessian,gradient,lower,upper):
    work=_WORK.get()
    if work is None:return False,None
    n=len(gradient)
    supported=(_KERNEL is not None and 1<=n<=10
        and all(type(a) is np.ndarray and a.dtype==np.dtype('float64') and a.flags.c_contiguous
            and np.isfinite(a).all() for a in (hessian,gradient,lower,upper))
        and hessian.shape==(n,n) and gradient.shape==lower.shape==upper.shape==(n,)
        and np.all(np.diag(hessian)>0) and np.all(lower<=upper))
    if not supported:
        work['fallbacks']+=1;return False,None
    solution,cycles=_KERNEL(hessian,gradient,lower,upper)
    work['calls']+=1;work['cycles'].append(cycles)
    return True,solution if cycles>0 else None
def receipt():
    w=_WORK.get()
    return dict(identity='C125_NUMBA_COORDINATE_DESCENT_NO_FASTMATH',enabled=w is not None,
        kernel_available=_KERNEL is not None,numpy=np.__version__,numba=_NUMBA_VERSION,
        calls=0 if w is None else w['calls'],cycles=[] if w is None else list(w['cycles']),
        fallbacks=0 if w is None else w['fallbacks'],
        settings=dict(max_cycles=10000,tolerance=1e-12,fastmath=False))

def prepare():
    """Compile/load the concrete float64 C-array signature before controller start."""
    import time
    started=time.monotonic_ns()
    if _KERNEL is None:
        return dict(available=False,warmup_ms=(time.monotonic_ns()-started)/1e6,signatures=[])
    h=np.eye(5,dtype=np.float64)
    g=np.zeros(5,dtype=np.float64)
    solution,cycles=_KERNEL(h,g,-np.ones(5,dtype=np.float64),np.ones(5,dtype=np.float64))
    assert cycles==1 and np.array_equal(solution,g)
    return dict(available=True,warmup_ms=(time.monotonic_ns()-started)/1e6,
        signatures=[str(x) for x in _KERNEL.signatures])
