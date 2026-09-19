"""Offline native prototype of the original affine-ball search, no fastmath."""
import numpy as np
from contextvars import ContextVar
from contextlib import contextmanager
import time
try:
    import numba
except ImportError:
    numba=None

def _compile(function):
    if numba is None or np.__version__!='1.26.4' or numba.__version__!='0.61.2':
        return function
    return numba.njit(cache=True,fastmath=False,error_model='numpy')(function)


@_compile
def evaluate(point,matrix,offset,gram,identity,value,multiplier):
    try:
        residual=np.linalg.solve(identity+multiplier*gram,value)
    except Exception:
        return np.empty(0,dtype=np.float64),np.nan,False
    candidate=point-multiplier*matrix.T@residual
    vector=matrix@candidate+offset
    norm=np.sqrt(np.dot(vector,vector))
    return candidate,norm,np.isfinite(candidate).all() and np.isfinite(norm)

@_compile
def kernel(point,matrix,offset,gram,identity,radius,tolerance):
    value=matrix@point+offset
    if not np.isfinite(value).all() or radius<0:return None
    if np.sqrt(np.dot(value,value))<=radius+tolerance:return point.copy()
    upper=1.0
    candidate,norm,valid=evaluate(point,matrix,offset,gram,identity,value,upper)
    while valid and norm>radius+tolerance and upper<1e12:
        upper*=2
        candidate,norm,valid=evaluate(point,matrix,offset,gram,identity,value,upper)
    if not valid or norm>radius+tolerance:return None
    upper_result=candidate
    lower=0.0
    for _ in range(80):
        middle=(lower+upper)/2
        candidate,norm,valid=evaluate(point,matrix,offset,gram,identity,value,middle)
        if not valid:return None
        if norm>radius:lower=middle
        else:
            upper=middle
            upper_result=candidate
        if upper-lower<=tolerance*max(1.0,upper):break
    return upper_result


_WORK=ContextVar('compiled_projection_work',default=None)
_PREPARED=False
_AVAILABLE=numba is not None and np.__version__=='1.26.4' and numba.__version__=='0.61.2'
_F64=np.dtype('float64')

@contextmanager
def enabled():
    token=_WORK.set(dict(calls=0,fallbacks=0,violation_calls=0,violation_fallbacks=0))
    try:yield
    finally:_WORK.reset(token)

def prepare():
    """Compile only the actual fixed-k5 layout before any controller activity."""
    global _PREPARED
    started=time.monotonic_ns()
    if not _AVAILABLE:return dict(available=False,prepared=False,warmup_ms=(time.monotonic_ns()-started)/1e6,signatures=[])
    point=np.ones(15,dtype=np.float64)*.1
    matrix=np.eye(3,15,dtype=np.float64);offset=np.zeros(3,dtype=np.float64)
    gram=matrix@matrix.T;identity=np.eye(3,dtype=np.float64)
    for array in (matrix,offset,gram,identity):array.flags.writeable=False
    result=kernel(point,matrix,offset,gram,identity,.01,1e-12)
    if result is None or not np.isfinite(result).all() or len(kernel.signatures)!=1:
        raise RuntimeError('Projection kernel preparation failed')
    violation_value=violation_kernel(point,point-1.,point+1.,(matrix,matrix,matrix),(offset,offset,offset),(0.01,0.01,0.01))
    violation_value4=violation_kernel(point,point-1.,point+1.,(matrix,matrix,matrix,matrix),(offset,offset,offset,offset),(0.01,0.01,0.01,0.01))
    if not np.isfinite([violation_value,violation_value4]).all() or len(violation_kernel.signatures)!=2: raise RuntimeError('Violation preparation failed')
    _PREPARED=True
    return dict(available=True,prepared=True,warmup_ms=(time.monotonic_ns()-started)/1e6,signatures=[str(x) for x in kernel.signatures])

def project(point,matrix,offset,gram,identity,radius,tolerance):
    work=_WORK.get()
    if work is None:return False,None
    arrays=(point,matrix,offset,gram,identity)
    supported=(_PREPARED and all(type(a) is np.ndarray and a.dtype==_F64 and a.flags.c_contiguous and a.flags.aligned for a in arrays)
        and point.shape in ((9,),(15,)) and matrix.shape==(3,len(point)) and offset.shape==(3,) and gram.shape==identity.shape==(3,3)
        and point.flags.writeable and all(not a.flags.writeable for a in arrays[1:])
        and type(radius) in (float,np.float64) and type(tolerance) in (float,np.float64))
    if not supported:
        work['fallbacks']+=1;return False,None
    value=kernel(point,matrix,offset,gram,identity,radius,tolerance)
    work['calls']+=1
    return True,value

def receipt():
    work=_WORK.get()
    return dict(identity='C143_NUMBA_AFFINE_BALL_ORIGINAL_SEARCH',enabled=work is not None,prepared=_PREPARED,
        kernel_available=_AVAILABLE,calls=0 if work is None else work['calls'],fallbacks=0 if work is None else work['fallbacks'],
        native_violation=dict(identity='C149_NATIVE_MAX_VIOLATION',calls=0 if work is None else work['violation_calls'],fallbacks=0 if work is None else work['violation_fallbacks']),
        settings=dict(max_bisections=80,bracket_upper_limit=1e12,fastmath=False,unsupported_or_unprepared='ORIGINAL_NUMPY'))


@_compile
def violation_kernel(point,lower,upper,matrices,offsets,radii):
    # Preserve NumPy's Euclidean norm as sqrt(dot), avoiding a different nrm2.
    lo=np.max(lower-point)
    hi=np.max(point-upper)
    if not np.isfinite(lo) or not np.isfinite(hi):return np.inf
    result=max(lo,hi,0.0)
    for i in range(len(matrices)):
        vector=matrices[i]@point+offsets[i]
        value=np.sqrt(np.dot(vector,vector))-radii[i]
        if not np.isfinite(value):return np.inf
        result=max(result,value)
    return result


def project_violation(point,lower,upper,balls):
    work=_WORK.get()
    if work is None:return False,None
    arrays=(point,lower,upper)
    supported=(_PREPARED and len(balls) in (3,4) and all(type(a) is np.ndarray and a.dtype==_F64 and a.shape==point.shape and point.shape in ((9,),(15,)) and a.flags.c_contiguous and a.flags.aligned and a.flags.writeable for a in arrays)
        and all(type(b.matrix) is np.ndarray and b.matrix.dtype==_F64 and b.matrix.shape==(3,len(point)) and b.matrix.flags.c_contiguous and b.matrix.flags.aligned and not b.matrix.flags.writeable
            and type(b.offset) is np.ndarray and b.offset.dtype==_F64 and b.offset.shape==(3,) and b.offset.flags.c_contiguous and b.offset.flags.aligned and not b.offset.flags.writeable
            and type(b.radius) in (float,np.float64) for b in balls))
    if not supported:
        work['violation_fallbacks']+=1
        return False,None
    value=violation_kernel(point,lower,upper,tuple(b.matrix for b in balls),tuple(b.offset for b in balls),tuple(float(b.radius) for b in balls))
    work['violation_calls']+=1
    return True,value
