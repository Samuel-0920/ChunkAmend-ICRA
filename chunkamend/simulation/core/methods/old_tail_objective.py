"""Request-local evidence of old-tail quadratic construction, not an endpoint."""
from contextvars import ContextVar
from contextlib import contextmanager
import hashlib
import numpy as np
_WORK=ContextVar('old_tail_objective',default=None)
@contextmanager
def capture():
    work=[];token=_WORK.set(work)
    try:yield work
    finally:_WORK.reset(token)
def record(targets,mu):
    work=_WORK.get()
    if work is not None:
        work.append(dict(length=len(targets),mu=float(mu),targets_sha256=hashlib.sha256(np.ascontiguousarray(targets,dtype=np.float64).tobytes()).hexdigest()))
