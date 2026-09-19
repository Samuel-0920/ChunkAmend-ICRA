"""Conservative box / affine-ball infeasibility certificates, opt-in."""
from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction as F
import numpy as np

_WORK=ContextVar('box_ball_infeasibility_work',default=None)

@contextmanager
def enabled():
    token=_WORK.set(dict(checks=0,exact_checks=0,certificates=[]))
    try:yield
    finally:_WORK.reset(token)

def receipt():
    work=_WORK.get()
    return dict(identity='C122_EXACT_RATIONAL_SUPPORT_WITH_FLOAT_GUARD',enabled=work is not None,
        checks=0 if work is None else work['checks'],exact_checks=0 if work is None else work['exact_checks'],
        certificates=[] if work is None else list(work['certificates']))

def certify_ball(point,lower,upper,ball,tolerance):
    """Return a sufficient certificate, never a feasibility decision from absence."""
    a=np.asarray(ball.matrix);b=np.asarray(ball.offset);l=np.asarray(lower);u=np.asarray(upper);p=np.asarray(point)
    if (a.ndim!=2 or a.shape[0]!=3 or not 1<=a.shape[1]<=27 or b.shape!=(3,)
            or l.shape!=(a.shape[1],) or u.shape!=l.shape or p.shape!=l.shape
            or any(x.dtype!=np.dtype(np.float64) or not np.isfinite(x).all() or np.max(np.abs(x))>1e6 for x in (a,b,l,u,p))
            or np.any(l>u) or not np.isfinite(ball.radius) or not 0<=ball.radius<=1e6
            or tolerance!=1e-9):
        return None
    y=a@p+b # Any finite nonzero direction is valid; it need not be an exact residual.
    ynorm=float(np.linalg.norm(y))
    if not np.isfinite(ynorm) or ynorm==0:return None
    w=y@a
    approximate_lower=float(y@b+np.sum(np.where(w>=0,w*l,w*u)))
    if approximate_lower<=(ball.radius+tolerance)*ynorm:return None
    work=_WORK.get()
    if work is not None:work['exact_checks']+=1
    # The floating prefilter can only avoid work. It never establishes rejection.
    af=[[F(float(v)) for v in row] for row in a]
    bf=[F(float(v)) for v in b];yf=[F(float(v)) for v in y]
    lf=[F(float(v)) for v in l];uf=[F(float(v)) for v in u]
    radius=F(float(ball.radius));tol=F(float(tolerance))
    extent=[max(abs(lo),abs(hi)) for lo,hi in zip(lf,uf)]
    scale=max([F(1),radius]+[abs(bf[i])+sum(abs(v)*e for v,e in zip(af[i],extent)) for i in range(3)])
    # 1024*float64 epsilon times a magnitude bound covers the <=27-term
    # matmul, addition, 3-vector norm and norm-radius rounding. Magnitudes
    # above 1e6 per input and other dtypes use the original solver instead.
    guard=scale/F(2**42)
    weights=[sum(yf[i]*af[i][j] for i in range(3)) for j in range(a.shape[1])]
    support=sum(v*z for v,z in zip(yf,bf))+sum(w*(lo if w>=0 else hi) for w,lo,hi in zip(weights,lf,uf))
    direction_squared=sum(v*v for v in yf);allowed=radius+tol+guard
    gap=support*support-allowed*allowed*direction_squared
    if support<=0 or gap<=0:return None
    return dict(ball=ball.name,direction=y.tolist(),support_lower_exact=str(support),
        allowed_radius_exact=str(allowed),direction_squared_norm_exact=str(direction_squared),
        positive_squared_gap_exact=str(gap),float_error_guard_exact=str(guard),magnitude_bound_exact=str(scale))

def reject_if_certified(point,lower,upper,balls,*,tolerance,iteration):
    work=_WORK.get()
    if work is None or iteration not in (0,4,16,64):return None
    work['checks']+=1
    for ball in balls:
        certificate=certify_ball(point,lower,upper,ball,tolerance)
        if certificate is not None:
            certificate=dict(certificate,iteration=iteration)
            work['certificates'].append(certificate)
            return certificate
    return None
