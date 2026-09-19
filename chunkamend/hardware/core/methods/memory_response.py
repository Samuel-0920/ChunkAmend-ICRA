"""Offline research candidate: action plus one previous displacement, no intercept.

This module is not installed in the running policy or simulation launcher.
"""
import numpy as np

WINDOW = 64
RIDGE = 1e-4
MIN_SAMPLES = 8
CONDITION_LIMIT = 1e8
STATE_SCALE_M = 0.05

def fit_memory_response(completed_actions, completed_displacements, *, preceding_displacement=None, state_ridge=RIDGE):
    if state_ridge not in (1e-4,1e-3):raise ValueError('Nonfrozen state ridge')
    u=np.asarray(completed_actions,dtype=np.float64)
    d=np.asarray(completed_displacements,dtype=np.float64)
    if u.ndim!=2 or u.shape[1:]!=(3,) or d.shape!=u.shape or not np.isfinite(u).all() or not np.isfinite(d).all():
        raise ValueError('Expected finite chronological completed [N,3] histories')
    end=len(u);start=max(1,end-WINDOW) if preceding_displacement is None else 0
    if preceding_displacement is not None:
        preceding_displacement=np.asarray(preceding_displacement,dtype=np.float64)
        if end>WINDOW or preceding_displacement.shape!=(3,) or not np.isfinite(preceding_displacement).all():
            raise ValueError("Invalid preceding completed displacement")
    if end-start<MIN_SAMPLES:
        return dict(available=False,reason='INSUFFICIENT_COMPLETED_PAIRS',start=start,end=end)
    lag=d[start-1:end-1] if preceding_displacement is None else np.concatenate((preceding_displacement[None],d[:-1]),axis=0)
    x=np.concatenate((u[start:end],lag/STATE_SCALE_M),axis=1)
    y=d[start:end]
    system=x.T@x+np.diag(np.asarray([RIDGE]*3+[state_ridge]*3))
    condition=float(np.linalg.cond(system))
    if not np.isfinite(condition) or condition>CONDITION_LIMIT:
        return dict(available=False,reason='ILL_CONDITIONED_NORMAL_SYSTEM',start=start,end=end,condition=condition)
    coefficients=np.linalg.solve(system,x.T@y)
    if not np.isfinite(coefficients).all():
        return dict(available=False,reason='NONFINITE_FIT',start=start,end=end,condition=condition)
    return dict(available=True,reason='AVAILABLE',start=start,end=end,condition=condition,
                coefficients=coefficients,rank=int(np.linalg.matrix_rank(x)),
                state_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(coefficients[3:]/STATE_SCALE_M)))))

def predict_memory_response(fit, planned_actions, last_completed_displacement):
    u=np.asarray(planned_actions,dtype=np.float64)
    previous=np.asarray(last_completed_displacement,dtype=np.float64).copy()
    if not fit['available'] or u.ndim!=2 or u.shape[1:]!=(3,) or previous.shape!=(3,):
        raise ValueError('Invalid forecast input')
    if not np.isfinite(u).all() or not np.isfinite(previous).all():
        raise ValueError('Nonfinite forecast input')
    result=[]
    for action in u:
        previous=np.concatenate((action,previous/STATE_SCALE_M))@fit['coefficients']
        result.append(previous.copy())
    return np.asarray(result).reshape((-1,3))
