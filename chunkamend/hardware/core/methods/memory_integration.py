"""Opt-in C117 Q3 memory fit; original Q3/Q6 remain the default paths."""
from dataclasses import replace
import numpy as np
from chunkamend.hardware.core.methods.m1_types import ResponseFit, RESPONSE_PREDICTION_Q3_MEMORY
from chunkamend.hardware.core.methods.rap_response import fit_completed_translation_response
from chunkamend.hardware.core.methods.memory_response import fit_memory_response, STATE_SCALE_M
from chunkamend.hardware.core.methods.temporal_affine import lift_memory_response

def fit_memory_adapter(actions, displacements, *, preceding_displacement=None, state_ridge=1e-4):
    """Only completed history; unavailable memory uses the original static fit.

    The fallback matches the C115 forecasting experiment. It is explicit in
    memory_fit_diagnostics and never labelled as an available memory model.
    """
    if state_ridge not in (1e-4,1e-3):raise ValueError('Unfrozen integrated state ridge')
    try:
        fit = fit_memory_response(actions, displacements, preceding_displacement=preceding_displacement, state_ridge=state_ridge)
    except (ValueError, np.linalg.LinAlgError) as error:
        fit = dict(available=False, reason=type(error).__name__)
    if not fit['available']:
        static = fit_completed_translation_response(actions, displacements)
        return replace(static, memory_fit_diagnostics=dict(
            available=False, reason=fit['reason'], fallback='ORIGINAL_STATIC_Q3',
            fallback_available=static.available))
    u = np.asarray(actions, dtype=np.float64)
    d = np.asarray(displacements, dtype=np.float64)
    start, end = fit['start'], fit['end']
    lag = d[start-1:end-1] if preceding_displacement is None else np.concatenate((np.asarray(preceding_displacement)[None],d[:-1]),axis=0)
    x = np.concatenate((u[start:end], lag/STATE_SCALE_M), axis=1)
    residual = float(np.sqrt(np.mean((x @ fit['coefficients'] - d[start:end])**2)))
    if not np.isfinite(residual):
        static = fit_completed_translation_response(actions, displacements)
        return replace(static, memory_fit_diagnostics=dict(
            available=False, reason='NONFINITE_RESIDUAL', fallback='ORIGINAL_STATIC_Q3',
            fallback_available=static.available))
    temporal = lift_memory_response(fit, d[-1])
    b = fit['coefficients'][:3].copy(); b.setflags(write=False)
    return ResponseFit(matrix=b, available=True, reason='AVAILABLE',
        sample_count=end-start, window_start=start, window_end=end,
        condition_number=fit['condition'], residual_rmse=residual, rank=fit['rank'],
        prediction_semantics=RESPONSE_PREDICTION_Q3_MEMORY, temporal_response=temporal,
        memory_fit_diagnostics=dict(available=True, reason='AVAILABLE', fallback=None,
            coefficients=fit['coefficients'].tolist(), initial_displacement=d[-1].tolist(),
            state_spectral_radius=fit['state_spectral_radius'], state_scale_m=STATE_SCALE_M,
            window=64, ridge=1e-4, min_samples=8, condition_limit=1e8, intercept=False))


