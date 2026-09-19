"""Dormant 6D-action to TCP-displacement response candidate.

This Q-only candidate adds rotation covariates to the completed-action response
fit. It is deliberately not wired into the active RAP policy path. Callers
must provide translation and rotation columns already expressed in one common,
explicit response space; this module performs no representation conversion.

The fitted matrix has shape (6, 3). Translation-only RAP perturbations use only
its first three rows, which is exactly equivalent to augmenting every
translation perturbation with a frozen zero rotation perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chunkamend.hardware.core.methods.rap_response import CONDITION_LIMIT
from chunkamend.hardware.core.methods.rap_response import MIN_SAMPLES
from chunkamend.hardware.core.methods.rap_response import RESPONSE_WINDOW
from chunkamend.hardware.core.methods.rap_response import RIDGE


Q6D_ACTION_REPRESENTATION = (
    "CALLER_SUPPLIED_COMMON_RESPONSE_SPACE__TRANSLATION3_ROTATION3__NO_IMPLICIT_NORMALIZATION"
)
Q6D_ACTION_DIM = 6
TCP_DISPLACEMENT_DIM = 3
Q6D_CANDIDATE_ROTATION_PERTURBATION = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SixDResponseFit:
    """Diagnostics and coefficients for the isolated Q 6x3 response fit."""

    matrix: np.ndarray | None
    available: bool
    reason: str
    action_representation: str = Q6D_ACTION_REPRESENTATION
    sample_count: int = 0
    window_start: int = 0
    window_end: int = 0
    condition_number: float = float("nan")
    residual_rmse: float = float("nan")
    rank: int = 0
    fit_age: int = 0

    @property
    def translation_intervention_matrix(self) -> np.ndarray | None:
        """Return the 3x3 adapter for a translation-only candidate delta.

        No active-path ResponseFit is returned: integration must remain an
        explicit later step because the full design rank is 6D and must not be
        silently relabelled as the current 3D response identity.
        """

        if not self.available or self.matrix is None:
            return None
        matrix = np.asarray(self.matrix)
        if matrix.shape != (Q6D_ACTION_DIM, TCP_DISPLACEMENT_DIM) or not np.isfinite(matrix).all():
            return None
        result = np.ascontiguousarray(matrix[:TCP_DISPLACEMENT_DIM]).copy()
        result.flags.writeable = False
        return result


def _unavailable(
    reason: str,
    *,
    action_representation: str = Q6D_ACTION_REPRESENTATION,
    sample_count: int = 0,
    window_start: int = 0,
    window_end: int = 0,
    condition_number: float = float("nan"),
    rank: int = 0,
    fit_age: int = 0,
) -> SixDResponseFit:
    return SixDResponseFit(
        matrix=None,
        available=False,
        reason=reason,
        action_representation=action_representation,
        sample_count=sample_count,
        window_start=window_start,
        window_end=window_end,
        condition_number=condition_number,
        residual_rmse=float("nan"),
        rank=rank,
        fit_age=fit_age,
    )


def fit_completed_six_d_response(
    actions: np.ndarray,
    tcp_displacements: np.ndarray,
    *,
    action_representation: str,
    min_samples: int = MIN_SAMPLES,
    ridge: float = RIDGE,
    rotation_ridge: float = RIDGE,
    condition_limit: float = CONDITION_LIMIT,
    fit_age: int = 0,
) -> SixDResponseFit:
    """Fit the latest 64 completed, chronological 6D action/TCP pairs.

    Actions must have columns translation[0:3], rotation[3:6] already expressed
    in the exact caller-declared common response space. Supplying a
    controller-bound representation label alone is insufficient: until an
    external normalization contract is frozen, only the exact identity constant
    above is accepted.
    """

    valid_fit_age = fit_age if isinstance(fit_age, int) and not isinstance(fit_age, bool) else 0
    if action_representation != Q6D_ACTION_REPRESENTATION:
        return _unavailable(
            "UNBOUND_Q6D_ACTION_REPRESENTATION",
            action_representation=(action_representation if isinstance(action_representation, str) else "INVALID"),
            fit_age=valid_fit_age,
        )
    if (
        min_samples != MIN_SAMPLES
        or ridge != RIDGE
        or rotation_ridge not in (RIDGE, 1e-3)
        or condition_limit != CONDITION_LIMIT
        or isinstance(fit_age, bool)
        or not isinstance(fit_age, int)
        or fit_age < 0
    ):
        return _unavailable("NONFROZEN_Q6D_RESPONSE_CONFIG", fit_age=valid_fit_age)

    try:
        action_array = np.asarray(actions, dtype=np.float64)
        displacement_array = np.asarray(tcp_displacements, dtype=np.float64)
    except (TypeError, ValueError):
        return _unavailable("NONFINITE_OR_SHAPE", fit_age=fit_age)
    if (
        action_array.ndim != 2
        or action_array.shape[1:] != (Q6D_ACTION_DIM,)
        or displacement_array.shape != (len(action_array), TCP_DISPLACEMENT_DIM)
        or not np.isfinite(action_array).all()
        or not np.isfinite(displacement_array).all()
    ):
        return _unavailable("NONFINITE_OR_SHAPE", fit_age=fit_age)

    pair_count = len(action_array)
    window_start = max(0, pair_count - RESPONSE_WINDOW)
    window_end = pair_count
    action_window = action_array[window_start:window_end]
    displacement_window = displacement_array[window_start:window_end]
    sample_count = len(action_window)
    rank = int(np.linalg.matrix_rank(action_window)) if sample_count else 0
    penalty = RIDGE * np.eye(Q6D_ACTION_DIM)
    if rotation_ridge != RIDGE:
        penalty[3:, 3:] = rotation_ridge * np.eye(3)
    system = action_window.T @ action_window + penalty
    condition_number = float(np.linalg.cond(system))
    diagnostics = {
        "sample_count": sample_count,
        "window_start": window_start,
        "window_end": window_end,
        "condition_number": condition_number,
        "rank": rank,
        "fit_age": fit_age,
    }
    if sample_count < MIN_SAMPLES:
        return _unavailable("INSUFFICIENT_COMPLETED_PAIRS", **diagnostics)
    if not np.isfinite(condition_number) or condition_number > CONDITION_LIMIT:
        return _unavailable("ILL_CONDITIONED_NORMAL_SYSTEM", **diagnostics)

    try:
        matrix = np.linalg.solve(system, action_window.T @ displacement_window)
    except np.linalg.LinAlgError:
        return _unavailable("NORMAL_SOLVE_FAILURE", **diagnostics)
    residual = action_window @ matrix - displacement_window
    residual_rmse = float(np.sqrt(np.mean(residual**2)))
    if not np.isfinite(matrix).all() or not np.isfinite(residual_rmse):
        return _unavailable("NONFINITE_FIT", **diagnostics)
    return SixDResponseFit(
        matrix=matrix,
        available=True,
        reason="AVAILABLE",
        residual_rmse=residual_rmse,
        **diagnostics,
    )
