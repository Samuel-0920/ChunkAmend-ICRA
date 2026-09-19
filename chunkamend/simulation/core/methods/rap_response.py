"""Frozen M2 causal normalized-action to TCP-displacement response fit."""

import numpy as np

from chunkamend.simulation.core.methods.m1_types import ResponseFit

from chunkamend.simulation.core.shared.feedback_window import RESPONSE_WINDOW
MIN_SAMPLES = 8
RIDGE = 1e-4
CONDITION_LIMIT = 1e8
PREQUENTIAL_MIN_PREDICTIONS = 8
EW_HALF_LIFE_COMPLETED_PAIRS = 2


def _unavailable(
    reason: str,
    *,
    sample_count: int = 0,
    window_start: int = 0,
    window_end: int = 0,
    condition_number: float = float("nan"),
    rank: int = 0,
    fit_age: int = 0,
) -> ResponseFit:
    return ResponseFit(
        matrix=None,
        available=False,
        reason=reason,
        sample_count=sample_count,
        window_start=window_start,
        window_end=window_end,
        condition_number=condition_number,
        residual_rmse=float("nan"),
        rank=rank,
        fit_age=fit_age,
    )


def fit_completed_translation_response(
    actions: np.ndarray,
    tcp_displacements: np.ndarray,
    *,
    min_samples: int = MIN_SAMPLES,
    ridge: float = RIDGE,
    condition_limit: float = CONDITION_LIMIT,
    fit_age: int = 0,
) -> ResponseFit:
    """Fit only the latest 64 completed chronological pairs.

    The primary M2 constants are deliberately rejected when changed, so an
    experimental response variant cannot silently retain the primary method
    identity. Window indices are half-open indices into the supplied completed
    pair sequence.
    """

    if (
        min_samples != MIN_SAMPLES
        or ridge != RIDGE
        or condition_limit != CONDITION_LIMIT
        or isinstance(fit_age, bool)
        or not isinstance(fit_age, int)
        or fit_age < 0
    ):
        valid_fit_age = fit_age if isinstance(fit_age, int) and not isinstance(fit_age, bool) else 0
        return _unavailable("NONPRIMARY_RESPONSE_CONFIG", fit_age=valid_fit_age)

    try:
        action_array = np.asarray(actions, dtype=np.float64)
        displacement_array = np.asarray(tcp_displacements, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return _unavailable("NONFINITE_OR_SHAPE", fit_age=fit_age)
    if (
        action_array.ndim != 2
        or action_array.shape[1:] != (3,)
        or displacement_array.shape != action_array.shape
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
    system = action_window.T @ action_window + RIDGE * np.eye(3)
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
    return ResponseFit(
        matrix=matrix,
        available=True,
        reason="AVAILABLE",
        residual_rmse=residual_rmse,
        **diagnostics,
    )


def fit_completed_translation_response_exponentially_weighted(
    actions: np.ndarray,
    tcp_displacements: np.ndarray,
    *,
    min_samples: int = MIN_SAMPLES,
    ridge: float = RIDGE,
    condition_limit: float = CONDITION_LIMIT,
    fit_age: int = 0,
    half_life_completed_pairs: int = EW_HALF_LIFE_COMPLETED_PAIRS,
) -> ResponseFit:
    """Fit a causal latest-64 response with one frozen recency mechanism.

    The newest completed pair has age zero and weight one before
    normalization.  Weights are normalized to sum to the current window size,
    preserving the primary ridge scale rather than introducing shrinkage as a
    second mechanism.  The caller supplies completed pairs only.
    """

    if (
        min_samples != MIN_SAMPLES
        or ridge != RIDGE
        or condition_limit != CONDITION_LIMIT
        or isinstance(fit_age, bool)
        or not isinstance(fit_age, int)
        or fit_age < 0
        or isinstance(half_life_completed_pairs, bool)
        or not isinstance(half_life_completed_pairs, int)
        or half_life_completed_pairs != EW_HALF_LIFE_COMPLETED_PAIRS
    ):
        valid_fit_age = fit_age if isinstance(fit_age, int) and not isinstance(fit_age, bool) else 0
        return _unavailable("NONFROZEN_EW_RESPONSE_CONFIG", fit_age=valid_fit_age)

    action_array = np.asarray(actions, dtype=np.float64)
    displacement_array = np.asarray(tcp_displacements, dtype=np.float64)
    if (
        action_array.ndim != 2
        or action_array.shape[1:] != (3,)
        or displacement_array.shape != action_array.shape
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
    diagnostics = {
        "sample_count": sample_count,
        "window_start": window_start,
        "window_end": window_end,
        "condition_number": float("nan"),
        "rank": rank,
        "fit_age": fit_age,
    }
    if sample_count < MIN_SAMPLES:
        return _unavailable("INSUFFICIENT_COMPLETED_PAIRS", **diagnostics)

    ages = np.arange(sample_count - 1, -1, -1, dtype=np.float64)
    weights = np.power(2.0, -ages / EW_HALF_LIFE_COMPLETED_PAIRS)
    weight_sum = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return _unavailable("NONFINITE_EW_WEIGHTS", **diagnostics)
    weights *= sample_count / weight_sum
    if (
        not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not np.isclose(float(np.sum(weights)), float(sample_count), rtol=0.0, atol=1e-12)
    ):
        return _unavailable("INVALID_EW_WEIGHTS", **diagnostics)

    weighted_actions = weights[:, None] * action_window
    system = action_window.T @ weighted_actions + RIDGE * np.eye(3)
    condition_number = float(np.linalg.cond(system))
    diagnostics["condition_number"] = condition_number
    if not np.isfinite(condition_number) or condition_number > CONDITION_LIMIT:
        return _unavailable("ILL_CONDITIONED_NORMAL_SYSTEM", **diagnostics)
    try:
        matrix = np.linalg.solve(
            system,
            action_window.T @ (weights[:, None] * displacement_window),
        )
    except np.linalg.LinAlgError:
        return _unavailable("NORMAL_SOLVE_FAILURE", **diagnostics)
    residual = action_window @ matrix - displacement_window
    residual_rmse = float(
        np.sqrt(
            np.sum(weights[:, None] * np.square(residual), dtype=np.float64)
            / (3.0 * float(np.sum(weights)))
        )
    )
    if not np.isfinite(matrix).all() or not np.isfinite(residual_rmse):
        return _unavailable("NONFINITE_FIT", **diagnostics)
    return ResponseFit(
        matrix=matrix,
        available=True,
        reason="AVAILABLE",
        residual_rmse=residual_rmse,
        **diagnostics,
    )


def fit_completed_translation_response_prequential(
    actions: np.ndarray,
    tcp_displacements: np.ndarray,
    *,
    min_samples: int = MIN_SAMPLES,
    ridge: float = RIDGE,
    condition_limit: float = CONDITION_LIMIT,
    fit_age: int = 0,
    minimum_predictions: int = PREQUENTIAL_MIN_PREDICTIONS,
) -> ResponseFit:
    """Fit the primary response and require causal one-step prediction evidence.

    Every validation residual is produced by a model fit only to chronologically
    earlier completed pairs.  The latest-64 final estimator remains identical to
    the primary estimator once the additional adequacy gate passes.
    """

    if minimum_predictions != PREQUENTIAL_MIN_PREDICTIONS:
        return _unavailable("NONPRIMARY_PREQUENTIAL_CONFIG", fit_age=fit_age)
    primary = fit_completed_translation_response(
        actions,
        tcp_displacements,
        min_samples=min_samples,
        ridge=ridge,
        condition_limit=condition_limit,
        fit_age=fit_age,
    )
    if not primary.available:
        return primary

    action_array = np.asarray(actions, dtype=np.float64)
    displacement_array = np.asarray(tcp_displacements, dtype=np.float64)
    start = max(0, len(action_array) - RESPONSE_WINDOW)
    action_window = action_array[start:]
    displacement_window = displacement_array[start:]
    errors: list[np.ndarray] = []
    for target in range(MIN_SAMPLES, len(action_window)):
        train_actions = action_window[:target]
        train_displacements = displacement_window[:target]
        if np.linalg.matrix_rank(train_actions) != 3:
            continue
        system = train_actions.T @ train_actions + RIDGE * np.eye(3)
        condition = float(np.linalg.cond(system))
        if not np.isfinite(condition) or condition > CONDITION_LIMIT:
            continue
        try:
            matrix = np.linalg.solve(system, train_actions.T @ train_displacements)
        except np.linalg.LinAlgError:
            continue
        error = action_window[target] @ matrix - displacement_window[target]
        if np.isfinite(error).all():
            errors.append(error)

    count = len(errors)
    prequential_rmse = (
        float(np.sqrt(np.mean(np.asarray(errors, dtype=np.float64) ** 2))) if errors else float("nan")
    )
    common = {
        "sample_count": primary.sample_count,
        "window_start": primary.window_start,
        "window_end": primary.window_end,
        "condition_number": primary.condition_number,
        "residual_rmse": primary.residual_rmse,
        "rank": primary.rank,
        "fit_age": primary.fit_age,
        "prequential_rmse": prequential_rmse,
        "prequential_count": count,
    }
    if primary.rank != 3:
        return ResponseFit(matrix=None, available=False, reason="FINAL_RESPONSE_RANK_DEFICIENT", **common)
    if count < PREQUENTIAL_MIN_PREDICTIONS or not np.isfinite(prequential_rmse):
        return ResponseFit(matrix=None, available=False, reason="INSUFFICIENT_PREQUENTIAL_PREDICTIONS", **common)
    return ResponseFit(matrix=primary.matrix.copy(), available=True, reason="AVAILABLE", **common)
