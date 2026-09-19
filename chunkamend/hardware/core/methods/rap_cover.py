from chunkamend.hardware.core.methods import compiled_bridge
"""Pure M2 RAP-COVER execution projection and counterfactual-veto core."""

from dataclasses import dataclass
from dataclasses import replace

import numpy as np
from chunkamend.hardware.core.methods.numeric_dispatch import clip_scalar_f64

from chunkamend.hardware.core.methods.m1_types import RESPONSE_ALTERNATIVE_RAW
from chunkamend.hardware.core.methods.m1_types import RESPONSE_ALTERNATIVE_SELECTED
from chunkamend.hardware.core.methods.m1_types import RESPONSE_PREDICTION_Q6_ALTERNATIVE_OFFSETS
from chunkamend.hardware.core.methods.m1_types import RESPONSE_PREDICTION_Q3_MEMORY
from chunkamend.hardware.core.methods.m1_types import RapCandidateSpec
from chunkamend.hardware.core.methods.m1_types import RapCandidateTrace
from chunkamend.hardware.core.methods.m1_types import RapConfig
from chunkamend.hardware.core.methods.m1_types import RapDecision
from chunkamend.hardware.core.methods.m1_types import RapReason
from chunkamend.hardware.core.methods.m1_types import RapTelemetry
from chunkamend.hardware.core.methods.m1_types import valid_remaining_shape
from chunkamend.hardware.core.methods.m1_types import ResponseFit
from chunkamend.hardware.core.methods.m1_types import RiskFeature
from chunkamend.hardware.core.methods.m1_types import predict_response_displacements
from chunkamend.hardware.core.methods.m1_types import response_prediction_contract_valid
from chunkamend.hardware.core.methods.rap_projector import TERMINAL_CONSTRAINT_EXECUTED_INTERNAL
from chunkamend.hardware.core.methods.rap_projector import TERMINAL_CONSTRAINT_MODES
from chunkamend.hardware.core.methods.rap_projector import TERMINAL_CONSTRAINT_PREFIX_EXIT
from chunkamend.hardware.core.methods.rap_projector import ProjectorQueryContextCache
from chunkamend.hardware.core.methods.rap_projector import evaluate_physical_counterfactual
from chunkamend.hardware.core.methods.rap_projector import evaluate_physical_counterfactual_cached
from chunkamend.hardware.core.methods.rap_projector import physical_counterfactual_veto
from chunkamend.hardware.core.methods.rap_projector import project_execution_aware_candidate
from chunkamend.hardware.core.methods.rap_projector import project_execution_aware_candidate_cached
from chunkamend.hardware.core.methods.rap_projector import project_execution_aware_candidate_cached_converged_representation_repair
from chunkamend.hardware.core.methods.rap_projector import project_execution_aware_candidate_cached_dykstra
from chunkamend.hardware.core.methods.rap_projector import project_execution_aware_candidate_cached_representation_aware
from chunkamend.hardware.core.methods.rap_projector import project_execution_aware_candidate_cached_terminal_endpoint_reuse
from chunkamend.hardware.core.methods.rap_projector import project_execution_aware_candidate_cached_validator_only
from chunkamend.hardware.core.methods.rap_response import CONDITION_LIMIT
from chunkamend.hardware.core.methods.rap_response import MIN_SAMPLES

HORIZON = 15
CONTROL_PERIOD_SECONDS = 0.05
METHOD_ID = "RAP-COVER-EXECUTION-PROJECTION-CF-VETO"
OBJECTIVE_LEGACY_NEXT = "STRIDE_PLUS_ONE"
OBJECTIVE_EXECUTED_ONLY = "EXECUTED_ONLY"
SELECTION_ACTION = "ACTION_OBJECTIVE"
RANKING_ACTION = "ACTION_SCORE"
RANKING_ELIGIBLE_PHYSICAL = "ELIGIBLE_PHYSICAL_RELATIVE"
SELECTION_PHYSICAL = "PHYSICAL_RELATIVE"
SELECTION_TCP_HORIZON_ISJ = "PREDICTED_TCP_HORIZON_INTEGRATED_SQUARED_JERK"
REVERSAL_LEGACY = "LEGACY_MIXED"
REVERSAL_RESPONSE_TCP = "RESPONSE_TCP"
CANDIDATE_EVALUATION_FULL = "FULL_GRID"
CANDIDATE_EVALUATION_ZERO_FALLBACK = "GRID10_ZERO_ELIGIBLE_FULL_FALLBACK"
CANDIDATE_EVALUATION_TIERED_FALLBACK = "GRID10_AT_MOST_ONE_TIERED_FALLBACK"
CANDIDATE_EVALUATION_POLICIES = {
    CANDIDATE_EVALUATION_FULL,
    CANDIDATE_EVALUATION_ZERO_FALLBACK,
    CANDIDATE_EVALUATION_TIERED_FALLBACK,
}
PREPROJECTION_BOUND_DISABLED = "DISABLED"
PREPROJECTION_BOUND_BRIDGE_MIN_GAIN = "BRIDGE_MIN_GAIN_NECESSARY_CONDITION"
PREPROJECTION_BOUND_MODES = {
    PREPROJECTION_BOUND_DISABLED,
    PREPROJECTION_BOUND_BRIDGE_MIN_GAIN,
}
PREPROJECTION_BOUND_ABSOLUTE_MARGIN = 1e-9
PREPROJECTION_BOUND_RELATIVE_MARGIN = 1e-7
PROJECTOR_CONTEXT_CACHE_DISABLED = "DISABLED"
PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY = "QUERY_PREFIX_LAZY_READONLY"
PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE = (
    "QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE"
)
PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_DYKSTRA = (
    "QUERY_PREFIX_LAZY_READONLY_DYKSTRA_REPRESENTATION_AWARE"
)
PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_REPRESENTATION_AWARE = (
    "QUERY_PREFIX_LAZY_READONLY_REPRESENTATION_AWARE"
)
PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_CONVERGED_REPRESENTATION_REPAIR = (
    "QUERY_PREFIX_LAZY_READONLY_CONVERGED_REPRESENTATION_REPAIR"
)
PROJECTOR_CONTEXT_CACHE_MODES = {
    PROJECTOR_CONTEXT_CACHE_DISABLED,
    PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY,
    PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE,
    PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_DYKSTRA,
    PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_REPRESENTATION_AWARE,
    PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_CONVERGED_REPRESENTATION_REPAIR,
}
TERMINAL_PROJECTION_REF = "PROJECT_AND_VALIDATE"
TERMINAL_PROJECTION_VALIDATOR_ONLY = "VALIDATE_ONLY"
TERMINAL_PROJECTION_ROLES = {
    TERMINAL_PROJECTION_REF,
    TERMINAL_PROJECTION_VALIDATOR_ONLY,
}


@dataclass(frozen=True)
class BridgeResult:
    candidate: np.ndarray
    correction: np.ndarray
    lower_correction: np.ndarray
    upper_correction: np.ndarray
    ideal_correction: np.ndarray
    relative_cap_hit: bool
    correction_cap_hit: bool


@dataclass(frozen=True)
class _Candidate:
    chunk: np.ndarray
    score: float
    k: int
    alpha: float
    correction_l2: float | None = None
    objective: float | None = None
    trace_index: int = -1


def _direction(values: np.ndarray, epsilon: float) -> np.ndarray | None:
    vector = np.mean(values, axis=0)
    norm = np.linalg.norm(vector)
    return vector / norm if np.isfinite(vector).all() and norm > epsilon else None


def sustained_reversal(
    *,
    raw_chunk: np.ndarray,
    executed_action_history: np.ndarray,
    execution_stride: int,
    direction_window: int,
    reversal_cos_threshold: float,
    min_consistent_steps: int,
    velocity_epsilon: float,
    completed_tcp_history: np.ndarray | None = None,
    response: ResponseFit | None = None,
    direction_space: str = REVERSAL_LEGACY,
) -> bool:
    """Causal hard rollback retained from the preregistered RAP path."""
    raw, actions = np.asarray(raw_chunk), np.asarray(executed_action_history)
    if (
        isinstance(execution_stride, bool)
        or not isinstance(execution_stride, int)
        or execution_stride not in {3, 5, 9}
        or isinstance(direction_window, bool)
        or not isinstance(direction_window, int)
        or direction_window < 1
        or isinstance(min_consistent_steps, bool)
        or not isinstance(min_consistent_steps, int)
        or min_consistent_steps < 1
        or not np.isfinite((reversal_cos_threshold, velocity_epsilon)).all()
        or not -1 <= reversal_cos_threshold <= 1
        or velocity_epsilon <= 0
        or direction_space not in {REVERSAL_LEGACY, REVERSAL_RESPONSE_TCP}
    ):
        raise ValueError("invalid frozen reversal context")
    if (
        not valid_remaining_shape(raw, 7)
        or actions.ndim != 2
        or actions.shape[1] < 3
        or len(actions) < 1
        or not np.isfinite(raw).all()
        or not np.isfinite(actions).all()
    ):
        return False
    count = min(direction_window, execution_stride, len(raw))
    reference_values = actions[-max(count, min_consistent_steps) :, :3]
    candidate_values = raw[:count, :3]
    tcp = None if completed_tcp_history is None else np.asarray(completed_tcp_history)
    tcp_ready = bool(
        tcp is not None
        and tcp.ndim == 2
        and tcp.shape[1] == 3
        and len(tcp) >= 2
        and np.isfinite(tcp).all()
    )
    if direction_space == REVERSAL_RESPONSE_TCP:
        if not tcp_ready or response is None or not _response_ready(response):
            raise ValueError("response-space reversal requires causal TCP history and an available response")
        reference_values = np.diff(tcp, axis=0)[-max(count, min_consistent_steps) :]
        candidate_values = predict_response_displacements(
            raw,
            response,
            alternative=RESPONSE_ALTERNATIVE_RAW,
        )[:count]
    elif tcp_ready:
        # Preserve the frozen M4 mixed-coordinate behavior only in the reference branch.
        reference_values = np.diff(tcp, axis=0)[-max(count, min_consistent_steps) :]
    reference = _direction(reference_values, velocity_epsilon)
    direction = _direction(candidate_values, velocity_epsilon)
    if reference is None or direction is None:
        return False
    support = sum(
        np.dot(reference, value / np.linalg.norm(value)) < reversal_cos_threshold
        for value in candidate_values
        if np.linalg.norm(value) > velocity_epsilon
    )
    return float(np.dot(reference, direction)) < reversal_cos_threshold and support >= min_consistent_steps


def _expanded(weights: tuple[float, ...], count: int) -> np.ndarray:
    return np.asarray([weights[min(index, len(weights) - 1)] for index in range(count)], dtype=np.float64)


def local_translation_components(chunk_norm: np.ndarray, history_norm: np.ndarray, count: int) -> np.ndarray | None:
    """Causal normalized-translation second-difference RMS components."""
    chunk, history = np.asarray(chunk_norm), np.asarray(history_norm)
    if (
        not valid_remaining_shape(chunk, 7)
        or history.ndim != 2
        or history.shape[1] != 7
        or len(history) < 2
        or not 1 <= count <= len(chunk)
        or not np.issubdtype(chunk.dtype, np.floating)
        or not np.isfinite(chunk).all()
        or not np.isfinite(history).all()
    ):
        return None
    sequence = np.concatenate((history[-2:, :3], chunk[:, :3]), axis=0)
    second_difference = sequence[2 : count + 2] - 2 * sequence[1 : count + 1] + sequence[:count]
    components = np.sqrt(np.mean(second_difference**2, axis=1))
    return components if np.isfinite(components).all() else None


def local_translation_objective(
    chunk_norm: np.ndarray,
    history_norm: np.ndarray,
    *,
    count: int,
    objective_weights: tuple[float, ...],
) -> float | None:
    if (
        not objective_weights
        or not np.isfinite(objective_weights).all()
        or any(weight <= 0 for weight in objective_weights)
    ):
        return None
    components = local_translation_components(chunk_norm, history_norm, count)
    if components is None:
        return None
    weights = _expanded(objective_weights, count)
    value = float(np.sqrt(np.sum(weights * components**2) / np.sum(weights)))
    return value if np.isfinite(value) else None


def _bounded_cast(
    raw: np.ndarray, correction: np.ndarray, lower: np.ndarray, upper: np.ndarray, *, prefix_length: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    candidate = raw.copy()
    raw64 = raw[:prefix_length, :3].astype(np.float64)
    cast = (raw64 + correction).astype(raw.dtype)
    for _ in range(3):
        actual = cast.astype(np.float64) - raw64
        below, above = actual < lower, actual > upper
        if not np.any(below | above):
            candidate[:prefix_length, :3] = cast
            return candidate, actual
        cast = np.where(below, np.nextafter(cast, np.asarray(np.inf, dtype=raw.dtype)), cast)
        cast = np.where(above, np.nextafter(cast, np.asarray(-np.inf, dtype=raw.dtype)), cast)
    return None, None


def bridge_prefix(
    *,
    raw_norm: np.ndarray,
    history_norm: np.ndarray,
    prefix_length: int,
    objective_count: int,
    alpha: float,
    transition_weights: tuple[float, ...],
    objective_weights: tuple[float, ...],
    relative_cap: float,
    correction_cap: float | None,
    allow_full_executed_prefix: bool = False,
    reference_targets: np.ndarray | None = None,
    reference_rho: float = 1.0,
) -> BridgeResult | None:
    """Bounded minimum-jerk bridge with the exact directional correction box."""
    raw_input = np.asarray(raw_norm)
    raw = raw_input.astype(np.float64, copy=False)
    history = np.asarray(history_norm, dtype=np.float64)
    if (
        not np.issubdtype(raw_input.dtype, np.floating)
        or not valid_remaining_shape(raw, 7)
        or history.ndim != 2
        or history.shape[1] < 3
        or len(history) < 2
        or isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or not (1 <= prefix_length < len(raw) or (allow_full_executed_prefix and prefix_length == len(raw)))
        or not isinstance(allow_full_executed_prefix, bool)
        or not (
            prefix_length < objective_count <= len(raw)
            or (
                allow_full_executed_prefix
                and prefix_length == objective_count <= len(raw)
            )
        )
        or not 0 < alpha <= 1
        or len(transition_weights) != prefix_length
        or not objective_weights
        or relative_cap < 0
        or (
            correction_cap is not None
            and (
                isinstance(correction_cap, bool)
                or not isinstance(correction_cap, (int, float, np.integer, np.floating))
                or correction_cap <= 0
                or not np.isfinite(correction_cap)
            )
        )
        or not np.isfinite([relative_cap, *transition_weights, *objective_weights]).all()
        or any(value < 0 for value in transition_weights)
        or any(value <= 0 for value in objective_weights)
        or not np.isfinite(raw).all()
        or not np.isfinite(history).all()
    ):
        return None

    targets = None
    if reference_targets is not None:
        targets = np.asarray(reference_targets, dtype=np.float64)
        if (targets.ndim != 2 or targets.shape[1] != 3 or len(targets) not in (1, 2)
                or len(targets) > prefix_length or not np.isfinite(targets).all()
                or isinstance(reference_rho, bool) or not isinstance(reference_rho, (int, float, np.integer, np.floating))
                or not np.isfinite(reference_rho) or reference_rho <= 0):
            raise ValueError('Invalid old RAW-tail reference')
    sequence = np.concatenate((history[-2:, :3], raw[:, :3]), axis=0)
    residual = sequence[2 : objective_count + 2] - 2 * sequence[1 : objective_count + 1] + sequence[:objective_count]
    design = np.zeros((objective_count, prefix_length), dtype=np.float64)
    for row in range(objective_count):
        for column, coefficient in ((row, 1.0), (row - 1, -2.0), (row - 2, 1.0)):
            if 0 <= column < prefix_length:
                design[row, column] += coefficient
    sqrt_weights = np.sqrt(_expanded(objective_weights, objective_count))
    weighted_design = design * sqrt_weights[:, None]
    taper = np.asarray(transition_weights, dtype=np.float64)
    correction = np.zeros((prefix_length, 3), dtype=np.float64)
    ideal_correction = np.zeros_like(correction)
    lower = np.zeros_like(correction)
    upper = np.zeros_like(correction)

    hessian = weighted_design.T @ weighted_design
    if np.any(np.diag(hessian) <= 0) or not np.isfinite(hessian).all():
        return None
    solve_hessian = hessian
    if targets is not None:
        # Preserve the original R0 ideal and correction box. The old-tail term
        # changes only the bounded solve, exactly as in historical R1/R2.
        mu = float(reference_rho) * float(np.sum(_expanded(objective_weights, objective_count))) / len(targets)
        solve_hessian = hessian.copy()
        solve_hessian[:len(targets), :len(targets)] += mu * np.eye(len(targets))
        from chunkamend.hardware.core.methods import old_tail_objective
        old_tail_objective.record(targets, mu)
    for dimension in range(3):
        weighted_residual = residual[:, dimension] * sqrt_weights
        try:
            ideal = np.linalg.lstsq(weighted_design, -weighted_residual, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        taper_limit = np.abs(ideal) * taper * alpha
        relative_limit = np.abs(ideal) * relative_cap
        limit = np.minimum(taper_limit, relative_limit)
        if correction_cap is not None:
            limit = np.minimum(limit, correction_cap)
        lower_dimension = np.minimum(0.0, np.sign(ideal) * limit)
        upper_dimension = np.maximum(0.0, np.sign(ideal) * limit)
        gradient = weighted_design.T @ weighted_residual
        if targets is not None:
            gradient[:len(targets)] += mu * (raw[:len(targets), dimension] - targets[:, dimension])
        compiled, solution = compiled_bridge.solve(solve_hessian, gradient, lower_dimension, upper_dimension)
        if compiled:
            if solution is None:
                return None
        else:
            solution = np.zeros(prefix_length, dtype=np.float64)
            for _ in range(10_000):
                previous = solution.copy()
                for index in range(prefix_length):
                    partial = gradient[index] + np.dot(solve_hessian[index], solution) - solve_hessian[index, index] * solution[index]
                    solution[index] = clip_scalar_f64(
                        -partial / solve_hessian[index, index], lower_dimension[index], upper_dimension[index]
                    )
                if np.max(np.abs(solution - previous)) <= 1e-12:
                    break
            else:
                return None
        ideal_correction[:, dimension] = ideal
        correction[:, dimension] = solution
        lower[:, dimension] = lower_dimension
        upper[:, dimension] = upper_dimension

    candidate, actual = _bounded_cast(raw_input, correction, lower, upper, prefix_length=prefix_length)
    if candidate is None or actual is None or not np.isfinite(candidate).all():
        return None
    ideal_abs = np.abs(ideal_correction)
    solved_abs = np.abs(correction)
    relative_limit = ideal_abs * relative_cap
    taper_limit = ideal_abs * taper[:, None] * alpha
    relative_active = relative_limit <= taper_limit + 1e-15
    if correction_cap is not None:
        relative_active &= relative_limit <= correction_cap + 1e-15
        correction_active = (correction_cap <= taper_limit + 1e-15) & (
            correction_cap <= relative_limit + 1e-15
        )
    else:
        correction_active = np.zeros_like(relative_active, dtype=bool)
    relative_hit = bool(
        np.any(relative_active & (relative_limit > 0) & np.isclose(solved_abs, relative_limit, rtol=0, atol=1e-10))
    )
    correction_hit = bool(
        correction_cap is not None
        and np.any(correction_active & np.isclose(solved_abs, correction_cap, rtol=0, atol=1e-10))
    )
    return BridgeResult(
        candidate=candidate,
        correction=actual,
        lower_correction=lower,
        upper_correction=upper,
        ideal_correction=ideal_correction,
        relative_cap_hit=relative_hit,
        correction_cap_hit=correction_hit,
    )


def time_aligned_plan_disagreement(
    previous_chunk: np.ndarray | None,
    current_chunk: np.ndarray,
    *,
    previous_executed_count: int,
    logical_timestep: int,
) -> RiskFeature:
    current = np.asarray(current_chunk)

    def unavailable() -> RiskFeature:
        return RiskFeature(
            "time_aligned_previous_plan_disagreement",
            None,
            (),
            "normalized_action_translation",
            logical_timestep,
            "UNAVAILABLE",
        )

    if previous_chunk is None or previous_executed_count < 0:
        return unavailable()
    previous = np.asarray(previous_chunk)
    overlap = min(HORIZON - previous_executed_count, HORIZON)
    if (
        previous.shape != (HORIZON, 7)
        or current.shape != (HORIZON, 7)
        or overlap <= 0
        or not np.isfinite(previous).all()
        or not np.isfinite(current).all()
    ):
        return unavailable()
    value = float(
        np.sqrt(
            np.mean(
                (previous[previous_executed_count : previous_executed_count + overlap, :3] - current[:overlap, :3]) ** 2
            )
        )
    )
    return RiskFeature(
        "time_aligned_previous_plan_disagreement",
        value,
        (),
        "normalized_action_translation",
        logical_timestep,
        "CANDIDATE_SELECTION",
    )


def causal_risk_features(
    *,
    executed_tcp: np.ndarray,
    raw_chunk: np.ndarray,
    previous_raw_chunk: np.ndarray | None,
    previous_executed_count: int,
    logical_timestep: int,
    tcp_translation_units: str,
) -> dict[str, RiskFeature]:
    if not tcp_translation_units:
        raise ValueError("tcp_translation_units is required")
    tcp, raw = np.asarray(executed_tcp), np.asarray(raw_chunk)
    common = {
        "shape": (),
        "logical_timestamp": logical_timestep,
        "latest_availability": "CANDIDATE_SELECTION",
    }
    boundary = (
        None
        if tcp.ndim != 2 or len(tcp) < 3 or not np.isfinite(tcp).all()
        else float(np.linalg.norm(tcp[-1, :3] - 2 * tcp[-2, :3] + tcp[-3, :3]))
    )
    next_component = (
        None
        if raw.shape != (HORIZON, 7) or not np.isfinite(raw).all()
        else float(np.linalg.norm(raw[2, :3] - 2 * raw[1, :3] + raw[0, :3]))
    )
    return {
        "boundary_translation_second_difference": RiskFeature(
            "boundary_translation_second_difference", boundary, units=tcp_translation_units, **common
        ),
        "next_in_chunk_translation_second_difference": RiskFeature(
            "next_in_chunk_translation_second_difference",
            next_component,
            units="normalized_action_translation",
            **common,
        ),
        "time_aligned_previous_plan_disagreement": time_aligned_plan_disagreement(
            previous_raw_chunk,
            raw,
            previous_executed_count=previous_executed_count,
            logical_timestep=logical_timestep,
        ),
    }


def _select(raw: np.ndarray, no_op_score: float, candidates: list[_Candidate]) -> RapDecision:
    choices = [_Candidate(raw.copy(), no_op_score, 0, 0.0, 0.0), *candidates]
    best = min(
        choices,
        key=lambda item: (
            item.score,
            float(np.linalg.norm(item.chunk - raw)) if item.correction_l2 is None else item.correction_l2,
            item.k,
            item.alpha,
        ),
    )
    return RapDecision(
        best.chunk,
        best.k,
        best.alpha,
        RapReason.NO_OP if best.k == 0 else RapReason.SELECTED,
    )


def _response_ready(
    response: ResponseFit,
    *,
    require_full_rank: bool = False,
    minimum_prequential_predictions: int = 0,
) -> bool:
    maximum_rank = (
        6
        if response.prediction_semantics in {RESPONSE_PREDICTION_Q6_ALTERNATIVE_OFFSETS, RESPONSE_PREDICTION_Q3_MEMORY}
        else 3
    )
    return bool(
        response.available
        and response.reason == "AVAILABLE"
        and response_prediction_contract_valid(response)
        and response.sample_count >= MIN_SAMPLES
        and response.window_start >= 0
        and response.window_end - response.window_start == response.sample_count
        and np.isfinite(response.condition_number)
        and response.condition_number <= CONDITION_LIMIT
        and np.isfinite(response.residual_rmse)
        and (response.rank == maximum_rank if require_full_rank else 0 <= response.rank <= maximum_rank)
        and response.fit_age >= 0
        and (
            minimum_prequential_predictions == 0
            or (
                response.prequential_count >= minimum_prequential_predictions
                and np.isfinite(response.prequential_rmse)
                and response.prequential_rmse >= 0
            )
        )
    )


def _rows(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in np.asarray(value))


def _values(value: np.ndarray | None) -> tuple[float, ...]:
    return () if value is None else tuple(float(item) for item in value)


def physical_relative_objective(
    raw_values: np.ndarray,
    candidate_values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Dimensionless ratio of weighted physical continuity magnitudes.

    The aggregation happens before division.  This makes a component that is
    zero in both alternatives neutral instead of spuriously treating 0/0 as a
    perfect component-level improvement.
    """

    raw = np.asarray(raw_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if (
        raw.shape != candidate.shape
        or raw.shape != weight.shape
        or raw.ndim != 1
        or raw.size == 0
        or not np.isfinite(raw).all()
        or not np.isfinite(candidate).all()
        or not np.isfinite(weight).all()
        or np.any(raw < 0)
        or np.any(candidate < 0)
        or np.any(weight <= 0)
    ):
        return float("inf")
    raw_objective = float(np.dot(weight, raw) / np.sum(weight))
    candidate_objective = float(np.dot(weight, candidate) / np.sum(weight))
    if raw_objective <= 1e-12:
        return float("inf")
    return candidate_objective / raw_objective


def predicted_tcp_horizon_integrated_squared_jerk(
    chunk: np.ndarray,
    observed_displacements: np.ndarray,
    response: ResponseFit,
    *,
    control_period_seconds: float = CONTROL_PERIOD_SECONDS,
    alternative: str = RESPONSE_ALTERNATIVE_RAW,
) -> float:
    """Causal full-horizon TCP ISJ predicted from displacement increments."""

    value = np.asarray(chunk)
    observed = np.asarray(observed_displacements, dtype=np.float64)
    if (
        not valid_remaining_shape(value, 7)
        or not np.issubdtype(value.dtype, np.floating)
        or observed.shape != (2, 3)
        or not response.available
        or response.reason != "AVAILABLE"
        or control_period_seconds != CONTROL_PERIOD_SECONDS
        or not np.isfinite(value).all()
        or not np.isfinite(observed).all()
        or not response_prediction_contract_valid(response)
    ):
        raise ValueError("invalid predicted TCP horizon jerk inputs")
    predicted_displacements = predict_response_displacements(
        value,
        response,
        alternative=alternative,
    )
    displacement_sequence = np.concatenate((observed, predicted_displacements), axis=0)
    second_difference = (
        displacement_sequence[2:]
        - 2.0 * displacement_sequence[1:-1]
        + displacement_sequence[:-2]
    )
    with np.errstate(over="ignore", invalid="ignore"):
        objective = float(
            np.sum(np.square(second_difference), dtype=np.float64)
            / control_period_seconds**5
        )
    if not np.isfinite(objective) or objective < 0.0:
        raise ValueError("nonfinite predicted TCP horizon jerk objective")
    return objective


def paired_predicted_tcp_horizon_jerk_ratio(
    raw_objective: float,
    candidate_objective: float,
) -> float:
    """Dimensionless paired calibration after aggregate horizon/axis summation."""

    if (
        isinstance(raw_objective, bool)
        or isinstance(candidate_objective, bool)
        or not np.isfinite((raw_objective, candidate_objective)).all()
        or raw_objective < 0.0
        or candidate_objective < 0.0
    ):
        return float("inf")
    if raw_objective == 0.0:
        return 1.0 if candidate_objective == 0.0 else float("inf")
    return float(candidate_objective / raw_objective)


def _candidate_tier(policy: str, alpha: float) -> int:
    """Return the frozen evaluation tier; FULL_GRID preserves legacy ordering."""
    if policy == CANDIDATE_EVALUATION_FULL:
        return 0
    if alpha in {0.1, 0.2}:
        return 0
    if policy == CANDIDATE_EVALUATION_TIERED_FALLBACK and alpha == 0.05:
        return 1
    if alpha in {0.01, 0.02, 0.05}:
        return 1 if policy == CANDIDATE_EVALUATION_ZERO_FALLBACK else 2
    raise ValueError("adaptive candidate policy received a non-M4 alpha")


def _evaluate_candidate_tier(policy: str, tier: int, eligible_count: int) -> bool:
    """Freeze a whole tier from the eligible count at its entry boundary."""
    if policy == CANDIDATE_EVALUATION_FULL or tier == 0:
        return True
    if policy == CANDIDATE_EVALUATION_ZERO_FALLBACK:
        return eligible_count == 0
    if policy == CANDIDATE_EVALUATION_TIERED_FALLBACK:
        return eligible_count <= 1 if tier == 1 else eligible_count == 0
    raise ValueError("invalid adaptive candidate evaluation policy")


def preprojection_min_gain_bound_fails(
    raw_objective: float,
    bridge_objective: float,
    min_gain: float,
) -> bool:
    """Conservatively reject a candidate before physical projection.

    ``bridge_prefix`` minimizes the frozen action objective inside the exact
    correction box later passed to the physical projector.  The projector is
    constrained to that same box, so its objective cannot improve on the
    bridge objective.  A strict numerical margin keeps missing/nonfinite and
    threshold-adjacent cases on the existing fail-open TIERED path.
    """

    if (
        isinstance(raw_objective, bool)
        or isinstance(bridge_objective, bool)
        or isinstance(min_gain, bool)
        or not np.isfinite((raw_objective, bridge_objective, min_gain)).all()
        or raw_objective <= 0.0
        or bridge_objective < 0.0
        or not 0.0 <= min_gain <= 1.0
    ):
        return False
    threshold = raw_objective * (1.0 - min_gain)
    margin = max(
        PREPROJECTION_BOUND_ABSOLUTE_MARGIN,
        abs(raw_objective) * PREPROJECTION_BOUND_RELATIVE_MARGIN,
    )
    return bool(bridge_objective > threshold + margin)


def _validate_ranking_penalty(value, ranking_mode):
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value) or value < 0
        or ranking_mode != RANKING_ELIGIBLE_PHYSICAL
    ):
        raise ValueError("invalid candidate ranking correction penalty")


def select_eligible_candidate(candidates, traces, ranking_mode, correction_penalty=None):
    """Rank an already admitted set; do not change eligibility or RAW fallback."""
    _validate_ranking_penalty(correction_penalty, ranking_mode)
    def action_key(item):
        return (item.score, item.correction_l2, item.k, item.alpha)
    if ranking_mode == RANKING_ELIGIBLE_PHYSICAL and all(
        np.isfinite(traces[item.trace_index].physical_relative_objective) for item in candidates
    ):
        if correction_penalty is not None and correction_penalty > 0:
            fractions = [traces[item.trace_index].correction_fraction_of_cap for item in candidates]
            if not all(np.isfinite(v) and v >= 0 for v in fractions):
                return min(candidates, key=action_key)
            scores = {item.trace_index: float(traces[item.trace_index].physical_relative_objective)
                      + float(correction_penalty) * float(traces[item.trace_index].correction_fraction_of_cap)
                      for item in candidates}
            if not all(np.isfinite(v) for v in scores.values()):
                return min(candidates, key=action_key)
            return min(candidates, key=lambda item: (scores[item.trace_index], *action_key(item)))
        return min(candidates, key=lambda item: (
            traces[item.trace_index].physical_relative_objective, *action_key(item)
        ))
    return min(candidates, key=action_key)


class RapCoverCore:
    def __init__(
        self,
        config: RapConfig,
        *,
        method_identity: str = METHOD_ID,
        objective_count_mode: str = OBJECTIVE_LEGACY_NEXT,
        selection_mode: str = SELECTION_ACTION,
        candidate_ranking_mode: str = RANKING_ACTION,
        candidate_ranking_correction_penalty: float | None = None,
        reversal_space: str = REVERSAL_LEGACY,
        uncertainty_scale: float | None = None,
        minimum_prequential_predictions: int = 0,
        candidate_evaluation_policy: str = CANDIDATE_EVALUATION_FULL,
        preprojection_objective_bound: str = PREPROJECTION_BOUND_DISABLED,
        projector_context_cache_mode: str = PROJECTOR_CONTEXT_CACHE_DISABLED,
        terminal_constraint_mode: str = TERMINAL_CONSTRAINT_PREFIX_EXIT,
        terminal_projection_role: str = TERMINAL_PROJECTION_REF,
        terminal_veto: bool = True,
    ) -> None:
        if (
            type(terminal_veto) is not bool
            or (not terminal_veto and (terminal_projection_role != TERMINAL_PROJECTION_VALIDATOR_ONLY
                or terminal_constraint_mode != TERMINAL_CONSTRAINT_EXECUTED_INTERNAL))
            or not isinstance(method_identity, str)
            or not method_identity
            or objective_count_mode not in {OBJECTIVE_LEGACY_NEXT, OBJECTIVE_EXECUTED_ONLY}
            or selection_mode
            not in {SELECTION_ACTION, SELECTION_PHYSICAL, SELECTION_TCP_HORIZON_ISJ}
            or candidate_ranking_mode not in {RANKING_ACTION, RANKING_ELIGIBLE_PHYSICAL}
            or (candidate_ranking_mode != RANKING_ACTION and selection_mode != SELECTION_ACTION)
            or reversal_space not in {REVERSAL_LEGACY, REVERSAL_RESPONSE_TCP}
            or (
                uncertainty_scale is not None
                and (not np.isfinite(uncertainty_scale) or uncertainty_scale <= 0)
            )
            or isinstance(minimum_prequential_predictions, bool)
            or not isinstance(minimum_prequential_predictions, int)
            or minimum_prequential_predictions < 0
            or ((uncertainty_scale is None) != (minimum_prequential_predictions == 0))
            or candidate_evaluation_policy not in CANDIDATE_EVALUATION_POLICIES
            or preprojection_objective_bound not in PREPROJECTION_BOUND_MODES
            or projector_context_cache_mode not in PROJECTOR_CONTEXT_CACHE_MODES
            or terminal_constraint_mode not in TERMINAL_CONSTRAINT_MODES
            or terminal_projection_role not in TERMINAL_PROJECTION_ROLES
            or (
                terminal_constraint_mode == TERMINAL_CONSTRAINT_EXECUTED_INTERNAL
                and projector_context_cache_mode
                != PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE
            )
            or (
                preprojection_objective_bound != PREPROJECTION_BOUND_DISABLED
                and selection_mode != SELECTION_ACTION
            )
        ):
            raise ValueError("invalid COVER variant contract")
        self.config = config
        self.method_identity = method_identity
        self.objective_count_mode = objective_count_mode
        self.selection_mode = selection_mode
        self.candidate_ranking_mode = candidate_ranking_mode
        _validate_ranking_penalty(candidate_ranking_correction_penalty, candidate_ranking_mode)
        self.candidate_ranking_correction_penalty = candidate_ranking_correction_penalty
        self.reversal_space = reversal_space
        self.uncertainty_scale = uncertainty_scale
        self.minimum_prequential_predictions = minimum_prequential_predictions
        self.candidate_evaluation_policy = candidate_evaluation_policy
        self.preprojection_objective_bound = preprojection_objective_bound
        self.projector_context_cache_mode = projector_context_cache_mode
        self.terminal_constraint_mode = terminal_constraint_mode
        self.terminal_projection_role = terminal_projection_role
        self.terminal_veto = terminal_veto

    def _decision(
        self,
        raw: np.ndarray,
        *,
        selected: np.ndarray | None,
        k: int,
        alpha: float,
        reason: RapReason,
        raw_objective: float,
        traces: list[RapCandidateTrace],
        response: ResponseFit,
        risk_eligible: bool,
        risk_eligibility_ablated: bool,
        reversal_rollback: bool,
        projector_cache: ProjectorQueryContextCache | None = None,
    ) -> RapDecision:
        chosen = raw.copy() if selected is None else selected.copy()
        cache_receipt = (
            projector_cache.receipt()
            if projector_cache is not None
            else {
                "initialized": False,
                "builds_by_k": (),
                "hits_by_k": (),
                "gram_build_count": 0,
                "failure_by_k": (),
            }
        )
        telemetry = RapTelemetry(
            method_id=self.method_identity,
            raw_chunk=raw.copy(),
            selected_chunk=chosen.copy(),
            selected_reason=reason.value,
            raw_objective=raw_objective,
            candidate_traces=tuple(traces),
            response_sample_count=response.sample_count,
            response_window_start=response.window_start,
            response_window_end=response.window_end,
            response_rank=response.rank,
            response_condition_number=response.condition_number,
            response_residual_rmse=response.residual_rmse,
            risk_eligible=risk_eligible,
            risk_eligibility_ablated=risk_eligibility_ablated,
            reversal_rollback=reversal_rollback,
            selection_mode=self.selection_mode,
            candidate_ranking_mode=self.candidate_ranking_mode,
            projector_context_cache_mode=self.projector_context_cache_mode,
            projector_context_cache_initialized=bool(cache_receipt["initialized"]),
            projector_context_builds_by_k=tuple(cache_receipt["builds_by_k"]),
            projector_context_hits_by_k=tuple(cache_receipt["hits_by_k"]),
            projector_context_gram_build_count=int(cache_receipt["gram_build_count"]),
            projector_context_failure_by_k=tuple(cache_receipt["failure_by_k"]),
        )
        return RapDecision(chosen, k, alpha, reason, telemetry)

    def apply(
        self,
        raw_chunk: np.ndarray,
        *,
        stride: int,
        candidate_specs: list[RapCandidateSpec],
        completed_actions: np.ndarray,
        response: ResponseFit,
        observed_displacements: np.ndarray | None = None,
        previous_action: np.ndarray | None = None,
        risk_eligible: bool = True,
        ablate_risk_trust_eligibility: bool = False,
        reversal_context: dict | None = None,
        reference_targets: np.ndarray | None = None,
        reference_rho: float = 1.0,
    ) -> RapDecision:
        del previous_action  # The primary physical projector uses observed TCP displacements instead.
        raw_input = np.asarray(raw_chunk)
        if not valid_remaining_shape(raw_input, 7) or not np.issubdtype(raw_input.dtype, np.floating):
            return RapDecision(raw_input.copy(), 0, 0.0, RapReason.NONFINITE_INPUT)
        raw = raw_input.copy()
        if not np.isfinite(raw).all():
            return RapDecision(raw, 0, 0.0, RapReason.NONFINITE_INPUT)
        traces: list[RapCandidateTrace] = []
        raw_objective = float("nan")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride not in {3, 5, 9}:
            return self._decision(
                raw,
                selected=None,
                k=0,
                alpha=0.0,
                reason=RapReason.NO_ELIGIBLE_CANDIDATE,
                raw_objective=raw_objective,
                traces=traces,
                response=response,
                risk_eligible=risk_eligible,
                risk_eligibility_ablated=ablate_risk_trust_eligibility,
                reversal_rollback=False,
            )
        if not risk_eligible and not ablate_risk_trust_eligibility:
            return self._decision(
                raw,
                selected=None,
                k=0,
                alpha=0.0,
                reason=RapReason.NO_ELIGIBLE_CANDIDATE,
                raw_objective=raw_objective,
                traces=traces,
                response=response,
                risk_eligible=False,
                risk_eligibility_ablated=False,
                reversal_rollback=False,
            )
        require_full_rank = self.uncertainty_scale is not None
        reversal_rollback = False
        if reversal_context:
            try:
                reversal_rollback = sustained_reversal(
                    raw_chunk=raw,
                    executed_action_history=np.asarray(reversal_context["executed_action_history"]),
                    execution_stride=stride,
                    direction_window=reversal_context["direction_window"],
                    reversal_cos_threshold=reversal_context["reversal_cos_threshold"],
                    min_consistent_steps=reversal_context["min_consistent_steps"],
                    velocity_epsilon=reversal_context["velocity_epsilon"],
                    completed_tcp_history=reversal_context.get("completed_tcp_history"),
                    response=response,
                    direction_space=self.reversal_space,
                )
            except (KeyError, TypeError, ValueError):
                reversal_rollback = True
        if reversal_rollback:
            return self._decision(
                raw,
                selected=None,
                k=0,
                alpha=0.0,
                reason=RapReason.REVERSAL_ROLLBACK,
                raw_objective=raw_objective,
                traces=traces,
                response=response,
                risk_eligible=risk_eligible,
                risk_eligibility_ablated=ablate_risk_trust_eligibility,
                reversal_rollback=True,
            )
        if not _response_ready(
            response,
            require_full_rank=require_full_rank,
            minimum_prequential_predictions=self.minimum_prequential_predictions,
        ):
            return self._decision(
                raw,
                selected=None,
                k=0,
                alpha=0.0,
                reason=RapReason.RESPONSE_UNAVAILABLE,
                raw_objective=raw_objective,
                traces=traces,
                response=response,
                risk_eligible=risk_eligible,
                risk_eligibility_ablated=ablate_risk_trust_eligibility,
                reversal_rollback=False,
            )
        observed = np.asarray(observed_displacements)
        if observed.shape != (2, 3) or not np.isfinite(observed).all():
            return self._decision(
                raw,
                selected=None,
                k=0,
                alpha=0.0,
                reason=RapReason.PROJECTOR_UNAVAILABLE,
                raw_objective=raw_objective,
                traces=traces,
                response=response,
                risk_eligible=risk_eligible,
                risk_eligibility_ablated=ablate_risk_trust_eligibility,
                reversal_rollback=False,
            )

        objective_count = (
            min(len(raw), stride + 1)
            if self.objective_count_mode == OBJECTIVE_LEGACY_NEXT
            else min(len(raw), stride)
        )
        raw_components = local_translation_components(raw, completed_actions, objective_count)
        raw_score = local_translation_objective(
            raw,
            completed_actions,
            count=objective_count,
            objective_weights=self.config.objective_weights,
        )
        if raw_score is None or raw_components is None:
            return self._decision(
                raw,
                selected=None,
                k=0,
                alpha=0.0,
                reason=RapReason.NO_ELIGIBLE_CANDIDATE,
                raw_objective=raw_objective,
                traces=traces,
                response=response,
                risk_eligible=risk_eligible,
                risk_eligibility_ablated=ablate_risk_trust_eligibility,
                reversal_rollback=False,
            )
        raw_objective = raw_score
        raw_predicted_tcp_horizon_isj = float("nan")
        if self.selection_mode == SELECTION_TCP_HORIZON_ISJ:
            try:
                raw_predicted_tcp_horizon_isj = (
                    predicted_tcp_horizon_integrated_squared_jerk(
                        raw,
                        observed,
                        response,
                    )
                )
            except ValueError:
                return self._decision(
                    raw,
                    selected=None,
                    k=0,
                    alpha=0.0,
                    reason=RapReason.NO_ELIGIBLE_CANDIDATE,
                    raw_objective=raw_objective,
                    traces=traces,
                    response=response,
                    risk_eligible=risk_eligible,
                    risk_eligibility_ablated=ablate_risk_trust_eligibility,
                    reversal_rollback=False,
                )
        candidates: list[_Candidate] = []
        eligible_by_alpha: dict[float, list[_Candidate]] = {}
        projector_failure = False
        physical_veto_failure = False
        projector_cache: ProjectorQueryContextCache | None = None

        if self.candidate_evaluation_policy == CANDIDATE_EVALUATION_FULL:
            ordered_specs = sorted(
                candidate_specs,
                key=lambda item: (item.k, item.alpha, item.transition_weights, item.relative_cap),
            )
        else:
            ordered_specs = sorted(
                candidate_specs,
                key=lambda item: (
                    _candidate_tier(self.candidate_evaluation_policy, item.alpha),
                    item.k,
                    item.alpha,
                    item.transition_weights,
                    item.relative_cap,
                ),
            )
        active_tier = -1
        evaluate_active_tier = True
        for spec in ordered_specs:
            tier = _candidate_tier(self.candidate_evaluation_policy, spec.alpha)
            if tier != active_tier:
                active_tier = tier
                evaluate_active_tier = _evaluate_candidate_tier(
                    self.candidate_evaluation_policy, tier, len(candidates)
                )
            if not evaluate_active_tier:
                continue
            trace = RapCandidateTrace(
                k=spec.k,
                alpha=spec.alpha,
                status="NOT_EVALUATED",
                raw_local_components=_values(raw_components),
            )
            if spec.k > min(stride, len(raw) if self.terminal_constraint_mode == "EXECUTED_INTERNAL" else len(raw) - 1, self.config.k_max):
                traces.append(replace(trace, status="OUT_OF_RANGE"))
                continue
            bridge = bridge_prefix(
                raw_norm=raw,
                history_norm=completed_actions,
                reference_targets=reference_targets,
                reference_rho=reference_rho,
                prefix_length=spec.k,
                objective_count=objective_count,
                alpha=spec.alpha,
                transition_weights=spec.transition_weights,
                objective_weights=self.config.objective_weights,
                relative_cap=spec.relative_cap,
                correction_cap=self.config.correction_cap,
                allow_full_executed_prefix=(
                    self.objective_count_mode == OBJECTIVE_EXECUTED_ONLY
                ),
            )
            if bridge is None:
                traces.append(replace(trace, status="BRIDGE_FAILURE"))
                continue
            bridge_components = local_translation_components(bridge.candidate, completed_actions, objective_count)
            bridge_objective = local_translation_objective(
                bridge.candidate,
                completed_actions,
                count=objective_count,
                objective_weights=self.config.objective_weights,
            )
            trace = replace(
                trace,
                bridge_objective=float("nan") if bridge_objective is None else bridge_objective,
                lower_correction=_rows(bridge.lower_correction),
                upper_correction=_rows(bridge.upper_correction),
                bridge_local_components=_values(bridge_components),
                relative_cap_hit=bridge.relative_cap_hit,
                correction_cap_hit=bridge.correction_cap_hit,
            )
            if bridge_objective is None or bridge_components is None or np.array_equal(bridge.candidate, raw):
                traces.append(replace(trace, status="BRIDGE_NO_EFFECT"))
                continue
            if (
                self.preprojection_objective_bound
                == PREPROJECTION_BOUND_BRIDGE_MIN_GAIN
                and preprojection_min_gain_bound_fails(
                    raw_objective,
                    bridge_objective,
                    self.config.min_gain,
                )
            ):
                traces.append(
                    replace(trace, status="PREPROJECTION_MIN_GAIN_BOUND_FAIL")
                )
                continue
            if (
                self.projector_context_cache_mode
                == PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY
            ):
                if projector_cache is None:
                    projector_cache = ProjectorQueryContextCache(
                        raw,
                        observed,
                        response,
                        terminal_constraint_mode=self.terminal_constraint_mode,
                    )
                projection = project_execution_aware_candidate_cached(
                    projector_cache,
                    bridge.candidate,
                    prefix_length=spec.k,
                    lower_correction=bridge.lower_correction,
                    upper_correction=bridge.upper_correction,
                )
            elif (
                self.projector_context_cache_mode
                == PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE
            ):
                if projector_cache is None:
                    projector_cache = ProjectorQueryContextCache(
                        raw,
                        observed,
                        response,
                        terminal_constraint_mode=self.terminal_constraint_mode,
                    )
                projector = (
                    project_execution_aware_candidate_cached_validator_only
                    if self.terminal_projection_role == TERMINAL_PROJECTION_VALIDATOR_ONLY
                    else project_execution_aware_candidate_cached_terminal_endpoint_reuse
                )
                projection = projector(
                    projector_cache,
                    bridge.candidate,
                    prefix_length=spec.k,
                    lower_correction=bridge.lower_correction,
                    upper_correction=bridge.upper_correction,
                )
            elif (
                self.projector_context_cache_mode
                == PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_DYKSTRA
            ):
                if projector_cache is None:
                    projector_cache = ProjectorQueryContextCache(
                        raw,
                        observed,
                        response,
                        terminal_constraint_mode=self.terminal_constraint_mode,
                    )
                projection = project_execution_aware_candidate_cached_dykstra(
                    projector_cache,
                    bridge.candidate,
                    prefix_length=spec.k,
                    lower_correction=bridge.lower_correction,
                    upper_correction=bridge.upper_correction,
                )
            elif (
                self.projector_context_cache_mode
                == PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_REPRESENTATION_AWARE
            ):
                if projector_cache is None:
                    projector_cache = ProjectorQueryContextCache(
                        raw,
                        observed,
                        response,
                        terminal_constraint_mode=self.terminal_constraint_mode,
                    )
                projection = project_execution_aware_candidate_cached_representation_aware(
                    projector_cache,
                    bridge.candidate,
                    prefix_length=spec.k,
                    lower_correction=bridge.lower_correction,
                    upper_correction=bridge.upper_correction,
                )
            elif (
                self.projector_context_cache_mode
                == PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_CONVERGED_REPRESENTATION_REPAIR
            ):
                if projector_cache is None:
                    projector_cache = ProjectorQueryContextCache(
                        raw,
                        observed,
                        response,
                        terminal_constraint_mode=self.terminal_constraint_mode,
                    )
                projection = project_execution_aware_candidate_cached_converged_representation_repair(
                    projector_cache,
                    bridge.candidate,
                    prefix_length=spec.k,
                    lower_correction=bridge.lower_correction,
                    upper_correction=bridge.upper_correction,
                )
            else:
                projection = project_execution_aware_candidate(
                    raw,
                    bridge.candidate,
                    observed,
                    response,
                    prefix_length=spec.k,
                    lower_correction=bridge.lower_correction,
                    upper_correction=bridge.upper_correction,
                )
            if projection.candidate is None:
                projector_failure = True
                traces.append(
                    replace(
                        trace,
                        status=f"PROJECTOR:{projection.status}",
                        projection_iterations=projection.iterations,
                        projection_max_violation=projection.max_violation,
                        projection_slacks=projection.constraint_slacks,
                        projection_dykstra_cycles=projection.dykstra_cycles,
                        projection_cyclic_completion_cycles=projection.cyclic_completion_cycles,
                        representation_repair_invocations=projection.representation_repair_invocations,
                        representation_repair_steps=projection.representation_repair_steps,
                        representation_repair_ulp_neighbor_evaluations=(
                            projection.representation_repair_ulp_neighbor_evaluations
                        ),
                    )
                )
                continue
            projected = projection.candidate
            try:
                if projector_cache is None:
                    counterfactual = evaluate_physical_counterfactual(
                        raw,
                        projected,
                        observed,
                        response,
                        prefix_length=spec.k,
                        terminal_constraint_mode=self.terminal_constraint_mode,
                    )
                else:
                    counterfactual = evaluate_physical_counterfactual_cached(
                        projector_cache,
                        projected,
                        prefix_length=spec.k,
                    )
                veto = physical_counterfactual_veto(counterfactual, terminal_veto=self.terminal_veto)
            except ValueError:
                projector_failure = True
                traces.append(replace(trace, status="COUNTERFACTUAL_INVALID"))
                continue
            projected_components = local_translation_components(projected, completed_actions, objective_count)
            projected_objective = local_translation_objective(
                projected,
                completed_actions,
                count=objective_count,
                objective_weights=self.config.objective_weights,
            )
            correction = projected[: spec.k, :3].astype(np.float64) - raw[: spec.k, :3].astype(np.float64)
            correction_l2 = float(np.linalg.norm(correction))
            correction_max_abs = float(np.max(np.abs(correction)))
            action_relative_gain = (
                (raw_objective - projected_objective) / raw_objective
                if projected_objective is not None and raw_objective > 0
                else float("-inf")
            )
            action_selection_score = (
                projected_objective + self.config.correction_penalty * correction_l2 / np.sqrt(spec.k)
                if projected_objective is not None
                else float("inf")
            )
            raw_physical_values = np.asarray(
                [
                    counterfactual.raw.boundary_jerk,
                    counterfactual.raw.next_jerk,
                    counterfactual.raw.prefix_exit_jerk,
                ],
                dtype=np.float64,
            )
            candidate_physical_values = np.asarray(
                [
                    counterfactual.candidate.boundary_jerk,
                    counterfactual.candidate.next_jerk,
                    counterfactual.candidate.prefix_exit_jerk,
                ],
                dtype=np.float64,
            )
            physical_weights = np.asarray([1.0, 0.5, 0.25], dtype=np.float64)
            physical_relative_score = physical_relative_objective(
                raw_physical_values,
                candidate_physical_values,
                physical_weights,
            )
            correction_fraction_of_cap = (
                float("nan")
                if self.config.correction_cap is None
                else float(
                    correction_l2
                    / (self.config.correction_cap * np.sqrt(3.0 * spec.k))
                )
            )
            candidate_predicted_tcp_horizon_isj = float("nan")
            paired_tcp_horizon_jerk_ratio = float("nan")
            if self.selection_mode == SELECTION_TCP_HORIZON_ISJ:
                try:
                    candidate_predicted_tcp_horizon_isj = (
                        predicted_tcp_horizon_integrated_squared_jerk(
                            projected,
                            observed,
                            response,
                            alternative=RESPONSE_ALTERNATIVE_SELECTED,
                        )
                    )
                    paired_tcp_horizon_jerk_ratio = (
                        paired_predicted_tcp_horizon_jerk_ratio(
                            raw_predicted_tcp_horizon_isj,
                            candidate_predicted_tcp_horizon_isj,
                        )
                    )
                except ValueError:
                    traces.append(replace(trace, status="TCP_HORIZON_OBJECTIVE_INVALID"))
                    continue
            predicted_boundary_improvement = float(
                counterfactual.raw.boundary_jerk - counterfactual.candidate.boundary_jerk
            )
            uncertainty_threshold = (
                float(np.sqrt(3.0) * response.prequential_rmse * self.uncertainty_scale)
                if self.uncertainty_scale is not None
                else float("nan")
            )
            if self.selection_mode == SELECTION_TCP_HORIZON_ISJ:
                relative_gain = 1.0 - paired_tcp_horizon_jerk_ratio
                selection_objective = paired_tcp_horizon_jerk_ratio
                selection_score = (
                    paired_tcp_horizon_jerk_ratio
                    + self.config.correction_penalty * correction_fraction_of_cap
                )
                no_op_selection_score = 1.0
            elif self.selection_mode == SELECTION_PHYSICAL:
                relative_gain = 1.0 - physical_relative_score
                selection_objective = physical_relative_score
                selection_score = (
                    physical_relative_score
                    + self.config.correction_penalty * correction_fraction_of_cap
                )
                no_op_selection_score = 1.0
            else:
                relative_gain = action_relative_gain
                selection_objective = projected_objective
                selection_score = action_selection_score
                no_op_selection_score = raw_objective
            trace = replace(
                trace,
                projected_objective=float("nan") if projected_objective is None else projected_objective,
                relative_gain=relative_gain,
                selection_score=selection_score,
                correction_l2=correction_l2,
                correction_max_abs=correction_max_abs,
                projected_local_components=_values(projected_components),
                projection_iterations=projection.iterations,
                projection_max_violation=projection.max_violation,
                projection_slacks=projection.constraint_slacks,
                projection_dykstra_cycles=projection.dykstra_cycles,
                projection_cyclic_completion_cycles=projection.cyclic_completion_cycles,
                representation_repair_invocations=projection.representation_repair_invocations,
                representation_repair_steps=projection.representation_repair_steps,
                representation_repair_ulp_neighbor_evaluations=(
                    projection.representation_repair_ulp_neighbor_evaluations
                ),
                raw_physical=counterfactual.raw.as_pairs(),
                candidate_physical=counterfactual.candidate.as_pairs(),
                veto_reason=veto.reason,
                predicted_boundary_improvement=predicted_boundary_improvement,
                uncertainty_threshold=uncertainty_threshold,
                physical_relative_objective=physical_relative_score,
                correction_fraction_of_cap=correction_fraction_of_cap,
                raw_predicted_tcp_horizon_integrated_squared_jerk=(
                    raw_predicted_tcp_horizon_isj
                ),
                candidate_predicted_tcp_horizon_integrated_squared_jerk=(
                    candidate_predicted_tcp_horizon_isj
                ),
                paired_predicted_tcp_horizon_jerk_ratio=(
                    paired_tcp_horizon_jerk_ratio
                ),
            )
            if not veto.allowed:
                physical_veto_failure = True
                traces.append(replace(trace, status="PHYSICAL_VETO"))
                continue
            if self.uncertainty_scale is not None and not predicted_boundary_improvement > uncertainty_threshold:
                traces.append(replace(trace, status="UNCERTAINTY_GATE_FAIL"))
                continue
            if projected_objective is None or projected_components is None:
                traces.append(replace(trace, status="NONFINITE_OBJECTIVE"))
                continue
            if relative_gain < self.config.min_gain:
                traces.append(replace(trace, status="MIN_GAIN_FAIL"))
                continue
            if np.any(projected_components > raw_components + self.config.max_local_component_increase):
                traces.append(replace(trace, status="LOCAL_COMPONENT_INCREASE_FAIL"))
                continue
            predecessors = [item for item in eligible_by_alpha.get(spec.alpha, []) if item.k < spec.k]
            if predecessors:
                predecessor_k = max(item.k for item in predecessors)
                predecessor = min(
                    (item for item in predecessors if item.k == predecessor_k),
                    key=lambda item: (item.objective, item.score, item.correction_l2),
                )
                if predecessor.objective is None or not selection_objective < predecessor.objective:
                    traces.append(replace(trace, status="SHORTER_PREFIX_DOMINANCE_FAIL"))
                    continue
            if not selection_score < no_op_selection_score - self.config.noop_selection_margin:
                traces.append(replace(trace, status="NOOP_SCORE_FAIL"))
                continue
            candidate = _Candidate(
                chunk=projected,
                score=selection_score,
                k=spec.k,
                alpha=spec.alpha,
                correction_l2=correction_l2,
                objective=selection_objective,
                trace_index=len(traces),
            )
            candidates.append(candidate)
            eligible_by_alpha.setdefault(spec.alpha, []).append(candidate)
            traces.append(replace(trace, status="ELIGIBLE"))

        if candidates:
            selected = select_eligible_candidate(candidates, traces, self.candidate_ranking_mode, self.candidate_ranking_correction_penalty)
            traces[selected.trace_index] = replace(traces[selected.trace_index], status="SELECTED")
            return self._decision(
                raw,
                selected=selected.chunk,
                k=selected.k,
                alpha=selected.alpha,
                reason=RapReason.SELECTED,
                raw_objective=raw_objective,
                traces=traces,
                response=response,
                risk_eligible=risk_eligible,
                risk_eligibility_ablated=ablate_risk_trust_eligibility,
                reversal_rollback=False,
                projector_cache=projector_cache,
            )
        reason = (
            RapReason.PROJECTOR_FAILURE
            if projector_failure
            else RapReason.PHYSICAL_VETO
            if physical_veto_failure
            else RapReason.NO_ELIGIBLE_CANDIDATE
        )
        return self._decision(
            raw,
            selected=None,
            k=0,
            alpha=0.0,
            reason=reason,
            raw_objective=raw_objective,
            traces=traces,
            response=response,
            risk_eligible=risk_eligible,
            risk_eligibility_ablated=ablate_risk_trust_eligibility,
            reversal_rollback=False,
            projector_cache=projector_cache,
        )
