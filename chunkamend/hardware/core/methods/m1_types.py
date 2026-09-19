"""Shared, runtime-independent data contracts for the M1 method cores."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

RESPONSE_PREDICTION_TRANSLATION_ONLY = "TRANSLATION_ONLY_3X3"
RESPONSE_PREDICTION_Q6_ALTERNATIVE_OFFSETS = (
    "Q6_TRANSLATION_MATRIX_PLUS_RAW_AND_SELECTED_ROTATION_OFFSETS"
)
RESPONSE_PREDICTION_Q3_MEMORY = "Q3_ARX1_FULL_TEMPORAL_AFFINE"
RESPONSE_PREDICTION_SEMANTICS = {
    RESPONSE_PREDICTION_Q3_MEMORY,
    RESPONSE_PREDICTION_TRANSLATION_ONLY,
    RESPONSE_PREDICTION_Q6_ALTERNATIVE_OFFSETS,
}
def valid_remaining_shape(value, columns):
    """Accept the H15 remaining window while preserving the frozen K=5 floor."""
    return value.ndim == 2 and value.shape[1] == columns and 5 <= value.shape[0] <= 15

RESPONSE_ALTERNATIVE_RAW = "RAW"
RESPONSE_ALTERNATIVE_SELECTED = "SELECTED"


class RapReason(str, Enum):
    NO_OP = "NO_OP"
    SELECTED = "SELECTED"
    NO_ELIGIBLE_CANDIDATE = "NO_ELIGIBLE_CANDIDATE"
    NONFINITE_INPUT = "NONFINITE_INPUT"
    PROJECTOR_UNAVAILABLE = "PROJECTOR_UNAVAILABLE"
    PROJECTOR_FAILURE = "PROJECTOR_FAILURE"
    PHYSICAL_VETO = "PHYSICAL_VETO"
    REVERSAL_ROLLBACK = "REVERSAL_ROLLBACK"
    RESPONSE_UNAVAILABLE = "RESPONSE_UNAVAILABLE"


@dataclass(frozen=True)
class RiskFeature:
    name: str
    value: float | None
    shape: tuple[int, ...]
    units: str
    logical_timestamp: int
    latest_availability: str
    missing_behavior: str = "UNAVAILABLE"


@dataclass(frozen=True)
class ResponseFit:
    """Causal action-to-TCP-displacement response prediction contract.

    The historical Q3 path uses only ``matrix``.  A Q6 query additionally
    binds the rotation contribution for the two physically distinct
    alternatives: exact raw fallback and an actively selected translation
    candidate.  Candidate construction remains translation-only.
    """

    matrix: np.ndarray | None
    available: bool
    reason: str
    sample_count: int = 0
    window_start: int = 0
    window_end: int = 0
    condition_number: float = float("nan")
    residual_rmse: float = float("nan")
    rank: int = 0
    fit_age: int = 0
    prequential_rmse: float = float("nan")
    prequential_count: int = 0
    prediction_semantics: str = RESPONSE_PREDICTION_TRANSLATION_ONLY
    raw_prediction_offset: np.ndarray | None = None
    selected_prediction_offset: np.ndarray | None = None
    temporal_response: object | None = None
    memory_fit_diagnostics: dict | None = None


def response_prediction_contract_valid(response: ResponseFit) -> bool:
    try:
        if not isinstance(response, ResponseFit) or response.prediction_semantics not in RESPONSE_PREDICTION_SEMANTICS:
            return False
        matrix = np.asarray(response.matrix)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            return False
        if response.prediction_semantics == RESPONSE_PREDICTION_Q3_MEMORY:
            from chunkamend.hardware.core.methods.temporal_affine import TemporalResponse
            temporal = response.temporal_response
            return bool(isinstance(temporal, TemporalResponse)
                and temporal.matrix.shape == (30, 30) and temporal.offset.shape == (30,)
                and temporal.initial_displacement.shape == (3,)
                and np.isfinite(temporal.matrix).all() and np.isfinite(temporal.offset).all()
                and np.isfinite(temporal.initial_displacement).all()
                and response.raw_prediction_offset is None and response.selected_prediction_offset is None)
        if response.temporal_response is not None:
            return False
        if response.prediction_semantics == RESPONSE_PREDICTION_TRANSLATION_ONLY:
            return response.raw_prediction_offset is None and response.selected_prediction_offset is None
        raw_offset = np.asarray(response.raw_prediction_offset)
        selected_offset = np.asarray(response.selected_prediction_offset)
        return bool(
            valid_remaining_shape(raw_offset, 3)
            and selected_offset.shape == raw_offset.shape
            and np.isfinite(raw_offset).all()
            and np.isfinite(selected_offset).all()
        )
    except (TypeError, ValueError, OverflowError, AttributeError):
        return False


def predict_response_displacements(
    chunk: np.ndarray,
    response: ResponseFit,
    *,
    alternative: str,
) -> np.ndarray:
    value = np.asarray(chunk)
    if (
        not valid_remaining_shape(value, 7)
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
        or alternative not in {RESPONSE_ALTERNATIVE_RAW, RESPONSE_ALTERNATIVE_SELECTED}
        or not response.available
        or not response_prediction_contract_valid(response)
    ):
        raise ValueError("invalid response prediction contract input")
    if (response.prediction_semantics == RESPONSE_PREDICTION_Q6_ALTERNATIVE_OFFSETS
            and response.raw_prediction_offset.shape[0] != value.shape[0]):
        raise ValueError("Q6 response offsets must match the actual remaining horizon")
    if response.prediction_semantics == RESPONSE_PREDICTION_Q3_MEMORY:
        return np.ascontiguousarray(response.temporal_response.predict(value[:, :3]))
    predicted = value[:, :3].astype(np.float64) @ np.asarray(response.matrix, dtype=np.float64)
    if response.prediction_semantics == RESPONSE_PREDICTION_Q6_ALTERNATIVE_OFFSETS:
        offset = (
            response.raw_prediction_offset
            if alternative == RESPONSE_ALTERNATIVE_RAW
            else response.selected_prediction_offset
        )
        predicted = predicted + np.asarray(offset, dtype=np.float64)
    if predicted.shape != (len(value), 3):
        raise ValueError("invalid response prediction shape")
    # Do not reject finite-input floating overflow here.  Historical callers
    # map it to their own controlled failure (physical-continuity or jerk),
    # and the cached and uncached projector paths must retain the same receipt.
    return np.ascontiguousarray(predicted)


@dataclass(frozen=True)
class RapConfig:
    """M2 primary RAP configuration; projection constants are structural."""

    k_max: int
    correction_cap: float | None
    objective_weights: tuple[float, ...]
    min_gain: float = 0.01
    correction_penalty: float = 0.25
    max_local_component_increase: float = 0.05
    noop_selection_margin: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.objective_weights, tuple):
            raise ValueError("objective_weights must be a nonempty tuple")
        values = (
            self.min_gain,
            self.correction_penalty,
            self.max_local_component_increase,
            self.noop_selection_margin,
            *self.objective_weights,
        )
        if isinstance(self.k_max, bool) or not isinstance(self.k_max, int) or self.k_max < 1:
            raise ValueError("RapConfig requires integer k_max >= 1")
        if not self.objective_weights or not np.isfinite(values).all():
            raise ValueError("RapConfig values must be finite")
        if (
            self.correction_cap is not None
            and (
                isinstance(self.correction_cap, bool)
                or not isinstance(self.correction_cap, (int, float, np.integer, np.floating))
                or not np.isfinite(self.correction_cap)
                or self.correction_cap <= 0
            )
        ):
            raise ValueError("correction cap must be positive finite or None")
        if any(weight <= 0 for weight in self.objective_weights):
            raise ValueError("objective weights must be positive")
        if (
            not 0 <= self.min_gain <= 1
            or self.correction_penalty < 0
            or self.max_local_component_increase < 0
            or self.noop_selection_margin < 0
        ):
            raise ValueError("invalid M2 RAP selection safeguards")
        if (
            self.min_gain != 0.01
            or self.correction_penalty != 0.25
            or self.max_local_component_increase != 0.05
            or self.noop_selection_margin != 0.0
        ):
            raise ValueError("M2 RAP selection safeguards are frozen at 0.01/0.25/0.05/0.0")


@dataclass(frozen=True)
class RapCandidateSpec:
    k: int
    alpha: float
    transition_weights: tuple[float, ...]
    relative_cap: float

    def __post_init__(self) -> None:
        values = (*self.transition_weights, self.alpha, self.relative_cap)
        if (
            isinstance(self.k, bool)
            or not isinstance(self.k, int)
            or not 1 <= self.k < 15
            or not 0 < self.alpha <= 1
            or len(self.transition_weights) != self.k
            or not np.isfinite(values).all()
            or any(value < 0 for value in self.transition_weights)
            or self.relative_cap < 0
        ):
            raise ValueError("invalid explicit RAP candidate specification")


@dataclass(frozen=True)
class RapCandidateTrace:
    k: int
    alpha: float
    status: str
    bridge_objective: float = float("nan")
    projected_objective: float = float("nan")
    relative_gain: float = float("nan")
    selection_score: float = float("nan")
    correction_l2: float = float("nan")
    correction_max_abs: float = float("nan")
    relative_cap_hit: bool = False
    correction_cap_hit: bool = False
    projection_iterations: int = 0
    projection_dykstra_cycles: int = 0
    projection_cyclic_completion_cycles: int = 0
    representation_repair_invocations: int = 0
    representation_repair_steps: int = 0
    representation_repair_ulp_neighbor_evaluations: int = 0
    projection_max_violation: float = float("nan")
    lower_correction: tuple[tuple[float, ...], ...] = ()
    upper_correction: tuple[tuple[float, ...], ...] = ()
    raw_local_components: tuple[float, ...] = ()
    bridge_local_components: tuple[float, ...] = ()
    projected_local_components: tuple[float, ...] = ()
    projection_slacks: tuple[tuple[str, float], ...] = ()
    raw_physical: tuple[tuple[str, float], ...] = ()
    candidate_physical: tuple[tuple[str, float], ...] = ()
    veto_reason: str = "NOT_EVALUATED"
    predicted_boundary_improvement: float = float("nan")
    uncertainty_threshold: float = float("nan")
    physical_relative_objective: float = float("nan")
    correction_fraction_of_cap: float = float("nan")
    raw_predicted_tcp_horizon_integrated_squared_jerk: float = float("nan")
    candidate_predicted_tcp_horizon_integrated_squared_jerk: float = float("nan")
    paired_predicted_tcp_horizon_jerk_ratio: float = float("nan")


@dataclass(frozen=True)
class RapTelemetry:
    method_id: str
    raw_chunk: np.ndarray
    selected_chunk: np.ndarray
    selected_reason: str
    raw_objective: float
    candidate_traces: tuple[RapCandidateTrace, ...]
    response_sample_count: int
    response_window_start: int
    response_window_end: int
    response_rank: int
    response_condition_number: float
    response_residual_rmse: float
    risk_eligible: bool
    risk_eligibility_ablated: bool
    reversal_rollback: bool
    selection_mode: str = "ACTION_OBJECTIVE"
    candidate_ranking_mode: str = "ACTION_SCORE"
    projector_context_cache_mode: str = "DISABLED"
    projector_context_cache_initialized: bool = False
    projector_context_builds_by_k: tuple[tuple[int, int], ...] = ()
    projector_context_hits_by_k: tuple[tuple[int, int], ...] = ()
    projector_context_gram_build_count: int = 0
    projector_context_failure_by_k: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class RapDecision:
    chunk: np.ndarray
    k: int
    alpha: float
    reason: RapReason
    telemetry: RapTelemetry | None = None
