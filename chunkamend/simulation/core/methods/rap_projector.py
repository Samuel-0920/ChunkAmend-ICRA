"""M2 execution-aware physical projection and independent counterfactual veto."""

from dataclasses import dataclass

import numpy as np
from chunkamend.simulation.core.methods import compiled_projection
from chunkamend.simulation.core.methods.numeric_dispatch import solve3_f64
from chunkamend.simulation.core.methods import exact_cycle, box_certificate

from chunkamend.simulation.core.methods.m1_types import RESPONSE_ALTERNATIVE_RAW
from chunkamend.simulation.core.methods.m1_types import RESPONSE_ALTERNATIVE_SELECTED
from chunkamend.simulation.core.methods.m1_types import RESPONSE_PREDICTION_Q6_ALTERNATIVE_OFFSETS
from chunkamend.simulation.core.methods.m1_types import RESPONSE_PREDICTION_TRANSLATION_ONLY
from chunkamend.simulation.core.methods.m1_types import valid_remaining_shape
from chunkamend.simulation.core.methods.m1_types import RESPONSE_PREDICTION_Q3_MEMORY
from chunkamend.simulation.core.methods.m1_types import ResponseFit
from chunkamend.simulation.core.methods.m1_types import predict_response_displacements
from chunkamend.simulation.core.methods.m1_types import response_prediction_contract_valid

HORIZON = 10
PHYSICAL_TOLERANCE = 1e-12
BOUNDARY_JERK_IMPROVEMENT = 1e-7
PROJECTION_TOLERANCE = 1e-9
COUNTERFACTUAL_VETO_TOLERANCE = PHYSICAL_TOLERANCE + PROJECTION_TOLERANCE
MAX_PROJECTION_CYCLES = 128
DYKSTRA_PRIMARY_CYCLES = 16
SIMPLIFIED_ABLATION_ID = "RAP-ACTIONSPACE-SIMPLIFIED-ABLATION"
TERMINAL_CONSTRAINT_PREFIX_EXIT = "PREFIX_EXIT"
TERMINAL_CONSTRAINT_EXECUTED_INTERNAL = "EXECUTED_INTERNAL"
TERMINAL_CONSTRAINT_MODES = {
    TERMINAL_CONSTRAINT_PREFIX_EXIT,
    TERMINAL_CONSTRAINT_EXECUTED_INTERNAL,
}


@dataclass(frozen=True)
class PhysicalContinuityComponents:
    boundary_velocity: float
    boundary_jerk: float
    next_jerk: float
    prefix_exit_jerk: float
    terminal_constraint_mode: str = TERMINAL_CONSTRAINT_PREFIX_EXIT

    def as_pairs(self) -> tuple[tuple[str, float], ...]:
        terminal_name = (
            "terminal_internal_jerk"
            if self.terminal_constraint_mode == TERMINAL_CONSTRAINT_EXECUTED_INTERNAL
            else "prefix_exit_jerk"
        )
        return (
            ("boundary_velocity", self.boundary_velocity),
            ("boundary_jerk", self.boundary_jerk),
            ("next_jerk", self.next_jerk),
            (terminal_name, self.prefix_exit_jerk),
        )


@dataclass(frozen=True)
class PhysicalCounterfactual:
    raw: PhysicalContinuityComponents
    candidate: PhysicalContinuityComponents


@dataclass(frozen=True)
class PhysicalVetoDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class ExecutionProjectionResult:
    candidate: np.ndarray | None
    status: str
    iterations: int
    max_violation: float
    constraint_slacks: tuple[tuple[str, float], ...] = ()
    dykstra_cycles: int = 0
    cyclic_completion_cycles: int = 0
    representation_repair_invocations: int = 0
    representation_repair_steps: int = 0
    representation_repair_ulp_neighbor_evaluations: int = 0


@dataclass(frozen=True)
class _AffineBall:
    name: str
    matrix: np.ndarray
    offset: np.ndarray
    radius: float


def _response_matrix(response: ResponseFit) -> np.ndarray:
    matrix = np.asarray(response.matrix)
    if not response.available or not response_prediction_contract_valid(response):
        raise ValueError("finite available 3x3 response is required")
    return matrix.astype(np.float64, copy=False)


def _physical_vectors(
    predicted: np.ndarray,
    observed_displacements: np.ndarray,
    *,
    prefix_length: int,
    terminal_constraint_mode: str = TERMINAL_CONSTRAINT_PREFIX_EXIT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(observed_displacements, dtype=np.float64)
    if (
        not valid_remaining_shape(predicted, 3)
        or observed.shape != (2, 3)
        or not np.isfinite(predicted).all()
        or not np.isfinite(observed).all()
        or isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or not 1 <= prefix_length <= min(len(predicted), HORIZON - 1)
        or (terminal_constraint_mode == TERMINAL_CONSTRAINT_PREFIX_EXIT and prefix_length >= len(predicted))
        or terminal_constraint_mode not in TERMINAL_CONSTRAINT_MODES
        or (
            terminal_constraint_mode == TERMINAL_CONSTRAINT_EXECUTED_INTERNAL
            and prefix_length < 3
        )
    ):
        raise ValueError("invalid physical continuity inputs")
    boundary_velocity = predicted[0] - observed[-1]
    boundary_jerk = predicted[0] - 2 * observed[-1] + observed[-2]
    next_jerk = predicted[1] - 2 * predicted[0] + observed[-1]
    if terminal_constraint_mode == TERMINAL_CONSTRAINT_EXECUTED_INTERNAL:
        prefix_exit = (
            predicted[prefix_length - 1]
            - 2 * predicted[prefix_length - 2]
            + predicted[prefix_length - 3]
        )
    elif prefix_length == 1:
        prefix_exit = predicted[1] - 2 * predicted[0] + observed[-1]
    else:
        prefix_exit = predicted[prefix_length] - 2 * predicted[prefix_length - 1] + predicted[prefix_length - 2]
    return boundary_velocity, boundary_jerk, next_jerk, prefix_exit


def physical_continuity_components(
    chunk: np.ndarray,
    observed_displacements: np.ndarray,
    response: ResponseFit,
    *,
    prefix_length: int,
    terminal_constraint_mode: str = TERMINAL_CONSTRAINT_PREFIX_EXIT,
    alternative: str = RESPONSE_ALTERNATIVE_RAW,
) -> PhysicalContinuityComponents:
    value = np.asarray(chunk)
    if not valid_remaining_shape(value, 7) or not np.isfinite(value).all():
        raise ValueError("chunk must be finite [10,7]")
    predicted = predict_response_displacements(value, response, alternative=alternative)
    vectors = _physical_vectors(
        predicted,
        observed_displacements,
        prefix_length=prefix_length,
        terminal_constraint_mode=terminal_constraint_mode,
    )
    norms = tuple(float(np.linalg.norm(vector)) for vector in vectors)
    if not np.isfinite(norms).all():
        raise ValueError("nonfinite physical continuity component")
    return PhysicalContinuityComponents(
        *norms,
        terminal_constraint_mode=terminal_constraint_mode,
    )


def evaluate_physical_counterfactual(
    raw: np.ndarray,
    candidate: np.ndarray,
    observed_displacements: np.ndarray,
    response: ResponseFit,
    *,
    prefix_length: int,
    terminal_constraint_mode: str = TERMINAL_CONSTRAINT_PREFIX_EXIT,
) -> PhysicalCounterfactual:
    raw_value, candidate_value = np.asarray(raw), np.asarray(candidate)
    if (
        not valid_remaining_shape(raw_value, 7)
        or candidate_value.shape != raw_value.shape
        or not np.isfinite(raw_value).all()
        or not np.isfinite(candidate_value).all()
        or not np.array_equal(candidate_value[prefix_length:], raw_value[prefix_length:])
        or not np.array_equal(candidate_value[:, 3:], raw_value[:, 3:])
    ):
        raise ValueError("counterfactual locality or input validation failed")
    return PhysicalCounterfactual(
        raw=physical_continuity_components(
            raw_value,
            observed_displacements,
            response,
            prefix_length=prefix_length,
            terminal_constraint_mode=terminal_constraint_mode,
            alternative=RESPONSE_ALTERNATIVE_RAW,
        ),
        candidate=physical_continuity_components(
            candidate_value,
            observed_displacements,
            response,
            prefix_length=prefix_length,
            terminal_constraint_mode=terminal_constraint_mode,
            alternative=RESPONSE_ALTERNATIVE_SELECTED,
        ),
    )


def physical_counterfactual_veto(
    counterfactual: PhysicalCounterfactual, *, tolerance: float = COUNTERFACTUAL_VETO_TOLERANCE, terminal_veto: bool = True
) -> PhysicalVetoDecision:
    if (
        type(terminal_veto) is not bool
        or not isinstance(counterfactual, PhysicalCounterfactual)
        or tolerance != COUNTERFACTUAL_VETO_TOLERANCE
    ):
        raise ValueError("M2 physical veto requires its frozen tolerance")
    raw, candidate = counterfactual.raw, counterfactual.candidate
    if (
        raw.terminal_constraint_mode not in TERMINAL_CONSTRAINT_MODES
        or candidate.terminal_constraint_mode != raw.terminal_constraint_mode
    ):
        return PhysicalVetoDecision(allowed=False, reason="INVALID_PHYSICAL_DIAGNOSTIC")
    values = tuple(value for _, value in raw.as_pairs() + candidate.as_pairs())
    if not np.isfinite(values).all() or any(value < 0 for value in values):
        return PhysicalVetoDecision(allowed=False, reason="INVALID_PHYSICAL_DIAGNOSTIC")
    if not candidate.boundary_jerk + tolerance < raw.boundary_jerk:
        return PhysicalVetoDecision(allowed=False, reason="BOUNDARY_JERK_NOT_STRICTLY_IMPROVED")
    if candidate.boundary_velocity > raw.boundary_velocity + tolerance:
        return PhysicalVetoDecision(allowed=False, reason="BOUNDARY_VELOCITY_WORSE")
    if candidate.next_jerk > raw.next_jerk + tolerance:
        return PhysicalVetoDecision(allowed=False, reason="NEXT_JERK_WORSE")
    if terminal_veto and candidate.prefix_exit_jerk > raw.prefix_exit_jerk + tolerance:
        reason = (
            "TERMINAL_INTERNAL_JERK_WORSE"
            if raw.terminal_constraint_mode == TERMINAL_CONSTRAINT_EXECUTED_INTERNAL
            else "PREFIX_EXIT_JERK_WORSE"
        )
        return PhysicalVetoDecision(allowed=False, reason=reason)
    return PhysicalVetoDecision(allowed=True, reason="ACCEPTED")


def _affine_balls(
    raw: np.ndarray,
    observed_displacements: np.ndarray,
    response: ResponseFit,
    *,
    prefix_length: int,
    terminal_constraint_mode: str = TERMINAL_CONSTRAINT_PREFIX_EXIT,
) -> tuple[_AffineBall, ...]:
    if response.prediction_semantics == RESPONSE_PREDICTION_Q3_MEMORY:
        _response_matrix(response)
        from chunkamend.simulation.core.methods.temporal_affine import build_prefix_constraints
        prefix = build_prefix_constraints(response.temporal_response, np.asarray(raw)[:, :3],
            observed_displacements, prefix_length=prefix_length,
            terminal_constraint_mode=terminal_constraint_mode)
        return prefix.projection_balls + (prefix.terminal_ball,)
    matrix = _response_matrix(response)
    raw_value = np.asarray(raw)
    raw_predicted = predict_response_displacements(
        raw_value, response, alternative=RESPONSE_ALTERNATIVE_RAW
    )
    selected_base_predicted = predict_response_displacements(
        raw_value, response, alternative=RESPONSE_ALTERNATIVE_SELECTED
    )
    raw_vectors = _physical_vectors(
        raw_predicted,
        observed_displacements,
        prefix_length=prefix_length,
        terminal_constraint_mode=terminal_constraint_mode,
    )
    selected_base_vectors = _physical_vectors(
        selected_base_predicted,
        observed_displacements,
        prefix_length=prefix_length,
        terminal_constraint_mode=terminal_constraint_mode,
    )
    variable_count = 3 * prefix_length

    def ball(name: str, offset: np.ndarray, coefficients: dict[int, float], radius: float) -> _AffineBall:
        affine = np.zeros((3, variable_count), dtype=np.float64)
        for index, coefficient in coefficients.items():
            if 0 <= index < prefix_length:
                affine[:, 3 * index : 3 * index + 3] = coefficient * matrix.T
        return _AffineBall(name, affine, offset, radius)

    raw_boundary_velocity, raw_boundary_jerk, raw_next_jerk, raw_prefix_exit = raw_vectors
    boundary_velocity, boundary_jerk, next_jerk, prefix_exit = selected_base_vectors
    boundary_radius = float(np.linalg.norm(raw_boundary_jerk)) - BOUNDARY_JERK_IMPROVEMENT
    if boundary_radius <= 0:
        raise ValueError("NONPOSITIVE_BOUNDARY_JERK_RADIUS")
    if terminal_constraint_mode == TERMINAL_CONSTRAINT_EXECUTED_INTERNAL:
        exit_name = "terminal_internal_jerk"
        exit_coefficients = {
            prefix_length - 3: 1.0,
            prefix_length - 2: -2.0,
            prefix_length - 1: 1.0,
        }
    else:
        exit_name = "prefix_exit_jerk"
        exit_coefficients = (
            {0: -2.0}
            if prefix_length == 1
            else {prefix_length - 2: 1.0, prefix_length - 1: -2.0}
        )
    return (
        ball("boundary_jerk", boundary_jerk, {0: 1.0}, boundary_radius),
        ball(
            "boundary_velocity",
            boundary_velocity,
            {0: 1.0},
            float(np.linalg.norm(raw_boundary_velocity)) + PHYSICAL_TOLERANCE,
        ),
        ball(
            "next_jerk",
            next_jerk,
            {0: -2.0, 1: 1.0},
            float(np.linalg.norm(raw_next_jerk)) + PHYSICAL_TOLERANCE,
        ),
        ball(
            exit_name,
            prefix_exit,
            exit_coefficients,
            float(np.linalg.norm(raw_prefix_exit)) + PHYSICAL_TOLERANCE,
        ),
    )


def _unique_balls(balls: tuple[_AffineBall, ...]) -> tuple[_AffineBall, ...]:
    unique: list[_AffineBall] = []
    for item in balls:
        if any(
            item.radius == other.radius
            and np.array_equal(item.matrix, other.matrix)
            and np.array_equal(item.offset, other.offset)
            for other in unique
        ):
            continue
        unique.append(item)
    return tuple(unique)


def _readonly_array(value: np.ndarray) -> np.ndarray:
    """Return an owned read-only array for a published cache entry."""

    output = np.array(value, copy=True)
    output.setflags(write=False)
    return output


def _affine_balls_from_query_invariants(
    matrix: np.ndarray,
    predicted_raw: np.ndarray,
    predicted_selected_base: np.ndarray,
    observed_displacements: np.ndarray,
    *,
    prefix_length: int,
    terminal_constraint_mode: str = TERMINAL_CONSTRAINT_PREFIX_EXIT,
) -> tuple[tuple[_AffineBall, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build the same four balls as ``_affine_balls`` from query-local invariants."""

    raw_vectors = _physical_vectors(
        predicted_raw,
        observed_displacements,
        prefix_length=prefix_length,
        terminal_constraint_mode=terminal_constraint_mode,
    )
    selected_base_vectors = _physical_vectors(
        predicted_selected_base,
        observed_displacements,
        prefix_length=prefix_length,
        terminal_constraint_mode=terminal_constraint_mode,
    )
    variable_count = 3 * prefix_length

    def ball(name: str, offset: np.ndarray, coefficients: dict[int, float], radius: float) -> _AffineBall:
        affine = np.zeros((3, variable_count), dtype=np.float64)
        for index, coefficient in coefficients.items():
            if 0 <= index < prefix_length:
                affine[:, 3 * index : 3 * index + 3] = coefficient * matrix.T
        return _AffineBall(name, affine, offset, radius)

    raw_boundary_velocity, raw_boundary_jerk, raw_next_jerk, raw_prefix_exit = raw_vectors
    boundary_velocity, boundary_jerk, next_jerk, prefix_exit = selected_base_vectors
    boundary_radius = float(np.linalg.norm(raw_boundary_jerk)) - BOUNDARY_JERK_IMPROVEMENT
    if boundary_radius <= 0:
        raise ValueError("NONPOSITIVE_BOUNDARY_JERK_RADIUS")
    if terminal_constraint_mode == TERMINAL_CONSTRAINT_EXECUTED_INTERNAL:
        exit_name = "terminal_internal_jerk"
        exit_coefficients = {
            prefix_length - 3: 1.0,
            prefix_length - 2: -2.0,
            prefix_length - 1: 1.0,
        }
    else:
        exit_name = "prefix_exit_jerk"
        exit_coefficients = (
            {0: -2.0}
            if prefix_length == 1
            else {prefix_length - 2: 1.0, prefix_length - 1: -2.0}
        )
    return (
        (
            ball("boundary_jerk", boundary_jerk, {0: 1.0}, boundary_radius),
            ball(
                "boundary_velocity",
                boundary_velocity,
                {0: 1.0},
                float(np.linalg.norm(raw_boundary_velocity)) + PHYSICAL_TOLERANCE,
            ),
            ball(
                "next_jerk",
                next_jerk,
                {0: -2.0, 1: 1.0},
                float(np.linalg.norm(raw_next_jerk)) + PHYSICAL_TOLERANCE,
            ),
            ball(
                exit_name,
                prefix_exit,
                exit_coefficients,
                float(np.linalg.norm(raw_prefix_exit)) + PHYSICAL_TOLERANCE,
            ),
        ),
        raw_vectors,
    )


@dataclass(frozen=True)
class PrefixProjectionContext:
    """Published immutable projector invariants for one query/prefix pair."""

    prefix_length: int
    balls: tuple[_AffineBall, ...]
    unique_balls: tuple[_AffineBall, ...]
    unique_ball_grams: tuple[np.ndarray, ...]
    raw_physical: PhysicalContinuityComponents
    terminal_constraint_mode: str = TERMINAL_CONSTRAINT_PREFIX_EXIT


class ProjectorQueryContextCache:
    """Single-query, single-threaded lazy cache with immutable published entries."""

    def __init__(
        self,
        raw: np.ndarray,
        observed_displacements: np.ndarray,
        response: ResponseFit,
        *,
        terminal_constraint_mode: str = TERMINAL_CONSTRAINT_PREFIX_EXIT,
    ) -> None:
        raw_value = np.asarray(raw)
        observed_value = np.asarray(observed_displacements, dtype=np.float64)
        matrix_value = _response_matrix(response)
        if (
            not valid_remaining_shape(raw_value, 7)
            or not np.issubdtype(raw_value.dtype, np.floating)
            or not np.isfinite(raw_value).all()
            or observed_value.shape != (2, 3)
            or not np.isfinite(observed_value).all()
            or terminal_constraint_mode not in TERMINAL_CONSTRAINT_MODES
        ):
            raise ValueError("invalid projector query context")
        # Preserve the frozen Q3 cache arithmetic exactly, including its
        # controlled deferred handling of finite-input matmul overflow.  Q6
        # adds only the alternative-specific rotation offsets to that same
        # translation prediction.
        if (response.prediction_semantics == RESPONSE_PREDICTION_Q6_ALTERNATIVE_OFFSETS
                and response.raw_prediction_offset.shape[0] != len(raw_value)):
            raise ValueError("Q6 response offsets must match the actual remaining horizon")
        translated_value = raw_value[:, :3].astype(np.float64) @ matrix_value
        if response.prediction_semantics == RESPONSE_PREDICTION_Q6_ALTERNATIVE_OFFSETS:
            raw_offset_value = np.asarray(response.raw_prediction_offset, dtype=np.float64)
            selected_offset_value = np.asarray(
                response.selected_prediction_offset, dtype=np.float64
            )
            predicted_raw_value = translated_value + raw_offset_value
            predicted_selected_base_value = translated_value + selected_offset_value
        else:
            selected_offset_value = None
            predicted_raw_value = translated_value
            predicted_selected_base_value = translated_value
        self.temporal_response = response.temporal_response
        if response.prediction_semantics == RESPONSE_PREDICTION_Q3_MEMORY:
            if not np.array_equal(self.temporal_response.initial_displacement, observed_value[-1]):
                raise ValueError("MEMORY_INITIAL_STATE_MISMATCH")
            predicted_raw_value = self.temporal_response.predict(raw_value[:, :3])
            predicted_selected_base_value = predicted_raw_value
        self.raw = _readonly_array(raw_value)
        self.observed_displacements = _readonly_array(observed_value)
        self.response_matrix = _readonly_array(matrix_value)
        self.predicted_raw = _readonly_array(predicted_raw_value)
        self.predicted_selected_base = _readonly_array(predicted_selected_base_value)
        self.response_prediction_semantics = response.prediction_semantics
        self.selected_prediction_offset = (
            None
            if selected_offset_value is None
            else _readonly_array(selected_offset_value)
        )
        self.identity3 = _readonly_array(np.eye(3, dtype=np.float64))
        self.terminal_constraint_mode = terminal_constraint_mode
        self._entries: dict[int, PrefixProjectionContext] = {}
        self._failures: dict[int, str] = {}
        self._build_counts: dict[int, int] = {}
        self._hit_counts: dict[int, int] = {}

    def _validate_prefix_length(self, prefix_length: int) -> None:
        if (
            isinstance(prefix_length, bool)
            or not isinstance(prefix_length, int)
            or not 1 <= prefix_length <= min(len(self.raw), HORIZON - 1)
        ):
            raise ValueError("invalid physical continuity inputs")

    def _for_prefix(
        self,
        prefix_length: int,
        *,
        include_terminal_projection: bool,
    ) -> PrefixProjectionContext:
        self._validate_prefix_length(prefix_length)
        if prefix_length in self._entries:
            expected_ball_count = 4 if include_terminal_projection else 3
            if len(self._entries[prefix_length].balls) != expected_ball_count:
                raise ValueError("mixed terminal projection role in one query cache")
            self._hit_counts[prefix_length] = self._hit_counts.get(prefix_length, 0) + 1
            return self._entries[prefix_length]
        if prefix_length in self._failures:
            self._hit_counts[prefix_length] = self._hit_counts.get(prefix_length, 0) + 1
            raise ValueError(self._failures[prefix_length])
        self._build_counts[prefix_length] = self._build_counts.get(prefix_length, 0) + 1
        try:
            if self.temporal_response is not None:
                from chunkamend.simulation.core.methods.temporal_affine import build_prefix_constraints
                memory_prefix = build_prefix_constraints(self.temporal_response, self.raw[:, :3],
                    self.observed_displacements, prefix_length=prefix_length,
                    terminal_constraint_mode=self.terminal_constraint_mode)
                built = memory_prefix.projection_balls + (memory_prefix.terminal_ball,)
                vectors = _physical_vectors(self.predicted_raw, self.observed_displacements,
                    prefix_length=prefix_length, terminal_constraint_mode=self.terminal_constraint_mode)
            else:
                built, vectors = _affine_balls_from_query_invariants(
                    self.response_matrix, self.predicted_raw, self.predicted_selected_base,
                    self.observed_displacements, prefix_length=prefix_length,
                    terminal_constraint_mode=self.terminal_constraint_mode,
                )
            projected_balls = built if include_terminal_projection else built[:3]
            balls = tuple(
                _AffineBall(
                    item.name,
                    _readonly_array(item.matrix),
                    _readonly_array(item.offset),
                    item.radius,
                )
                for item in projected_balls
            )
            unique_balls = _unique_balls(balls)
            grams = tuple(
                _readonly_array(item.matrix @ item.matrix.T) for item in unique_balls
            )
            norms = tuple(float(np.linalg.norm(vector)) for vector in vectors)
            if not np.isfinite(norms).all():
                raise ValueError("nonfinite physical continuity component")
            entry = PrefixProjectionContext(
                prefix_length=prefix_length,
                balls=balls,
                unique_balls=unique_balls,
                unique_ball_grams=grams,
                raw_physical=PhysicalContinuityComponents(
                    *norms,
                    terminal_constraint_mode=self.terminal_constraint_mode,
                ),
                terminal_constraint_mode=self.terminal_constraint_mode,
            )
        except ValueError as error:
            status = str(error)
            self._failures[prefix_length] = status
            raise ValueError(status) from error
        self._entries[prefix_length] = entry
        return entry

    def for_prefix(self, prefix_length: int) -> PrefixProjectionContext:
        """Return the frozen four-ball projection context used by REF."""

        return self._for_prefix(prefix_length, include_terminal_projection=True)

    def for_prefix_validator_only(self, prefix_length: int) -> PrefixProjectionContext:
        """Return the cached three-core-ball context used by VALIDATOR_ONLY."""

        return self._for_prefix(prefix_length, include_terminal_projection=False)

    def for_prefix_physical(self, prefix_length: int) -> PrefixProjectionContext:
        """Reuse either arm's prefix entry for the unchanged final physical veto."""

        self._validate_prefix_length(prefix_length)
        if prefix_length in self._entries:
            self._hit_counts[prefix_length] = self._hit_counts.get(prefix_length, 0) + 1
            return self._entries[prefix_length]
        return self.for_prefix(prefix_length)

    def receipt(self) -> dict[str, object]:
        return {
            "initialized": True,
            "builds_by_k": tuple(sorted(self._build_counts.items())),
            "hits_by_k": tuple(sorted(self._hit_counts.items())),
            "gram_build_count": sum(
                len(entry.unique_ball_grams) for entry in self._entries.values()
            ),
            "failure_by_k": tuple(sorted(self._failures.items())),
        }


def _project_affine_ball(point: np.ndarray, ball: _AffineBall, *, tolerance: float) -> np.ndarray | None:
    value = ball.matrix @ point + ball.offset
    if not np.isfinite(value).all() or ball.radius < 0:
        return None
    if float(np.linalg.norm(value)) <= ball.radius + tolerance:
        return point.copy()
    gram = ball.matrix @ ball.matrix.T

    def projected(multiplier: float) -> tuple[np.ndarray, float] | None:
        try:
            residual = np.linalg.solve(np.eye(3) + multiplier * gram, value)
        except np.linalg.LinAlgError:
            return None
        candidate = point - multiplier * ball.matrix.T @ residual
        norm = float(np.linalg.norm(ball.matrix @ candidate + ball.offset))
        return (candidate, norm) if np.isfinite(candidate).all() and np.isfinite(norm) else None

    upper = 1.0
    result = projected(upper)
    while result is not None and result[1] > ball.radius + tolerance and upper < 1e12:
        upper *= 2
        result = projected(upper)
    if result is None or result[1] > ball.radius + tolerance:
        return None
    lower = 0.0
    for _ in range(80):
        middle = (lower + upper) / 2
        result = projected(middle)
        if result is None:
            return None
        if result[1] > ball.radius:
            lower = middle
        else:
            upper = middle
        if upper - lower <= tolerance * max(1.0, upper):
            break
    result = projected(upper)
    return None if result is None else result[0]


def _project_affine_ball_cached(
    point: np.ndarray,
    ball: _AffineBall,
    gram: np.ndarray,
    identity3: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray | None:
    """Byte-equivalent ball projection with query/prefix invariants supplied read-only."""

    value = ball.matrix @ point + ball.offset
    if not np.isfinite(value).all() or ball.radius < 0:
        return None
    if float(np.linalg.norm(value)) <= ball.radius + tolerance:
        return point.copy()

    def projected(multiplier: float) -> tuple[np.ndarray, float] | None:
        try:
            residual = np.linalg.solve(identity3 + multiplier * gram, value)
        except np.linalg.LinAlgError:
            return None
        candidate = point - multiplier * ball.matrix.T @ residual
        norm = float(np.linalg.norm(ball.matrix @ candidate + ball.offset))
        return (candidate, norm) if np.isfinite(candidate).all() and np.isfinite(norm) else None

    upper = 1.0
    result = projected(upper)
    while result is not None and result[1] > ball.radius + tolerance and upper < 1e12:
        upper *= 2
        result = projected(upper)
    if result is None or result[1] > ball.radius + tolerance:
        return None
    lower = 0.0
    for _ in range(80):
        middle = (lower + upper) / 2
        result = projected(middle)
        if result is None:
            return None
        if result[1] > ball.radius:
            lower = middle
        else:
            upper = middle
        if upper - lower <= tolerance * max(1.0, upper):
            break
    result = projected(upper)
    return None if result is None else result[0]


def _project_affine_ball_cached_terminal_endpoint_reuse(
    point: np.ndarray,
    ball: _AffineBall,
    gram: np.ndarray,
    identity3: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray | None:
    """Reuse the already evaluated terminal upper endpoint without changing the search."""

    used,compiled=compiled_projection.project(point,ball.matrix,ball.offset,gram,identity3,ball.radius,tolerance)
    if used:return compiled

    value = ball.matrix @ point + ball.offset
    if not np.isfinite(value).all() or ball.radius < 0:
        return None
    if float(np.linalg.norm(value)) <= ball.radius + tolerance:
        return point.copy()

    def projected(multiplier: float) -> tuple[np.ndarray, float] | None:
        try:
            residual = solve3_f64(identity3 + multiplier * gram, value)
        except np.linalg.LinAlgError:
            return None
        candidate = point - multiplier * ball.matrix.T @ residual
        norm = float(np.linalg.norm(ball.matrix @ candidate + ball.offset))
        return (candidate, norm) if np.isfinite(candidate).all() and np.isfinite(norm) else None

    upper = 1.0
    result = projected(upper)
    while result is not None and result[1] > ball.radius + tolerance and upper < 1e12:
        upper *= 2
        result = projected(upper)
    if result is None or result[1] > ball.radius + tolerance:
        return None
    upper_result = result
    lower = 0.0
    for _ in range(80):
        middle = (lower + upper) / 2
        result = projected(middle)
        if result is None:
            return None
        if result[1] > ball.radius:
            lower = middle
        else:
            upper = middle
            upper_result = result
        if upper - lower <= tolerance * max(1.0, upper):
            break
    return upper_result[0]


def _max_violation(point: np.ndarray, lower: np.ndarray, upper: np.ndarray, balls: tuple[_AffineBall, ...]) -> float:
    handled, value = compiled_projection.project_violation(point, lower, upper, balls)
    if handled:
        return value
    values = [float(np.max(lower - point)), float(np.max(point - upper)), 0.0]
    values.extend(float(np.linalg.norm(ball.matrix @ point + ball.offset)) - ball.radius for ball in balls)
    return float(max(values)) if np.isfinite(values).all() else float("inf")


def _slacks(
    point: np.ndarray, lower: np.ndarray, upper: np.ndarray, balls: tuple[_AffineBall, ...]
) -> tuple[tuple[str, float], ...]:
    values = [("box_lower", float(np.min(point - lower))), ("box_upper", float(np.min(upper - point)))]
    values.extend((ball.name, float(ball.radius - np.linalg.norm(ball.matrix @ point + ball.offset))) for ball in balls)
    return tuple(values)


def _bounded_cast(
    raw: np.ndarray, correction: np.ndarray, lower: np.ndarray, upper: np.ndarray, *, prefix_length: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    candidate = raw.copy()
    target = np.asarray(raw[:prefix_length, :3], dtype=np.float64) + correction
    cast = target.astype(raw.dtype)
    raw64 = raw[:prefix_length, :3].astype(np.float64)
    for _ in range(3):
        actual = cast.astype(np.float64) - raw64
        below, above = actual < lower, actual > upper
        if not np.any(below | above):
            candidate[:prefix_length, :3] = cast
            return candidate, actual
        cast = np.where(below, np.nextafter(cast, np.asarray(np.inf, dtype=raw.dtype)), cast)
        cast = np.where(above, np.nextafter(cast, np.asarray(-np.inf, dtype=raw.dtype)), cast)
    return None, None


def project_execution_aware_candidate(
    raw: np.ndarray,
    bridge_candidate: np.ndarray,
    observed_displacements: np.ndarray,
    response: ResponseFit,
    *,
    prefix_length: int,
    lower_correction: np.ndarray,
    upper_correction: np.ndarray,
    max_iterations: int = MAX_PROJECTION_CYCLES,
    tolerance: float = PROJECTION_TOLERANCE,
) -> ExecutionProjectionResult:
    """Deterministic cyclic feasibility projection over four balls then box."""

    raw_value, bridge = np.asarray(raw), np.asarray(bridge_candidate)
    lower, upper = np.asarray(lower_correction, dtype=np.float64), np.asarray(upper_correction, dtype=np.float64)
    if (
        not valid_remaining_shape(raw_value, 7)
        or bridge.shape != raw_value.shape
        or not np.issubdtype(raw_value.dtype, np.floating)
        or bridge.dtype != raw_value.dtype
        or not np.isfinite(raw_value).all()
        or not np.isfinite(bridge).all()
        or isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or not 1 <= prefix_length <= min(len(raw_value), HORIZON - 1)
        or lower.shape != (prefix_length, 3)
        or upper.shape != lower.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
        or max_iterations != MAX_PROJECTION_CYCLES
        or tolerance != PROJECTION_TOLERANCE
        or not np.array_equal(bridge[prefix_length:], raw_value[prefix_length:])
        or not np.array_equal(bridge[:, 3:], raw_value[:, 3:])
    ):
        return ExecutionProjectionResult(None, "INVALID_PROJECTION_INPUT", 0, float("inf"))
    target = (bridge[:prefix_length, :3].astype(np.float64) - raw_value[:prefix_length, :3]).reshape(-1)
    lower_flat, upper_flat = lower.reshape(-1), upper.reshape(-1)
    if np.any(target < lower_flat - tolerance) or np.any(target > upper_flat + tolerance):
        return ExecutionProjectionResult(None, "BRIDGE_OUTSIDE_CORRECTION_BOX", 0, float("inf"))
    try:
        balls = _affine_balls(raw_value, observed_displacements, response, prefix_length=prefix_length)
    except ValueError as error:
        return ExecutionProjectionResult(None, str(error), 0, float("inf"))

    point = target.copy()
    violation = _max_violation(point, lower_flat, upper_flat, balls)
    iterations = 0
    if violation > tolerance:
        for iterations in range(1, max_iterations + 1):
            for ball in _unique_balls(balls):
                point = _project_affine_ball(point, ball, tolerance=tolerance)
                if point is None:
                    return ExecutionProjectionResult(None, "INDIVIDUAL_SET_FAILURE", iterations, float("inf"))
            point = np.clip(point, lower_flat, upper_flat)
            violation = _max_violation(point, lower_flat, upper_flat, balls)
            if violation <= tolerance:
                break
        else:
            return ExecutionProjectionResult(
                None,
                "PROJECTION_NOT_CONVERGED",
                iterations,
                violation,
                _slacks(point, lower_flat, upper_flat, balls),
            )

    correction = point.reshape(prefix_length, 3)
    candidate, actual = _bounded_cast(raw_value, correction, lower, upper, prefix_length=prefix_length)
    if candidate is None or actual is None:
        return ExecutionProjectionResult(None, "DTYPE_BOX_ROUNDTRIP_FAILURE", iterations, float("inf"))
    final_point = actual.reshape(-1)
    violation = _max_violation(final_point, lower_flat, upper_flat, balls)
    if (
        violation > tolerance
        or not np.array_equal(candidate[prefix_length:], raw_value[prefix_length:])
        or not np.array_equal(candidate[:, 3:], raw_value[:, 3:])
    ):
        return ExecutionProjectionResult(
            None,
            "FINAL_FEASIBILITY_FAILURE",
            iterations,
            violation,
            _slacks(final_point, lower_flat, upper_flat, balls),
        )
    return ExecutionProjectionResult(
        candidate,
        "PROJECTED",
        iterations,
        violation,
        _slacks(final_point, lower_flat, upper_flat, balls),
    )


def project_execution_aware_candidate_cached(
    context: ProjectorQueryContextCache,
    bridge_candidate: np.ndarray,
    *,
    prefix_length: int,
    lower_correction: np.ndarray,
    upper_correction: np.ndarray,
    max_iterations: int = MAX_PROJECTION_CYCLES,
    tolerance: float = PROJECTION_TOLERANCE,
) -> ExecutionProjectionResult:
    """The frozen projection algorithm with query/prefix invariants reused read-only."""

    if not isinstance(context, ProjectorQueryContextCache):
        return ExecutionProjectionResult(None, "INVALID_PROJECTION_INPUT", 0, float("inf"))
    raw_value, bridge = context.raw, np.asarray(bridge_candidate)
    lower = np.asarray(lower_correction, dtype=np.float64)
    upper = np.asarray(upper_correction, dtype=np.float64)
    if (
        not valid_remaining_shape(raw_value, 7)
        or bridge.shape != raw_value.shape
        or not np.issubdtype(raw_value.dtype, np.floating)
        or bridge.dtype != raw_value.dtype
        or not np.isfinite(raw_value).all()
        or not np.isfinite(bridge).all()
        or isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or not 1 <= prefix_length <= min(len(raw_value), HORIZON - 1)
        or lower.shape != (prefix_length, 3)
        or upper.shape != lower.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
        or max_iterations != MAX_PROJECTION_CYCLES
        or tolerance != PROJECTION_TOLERANCE
        or not np.array_equal(bridge[prefix_length:], raw_value[prefix_length:])
        or not np.array_equal(bridge[:, 3:], raw_value[:, 3:])
    ):
        return ExecutionProjectionResult(None, "INVALID_PROJECTION_INPUT", 0, float("inf"))
    target = (
        bridge[:prefix_length, :3].astype(np.float64)
        - raw_value[:prefix_length, :3]
    ).reshape(-1)
    lower_flat, upper_flat = lower.reshape(-1), upper.reshape(-1)
    if np.any(target < lower_flat - tolerance) or np.any(target > upper_flat + tolerance):
        return ExecutionProjectionResult(None, "BRIDGE_OUTSIDE_CORRECTION_BOX", 0, float("inf"))
    try:
        prefix = context.for_prefix(prefix_length)
    except ValueError as error:
        return ExecutionProjectionResult(None, str(error), 0, float("inf"))
    balls = prefix.balls
    point = target.copy()
    violation = _max_violation(point, lower_flat, upper_flat, balls)
    iterations = 0
    if violation > tolerance:
        for iterations in range(1, max_iterations + 1):
            for ball, gram in zip(
                prefix.unique_balls,
                prefix.unique_ball_grams,
                strict=True,
            ):
                point = _project_affine_ball_cached(
                    point,
                    ball,
                    gram,
                    context.identity3,
                    tolerance=tolerance,
                )
                if point is None:
                    return ExecutionProjectionResult(
                        None,
                        "INDIVIDUAL_SET_FAILURE",
                        iterations,
                        float("inf"),
                    )
            point = np.clip(point, lower_flat, upper_flat)
            violation = _max_violation(point, lower_flat, upper_flat, balls)
            if violation <= tolerance:
                break
        else:
            return ExecutionProjectionResult(
                None,
                "PROJECTION_NOT_CONVERGED",
                iterations,
                violation,
                _slacks(point, lower_flat, upper_flat, balls),
            )

    correction = point.reshape(prefix_length, 3)
    candidate, actual = _bounded_cast(
        raw_value,
        correction,
        lower,
        upper,
        prefix_length=prefix_length,
    )
    if candidate is None or actual is None:
        return ExecutionProjectionResult(
            None,
            "DTYPE_BOX_ROUNDTRIP_FAILURE",
            iterations,
            float("inf"),
        )
    final_point = actual.reshape(-1)
    violation = _max_violation(final_point, lower_flat, upper_flat, balls)
    if (
        violation > tolerance
        or not np.array_equal(candidate[prefix_length:], raw_value[prefix_length:])
        or not np.array_equal(candidate[:, 3:], raw_value[:, 3:])
    ):
        return ExecutionProjectionResult(
            None,
            "FINAL_FEASIBILITY_FAILURE",
            iterations,
            violation,
            _slacks(final_point, lower_flat, upper_flat, balls),
        )
    return ExecutionProjectionResult(
        candidate,
        "PROJECTED",
        iterations,
        violation,
        _slacks(final_point, lower_flat, upper_flat, balls),
    )


def project_execution_aware_candidate_cached_terminal_endpoint_reuse(
    context: ProjectorQueryContextCache,
    bridge_candidate: np.ndarray,
    *,
    prefix_length: int,
    lower_correction: np.ndarray,
    upper_correction: np.ndarray,
    max_iterations: int = MAX_PROJECTION_CYCLES,
    tolerance: float = PROJECTION_TOLERANCE,
) -> ExecutionProjectionResult:
    """The cached projector with only terminal upper-endpoint recomputation removed."""

    if not isinstance(context, ProjectorQueryContextCache):
        return ExecutionProjectionResult(None, "INVALID_PROJECTION_INPUT", 0, float("inf"))
    raw_value, bridge = context.raw, np.asarray(bridge_candidate)
    lower = np.asarray(lower_correction, dtype=np.float64)
    upper = np.asarray(upper_correction, dtype=np.float64)
    if (
        not valid_remaining_shape(raw_value, 7)
        or bridge.shape != raw_value.shape
        or not np.issubdtype(raw_value.dtype, np.floating)
        or bridge.dtype != raw_value.dtype
        or not np.isfinite(raw_value).all()
        or not np.isfinite(bridge).all()
        or isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or not 1 <= prefix_length <= min(len(raw_value), HORIZON - 1)
        or lower.shape != (prefix_length, 3)
        or upper.shape != lower.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
        or max_iterations != MAX_PROJECTION_CYCLES
        or tolerance != PROJECTION_TOLERANCE
        or not np.array_equal(bridge[prefix_length:], raw_value[prefix_length:])
        or not np.array_equal(bridge[:, 3:], raw_value[:, 3:])
    ):
        return ExecutionProjectionResult(None, "INVALID_PROJECTION_INPUT", 0, float("inf"))
    target = (
        bridge[:prefix_length, :3].astype(np.float64)
        - raw_value[:prefix_length, :3]
    ).reshape(-1)
    lower_flat, upper_flat = lower.reshape(-1), upper.reshape(-1)
    if np.any(target < lower_flat - tolerance) or np.any(target > upper_flat + tolerance):
        return ExecutionProjectionResult(None, "BRIDGE_OUTSIDE_CORRECTION_BOX", 0, float("inf"))
    try:
        prefix = context.for_prefix(prefix_length)
    except ValueError as error:
        return ExecutionProjectionResult(None, str(error), 0, float("inf"))
    balls = prefix.balls
    point = target.copy()
    violation = _max_violation(point, lower_flat, upper_flat, balls)
    iterations = 0
    if violation > tolerance:
        for iterations in range(1, max_iterations + 1):
            for ball, gram in zip(
                prefix.unique_balls,
                prefix.unique_ball_grams,
                strict=True,
            ):
                point = _project_affine_ball_cached_terminal_endpoint_reuse(
                    point,
                    ball,
                    gram,
                    context.identity3,
                    tolerance=tolerance,
                )
                if point is None:
                    return ExecutionProjectionResult(
                        None,
                        "INDIVIDUAL_SET_FAILURE",
                        iterations,
                        float("inf"),
                    )
            point = np.clip(point, lower_flat, upper_flat)
            violation = _max_violation(point, lower_flat, upper_flat, balls)
            if violation <= tolerance:
                break
        else:
            return ExecutionProjectionResult(
                None,
                "PROJECTION_NOT_CONVERGED",
                iterations,
                violation,
                _slacks(point, lower_flat, upper_flat, balls),
            )

    correction = point.reshape(prefix_length, 3)
    candidate, actual = _bounded_cast(
        raw_value,
        correction,
        lower,
        upper,
        prefix_length=prefix_length,
    )
    if candidate is None or actual is None:
        return ExecutionProjectionResult(
            None,
            "DTYPE_BOX_ROUNDTRIP_FAILURE",
            iterations,
            float("inf"),
        )
    final_point = actual.reshape(-1)
    violation = _max_violation(final_point, lower_flat, upper_flat, balls)
    if (
        violation > tolerance
        or not np.array_equal(candidate[prefix_length:], raw_value[prefix_length:])
        or not np.array_equal(candidate[:, 3:], raw_value[:, 3:])
    ):
        return ExecutionProjectionResult(
            None,
            "FINAL_FEASIBILITY_FAILURE",
            iterations,
            violation,
            _slacks(final_point, lower_flat, upper_flat, balls),
        )
    return ExecutionProjectionResult(
        candidate,
        "PROJECTED",
        iterations,
        violation,
        _slacks(final_point, lower_flat, upper_flat, balls),
    )


def project_execution_aware_candidate_cached_validator_only(
    context: ProjectorQueryContextCache,
    bridge_candidate: np.ndarray,
    *,
    prefix_length: int,
    lower_correction: np.ndarray,
    upper_correction: np.ndarray,
    max_iterations: int = MAX_PROJECTION_CYCLES,
    tolerance: float = PROJECTION_TOLERANCE,
) -> ExecutionProjectionResult:
    """Project against the physical core while leaving terminal to the final veto.

    This is the prospective Task143 implementation of the Task141/142
    VALIDATOR_ONLY arm.  It inherits the accepted endpoint-reuse cyclic projector; the optional
    exact-cycle context skips byte-identical repetitions and records actual work.
    The fourth (terminal internal jerk) affine
    ball is omitted from projection and final projector feasibility.  The
    unchanged caller still evaluates all four physical components and rejects
    a terminal failure before candidate selection.
    """

    if not isinstance(context, ProjectorQueryContextCache):
        return ExecutionProjectionResult(None, "INVALID_PROJECTION_INPUT", 0, float("inf"))
    raw_value, bridge = context.raw, np.asarray(bridge_candidate)
    lower = np.asarray(lower_correction, dtype=np.float64)
    upper = np.asarray(upper_correction, dtype=np.float64)
    if (
        not valid_remaining_shape(raw_value, 7)
        or bridge.shape != raw_value.shape
        or not np.issubdtype(raw_value.dtype, np.floating)
        or bridge.dtype != raw_value.dtype
        or not np.isfinite(raw_value).all()
        or not np.isfinite(bridge).all()
        or isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or not 1 <= prefix_length <= min(len(raw_value), HORIZON - 1)
        or lower.shape != (prefix_length, 3)
        or upper.shape != lower.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
        or max_iterations != MAX_PROJECTION_CYCLES
        or tolerance != PROJECTION_TOLERANCE
        or not np.array_equal(bridge[prefix_length:], raw_value[prefix_length:])
        or not np.array_equal(bridge[:, 3:], raw_value[:, 3:])
    ):
        return ExecutionProjectionResult(None, "INVALID_PROJECTION_INPUT", 0, float("inf"))
    target = (
        bridge[:prefix_length, :3].astype(np.float64)
        - raw_value[:prefix_length, :3]
    ).reshape(-1)
    lower_flat, upper_flat = lower.reshape(-1), upper.reshape(-1)
    if np.any(target < lower_flat - tolerance) or np.any(target > upper_flat + tolerance):
        return ExecutionProjectionResult(None, "BRIDGE_OUTSIDE_CORRECTION_BOX", 0, float("inf"))
    try:
        prefix = context.for_prefix_validator_only(prefix_length)
    except ValueError as error:
        return ExecutionProjectionResult(None, str(error), 0, float("inf"))
    cycle_work = exact_cycle.current_work()
    if cycle_work is not None:
        cycle_work["valid_projector_calls"] += 1
    balls = prefix.balls
    point = target.copy()
    violation = _max_violation(point, lower_flat, upper_flat, balls)
    iterations = 0
    if violation > tolerance:
        if box_certificate.reject_if_certified(point, lower_flat, upper_flat, balls, tolerance=tolerance, iteration=0):
            return ExecutionProjectionResult(None, "PROVEN_BOX_BALL_INFEASIBLE", 0, violation, _slacks(point, lower_flat, upper_flat, balls))
        for iterations in range(1, max_iterations + 1):
            if cycle_work is not None:
                cycle_work["executed_cycles"] += 1
                cycle_input_bytes = point.tobytes()
            for ball, gram in zip(
                prefix.unique_balls,
                prefix.unique_ball_grams,
                strict=True,
            ):
                point = _project_affine_ball_cached_terminal_endpoint_reuse(
                    point,
                    ball,
                    gram,
                    context.identity3,
                    tolerance=tolerance,
                )
                if point is None:
                    return ExecutionProjectionResult(
                        None, "INDIVIDUAL_SET_FAILURE", iterations, float("inf")
                    )
            point = np.clip(point, lower_flat, upper_flat)
            violation = _max_violation(point, lower_flat, upper_flat, balls)
            if violation <= tolerance:
                break
            if box_certificate.reject_if_certified(point, lower_flat, upper_flat, balls, tolerance=tolerance, iteration=iterations):
                return ExecutionProjectionResult(None, "PROVEN_BOX_BALL_INFEASIBLE", iterations, violation, _slacks(point, lower_flat, upper_flat, balls))
            if cycle_work is not None and point.tobytes() == cycle_input_bytes:
                omitted = max_iterations - iterations
                cycle_work["skipped_cycles"] += omitted
                cycle_work["fixed_points"].append(dict(
                    projector_call=cycle_work["valid_projector_calls"],
                    executed_cycles=iterations, skipped_cycles=omitted,
                    equivalent_baseline_cycles=max_iterations, violation=violation))
                return ExecutionProjectionResult(
                    None, "PROJECTION_NOT_CONVERGED", max_iterations, violation,
                    _slacks(point, lower_flat, upper_flat, balls),
                )
        else:
            return ExecutionProjectionResult(
                None,
                "PROJECTION_NOT_CONVERGED",
                iterations,
                violation,
                _slacks(point, lower_flat, upper_flat, balls),
            )
    candidate, actual = _bounded_cast(
        raw_value,
        point.reshape(prefix_length, 3),
        lower,
        upper,
        prefix_length=prefix_length,
    )
    if candidate is None or actual is None:
        return ExecutionProjectionResult(
            None, "DTYPE_BOX_ROUNDTRIP_FAILURE", iterations, float("inf")
        )
    final_point = actual.reshape(-1)
    violation = _max_violation(final_point, lower_flat, upper_flat, balls)
    if (
        violation > tolerance
        or not np.array_equal(candidate[prefix_length:], raw_value[prefix_length:])
        or not np.array_equal(candidate[:, 3:], raw_value[:, 3:])
    ):
        return ExecutionProjectionResult(
            None,
            "FINAL_FEASIBILITY_FAILURE",
            iterations,
            violation,
            _slacks(final_point, lower_flat, upper_flat, balls),
        )
    return ExecutionProjectionResult(
        candidate,
        "PROJECTED",
        iterations,
        violation,
        _slacks(final_point, lower_flat, upper_flat, balls),
    )


def _dykstra_cycles(
    start: np.ndarray,
    balls: tuple[_AffineBall, ...],
    grams: tuple[np.ndarray, ...],
    identity3: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_cycles: int,
    tolerance: float,
) -> tuple[np.ndarray | None, int, float, str]:
    """Project onto the intersection of the same affine balls and box."""

    point = np.asarray(start, dtype=np.float64).copy()
    ball_residuals = [np.zeros_like(point) for _ in balls]
    box_residual = np.zeros_like(point)
    violation = float("inf")
    for cycle in range(1, max_cycles + 1):
        for index, (ball, gram) in enumerate(zip(balls, grams, strict=True)):
            shifted = point + ball_residuals[index]
            projected = _project_affine_ball_cached_terminal_endpoint_reuse(
                shifted,
                ball,
                gram,
                identity3,
                tolerance=1e-12,
            )
            if projected is None:
                return None, cycle, float("inf"), "INDIVIDUAL_SET_FAILURE"
            ball_residuals[index] = shifted - projected
            point = projected
        shifted = point + box_residual
        projected = np.clip(shifted, lower, upper)
        box_residual = shifted - projected
        point = projected
        violation = _max_violation(point, lower, upper, balls)
        if violation <= tolerance:
            return point, cycle, violation, "PROJECTED"
    return point, max_cycles, violation, "PROJECTION_NOT_CONVERGED"


def _cyclic_completion_cycles(
    start: np.ndarray,
    balls: tuple[_AffineBall, ...],
    grams: tuple[np.ndarray, ...],
    identity3: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_cycles: int,
    tolerance: float,
) -> tuple[np.ndarray | None, int, float, str]:
    """Complete a Dykstra warm start with the accepted cyclic set order."""

    point = np.asarray(start, dtype=np.float64).copy()
    violation = _max_violation(point, lower, upper, balls)
    for cycle in range(1, max_cycles + 1):
        for ball, gram in zip(balls, grams, strict=True):
            point = _project_affine_ball_cached_terminal_endpoint_reuse(
                point,
                ball,
                gram,
                identity3,
                tolerance=tolerance,
            )
            if point is None:
                return None, cycle, float("inf"), "INDIVIDUAL_SET_FAILURE"
        point = np.clip(point, lower, upper)
        violation = _max_violation(point, lower, upper, balls)
        if violation <= tolerance:
            return point, cycle, violation, "PROJECTED"
    return point, max_cycles, violation, "PROJECTION_NOT_CONVERGED"


def _repair_representable_point(
    raw: np.ndarray,
    candidate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    balls: tuple[_AffineBall, ...],
    *,
    prefix_length: int,
    tolerance: float,
    max_steps: int = 256,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """Greedily move by raw-dtype ULPs until original constraints pass."""

    cast = candidate[:prefix_length, :3].copy().reshape(-1)
    raw64 = raw[:prefix_length, :3].astype(np.float64).reshape(-1)

    def evaluate(values: np.ndarray) -> tuple[tuple[float, float], np.ndarray]:
        point = values.astype(np.float64) - raw64
        violations = [float(np.max(lower - point)), float(np.max(point - upper)), 0.0]
        violations.extend(
            float(np.linalg.norm(ball.matrix @ point + ball.offset)) - ball.radius
            for ball in balls
        )
        positive = np.maximum(np.asarray(violations, dtype=np.float64), 0.0)
        return (float(np.max(positive)), float(np.dot(positive, positive))), point

    score, point = evaluate(cast)
    steps = 0
    neighbor_evaluations = 0
    while score[0] > tolerance and steps < max_steps:
        best_score = score
        best_cast = None
        best_point = None
        for index in range(cast.size):
            for direction in (-np.inf, np.inf):
                trial = cast.copy()
                trial[index] = np.nextafter(
                    trial[index], np.asarray(direction, dtype=raw.dtype)
                )
                if not np.isfinite(trial[index]):
                    continue
                neighbor_evaluations += 1
                trial_score, trial_point = evaluate(trial)
                if trial_score < best_score:
                    best_score = trial_score
                    best_cast = trial
                    best_point = trial_point
        if best_cast is None or best_point is None:
            break
        cast, point, score = best_cast, best_point, best_score
        steps += 1
    repaired = candidate.copy()
    repaired[:prefix_length, :3] = cast.reshape(prefix_length, 3)
    return repaired, point, score[0], steps, neighbor_evaluations


def project_execution_aware_candidate_cached_dykstra(
    context: ProjectorQueryContextCache,
    bridge_candidate: np.ndarray,
    *,
    prefix_length: int,
    lower_correction: np.ndarray,
    upper_correction: np.ndarray,
    max_iterations: int = MAX_PROJECTION_CYCLES,
    tolerance: float = PROJECTION_TOLERANCE,
) -> ExecutionProjectionResult:
    """Deterministic Dykstra projection over the frozen affine balls and box.

    Acceptance is always checked against the original constraints.  If the
    float64 solution is lost only during the raw-dtype round trip, a bounded
    deterministic ULP repair searches neighboring representable values; this
    changes neither the accepted feasible set nor the physical veto.
    """

    if not isinstance(context, ProjectorQueryContextCache):
        return ExecutionProjectionResult(None, "INVALID_PROJECTION_INPUT", 0, float("inf"))
    raw_value, bridge = context.raw, np.asarray(bridge_candidate)
    lower = np.asarray(lower_correction, dtype=np.float64)
    upper = np.asarray(upper_correction, dtype=np.float64)
    if (
        not valid_remaining_shape(raw_value, 7)
        or bridge.shape != raw_value.shape
        or not np.issubdtype(raw_value.dtype, np.floating)
        or bridge.dtype != raw_value.dtype
        or not np.isfinite(raw_value).all()
        or not np.isfinite(bridge).all()
        or isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or not 1 <= prefix_length <= min(len(raw_value), HORIZON - 1)
        or lower.shape != (prefix_length, 3)
        or upper.shape != lower.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
        or max_iterations != MAX_PROJECTION_CYCLES
        or tolerance != PROJECTION_TOLERANCE
        or not np.array_equal(bridge[prefix_length:], raw_value[prefix_length:])
        or not np.array_equal(bridge[:, 3:], raw_value[:, 3:])
    ):
        return ExecutionProjectionResult(None, "INVALID_PROJECTION_INPUT", 0, float("inf"))
    target = (
        bridge[:prefix_length, :3].astype(np.float64)
        - raw_value[:prefix_length, :3]
    ).reshape(-1)
    lower_flat, upper_flat = lower.reshape(-1), upper.reshape(-1)
    if np.any(target < lower_flat - tolerance) or np.any(target > upper_flat + tolerance):
        return ExecutionProjectionResult(None, "BRIDGE_OUTSIDE_CORRECTION_BOX", 0, float("inf"))
    try:
        prefix = context.for_prefix(prefix_length)
    except ValueError as error:
        return ExecutionProjectionResult(None, str(error), 0, float("inf"))
    balls = prefix.balls
    unique = prefix.unique_balls
    initial_violation = _max_violation(target, lower_flat, upper_flat, balls)
    if initial_violation <= tolerance:
        point, iterations, status = target, 0, "PROJECTED"
    else:
        point, iterations, _, status = _dykstra_cycles(
            target,
            unique,
            prefix.unique_ball_grams,
            context.identity3,
            lower_flat,
            upper_flat,
            max_cycles=min(DYKSTRA_PRIMARY_CYCLES, max_iterations),
            tolerance=tolerance,
        )
        if point is not None and status == "PROJECTION_NOT_CONVERGED" and iterations < max_iterations:
            completion, extra, _, status = _cyclic_completion_cycles(
                point,
                unique,
                prefix.unique_ball_grams,
                context.identity3,
                lower_flat,
                upper_flat,
                max_cycles=max_iterations - iterations,
                tolerance=tolerance,
            )
            point = completion
            iterations += extra
    if point is None:
        return ExecutionProjectionResult(None, status, iterations, float("inf"))
    if status != "PROJECTED":
        candidate, actual = _bounded_cast(
            raw_value,
            point.reshape(prefix_length, 3),
            lower,
            upper,
            prefix_length=prefix_length,
        )
        if candidate is not None and actual is not None:
            candidate, final_point, violation, _, _ = _repair_representable_point(
                raw_value,
                candidate,
                lower_flat,
                upper_flat,
                balls,
                prefix_length=prefix_length,
                tolerance=tolerance,
            )
            if violation <= tolerance:
                return ExecutionProjectionResult(
                    candidate,
                    "PROJECTED",
                    iterations,
                    violation,
                    _slacks(final_point, lower_flat, upper_flat, balls),
                )
        violation = _max_violation(point, lower_flat, upper_flat, balls)
        return ExecutionProjectionResult(
            None,
            status,
            iterations,
            violation,
            _slacks(point, lower_flat, upper_flat, balls),
        )

    def finalize(value: np.ndarray):
        candidate, actual = _bounded_cast(
            raw_value,
            value.reshape(prefix_length, 3),
            lower,
            upper,
            prefix_length=prefix_length,
        )
        if candidate is None or actual is None:
            return None, None, float("inf")
        final = actual.reshape(-1)
        return candidate, final, _max_violation(final, lower_flat, upper_flat, balls)

    candidate, final_point, violation = finalize(point)
    locality_ok = bool(
        candidate is not None
        and np.array_equal(candidate[prefix_length:], raw_value[prefix_length:])
        and np.array_equal(candidate[:, 3:], raw_value[:, 3:])
    )
    if violation > tolerance or not locality_ok:
        if candidate is not None and final_point is not None:
            candidate, final_point, violation, _, _ = _repair_representable_point(
                raw_value,
                candidate,
                lower_flat,
                upper_flat,
                balls,
                prefix_length=prefix_length,
                tolerance=tolerance,
            )
            locality_ok = bool(
                np.array_equal(candidate[prefix_length:], raw_value[prefix_length:])
                and np.array_equal(candidate[:, 3:], raw_value[:, 3:])
            )
    if candidate is None or final_point is None:
        return ExecutionProjectionResult(None, "DTYPE_BOX_ROUNDTRIP_FAILURE", iterations, float("inf"))
    if violation > tolerance or not locality_ok:
        return ExecutionProjectionResult(
            None,
            "FINAL_FEASIBILITY_FAILURE",
            iterations,
            violation,
            _slacks(final_point, lower_flat, upper_flat, balls),
        )
    return ExecutionProjectionResult(
        candidate,
        "PROJECTED",
        iterations,
        violation,
        _slacks(final_point, lower_flat, upper_flat, balls),
    )


def _project_execution_aware_candidate_cached_representation_aware(
    context: ProjectorQueryContextCache,
    bridge_candidate: np.ndarray,
    *,
    prefix_length: int,
    lower_correction: np.ndarray,
    upper_correction: np.ndarray,
    max_iterations: int = MAX_PROJECTION_CYCLES,
    tolerance: float = PROJECTION_TOLERANCE,
    repair_nonconverged_endpoint: bool,
) -> ExecutionProjectionResult:
    """Accepted cyclic projector plus configurable raw-dtype feasibility repair.

    The float64 projection algorithm, set order, four affine balls, correction
    box, tolerance, and 128-cycle budget are identical to the accepted Bdev
    projector.  Repair is attempted only after the raw-dtype round trip, or on
    the final representable point after cycle-budget exhaustion; every repaired
    point is rechecked against the original constraints and locality contract.
    """

    if not isinstance(context, ProjectorQueryContextCache):
        return ExecutionProjectionResult(None, "INVALID_PROJECTION_INPUT", 0, float("inf"))
    raw_value, bridge = context.raw, np.asarray(bridge_candidate)
    lower = np.asarray(lower_correction, dtype=np.float64)
    upper = np.asarray(upper_correction, dtype=np.float64)
    if (
        not valid_remaining_shape(raw_value, 7)
        or bridge.shape != raw_value.shape
        or not np.issubdtype(raw_value.dtype, np.floating)
        or bridge.dtype != raw_value.dtype
        or not np.isfinite(raw_value).all()
        or not np.isfinite(bridge).all()
        or isinstance(prefix_length, bool)
        or not isinstance(prefix_length, int)
        or not 1 <= prefix_length <= min(len(raw_value), HORIZON - 1)
        or lower.shape != (prefix_length, 3)
        or upper.shape != lower.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
        or max_iterations != MAX_PROJECTION_CYCLES
        or tolerance != PROJECTION_TOLERANCE
        or not np.array_equal(bridge[prefix_length:], raw_value[prefix_length:])
        or not np.array_equal(bridge[:, 3:], raw_value[:, 3:])
    ):
        return ExecutionProjectionResult(None, "INVALID_PROJECTION_INPUT", 0, float("inf"))
    target = (
        bridge[:prefix_length, :3].astype(np.float64)
        - raw_value[:prefix_length, :3].astype(np.float64)
    ).reshape(-1)
    lower_flat, upper_flat = lower.reshape(-1), upper.reshape(-1)
    if np.any(target < lower_flat - tolerance) or np.any(target > upper_flat + tolerance):
        return ExecutionProjectionResult(None, "BRIDGE_OUTSIDE_CORRECTION_BOX", 0, float("inf"))
    try:
        prefix = context.for_prefix(prefix_length)
    except ValueError as error:
        return ExecutionProjectionResult(None, str(error), 0, float("inf"))

    balls = prefix.balls
    point = target.copy()
    violation = _max_violation(point, lower_flat, upper_flat, balls)
    iterations = 0
    status = "PROJECTED"
    if violation > tolerance:
        status = "PROJECTION_NOT_CONVERGED"
        for iterations in range(1, max_iterations + 1):
            for ball, gram in zip(
                prefix.unique_balls,
                prefix.unique_ball_grams,
                strict=True,
            ):
                point = _project_affine_ball_cached_terminal_endpoint_reuse(
                    point,
                    ball,
                    gram,
                    context.identity3,
                    tolerance=tolerance,
                )
                if point is None:
                    return ExecutionProjectionResult(
                        None,
                        "INDIVIDUAL_SET_FAILURE",
                        iterations,
                        float("inf"),
                        cyclic_completion_cycles=iterations,
                    )
            point = np.clip(point, lower_flat, upper_flat)
            violation = _max_violation(point, lower_flat, upper_flat, balls)
            if violation <= tolerance:
                status = "PROJECTED"
                break

    repair_invocations = 0
    repair_steps = 0
    repair_neighbor_evaluations = 0

    def repair(candidate: np.ndarray):
        nonlocal repair_invocations, repair_steps, repair_neighbor_evaluations
        repair_invocations += 1
        repaired, repaired_point, repaired_violation, steps, evaluations = (
            _repair_representable_point(
                raw_value,
                candidate,
                lower_flat,
                upper_flat,
                balls,
                prefix_length=prefix_length,
                tolerance=tolerance,
            )
        )
        repair_steps += steps
        repair_neighbor_evaluations += evaluations
        return repaired, repaired_point, repaired_violation

    if status != "PROJECTED" and not repair_nonconverged_endpoint:
        return ExecutionProjectionResult(
            None,
            status,
            iterations,
            violation,
            _slacks(point, lower_flat, upper_flat, balls),
            cyclic_completion_cycles=iterations,
        )
    if status != "PROJECTED":
        candidate, actual = _bounded_cast(
            raw_value,
            point.reshape(prefix_length, 3),
            lower,
            upper,
            prefix_length=prefix_length,
        )
        if candidate is not None and actual is not None:
            candidate, final_point, violation = repair(candidate)
            locality_ok = bool(
                np.array_equal(candidate[prefix_length:], raw_value[prefix_length:])
                and np.array_equal(candidate[:, 3:], raw_value[:, 3:])
            )
            if violation <= tolerance and locality_ok:
                return ExecutionProjectionResult(
                    candidate,
                    "PROJECTED",
                    iterations,
                    violation,
                    _slacks(final_point, lower_flat, upper_flat, balls),
                    cyclic_completion_cycles=iterations,
                    representation_repair_invocations=repair_invocations,
                    representation_repair_steps=repair_steps,
                    representation_repair_ulp_neighbor_evaluations=repair_neighbor_evaluations,
                )
        violation = _max_violation(point, lower_flat, upper_flat, balls)
        return ExecutionProjectionResult(
            None,
            status,
            iterations,
            violation,
            _slacks(point, lower_flat, upper_flat, balls),
            cyclic_completion_cycles=iterations,
            representation_repair_invocations=repair_invocations,
            representation_repair_steps=repair_steps,
            representation_repair_ulp_neighbor_evaluations=repair_neighbor_evaluations,
        )

    candidate, actual = _bounded_cast(
        raw_value,
        point.reshape(prefix_length, 3),
        lower,
        upper,
        prefix_length=prefix_length,
    )
    if candidate is None or actual is None:
        return ExecutionProjectionResult(
            None,
            "DTYPE_BOX_ROUNDTRIP_FAILURE",
            iterations,
            float("inf"),
            cyclic_completion_cycles=iterations,
        )
    final_point = actual.reshape(-1)
    violation = _max_violation(final_point, lower_flat, upper_flat, balls)
    locality_ok = bool(
        np.array_equal(candidate[prefix_length:], raw_value[prefix_length:])
        and np.array_equal(candidate[:, 3:], raw_value[:, 3:])
    )
    if violation > tolerance or not locality_ok:
        candidate, final_point, violation = repair(candidate)
        locality_ok = bool(
            np.array_equal(candidate[prefix_length:], raw_value[prefix_length:])
            and np.array_equal(candidate[:, 3:], raw_value[:, 3:])
        )
    if violation > tolerance or not locality_ok:
        return ExecutionProjectionResult(
            None,
            "FINAL_FEASIBILITY_FAILURE",
            iterations,
            violation,
            _slacks(final_point, lower_flat, upper_flat, balls),
            cyclic_completion_cycles=iterations,
            representation_repair_invocations=repair_invocations,
            representation_repair_steps=repair_steps,
            representation_repair_ulp_neighbor_evaluations=repair_neighbor_evaluations,
        )
    return ExecutionProjectionResult(
        candidate,
        "PROJECTED",
        iterations,
        violation,
        _slacks(final_point, lower_flat, upper_flat, balls),
        cyclic_completion_cycles=iterations,
        representation_repair_invocations=repair_invocations,
        representation_repair_steps=repair_steps,
        representation_repair_ulp_neighbor_evaluations=repair_neighbor_evaluations,
    )


def project_execution_aware_candidate_cached_representation_aware(
    context: ProjectorQueryContextCache,
    bridge_candidate: np.ndarray,
    *,
    prefix_length: int,
    lower_correction: np.ndarray,
    upper_correction: np.ndarray,
    max_iterations: int = MAX_PROJECTION_CYCLES,
    tolerance: float = PROJECTION_TOLERANCE,
) -> ExecutionProjectionResult:
    """Task128 Q2 repair, including its nonconverged-endpoint development behavior."""

    return _project_execution_aware_candidate_cached_representation_aware(
        context,
        bridge_candidate,
        prefix_length=prefix_length,
        lower_correction=lower_correction,
        upper_correction=upper_correction,
        max_iterations=max_iterations,
        tolerance=tolerance,
        repair_nonconverged_endpoint=True,
    )


def project_execution_aware_candidate_cached_converged_representation_repair(
    context: ProjectorQueryContextCache,
    bridge_candidate: np.ndarray,
    *,
    prefix_length: int,
    lower_correction: np.ndarray,
    upper_correction: np.ndarray,
    max_iterations: int = MAX_PROJECTION_CYCLES,
    tolerance: float = PROJECTION_TOLERANCE,
) -> ExecutionProjectionResult:
    """Repair final dtype only after the governing cyclic solver converges.

    A cycle-budget hit remains ``PROJECTION_NOT_CONVERGED`` and returns no
    candidate, as required by the M2 failure-closed fallback.  A converged
    continuous point still receives the bounded Task128 ULP repair after its
    raw-dtype round trip, followed by the unchanged original-constraint and
    locality checks.
    """

    return _project_execution_aware_candidate_cached_representation_aware(
        context,
        bridge_candidate,
        prefix_length=prefix_length,
        lower_correction=lower_correction,
        upper_correction=upper_correction,
        max_iterations=max_iterations,
        tolerance=tolerance,
        repair_nonconverged_endpoint=False,
    )


def evaluate_physical_counterfactual_cached(
    context: ProjectorQueryContextCache,
    candidate: np.ndarray,
    *,
    prefix_length: int,
) -> PhysicalCounterfactual:
    """Evaluate the candidate while reusing the immutable raw physical context."""

    if not isinstance(context, ProjectorQueryContextCache):
        raise ValueError("counterfactual context validation failed")
    candidate_value = np.asarray(candidate)
    raw_value = context.raw
    if (
        candidate_value.shape != raw_value.shape
        or not np.isfinite(candidate_value).all()
        or not np.array_equal(candidate_value[prefix_length:], raw_value[prefix_length:])
        or not np.array_equal(candidate_value[:, 3:], raw_value[:, 3:])
    ):
        raise ValueError("counterfactual locality or input validation failed")
    prefix = context.for_prefix_physical(prefix_length)
    # Match the uncached predictor's operation order byte-for-byte.  Reusing
    # ``raw_prediction + correction`` is algebraically equivalent but differs
    # by an ULP on ordinary float fixtures and would perturb the frozen Q3
    # comparator.
    if context.temporal_response is not None:
        predicted = context.temporal_response.predict(candidate_value[:, :3])
    else:
        predicted = candidate_value[:, :3].astype(np.float64) @ context.response_matrix
    if context.response_prediction_semantics == RESPONSE_PREDICTION_Q6_ALTERNATIVE_OFFSETS:
        predicted = predicted + context.selected_prediction_offset
    vectors = _physical_vectors(
        predicted,
        context.observed_displacements,
        prefix_length=prefix_length,
        terminal_constraint_mode=context.terminal_constraint_mode,
    )
    norms = tuple(float(np.linalg.norm(vector)) for vector in vectors)
    if not np.isfinite(norms).all():
        raise ValueError("nonfinite physical continuity component")
    return PhysicalCounterfactual(
        raw=prefix.raw_physical,
        candidate=PhysicalContinuityComponents(
            *norms,
            terminal_constraint_mode=context.terminal_constraint_mode,
        ),
    )


def project_translation_prefix_simplified_ablation(
    raw: np.ndarray, candidate: np.ndarray, *, k: int, response: ResponseFit, max_norm: float
) -> tuple[np.ndarray | None, str]:
    """Historical action-space row clipping for RAP-ACTIONSPACE-SIMPLIFIED-ABLATION."""
    raw_value, candidate_value = np.asarray(raw), np.asarray(candidate)
    if not response.available or response.matrix is None or not np.isfinite(max_norm) or max_norm < 0:
        return None, "PROJECTOR_UNAVAILABLE"
    correction = candidate_value[:k, :3] - raw_value[:k, :3]
    norm = np.linalg.norm(correction, axis=1, keepdims=True)
    output = candidate_value.copy()
    output[:k, :3] = raw_value[:k, :3] + correction * np.minimum(1.0, max_norm / np.maximum(norm, 1e-12))
    return (output, "PROJECTED") if np.isfinite(output).all() else (None, "PROJECTOR_FAILURE")


def continuity_not_worse_simplified_ablation(
    raw: np.ndarray,
    candidate: np.ndarray,
    *,
    previous_action: np.ndarray | None,
    response: ResponseFit | None = None,
) -> bool:
    """Historical one-step comparator for RAP-ACTIONSPACE-SIMPLIFIED-ABLATION."""
    if previous_action is None:
        return True
    previous = np.asarray(previous_action)[:3]
    if previous.shape != (3,) or not np.isfinite(previous).all():
        return False
    raw_delta, candidate_delta = raw[0, :3] - previous, candidate[0, :3] - previous
    if response is not None and response.available and response.matrix is not None:
        if response.prediction_semantics != RESPONSE_PREDICTION_TRANSLATION_ONLY:
            return False
        raw_delta, candidate_delta = raw_delta @ response.matrix, candidate_delta @ response.matrix
    return bool(np.linalg.norm(candidate_delta) <= np.linalg.norm(raw_delta))


# Compatibility names are aliases to the explicitly named nonprimary ablation.
project_translation_prefix = project_translation_prefix_simplified_ablation
continuity_not_worse = continuity_not_worse_simplified_ablation
