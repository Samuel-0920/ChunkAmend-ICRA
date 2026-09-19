"""Finite-horizon affine form of C115's Q3 memory response.

Only the first K normalized translation actions are correction variables.
All later displacement predictions still include their propagated effects.
The terminal constraint remains a final veto, outside the projection balls.
C146 adapts this map to the actual remaining horizon; simulation is not released.
"""
from dataclasses import dataclass
import numpy as np

from chunkamend.hardware.core.methods.memory_response import STATE_SCALE_M
from chunkamend.hardware.core.methods import rap_projector as projector


def _finite(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite {shape}")
    return value


def _owned(value):
    value = np.array(value, dtype=np.float64, copy=True)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class TemporalResponse:
    matrix: np.ndarray
    offset: np.ndarray
    initial_displacement: np.ndarray

    def predict(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if not projector.valid_remaining_shape(actions, 3) or not np.isfinite(actions).all():
            raise ValueError("Invalid remaining actions")
        width = 3 * len(actions)
        value = self.matrix[:width, :width] @ actions.reshape(-1) + self.offset[:width]
        if not np.isfinite(value).all():
            raise ValueError("NONFINITE_MEMORY_PREDICTION")
        return value.reshape(len(actions), 3)


def lift_memory_response(fit, last_completed_displacement):
    """Column-flattened D = L U + b, from row recurrence d_i=u_i B+d_(i-1) A.

    A=C/STATE_SCALE_M; L[i,j]=(B A**(i-j)).T for j<=i.
    b[i]=last_completed_displacement A**(i+1).
    No spectral clipping or new stability threshold is introduced.
    """
    if not fit.get("available", False):
        raise ValueError("UNAVAILABLE_MEMORY_FIT")
    coefficients = _finite(fit["coefficients"], (6, 3), "coefficients")
    initial = _finite(last_completed_displacement, (3,), "initial displacement")
    b_action = coefficients[:3]
    a_state = coefficients[3:] / STATE_SCALE_M
    horizon = projector.HORIZON
    blocks = []
    power = np.eye(3)
    for _ in range(horizon):
        blocks.append((b_action @ power).T)
        power = power @ a_state
    matrix = np.zeros((3 * horizon, 3 * horizon))
    for i in range(horizon):
        for j in range(i + 1):
            matrix[3*i:3*i+3, 3*j:3*j+3] = blocks[i-j]
    offsets = []
    previous = initial.copy()
    for _ in range(horizon):
        previous = previous @ a_state
        offsets.append(previous)
    offset = np.asarray(offsets).reshape(-1)
    if not np.isfinite(matrix).all() or not np.isfinite(offset).all():
        raise ValueError("NONFINITE_TEMPORAL_MAP")
    return TemporalResponse(_owned(matrix), _owned(offset), _owned(initial))


@dataclass(frozen=True)
class TemporalPrefixConstraints:
    prefix_length: int
    predicted_raw: np.ndarray
    correction_matrix: np.ndarray
    projection_balls: tuple
    terminal_ball: object
    raw_physical: object
    terminal_constraint_mode: str

    def predict_correction(self, correction):
        delta = _finite(correction, (self.prefix_length, 3), "prefix correction")
        value = self.predicted_raw + (self.correction_matrix @ delta.reshape(-1)).reshape(-1, 3)
        if not np.isfinite(value).all():
            raise ValueError("NONFINITE_MEMORY_PREDICTION")
        return value

    def counterfactual(self, correction):
        delta = _finite(correction, (self.prefix_length, 3), "prefix correction").reshape(-1)
        balls = self.projection_balls + (self.terminal_ball,)
        norms = [float(np.linalg.norm(ball.matrix @ delta + ball.offset)) for ball in balls]
        candidate = projector.PhysicalContinuityComponents(
            norms[1], norms[0], norms[2], norms[3], self.terminal_constraint_mode
        )
        return projector.PhysicalCounterfactual(self.raw_physical, candidate)


def build_prefix_constraints(response, raw_actions, observed_displacements, *,
                             prefix_length=5,
                             terminal_constraint_mode=projector.TERMINAL_CONSTRAINT_EXECUTED_INTERNAL):
    observed = _finite(observed_displacements, (2, 3), "observed displacements")
    if not np.array_equal(response.initial_displacement, observed[-1]):
        raise ValueError("MEMORY_INITIAL_STATE_MISMATCH")
    predicted = response.predict(raw_actions)
    # Reuse the original input/mode validation and physical definitions.
    raw_vectors = projector._physical_vectors(
        predicted, observed, prefix_length=prefix_length,
        terminal_constraint_mode=terminal_constraint_mode,
    )
    jac = response.matrix[:3 * len(predicted), :3 * prefix_length].reshape(len(predicted), 3, -1)
    bv, bj, nj, tj = raw_vectors
    boundary_radius = float(np.linalg.norm(bj)) - projector.BOUNDARY_JERK_IMPROVEMENT
    if not np.isfinite(boundary_radius) or boundary_radius <= 0:
        raise ValueError("NONPOSITIVE_BOUNDARY_JERK_RADIUS")
    if terminal_constraint_mode == projector.TERMINAL_CONSTRAINT_EXECUTED_INTERNAL:
        terminal_name = "terminal_internal_jerk"
        terminal_jac = jac[prefix_length-1] - 2 * jac[prefix_length-2] + jac[prefix_length-3]
    elif prefix_length == 1:
        terminal_name = "prefix_exit_jerk"
        terminal_jac = jac[1] - 2 * jac[0]
    else:
        terminal_name = "prefix_exit_jerk"
        terminal_jac = jac[prefix_length] - 2 * jac[prefix_length-1] + jac[prefix_length-2]

    def ball(name, matrix, offset, radius):
        if not np.isfinite(matrix).all() or not np.isfinite(radius):
            raise ValueError("NONFINITE_MEMORY_CONSTRAINT")
        return projector._AffineBall(name, _owned(matrix), _owned(offset), radius)

    balls = (
        ball("boundary_jerk", jac[0], bj, boundary_radius),
        ball("boundary_velocity", jac[0], bv, float(np.linalg.norm(bv)) + projector.PHYSICAL_TOLERANCE),
        ball("next_jerk", jac[1] - 2 * jac[0], nj, float(np.linalg.norm(nj)) + projector.PHYSICAL_TOLERANCE),
    )
    terminal = ball(terminal_name, terminal_jac, tj, float(np.linalg.norm(tj)) + projector.PHYSICAL_TOLERANCE)
    raw_physical = projector.PhysicalContinuityComponents(
        *(float(np.linalg.norm(vector)) for vector in raw_vectors),
        terminal_constraint_mode=terminal_constraint_mode,
    )
    return TemporalPrefixConstraints(
        prefix_length, _owned(predicted), _owned(response.matrix[:3 * len(predicted), :3 * prefix_length]),
        balls, terminal, raw_physical, terminal_constraint_mode,
    )
