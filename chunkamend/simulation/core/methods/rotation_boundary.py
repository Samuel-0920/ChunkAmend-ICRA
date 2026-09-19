"""Minimal causal rotation-command boundary correction for RAP-COVER.

The treatment operates only on the five rotation commands that will be
executed before the next query.  It interpolates rotation *increments* on
SO(3), keeps translation, gripper, and the unexecuted suffix byte-identical,
and introduces no tunable magnitude cap.  The existing LIBERO OSC_POSE
normalized input cube is the governing bound; a candidate outside it is
rejected and the input chunk is returned exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import numpy as np


ROTATION_START = 3
ROTATION_STOP = 6
EXECUTED_PREFIX = 5
CONTROLLER_INPUT_LIMIT = 1.0
CONTROLLER_ROTATION_SCALE_RAD = 0.5
SO3_PI_MARGIN_RAD = 1e-7
SELECTED_UPSTREAM_REASON = "SELECTED"

ROT_REF = "ROT_REF"
ROT_E0 = "ROT_MINJERK_E0"
ROT_E25 = "ROT_MINJERK_E25"
ROT_E50 = "ROT_MINJERK_E50"
ROT_E75 = "ROT_MINJERK_E75"
ROTATION_ARMS = (ROT_REF, ROT_E0, ROT_E25, ROT_E50, ROT_E75)

ENDPOINT_FRACTIONS = {
    ROT_E0: 0.0,
    ROT_E25: 0.25,
    ROT_E50: 0.5,
    ROT_E75: 0.75,
}


def minimum_jerk_weights(endpoint_fraction: float) -> np.ndarray:
    """Return five samples of the unique C2 quintic endpoint transition."""
    if endpoint_fraction not in (0.0, 0.25, 0.5, 0.75):
        raise ValueError("rotation endpoint must be exactly 0, 0.25, 0.5, or 0.75")
    tau = np.linspace(0.0, 1.0, EXECUTED_PREFIX, dtype=np.float64)
    smoothstep = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    weights = endpoint_fraction + (1.0 - endpoint_fraction) * (1.0 - smoothstep)
    expected = {
        0.0: np.array([1.0, 0.896484375, 0.5, 0.103515625, 0.0]),
        0.25: np.array([1.0, 0.92236328125, 0.625, 0.32763671875, 0.25]),
        0.5: np.array([1.0, 0.9482421875, 0.75, 0.5517578125, 0.5]),
        0.75: np.array([1.0, 0.97412109375, 0.875, 0.77587890625, 0.75]),
    }[endpoint_fraction]
    if not np.array_equal(weights, expected):
        raise RuntimeError("minimum-jerk weight construction drifted")
    return weights


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _exp_so3(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("invalid SO(3) exponential input")
    theta = float(np.linalg.norm(vector))
    generator = _skew(vector)
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64) + generator + 0.5 * (generator @ generator)
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) / theta * generator
        + (1.0 - math.cos(theta)) / (theta * theta) * (generator @ generator)
    )


def _log_so3(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("invalid SO(3) logarithm input")
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cosine)
    if theta >= math.pi - SO3_PI_MARGIN_RAD:
        raise ValueError("SO(3) principal-log branch is not trustworthy")
    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if theta < 1e-10:
        return 0.5 * vee
    return theta / (2.0 * math.sin(theta)) * vee


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass(frozen=True)
class RotationBoundaryUpdate:
    key: tuple[str, str]
    logical_timestep: int
    final_executed_rotation: np.ndarray
    expected_previous_logical_timestep: int | None = None


@dataclass(frozen=True)
class RotationBoundaryPreview:
    actions: np.ndarray
    update: RotationBoundaryUpdate
    receipt: dict


@dataclass(frozen=True)
class _RotationEpisodeState:
    logical_timestep: int
    final_executed_rotation: np.ndarray


class RotationBoundaryCorrector:
    """Transactional per-run causal state for the three frozen rotation arms."""

    def __init__(self) -> None:
        self._episodes: dict[tuple[str, str], _RotationEpisodeState] = {}

    @staticmethod
    def _context(value: Mapping) -> tuple[str, str, bool, int, int, int]:
        required = {
            "run_id",
            "episode_id",
            "episode_start",
            "logical_timestep",
            "stride",
            "previous_executed_count",
        }
        if not isinstance(value, Mapping) or not required <= set(value):
            raise ValueError("rotation correction requires complete causal context")
        run_id = value["run_id"]
        episode_id = value["episode_id"]
        episode_start = value["episode_start"]
        logical_timestep = value["logical_timestep"]
        stride = value["stride"]
        previous_count = value["previous_executed_count"]
        if (
            not isinstance(run_id, str)
            or not run_id
            or not isinstance(episode_id, str)
            or not episode_id
            or not isinstance(episode_start, bool)
            or isinstance(logical_timestep, bool)
            or not isinstance(logical_timestep, int)
            or logical_timestep < 0
            or stride != EXECUTED_PREFIX
            or isinstance(previous_count, bool)
            or not isinstance(previous_count, int)
        ):
            raise ValueError("invalid rotation correction causal context")
        return run_id, episode_id, episode_start, logical_timestep, stride, previous_count

    @staticmethod
    def _validate_actions(actions: np.ndarray) -> np.ndarray:
        value = np.asarray(actions)
        if value.shape != (10, 7) or not np.issubdtype(value.dtype, np.floating):
            raise ValueError("rotation correction requires a floating (10,7) chunk")
        if not np.isfinite(value).all():
            raise ValueError("rotation correction input must be finite")
        return value

    @staticmethod
    def _interpolate(raw_rotation: np.ndarray, previous_rotation: np.ndarray, weight: float) -> np.ndarray:
        if weight == 0.0:
            return raw_rotation.copy()
        if weight == 1.0:
            return previous_rotation.copy()
        raw_increment = _exp_so3(CONTROLLER_ROTATION_SCALE_RAD * raw_rotation)
        previous_increment = _exp_so3(CONTROLLER_ROTATION_SCALE_RAD * previous_rotation)
        relative = previous_increment @ raw_increment.T
        correction = _log_so3(relative)
        corrected_increment = _exp_so3(weight * correction) @ raw_increment
        return _log_so3(corrected_increment) / CONTROLLER_ROTATION_SCALE_RAD

    def preview(
        self,
        context: Mapping,
        actions: np.ndarray,
        *,
        rotation_arm: str,
        upstream_reason: str,
    ) -> RotationBoundaryPreview:
        if rotation_arm not in ROTATION_ARMS:
            raise ValueError("unknown frozen rotation arm")
        source = self._validate_actions(actions)
        run_id, episode_id, episode_start, logical_timestep, stride, previous_count = self._context(context)
        key = (run_id, episode_id)
        previous = self._episodes.get(key)
        result = source.copy()
        reason = "ROT_REF"
        applied = False
        fallback = False
        weights: np.ndarray | None = None

        if episode_start:
            if previous is not None or logical_timestep != 0 or previous_count != 0:
                raise ValueError("invalid rotation episode start")
            reason = "EPISODE_START_NO_HISTORY"
        elif previous is None:
            raise ValueError("rotation episode continuation without state")
        elif previous_count != stride or logical_timestep != previous.logical_timestep + stride:
            raise ValueError("stale or skipped rotation causal context")
        elif rotation_arm == ROT_REF:
            reason = "ROT_REF"
        elif upstream_reason != SELECTED_UPSTREAM_REASON:
            reason = "UPSTREAM_EXACT_RAW_FALLBACK"
        else:
            endpoint = ENDPOINT_FRACTIONS[rotation_arm]
            weights = minimum_jerk_weights(endpoint)
            raw_prefix = source[:EXECUTED_PREFIX, ROTATION_START:ROTATION_STOP]
            previous_rotation = previous.final_executed_rotation
            if (
                np.max(np.abs(raw_prefix)) > CONTROLLER_INPUT_LIMIT
                or np.max(np.abs(previous_rotation)) > CONTROLLER_INPUT_LIMIT
            ):
                reason = "CONTROLLER_INPUT_BOUND_FALLBACK"
                fallback = True
            else:
                try:
                    corrected = np.stack(
                        [
                            self._interpolate(raw_prefix[index], previous_rotation, float(weight))
                            for index, weight in enumerate(weights)
                        ],
                        axis=0,
                    ).astype(source.dtype, copy=False)
                except (FloatingPointError, ValueError):
                    reason = "SO3_NUMERICAL_FALLBACK"
                    fallback = True
                else:
                    if (
                        not np.isfinite(corrected).all()
                        or np.max(np.abs(corrected)) > CONTROLLER_INPUT_LIMIT
                    ):
                        reason = "FINAL_CONTROLLER_BOUND_FALLBACK"
                        fallback = True
                    else:
                        result[:EXECUTED_PREFIX, ROTATION_START:ROTATION_STOP] = corrected
                        applied = True
                        reason = "APPLIED"

        if not np.array_equal(result[:, :ROTATION_START], source[:, :ROTATION_START]):
            raise RuntimeError("rotation treatment changed translation bytes")
        if not np.array_equal(result[:, ROTATION_STOP:], source[:, ROTATION_STOP:]):
            raise RuntimeError("rotation treatment changed gripper bytes")
        if not np.array_equal(
            result[EXECUTED_PREFIX:, ROTATION_START:ROTATION_STOP],
            source[EXECUTED_PREFIX:, ROTATION_START:ROTATION_STOP],
        ):
            raise RuntimeError("rotation treatment changed the unexecuted suffix")
        if (fallback or not applied) and not np.array_equal(result, source):
            raise RuntimeError("rotation fallback/no-op is not byte-exact")

        final_rotation = result[EXECUTED_PREFIX - 1, ROTATION_START:ROTATION_STOP].copy()
        update = RotationBoundaryUpdate(
            key, logical_timestep, final_rotation,
            None if previous is None else previous.logical_timestep,
        )
        before = source[:EXECUTED_PREFIX, ROTATION_START:ROTATION_STOP].astype(np.float64)
        after = result[:EXECUTED_PREFIX, ROTATION_START:ROTATION_STOP].astype(np.float64)
        receipt = {
            "schema_version": "rap-cover-rotation-boundary-v2-cas",
            "commit_semantics": "COMPARE_PREVIOUS_TIMESTEP_AND_ADVANCE",
            "rotation_arm": rotation_arm,
            "reason": reason,
            "applied": applied,
            "exact_fallback": fallback,
            "endpoint_fraction": None if weights is None else float(weights[-1]),
            "weights": None if weights is None else weights.tolist(),
            "input_action_sha256": _sha256(source),
            "output_action_sha256": _sha256(result),
            "rotation_prefix_changed": not np.array_equal(before, after),
            "translation_bytes_equal": True,
            "gripper_bytes_equal": True,
            "suffix_rotation_bytes_equal": True,
            "first_rotation_equals_previous": bool(
                applied
                and np.array_equal(
                    result[0, ROTATION_START:ROTATION_STOP], previous.final_executed_rotation
                )
            ),
            "correction_l2": float(np.linalg.norm(after - before)),
            "correction_max_abs": float(np.max(np.abs(after - before))),
            "controller_input_limit": CONTROLLER_INPUT_LIMIT,
            "new_tunable_rotation_cap": None,
        }
        return RotationBoundaryPreview(result, update, receipt)

    def commit(self, update: RotationBoundaryUpdate) -> None:
        if not isinstance(update, RotationBoundaryUpdate):
            raise ValueError("invalid rotation update")
        if (not isinstance(update.key, tuple) or len(update.key) != 2
                or not all(isinstance(x, str) and x for x in update.key)
                or type(update.logical_timestep) is not int or update.logical_timestep < 0):
            raise ValueError("invalid rotation update identity")
        previous = self._episodes.get(update.key)
        expected = update.expected_previous_logical_timestep
        if expected is None:
            if previous is not None or update.logical_timestep != 0:
                raise ValueError("stale rotation episode-start commit")
        elif (type(expected) is not int or previous is None
              or previous.logical_timestep != expected
              or update.logical_timestep != expected + EXECUTED_PREFIX):
            raise ValueError("stale, duplicate or skipped rotation commit")
        try:
            rotation = np.asarray(update.final_executed_rotation)
            valid = (rotation.shape == (3,) and np.issubdtype(rotation.dtype, np.floating)
                     and np.isfinite(rotation).all())
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("invalid rotation update values")
        self._episodes[update.key] = _RotationEpisodeState(
            update.logical_timestep,
            rotation.copy(),
        )
