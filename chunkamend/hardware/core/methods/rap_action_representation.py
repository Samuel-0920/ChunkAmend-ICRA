"""Explicit action-representation handoff for future RAP response work.

This module is deliberately not wired into the active RAP policy path.  It
tracks three representations without claiming that a controller-bound command
is the robot's realized physical state:

1. the selected normalized physical action emitted by RAP;
2. the post-output-transform physical action emitted by Policy;
3. the post-rotation, controller-bound command submitted by an integration.

The current 3x3 translation response remains unchanged.  A future response
candidate may consume ``completed_controller_bound_prefix`` only after its own
scientific identity and normalization contract have been frozen.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib

import numpy as np

from chunkamend.hardware.core.methods.m1_integration import COVER_SCREEN_METHOD_IDS
from chunkamend.hardware.core.methods.m1_integration import M1Context
from chunkamend.hardware.core.methods.m1_integration import MethodId


SCHEMA_VERSION = "rap-cover-action-representation-v1"
ACTION_HORIZON = 15
PHYSICAL_ACTION_DIM = 7
TRANSLATION_STOP = 3
ROTATION_STOP = 6
CONTROLLER_BOUND_SEMANTICS = "COMMAND_SUBMITTED_TO_CONTROLLER__NOT_REALIZED_ROBOT_STATE"

_RAP_METHOD_IDS = {
    MethodId.RAP,
    MethodId.RAP_NO_RISK_TRUST_ELIGIBILITY,
    MethodId.RAP_RISK_GATED_ABLATION,
    *COVER_SCREEN_METHOD_IDS,
}


def _immutable_action_copy(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (ACTION_HORIZON, PHYSICAL_ACTION_DIM)
        or not np.issubdtype(array.dtype, np.floating)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{label} must be a finite floating (15,7) action chunk")
    result = np.ascontiguousarray(array).copy()
    result.flags.writeable = False
    return result


def _immutable_prefix_copy(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value).copy()
    result.flags.writeable = False
    return result


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


@dataclass(frozen=True)
class RapActionRepresentationPreview:
    """Uncommitted representation handoff for one logical RAP query."""

    key: tuple[str, str, MethodId]
    logical_timestep: int
    stride: int
    expected_previous_logical_timestep: int | None
    selected_normalized_physical_actions: np.ndarray
    post_output_physical_actions: np.ndarray
    completed_controller_bound_prefix: np.ndarray


@dataclass(frozen=True)
class _CommittedActionRepresentations:
    logical_timestep: int
    stride: int
    selected_normalized_physical_actions: np.ndarray
    post_output_physical_actions: np.ndarray
    post_rotation_controller_bound_actions: np.ndarray


class RapActionRepresentationTracker:
    """Transactional, per-episode representation contract.

    ``preview`` is read-only. ``commit`` validates the post-rotation handoff and
    is the only operation that advances state.  Invalid or duplicate commits do
    not mutate the last valid record.
    """

    def __init__(self) -> None:
        self._committed: dict[tuple[str, str, MethodId], _CommittedActionRepresentations] = {}

    @staticmethod
    def _context(value: M1Context | Mapping) -> M1Context:
        context = value if isinstance(value, M1Context) else M1Context.from_mapping(dict(value))
        if context.method_id not in _RAP_METHOD_IDS:
            raise ValueError("action representation contract requires a RAP method identity")
        return context

    def preview(
        self,
        context: M1Context | Mapping,
        *,
        selected_normalized_physical_actions: np.ndarray,
        post_output_physical_actions: np.ndarray,
    ) -> RapActionRepresentationPreview:
        parsed = self._context(context)
        selected = _immutable_action_copy(
            selected_normalized_physical_actions,
            label="selected_normalized_physical_actions",
        )
        post_output = _immutable_action_copy(
            post_output_physical_actions,
            label="post_output_physical_actions",
        )
        key = (parsed.run_id, parsed.episode_id, parsed.method_id)
        previous = self._committed.get(key)

        if parsed.episode_start:
            if previous is not None:
                raise ValueError("duplicate action representation episode start")
            expected_previous_logical_timestep = None
            completed = np.empty((0, PHYSICAL_ACTION_DIM), dtype=post_output.dtype)
        else:
            if previous is None:
                raise ValueError("action representation continuation without committed state")
            if (
                parsed.stride != previous.stride
                or parsed.previous_executed_count != previous.stride
                or parsed.logical_timestep != previous.logical_timestep + previous.stride
            ):
                raise ValueError("stale or skipped action representation context")
            expected_previous_logical_timestep = previous.logical_timestep
            completed = previous.post_rotation_controller_bound_actions[: parsed.previous_executed_count]

        return RapActionRepresentationPreview(
            key=key,
            logical_timestep=parsed.logical_timestep,
            stride=parsed.stride,
            expected_previous_logical_timestep=expected_previous_logical_timestep,
            selected_normalized_physical_actions=selected,
            post_output_physical_actions=post_output,
            completed_controller_bound_prefix=_immutable_prefix_copy(completed),
        )

    def preview_policy_output(
        self,
        context: M1Context | Mapping,
        output: Mapping,
    ) -> RapActionRepresentationPreview:
        """Bind the existing Policy output and RAP receipt to this contract."""
        parsed = self._context(context)
        if not isinstance(output, Mapping):
            raise ValueError("policy output must be a mapping")
        runtime = output.get("m1_runtime")
        if not isinstance(runtime, Mapping):
            raise ValueError("policy output is missing the RAP runtime receipt")
        expected_identity = {
            "method_id": parsed.method_id.value,
            "run_id": parsed.run_id,
            "episode_id": parsed.episode_id,
            "logical_timestep": parsed.logical_timestep,
            "stride": parsed.stride,
        }
        if any(runtime.get(name) != value for name, value in expected_identity.items()):
            raise ValueError("policy output and action representation context disagree")
        if "selected_normalized_physical_actions" not in runtime or "actions" not in output:
            raise ValueError("policy output lacks an action representation")
        return self.preview(
            parsed,
            selected_normalized_physical_actions=runtime["selected_normalized_physical_actions"],
            post_output_physical_actions=output["actions"],
        )

    def commit(
        self,
        preview: RapActionRepresentationPreview,
        post_rotation_controller_bound_actions: np.ndarray,
    ) -> dict:
        if not isinstance(preview, RapActionRepresentationPreview):
            raise ValueError("invalid action representation preview")
        controller_bound = _immutable_action_copy(
            post_rotation_controller_bound_actions,
            label="post_rotation_controller_bound_actions",
        )
        previous = self._committed.get(preview.key)
        if preview.expected_previous_logical_timestep is None:
            if previous is not None:
                raise ValueError("stale or duplicate action representation episode-start commit")
        elif previous is None or previous.logical_timestep != preview.expected_previous_logical_timestep:
            raise ValueError("stale or duplicate action representation commit")

        post_output = preview.post_output_physical_actions
        translation_equal = _bytes_equal(
            post_output[:, :TRANSLATION_STOP],
            controller_bound[:, :TRANSLATION_STOP],
        )
        gripper_equal = _bytes_equal(
            post_output[:, ROTATION_STOP:],
            controller_bound[:, ROTATION_STOP:],
        )
        suffix_rotation_equal = _bytes_equal(
            post_output[preview.stride :, TRANSLATION_STOP:ROTATION_STOP],
            controller_bound[preview.stride :, TRANSLATION_STOP:ROTATION_STOP],
        )
        if not translation_equal:
            raise ValueError("post-rotation handoff changed translation bytes")
        if not gripper_equal:
            raise ValueError("post-rotation handoff changed gripper bytes")
        if not suffix_rotation_equal:
            raise ValueError("post-rotation handoff changed unexecuted rotation bytes")

        selected = _immutable_action_copy(
            preview.selected_normalized_physical_actions,
            label="selected_normalized_physical_actions",
        )
        post_output_copy = _immutable_action_copy(
            post_output,
            label="post_output_physical_actions",
        )
        controller_bound_copy = _immutable_action_copy(
            controller_bound,
            label="post_rotation_controller_bound_actions",
        )
        self._committed[preview.key] = _CommittedActionRepresentations(
            logical_timestep=preview.logical_timestep,
            stride=preview.stride,
            selected_normalized_physical_actions=selected,
            post_output_physical_actions=post_output_copy,
            post_rotation_controller_bound_actions=controller_bound_copy,
        )

        rotation_prefix_changed = not _bytes_equal(
            post_output[: preview.stride, TRANSLATION_STOP:ROTATION_STOP],
            controller_bound[: preview.stride, TRANSLATION_STOP:ROTATION_STOP],
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": preview.key[0],
            "episode_id": preview.key[1],
            "method_id": preview.key[2].value,
            "logical_timestep": preview.logical_timestep,
            "stride": preview.stride,
            "controller_bound_semantics": CONTROLLER_BOUND_SEMANTICS,
            "selected_normalized_physical_actions_sha256": _sha256(selected),
            "post_output_physical_actions_sha256": _sha256(post_output_copy),
            "post_rotation_controller_bound_actions_sha256": _sha256(controller_bound_copy),
            "completed_controller_bound_prefix_sha256": _sha256(
                preview.completed_controller_bound_prefix
            ),
            "translation_bytes_equal": translation_equal,
            "gripper_bytes_equal": gripper_equal,
            "suffix_rotation_bytes_equal": suffix_rotation_equal,
            "rotation_prefix_changed": rotation_prefix_changed,
            "feeds_active_response": False,
        }
