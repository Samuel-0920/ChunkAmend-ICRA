"""Common active Q3/Q6 response integration for paired RAP-COVER experiments.

The response mode changes exactly one scientific variable: the unchanged core
uses either the production Q3 translation fit or a Q6 fit augmented with
controller-bound rotation.  Candidate perturbations remain translation-only.
For Q6, the core receives the translation coefficient rows plus the complete
raw/selected rotation response offsets, so its absolute TCP predictions equal
the full 6x3 fit.  Selector, projector, physical veto, candidate space,
tie-break and raw fallback are not reimplemented here.

Both estimator arms use this one transactional state machine and the same
post-output, post-rotation representation handoff.  It is exposed through
``M1RapState.for_active_response``; a dispatcher may provide controller-bound
counterfactuals but must not reimplement response fitting or COVER decisions.
Missing, stale or mismatched handoffs fail closed before they can enter causal
history.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from chunkamend.hardware.core.methods import rap_cover as _rap_cover
from chunkamend.hardware.core.methods.m1_integration import M1Context
from chunkamend.hardware.core.methods.m1_integration import RapRuntimeConfig
from chunkamend.hardware.core.methods.m1_types import RESPONSE_PREDICTION_Q6_ALTERNATIVE_OFFSETS
from chunkamend.hardware.core.methods.m1_types import RapDecision
from chunkamend.hardware.core.methods.m1_types import ResponseFit
from chunkamend.hardware.core.methods.rap_action_representation import CONTROLLER_BOUND_SEMANTICS
from chunkamend.hardware.core.methods.rap_action_representation import SCHEMA_VERSION as REPRESENTATION_SCHEMA_VERSION
from chunkamend.hardware.core.methods.rap_response import fit_completed_translation_response
from chunkamend.hardware.core.methods.rap_response_6d import Q6D_ACTION_REPRESENTATION
from chunkamend.hardware.core.methods.rap_response_6d import SixDResponseFit
from chunkamend.hardware.core.methods.rap_response_6d import fit_completed_six_d_response
from chunkamend.hardware.core.methods.response_experiment import ResponseExperiment, fit_response_experiment

ACTIVE_Q6_RESPONSE_IDENTITY = (
    "Q6_TRANSLATION_PLUS_ACTIVE_QUANTILE_CONTROLLER_ROTATION__"
    "FULL_PREDICTION_WITH_TRANSLATION_ONLY_INTERVENTION"
)
ACTIVE_Q3_RESPONSE_IDENTITY = "Q3_NORMALIZED_TRANSLATION__COMMON_ACTIVE_RESPONSE_INTEGRATION"
ACTIVE_RESPONSE_Q3 = "Q3_TRANSLATION"
ACTIVE_RESPONSE_Q6 = "Q6_TRANSLATION_PLUS_CONTROLLER_ROTATION"
ACTIVE_RESPONSE_MODES = {ACTIVE_RESPONSE_Q3, ACTIVE_RESPONSE_Q6}
ACTION_HORIZON = 15
PHYSICAL_ACTION_DIM = 7
RESPONSE_ACTION_DIM = 6
TCP_DIM = 3
NORMALIZATION_EPSILON = 1e-6


def _finite_matrix(value, *, shape: tuple[int, int], label: str) -> np.ndarray:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if (
        result.shape != shape
        or not np.issubdtype(result.dtype, np.floating)
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"{label} must be a finite floating {shape} matrix")
    return np.ascontiguousarray(result).copy()


def _finite_vector(value, *, size: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite ({size},) vector")
    return np.ascontiguousarray(result)


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@dataclass(frozen=True)
class _ActiveEpisodeState:
    previous_raw_chunk: np.ndarray
    previous_selected_normalized_chunk: np.ndarray
    previous_controller_bound_physical_chunk: np.ndarray
    completed_selected_normalized_history: np.ndarray
    completed_q6_history: np.ndarray
    completed_tcp_history: np.ndarray
    previous_query_tcp: np.ndarray
    last_logical_timestep: int


@dataclass(frozen=True)
class ActiveResponseUpdate:
    key: tuple[str, str, object]
    expected_previous_logical_timestep: int | None
    state_before_current_commit: _ActiveEpisodeState


@dataclass(frozen=True)
class ActiveResponsePreview:
    actions: np.ndarray
    raw_chunk: np.ndarray
    selected_chunk: np.ndarray
    update: ActiveResponseUpdate | None
    causal_status: str
    reason: str
    decision: RapDecision | None
    six_d_response: SixDResponseFit | None
    response_adapter: ResponseFit
    raw_controller_rotation: np.ndarray
    selected_controller_rotation: np.ndarray
    response_mode: str
    response_identity: str


class ActiveResponseState:
    """One transactional integration path whose only arm variable is Q3/Q6."""

    def __init__(
        self,
        *,
        action_q01: np.ndarray,
        action_q99: np.ndarray,
        normalization_identity: str,
        response_mode: str,
        response_experiment: ResponseExperiment | None = None,
    ) -> None:
        q01 = _finite_vector(action_q01, size=PHYSICAL_ACTION_DIM, label="action_q01")
        q99 = _finite_vector(action_q99, size=PHYSICAL_ACTION_DIM, label="action_q99")
        if np.any(q99 <= q01):
            raise ValueError("active action quantile span must be positive")
        if not isinstance(normalization_identity, str) or not normalization_identity:
            raise ValueError("normalization identity must be explicit")
        if response_mode not in ACTIVE_RESPONSE_MODES:
            raise ValueError("active response mode must be Q3 or Q6")
        self._q01 = q01.copy()
        self._q99 = q99.copy()
        self._normalization_identity = normalization_identity
        self._response_mode = response_mode
        if response_experiment is not None and (
            not isinstance(response_experiment, ResponseExperiment) or response_mode != ACTIVE_RESPONSE_Q3
        ):
            raise ValueError("window/ridge development overrides are explicitly Q3-only")
        self._response_experiment = response_experiment
        self._episodes: dict[tuple[str, str, object], _ActiveEpisodeState] = {}

    def _normalize_controller_rotation(self, controller_bound: np.ndarray) -> np.ndarray:
        rotation = controller_bound[:, 3:6].astype(np.float64, copy=False)
        result = (
            2.0 * (rotation - self._q01[3:6])
            / (self._q99[3:6] - self._q01[3:6] + NORMALIZATION_EPSILON)
            - 1.0
        )
        if not np.isfinite(result).all():
            raise ValueError("nonfinite active-quantile controller rotation")
        return np.ascontiguousarray(result)

    @staticmethod
    def adapt_response(
        value: SixDResponseFit,
        *,
        raw_rotation: np.ndarray,
        selected_rotation: np.ndarray,
    ) -> ResponseFit:
        full_matrix = value.matrix
        matrix = None if full_matrix is None else full_matrix[:3]
        raw_offset = None if full_matrix is None else raw_rotation @ full_matrix[3:]
        selected_offset = None if full_matrix is None else selected_rotation @ full_matrix[3:]
        return ResponseFit(
            matrix=matrix,
            available=value.available and matrix is not None,
            reason=value.reason if matrix is not None or not value.available else "INVALID_Q6_TRANSLATION_ADAPTER",
            sample_count=value.sample_count,
            window_start=value.window_start,
            window_end=value.window_end,
            condition_number=value.condition_number,
            residual_rmse=value.residual_rmse,
            rank=value.rank,
            fit_age=value.fit_age,
            prediction_semantics=RESPONSE_PREDICTION_Q6_ALTERNATIVE_OFFSETS,
            raw_prediction_offset=raw_offset,
            selected_prediction_offset=selected_offset,
        )

    @staticmethod
    def _copy_state(value: _ActiveEpisodeState) -> _ActiveEpisodeState:
        return _ActiveEpisodeState(
            value.previous_raw_chunk.copy(),
            value.previous_selected_normalized_chunk.copy(),
            value.previous_controller_bound_physical_chunk.copy(),
            value.completed_selected_normalized_history.copy(),
            value.completed_q6_history.copy(),
            value.completed_tcp_history.copy(),
            value.previous_query_tcp.copy(),
            value.last_logical_timestep,
        )

    def _structural_fallback(
        self,
        raw_full: np.ndarray,
        *,
        status: str,
        update: ActiveResponseUpdate | None = None,
        raw_controller_rotation: np.ndarray | None = None,
        selected_controller_rotation: np.ndarray | None = None,
    ) -> ActiveResponsePreview:
        raw = np.ascontiguousarray(raw_full[:, :PHYSICAL_ACTION_DIM]).copy()
        empty_rotation = np.empty((0, TCP_DIM), dtype=np.float64)
        return ActiveResponsePreview(
            actions=np.ascontiguousarray(raw_full).copy(),
            raw_chunk=raw,
            selected_chunk=raw.copy(),
            update=update,
            causal_status=status,
            reason=status,
            decision=None,
            six_d_response=None,
            response_adapter=ResponseFit(matrix=None, available=False, reason=status),
            raw_controller_rotation=(
                empty_rotation.copy()
                if raw_controller_rotation is None
                else np.ascontiguousarray(raw_controller_rotation).copy()
            ),
            selected_controller_rotation=(
                empty_rotation.copy()
                if selected_controller_rotation is None
                else np.ascontiguousarray(selected_controller_rotation).copy()
            ),
            response_mode=self._response_mode,
            response_identity=(
                self._response_experiment.identity
                if self._response_experiment is not None
                else ACTIVE_Q3_RESPONSE_IDENTITY
                if self._response_mode == ACTIVE_RESPONSE_Q3
                else ACTIVE_Q6_RESPONSE_IDENTITY
            ),
        )

    def preview(
        self,
        context: M1Context,
        raw_actions: np.ndarray,
        config: RapRuntimeConfig,
        *,
        raw_controller_bound_physical_actions: np.ndarray,
        selected_controller_bound_physical_actions: np.ndarray,
    ) -> ActiveResponsePreview:
        """Compute one Q3 or Q6 decision from completed evidence only."""
        if self._response_experiment is not None and config.method_identity != self._response_experiment.identity:
            raise ValueError("response experiment and runtime identities disagree")
        if not isinstance(context, M1Context):
            raise ValueError("active response preview requires a validated M1Context")
        if context.method_id is not config.method_id:
            raise ValueError("active response context/config method identity mismatch")
        if config.response_rotation_ridge != 1e-4 and (self._response_mode != ACTIVE_RESPONSE_Q6 or self._response_experiment is not None):
            raise ValueError("group ridge requires the active Q6 response")
        if config.response_mode != "PRIMARY" or config.risk is not None:
            raise ValueError("active response comparison requires the primary risk-ablated runtime")
        raw_full = np.asarray(raw_actions)
        if (
            raw_full.ndim != 2
            or raw_full.shape[0] != ACTION_HORIZON
            or raw_full.shape[1] < PHYSICAL_ACTION_DIM
            or not np.issubdtype(raw_full.dtype, np.floating)
        ):
            raise ValueError("raw_actions must be floating [15,D>=7]")
        if raw_full.shape not in {
            (ACTION_HORIZON, PHYSICAL_ACTION_DIM),
            (ACTION_HORIZON, 32),
        }:
            return self._structural_fallback(raw_full, status="SHAPE_MISMATCH")
        if not np.isfinite(raw_full).all():
            return self._structural_fallback(raw_full, status="NONFINITE_INPUT")
        raw = np.ascontiguousarray(raw_full[:, :PHYSICAL_ACTION_DIM]).copy()
        raw_controller = _finite_matrix(
            raw_controller_bound_physical_actions,
            shape=(ACTION_HORIZON, PHYSICAL_ACTION_DIM),
            label="raw_controller_bound_physical_actions",
        )
        selected_controller = _finite_matrix(
            selected_controller_bound_physical_actions,
            shape=(ACTION_HORIZON, PHYSICAL_ACTION_DIM),
            label="selected_controller_bound_physical_actions",
        )
        raw_rotation = self._normalize_controller_rotation(raw_controller)
        selected_rotation = self._normalize_controller_rotation(selected_controller)
        key = (context.run_id, context.episode_id, context.method_id)
        previous = self._episodes.get(key)

        if context.episode_start:
            if previous is not None:
                raise ValueError("duplicate active response episode start")
            state = _ActiveEpisodeState(
                previous_raw_chunk=raw.copy(),
                previous_selected_normalized_chunk=np.empty((0, PHYSICAL_ACTION_DIM), dtype=raw.dtype),
                previous_controller_bound_physical_chunk=np.empty(
                    (0, PHYSICAL_ACTION_DIM), dtype=np.float64
                ),
                completed_selected_normalized_history=np.empty(
                    (0, PHYSICAL_ACTION_DIM), dtype=raw.dtype
                ),
                completed_q6_history=np.empty((0, RESPONSE_ACTION_DIM), dtype=np.float64),
                completed_tcp_history=context.current_tcp_translation[None, :].astype(
                    np.float64, copy=True
                ),
                previous_query_tcp=context.current_tcp_translation.astype(np.float64, copy=True),
                last_logical_timestep=context.logical_timestep,
            )
            expected_previous = None
        else:
            if previous is None:
                raise ValueError("active response continuation without committed state")
            if (
                context.previous_executed_count != context.stride
                or previous.previous_selected_normalized_chunk.shape
                != (ACTION_HORIZON, PHYSICAL_ACTION_DIM)
                or previous.previous_controller_bound_physical_chunk.shape
                != (ACTION_HORIZON, PHYSICAL_ACTION_DIM)
            ):
                return self._structural_fallback(
                    raw_full,
                    status="SHAPE_MISMATCH",
                    raw_controller_rotation=raw_controller[:, 3:6],
                    selected_controller_rotation=selected_controller[:, 3:6],
                )
            completed_tcp = context.completed_tcp_positions.astype(np.float64, copy=True)
            if completed_tcp.shape != (context.stride, TCP_DIM):
                return self._structural_fallback(
                    raw_full,
                    status="SHAPE_MISMATCH",
                    raw_controller_rotation=raw_controller[:, 3:6],
                    selected_controller_rotation=selected_controller[:, 3:6],
                )
            if not np.array_equal(context.current_tcp_translation, completed_tcp[-1]):
                return self._structural_fallback(
                    raw_full,
                    status="TIMESTAMP_MISMATCH",
                    raw_controller_rotation=raw_controller[:, 3:6],
                    selected_controller_rotation=selected_controller[:, 3:6],
                )
            if not np.isfinite(completed_tcp).all():
                return self._structural_fallback(
                    raw_full,
                    status="NONFINITE_INPUT",
                    raw_controller_rotation=raw_controller[:, 3:6],
                    selected_controller_rotation=selected_controller[:, 3:6],
                )
            if context.logical_timestep != previous.last_logical_timestep + context.stride:
                anchor = _ActiveEpisodeState(
                    previous_raw_chunk=raw.copy(),
                    previous_selected_normalized_chunk=np.empty(
                        (0, PHYSICAL_ACTION_DIM), dtype=raw.dtype
                    ),
                    previous_controller_bound_physical_chunk=np.empty(
                        (0, PHYSICAL_ACTION_DIM), dtype=np.float64
                    ),
                    completed_selected_normalized_history=np.empty(
                        (0, PHYSICAL_ACTION_DIM), dtype=raw.dtype
                    ),
                    completed_q6_history=np.empty(
                        (0, RESPONSE_ACTION_DIM), dtype=np.float64
                    ),
                    completed_tcp_history=context.current_tcp_translation[None, :].astype(
                        np.float64, copy=True
                    ),
                    previous_query_tcp=context.current_tcp_translation.astype(
                        np.float64, copy=True
                    ),
                    last_logical_timestep=context.logical_timestep,
                )
                return self._structural_fallback(
                    raw_full,
                    status="TIMESTAMP_MISMATCH",
                    update=ActiveResponseUpdate(
                        key,
                        previous.last_logical_timestep,
                        self._copy_state(anchor),
                    ),
                    raw_controller_rotation=raw_controller[:, 3:6],
                    selected_controller_rotation=selected_controller[:, 3:6],
                )
            completed_selected = previous.previous_selected_normalized_chunk[: context.stride].copy()
            completed_controller = previous.previous_controller_bound_physical_chunk[
                : context.stride
            ].copy()
            q6_rows = np.concatenate(
                (
                    completed_selected[:, :3].astype(np.float64, copy=False),
                    self._normalize_controller_rotation(completed_controller),
                ),
                axis=1,
            )
            state = _ActiveEpisodeState(
                previous_raw_chunk=raw.copy(),
                previous_selected_normalized_chunk=np.empty((0, PHYSICAL_ACTION_DIM), dtype=raw.dtype),
                previous_controller_bound_physical_chunk=np.empty(
                    (0, PHYSICAL_ACTION_DIM), dtype=np.float64
                ),
                completed_selected_normalized_history=np.concatenate(
                    (previous.completed_selected_normalized_history, completed_selected), axis=0
                ),
                completed_q6_history=np.concatenate(
                    (previous.completed_q6_history, q6_rows), axis=0
                ),
                completed_tcp_history=np.concatenate(
                    (previous.completed_tcp_history, completed_tcp), axis=0
                ),
                previous_query_tcp=context.current_tcp_translation.astype(np.float64, copy=True),
                last_logical_timestep=context.logical_timestep,
            )
            expected_previous = previous.last_logical_timestep

        displacements = np.diff(state.completed_tcp_history, axis=0)
        if self._response_mode == ACTIVE_RESPONSE_Q3:
            six_d = None
            if self._response_experiment is None:
                response = fit_completed_translation_response(
                    state.completed_selected_normalized_history[:, :3],
                    displacements,
                    min_samples=config.response.min_samples,
                    ridge=config.response.ridge,
                    condition_limit=config.response.condition_limit,
                )
                response_identity = ACTIVE_Q3_RESPONSE_IDENTITY
            else:
                response = fit_response_experiment(
                    state.completed_selected_normalized_history[:, :3],
                    displacements, self._response_experiment,
                )
                response_identity = self._response_experiment.identity
        else:
            six_d = fit_completed_six_d_response(
                state.completed_q6_history,
                displacements,
                action_representation=Q6D_ACTION_REPRESENTATION,
                rotation_ridge=config.response_rotation_ridge,
                min_samples=config.response.min_samples,
                ridge=config.response.ridge,
                condition_limit=config.response.condition_limit,
            )
            response = self.adapt_response(
                six_d,
                raw_rotation=raw_rotation,
                selected_rotation=selected_rotation,
            )
            response_identity = ACTIVE_Q6_RESPONSE_IDENTITY if config.response_rotation_ridge == 1e-4 else "Q6_GROUP_RIDGE_T0001_R001"
        completed_actions = state.completed_selected_normalized_history
        decision = _rap_cover.RapCoverCore(
            config.core,
            method_identity=config.method_identity,
            objective_count_mode=config.objective_count_mode,
            selection_mode=config.selection_mode,
            candidate_ranking_mode=config.candidate_ranking_mode,
            reversal_space=config.reversal_space,
            uncertainty_scale=config.uncertainty_scale,
            minimum_prequential_predictions=config.minimum_prequential_predictions,
            candidate_evaluation_policy=config.candidate_evaluation_policy,
            preprojection_objective_bound=config.preprojection_objective_bound,
            projector_context_cache_mode=config.projector_context_cache_mode,
            terminal_constraint_mode=config.terminal_constraint_mode,
            terminal_projection_role=config.terminal_projection_role,
        ).apply(
            raw,
            stride=context.stride,
            candidate_specs=list(config.candidate_specs),
            completed_actions=completed_actions,
            response=response,
            observed_displacements=displacements[-2:] if len(displacements) >= 2 else None,
            previous_action=completed_actions[-1].copy() if len(completed_actions) else None,
            risk_eligible=True,
            ablate_risk_trust_eligibility=True,
            reversal_context={
                "executed_action_history": completed_actions,
                "completed_tcp_history": state.completed_tcp_history,
                "direction_window": config.reversal.direction_window,
                "reversal_cos_threshold": config.reversal.reversal_cos_threshold,
                "min_consistent_steps": config.reversal.min_consistent_steps,
                "velocity_epsilon": config.reversal.velocity_epsilon,
            },
        )
        selected_full = np.ascontiguousarray(raw_full).copy()
        selected_full[:, :PHYSICAL_ACTION_DIM] = decision.chunk
        update = ActiveResponseUpdate(key, expected_previous, self._copy_state(state))
        return ActiveResponsePreview(
            actions=selected_full,
            raw_chunk=raw,
            selected_chunk=decision.chunk.copy(),
            update=update,
            causal_status="AVAILABLE" if response.available else "RESPONSE_UNAVAILABLE",
            reason=decision.reason.value,
            decision=decision,
            six_d_response=six_d,
            response_adapter=response,
            raw_controller_rotation=raw_controller[:, 3:6].copy(),
            selected_controller_rotation=selected_controller[:, 3:6].copy(),
            response_mode=self._response_mode,
            response_identity=response_identity,
        )

    def commit(
        self,
        preview: ActiveResponsePreview,
        *,
        controller_bound_physical_actions: np.ndarray,
        representation_receipt: dict,
    ) -> None:
        """Commit only a byte-bound V1 representation-contract handoff."""
        if not isinstance(preview, ActiveResponsePreview):
            raise ValueError("invalid active response preview")
        if preview.update is None:
            return
        controller = _finite_matrix(
            controller_bound_physical_actions,
            shape=(ACTION_HORIZON, PHYSICAL_ACTION_DIM),
            label="controller_bound_physical_actions",
        )
        selected = _finite_matrix(
            preview.selected_chunk,
            shape=(ACTION_HORIZON, PHYSICAL_ACTION_DIM),
            label="selected_normalized_actions",
        )
        update = preview.update
        expected_rotation = (
            preview.selected_controller_rotation
            if preview.decision is not None and preview.decision.reason.value == "SELECTED"
            else preview.raw_controller_rotation
        )
        if not (
            controller[:, 3:6].dtype == expected_rotation.dtype
            and np.array_equal(controller[:, 3:6], expected_rotation)
        ):
            raise ValueError("active response controller rotation counterfactual mismatch")
        previous = self._episodes.get(update.key)
        if update.expected_previous_logical_timestep is None:
            if previous is not None:
                raise ValueError("stale active response episode-start commit")
        elif previous is None or previous.last_logical_timestep != update.expected_previous_logical_timestep:
            raise ValueError("stale or duplicate active response commit")
        if not isinstance(representation_receipt, dict) or (
            representation_receipt.get("schema_version") != REPRESENTATION_SCHEMA_VERSION
            or representation_receipt.get("run_id") != update.key[0]
            or representation_receipt.get("episode_id") != update.key[1]
            or representation_receipt.get("method_id") != update.key[2].value
            or representation_receipt.get("logical_timestep")
            != update.state_before_current_commit.last_logical_timestep
            or representation_receipt.get("controller_bound_semantics")
            != CONTROLLER_BOUND_SEMANTICS
            or representation_receipt.get("translation_bytes_equal") is not True
            or representation_receipt.get("gripper_bytes_equal") is not True
            or representation_receipt.get("suffix_rotation_bytes_equal") is not True
            or representation_receipt.get("selected_normalized_physical_actions_sha256")
            != _sha256(selected)
            or representation_receipt.get("post_rotation_controller_bound_actions_sha256")
            != _sha256(controller)
        ):
            raise ValueError("active response representation receipt mismatch")
        state = update.state_before_current_commit
        self._episodes[update.key] = _ActiveEpisodeState(
            state.previous_raw_chunk.copy(),
            selected.copy(),
            controller.copy(),
            state.completed_selected_normalized_history.copy(),
            state.completed_q6_history.copy(),
            state.completed_tcp_history.copy(),
            state.previous_query_tcp.copy(),
            state.last_logical_timestep,
        )

    def receipt(self, preview: ActiveResponsePreview) -> dict:
        response = preview.response_adapter
        decision = preview.decision
        return {
            **({"group_regularization": {"translation_ridge": 1e-4, "rotation_ridge": 1e-3, "intercept": False}} if preview.response_identity == "Q6_GROUP_RIDGE_T0001_R001" else {}),
            "schema_version": "rap-cover-common-active-response-v2",
            "response_mode": preview.response_mode,
            "response_identity": preview.response_identity,
            "response_experiment": (None if self._response_experiment is None
                                    else self._response_experiment.receipt()),
            "normalization_identity": self._normalization_identity,
            "action_representation": (
                "NORMALIZED_TRANSLATION_3"
                if preview.six_d_response is None
                else preview.six_d_response.action_representation
            ),
            "causal_status": preview.causal_status,
            "reason": preview.reason,
            "available": response.available,
            "fit_reason": response.reason,
            "sample_count": response.sample_count,
            "window_start": response.window_start,
            "window_end": response.window_end,
            "rank": response.rank,
            "condition_number": (
                response.condition_number if np.isfinite(response.condition_number) else None
            ),
            "residual_rmse": response.residual_rmse if np.isfinite(response.residual_rmse) else None,
            "translation_intervention_only": True,
            "full_q6_prediction_contract": preview.six_d_response is not None,
            "alternative_rotation_offsets": (
                None
                if preview.six_d_response is None
                else "RAW_FALLBACK_VS_SELECTED_ROTATION_CORRECTION"
            ),
            "selector_projector_veto_candidate_space_unchanged": True,
            "selected_k": 0 if decision is None else decision.k,
            "selected_alpha": 0.0 if decision is None else decision.alpha,
            "selected_reason": preview.reason,
        }


# Compatibility names keep the rejected and corrected Q6 commits directly
# comparable while the experiment uses the common M1RapState factory below.
Q6ActiveUpdate = ActiveResponseUpdate
Q6ActivePreview = ActiveResponsePreview


class Q6ActiveResponseState(ActiveResponseState):
    def __init__(
        self,
        *,
        action_q01: np.ndarray,
        action_q99: np.ndarray,
        normalization_identity: str,
    ) -> None:
        super().__init__(
            action_q01=action_q01,
            action_q99=action_q99,
            normalization_identity=normalization_identity,
            response_mode=ACTIVE_RESPONSE_Q6,
        )
