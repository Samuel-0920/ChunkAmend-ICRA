"""Transactional runtime integration for the frozen M4 comparison methods."""

import copy
from dataclasses import dataclass
from enum import Enum
import json
import numbers
from pathlib import Path

import numpy as np

from chunkamend.hardware.core.methods import rap_cover as _rap_cover
from chunkamend.hardware.core.methods.m1_types import RapCandidateSpec
from chunkamend.hardware.core.methods.m1_types import RapConfig
from chunkamend.hardware.core.methods.m1_types import RapDecision
from chunkamend.hardware.core.methods.m1_types import ResponseFit
from chunkamend.hardware.core.methods.m1_types import RiskFeature
from chunkamend.hardware.core.methods.rap_cover import causal_risk_features
from chunkamend.hardware.core.methods.rap_response import CONDITION_LIMIT
from chunkamend.hardware.core.methods.rap_response import EW_HALF_LIFE_COMPLETED_PAIRS
from chunkamend.hardware.core.methods.rap_response import MIN_SAMPLES
from chunkamend.hardware.core.methods.rap_response import RIDGE
from chunkamend.hardware.core.methods.rap_response import fit_completed_translation_response
from chunkamend.hardware.core.methods.rap_response import fit_completed_translation_response_exponentially_weighted
from chunkamend.hardware.core.methods.rap_response import fit_completed_translation_response_prequential
from chunkamend.hardware.core.methods.rap_risk import RapRiskConfig
from chunkamend.hardware.core.methods.rap_risk import RapRiskDecision
from chunkamend.hardware.core.methods.rap_risk import RapRiskOperatingPoint
from chunkamend.hardware.core.methods.rap_risk import evaluate_rap_risk
from chunkamend.hardware.core.methods.temporal_ensemble import ActTeDecision
from chunkamend.hardware.core.methods.temporal_ensemble import ActTemporalEnsemble
from chunkamend.hardware.core.methods.temporal_ensemble import MatchedCadenceTemporalEnsemble

CONTEXT_KEY = "__openpi_m1_context__"
SCHEMA_VERSION = "1.0"


class MethodId(str, Enum):
    VANILLA = "VANILLA"
    TE = "TE"
    TE_MATCHED_CADENCE_CONTROL = "TE_MATCHED_CADENCE_CONTROL"
    RTC_D0_MECHANISM_CONTROL = "RTC_D0"
    RTC_D0 = RTC_D0_MECHANISM_CONTROL
    RAP = "RAP"
    RAP_NO_RISK_TRUST_ELIGIBILITY = "RAP_NO_RISK_TRUST_ELIGIBILITY"
    RAP_RISK_GATED_ABLATION = "RAP_RISK_GATED_ABLATION"
    COVER_M4_REF = "COVER_M4_REF"
    COVER_EXEC_K4 = "COVER_EXEC_K4"
    COVER_UNCERT_SNR1 = "COVER_UNCERT_SNR1"
    COVER_PHYS_SCORE = "COVER_PHYS_SCORE"
    COVER_REV_PHYS = "COVER_REV_PHYS"
    COVER_TCP_JERK_HORIZON = "COVER_TCP_JERK_HORIZON"
    COVER_EW_RESPONSE = "COVER_EW_RESPONSE"
    COVER_GRID10 = "COVER_GRID10"
    COVER_GRID5 = "COVER_GRID5"
    COVER_GRID10_ZERO_FALLBACK = "COVER_GRID10_ZERO_FALLBACK"
    COVER_GRID10_TIERED_FALLBACK = "COVER_GRID10_TIERED_FALLBACK"
    COVER_TIERED_PREPROJECTION_BOUND = "COVER_TIERED_PREPROJECTION_BOUND"
    COVER_PROJECTOR_CONTEXT_CACHE = "COVER_PROJECTOR_CONTEXT_CACHE"
    COVER_PROJECTOR_TERMINAL_ENDPOINT_REUSE = "COVER_PROJECTOR_TERMINAL_ENDPOINT_REUSE"
    COVER_K5_ALPHA5 = "COVER_K5_ALPHA5"
    COVER_K5_ALPHA5_NO_RELATIVE_CAP = "COVER_K5_ALPHA5_NO_RELATIVE_CAP"
    COVER_K5_ALPHA_COARSE_SCAN = "COVER_K5_ALPHA_COARSE_SCAN"
    COVER_K5_ALPHA_COARSE_A6_FREE = "COVER_K5_ALPHA_COARSE_A6_FREE"
    COVER_K5_ALPHA_FINE_A6_FREE = "COVER_K5_ALPHA_FINE_A6_FREE"
    COVER_K5_ALPHA_COARSE_A6_FREE_NO_CORRECTION_CAP = (
        "COVER_K5_ALPHA_COARSE_A6_FREE_NO_CORRECTION_CAP"
    )
    COVER_K5_ALPHA_FINE5_A6_FREE_NO_CORRECTION_CAP = (
        "COVER_K5_ALPHA_FINE5_A6_FREE_NO_CORRECTION_CAP"
    )
    COVER_TERMINAL_REF_CAP003 = "COVER_TERMINAL_REF_CAP003"
    COVER_TERMINAL_VALIDATOR_ONLY_CAP003 = "COVER_TERMINAL_VALIDATOR_ONLY_CAP003"
    COVER_ALPHA_BANK_065_095_VALIDATOR_ONLY_CAP003 = (
        "COVER_ALPHA_BANK_065_095_VALIDATOR_ONLY_CAP003"
    )


TASK105_COVER_SCREEN_METHOD_IDS = {
    MethodId.COVER_M4_REF,
    MethodId.COVER_EXEC_K4,
    MethodId.COVER_UNCERT_SNR1,
    MethodId.COVER_PHYS_SCORE,
    MethodId.COVER_REV_PHYS,
}
COVER_SCREEN_METHOD_IDS = TASK105_COVER_SCREEN_METHOD_IDS | {
    MethodId.COVER_TCP_JERK_HORIZON,
    MethodId.COVER_EW_RESPONSE,
    MethodId.COVER_GRID10,
    MethodId.COVER_GRID5,
    MethodId.COVER_GRID10_ZERO_FALLBACK,
    MethodId.COVER_GRID10_TIERED_FALLBACK,
    MethodId.COVER_TIERED_PREPROJECTION_BOUND,
    MethodId.COVER_PROJECTOR_CONTEXT_CACHE,
    MethodId.COVER_PROJECTOR_TERMINAL_ENDPOINT_REUSE,
    MethodId.COVER_K5_ALPHA5,
    MethodId.COVER_K5_ALPHA5_NO_RELATIVE_CAP,
    MethodId.COVER_K5_ALPHA_COARSE_SCAN,
    MethodId.COVER_K5_ALPHA_COARSE_A6_FREE,
    MethodId.COVER_K5_ALPHA_FINE_A6_FREE,
    MethodId.COVER_K5_ALPHA_COARSE_A6_FREE_NO_CORRECTION_CAP,
    MethodId.COVER_K5_ALPHA_FINE5_A6_FREE_NO_CORRECTION_CAP,
    MethodId.COVER_TERMINAL_REF_CAP003,
    MethodId.COVER_TERMINAL_VALIDATOR_ONLY_CAP003,
    MethodId.COVER_ALPHA_BANK_065_095_VALIDATOR_ONLY_CAP003,
}


class RuntimeMethod(str, Enum):
    NO_M1_RUNTIME = "NO_M1_RUNTIME"
    VANILLA = "VANILLA"
    TE = "TE"
    TE_MATCHED_CADENCE_CONTROL = "TE_MATCHED_CADENCE_CONTROL"
    RTC_D0_MECHANISM_CONTROL = "RTC_D0"
    RTC_D0 = RTC_D0_MECHANISM_CONTROL
    RAP = "RAP"
    RAP_NO_RISK_TRUST_ELIGIBILITY = "RAP_NO_RISK_TRUST_ELIGIBILITY"
    RAP_RISK_GATED_ABLATION = "RAP_RISK_GATED_ABLATION"


@dataclass(frozen=True)
class M1RuntimeConfig:
    """Explicit method configuration; no method parameters are silently selected."""

    method: RuntimeMethod = RuntimeMethod.NO_M1_RUNTIME
    te_decay: float | None = None
    te_history_cap: int | None = None
    rap_runtime: "RapRuntimeConfig | None" = None
    rtc_runtime: "RtcD0RuntimeConfig | None" = None

    def __post_init__(self) -> None:
        if self.method is RuntimeMethod.TE:
            if self.te_decay is not None or self.te_history_cap is not None:
                raise ValueError("primary ACT TE has frozen decay=0.01 and horizon=10")
        elif self.method is RuntimeMethod.TE_MATCHED_CADENCE_CONTROL:
            if self.te_decay is None or not np.isfinite(self.te_decay) or self.te_decay < 0:
                raise ValueError("matched-cadence TE control requires finite non-negative te_decay")
            if self.te_history_cap is not None and (
                isinstance(self.te_history_cap, bool)
                or not isinstance(self.te_history_cap, int)
                or self.te_history_cap < 1
            ):
                raise ValueError("te_history_cap is None or an integer >= 1")
        elif self.te_decay is not None or self.te_history_cap is not None:
            raise ValueError("TE parameters require the matched-cadence TE control")
        if self.method in {
            RuntimeMethod.RAP,
            RuntimeMethod.RAP_NO_RISK_TRUST_ELIGIBILITY,
            RuntimeMethod.RAP_RISK_GATED_ABLATION,
        }:
            if self.rap_runtime is None:
                raise ValueError("RAP requires explicit runtime config")
        elif self.rap_runtime is not None:
            raise ValueError("RAP runtime config requires RAP runtime")
        if self.method is RuntimeMethod.RTC_D0:
            if self.rtc_runtime is None:
                raise ValueError("RTC_D0 requires explicit runtime config")
        elif self.rtc_runtime is not None:
            raise ValueError("RTC runtime config requires RTC_D0 runtime")


class M1MethodNotActiveError(ValueError):
    """Raised before model sampling when a dormant method is requested."""

    code = "M1_METHOD_NOT_ACTIVE"


class M1ContextRequiredError(ValueError):
    code = "M1_CONTEXT_REQUIRED"


@dataclass(frozen=True)
class RtcD0RuntimeConfig:
    beta: float = 5.0
    num_steps: int = 10

    def __post_init__(self) -> None:
        if (
            isinstance(self.beta, bool)
            or not isinstance(self.beta, numbers.Real)
            or not np.isfinite(float(self.beta))
            or float(self.beta) != 5.0
        ):
            raise ValueError("primary synchronous RTC-d0 freezes beta=5")
        if isinstance(self.num_steps, bool) or not isinstance(self.num_steps, int) or self.num_steps != 10:
            raise ValueError("primary synchronous RTC-d0 freezes num_steps=10")


@dataclass(frozen=True)
class M1Context:
    schema_version: str
    run_id: str
    episode_id: str
    method_id: MethodId
    episode_start: bool
    logical_timestep: int
    stride: int
    previous_executed_count: int
    completed_tcp_positions: np.ndarray
    current_tcp_translation: np.ndarray
    execution_mode: str | None = None

    @classmethod
    def from_mapping(cls, value: dict) -> "M1Context":
        required = {
            "schema_version",
            "run_id",
            "episode_id",
            "method_id",
            "episode_start",
            "logical_timestep",
            "stride",
            "previous_executed_count",
            "completed_tcp_positions",
            "current_tcp_translation",
        }
        if set(value) not in (required, required | {"execution_mode"}):
            raise ValueError("invalid M1 context schema")
        if (
            value["schema_version"] != SCHEMA_VERSION
            or not isinstance(value["run_id"], str)
            or not value["run_id"]
            or not isinstance(value["episode_id"], str)
            or not value["episode_id"]
        ):
            raise ValueError("invalid M1 context identity")
        if (
            not isinstance(value["episode_start"], bool)
            or isinstance(value["logical_timestep"], bool)
            or not isinstance(value["logical_timestep"], int)
            or value["logical_timestep"] < 0
        ):
            raise ValueError("invalid M1 logical timestep")
        try:
            method = MethodId(value["method_id"])
        except ValueError as error:
            raise ValueError("invalid M1 method") from error
        execution_mode = value.get("execution_mode")
        if "execution_mode" in value and (
            execution_mode != "VANILLA_VARIABLE_H10" or method is not MethodId.VANILLA
        ):
            raise ValueError("invalid variable execution identity")
        allowed_strides = {1} if method is MethodId.TE else {3, 5, 9}
        if (
            isinstance(value["stride"], bool)
            or not isinstance(value["stride"], int)
            or value["stride"] not in allowed_strides
            or isinstance(value["previous_executed_count"], bool)
            or not isinstance(value["previous_executed_count"], int)
            or not 0 <= value["previous_executed_count"] <= (9 if execution_mode else value["stride"])
        ):
            raise ValueError("invalid M1 execution count")
        completed = np.asarray(value["completed_tcp_positions"])
        current = np.asarray(value["current_tcp_translation"])
        if (
            completed.shape != (value["previous_executed_count"], 3)
            or current.shape != (3,)
            or not np.issubdtype(completed.dtype, np.number)
            or not np.issubdtype(current.dtype, np.number)
            or not np.isfinite(completed).all()
            or not np.isfinite(current).all()
        ):
            raise ValueError("invalid M1 TCP feedback")
        if value["episode_start"] and (
            value["logical_timestep"] != 0 or value["previous_executed_count"] != 0 or completed.shape != (0, 3)
        ):
            raise ValueError("invalid M1 episode start")
        return cls(
            value["schema_version"],
            value["run_id"],
            value["episode_id"],
            method,
            value["episode_start"],
            value["logical_timestep"],
            value["stride"],
            value["previous_executed_count"],
            completed.copy(),
            current.copy(),
            execution_mode,
        )


@dataclass(frozen=True)
class _EpisodeState:
    method_id: MethodId
    stride: int
    last_logical_query_timestep: int
    execution_mode: str | None = None


@dataclass(frozen=True)
class LifecycleUpdate:
    key: tuple[str, str]
    state: _EpisodeState


class M1LifecycleTracker:
    """Server/Policy-side validation state, keyed by operational episode identity."""

    def __init__(self) -> None:
        self._episodes: dict[tuple[str, str], _EpisodeState] = {}

    def preview(self, value: dict) -> tuple[M1Context, LifecycleUpdate]:
        context = M1Context.from_mapping(value)
        key = (context.run_id, context.episode_id)
        previous = self._episodes.get(key)
        if context.episode_start:
            if previous is not None:
                raise ValueError("duplicate M1 episode start")
            return context, LifecycleUpdate(
                key, _EpisodeState(context.method_id, context.stride, context.logical_timestep, context.execution_mode)
            )
        if previous is None:
            raise ValueError("episode change without M1 episode start")
        if previous.method_id is not context.method_id:
            raise ValueError("M1 method switch")
        if previous.execution_mode != context.execution_mode:
            raise ValueError("M1 execution mode switch")
        if previous.stride != context.stride and context.execution_mode is None:
            raise ValueError("M1 stride switch")
        if (
            context.previous_executed_count != previous.stride
            or context.logical_timestep != previous.last_logical_query_timestep + previous.stride
        ):
            raise ValueError("stale or skipped M1 logical timestep")
        return context, LifecycleUpdate(key, _EpisodeState(context.method_id, context.stride, context.logical_timestep, context.execution_mode))

    def commit(self, update: LifecycleUpdate) -> None:
        self._episodes[update.key] = update.state


@dataclass(frozen=True)
class TeUpdate:
    key: tuple[str, str, MethodId]
    ensemble: ActTemporalEnsemble | MatchedCadenceTemporalEnsemble
    decision: ActTeDecision | None


class M1TeState:
    """Transactional primary ACT TE plus an explicit legacy control."""

    def __init__(self) -> None:
        self._ensembles: dict[
            tuple[str, str, MethodId], ActTemporalEnsemble | MatchedCadenceTemporalEnsemble
        ] = {}

    def preview(
        self, context: M1Context, raw_actions: np.ndarray, config: M1RuntimeConfig
    ) -> tuple[np.ndarray, TeUpdate]:
        raw = np.asarray(raw_actions)
        if (
            raw.shape != (10, 32)
            or not np.issubdtype(raw.dtype, np.floating)
            or not np.isfinite(raw).all()
            or config.method not in {RuntimeMethod.TE, RuntimeMethod.TE_MATCHED_CADENCE_CONTROL}
        ):
            raise ValueError("invalid normalized TE sample")
        key = (context.run_id, context.episode_id, context.method_id)
        if config.method is RuntimeMethod.TE:
            if context.method_id is not MethodId.TE or context.stride != 1:
                raise ValueError("primary ACT TE requires TE identity and stride 1")
            ensemble = copy.deepcopy(self._ensembles.get(key, ActTemporalEnsemble()))
            if not isinstance(ensemble, ActTemporalEnsemble):
                raise ValueError("TE state identity mismatch")
            decision = ensemble.step(
                query_timestep=context.logical_timestep,
                chunk=raw,
                episode_id=context.episode_id,
                method_id=context.method_id.value,
            )
            result = raw.copy()
            result[0] = decision.normalized_model_row
            return result, TeUpdate(key, ensemble, decision)
        if context.method_id is not MethodId.TE_MATCHED_CADENCE_CONTROL:
            raise ValueError("matched-cadence TE state identity mismatch")
        ensemble = copy.deepcopy(
            self._ensembles.get(key, MatchedCadenceTemporalEnsemble(history_cap=config.te_history_cap))
        )
        if not isinstance(ensemble, MatchedCadenceTemporalEnsemble):
            raise ValueError("TE state identity mismatch")
        raw_chunk = raw[:, :7].copy()
        ensemble.add(
            query_timestep=context.logical_timestep,
            chunk=raw_chunk,
            episode_id=context.episode_id,
            method_id=context.method_id.value,
        )
        output = np.stack(
            [
                ensemble.action_at(
                    timestep=context.logical_timestep + offset,
                    episode_id=context.episode_id,
                    method_id=context.method_id.value,
                    decay=config.te_decay,
                )
                for offset in range(10)
            ]
        )
        result = raw.copy()
        result[:, :7] = output
        return result, TeUpdate(key, ensemble, None)

    def commit(self, update: TeUpdate) -> None:
        self._ensembles[update.key] = update.ensemble


@dataclass(frozen=True)
class RtcD0Update:
    key: tuple[str, str, MethodId]
    chunk: np.ndarray


class M1RtcD0State:
    """Policy-owned prior store for the nonprimary synchronous RTC-d0 control."""

    def __init__(self) -> None:
        self._priors: dict[tuple[str, str, MethodId], np.ndarray] = {}

    def previous(self, context: M1Context) -> np.ndarray | None:
        key = (context.run_id, context.episode_id, context.method_id)
        if context.episode_start:
            return None
        if key not in self._priors:
            raise ValueError("RTC prior missing for non-start query")
        return self._priors[key].copy()

    def pending(self, context: M1Context, chunk: np.ndarray) -> RtcD0Update:
        value = np.asarray(chunk)
        if value.shape != (10, 32) or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError("invalid RTC normalized chunk")
        return RtcD0Update((context.run_id, context.episode_id, context.method_id), value.copy())

    def commit(self, update: RtcD0Update) -> None:
        self._priors[update.key] = update.chunk.copy()


class RapCausalStatus(str, Enum):
    """Fail-closed status for dormant RAP causal-state preparation."""

    AVAILABLE = "AVAILABLE"
    RESPONSE_UNAVAILABLE = "RESPONSE_UNAVAILABLE"
    TIMESTAMP_MISMATCH = "TIMESTAMP_MISMATCH"
    SHAPE_MISMATCH = "SHAPE_MISMATCH"
    NONFINITE_INPUT = "NONFINITE_INPUT"


@dataclass(frozen=True)
class _RapEpisodeState:
    previous_raw_chunk: np.ndarray
    previous_selected_chunk: np.ndarray
    completed_action_history: np.ndarray
    completed_tcp_history: np.ndarray
    previous_query_tcp: np.ndarray
    last_logical_timestep: int


@dataclass(frozen=True)
class RapCausalSnapshot:
    """Causal-only inputs for a future RAP/online-Risk integration."""

    status: RapCausalStatus
    current_raw_chunk: np.ndarray | None
    previous_raw_chunk: np.ndarray | None
    previous_selected_chunk: np.ndarray | None
    completed_action_history: np.ndarray
    completed_tcp_history: np.ndarray
    completed_tcp_displacements: np.ndarray
    previous_action: np.ndarray | None
    response_fit: ResponseFit
    risk_features: dict[str, RiskFeature]


@dataclass(frozen=True)
class RapCausalUpdate:
    key: tuple[str, str, MethodId]
    state_before_selected: _RapEpisodeState


class M1RapCausalState:
    """Transactional causal history keyed by run, episode, and dormant RAP method.

    This class prepares no correction and makes no eligibility or risk decision.
    """

    def __init__(self) -> None:
        self._episodes: dict[tuple[str, str, MethodId], _RapEpisodeState] = {}

    @staticmethod
    def _empty_response(*, reason: str) -> ResponseFit:
        return ResponseFit(matrix=None, available=False, reason=reason)

    @staticmethod
    def _empty_snapshot(status: RapCausalStatus, raw: np.ndarray | None = None) -> RapCausalSnapshot:
        empty_actions = np.empty((0, 7), dtype=np.float64)
        empty_tcp = np.empty((0, 3), dtype=np.float64)
        return RapCausalSnapshot(
            status,
            raw,
            None,
            None,
            empty_actions,
            empty_tcp,
            empty_tcp.copy(),
            None,
            M1RapCausalState._empty_response(reason=status.value),
            {},
        )

    @staticmethod
    def _copy_state(state: _RapEpisodeState) -> _RapEpisodeState:
        return _RapEpisodeState(
            state.previous_raw_chunk.copy(),
            state.previous_selected_chunk.copy(),
            state.completed_action_history.copy(),
            state.completed_tcp_history.copy(),
            state.previous_query_tcp.copy(),
            state.last_logical_timestep,
        )

    @staticmethod
    def _interval_displacements(previous_tcp: np.ndarray, completed_tcp: np.ndarray) -> np.ndarray:
        points = np.concatenate((previous_tcp[None, :], completed_tcp), axis=0)
        return np.diff(points, axis=0)

    def preview(
        self,
        context: M1Context,
        raw_actions: np.ndarray,
        *,
        min_samples: int,
        ridge: float,
        condition_limit: float,
        tcp_translation_units: str,
        response_mode: str = "PRIMARY",
        minimum_prequential_predictions: int = 0,
        response_half_life_completed_pairs: int = 0,
    ) -> tuple[RapCausalSnapshot, RapCausalUpdate | None]:
        """Prepare causal state without committing the current selected chunk."""
        if context.method_id not in {
            MethodId.RAP,
            MethodId.RAP_NO_RISK_TRUST_ELIGIBILITY,
            MethodId.RAP_RISK_GATED_ABLATION,
            *COVER_SCREEN_METHOD_IDS,
        }:
            raise ValueError("RAP causal state requires a RAP method identity")
        raw = np.asarray(raw_actions)
        if raw.shape != (10, 32) or not np.issubdtype(raw.dtype, np.floating):
            return self._empty_snapshot(RapCausalStatus.SHAPE_MISMATCH), None
        if not np.isfinite(raw).all():
            return self._empty_snapshot(RapCausalStatus.NONFINITE_INPUT), None
        current_raw = raw[:, :7].copy()
        key = (context.run_id, context.episode_id, context.method_id)
        previous = self._episodes.get(key)
        if context.episode_start:
            if previous is not None:
                raise ValueError("duplicate RAP causal episode start")
            state = _RapEpisodeState(
                previous_raw_chunk=current_raw.copy(),
                previous_selected_chunk=np.empty((0, 7), dtype=current_raw.dtype),
                completed_action_history=np.empty((0, 7), dtype=current_raw.dtype),
                completed_tcp_history=context.current_tcp_translation[None, :].copy(),
                previous_query_tcp=context.current_tcp_translation.copy(),
                last_logical_timestep=context.logical_timestep,
            )
            response_functions = {
                "PRIMARY": fit_completed_translation_response,
                "PREQUENTIAL": fit_completed_translation_response_prequential,
                "EXPONENTIALLY_WEIGHTED": (
                    fit_completed_translation_response_exponentially_weighted
                ),
            }
            if response_mode not in response_functions:
                raise ValueError("invalid COVER response mode")
            response_function = response_functions[response_mode]
            response_kwargs = {
                "min_samples": min_samples,
                "ridge": ridge,
                "condition_limit": condition_limit,
            }
            if response_mode == "PREQUENTIAL":
                response_kwargs["minimum_predictions"] = minimum_prequential_predictions
            elif response_mode == "EXPONENTIALLY_WEIGHTED":
                response_kwargs["half_life_completed_pairs"] = (
                    response_half_life_completed_pairs
                )
            response = response_function(
                state.completed_action_history[:, :3],
                np.empty((0, 3)),
                **response_kwargs,
            )
            features = causal_risk_features(
                executed_tcp=state.completed_tcp_history,
                raw_chunk=current_raw,
                previous_raw_chunk=None,
                previous_executed_count=0,
                logical_timestep=context.logical_timestep,
                tcp_translation_units=tcp_translation_units,
            )
            snapshot = RapCausalSnapshot(
                RapCausalStatus.RESPONSE_UNAVAILABLE,
                current_raw,
                None,
                None,
                state.completed_action_history.copy(),
                state.completed_tcp_history.copy(),
                np.empty((0, 3), dtype=current_raw.dtype),
                None,
                response,
                features,
            )
            return snapshot, RapCausalUpdate(key, self._copy_state(state))
        if previous is None:
            raise ValueError("RAP causal episode change without start")
        completed_tcp = context.completed_tcp_positions.copy()
        if context.previous_executed_count != context.stride or completed_tcp.shape != (context.stride, 3):
            return self._empty_snapshot(RapCausalStatus.SHAPE_MISMATCH, current_raw), None
        if not np.array_equal(context.current_tcp_translation, completed_tcp[-1]):
            return self._empty_snapshot(RapCausalStatus.TIMESTAMP_MISMATCH, current_raw), None
        if not np.isfinite(completed_tcp).all():
            return self._empty_snapshot(RapCausalStatus.NONFINITE_INPUT, current_raw), None
        if context.logical_timestep != previous.last_logical_timestep + context.stride:
            anchor = _RapEpisodeState(
                current_raw.copy(),
                np.empty((0, 7), dtype=current_raw.dtype),
                np.empty((0, 7), dtype=current_raw.dtype),
                context.current_tcp_translation[None, :].copy(),
                context.current_tcp_translation.copy(),
                context.logical_timestep,
            )
            return RapCausalSnapshot(
                RapCausalStatus.TIMESTAMP_MISMATCH,
                current_raw,
                None,
                None,
                anchor.completed_action_history.copy(),
                anchor.completed_tcp_history.copy(),
                np.empty((0, 3), dtype=current_raw.dtype),
                None,
                self._empty_response(reason="TIMESTAMP_MISMATCH"),
                {},
            ), RapCausalUpdate(key, anchor)
        completed_actions = previous.previous_selected_chunk[: context.stride].copy()
        if (
            completed_actions.shape != (context.stride, 7)
            or not np.isfinite(completed_actions).all()
            or not np.isfinite(completed_tcp).all()
        ):
            return self._empty_snapshot(RapCausalStatus.NONFINITE_INPUT, current_raw), None
        displacements = self._interval_displacements(previous.previous_query_tcp, completed_tcp)
        state = _RapEpisodeState(
            previous_raw_chunk=current_raw.copy(),
            previous_selected_chunk=np.empty((0, 7), dtype=current_raw.dtype),
            completed_action_history=np.concatenate((previous.completed_action_history, completed_actions), axis=0),
            completed_tcp_history=np.concatenate((previous.completed_tcp_history, completed_tcp), axis=0),
            previous_query_tcp=context.current_tcp_translation.copy(),
            last_logical_timestep=context.logical_timestep,
        )
        all_displacements = np.diff(state.completed_tcp_history, axis=0)
        response_functions = {
            "PRIMARY": fit_completed_translation_response,
            "PREQUENTIAL": fit_completed_translation_response_prequential,
            "EXPONENTIALLY_WEIGHTED": (
                fit_completed_translation_response_exponentially_weighted
            ),
        }
        if response_mode not in response_functions:
            raise ValueError("invalid COVER response mode")
        response_function = response_functions[response_mode]
        response_kwargs = {
            "min_samples": min_samples,
            "ridge": ridge,
            "condition_limit": condition_limit,
        }
        if response_mode == "PREQUENTIAL":
            response_kwargs["minimum_predictions"] = minimum_prequential_predictions
        elif response_mode == "EXPONENTIALLY_WEIGHTED":
            response_kwargs["half_life_completed_pairs"] = response_half_life_completed_pairs
        response = response_function(
            state.completed_action_history[:, :3],
            all_displacements,
            **response_kwargs,
        )
        features = causal_risk_features(
            executed_tcp=state.completed_tcp_history,
            raw_chunk=current_raw,
            previous_raw_chunk=previous.previous_raw_chunk,
            previous_executed_count=context.previous_executed_count,
            logical_timestep=context.logical_timestep,
            tcp_translation_units=tcp_translation_units,
        )
        status = RapCausalStatus.AVAILABLE if response.available else RapCausalStatus.RESPONSE_UNAVAILABLE
        snapshot = RapCausalSnapshot(
            status,
            current_raw,
            previous.previous_raw_chunk.copy(),
            previous.previous_selected_chunk.copy(),
            state.completed_action_history.copy(),
            state.completed_tcp_history.copy(),
            displacements,
            completed_actions[-1].copy(),
            response,
            features,
        )
        return snapshot, RapCausalUpdate(key, state)

    def commit(self, update: RapCausalUpdate, selected_chunk: np.ndarray) -> None:
        """Commit only the selected chunk for derivation at the next query."""
        selected = np.asarray(selected_chunk)
        if (
            selected.shape != (10, 7)
            or not np.issubdtype(selected.dtype, np.floating)
            or not np.isfinite(selected).all()
        ):
            raise ValueError("invalid selected normalized RAP chunk")
        state = update.state_before_selected
        self._episodes[update.key] = _RapEpisodeState(
            state.previous_raw_chunk.copy(),
            selected.copy(),
            state.completed_action_history.copy(),
            state.completed_tcp_history.copy(),
            state.previous_query_tcp.copy(),
            state.last_logical_timestep,
        )


@dataclass(frozen=True)
class RapResponseRuntimeConfig:
    min_samples: int
    ridge: float
    condition_limit: float

    def __post_init__(self) -> None:
        if self.min_samples != MIN_SAMPLES or self.ridge != RIDGE or self.condition_limit != CONDITION_LIMIT:
            raise ValueError("M2 RAP response config is frozen at 8/1e-4/1e8")


@dataclass(frozen=True)
class RapReversalRuntimeConfig:
    direction_window: int
    reversal_cos_threshold: float
    min_consistent_steps: int
    velocity_epsilon: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.direction_window, bool)
            or isinstance(self.min_consistent_steps, bool)
            or not isinstance(self.direction_window, int)
            or not isinstance(self.min_consistent_steps, int)
            or self.direction_window < 1
            or self.min_consistent_steps < 1
            or not np.isfinite((self.reversal_cos_threshold, self.velocity_epsilon)).all()
            or not -1 <= self.reversal_cos_threshold <= 1
            or self.velocity_epsilon <= 0
        ):
            raise ValueError("invalid RAP reversal config")


@dataclass(frozen=True)
class RapRuntimeConfig:
    core: RapConfig
    candidate_specs: tuple[RapCandidateSpec, ...]
    response: RapResponseRuntimeConfig
    reversal: RapReversalRuntimeConfig
    risk: RapRiskConfig | None = None
    tcp_translation_units: str = "m"
    method_id: MethodId = MethodId.RAP
    method_identity: str = "RAP-COVER-EXECUTION-PROJECTION-CF-VETO"
    objective_count_mode: str = _rap_cover.OBJECTIVE_LEGACY_NEXT
    response_mode: str = "PRIMARY"
    response_rotation_ridge: float = 1e-4
    selection_mode: str = _rap_cover.SELECTION_ACTION
    candidate_ranking_mode: str = _rap_cover.RANKING_ACTION
    reversal_space: str = _rap_cover.REVERSAL_LEGACY
    uncertainty_scale: float | None = None
    minimum_prequential_predictions: int = 0
    response_half_life_completed_pairs: int = 0
    candidate_evaluation_policy: str = _rap_cover.CANDIDATE_EVALUATION_FULL
    preprojection_objective_bound: str = _rap_cover.PREPROJECTION_BOUND_DISABLED
    projector_context_cache_mode: str = _rap_cover.PROJECTOR_CONTEXT_CACHE_DISABLED
    terminal_constraint_mode: str = _rap_cover.TERMINAL_CONSTRAINT_PREFIX_EXIT
    terminal_projection_role: str = _rap_cover.TERMINAL_PROJECTION_REF

    def __post_init__(self) -> None:
        if isinstance(self.core.k_max, bool):
            raise ValueError("invalid RAP core runtime config")
        ordered = tuple(
            sorted(self.candidate_specs, key=lambda x: (x.k, x.alpha, x.transition_weights, x.relative_cap))
        )
        if len({(item.k, item.alpha, tuple(item.transition_weights), item.relative_cap) for item in ordered}) != len(
            ordered
        ):
            raise ValueError("duplicate RAP candidate spec")
        if self.tcp_translation_units != "m":
            raise ValueError("M4 RAP-COVER requires LIBERO TCP translation units in meters")
        if self.risk is not None and self.risk.tcp_translation_units != self.tcp_translation_units:
            raise ValueError("RAP risk and causal TCP units disagree")
        if (
            not isinstance(self.method_id, MethodId)
            or not isinstance(self.method_identity, str)
            or not self.method_identity
            or self.objective_count_mode
            not in {_rap_cover.OBJECTIVE_LEGACY_NEXT, _rap_cover.OBJECTIVE_EXECUTED_ONLY}
            or self.response_mode not in {
                "PRIMARY",
                "PREQUENTIAL",
                "EXPONENTIALLY_WEIGHTED",
            }
            or self.selection_mode
            not in {
                _rap_cover.SELECTION_ACTION,
                _rap_cover.SELECTION_PHYSICAL,
                _rap_cover.SELECTION_TCP_HORIZON_ISJ,
            }
            or self.response_rotation_ridge not in (1e-4, 1e-3)
            or (self.response_rotation_ridge != 1e-4 and self.response_mode != "PRIMARY")
            or self.candidate_ranking_mode not in {_rap_cover.RANKING_ACTION, _rap_cover.RANKING_ELIGIBLE_PHYSICAL}
            or (self.candidate_ranking_mode != _rap_cover.RANKING_ACTION and self.selection_mode != _rap_cover.SELECTION_ACTION)
            or self.reversal_space not in {_rap_cover.REVERSAL_LEGACY, _rap_cover.REVERSAL_RESPONSE_TCP}
            or (
                self.uncertainty_scale is not None
                and (not np.isfinite(self.uncertainty_scale) or self.uncertainty_scale <= 0)
            )
            or isinstance(self.minimum_prequential_predictions, bool)
            or not isinstance(self.minimum_prequential_predictions, int)
            or self.minimum_prequential_predictions < 0
            or isinstance(self.response_half_life_completed_pairs, bool)
            or not isinstance(self.response_half_life_completed_pairs, int)
            or (
                self.response_mode == "EXPONENTIALLY_WEIGHTED"
                and self.response_half_life_completed_pairs
                != EW_HALF_LIFE_COMPLETED_PAIRS
            )
            or (
                self.response_mode != "EXPONENTIALLY_WEIGHTED"
                and self.response_half_life_completed_pairs != 0
            )
            or (
                self.response_mode == "EXPONENTIALLY_WEIGHTED"
                and self.minimum_prequential_predictions != 0
            )
            or self.candidate_evaluation_policy
            not in _rap_cover.CANDIDATE_EVALUATION_POLICIES
            or self.preprojection_objective_bound
            not in _rap_cover.PREPROJECTION_BOUND_MODES
            or self.projector_context_cache_mode
            not in _rap_cover.PROJECTOR_CONTEXT_CACHE_MODES
            or self.terminal_constraint_mode
            not in _rap_cover.TERMINAL_CONSTRAINT_MODES
            or self.terminal_projection_role not in _rap_cover.TERMINAL_PROJECTION_ROLES
            or (
                self.method_id is MethodId.COVER_TERMINAL_REF_CAP003
                and self.terminal_projection_role != _rap_cover.TERMINAL_PROJECTION_REF
            )
            or (
                self.method_id
                in {
                    MethodId.COVER_TERMINAL_VALIDATOR_ONLY_CAP003,
                    MethodId.COVER_ALPHA_BANK_065_095_VALIDATOR_ONLY_CAP003,
                }
                and self.terminal_projection_role
                != _rap_cover.TERMINAL_PROJECTION_VALIDATOR_ONLY
            )
            or (
                self.terminal_constraint_mode
                == _rap_cover.TERMINAL_CONSTRAINT_EXECUTED_INTERNAL
                and self.projector_context_cache_mode
                != _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE
            )
            or (
                self.preprojection_objective_bound
                != _rap_cover.PREPROJECTION_BOUND_DISABLED
                and self.selection_mode != _rap_cover.SELECTION_ACTION
            )
        ):
            raise ValueError("invalid COVER runtime variant")
        object.__setattr__(self, "candidate_specs", ordered)


@dataclass(frozen=True)
class RapRuntimePreview:
    actions: np.ndarray
    raw_chunk: np.ndarray
    causal_update: RapCausalUpdate | None
    selected_chunk: np.ndarray
    causal_status: str
    reason: str
    decision: RapDecision | None
    risk: RapRiskDecision | None
    response_available: bool
    response: ResponseFit | None = None
    risk_features: dict[str, RiskFeature] | None = None
    risk_diagnostic_only: bool = True


class M1RapState:
    """Transactional active wrapper; the selected chunk is committed only by Policy."""

    def __init__(self) -> None:
        self._causal = M1RapCausalState()
        self._active_response_state = None

    @classmethod
    def for_active_response(
        cls,
        *,
        response_mode: str,
        action_q01: np.ndarray,
        action_q99: np.ndarray,
        normalization_identity: str,
        response_experiment=None,
    ) -> "M1RapState":
        """Build the common Q3/Q6 integration used by a paired experiment.

        The ordinary Policy-owned Q3 path remains the default.  This explicit
        factory is for integrations that must commit the post-rotation
        controller command before either estimator may advance its history.
        Both arms use the same state-machine type and public methods.
        """

        from chunkamend.hardware.core.methods.rap_response_6d_active import ActiveResponseState

        result = cls()
        result._active_response_state = ActiveResponseState(
            action_q01=action_q01,
            action_q99=action_q99,
            normalization_identity=normalization_identity,
            response_mode=response_mode,
            response_experiment=response_experiment,
        )
        return result

    def preview_active_response(
        self,
        context: M1Context,
        raw_actions: np.ndarray,
        config: RapRuntimeConfig,
        *,
        raw_controller_bound_physical_actions: np.ndarray,
        selected_controller_bound_physical_actions: np.ndarray,
    ):
        if self._active_response_state is None:
            raise ValueError("M1RapState active-response integration is not configured")
        return self._active_response_state.preview(
            context,
            raw_actions,
            config,
            raw_controller_bound_physical_actions=raw_controller_bound_physical_actions,
            selected_controller_bound_physical_actions=selected_controller_bound_physical_actions,
        )

    def commit_active_response(
        self,
        preview,
        *,
        controller_bound_physical_actions: np.ndarray,
        representation_receipt: dict,
    ) -> None:
        if self._active_response_state is None:
            raise ValueError("M1RapState active-response integration is not configured")
        self._active_response_state.commit(
            preview,
            controller_bound_physical_actions=controller_bound_physical_actions,
            representation_receipt=representation_receipt,
        )

    def active_response_receipt(self, preview) -> dict:
        if self._active_response_state is None:
            raise ValueError("M1RapState active-response integration is not configured")
        return self._active_response_state.receipt(preview)

    def preview(self, context: M1Context, raw_actions: np.ndarray, config: RapRuntimeConfig) -> RapRuntimePreview:
        snapshot, update = self._causal.preview(
            context,
            raw_actions,
            min_samples=config.response.min_samples,
            ridge=config.response.ridge,
            condition_limit=config.response.condition_limit,
            tcp_translation_units=config.tcp_translation_units,
            response_mode=config.response_mode,
            minimum_prequential_predictions=config.minimum_prequential_predictions,
            response_half_life_completed_pairs=(
                config.response_half_life_completed_pairs
            ),
        )
        raw = np.asarray(raw_actions).copy()
        if snapshot.current_raw_chunk is None:
            return RapRuntimePreview(
                actions=raw,
                raw_chunk=raw[:, :7].copy(),
                causal_update=None,
                selected_chunk=raw[:, :7].copy(),
                causal_status=snapshot.status.value,
                reason=snapshot.status.value,
                decision=None,
                risk=None,
                response_available=False,
                response=snapshot.response_fit,
                risk_features=snapshot.risk_features,
            )
        risk = (
            None
            if config.risk is None
            else evaluate_rap_risk(
                config.risk,
                stride=context.stride,
                logical_timestep=context.logical_timestep,
                features=snapshot.risk_features,
            )
        )
        risk_gated = context.method_id is MethodId.RAP_RISK_GATED_ABLATION
        if risk_gated and risk is None:
            raise ValueError("RAP risk-gated ablation requires an explicit selected risk config")
        ablated = not risk_gated
        structural_failures = {
            RapCausalStatus.TIMESTAMP_MISMATCH,
            RapCausalStatus.SHAPE_MISMATCH,
            RapCausalStatus.NONFINITE_INPUT,
        }
        if snapshot.status in structural_failures:
            return RapRuntimePreview(
                raw,
                raw[:, :7].copy(),
                update,
                raw[:, :7].copy(),
                snapshot.status.value,
                snapshot.status.value,
                None,
                risk,
                snapshot.response_fit.available,
                snapshot.response_fit,
                snapshot.risk_features,
                not risk_gated,
            )
        if risk_gated and not risk.available:
            reason = risk.reason.value
            return RapRuntimePreview(
                raw,
                raw[:, :7].copy(),
                update,
                raw[:, :7].copy(),
                snapshot.status.value,
                reason,
                None,
                risk,
                snapshot.response_fit.available,
                snapshot.response_fit,
                snapshot.risk_features,
                False,
            )
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
            snapshot.current_raw_chunk,
            stride=context.stride,
            candidate_specs=list(config.candidate_specs),
            completed_actions=snapshot.completed_action_history,
            response=snapshot.response_fit,
            observed_displacements=(
                snapshot.completed_tcp_displacements[-2:] if len(snapshot.completed_tcp_displacements) >= 2 else None
            ),
            previous_action=snapshot.previous_action,
            risk_eligible=True if risk is None else risk.eligible,
            ablate_risk_trust_eligibility=ablated,
            reversal_context={
                "executed_action_history": snapshot.completed_action_history,
                "completed_tcp_history": snapshot.completed_tcp_history,
                "direction_window": config.reversal.direction_window,
                "reversal_cos_threshold": config.reversal.reversal_cos_threshold,
                "min_consistent_steps": config.reversal.min_consistent_steps,
                "velocity_epsilon": config.reversal.velocity_epsilon,
            },
        )
        selected = raw.copy()
        selected[:, :7] = decision.chunk
        return RapRuntimePreview(
            selected,
            snapshot.current_raw_chunk.copy(),
            update,
            decision.chunk.copy(),
            snapshot.status.value,
            decision.reason.value,
            decision,
            risk,
            snapshot.response_fit.available,
            snapshot.response_fit,
            snapshot.risk_features,
            not risk_gated,
        )

    def commit(self, update: RapCausalUpdate | None, selected_chunk: np.ndarray) -> None:
        if update is not None:
            self._causal.commit(update, selected_chunk)


def load_rap_runtime_config(path: str) -> RapRuntimeConfig:
    data = json.loads(Path(path).read_text())

    def require(value, keys, label):
        if not isinstance(value, dict) or set(value) != set(keys):
            raise ValueError(f"invalid RAP runtime schema: {label}")
        return value

    if not isinstance(data, dict):
        raise ValueError("invalid RAP runtime schema: top-level")
    version = data.get("schema_version")
    method_id = MethodId.RAP
    method_identity = "RAP-COVER-EXECUTION-PROJECTION-CF-VETO"
    objective_count_mode = _rap_cover.OBJECTIVE_LEGACY_NEXT
    response_mode = "PRIMARY"
    selection_mode = _rap_cover.SELECTION_ACTION
    reversal_space = _rap_cover.REVERSAL_LEGACY
    uncertainty_scale = None
    minimum_prequential_predictions = 0
    response_half_life_completed_pairs = 0
    candidate_evaluation_policy = _rap_cover.CANDIDATE_EVALUATION_FULL
    preprojection_objective_bound = _rap_cover.PREPROJECTION_BOUND_DISABLED
    projector_context_cache_mode = _rap_cover.PROJECTOR_CONTEXT_CACHE_DISABLED
    terminal_constraint_mode = _rap_cover.TERMINAL_CONSTRAINT_PREFIX_EXIT
    terminal_projection_role = _rap_cover.TERMINAL_PROJECTION_REF
    if version == "1.0":
        data = require(
            data,
            {"schema_version", "core", "candidate_specs", "response", "reversal", "risk"},
            "top-level",
        )
        tcp_translation_units = data["risk"].get("tcp_translation_units") if isinstance(data["risk"], dict) else None
    elif version == "2.0":
        data = require(
            data,
            {
                "schema_version",
                "method_identity",
                "tcp_translation_units",
                "core",
                "candidate_specs",
                "response",
                "reversal",
            },
            "top-level",
        )
        if data["method_identity"] != "RAP-COVER-EXECUTION-PROJECTION-CF-VETO":
            raise ValueError("invalid RAP runtime schema: method_identity")
        tcp_translation_units = data["tcp_translation_units"]
    elif version in {"3.0", "7.0", "8.0", "9.0", "10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0", "17.0", "18.0", "19.0", "20.0", "21.0", "22.0", "23.0", "24.0"}:
        data = require(
            data,
            {
                "schema_version",
                "method_id",
                "method_identity",
                "tcp_translation_units",
                "variant",
                "core",
                "candidate_specs",
                "response",
                "reversal",
            },
            "top-level",
        )
        try:
            method_id = MethodId(data["method_id"])
        except ValueError as error:
            raise ValueError("invalid COVER screen method_id") from error
        allowed_method_ids = (
            TASK105_COVER_SCREEN_METHOD_IDS
            if version == "3.0"
            else {MethodId.COVER_TCP_JERK_HORIZON}
            if version == "7.0"
            else {MethodId.COVER_EW_RESPONSE}
            if version == "8.0"
            else {MethodId.COVER_GRID10, MethodId.COVER_GRID5}
            if version == "9.0"
            else {
                MethodId.COVER_GRID10_ZERO_FALLBACK,
                MethodId.COVER_GRID10_TIERED_FALLBACK,
            }
            if version == "10.0"
            else {MethodId.COVER_TIERED_PREPROJECTION_BOUND}
            if version == "11.0"
            else {MethodId.COVER_PROJECTOR_CONTEXT_CACHE}
            if version == "12.0"
            else {MethodId.COVER_K5_ALPHA5}
            if version == "17.0"
            else {MethodId.COVER_K5_ALPHA5_NO_RELATIVE_CAP}
            if version == "18.0"
            else {
                MethodId.COVER_K5_ALPHA_COARSE_SCAN,
                MethodId.COVER_K5_ALPHA_COARSE_A6_FREE,
            }
            if version == "19.0"
            else {MethodId.COVER_K5_ALPHA_FINE_A6_FREE}
            if version == "20.0"
            else {MethodId.COVER_K5_ALPHA_COARSE_A6_FREE_NO_CORRECTION_CAP}
            if version == "21.0"
            else {MethodId.COVER_K5_ALPHA_FINE5_A6_FREE_NO_CORRECTION_CAP}
            if version == "22.0"
            else {
                MethodId.COVER_TERMINAL_REF_CAP003,
                MethodId.COVER_TERMINAL_VALIDATOR_ONLY_CAP003,
            }
            if version == "23.0"
            else {MethodId.COVER_ALPHA_BANK_065_095_VALIDATOR_ONLY_CAP003}
            if version == "24.0"
            else {MethodId.COVER_PROJECTOR_TERMINAL_ENDPOINT_REUSE}
        )
        if method_id not in allowed_method_ids:
            raise ValueError(f"schema-{version} method_id is outside its frozen stage contract")
        method_identity = data["method_identity"]
        tcp_translation_units = data["tcp_translation_units"]
        variant_keys = {
            "objective_count_mode",
            "response_mode",
            "selection_mode",
            "reversal_space",
            "uncertainty_scale",
            "minimum_prequential_predictions",
        }
        if version == "8.0":
            variant_keys.add("response_half_life_completed_pairs")
        if version in {"10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0", "17.0", "18.0", "19.0", "20.0", "21.0", "22.0", "23.0", "24.0"}:
            variant_keys.add("candidate_evaluation_policy")
        if version in {"11.0", "12.0", "13.0", "14.0", "15.0", "16.0", "17.0", "18.0", "19.0", "20.0", "21.0", "22.0", "23.0", "24.0"}:
            variant_keys.add("preprojection_objective_bound")
        if version in {"12.0", "13.0", "14.0", "15.0", "16.0", "17.0", "18.0", "19.0", "20.0", "21.0", "22.0", "23.0", "24.0"}:
            variant_keys.add("projector_context_cache_mode")
        if (
            version in {"20.0", "21.0", "22.0", "23.0", "24.0"}
            or version == "19.0"
            and method_id is MethodId.COVER_K5_ALPHA_COARSE_A6_FREE
        ):
            variant_keys.add("terminal_constraint_mode")
        if version in {"23.0", "24.0"}:
            variant_keys.add("terminal_projection_role")
        variant = require(data["variant"], variant_keys, "variant")
        objective_count_mode = variant["objective_count_mode"]
        response_mode = variant["response_mode"]
        selection_mode = variant["selection_mode"]
        reversal_space = variant["reversal_space"]
        uncertainty_scale = variant["uncertainty_scale"]
        minimum_prequential_predictions = variant["minimum_prequential_predictions"]
        if version == "8.0":
            response_half_life_completed_pairs = variant[
                "response_half_life_completed_pairs"
            ]
        if version in {"10.0", "11.0", "12.0", "13.0", "14.0", "15.0", "16.0", "17.0", "18.0", "19.0", "20.0", "21.0", "22.0", "23.0", "24.0"}:
            candidate_evaluation_policy = variant["candidate_evaluation_policy"]
        if version in {"11.0", "12.0", "13.0", "14.0", "15.0", "16.0", "17.0", "18.0", "19.0", "20.0", "21.0", "22.0", "23.0", "24.0"}:
            preprojection_objective_bound = variant[
                "preprojection_objective_bound"
            ]
        if version in {"12.0", "13.0", "14.0", "15.0", "16.0", "17.0", "18.0", "19.0", "20.0", "21.0", "22.0", "23.0", "24.0"}:
            projector_context_cache_mode = variant[
                "projector_context_cache_mode"
            ]
        if (
            version in {"20.0", "21.0", "22.0", "23.0", "24.0"}
            or version == "19.0"
            and method_id is MethodId.COVER_K5_ALPHA_COARSE_A6_FREE
        ):
            terminal_constraint_mode = variant["terminal_constraint_mode"]
        if version in {"23.0", "24.0"}:
            terminal_projection_role = variant["terminal_projection_role"]
    else:
        raise ValueError("invalid RAP runtime schema: version")
    core = require(data["core"], {"k_max", "correction_cap", "objective_weights"}, "core")
    if not isinstance(core["objective_weights"], list):
        raise ValueError("invalid RAP runtime schema: objective_weights")
    response = require(data["response"], {"min_samples", "ridge", "condition_limit"}, "response")
    reversal = require(
        data["reversal"],
        {"direction_window", "reversal_cos_threshold", "min_consistent_steps", "velocity_epsilon"},
        "reversal",
    )
    if not isinstance(data["candidate_specs"], list):
        raise ValueError("invalid RAP runtime schema: candidate_specs")
    candidates = []
    for item in data["candidate_specs"]:
        item = require(item, {"k", "alpha", "transition_weights", "relative_cap"}, "candidate")
        if not isinstance(item["transition_weights"], list):
            raise ValueError("invalid RAP runtime schema: transition_weights")
        candidates.append(
            RapCandidateSpec(
                k=item["k"],
                alpha=item["alpha"],
                transition_weights=tuple(item["transition_weights"]),
                relative_cap=item["relative_cap"],
            )
        )
    risk_config = None
    if version == "1.0":
        risk = require(data["risk"], {"tcp_translation_units", "next_weight", "operating_points"}, "risk")
        operating = require(risk["operating_points"], {"3", "5", "9"}, "operating_points")
        points = tuple(
            (
                int(stride),
                RapRiskOperatingPoint(
                    **require(operating[stride], {"boundary_scale", "next_scale", "cutoff"}, "operating_point")
                ),
            )
            for stride in ("3", "5", "9")
        )
        risk_config = RapRiskConfig(risk["tcp_translation_units"], risk["next_weight"], points)
    result = RapRuntimeConfig(
        RapConfig(
            k_max=core["k_max"],
            correction_cap=core["correction_cap"],
            objective_weights=tuple(core["objective_weights"]),
        ),
        tuple(candidates),
        RapResponseRuntimeConfig(**response),
        RapReversalRuntimeConfig(**reversal),
        risk=risk_config,
        tcp_translation_units=tcp_translation_units,
        method_id=method_id,
        method_identity=method_identity,
        objective_count_mode=objective_count_mode,
        response_mode=response_mode,
        selection_mode=selection_mode,
        reversal_space=reversal_space,
        uncertainty_scale=uncertainty_scale,
        minimum_prequential_predictions=minimum_prequential_predictions,
        response_half_life_completed_pairs=response_half_life_completed_pairs,
        candidate_evaluation_policy=candidate_evaluation_policy,
        preprojection_objective_bound=preprojection_objective_bound,
        projector_context_cache_mode=projector_context_cache_mode,
        terminal_constraint_mode=terminal_constraint_mode,
        terminal_projection_role=terminal_projection_role,
    )
    if version == "2.0":
        expected_tapers = {
            1: (1.0,),
            2: (1.0, 0.25),
            3: (1.0, 0.625, 0.25),
            4: (1.0, 0.75, 0.5, 0.25),
            5: (1.0, 0.8125, 0.625, 0.4375, 0.25),
        }
        expected_specs = tuple(
            sorted(
                (
                    k,
                    alpha,
                    expected_tapers[k],
                    0.2,
                )
                for k in range(1, 6)
                for alpha in (0.01, 0.02, 0.05, 0.1, 0.2)
            )
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap) for item in result.candidate_specs
        )
        if (
            result.core.k_max != 5
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen M4 RAP-COVER primary config")
    if version == "3.0":
        contracts = {
            MethodId.COVER_M4_REF: (
                "RAP-COVER-EXECUTION-PROJECTION-CF-VETO",
                5,
                _rap_cover.OBJECTIVE_LEGACY_NEXT,
                "PRIMARY",
                _rap_cover.SELECTION_ACTION,
                _rap_cover.REVERSAL_LEGACY,
                None,
                0,
            ),
            MethodId.COVER_EXEC_K4: (
                "RAP-COVER-V2-EXECUTED-ONLY-K4",
                4,
                _rap_cover.OBJECTIVE_EXECUTED_ONLY,
                "PRIMARY",
                _rap_cover.SELECTION_ACTION,
                _rap_cover.REVERSAL_LEGACY,
                None,
                0,
            ),
            MethodId.COVER_UNCERT_SNR1: (
                "RAP-COVER-V2-EXECUTED-K4-PREQUENTIAL-SNR1",
                4,
                _rap_cover.OBJECTIVE_EXECUTED_ONLY,
                "PREQUENTIAL",
                _rap_cover.SELECTION_ACTION,
                _rap_cover.REVERSAL_LEGACY,
                1.0,
                8,
            ),
            MethodId.COVER_PHYS_SCORE: (
                "RAP-COVER-V2-EXECUTED-K4-PHYSICAL-SCORE",
                4,
                _rap_cover.OBJECTIVE_EXECUTED_ONLY,
                "PRIMARY",
                _rap_cover.SELECTION_PHYSICAL,
                _rap_cover.REVERSAL_LEGACY,
                None,
                0,
            ),
            MethodId.COVER_REV_PHYS: (
                "RAP-COVER-V2-EXECUTED-K4-RESPONSE-SPACE-REVERSAL",
                4,
                _rap_cover.OBJECTIVE_EXECUTED_ONLY,
                "PRIMARY",
                _rap_cover.SELECTION_ACTION,
                _rap_cover.REVERSAL_RESPONSE_TCP,
                None,
                0,
            ),
        }
        expected_tapers = {
            1: (1.0,),
            2: (1.0, 0.25),
            3: (1.0, 0.625, 0.25),
            4: (1.0, 0.75, 0.5, 0.25),
            5: (1.0, 0.8125, 0.625, 0.4375, 0.25),
        }
        expected_contract = contracts[method_id]
        expected_k = expected_contract[1]
        expected_specs = tuple(
            sorted(
                (k, alpha, expected_tapers[k], 0.2)
                for k in range(1, expected_k + 1)
                for alpha in (0.01, 0.02, 0.05, 0.1, 0.2)
            )
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen Task105 COVER variant config")
    if version == "7.0":
        expected_tapers = {
            1: (1.0,),
            2: (1.0, 0.25),
            3: (1.0, 0.625, 0.25),
            4: (1.0, 0.75, 0.5, 0.25),
            5: (1.0, 0.8125, 0.625, 0.4375, 0.25),
        }
        expected_specs = tuple(
            sorted(
                (k, alpha, expected_tapers[k], 0.2)
                for k in range(1, 6)
                for alpha in (0.01, 0.02, 0.05, 0.1, 0.2)
            )
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_id,
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
        )
        expected_contract = (
            MethodId.COVER_TCP_JERK_HORIZON,
            "RAP-COVER-TASK106-S4-PREDICTED-TCP-HORIZON-ISJ-RANKING",
            5,
            _rap_cover.OBJECTIVE_LEGACY_NEXT,
            "PRIMARY",
            _rap_cover.SELECTION_TCP_HORIZON_ISJ,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen Task106 Stage4 TCP horizon jerk config")
    if version == "8.0":
        expected_tapers = {
            1: (1.0,),
            2: (1.0, 0.25),
            3: (1.0, 0.625, 0.25),
            4: (1.0, 0.75, 0.5, 0.25),
            5: (1.0, 0.8125, 0.625, 0.4375, 0.25),
        }
        expected_specs = tuple(
            sorted(
                (k, alpha, expected_tapers[k], 0.2)
                for k in range(1, 6)
                for alpha in (0.01, 0.02, 0.05, 0.1, 0.2)
            )
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_id,
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
        )
        expected_contract = (
            MethodId.COVER_EW_RESPONSE,
            "RAP-COVER-TASK108-EW-RESPONSE-HALFLIFE2",
            5,
            _rap_cover.OBJECTIVE_LEGACY_NEXT,
            "EXPONENTIALLY_WEIGHTED",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            EW_HALF_LIFE_COMPLETED_PAIRS,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen Task108 Wave2 EW response config")
    if version == "9.0":
        expected_tapers = {
            1: (1.0,),
            2: (1.0, 0.25),
            3: (1.0, 0.625, 0.25),
            4: (1.0, 0.75, 0.5, 0.25),
            5: (1.0, 0.8125, 0.625, 0.4375, 0.25),
        }
        allowed_alphas = {
            MethodId.COVER_GRID10: (0.1, 0.2),
            MethodId.COVER_GRID5: (0.2,),
        }[method_id]
        expected_specs = tuple(
            sorted(
                (k, alpha, expected_tapers[k], 0.2)
                for k in range(1, 6)
                for alpha in allowed_alphas
            )
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        expected_identity = {
            MethodId.COVER_GRID10: "RAP-COVER-TASK108-GRID10-K1TO5-ALPHA01-02",
            MethodId.COVER_GRID5: "RAP-COVER-TASK108-GRID5-K1TO5-ALPHA02",
        }[method_id]
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
        )
        expected_contract = (
            expected_identity,
            5,
            _rap_cover.OBJECTIVE_LEGACY_NEXT,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen Task108 Wave3 candidate-grid config")
    if version == "10.0":
        expected_tapers = {
            1: (1.0,),
            2: (1.0, 0.25),
            3: (1.0, 0.625, 0.25),
            4: (1.0, 0.75, 0.5, 0.25),
            5: (1.0, 0.8125, 0.625, 0.4375, 0.25),
        }
        expected_specs = tuple(
            sorted(
                (k, alpha, expected_tapers[k], 0.2)
                for k in range(1, 6)
                for alpha in (0.01, 0.02, 0.05, 0.1, 0.2)
            )
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        contracts = {
            MethodId.COVER_GRID10_ZERO_FALLBACK: (
                "RAP-COVER-TASK108-GRID10-ZERO-ELIGIBLE-FULL-FALLBACK",
                _rap_cover.CANDIDATE_EVALUATION_ZERO_FALLBACK,
            ),
            MethodId.COVER_GRID10_TIERED_FALLBACK: (
                "RAP-COVER-TASK108-GRID10-AT-MOST-ONE-TIERED-FALLBACK",
                _rap_cover.CANDIDATE_EVALUATION_TIERED_FALLBACK,
            ),
        }
        expected_identity, expected_policy = contracts[method_id]
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
        )
        expected_contract = (
            expected_identity,
            5,
            _rap_cover.OBJECTIVE_LEGACY_NEXT,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
            expected_policy,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen Task108 Wave4 adaptive fallback config")
    if version == "11.0":
        expected_tapers = {
            1: (1.0,),
            2: (1.0, 0.25),
            3: (1.0, 0.625, 0.25),
            4: (1.0, 0.75, 0.5, 0.25),
            5: (1.0, 0.8125, 0.625, 0.4375, 0.25),
        }
        expected_specs = tuple(
            sorted(
                (k, alpha, expected_tapers[k], 0.2)
                for k in range(1, 6)
                for alpha in (0.01, 0.02, 0.05, 0.1, 0.2)
            )
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
        )
        expected_contract = (
            "RAP-COVER-TASK108-TIERED-PREPROJECTION-MIN-GAIN-BOUND",
            5,
            _rap_cover.OBJECTIVE_LEGACY_NEXT,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
            _rap_cover.CANDIDATE_EVALUATION_TIERED_FALLBACK,
            _rap_cover.PREPROJECTION_BOUND_BRIDGE_MIN_GAIN,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen Task108 Wave6 preprojection-bound config")
    if version == "12.0":
        expected_tapers = {
            1: (1.0,),
            2: (1.0, 0.25),
            3: (1.0, 0.625, 0.25),
            4: (1.0, 0.75, 0.5, 0.25),
            5: (1.0, 0.8125, 0.625, 0.4375, 0.25),
        }
        expected_specs = tuple(
            sorted(
                (k, alpha, expected_tapers[k], 0.2)
                for k in range(1, 6)
                for alpha in (0.01, 0.02, 0.05, 0.1, 0.2)
            )
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
            result.projector_context_cache_mode,
        )
        expected_contract = (
            "RAP-COVER-TASK108-TIERED-PREPROJECTION-PROJECTOR-CONTEXT-CACHE",
            5,
            _rap_cover.OBJECTIVE_LEGACY_NEXT,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
            _rap_cover.CANDIDATE_EVALUATION_TIERED_FALLBACK,
            _rap_cover.PREPROJECTION_BOUND_BRIDGE_MIN_GAIN,
            _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen Task108 Wave7 projector-context-cache config")
    if version == "13.0":
        expected_tapers = {
            1: (1.0,),
            2: (1.0, 0.25),
            3: (1.0, 0.625, 0.25),
            4: (1.0, 0.75, 0.5, 0.25),
            5: (1.0, 0.8125, 0.625, 0.4375, 0.25),
        }
        expected_specs = tuple(
            sorted(
                (k, alpha, expected_tapers[k], 0.2)
                for k in range(1, 6)
                for alpha in (0.01, 0.02, 0.05, 0.1, 0.2)
            )
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
            result.projector_context_cache_mode,
        )
        expected_contract = (
            "RAP-COVER-TASK108-TIERED-PREPROJECTION-PROJECTOR-CONTEXT-CACHE-TERMINAL-ENDPOINT-REUSE",
            5,
            _rap_cover.OBJECTIVE_LEGACY_NEXT,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
            _rap_cover.CANDIDATE_EVALUATION_TIERED_FALLBACK,
            _rap_cover.PREPROJECTION_BOUND_BRIDGE_MIN_GAIN,
            _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen Task108 Wave9 terminal-endpoint-reuse config")
    if version == "14.0":
        expected_tapers = {
            1: (1.0,),
            2: (1.0, 0.25),
            3: (1.0, 0.625, 0.25),
            4: (1.0, 0.75, 0.5, 0.25),
            5: (1.0, 0.8125, 0.625, 0.4375, 0.25),
        }
        expected_specs = tuple(
            sorted(
                (k, alpha, expected_tapers[k], 0.2)
                for k in range(1, 6)
                for alpha in (0.01, 0.02, 0.05, 0.1, 0.2)
            )
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
            result.projector_context_cache_mode,
        )
        expected_contract = (
            "RAP-COVER-FAST-SCREEN-Q-DYKSTRA-REPRESENTATION-AWARE",
            5,
            _rap_cover.OBJECTIVE_LEGACY_NEXT,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
            _rap_cover.CANDIDATE_EVALUATION_TIERED_FALLBACK,
            _rap_cover.PREPROJECTION_BOUND_BRIDGE_MIN_GAIN,
            _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_DYKSTRA,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen fast-screen Q qualification config")
    if version == "15.0":
        expected_tapers = {
            1: (1.0,),
            2: (1.0, 0.25),
            3: (1.0, 0.625, 0.25),
            4: (1.0, 0.75, 0.5, 0.25),
            5: (1.0, 0.8125, 0.625, 0.4375, 0.25),
        }
        expected_specs = tuple(
            sorted(
                (k, alpha, expected_tapers[k], 0.2)
                for k in range(1, 6)
                for alpha in (0.01, 0.02, 0.05, 0.1, 0.2)
            )
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
            result.projector_context_cache_mode,
        )
        expected_contract = (
            "RAP-COVER-MODULE-LAB-Q2-CYCLIC-REPRESENTATION-REPAIR",
            5,
            _rap_cover.OBJECTIVE_LEGACY_NEXT,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
            _rap_cover.CANDIDATE_EVALUATION_TIERED_FALLBACK,
            _rap_cover.PREPROJECTION_BOUND_BRIDGE_MIN_GAIN,
            _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_REPRESENTATION_AWARE,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen Module-Lab Q2 repair-only config")
    if version == "16.0":
        expected_tapers = {
            1: (1.0,),
            2: (1.0, 0.25),
            3: (1.0, 0.625, 0.25),
            4: (1.0, 0.75, 0.5, 0.25),
            5: (1.0, 0.8125, 0.625, 0.4375, 0.25),
        }
        expected_specs = tuple(
            sorted(
                (k, alpha, expected_tapers[k], 0.2)
                for k in range(1, 6)
                for alpha in (0.01, 0.02, 0.05, 0.1, 0.2)
            )
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
            result.projector_context_cache_mode,
        )
        allowed_contracts = {
            (
                "RAP-COVER-EXECUTION-PROJECTION-CF-VETO",
                5,
                _rap_cover.OBJECTIVE_LEGACY_NEXT,
                "PRIMARY",
                _rap_cover.SELECTION_ACTION,
                _rap_cover.REVERSAL_LEGACY,
                None,
                0,
                0,
                _rap_cover.CANDIDATE_EVALUATION_FULL,
                _rap_cover.PREPROJECTION_BOUND_DISABLED,
                _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE,
            ),
            (
                "RAP-COVER-Q-CANDIDATE-CONVERGED-REPRESENTATION-REPAIR",
                5,
                _rap_cover.OBJECTIVE_LEGACY_NEXT,
                "PRIMARY",
                _rap_cover.SELECTION_ACTION,
                _rap_cover.REVERSAL_LEGACY,
                None,
                0,
                0,
                _rap_cover.CANDIDATE_EVALUATION_FULL,
                _rap_cover.PREPROJECTION_BOUND_DISABLED,
                _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_CONVERGED_REPRESENTATION_REPAIR,
            ),
        }
        if (
            actual_contract not in allowed_contracts
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen current-M4C Q development config")
    if version == "17.0":
        expected_specs = tuple(
            (5, alpha, (1.0, 0.8125, 0.625, 0.4375, 0.25), 0.2)
            for alpha in (0.2, 0.25, 0.3, 0.35, 0.4)
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
            result.projector_context_cache_mode,
        )
        expected_contract = (
            "RAP-COVER-K5-ALPHA5-0.20-0.40",
            5,
            _rap_cover.OBJECTIVE_LEGACY_NEXT,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
            _rap_cover.CANDIDATE_EVALUATION_FULL,
            _rap_cover.PREPROJECTION_BOUND_DISABLED,
            _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen K5-alpha5 development config")
    if version == "18.0":
        expected_specs = tuple(
            (5, alpha, (1.0, 0.8125, 0.625, 0.4375, 0.25), 1.0)
            for alpha in (0.2, 0.25, 0.3, 0.35, 0.4)
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
            result.projector_context_cache_mode,
        )
        expected_contract = (
            "RAP-COVER-K5-ALPHA5-NO-RELATIVE-CAP",
            5,
            _rap_cover.OBJECTIVE_LEGACY_NEXT,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
            _rap_cover.CANDIDATE_EVALUATION_FULL,
            _rap_cover.PREPROJECTION_BOUND_DISABLED,
            _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen K5-alpha5 no-relative-cap development config")
    if version == "19.0":
        expected_specs = tuple(
            (5, alpha, (1.0, 0.8125, 0.625, 0.4375, 0.25), 1.0)
            for alpha in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
            result.projector_context_cache_mode,
            result.terminal_constraint_mode,
        )
        expected_contract = (
            (
                "RAP-COVER-K5-ALPHA-COARSE-0.20-0.90",
                5,
                _rap_cover.OBJECTIVE_LEGACY_NEXT,
                "PRIMARY",
                _rap_cover.SELECTION_ACTION,
                _rap_cover.REVERSAL_LEGACY,
                None,
                0,
                0,
                _rap_cover.CANDIDATE_EVALUATION_FULL,
                _rap_cover.PREPROJECTION_BOUND_DISABLED,
                _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE,
                _rap_cover.TERMINAL_CONSTRAINT_PREFIX_EXIT,
            )
            if result.method_id is MethodId.COVER_K5_ALPHA_COARSE_SCAN
            else (
                "RAP-COVER-K5-ALPHA-COARSE-A6-FREE-0.20-0.90",
                5,
                _rap_cover.OBJECTIVE_EXECUTED_ONLY,
                "PRIMARY",
                _rap_cover.SELECTION_ACTION,
                _rap_cover.REVERSAL_LEGACY,
                None,
                0,
                0,
                _rap_cover.CANDIDATE_EVALUATION_FULL,
                _rap_cover.PREPROJECTION_BOUND_DISABLED,
                _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE,
                _rap_cover.TERMINAL_CONSTRAINT_EXECUTED_INTERNAL,
            )
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen K5 coarse-alpha development config")
    if version == "20.0":
        expected_specs = tuple(
            (5, alpha, (1.0, 0.8125, 0.625, 0.4375, 0.25), 1.0)
            for alpha in (0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7)
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
            result.projector_context_cache_mode,
            result.terminal_constraint_mode,
        )
        expected_contract = (
            "RAP-COVER-K5-ALPHA-FINE-A6-FREE-0.500-0.700",
            5,
            _rap_cover.OBJECTIVE_EXECUTED_ONLY,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
            _rap_cover.CANDIDATE_EVALUATION_FULL,
            _rap_cover.PREPROJECTION_BOUND_DISABLED,
            _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE,
            _rap_cover.TERMINAL_CONSTRAINT_EXECUTED_INTERNAL,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen K5 fine-alpha a6-free development config")
    if version == "21.0":
        expected_specs = tuple(
            (5, alpha, (1.0, 0.8125, 0.625, 0.4375, 0.25), 1.0)
            for alpha in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
            result.projector_context_cache_mode,
            result.terminal_constraint_mode,
        )
        expected_contract = (
            "RAP-COVER-K5-ALPHA-COARSE-A6-FREE-NO-CORRECTION-CAP-0.20-0.90",
            5,
            _rap_cover.OBJECTIVE_EXECUTED_ONLY,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
            _rap_cover.CANDIDATE_EVALUATION_FULL,
            _rap_cover.PREPROJECTION_BOUND_DISABLED,
            _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE,
            _rap_cover.TERMINAL_CONSTRAINT_EXECUTED_INTERNAL,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap is not None
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError(
                "invalid frozen K5 coarse-alpha a6-free no-correction-cap development config"
            )
    if version == "22.0":
        expected_specs = tuple(
            (5, alpha, (1.0, 0.8125, 0.625, 0.4375, 0.25), 1.0)
            for alpha in (0.7, 0.8, 0.9, 0.95, 1.0)
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
            result.projector_context_cache_mode,
            result.terminal_constraint_mode,
        )
        expected_contract = (
            "RAP-COVER-K5-ALPHA-FINE5-A6-FREE-NO-CORRECTION-CAP-0.70-1.00",
            5,
            _rap_cover.OBJECTIVE_EXECUTED_ONLY,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
            _rap_cover.CANDIDATE_EVALUATION_FULL,
            _rap_cover.PREPROJECTION_BOUND_DISABLED,
            _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE,
            _rap_cover.TERMINAL_CONSTRAINT_EXECUTED_INTERNAL,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap is not None
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError(
                "invalid frozen K5 fine5-alpha a6-free no-correction-cap development config"
            )
    if version == "23.0":
        expected_identity_and_role = {
            MethodId.COVER_TERMINAL_REF_CAP003: (
                "RAP-COVER-TERMINAL-REF-CAP003",
                _rap_cover.TERMINAL_PROJECTION_REF,
            ),
            MethodId.COVER_TERMINAL_VALIDATOR_ONLY_CAP003: (
                "RAP-COVER-TERMINAL-VALIDATOR-ONLY-CAP003",
                _rap_cover.TERMINAL_PROJECTION_VALIDATOR_ONLY,
            ),
        }
        expected_specs = tuple(
            (5, alpha, (1.0, 0.8125, 0.625, 0.4375, 0.25), 1.0)
            for alpha in (0.7, 0.8, 0.9, 0.95, 1.0)
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        expected_identity, expected_role = expected_identity_and_role[method_id]
        actual_contract = (
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
            result.projector_context_cache_mode,
            result.terminal_constraint_mode,
            result.terminal_projection_role,
        )
        expected_contract = (
            expected_identity,
            5,
            _rap_cover.OBJECTIVE_EXECUTED_ONLY,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
            _rap_cover.CANDIDATE_EVALUATION_FULL,
            _rap_cover.PREPROJECTION_BOUND_DISABLED,
            _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE,
            _rap_cover.TERMINAL_CONSTRAINT_EXECUTED_INTERNAL,
            expected_role,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.response != RapResponseRuntimeConfig(8, 0.0001, 100000000.0)
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen terminal REF/VALIDATOR_ONLY cap003 config")
    if version == "24.0":
        expected_specs = tuple(
            (5, alpha, (1.0, 0.8125, 0.625, 0.4375, 0.25), 1.0)
            for alpha in (0.65, 0.7, 0.8, 0.9, 0.95)
        )
        actual_specs = tuple(
            (item.k, item.alpha, item.transition_weights, item.relative_cap)
            for item in result.candidate_specs
        )
        actual_contract = (
            result.method_id,
            result.method_identity,
            result.core.k_max,
            result.objective_count_mode,
            result.response_mode,
            result.selection_mode,
            result.reversal_space,
            result.uncertainty_scale,
            result.minimum_prequential_predictions,
            result.response_half_life_completed_pairs,
            result.candidate_evaluation_policy,
            result.preprojection_objective_bound,
            result.projector_context_cache_mode,
            result.terminal_constraint_mode,
            result.terminal_projection_role,
        )
        expected_contract = (
            MethodId.COVER_ALPHA_BANK_065_095_VALIDATOR_ONLY_CAP003,
            "RAP-COVER-ALPHA-BANK-0.65-0.95-VALIDATOR-ONLY-CAP003",
            5,
            _rap_cover.OBJECTIVE_EXECUTED_ONLY,
            "PRIMARY",
            _rap_cover.SELECTION_ACTION,
            _rap_cover.REVERSAL_LEGACY,
            None,
            0,
            0,
            _rap_cover.CANDIDATE_EVALUATION_FULL,
            _rap_cover.PREPROJECTION_BOUND_DISABLED,
            _rap_cover.PROJECTOR_CONTEXT_CACHE_QUERY_PREFIX_LAZY_READONLY_TERMINAL_ENDPOINT_REUSE,
            _rap_cover.TERMINAL_CONSTRAINT_EXECUTED_INTERNAL,
            _rap_cover.TERMINAL_PROJECTION_VALIDATOR_ONLY,
        )
        if (
            actual_contract != expected_contract
            or result.core.correction_cap != 0.03
            or result.core.objective_weights != (1.0, 0.5, 0.25)
            or actual_specs != expected_specs
            or result.response != RapResponseRuntimeConfig(8, 0.0001, 100000000.0)
            or result.reversal != RapReversalRuntimeConfig(3, -0.5, 2, 1e-6)
            or result.risk is not None
        ):
            raise ValueError("invalid frozen alpha-bank successor cap003 config")
    return result


def _finite_or_none(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return value if np.isfinite(value) else None
    return value


def _rows_for_receipt(value) -> list[list[float | None]]:
    return [[_finite_or_none(item) for item in row] for row in value]


def serialize_rap_runtime_preview(
    context: M1Context,
    preview: RapRuntimePreview,
    config: RapRuntimeConfig | None = None,
) -> dict:
    """Serialize already-computed RAP telemetry without affecting selection state."""
    response = preview.response
    response_receipt = None
    if response is not None:
        numeric = {
            "condition_number": _finite_or_none(response.condition_number),
            "residual_rmse": _finite_or_none(response.residual_rmse),
        }
        if config is not None and config.response_mode == "PREQUENTIAL":
            numeric["prequential_rmse"] = _finite_or_none(response.prequential_rmse)
        response_receipt = {
            "available": response.available,
            "reason": response.reason,
            "sample_count": response.sample_count,
            "window_start": response.window_start,
            "window_end": response.window_end,
            "rank": response.rank,
            "fit_age": response.fit_age,
            "prequential_count": response.prequential_count,
            **numeric,
            "explicit_missing_numeric_fields": sorted(key for key, value in numeric.items() if value is None),
        }
    features = []
    for name, feature in sorted((preview.risk_features or {}).items()):
        features.append(
            {
                "name": name,
                "value": _finite_or_none(feature.value),
                "shape": list(feature.shape),
                "units": feature.units,
                "logical_timestamp": feature.logical_timestamp,
                "latest_availability": feature.latest_availability,
                "missing_behavior": feature.missing_behavior,
                "value_missing": feature.value is None or _finite_or_none(feature.value) is None,
            }
        )
    core = None
    if preview.decision is not None and preview.decision.telemetry is not None:
        telemetry = preview.decision.telemetry
        traces = []
        scalar_names = (
            "raw_predicted_tcp_horizon_integrated_squared_jerk",
            "candidate_predicted_tcp_horizon_integrated_squared_jerk",
            "paired_predicted_tcp_horizon_jerk_ratio",
            "bridge_objective",
            "projected_objective",
            "relative_gain",
            "selection_score",
            "correction_l2",
            "correction_max_abs",
            "projection_max_violation",
            "predicted_boundary_improvement",
            "uncertainty_threshold",
            "physical_relative_objective",
            "correction_fraction_of_cap",
        )
        for trace in telemetry.candidate_traces:
            scalars = {name: _finite_or_none(getattr(trace, name)) for name in scalar_names}
            traces.append(
                {
                    "k": trace.k,
                    "alpha": trace.alpha,
                    "status": trace.status,
                    **scalars,
                    "explicit_missing_numeric_fields": sorted(
                        name for name, value in scalars.items() if value is None
                    ),
                    "relative_cap_hit": trace.relative_cap_hit,
                    "correction_cap_hit": trace.correction_cap_hit,
                    "projection_iterations": trace.projection_iterations,
                    "projection_dykstra_cycles": trace.projection_dykstra_cycles,
                    "projection_cyclic_completion_cycles": trace.projection_cyclic_completion_cycles,
                    "representation_repair_invocations": trace.representation_repair_invocations,
                    "representation_repair_steps": trace.representation_repair_steps,
                    "representation_repair_ulp_neighbor_evaluations": (
                        trace.representation_repair_ulp_neighbor_evaluations
                    ),
                    "lower_correction": _rows_for_receipt(trace.lower_correction),
                    "upper_correction": _rows_for_receipt(trace.upper_correction),
                    "raw_local_components": [_finite_or_none(value) for value in trace.raw_local_components],
                    "bridge_local_components": [_finite_or_none(value) for value in trace.bridge_local_components],
                    "projected_local_components": [
                        _finite_or_none(value) for value in trace.projected_local_components
                    ],
                    "projection_slacks": [
                        {"name": name, "value": _finite_or_none(value)}
                        for name, value in trace.projection_slacks
                    ],
                    "raw_physical": [
                        {"name": name, "value": _finite_or_none(value)} for name, value in trace.raw_physical
                    ],
                    "candidate_physical": [
                        {"name": name, "value": _finite_or_none(value)}
                        for name, value in trace.candidate_physical
                    ],
                    "veto_reason": trace.veto_reason,
                }
            )
        core_numeric = {
            "raw_objective": _finite_or_none(telemetry.raw_objective),
            "response_condition_number": _finite_or_none(telemetry.response_condition_number),
            "response_residual_rmse": _finite_or_none(telemetry.response_residual_rmse),
        }
        core = {
            "method_identity": telemetry.method_id,
            "selected_reason": telemetry.selected_reason,
            **core_numeric,
            "explicit_missing_numeric_fields": sorted(
                name for name, value in core_numeric.items() if value is None
            ),
            "candidate_traces": traces,
            "response_sample_count": telemetry.response_sample_count,
            "response_window_start": telemetry.response_window_start,
            "response_window_end": telemetry.response_window_end,
            "response_rank": telemetry.response_rank,
            "risk_eligible_input": telemetry.risk_eligible,
            "risk_eligibility_ablated": telemetry.risk_eligibility_ablated,
            "reversal_rollback": telemetry.reversal_rollback,
            "selection_mode": telemetry.selection_mode,
        }
        if telemetry.candidate_ranking_mode != _rap_cover.RANKING_ACTION:
            core["candidate_ranking_mode"] = telemetry.candidate_ranking_mode
        if (
            telemetry.projector_context_cache_mode
            != _rap_cover.PROJECTOR_CONTEXT_CACHE_DISABLED
        ):
            core["engineering_cache"] = {
                "mode": telemetry.projector_context_cache_mode,
                "initialized": telemetry.projector_context_cache_initialized,
                "builds_by_k": [
                    {"k": k, "count": count}
                    for k, count in telemetry.projector_context_builds_by_k
                ],
                "hits_by_k": [
                    {"k": k, "count": count}
                    for k, count in telemetry.projector_context_hits_by_k
                ],
                "gram_build_count": telemetry.projector_context_gram_build_count,
                "failure_by_k": [
                    {"k": k, "status": status}
                    for k, status in telemetry.projector_context_failure_by_k
                ],
            }
    risk = None
    if preview.risk is not None:
        risk = {
            "available": preview.risk.available,
            "reason": preview.risk.reason.value,
            "score": _finite_or_none(preview.risk.score),
            "score_units": preview.risk.score_units,
            "eligible": preview.risk.eligible,
            "cutoff": _finite_or_none(preview.risk.cutoff),
            "operator": preview.risk.operator,
            "diagnostic_only": preview.risk_diagnostic_only,
        }
    method_identity = (
        config.method_identity
        if config is not None
        else (
            preview.decision.telemetry.method_id
            if preview.decision is not None and preview.decision.telemetry is not None
            else "RAP-COVER-EXECUTION-PROJECTION-CF-VETO"
        )
    )
    return {
        "schema_version": "2.0",
        "method_identity": method_identity,
        "method_id": context.method_id.value,
        "run_id": context.run_id,
        "episode_id": context.episode_id,
        "logical_timestep": context.logical_timestep,
        "stride": context.stride,
        **(
            {"terminal_constraint_mode": config.terminal_constraint_mode}
            if config is not None
            and config.terminal_constraint_mode
            != _rap_cover.TERMINAL_CONSTRAINT_PREFIX_EXIT
            else {}
        ),
        **(
            {"terminal_projection_role": config.terminal_projection_role}
            if config is not None
            and config.method_id
            in {
                MethodId.COVER_TERMINAL_REF_CAP003,
                MethodId.COVER_TERMINAL_VALIDATOR_ONLY_CAP003,
                MethodId.COVER_ALPHA_BANK_065_095_VALIDATOR_ONLY_CAP003,
            }
            else {}
        ),
        "causal_status": preview.causal_status,
        "reason": preview.reason,
        "k": 0 if preview.decision is None else preview.decision.k,
        "alpha": 0.0 if preview.decision is None else preview.decision.alpha,
        "risk_gate_active": not preview.risk_diagnostic_only,
        "risk": risk,
        "risk_features": features,
        "response": response_receipt,
        "core": core,
        "raw_normalized_actions": preview.raw_chunk.copy(),
        "selected_normalized_actions": preview.selected_chunk.copy(),
        "selected_normalized_physical_actions": preview.selected_chunk.copy(),
    }


def validate_runtime_context(config: M1RuntimeConfig, context: M1Context | None) -> None:
    if context is not None and context.execution_mode is not None and config.method is not RuntimeMethod.VANILLA:
        raise ValueError("variable execution requires active Vanilla runtime")
    if (
        config.method
        in {
            RuntimeMethod.TE,
            RuntimeMethod.TE_MATCHED_CADENCE_CONTROL,
            RuntimeMethod.RTC_D0,
            RuntimeMethod.RAP,
            RuntimeMethod.RAP_NO_RISK_TRUST_ELIGIBILITY,
            RuntimeMethod.RAP_RISK_GATED_ABLATION,
        }
        and context is None
    ):
        raise M1ContextRequiredError(M1ContextRequiredError.code)
    if context is None:
        return
    if config.method is RuntimeMethod.NO_M1_RUNTIME and context.method_id is not MethodId.VANILLA:
        raise M1MethodNotActiveError(M1MethodNotActiveError.code)
    if config.method is RuntimeMethod.VANILLA and context.method_id is not MethodId.VANILLA:
        raise M1MethodNotActiveError(M1MethodNotActiveError.code)
    if config.method is RuntimeMethod.TE and context.method_id is not MethodId.TE:
        raise ValueError("M1 runtime/context method mismatch")
    if (
        config.method is RuntimeMethod.TE_MATCHED_CADENCE_CONTROL
        and context.method_id is not MethodId.TE_MATCHED_CADENCE_CONTROL
    ):
        raise ValueError("M1 runtime/context method mismatch")
    if config.method is RuntimeMethod.RTC_D0 and context.method_id is not MethodId.RTC_D0:
        raise ValueError("M1 runtime/context method mismatch")
    if (
        config.method is RuntimeMethod.RAP
        and (config.rap_runtime is None or context.method_id is not config.rap_runtime.method_id)
    ):
        raise ValueError("M1 runtime/context method mismatch")
    if (
        config.method is RuntimeMethod.RAP_NO_RISK_TRUST_ELIGIBILITY
        and context.method_id is not MethodId.RAP_NO_RISK_TRUST_ELIGIBILITY
    ):
        raise ValueError("M1 runtime/context method mismatch")
    if (
        config.method is RuntimeMethod.RAP_RISK_GATED_ABLATION
        and context.method_id is not MethodId.RAP_RISK_GATED_ABLATION
    ):
        raise ValueError("M1 runtime/context method mismatch")


def strip_context(observation: dict) -> tuple[dict, dict | None]:
    """Copy/strip reserved context, preserving the no-context ordinary Policy path."""
    copied = dict(observation)
    context = copied.pop(CONTEXT_KEY, None)
    return copied, context
