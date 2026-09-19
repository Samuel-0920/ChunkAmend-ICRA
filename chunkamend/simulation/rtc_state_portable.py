from __future__ import annotations
"""Deterministic event-state core for the RTC paper's asynchronous Algorithm 1."""

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .rtc_math_portable import HORIZON
from .rtc_math_portable import MODEL_DIM
from .rtc_math_portable import right_pad_after_stride

PRIMARY_METHOD_ID = "RTC-ASYNC-OPENPI-ALG1"
CONTROLLER_PERIOD_SECONDS = 0.05
MIN_EXECUTION_HORIZON = 5
DELAY_BUFFER_SIZE = 10


class RtcStopReason(str, Enum):
    BUFFER_UNDERRUN = "BUFFER_UNDERRUN"
    INFEASIBLE_DELAY_STRIDE = "INFEASIBLE_DELAY_STRIDE"
    OBSERVED_DELAY_OVERFLOW = "OBSERVED_DELAY_OVERFLOW"


class RtcStoppedError(RuntimeError):
    pass


class RtcStaleResponseError(ValueError):
    pass


@dataclass(frozen=True)
class RtcTick:
    action: np.ndarray
    generation: int
    action_index: int
    buffer_cursor: int
    worker_wake_eligible: bool


@dataclass(frozen=True)
class RtcInferenceRequest:
    request_id: int
    generation: int
    s: int
    d_hat: int
    observation: np.ndarray
    previous_prior: np.ndarray
    remaining_rows: int


@dataclass(frozen=True)
class RtcSwap:
    request_id: int
    old_generation: int
    new_generation: int
    s: int
    d_hat: int
    d_observed: int
    delay_underestimated: bool
    next_action_index: int


class RtcAsyncState:
    """Single-controller/single-worker RTC scheduler without threads or clocks.

    The caller drives controller ticks and inference completion events. This
    makes Algorithm 1 indexing independently testable without claiming a
    real-time runtime. Inference receives a frozen latest observation and the
    unexecuted suffix ``A_cur[s:H]`` right-padded to H rows.
    """

    def __init__(self, initial_chunk: np.ndarray, *, d_init: int, s_min: int = MIN_EXECUTION_HORIZON) -> None:
        value = self._chunk(initial_chunk)
        if (
            isinstance(d_init, bool)
            or not isinstance(d_init, int)
            or not 0 <= d_init <= 5
            or isinstance(s_min, bool)
            or not isinstance(s_min, int)
            or s_min != MIN_EXECUTION_HORIZON
        ):
            raise ValueError("M3 RTC requires d_init in [0,5] and s_min=5")
        self._current = value.copy()
        self._cursor = 0
        self._generation = 0
        self._latest_observation: np.ndarray | None = None
        self._delay_history: deque[int] = deque((d_init,), maxlen=DELAY_BUFFER_SIZE)
        self._inflight: RtcInferenceRequest | None = None
        self._next_request_id = 0
        self._stop_reason: RtcStopReason | None = None
        self._s_min = s_min

    @staticmethod
    def _chunk(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        if (
            array.shape != (HORIZON, MODEL_DIM)
            or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all()
        ):
            raise ValueError("RTC chunk must be finite floating [10,32]")
        return array

    @staticmethod
    def _observation(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        if array.size == 0 or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("observation must be a nonempty finite numeric array")
        return array

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def delay_history(self) -> tuple[int, ...]:
        return tuple(self._delay_history)

    @property
    def inflight(self) -> RtcInferenceRequest | None:
        return self._inflight

    @property
    def stop_reason(self) -> RtcStopReason | None:
        return self._stop_reason

    def _require_running(self) -> None:
        if self._stop_reason is not None:
            raise RtcStoppedError(self._stop_reason.value)

    def controller_tick(self, observation: np.ndarray) -> RtcTick:
        self._require_running()
        observed = self._observation(observation)
        if self._cursor >= HORIZON:
            self._stop_reason = RtcStopReason.BUFFER_UNDERRUN
            raise RtcStoppedError(self._stop_reason.value)
        index = self._cursor
        action = self._current[index].copy()
        self._cursor += 1
        self._latest_observation = observed.copy()
        return RtcTick(
            action=action,
            generation=self._generation,
            action_index=index,
            buffer_cursor=self._cursor,
            worker_wake_eligible=self._inflight is None and self._cursor >= self._s_min,
        )

    def begin_inference(self) -> RtcInferenceRequest:
        self._require_running()
        if self._inflight is not None:
            raise RuntimeError("RTC permits only one in-flight inference")
        if self._cursor < self._s_min or self._latest_observation is None:
            raise RuntimeError("RTC worker cannot begin before s_min controller ticks")
        s = self._cursor
        d_hat = max(self._delay_history)
        if not d_hat <= s <= HORIZON - d_hat:
            self._stop_reason = RtcStopReason.INFEASIBLE_DELAY_STRIDE
            raise RtcStoppedError(self._stop_reason.value)
        request = RtcInferenceRequest(
            request_id=self._next_request_id,
            generation=self._generation,
            s=s,
            d_hat=d_hat,
            observation=self._latest_observation.copy(),
            previous_prior=right_pad_after_stride(self._current, s=s),
            remaining_rows=HORIZON - s,
        )
        self._next_request_id += 1
        self._inflight = request
        return request

    def complete_inference(self, request_id: int, new_chunk: np.ndarray) -> RtcSwap:
        self._require_running()
        if self._inflight is None or request_id != self._inflight.request_id:
            raise RtcStaleResponseError("stale or duplicate RTC inference response")
        request = self._inflight
        d_observed = self._cursor - request.s
        if d_observed > HORIZON - request.s:
            self._stop_reason = RtcStopReason.OBSERVED_DELAY_OVERFLOW
            raise RtcStoppedError(self._stop_reason.value)
        new_value = self._chunk(new_chunk)

        old_generation = self._generation
        self._current = new_value.copy()
        self._cursor = d_observed
        self._generation += 1
        self._delay_history.append(d_observed)
        self._inflight = None
        return RtcSwap(
            request_id=request.request_id,
            old_generation=old_generation,
            new_generation=self._generation,
            s=request.s,
            d_hat=request.d_hat,
            d_observed=d_observed,
            delay_underestimated=d_observed > request.d_hat,
            next_action_index=self._cursor,
        )
