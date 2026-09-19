"""Paper-faithful ACT temporal ensembling and an explicitly nonprimary legacy control."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

HORIZON = 10
MODEL_DIM = 32
PHYSICAL_DIM_COUNT = 7
ACT_DECAY = 0.01
PRIMARY_METHOD_ID = "TE-ACT-EVERY-STEP-OPENPI"
MATCHED_CADENCE_CONTROL_ID = "TE-MATCHED-CHUNK-CADENCE-ABLATION"


@dataclass(frozen=True)
class _Prediction:
    query: int
    chunk: np.ndarray


@dataclass(frozen=True)
class ActTeDecision:
    """One exact-target ACT ensemble result in normalized OpenPI action space."""

    normalized_model_row: np.ndarray
    contributor_queries: tuple[int, ...]
    target_offsets: tuple[int, ...]
    weights: tuple[float, ...]
    effective_sample_size: float
    contributor_normalized_actions: np.ndarray

    @property
    def action(self) -> np.ndarray:
        """Return the seven normalized physical action dimensions."""
        return self.normalized_model_row[:PHYSICAL_DIM_COUNT].copy()


class ActTemporalEnsemble:
    """ACT-style every-step, exact-absolute-target temporal ensemble.

    A fresh H=10 prediction must be supplied at every consecutive logical
    timestep. Predictions are stored with explicit occupancy, so an all-zero
    prediction is data rather than a sentinel. For target ``t`` the covering
    predictions are ordered oldest to newest and receive ACT's fixed
    ``exp(-0.01 * i)`` weights. All seven physical dimensions are averaged;
    dimensions 7:32 come unchanged from the current query's row zero.
    """

    def __init__(self) -> None:
        self._history: dict[tuple[str, str], list[_Prediction]] = defaultdict(list)
        self._last_query: dict[tuple[str, str], int] = {}

    def reset(self) -> None:
        self._history.clear()
        self._last_query.clear()

    def step(
        self,
        *,
        query_timestep: int,
        chunk: np.ndarray,
        episode_id: str,
        method_id: str,
    ) -> ActTeDecision:
        if isinstance(query_timestep, bool) or not isinstance(query_timestep, int) or query_timestep < 0:
            raise ValueError("query_timestep must be a non-negative integer")
        value = np.asarray(chunk)
        if (
            value.shape != (HORIZON, MODEL_DIM)
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()
            or not episode_id
            or not method_id
        ):
            raise ValueError("requires a finite floating [10,32] chunk and nonempty key")

        key = (episode_id, method_id)
        expected = 0 if key not in self._last_query else self._last_query[key] + 1
        if query_timestep != expected:
            raise ValueError(f"ACT TE requires every-step query {expected}, got {query_timestep}")

        retained = [prediction for prediction in self._history[key] if 0 <= query_timestep - prediction.query < HORIZON]
        retained.append(_Prediction(query_timestep, value.copy()))
        retained.sort(key=lambda prediction: prediction.query)

        offsets = tuple(query_timestep - prediction.query for prediction in retained)
        physical = np.stack(
            [
                prediction.chunk[offset, :PHYSICAL_DIM_COUNT]
                for prediction, offset in zip(retained, offsets, strict=True)
            ]
        )
        unnormalized = np.exp(-ACT_DECAY * np.arange(len(retained), dtype=np.float64))
        weights = unnormalized / np.sum(unnormalized)
        model_row = value[0].copy()
        model_row[:PHYSICAL_DIM_COUNT] = np.average(physical, axis=0, weights=weights).astype(value.dtype, copy=False)

        # Commit only after all validation and arithmetic succeeds.
        self._history[key] = retained
        self._last_query[key] = query_timestep
        return ActTeDecision(
            normalized_model_row=model_row,
            contributor_queries=tuple(prediction.query for prediction in retained),
            target_offsets=offsets,
            weights=tuple(float(weight) for weight in weights),
            effective_sample_size=float(1.0 / np.sum(weights**2)),
            contributor_normalized_actions=physical.copy(),
        )


class MatchedCadenceTemporalEnsemble:
    """Historical matched-cadence TE ablation; not the M3 primary ACT method."""

    def __init__(self, *, horizon: int = HORIZON, history_cap: int | None = None) -> None:
        if horizon != HORIZON or (history_cap is not None and history_cap < 1):
            raise ValueError("horizon is 10 and history_cap is positive or None")
        self.horizon, self.history_cap = horizon, history_cap
        self._history: dict[tuple[str, str], list[_Prediction]] = defaultdict(list)

    def reset(self) -> None:
        self._history.clear()

    def add(self, *, query_timestep: int, chunk: np.ndarray, episode_id: str, method_id: str) -> None:
        value = np.asarray(chunk)
        if value.shape != (self.horizon, 7) or not np.isfinite(value).all() or not episode_id or not method_id:
            raise ValueError("requires finite [10,7] chunk and nonempty key")
        key = (episode_id, method_id)
        self._history[key].append(_Prediction(query_timestep, value.copy()))
        if self.history_cap is not None:
            self._history[key] = self._history[key][-self.history_cap :]

    def action_at(self, *, timestep: int, episode_id: str, method_id: str, decay: float) -> np.ndarray:
        if not np.isfinite(decay) or decay < 0:
            raise ValueError("decay must be finite and non-negative")
        aligned = [
            prediction
            for prediction in self._history[(episode_id, method_id)]
            if 0 <= timestep - prediction.query < self.horizon
        ]
        if not aligned:
            raise LookupError("no prediction targets this logical timestep")
        aligned.sort(key=lambda prediction: prediction.query)
        if len(aligned) == 1:
            return aligned[0].chunk[timestep - aligned[0].query].copy()
        values = np.stack([prediction.chunk[timestep - prediction.query, :6] for prediction in aligned])
        weights = np.exp(-decay * np.arange(len(aligned), dtype=np.float64))
        result = np.empty(7, dtype=aligned[-1].chunk.dtype)
        result[:6] = np.average(values, axis=0, weights=weights).astype(result.dtype, copy=False)
        result[6] = aligned[-1].chunk[timestep - aligned[-1].query, 6]
        return result


# Source compatibility for the dormant M1 matched-cadence runtime. New M3
# code must import ActTemporalEnsemble or the explicit control name above.
TemporalEnsemble = MatchedCadenceTemporalEnsemble
