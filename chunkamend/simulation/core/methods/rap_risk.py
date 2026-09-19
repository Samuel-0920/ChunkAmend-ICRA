"""Structural, causal normalized-Risk scoring; no RAP correction is invoked."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from chunkamend.simulation.core.methods.m1_types import RiskFeature


class RapRiskReason(str, Enum):
    AVAILABLE = "AVAILABLE"
    FEATURE_UNAVAILABLE = "FEATURE_UNAVAILABLE"
    MALFORMED_INPUT = "MALFORMED_INPUT"
    NONFINITE_INPUT = "NONFINITE_INPUT"
    SHAPE_MISMATCH = "SHAPE_MISMATCH"
    UNITS_MISMATCH = "UNITS_MISMATCH"
    TIMESTAMP_MISMATCH = "TIMESTAMP_MISMATCH"
    STALE_INPUT = "STALE_INPUT"
    RISK_COMPUTATION_FAILURE = "RISK_COMPUTATION_FAILURE"
    RISK_NONFINITE = "RISK_NONFINITE"


@dataclass(frozen=True)
class RapRiskOperatingPoint:
    boundary_scale: float
    next_scale: float
    cutoff: float

    def __post_init__(self) -> None:
        values = (self.boundary_scale, self.next_scale, self.cutoff)
        if not np.isfinite(values).all() or self.boundary_scale <= 0 or self.next_scale <= 0 or self.cutoff < 0:
            raise ValueError("invalid RAP Risk operating point")


@dataclass(frozen=True)
class RapRiskConfig:
    tcp_translation_units: str
    next_weight: float
    operating_points: tuple[tuple[int, RapRiskOperatingPoint], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tcp_translation_units, str) or not self.tcp_translation_units:
            raise ValueError("tcp_translation_units is required")
        if not np.isfinite(self.next_weight) or self.next_weight <= 0:
            raise ValueError("next_weight must be finite and strictly positive")
        strides = tuple(stride for stride, _ in self.operating_points)
        if len(strides) != 3 or set(strides) != {3, 5, 9} or len(set(strides)) != len(strides):
            raise ValueError("RAP Risk requires exactly one operating point for strides 3, 5, and 9")
        if any(not isinstance(point, RapRiskOperatingPoint) for _, point in self.operating_points):
            raise ValueError("invalid RAP Risk operating point")

    def operating_point(self, stride: int) -> RapRiskOperatingPoint:
        for candidate_stride, point in self.operating_points:
            if candidate_stride == stride:
                return point
        raise ValueError("unsupported RAP Risk stride")


@dataclass(frozen=True)
class RapRiskDecision:
    available: bool
    score: float | None
    score_units: str
    eligible: bool
    reason: RapRiskReason
    stride: int
    logical_timestep: int
    boundary_component: float | None
    next_component: float | None
    cutoff: float | None
    operator: str


_BOUNDARY = "boundary_translation_second_difference"
_NEXT = "next_in_chunk_translation_second_difference"
_SCORE_UNITS = "dimensionless_normalized_risk"


def _unavailable(reason: RapRiskReason, *, stride: int, logical_timestep: int) -> RapRiskDecision:
    return RapRiskDecision(False, None, _SCORE_UNITS, False, reason, stride, logical_timestep, None, None, None, ">")


def _validate_feature(feature: RiskFeature | None, *, expected_name: str, expected_units: str, logical_timestep: int) -> tuple[RapRiskReason | None, float | None]:
    if feature is None or feature.value is None:
        return RapRiskReason.FEATURE_UNAVAILABLE, None
    if feature.name != expected_name:
        return RapRiskReason.MALFORMED_INPUT, None
    if feature.shape != ():
        return RapRiskReason.SHAPE_MISMATCH, None
    if feature.units != expected_units:
        return RapRiskReason.UNITS_MISMATCH, None
    if feature.logical_timestamp != logical_timestep:
        return RapRiskReason.TIMESTAMP_MISMATCH, None
    if feature.latest_availability != "CANDIDATE_SELECTION":
        return RapRiskReason.STALE_INPUT, None
    try:
        value = float(feature.value)
    except (TypeError, ValueError):
        return RapRiskReason.MALFORMED_INPUT, None
    if not np.isfinite(value):
        return RapRiskReason.NONFINITE_INPUT, None
    if value < 0:
        return RapRiskReason.MALFORMED_INPUT, None
    return None, value


def evaluate_rap_risk(
    config: RapRiskConfig,
    *,
    stride: int,
    logical_timestep: int,
    features: dict[str, RiskFeature],
) -> RapRiskDecision:
    """Score exactly the two causal features; diagnostic fields are ignored."""
    try:
        point = config.operating_point(stride)
        boundary_reason, boundary = _validate_feature(
            features.get(_BOUNDARY), expected_name=_BOUNDARY,
            expected_units=config.tcp_translation_units, logical_timestep=logical_timestep,
        )
        if boundary_reason is not None:
            return _unavailable(boundary_reason, stride=stride, logical_timestep=logical_timestep)
        next_reason, next_value = _validate_feature(
            features.get(_NEXT), expected_name=_NEXT,
            expected_units="normalized_action_translation", logical_timestep=logical_timestep,
        )
        if next_reason is not None:
            return _unavailable(next_reason, stride=stride, logical_timestep=logical_timestep)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            boundary_component = float(np.divide(boundary, point.boundary_scale))
            next_component = float(np.divide(next_value, point.next_scale))
            score = float(np.add(boundary_component, config.next_weight * next_component))
        if not np.isfinite(score):
            return _unavailable(RapRiskReason.RISK_NONFINITE, stride=stride, logical_timestep=logical_timestep)
        return RapRiskDecision(True, score, _SCORE_UNITS, score > point.cutoff, RapRiskReason.AVAILABLE, stride, logical_timestep, boundary_component, next_component, point.cutoff, ">")
    except (AttributeError, TypeError, ValueError):
        return _unavailable(RapRiskReason.RISK_COMPUTATION_FAILURE, stride=stride, logical_timestep=logical_timestep)
