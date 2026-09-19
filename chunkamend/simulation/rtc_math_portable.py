from __future__ import annotations
"""RTC paper math plus the historical synchronous d=0 mechanism control."""

import numpy as np

HORIZON = 10
MODEL_DIM = 32
PHYSICAL_DIMS = slice(0, 7)
D0_CONTROL_ID = "RTC-D0-SYNC-OPENPI-EQ2-5"
DENOISING_STEPS = 10
MAX_GUIDANCE_WEIGHT = 5.0


def _tensor(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (HORIZON, MODEL_DIM)
        or not np.issubdtype(array.dtype, np.floating)
        or not np.isfinite(array).all()
    ):
        raise ValueError("expected finite floating [10,32] tensor")
    return array


def _validate_delay_stride(*, d: int, s: int) -> None:
    if (
        isinstance(d, bool)
        or not isinstance(d, int)
        or isinstance(s, bool)
        or not isinstance(s, int)
        or not 0 <= d <= s
        or s > HORIZON - d
    ):
        raise ValueError("requires integer 0 <= d <= s <= H-d for H=10")


def rtc_prefix_weights(*, d: int, s: int) -> np.ndarray:
    """RTC Equation 5 prefix weights for every feasible H=10 ``d,s`` pair."""
    _validate_delay_stride(d=d, s=s)
    indices = np.arange(HORIZON, dtype=np.int64)
    weights = np.zeros(HORIZON, dtype=np.float64)
    weights[indices < d] = 1.0
    soft = (indices >= d) & (indices < HORIZON - s)
    c = (HORIZON - s - indices[soft]) / (HORIZON - s - d + 1)
    weights[soft] = c * np.expm1(c) / np.expm1(1.0)
    return weights


def soft_mask(*, stride: int, d: int = 0) -> np.ndarray:
    """Compatibility wrapper for the nonprimary synchronous RTC-d0 control."""
    if stride not in {3, 5, 9} or d != 0:
        raise ValueError("RTC-d0 control requires d=0 and stride in {3,5,9}")
    return rtc_prefix_weights(d=d, s=stride)


def right_pad_after_stride(chunk: np.ndarray, *, s: int) -> np.ndarray:
    """Return ``chunk[s:H]`` at rows zero onward, with exactly ``s`` zero rows."""
    value = _tensor(chunk)
    if isinstance(s, bool) or not isinstance(s, int) or not 0 <= s <= HORIZON:
        raise ValueError("s must be an integer in [0,10]")
    result = np.zeros_like(value)
    result[: HORIZON - s] = value[s:]
    return result


def fixed_delay_execution(previous: np.ndarray, new: np.ndarray, *, d: int, s: int) -> tuple[np.ndarray, np.ndarray]:
    """Paper fixed-delay indexing: execute old[:d], new[d:s], then shift new by s."""
    previous_value, new_value = _tensor(previous), _tensor(new)
    _validate_delay_stride(d=d, s=s)
    executed = np.concatenate((previous_value[:d], new_value[d:s]), axis=0)
    return executed, right_pad_after_stride(new_value, s=s)


def denoised_estimate(x_t: np.ndarray, t: float, velocity: np.ndarray) -> np.ndarray:
    x_t, velocity = _tensor(x_t), _tensor(velocity)
    if not np.isfinite(t):
        raise ValueError("time must be finite")
    return x_t - t * velocity


def prior_error(previous_prior: np.ndarray, current_hat: np.ndarray) -> np.ndarray:
    """Error against an already aligned, right-padded RTC prior."""
    return _tensor(previous_prior) - _tensor(current_hat)


def aligned_error(previous: np.ndarray | None, current_hat: np.ndarray, *, stride: int) -> np.ndarray | None:
    """Legacy d=0 control alignment; the M3 primary uses ``prior_error``."""
    current_hat = _tensor(current_hat)
    if previous is None:
        return None
    previous = _tensor(previous)
    if stride not in {3, 5, 9}:
        raise ValueError("unsupported stride")
    error = np.zeros_like(current_hat)
    error[: HORIZON - stride] = previous[stride:] - current_hat[: HORIZON - stride]
    return error


def guidance_from_vjp(
    jacobian: np.ndarray,
    error: np.ndarray,
    *,
    stride: int | None = None,
    d: int = 0,
    s: int | None = None,
) -> np.ndarray:
    """Apply temporal RTC weights before VJP over the complete model tensor."""
    error = _tensor(error)
    jacobian = np.asarray(jacobian)
    if jacobian.shape != (HORIZON * MODEL_DIM, HORIZON * MODEL_DIM) or not np.isfinite(jacobian).all():
        raise ValueError("expected finite [320,320] Jacobian")
    if s is None:
        if stride is None:
            raise ValueError("s is required for primary RTC guidance")
        s = stride
    elif stride is not None and stride != s:
        raise ValueError("stride and s disagree")
    weighted = error * rtc_prefix_weights(d=d, s=s)[:, None]
    guidance = (jacobian.T @ weighted.reshape(-1)).reshape(HORIZON, MODEL_DIM)
    return guidance


def paper_guidance_weight_reverse_time(time: float, *, beta: float) -> float:
    """Paper Equation 2 guidance weight under OpenPI time ``t = 1 - tau``."""
    if not np.isfinite(time) or time < 0 or time > 1:
        raise ValueError("time must be finite and in [0,1]")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    denominator = time * (1.0 - time)
    if denominator == 0:
        return float(beta)
    raw_weight = (time**2 + (1.0 - time) ** 2) / denominator
    return float(min(beta, raw_weight))


def guided_velocity(velocity: np.ndarray, guidance: np.ndarray, *, time: float, beta: float) -> np.ndarray:
    velocity, guidance = _tensor(velocity), _tensor(guidance)
    weight = paper_guidance_weight_reverse_time(time, beta=beta)
    return velocity - weight * guidance


class RtcD0State:
    """Historical synchronous d=0 prior store; not the M3 asynchronous primary."""

    def __init__(self) -> None:
        self.previous_chunk: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_chunk = None

    def update(self, chunk: np.ndarray) -> None:
        self.previous_chunk = _tensor(chunk).copy()
