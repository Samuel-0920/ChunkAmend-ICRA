"""Checkpoint action coordinates and H15 RTC prior rebasing.

The checkpoint emits first-six-joint offsets relative to its query state.  A
prior from an older image must therefore be rebased before RTC guidance; the
gripper and unused latent dimensions are not state-relative.
"""
import numpy as np

from .contracts import HORIZON, MODEL_DIM


class UF850ActionCoordinates:
    def __init__(self, action_stats: dict, *, use_quantiles: bool):
        if use_quantiles:
            lo = np.asarray(action_stats["q01"], dtype=np.float64)
            scale = (np.asarray(action_stats["q99"], dtype=np.float64) - lo + 1e-6) / 2
            offset = lo + scale
        else:
            offset = np.asarray(action_stats["mean"], dtype=np.float64)
            scale = np.asarray(action_stats["std"], dtype=np.float64) + 1e-6
        if offset.shape != (7,) or scale.shape != (7,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Expected finite seven-dimensional checkpoint action normalization")
        self.offset, self.scale = offset, scale

    @staticmethod
    def state(value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("Expected finite UF850 [joint_1..joint_6, gripper] state")
        return value

    @staticmethod
    def chunk(value):
        value = np.asarray(value)
        if value.shape != (HORIZON, MODEL_DIM) or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError("Expected finite H15 x 32 model chunk")
        return value

    def absolute(self, normalized, query_state):
        normalized = self.chunk(normalized)
        physical = normalized[:, :7] * self.scale + self.offset
        physical[:, :6] += self.state(query_state)[:6]
        return physical.astype(np.float32)

    def normalized_from_absolute(self, normalized_template, physical, query_state):
        """Replace only the seven physical dimensions with absolute joint targets.

        The first six output dimensions are query-relative model offsets, so
        an IK result must be mapped back relative to the same query state.
        Latent dimensions 7..31 remain byte-identical to the model output.
        """
        template = self.chunk(normalized_template)
        physical = np.asarray(physical)
        if physical.shape != (HORIZON, 7) or not np.issubdtype(physical.dtype, np.floating) or not np.isfinite(physical).all():
            raise ValueError("Expected finite H15 x 7 absolute UF850 targets")
        state = self.state(query_state)
        result = template.copy()
        values = physical.astype(result.dtype, copy=False).copy()
        values[:, :6] -= state[:6].astype(values.dtype, copy=False)
        result[:, :7] = (values - self.offset.astype(values.dtype, copy=False)) / self.scale.astype(values.dtype, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("Absolute UF850 targets cannot be represented in checkpoint coordinates")
        return result

    def rebased_prior(self, previous_normalized, previous_state, current_state, stride: int):
        previous_normalized = self.chunk(previous_normalized)
        if isinstance(stride, bool) or not isinstance(stride, int) or not 0 <= stride <= HORIZON:
            raise ValueError("RTC stride must be an integer in [0, H15]")
        shift = (self.state(previous_state)[:6] - self.state(current_state)[:6]) / self.scale[:6]
        prior = np.zeros_like(previous_normalized)
        prior[: HORIZON - stride] = previous_normalized[stride:]
        prior[: HORIZON - stride, :6] += shift.astype(prior.dtype)
        return prior
