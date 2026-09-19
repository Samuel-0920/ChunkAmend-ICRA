"""Model-response and completed-feedback contracts; no hardware I/O."""
from dataclasses import dataclass
import numpy as np
HORIZON=15
MODEL_DIM=32
class AsyncExecutorFault(RuntimeError):
    pass

@dataclass(frozen=True)
class FeedbackSnapshot:
    normalized_actions: np.ndarray
    controller_actions: np.ndarray
    tcp_positions_m: np.ndarray
    anchor_joint: np.ndarray | None = None

def validate_result(result, request):
    if not isinstance(result, dict):
        raise AsyncExecutorFault('Inference callback must return a mapping')
    for key in ('request_id', 'generation'):
        if result.get(key) != getattr(request, key):
            raise AsyncExecutorFault(f'Inference response identity mismatch: {key}')
    normalized = np.asarray(result.get('normalized_actions'))
    physical = np.asarray(result.get('actions'))
    state = np.asarray(result.get('query_state'))
    if normalized.shape != (HORIZON, MODEL_DIM) or not np.issubdtype(normalized.dtype, np.floating) or (not np.isfinite(normalized).all()):
        raise AsyncExecutorFault('Inference response needs a finite normalized H15 x 32 chunk')
    if physical.shape != (HORIZON, 7) or not np.issubdtype(physical.dtype, np.floating) or (not np.isfinite(physical).all()):
        raise AsyncExecutorFault('Inference response needs finite H15 x 7 absolute joint targets')
    if state.shape != (7,) or not np.isfinite(state).all():
        raise AsyncExecutorFault('Inference response needs a finite seven-dimensional query state')
    if not np.array_equal(state, request.observation['observation/state']):
        raise AsyncExecutorFault('Inference query state differs from its request')
    return (normalized.copy(), physical.copy(), state.copy())


def snapshot_feedback(normalized_actions, controller_actions, tcp_positions_m, anchor_joint=None) -> FeedbackSnapshot:
    normalized_actions = np.asarray(normalized_actions, dtype=np.float64)
    controller_actions = np.asarray(controller_actions, dtype=np.float64)
    tcp_positions_m = np.asarray(tcp_positions_m, dtype=np.float64)
    if normalized_actions.ndim != 2 or normalized_actions.shape[1:] != (7,) or controller_actions.shape != normalized_actions.shape:
        raise ValueError('Completed feedback command histories must be N x 7')
    if tcp_positions_m.shape != (len(normalized_actions) + 1, 3):
        raise ValueError('Completed TCP history must be exactly (N+1) x 3')
    if not all((np.isfinite(item).all() for item in (normalized_actions, controller_actions, tcp_positions_m))):
        raise ValueError('Feedback snapshot contains nonfinite values')
    normalized_actions = np.ascontiguousarray(normalized_actions).copy()
    controller_actions = np.ascontiguousarray(controller_actions).copy()
    tcp_positions_m = np.ascontiguousarray(tcp_positions_m).copy()
    for item in (normalized_actions, controller_actions, tcp_positions_m):
        item.flags.writeable = False
    if anchor_joint is not None:
        from .coordinates import UF850ActionCoordinates
        anchor_joint = UF850ActionCoordinates.state(anchor_joint).copy()
        anchor_joint.flags.writeable = False
    return FeedbackSnapshot(normalized_actions, controller_actions, tcp_positions_m, anchor_joint)
