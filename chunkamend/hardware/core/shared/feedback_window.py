"""Portable completed-feedback window contract shared by controller and corrector."""
import numpy as np

RESPONSE_WINDOW = 64
FEEDBACK_SCHEMA = 'COMPLETED_FEEDBACK_TAIL64_V1'


def _count(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('Invalid ' + label)
    return int(value)


def completed_position(value):
    position = np.asarray(value)
    if position.shape != (3,) or not np.issubdtype(position.dtype, np.floating) or not np.isfinite(position).all():
        raise ValueError('Invalid completed TCP position')
    return position.copy()


def validate_feedback_window(metadata, history, controller_history, positions):
    if metadata['schema'] != FEEDBACK_SCHEMA:
        raise ValueError('Unknown feedback window schema')
    completed = _count(metadata['completed_pairs'], 'absolute completed pairs')
    start = _count(metadata['window_start'], 'absolute feedback window start')
    size = min(completed, RESPONSE_WINDOW)
    if start != completed - size:
        raise ValueError('Incomplete or shifted feedback window')
    for value, shape in ((history, (size, 7)), (controller_history, (size, 7)), (positions, (size + 1, 3))):
        value = np.asarray(value)
        if value.shape != shape or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError('Invalid completed feedback window')
    return completed, start


def snapshot_feedback(history, controller_history, positions, completed_pairs, *, include_memory_context=False):
    """Call while the controller lock is held; never include an unfinished step."""
    completed = _count(completed_pairs, 'absolute completed pairs')
    if len(history) != completed or len(controller_history) != completed or len(positions) != completed + 1:
        raise ValueError('Feedback ledger and absolute step mismatch')
    start = max(0, completed - RESPONSE_WINDOW)
    h = np.array(history[start:], copy=True).reshape(-1, 7)
    controllers = np.array(controller_history[start:], copy=True).reshape(-1, 7)
    tcp = np.array(positions[start:], copy=True).reshape(-1, 3)
    metadata = dict(schema=FEEDBACK_SCHEMA, completed_pairs=completed, window_start=start)
    validate_feedback_window(metadata, h, controllers, tcp)
    if include_memory_context:
        previous = None if start == 0 else np.asarray(positions[start],dtype=np.float64)-np.asarray(positions[start-1],dtype=np.float64)
        if previous is not None: previous.flags.writeable=False
        metadata['memory_context']=dict(schema='PRECEDING_COMPLETED_DISPLACEMENT_V1',completed_pairs=completed,window_start=start,
            preceding_row_index=start-1 if start else None,preceding_displacement=previous)
        validate_memory_context(metadata)
    for value in (h, controllers, tcp):
        value.flags.writeable = False
    return (h, controllers, tcp), metadata

def validate_memory_context(metadata):
    completed=_count(metadata['completed_pairs'],'completed pairs')
    start=_count(metadata['window_start'],'window start')
    if start != max(0,completed-RESPONSE_WINDOW):raise ValueError('Shifted memory window')
    context=metadata['memory_context']
    if (context['schema']!='PRECEDING_COMPLETED_DISPLACEMENT_V1'
            or _count(context['completed_pairs'],'memory completed pairs')!=completed
            or _count(context['window_start'],'memory window start')!=start):
        raise ValueError('Memory feedback identity mismatch')
    if start==0:
        if context['preceding_row_index'] is not None or context['preceding_displacement'] is not None:raise ValueError('No pre-episode displacement exists')
        return None
    if _count(context['preceding_row_index'],'preceding row')!=start-1:raise ValueError('Memory lag row mismatch')
    value=np.asarray(context['preceding_displacement'])
    if value.shape!=(3,) or value.dtype!=np.float64 or not np.isfinite(value).all():raise ValueError('Invalid memory lag value')
    return value
