"""UF850: six absolute joint targets (rad) and normalized gripper opening."""
import dataclasses
import numpy as np
from openpi import transforms


def parse_rgb(image):
    a = np.asarray(image)
    if a.ndim != 3:
        raise ValueError('Expected one RGB image')
    if a.shape[0] == 3 and a.shape[-1] != 3:
        a = a.transpose(1, 2, 0)
    if a.shape[-1] != 3 or not np.isfinite(a).all():
        raise ValueError('Invalid RGB image')
    if np.issubdtype(a.dtype, np.floating):
        if a.min() < 0 or a.max() > 1:
            raise ValueError('Float RGB must be in [0,1]')
        a = np.rint(a * 255).astype(np.uint8)
    if a.dtype != np.uint8:
        raise ValueError('RGB must be uint8 or normalized float')
    return a


@dataclasses.dataclass(frozen=True)
class UF850Inputs(transforms.DataTransformFn):
    def __call__(self, data):
        state = np.asarray(data['observation/state'], dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError('UF850 observation requires 6 joints + gripper; remove SDK slot 6 first')
        front = parse_rgb(data['observation/front_image'])
        wrist = parse_rgb(data['observation/wrist_image'])
        result = {'state': state, 'image': {'base_0_rgb': front, 'left_wrist_0_rgb': wrist,
                  'right_wrist_0_rgb': np.zeros_like(front)},
                  'image_mask': {'base_0_rgb': np.True_, 'left_wrist_0_rgb': np.True_,
                                 'right_wrist_0_rgb': np.False_}}
        if 'actions' in data:
            action = np.asarray(data['actions'], dtype=np.float32)
            if action.ndim != 2 or action.shape[-1] != 7 or not np.isfinite(action).all():
                raise ValueError('Expected H x 7 absolute joint/gripper targets')
            result['actions'] = action
        if 'prompt' in data:
            result['prompt'] = data['prompt']
        return result


@dataclasses.dataclass(frozen=True)
class UF850Outputs(transforms.DataTransformFn):
    def __call__(self, data):
        action = np.asarray(data['actions'])
        if action.ndim != 2 or action.shape[-1] < 7 or not np.isfinite(action).all():
            raise ValueError('Invalid model action chunk')
        # AbsoluteActions has already undone the six-joint delta transform.
        return {'actions': action[:, :7]}
