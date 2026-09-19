"""Task and home confirmation gate. Standard library only; no device access."""
import json
import math
from pathlib import Path


def load_spec(base):
    return json.loads((Path(base) / 'TASK_SPEC.json').read_text())


def validate_spec(spec, task, robot_ip, configured_home):
    required = ('task_id', 'task_name_en', 'task_prompt_en', 'objects',
                'target', 'success_condition', 'home_confirmation')
    missing = [k for k in required if not isinstance(spec.get(k), str) or not spec[k].strip()]
    if spec.get('confirmed') is not True or missing:
        raise ValueError('任务/起点尚待用户确认: ' + ', '.join(missing))
    count = spec.get('target_episodes')
    if count is not None and (type(count) is not int or count <= 0):
        raise ValueError('计划示教数量必须为空或正整数；不自动截断回合')
    if task != spec['task_prompt_en'] or robot_ip != spec['robot_ip']:
        raise ValueError('启动参数与已确认任务/IP不一致')
    home = spec.get('initial_pose_mm_degrees')
    if (not isinstance(home, list) or len(home) != 6
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in home)
            or home != configured_home):
        raise ValueError('任务起点未确认或与config.py不一致')
    return spec
