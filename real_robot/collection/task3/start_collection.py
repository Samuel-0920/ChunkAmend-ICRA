#!/usr/bin/python3
"""Human-started collection entry; launching it starts hardware without a prompt."""
import sys

import launch
from task_spec import load_spec, validate_spec


def main():
    spec = load_spec(launch.BASE)
    home = launch.configured_initial_pose()
    print('\n任务：' + spec['task_name_en'])
    print(spec['task_prompt_en'])
    print('成功条件：' + spec['success_condition'])
    print('数据目录：' + str(launch.BASE / 'captures'))
    print('机器人：' + spec['robot_ip'])
    print('归位起点 [x,y,z (mm), roll,pitch,yaw (度)]：' + str(home))
    print('完成当前任务后按B结束本回合。')
    print('B开始：归位→自动标定→录制；B结束：保存→归位；归位后A回收刚保存回合。')
    count = spec.get('target_episodes')
    print(f'目标 {count} 条示教；数量不自动截断动作，完成后自行退出。' if count else '采集数量不限，完成后自行退出。')
    if spec.get('collection_condition') == 'static':
        print('当前仅采静态：先90条，杯子回合内保持原位；松爪后柠檬进入杯内即可按B保存，无需额外停留。60条换位另行启动。')
    print('\n开始后将打开双相机并控制机器人归位。')
    validate_spec(spec, spec['task_prompt_en'], spec['robot_ip'], home)
    return launch.run_collection(spec['robot_ip'], spec['task_prompt_en'])


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (EOFError, KeyboardInterrupt):
        print('\n已取消。')
        raise SystemExit(130)
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
