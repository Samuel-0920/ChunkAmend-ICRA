#!/usr/bin/env python3
"""Read-only preflight or the explicit, human-started UF850 collection entry."""

import argparse
import datetime
import fcntl
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from task_spec import load_spec, validate_spec


BASE = Path(__file__).resolve().parent
PYTHON = os.environ.get("UF850_COLLECTION_PYTHON", sys.executable)
import shutil
ADB = os.environ.get("UF850_ADB", shutil.which("adb") or "adb")
DEFAULT_IP = "192.168.1.237"
DEFAULT_TASK = None
PREPARATION_BATCH = "preparation-20260907"
PREPARATION_TARGET = 20


def configured_initial_pose():
    spec = importlib.util.spec_from_file_location('operator_pose_config', BASE / 'webxr/config.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.INITIAL_POSITION)


def saved_task_home(task):
    path = BASE / 'TASK_HOME.json'
    if not path.exists():
        return None
    home = json.loads(path.read_text())
    if home['task'] != task:
        return None
    if home['initial_pose_mm_degrees'] != configured_initial_pose():
        raise RuntimeError('TASK_HOME.json与config.py的初始位姿不一致，请先核实配置')
    return home


def preparation_status(root=None):
    root = Path(root) if root is not None else BASE / "preparation" / PREPARATION_BATCH
    completed = 0
    for path in root.glob("session_*/episodes/episode_*/manifest.json"):
        manifest = json.loads(path.read_text())
        if (manifest.get("preparation_batch") != PREPARATION_BATCH
                or manifest.get("training_eligible") is not False
                or manifest.get("exclude_from_training") is not True):
            raise RuntimeError(f"练习数据标记不一致，停止累计: {path}")
        completed += 1
    return {"root": str(root), "completed_before": completed,
            "target": PREPARATION_TARGET, "preparation_batch": PREPARATION_BATCH,
            "collection_purpose": "initial_pose_and_teleop_practice",
            "training_eligible": False, "exclude_from_training": True}


def run_preparation(robot_ip, task):
    root = BASE / "preparation" / PREPARATION_BATCH
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".operator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("已有练习入口运行，请先退出它") from exc
        profile = preparation_status(root)
        home = saved_task_home(task)
        if home:
            profile.update(expected_initial_pose_mm_degrees=home['initial_pose_mm_degrees'],
                           task_home_id=home['task_id'], task_home_evidence=home['evidence'])
        _atomic_json(root / "PREPARATION.json", profile)
        if profile["completed_before"] >= PREPARATION_TARGET:
            print("20条练习已保存；本入口不再连接设备或新增回合。")
            return 0
        return run_collection(robot_ip, task, preparation=profile)


def camera_environment():
    """Run the reviewed serial/device/colour-format check without opening a stream."""
    spec = importlib.util.spec_from_file_location(
        "identity_check", BASE / "source/launchers/launch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE = BASE
    return module.camera_environment()


def _run_adb(*args):
    return subprocess.run(
        [ADB, *args], check=True, capture_output=True, text=True, timeout=10
    )


def quest_usb_preflight(timeout=180.0, poll_interval=2.0):
    if not Path(ADB).is_file():
        raise RuntimeError(f"未找到 ADB: {ADB}")
    deadline = time.monotonic() + timeout
    last_status = None
    while True:
        result = _run_adb("devices", "-l")
        devices = []
        problems = []
        for line in result.stdout.splitlines()[1:]:
            fields = line.split()
            if len(fields) < 2:
                continue
            if fields[1] == "device":
                devices.append(fields[0])
            elif fields[1] in {"unauthorized", "offline"}:
                problems.append(f"{fields[0]}:{fields[1]}")
        if len(devices) == 1 and not problems:
            print(f"Quest ADB 已授权: {devices[0]}", flush=True)
            return devices[0]

        if problems:
            status = "Quest 等待授权: " + ", ".join(problems)
            guidance = (
                "若没有弹窗：先在 Meta Horizon 手机应用的设备设置中确认开发者模式已开启；"
                "再到头显‘快速设置 → 设置 → 开发者’开启 MTP 通知，保持头显解锁后重新插拔 USB。"
                "同一电脑的不同 Linux 用户默认使用不同 ADB 密钥，当前用户需要单独授权；"
                "不需要撤销另一个用户的授权。若设置已开启仍无弹窗，请普通重启头显后重新连接。"
                "看到‘允许 USB 调试’后勾选‘始终允许此计算机’，再点‘允许’。脚本会自动继续。"
            )
        else:
            status = f"等待一台 Quest USB 设备；当前检测到 {len(devices)} 台已授权设备"
            guidance = "请连接并解锁 Quest，保持 USB 线连接。脚本会自动继续。"
        if status != last_status:
            print(status, flush=True)
            print(guidance, flush=True)
            last_status = status

        if time.monotonic() >= deadline:
            raise RuntimeError(
                status
                + "；等待授权超时。若上述设置正确但仍无弹窗，请普通重启 Quest，"
                "解锁并重新连接 USB，然后再次运行 AUTHORIZE_QUEST_ONLY.sh。"
            )
        time.sleep(poll_interval)


def _require_free_port(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Match the WebSocket server's bind/reuse policy: closed TCP sessions
        # in TIME_WAIT must not prevent restarting, but live listeners must.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        raise RuntimeError(f"本机端口 {port} 已被占用: {exc}") from exc
    finally:
        sock.close()


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _stop_process(process, name):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    print(f"已停止 {name}", flush=True)


def _wait_for_http(process, port=8080, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Quest 网页服务启动失败，退出码 {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("Quest 网页服务未在3秒内就绪")


def run_collection(robot_ip, task, preparation=None):
    with (BASE / '.camera-owner.lock').open('a') as ownership:
        try:
            fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('请先停止双相机预览或退出其他采集入口，再启动采集') from exc
        return _run_collection(robot_ip, task, preparation)


def _run_collection(robot_ip, task, preparation=None):
    ipaddress.IPv4Address(robot_ip)
    cameras = camera_environment()
    quest_serial = quest_usb_preflight(
        timeout=float(os.environ.get("UF850_ADB_AUTH_TIMEOUT", "180"))
    )
    _require_free_port(8080)
    _require_free_port(8765)

    env = os.environ.copy()
    for name in ["PYTHONPATH", "PYTHONHOME"]:
        env.pop(name, None)
    env.update(cameras)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "UF850_MANUAL_COLLECTION": "1",
            "UF850_TASK": task,
            "UF850_PREPARATION_PROFILE": json.dumps(preparation),
        }
    )

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_root = Path(preparation["root"]) if preparation else BASE / "captures"
    output = output_root / ("session_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    session_path = output / "session.json"
    session = {
        "robot_ip": robot_ip,
        "task_spec": load_spec(BASE),
        "task": task,
        "quest_serial": quest_serial,
        "cameras": json.loads((BASE / "CAMERAS.json").read_text()),
        "resolved_devices": {
            key: value
            for key, value in cameras.items()
            if key in {"MAIN_CAM_DEVICE", "WRIST_CAM_DEVICE"}
        },
        "operator_started_hardware_entry": True,
        "hardware_acceptance_completed": False,
        "status": "starting",
    }
    if preparation:
        session.update(preparation)
        session["initial_pose_source"] = (
            "saved_task_home"
            if 'expected_initial_pose_mm_degrees' in preparation
            else "operator_positioned_current_sdk_pose")
    else:
        session.update(collection_purpose='formal_demonstration', training_eligible=True,
                       exclude_from_training=False, review_required=True)
    _atomic_json(session_path, session)

    http_process = None
    control_process = None
    http_log = None
    adb_ports = []
    try:
        _run_adb("-s", quest_serial, "reverse", "tcp:8765", "tcp:8765")
        adb_ports.append(8765)
        _run_adb("-s", quest_serial, "reverse", "tcp:8080", "tcp:8080")
        adb_ports.append(8080)

        http_log = (output / "quest_http.log").open("w", encoding="utf-8")
        http_process = subprocess.Popen(
            [PYTHON, "-B", "-m", "http.server", "8080", "--bind", "127.0.0.1"],
            cwd=BASE / "webxr",
            env=env,
            stdout=http_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _wait_for_http(http_process)
        session["status"] = "running"
        _atomic_json(session_path, session)

        print("=" * 72, flush=True)
        print("真实硬件采集入口已启动", flush=True)
        if preparation:
            print(f"准备练习：已保存 {preparation['completed_before']}/20，本轮数据排除训练。", flush=True)
            print("核对已保存的任务起点；若当前位置有偏移，先自动归位，再等待标定。B结束后返回同一起点。", flush=True)
        print(f"任务: {task}", flush=True)
        print("Quest 常用入口：未知来源 → My project。等待终端显示 Quest 已连接。", flush=True)
        print("My project 使用 USB 转发连接 127.0.0.1:8765；服务就绪后完全退出并重新打开应用。", flush=True)
        print("浏览器备用入口：http://localhost:8080/index.html", flush=True)
        print("B开始：归位→自动标定→录制；B结束：保存→归位。归位完成后A可回收刚保存的一条。", flush=True)
        print(f"输出: {output}", flush=True)
        print("=" * 72, flush=True)

        control_process = subprocess.Popen(
            [
                PYTHON,
                "-B",
                "-u",
                "robot_control_with_data_collection_reset.py",
                robot_ip,
                str(output),
            ],
            cwd=BASE / "webxr",
            env=env,
        )
        return_code = control_process.wait()
        session["status"] = "stopped" if return_code == 0 else "failed"
        session["control_returncode"] = return_code
        _atomic_json(session_path, session)
        return return_code
    except KeyboardInterrupt:
        session["status"] = "operator_interrupt"
        _atomic_json(session_path, session)
        return 130
    except Exception as exc:
        session["status"] = "launch_failure"
        session["launch_error"] = str(exc)
        _atomic_json(session_path, session)
        raise
    finally:
        _stop_process(control_process, "采集控制进程")
        _stop_process(http_process, "Quest 网页服务")
        if http_log is not None:
            http_log.close()
        for port in adb_ports:
            subprocess.run(
                [ADB, "-s", quest_serial, "reverse", "--remove", f"tcp:{port}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["check", "plan", "authorize-quest", "collection", "preparation", "preparation-status"]
    )
    parser.add_argument("--robot-ip", default=DEFAULT_IP)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--site-ready",
        action="store_true",
        help="操作者已检查工作区、归位路径、急停和设备独占",
    )
    args = parser.parse_args()
    spec = load_spec(BASE)
    args.task = args.task or spec.get("task_prompt_en")

    if args.mode == "preparation-status":
        print(json.dumps(preparation_status(), ensure_ascii=False, indent=2))
        return 0

    if args.mode == "check":
        print(json.dumps(camera_environment(), indent=2))
        print("Identity/formats only; no image stream, model or robot connection.")
        return 0
    if args.mode == "plan":
        print(
            json.dumps(
                {
                    "robot_ip": args.robot_ip,
                    "task_spec": spec,
                    "inherited_home_requires_confirmation": not bool(spec.get("home_confirmation")),
                    "task": args.task,
                    "initial_pose_mm_degrees": configured_initial_pose(),
                    "saved_task_home": saved_task_home(args.task),
                    "initial_and_b_end_motion": True,
                    "cameras": json.loads((BASE / "CAMERAS.json").read_text()),
                    "quest_usb_required": True,
                    "output": str(BASE / "captures/session_<timestamp>"),
                    "format": "uf850_raw_capture_v1; not training ready",
                    "pending_site_checks": [
                        "工作区与急停",
                        "初始/归位路径",
                        "夹爪方向",
                        "采集与推理进程互斥",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.mode == "authorize-quest":
        serial = quest_usb_preflight(
            timeout=float(os.environ.get("UF850_ADB_AUTH_TIMEOUT", "180"))
        )
        print(f"Quest USB 调试授权完成: {serial}")
        print("未打开相机、未连接机器人。现在可以退出并运行正式采集入口。")
        return 0
    if not args.site_ready:
        parser.error("collection 需要 --site-ready，表示操作者已完成现场检查")
    validate_spec(spec, args.task, args.robot_ip, configured_initial_pose())
    if args.mode == "preparation":
        return run_preparation(args.robot_ip, args.task.strip())
    return run_collection(args.robot_ip, args.task.strip())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
