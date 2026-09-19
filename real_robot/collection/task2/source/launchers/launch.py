"""Camera identity and color-format validation; no robot commands."""
import json
from pathlib import Path
import subprocess
BASE = Path(__file__).resolve().parents[2]

def camera_environment():
    config = json.loads((BASE / 'CAMERAS.json').read_text())
    result = {}
    serials = []
    devices = []
    for role, prefix in [('front', 'MAIN'), ('wrist', 'WRIST')]:
        serial = config.get(role + '_serial')
        device = config.get(role + '_rgb_device')
        if not serial or not device:
            raise ValueError(f'{role} 相机尚未配置；请填写 {BASE / "CAMERAS.json"}')
        if not isinstance(serial, str) or not serial.isdigit():
            raise ValueError(f'{role}: 序列号必须是数字字符串')
        path = Path(device).resolve(strict=True)
        if not str(path).startswith('/dev/video') or not path.is_char_device():
            raise ValueError(f'{role}: 必须指定实际的 /dev/video 彩色设备或其稳定链接')
        info = subprocess.run(['udevadm', 'info', '--query=property', '--name', str(path)], check=True, capture_output=True, text=True, timeout=5).stdout
        properties = dict(line.split('=', 1) for line in info.splitlines() if '=' in line)
        detected = properties.get('ID_SERIAL_SHORT') or properties.get('ID_SERIAL', '').rsplit('_', 1)[-1]
        if detected != serial:
            raise ValueError(f'{role}: 设备 {path} 的序列号 {detected} 与配置 {serial} 不匹配')
        formats = subprocess.run(['v4l2-ctl', '--device', str(path), '--list-formats-ext'], check=True, capture_output=True, text=True, timeout=5).stdout
        if not any(fmt in formats for fmt in ["'YUYV'", "'MJPG'", "'RGB3'", "'BGR3'"]):
            raise ValueError(f'{role}: 尚未确认所选节点提供 RGB/彩色视频格式')
        serials.append(serial)
        devices.append(str(path))
        result[prefix + '_RS_SERIAL'] = serial
        result[prefix + '_CAM_DEVICE'] = str(path)
    if len(set(serials)) != 2 or len(set(devices)) != 2:
        raise ValueError('前置与腕部必须为两台独立相机')
    return result
