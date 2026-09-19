#!/usr/bin/env python3
"""
UF850 WebXR 遥操作控制 + 数据收集（录制结束自动归位版本）

功能:
1. VR遥操作控制 (Quest 3)
2. 实时数据收集 (30Hz采样)
3. GR00T LeRobot格式保存

按键说明:
  - Trigger: 控制夹爪开关
  - B键: 归位并自动标定后开始录制；再次按下保存并归位
  - Joystick X/Y: 增减末端关节 Joint 5/6
  - A键: 未录制且归位结束时，将本次刚保存的回合移到系统回收站
"""

import asyncio
import websockets
import json
import numpy as np
import time
from collections import deque
import sys
import threading
from pathlib import Path

# 添加父目录到路径，以便导入config
sys.path.insert(0, str(Path(__file__).parent))


import config  # config.py在real/目录下
from data_collection import CameraManager, DataCollector
from teleop_diagnostics import TeleopDiagnostics

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "datasets" / "uf850_teleop_reset"


class UF850WebXRControlWithDataCollection:
    """UF850 WebXR遥操作控制器 + 数据收集"""

    def __init__(self, robot_ip=None, dataset_path=None, preparation=None):
        dataset_path = (DEFAULT_DATASET_PATH if dataset_path is None
                        else Path(dataset_path).expanduser().resolve())
        self.robot_ip = robot_ip or config.ROBOT_IP
        self.preparation = preparation
        self._preparation_needs_home = False
        self.arm = None
        self._episode_lock = threading.Lock()
        self._resetting = False
        self._reset_failed = False
        self._client_active = False
        self._fault_reason = None
        self._fault_lock = threading.Lock()
        self._cached_timing = None
        self._command_timing = None
        self._gripper_timing = None
        self._camera_timing = None
        self._reset_task = None
        self._trash_task = None
        self._trashing = False
        self._latest_right_input = None
        self._require_trigger_release = False
        self._diagnostics = None
        self._last_input_ns = None
        self._last_tcp_diagnostic_ns = None
        self._filtered_pose = None

        # ========== 原有遥操作配置 ==========
        self.ENABLE_ROTATION = True
        self.ROTATION_MODE = "incremental"
        self.ROTATION_SCALE = 0.5
        self.ROTATION_DEADZONE = 0.05
        self.POSITION_SCALE = config.SCALE_FACTOR

        # 标定数据
        self.is_calibrated = False
        self.calibration_robot_pose = None
        self.calibration_vr_pos = None
        self.calibration_vr_rot_matrix = None
        self.calibration_robot_rot_matrix = None

        # 控制状态
        self.last_sent_pose = None
        self.last_command_time = None
        self.gripper_open = False
        self.last_trigger_state = False

        #夹爪
        self.gripper_target = 1.0
        self.gripper_deadzone = 0.02
        self.gripper_alpha = 0.4
        self.last_gripper_cmd_time = None
        self.GRIPPER_CMD_FREQ = 30
        self.last_sent_gripper = None
        self.GRIPPER_HOLD_DEADBAND = 0.012

        # 滤波
        self.position_buffer = deque(maxlen=3)
        self.rotation_buffer = deque(maxlen=3)

        # 统计
        self.frame_count = 0
        self.start_time = None

        # ========== 新增: 数据收集组件 ==========
        self.data_collector = DataCollector(
            dataset_path=dataset_path,
            record_freq=30,      # 30Hz — 独立录制线程精确控制
            control_freq=100,
            collection_metadata={**(preparation or {
                'collection_purpose': 'formal_demonstration',
                'training_eligible': True, 'exclude_from_training': False,
            }), 'translation_gain': 1.0 / self.POSITION_SCALE},
            max_episodes=(preparation['target'] - preparation['completed_before']) if preparation else None,
        )

        # ========== 新增: 相机管理 ==========
        self.camera_manager = None
        self.camera_thread = None
        self._camera_running = False
        self.latest_frames = {}
        self.camera_lock = threading.Lock()

        # ========== 新增: 按键状态跟踪 ==========
        self.last_b_button_state = False
        
        # ========== 新增: 暂停 ==========
        self.is_paused = False
        self.last_a_button_state = False

        # ========== 新增: Joystick 关节控制 ==========
        self.joystick_joint_offsets = np.zeros(6)
        self.JOYSTICK_SCALE = 0.01
        self.JOYSTICK_DEADZONE = 0.15

        # ========== A2+A3方案: 独立录制线程 + 状态/IK缓存 ==========
        self._latest_target_pose = None
        self._target_pose_lock = threading.Lock()
        self._record_thread = None
        self._record_running = False

        # A3: 后台状态+IK缓存（录制线程零SDK调用）
        self._cached_state = None
        self._cached_action = None
        self._cache_lock = threading.Lock()
        self._state_ik_thread = None
        self._monitor_thread = None
        self._monitor_running = False

        # 主循环计算的带 Joystick 偏移的目标关节角（供录制线程使用）
        self._latest_target_joints = None
        self._target_joints_lock = threading.Lock()

        print("\n" + "="*60)
        print("UF850 WebXR遥操作 + 数据收集系统")
        print("="*60)
        print("坐标系配置（半镜像模式 + 旋转控制）:")
        print("  Quest 3 WebXR: X=右, Y=上, Z=后(朝向用户)")
        print("  UF850 机械臂: X=前, Y=左, Z=上")
        print("  位置映射(半镜像):")
        print("    VR Z(后) → Robot X(前) [前后不镜像，取反]")
        print("    VR X(右) → Robot Y(左) [左右镜像，不取反]")
        print("    VR Y(上) → Robot Z(上)")
        print("="*60)
        print("按键说明:")
        print("  • Trigger: 控制夹爪开关")
        print("  • B键: 归位→自动标定→录制 / 结束保存并归位")
        print("  • A键: 未录制且归位完成时，回收本次刚保存的一条（可恢复）")
        print("  • Joystick X轴: 增减 Joint 6 (index 5)")
        print("  • Joystick Y轴: 增减 Joint 5 (index 4)")
        print("="*60)
        print(f"平移增益: {1.0/self.POSITION_SCALE:g}:1；目标平移限速: {config.TELEOP_MAX_LINEAR_SPEED_MM_S:g} mm/s")
        print(f"位置/旋转按时间滤波: {config.TELEOP_POSITION_FILTER_TAU_S:g}/{config.TELEOP_ROTATION_FILTER_TAU_S:g} s")
        print(f"数据集路径: {dataset_path}")
        print(f"采样频率: 30Hz (从100Hz控制循环下采样)")
        print("="*60 + "\n")

    def _checked_call(self, method, *args, **kwargs):
        # Serializes a stop latch against subsequent submission/enable calls.
        # A blocking SDK call may still delay stop; no hard-real-time guarantee.
        with self._fault_lock:
            if self._fault_reason:
                raise RuntimeError(self._fault_reason)
            code = method(*args, **kwargs)
            self._check_reset_result(code, getattr(method, '__name__', 'SDK command'))
            return code

    def start_diagnostics(self):
        self._diagnostics = TeleopDiagnostics(self.data_collector.dataset_path / 'teleop_diagnostics', {
            'schema_version': 1, 'clock': 'server_monotonic_ns',
            'translation_gain': 1.0 / self.POSITION_SCALE,
            'max_target_translation_speed_mm_s': config.TELEOP_MAX_LINEAR_SPEED_MM_S,
            'position_tau_s': config.TELEOP_POSITION_FILTER_TAU_S,
            'rotation_tau_s': config.TELEOP_ROTATION_FILTER_TAU_S,
            'max_control_dt_s': config.TELEOP_MAX_CONTROL_DT_S,
            'rotation_gain': self.ROTATION_SCALE,
            'saved_home_mm_degrees': config.INITIAL_POSITION,
            'input_clock_note': 'receive_ns is application dequeue time, not USB arrival time; client clocks are not synchronized',
            'coordinate_note': 'robot XYZ mm and RPY rad; client reference space/units are unverified unless supplied',
            'feedback_note': 'SDK read windows are not hardware acquisition timestamps; commands are not execution proof',
        })
        print('诊断日志: ' + str(self.data_collector.dataset_path / 'teleop_diagnostics'), flush=True)

    def _trace(self, kind, **fields):
        if self._diagnostics is not None:
            self._diagnostics.emit(kind, **fields)

    def _latch_fault(self, reason):
        with self._fault_lock:
            self._fault_reason = self._fault_reason or str(reason)
            self._reset_failed = True
            self.is_paused = True
            self.is_calibrated = False
        # Stop request is best effort, not physical stop confirmation.
        if self.arm is not None:
            try:
                self._check_reset_result(self.arm.set_state(4), 'stop request')
            except Exception as exc:
                print(f'Stop request failed: {exc}', flush=True)
        print(f'Collection fault latched: {reason}', flush=True)
        try:
            self._trace('fault', reason=str(reason))
        except Exception as exc:
            print(f'Diagnostic fault: {exc}', flush=True)

    # ========== 原有方法保持不变 ==========
    def vr_to_robot_position(self, x, y, z):
        """VR位置转换到机械臂坐标系（半镜像版本）"""
        robot_x = -z
        robot_y = x
        robot_z = y

        robot_x *= (1000.0 / self.POSITION_SCALE)
        robot_y *= (1000.0 / self.POSITION_SCALE)
        robot_z *= (1000.0 / self.POSITION_SCALE)

        return np.array([robot_x, robot_y, robot_z])

    def quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """四元数转旋转矩阵"""
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm < 1e-6:
            return np.eye(3)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz

        R = np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),      2*(xz + wy)],
            [    2*(xy + wz), 1 - 2*(xx + zz),      2*(yz - wx)],
            [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
        ])
        return R

    def rotation_matrix_to_rpy(self, R):
        """旋转矩阵转RPY角"""
        cos_pitch = np.hypot(R[0, 0], R[1, 0])
        pitch = np.arctan2(-R[2, 0], cos_pitch)
        if cos_pitch < 1e-8:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            yaw = 0
        else:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        return [roll, pitch, yaw]

    def rpy_to_rotation_matrix(self, roll, pitch, yaw):
        """RPY角转旋转矩阵"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cp*cy,  -cr*sy + sr*sp*cy,   sr*sy + cr*sp*cy],
            [cp*sy,   cr*cy + sr*sp*sy,  -sr*cy + cr*sp*sy],
            [ -sp,         sr*cp,              cr*cp]
        ])
        return R

    def vr_rotation_to_robot(self, qx, qy, qz, qw):
        """VR手柄旋转转换到机械臂坐标系（半镜像版本）"""
        R_vr = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)

        T_vr_to_robot_half_mirror = np.array([
            [ 0,  0, -1],
            [ 1,  0,  0],
            [ 0,  1,  0]
        ])

        R_robot = T_vr_to_robot_half_mirror @ R_vr @ T_vr_to_robot_half_mirror.T

        rpy = self.rotation_matrix_to_rpy(R_robot)
        rpy[0] = -rpy[0]
        rpy[2] = -rpy[2]

        R_robot_half_mirrored = self.rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])
        return R_robot_half_mirrored

    def rotation_matrix_to_axis_angle(self, R):
        """旋转矩阵转轴角表示"""
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle < 1e-6:
            return np.array([0, 0, 1, 0])

        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(angle))

        return np.array([axis[0], axis[1], axis[2], angle])

    def axis_angle_to_rotation_matrix(self, axis_angle):
        """轴角表示转旋转矩阵"""
        axis = axis_angle[:3]
        angle = axis_angle[3]

        if angle < 1e-6:
            return np.eye(3)

        axis = axis / np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c

        x, y, z = axis
        R = np.array([
            [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
            [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
            [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
        ])
        return R

    async def initialize_robot(self):
        """初始化机械臂 + 相机"""
        print("初始化UF850机械臂...")

        # ========== 新增: 初始化相机 ==========
        print("初始化相机...")
        try:
            self.camera_manager = CameraManager(
                camera_ids={
                    'wrist': config.WRIST_CAM_ID,
                    'front': config.MAIN_CAM_ID,
                },
                fps=30,
            )

            self.latest_frames = self.camera_manager.get_frames()
            self._camera_timing = dict(self.camera_manager.last_timing)

            # 启动相机采集线程
            self._camera_running = True
            self.camera_thread = threading.Thread(
                target=self._camera_capture_loop,
                daemon=True,
                name="CameraCaptureThread"
            )
            self.camera_thread.start()

            print("✓ 相机初始化完成\n")
        except Exception as e:
            print(f"⚠ 相机初始化失败: {e}")
            self._latch_fault(f"camera initialization: {e}")
            return False


        try:
            from xarm.wrapper import XArmAPI
            self.arm = XArmAPI(self.robot_ip, is_radian=True)
            print(f"✓ 连接成功: {self.robot_ip}")
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            return False

        # 在使能和运动指令之前，读取操作者摆好的练习起点。
        if self.preparation:
            self._capture_preparation_home()

        # 配置机械臂
        self._check_reset_result(self._checked_call(self.arm.clean_error), 'self.arm.clean_error')
        self._check_reset_result(self._checked_call(self.arm.clean_warn), 'self.arm.clean_warn')
        self._check_reset_result(self._checked_call(self.arm.motion_enable, enable=True), 'self.arm.motion_enable')
        self._check_reset_result(self._checked_call(self.arm.set_mode, 1), 'self.arm.set_mode')
        self._check_reset_result(self._checked_call(self.arm.set_state, 0), 'self.arm.set_state')

        if self._preparation_needs_home:
            print("当前位置偏离任务起点，正在自动归位；到达后才开放遥操...")
            try:
                await self._return_to_saved_home()
                receipt_path = self.data_collector.dataset_path / 'preparation_home.json'
                receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                receipt['startup_home_completed'] = True
                receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
                self._clear_teleop_reference()
                print("✓ 启动归位完成，等待 Quest 连接；按B自动标定并开始。")
            except (Exception, asyncio.CancelledError) as exc:
                self._latch_fault(f'启动归位失败: {exc}')
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return False
        else:
            self._move_to_initial_position()

        self._check_reset_result(self._checked_call(self.arm.set_mode, 1), 'self.arm.set_mode')
        self._check_reset_result(self._checked_call(self.arm.set_state, 0), 'self.arm.set_state')

        print("✓ 机械臂初始化完成\n")

        # ========== 启动后台线程 (A2+A3方案) ==========
        self._record_running = True

        # A3: 状态+IK缓存线程
        self._state_ik_thread = threading.Thread(
            target=self._state_ik_loop,
            daemon=True,
            name="StateIKThread"
        )
        self._state_ik_thread.start()

        # A2: 独立录制线程（零SDK调用）
        self._record_thread = threading.Thread(
            target=self._record_loop,
            daemon=True,
            name="RecordThread"
        )
        self._record_thread.start()

        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._debug_monitor_loop,
            daemon=True,
            name="DebugMonitorThread"
        )
        self._monitor_thread.start()

        print("✓ 状态缓存线程 + 独立录制线程 + 调试监控线程已启动\n")

        return True
            

    def _capture_preparation_home(self):
        code, state = self.arm.get_state()
        self._check_reset_result(code, '读取起点状态')
        if state == 1:
            raise RuntimeError('机械臂仍在运动，请先用示教器停稳并退出拖动/其他控制')
        code, pose = self.arm.get_position(is_radian=True)
        self._check_reset_result(code, '读取练习起始位姿')
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError('练习起始位姿必须是6个有效SDK反馈值')
        for index, axis in enumerate(('x', 'y', 'z')):
            low, high = config.WORKSPACE_LIMITS[axis]
            if not low <= pose[index] <= high:
                raise ValueError(f'当前起点{axis}={pose[index]:.1f}mm不在已配工作区[{low},{high}]，未使能遥操；请核实现场和限位')
        pose_degrees = pose.copy()
        pose_degrees[3:] = np.rad2deg(pose_degrees[3:])
        adopted = pose_degrees.copy()
        source = 'operator_positioned_current_sdk_pose'
        expected = self.preparation.get('expected_initial_pose_mm_degrees')
        if expected is not None:
            expected = np.asarray(expected, dtype=float)
            if expected.shape != (6,) or not np.isfinite(expected).all():
                raise ValueError('已保存的任务起点无效')
            for index, axis in enumerate(('x', 'y', 'z')):
                low, high = config.WORKSPACE_LIMITS[axis]
                if not low <= expected[index] <= high:
                    raise ValueError('已保存的任务起点不在已配工作区，未使能遥操')
            angle_delta = (pose_degrees[3:] - expected[3:] + 180) % 360 - 180
            self._preparation_needs_home = bool(
                np.max(np.abs(pose_degrees[:3] - expected[:3])) > 2.0
                or np.max(np.abs(angle_delta)) > 0.5)
            adopted = expected.copy()
            source = 'saved_task_home'
        record = {**self.preparation, 'initial_pose_mm_degrees': adopted.tolist(),
                  'observed_initial_pose_mm_degrees': pose_degrees.tolist(),
                  'initial_pose_source': source,
                  'startup_home_required': self._preparation_needs_home,
                  'startup_home_completed': False,
                  'captured_monotonic_ns': time.monotonic_ns(),
                  'return_path_verified': False}
        (self.data_collector.dataset_path / 'preparation_home.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        # 只更新当前进程；正式config.py和旧数据保持原样。
        config.INITIAL_POSITION = adopted.tolist()
        self.data_collector.collection_metadata.update(record)
        print(f"✓ 本次练习起点已读取: {config.INITIAL_POSITION}（mm、degree）；未发位置移动指令")

    def _move_to_initial_position(self):
        if self.preparation:
            print("准备模式：当前位置已满足起点要求，无需启动归位。")
            return
        print("移动到初始位置...")
        initial = config.INITIAL_POSITION.copy()
        initial[3:] = np.deg2rad(initial[3:])
        self._check_reset_result(self._checked_call(self.arm.set_mode, 0), 'self.arm.set_mode')
        self._check_reset_result(self._checked_call(self.arm.set_state, 0), 'self.arm.set_state')
        self._check_reset_result(self._checked_call(self.arm.set_position, *initial, wait=True), 'self.arm.set_position')

    def _camera_capture_loop(self):
        """相机采集线程 (30fps)"""
        print("相机采集线程已启动 (30fps)")
        while self._camera_running:
            try:
                frames = self.camera_manager.get_frames()
                with self.camera_lock:
                    self.latest_frames = frames
                    self._camera_timing = dict(self.camera_manager.last_timing)
                # Reads already block; do not add a full frame period.
                time.sleep(0.001)
            except Exception as e:
                with self.camera_lock:
                    self.latest_frames = {}
                    self._camera_timing = None
                self._latch_fault(f"camera read: {e}")
                break

        print("相机采集线程已停止")

    def _read_proc_status_kb(self, key):
        """读取 /proc/self/status 中的 kB 指标，用于轻量级调试日志。"""
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith(key + ":"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1])
        except Exception:
            pass
        return None

    def _read_mem_available_mb(self):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024.0
        except Exception:
            pass
        return None

    def _debug_monitor_loop(self):
        """定期打印资源状态，方便定位崩溃前的趋势。"""
        print("调试监控线程已启动 (30s间隔)", flush=True)

        while self._monitor_running:
            try:
                rss_kb = self._read_proc_status_kb("VmRSS")
                rss_mb = rss_kb / 1024.0 if rss_kb is not None else None
                mem_avail_mb = self._read_mem_available_mb()

                with self.camera_lock:
                    frame_keys = ",".join(sorted(self.latest_frames.keys())) if self.latest_frames else "none"

                with self._cache_lock:
                    has_state = self._cached_state is not None
                    has_action = self._cached_action is not None

                uptime = time.time() - self.start_time if self.start_time else 0.0
                rss_text = f"{rss_mb:.1f}MB" if rss_mb is not None else "unknown"
                avail_text = f"{mem_avail_mb:.0f}MB" if mem_avail_mb is not None else "unknown"

                print(
                    "[DebugMonitor] "
                    f"uptime={uptime:.1f}s "
                    f"rss={rss_text} "
                    f"mem_available={avail_text} "
                    f"recording={self.data_collector.is_recording} "
                    f"episode_frames={self.data_collector.episode_step_count} "
                    f"ws_frames={self.frame_count} "
                    f"camera_frames={frame_keys} "
                    f"cache_state={has_state} "
                    f"cache_action={has_action}",
                    flush=True,
                )
            except Exception as e:
                print(f"调试监控线程错误: {e}", flush=True)

            for _ in range(30):
                if not self._monitor_running:
                    break
                time.sleep(1.0)

        print("调试监控线程已停止", flush=True)

    def _state_ik_loop(self):
        """
        后台状态+IK缓存线程 (A3方案)
        
        持续以 ~100Hz 读取机械臂状态并预计算IK，将结果缓存。
        这样录制线程无需任何SDK调用，频率完全不受SDK阻塞影响。
        
        此线程是唯一做"读取"类SDK调用的线程，与主线程的"写入"类
        SDK调用（set_servo_cartesian, set_gripper_position）分离，
        减少锁竞争。
        """
        print("状态+IK缓存线程已启动 (~100Hz)")

        while self._record_running:
            try:
                read_start = time.monotonic_ns()
                code, angles = self.arm.get_servo_angle(is_radian=True)
                self._check_reset_result(code, 'joint feedback')
                code, gripper_pos = self.arm.get_gripper_position()
                self._check_reset_result(code, 'gripper feedback')
                state = np.append(angles, gripper_pos / 850.0)
                if not np.isfinite(state).all():
                    raise ValueError('Nonfinite SDK feedback')
                read_end = time.monotonic_ns()
                if self._diagnostics is not None and (
                        self._last_tcp_diagnostic_ns is None or
                        read_end - self._last_tcp_diagnostic_ns >= 1e9 / config.TELEOP_TCP_DIAGNOSTIC_HZ):
                    tcp_read_start = time.monotonic_ns()
                    code, tcp_pose = self.arm.get_position(is_radian=True)
                    tcp_read_end = time.monotonic_ns()
                    self._check_reset_result(code, 'diagnostic TCP feedback')
                    tcp_pose = np.asarray(tcp_pose, dtype=float)
                    if tcp_pose.shape != (6,) or not np.isfinite(tcp_pose).all():
                        raise ValueError('Invalid diagnostic TCP feedback')
                    self._trace('feedback', state_read_start_ns=read_start, state_read_end_ns=read_end,
                                joint_and_gripper_state=state.tolist(),
                                tcp_read_start_ns=tcp_read_start, tcp_read_end_ns=tcp_read_end,
                                tcp_pose_mm_radians=tcp_pose.tolist(),
                                resetting=self._resetting, calibrated=self.is_calibrated,
                                paused=self.is_paused, recording=self.data_collector.is_recording)
                    self._last_tcp_diagnostic_ns = tcp_read_end
                with self._target_joints_lock:
                    tj = list(self._latest_target_joints) if self._latest_target_joints is not None else None
                    command_timing = dict(self._command_timing) if self._command_timing else None
                    grip = self.last_sent_gripper
                    grip_timing = dict(self._gripper_timing) if self._gripper_timing else None
                with self._cache_lock:
                    if not self._resetting and self.is_calibrated and tj is not None and grip is not None:
                        self._cached_state = state
                        self._cached_action = np.append(tj, grip)
                        self._cached_timing = {'state_read_start_ns': read_start,
                            'state_read_end_ns': read_end, 'command': command_timing,
                            'gripper_command': grip_timing, 'execution_confirmed': False}
            except Exception as e:
                with self._cache_lock:
                    self._cached_state = self._cached_action = None
                self._latch_fault(f'feedback: {e}')
                break

            time.sleep(0.008)  # ~125Hz 上限，留余量给SDK锁竞争

        print("状态+IK缓存线程已停止")

    def _record_loop(self):
        """
        独立录制线程 (A2+A3方案) — 零SDK调用
        
        只从缓存读取 state/action，从相机缓存读取传感器数据。
        不调用任何SDK方法，因此定时精度不受SDK阻塞影响，
        能精确保持目标录制频率。
        """
        freq = self.data_collector.record_freq
        interval = 1.0 / freq
        print(f"录制线程已启动 (目标 {freq}Hz, 间隔 {interval*1000:.1f}ms, 零SDK调用)")

        while self._record_running:
            t0 = time.monotonic()

            try:
                with self._episode_lock:
                    # 仅在录制中且未暂停时执行
                    if self.data_collector.is_recording and not self.is_paused:
                        # 检查是否已有目标位姿
                        with self._target_pose_lock:
                            has_pose = self._latest_target_pose is not None

                        if has_pose:
                            # 1+2. 从缓存读取 state 和 action（零SDK调用！）
                            with self._cache_lock:
                                current_state = self._cached_state.copy() if self._cached_state is not None else None
                                current_action = self._cached_action.copy() if self._cached_action is not None else None
                                timing = dict(self._cached_timing) if self._cached_timing else None

                            # 3. 获取最新相机帧
                            with self.camera_lock:
                                frames = self.latest_frames.copy() if self.latest_frames else {}
                                camera_timing = dict(self._camera_timing) if self._camera_timing else None

                            # 4. 记录数据
                            if current_state is not None and current_action is not None:
                                now_ns = time.monotonic_ns()
                                if not timing or not camera_timing:
                                    raise RuntimeError('Missing timing evidence')
                                ages = [now_ns - timing['state_read_end_ns']] + [now_ns - v['read_end_ns'] for v in camera_timing.values()]
                                if any(age < 0 or age > 250_000_000 for age in ages):
                                    raise RuntimeError('State/camera cache older than engineering guard 250ms')
                                timing.update({'sample_ns': now_ns, 'sample_unix_ns': time.time_ns(), 'cameras': camera_timing})
                                self.data_collector.record_step(
                                    state=current_state,
                                    action=current_action,
                                    frames=frames,
                                    timestamp=now_ns / 1e9, timing=timing, is_intervention=True,
                                )
            except Exception as e:
                self._latch_fault(f"record: {e}")
                import traceback
                traceback.print_exc()

            # 精确控制录制频率：减去本次处理耗时后 sleep
            elapsed = time.monotonic() - t0
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

        print("录制线程已停止")

    def _get_current_state(self):
        """
        获取机械臂当前状态 (8维)

        Returns:
            np.ndarray [8]: [j0, j1, j2, j3, j4, j5, j6, gripper] (弧度)
        """
        # 获取关节角度 (弧度)
        code, angles = self.arm.get_servo_angle(is_radian=True)
        if code != 0:
            return None

        # 获取夹爪位置并归一化
        code, gripper_pos = self.arm.get_gripper_position()
        if code != 0:
            gripper_normalized = 0.5
        else:
            gripper_normalized = gripper_pos / 850.0

        # 组合为8维state (7个关节 + 1个夹爪)
        state = np.append(angles, gripper_normalized)
        return state

    def _compute_current_action(self, target_pose):
        """
        从目标笛卡尔位姿计算action (8维关节空间)

        Args:
            target_pose: [x, y, z, roll, pitch, yaw] (笛卡尔空间, mm和弧度)

        Returns:
            np.ndarray [8]: 目标关节角度 + 夹爪 (弧度)
        """
        # 方案1: 使用IK计算目标关节角度
        code, joint_angles = self.arm.get_inverse_kinematics(target_pose)
        if code != 0 or joint_angles is None:
            # IK失败，使用当前角度
            code, joint_angles = self.arm.get_servo_angle(is_radian=True)
            if code != 0:
                return None

        # 夹爪目标
        gripper_target = float(np.clip(self.gripper_target, 0.0, 1.0))

        action = np.append(joint_angles, gripper_target)
        return action

    def calibrate(self, vr_x, vr_y, vr_z, vr_qx, vr_qy, vr_qz, vr_qw):
        """标定"""
        if (getattr(self, 'preparation', None) and self.data_collector.max_episodes is not None
                and self.data_collector.retained_episode_count >= self.data_collector.max_episodes):
            print("20条准备练习已保存，请Ctrl+C退出；不再重新标定遥操。")
            return False
        print("\n" + "="*50)
        print("执行标定...")

        code, pose = self.arm.get_position(is_radian=True)
        if code != 0:
            print("✗ 获取机械臂位姿失败")
            return False

        self.calibration_robot_pose = pose
        self.calibration_vr_pos = self.vr_to_robot_position(vr_x, vr_y, vr_z)
        self.calibration_vr_rot_matrix = self.vr_rotation_to_robot(vr_qx, vr_qy, vr_qz, vr_qw)
        self.calibration_robot_rot_matrix = self.rpy_to_rotation_matrix(pose[3], pose[4], pose[5])

        print(f"机械臂位置: X={pose[0]:.1f}, Y={pose[1]:.1f}, Z={pose[2]:.1f} mm")
        print(f"机械臂姿态: Roll={np.rad2deg(pose[3]):.1f}°, Pitch={np.rad2deg(pose[4]):.1f}°, Yaw={np.rad2deg(pose[5]):.1f}°")

        self.is_calibrated = True
        self.last_sent_pose = pose
        self.last_command_time = time.monotonic()
        self._filtered_pose = list(pose)
        self.position_buffer.clear()
        self.rotation_buffer.clear()
        self.joystick_joint_offsets = np.zeros(6)

        self._trace('calibration', robot_pose_mm_radians=list(pose),
                    controller_position=[vr_x, vr_y, vr_z],
                    controller_quaternion_xyzw=[vr_qx, vr_qy, vr_qz, vr_qw])

        print("✓ 标定完成！")
        print("="*50 + "\n")
        return True

    def compute_target_pose(self, vr_x, vr_y, vr_z, vr_qx, vr_qy, vr_qz, vr_qw):
        """计算目标位姿（包含旋转）"""
        if not self.is_calibrated:
            return None

        # 位置部分
        current_vr_pos = self.vr_to_robot_position(vr_x, vr_y, vr_z)
        delta_pos = current_vr_pos - self.calibration_vr_pos

        if np.linalg.norm(delta_pos) < 2.0:
            delta_pos = np.zeros(3)

        target_pos = self.calibration_robot_pose[:3] + delta_pos

        # 工作空间限制
        target_pos[0] = np.clip(target_pos[0],
                                config.WORKSPACE_LIMITS['x'][0],
                                config.WORKSPACE_LIMITS['x'][1])
        target_pos[1] = np.clip(target_pos[1],
                                config.WORKSPACE_LIMITS['y'][0],
                                config.WORKSPACE_LIMITS['y'][1])
        target_pos[2] = np.clip(target_pos[2],
                                config.WORKSPACE_LIMITS['z'][0],
                                config.WORKSPACE_LIMITS['z'][1])

        # 旋转部分
        if self.ENABLE_ROTATION:
            current_vr_rot = self.vr_rotation_to_robot(vr_qx, vr_qy, vr_qz, vr_qw)

            if self.ROTATION_MODE == "incremental":
                R_relative = current_vr_rot @ self.calibration_vr_rot_matrix.T
                axis_angle = self.rotation_matrix_to_axis_angle(R_relative)
                axis_angle[3] *= self.ROTATION_SCALE
                R_relative_scaled = self.axis_angle_to_rotation_matrix(axis_angle)
                R_target = self.calibration_robot_rot_matrix @ R_relative_scaled
            else:
                R_target = current_vr_rot

            if self.last_sent_pose is not None:
                last_R = self.rpy_to_rotation_matrix(
                    self.last_sent_pose[3],
                    self.last_sent_pose[4],
                    self.last_sent_pose[5]
                )
                R_diff = R_target @ last_R.T
                angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))

                if angle_diff < self.ROTATION_DEADZONE:
                    target_rpy = self.last_sent_pose[3:6]
                else:
                    target_rpy = self.rotation_matrix_to_rpy(R_target)
            else:
                target_rpy = self.rotation_matrix_to_rpy(R_target)
        else:
            target_rpy = self.calibration_robot_pose[3:6]

        # 组合位姿
        target_pose = list(target_pos) + list(target_rpy)

        return target_pose

    def apply_filter(self, pose, dt=None):
        """按单调时钟间隔平滑目标，再限制目标平移速度。"""
        dt = 1.0 / config.CONTROL_FREQUENCY if dt is None else float(dt)
        if not np.isfinite(dt) or dt < 0:
            raise ValueError('Invalid control interval')
        used_dt = min(dt, config.TELEOP_MAX_CONTROL_DT_S)
        target = np.asarray(pose, dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError('Invalid target pose')
        previous = getattr(self, '_filtered_pose', None)
        if previous is None:
            previous = getattr(self, 'last_sent_pose', None)
        previous = target.copy() if previous is None else np.asarray(previous, dtype=float)
        position_alpha = -np.expm1(-used_dt / config.TELEOP_POSITION_FILTER_TAU_S)
        rotation_alpha = -np.expm1(-used_dt / config.TELEOP_ROTATION_FILTER_TAU_S)
        smooth_position = previous[:3] + position_alpha * (target[:3] - previous[:3])
        anchor = getattr(self, 'last_sent_pose', None)
        anchor = previous[:3] if anchor is None else np.asarray(anchor[:3], dtype=float)
        step = smooth_position - anchor
        distance = np.linalg.norm(step)
        limit = config.TELEOP_MAX_LINEAR_SPEED_MM_S * used_dt
        limited = distance > limit
        position = anchor + step * (limit / distance) if limited else smooth_position

        # Shortest-path quaternion SLERP avoids Euler branch discontinuities.
        def quaternion(rpy):
            cr, cp, cy = np.cos(np.asarray(rpy) / 2)
            sr, sp, sy = np.sin(np.asarray(rpy) / 2)
            return np.array([sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                             cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy])
        q0, q1 = quaternion(previous[3:]), quaternion(target[3:])
        dot = float(np.dot(q0, q1))
        if dot < 0:
            q1, dot = -q1, -dot
        dot = np.clip(dot, 0, 1)
        if dot > 1 - 1e-10:
            q = (1 - rotation_alpha) * q0 + rotation_alpha * q1
        else:
            angle = np.arccos(dot)
            q = (np.sin((1 - rotation_alpha) * angle) * q0
                 + np.sin(rotation_alpha * angle) * q1) / np.sin(angle)
        rotation = self.rotation_matrix_to_rpy(self.quaternion_to_rotation_matrix(*q))
        result = list(position) + list(rotation)
        self._filtered_pose = result
        self._last_filter_metrics = {
            'elapsed_s': dt, 'used_dt_s': used_dt,
            'position_alpha': float(position_alpha), 'rotation_alpha': float(rotation_alpha),
            'position_rate_limited': bool(limited),
            'smoothed_position_before_rate_limit_mm': smooth_position.tolist(),
        }
        return result

    def control_gripper(self):
        """控制夹爪开关"""
        if config.GRIPPER_TYPE == 0:
            return

        try:
            if config.GRIPPER_TYPE == 1:
                pos = 850 if self.gripper_open else 0
                self.arm.set_gripper_position(pos, wait=False)
            elif config.GRIPPER_TYPE == 2:
                pos = 84 if self.gripper_open else 0
                self.arm.set_gripper_g2_position(pos, speed=225, wait=False)
            elif config.GRIPPER_TYPE == 3:
                pos = 150 if self.gripper_open else 71
                self.arm.set_bio_gripper_g2_position(pos, speed=4500, wait=False)
        except Exception as e:
            print(f"✗ 夹爪控制失败: {e}")

    def _set_joint_servo(self, angles):
        """关节伺服命令，兼容不同 xArm SDK 版本"""
        q = [float(x) for x in angles]
        if hasattr(self.arm, "set_servo_angle_j"):
            return self._checked_call(self.arm.set_servo_angle_j, q, is_radian=True)
        try:
            return self._checked_call(self.arm.set_servo_angle, angle=q, is_radian=True, wait=False)
        except TypeError:
            return self._checked_call(self.arm.set_servo_angle, q, is_radian=True, wait=False)

    def set_gripper_continuous(self, value_01):
        if config.GRIPPER_TYPE != 1:
            raise RuntimeError('Only type 1 gripper pulse semantics audited')
        if not np.isfinite(value_01):
            raise ValueError('Nonfinite gripper target')
        pulse = int(round(float(np.clip(value_01, 0, 1)) * 850))
        started = time.monotonic_ns()
        code = self._checked_call(self.arm.set_gripper_position, pulse, wait=False)
        ended = time.monotonic_ns()
        self._check_reset_result(code, 'gripper submission')
        with self._target_joints_lock:
            self.last_sent_gripper = pulse / 850.0
            self._gripper_timing = {'submit_ns': started, 'return_ns': ended,
                                    'sdk_code': code, 'pulse': pulse, 'execution_confirmed': False}

    def _clear_teleop_reference(self):
        """归位后禁止复用上一段的标定、滤波和目标。"""
        self.is_calibrated = False
        self.calibration_robot_pose = None
        self.calibration_vr_pos = None
        self.calibration_vr_rot_matrix = None
        self.calibration_robot_rot_matrix = None
        self.last_sent_pose = None
        self.last_command_time = None
        self._filtered_pose = None
        self.last_gripper_cmd_time = None
        self.last_sent_gripper = None
        self._command_timing = self._gripper_timing = self._cached_timing = None
        self.position_buffer.clear()
        self.rotation_buffer.clear()
        self.joystick_joint_offsets = np.zeros(6)
        with self._target_pose_lock:
            self._latest_target_pose = None
        with self._target_joints_lock:
            self._latest_target_joints = None
        with self._cache_lock:
            self._cached_state = None
            self._cached_action = None
        # Keep physical button edge states across reset: a held button is not a new press.
        self._require_trigger_release = True

    @staticmethod
    def _check_reset_result(code, operation):
        if code != 0:
            raise RuntimeError(f"{operation}失败，SDK返回码: {code}")

    def _save_finished_episode(self):
        # 与单帧采样互斥，避免保存/清空缓冲时录制线程仍在写入。
        with self._episode_lock:
            self.data_collector.stop_episode(success=None, reason="operator_b_end")

    async def _return_to_saved_home(self):
        """启动和回合结束共用归位，并核对完成状态及实际反馈。"""
        self._trace('home_start', saved_home_mm_degrees=config.INITIAL_POSITION)
        initial = np.asarray(config.INITIAL_POSITION, dtype=float).copy()
        initial[3:] = np.deg2rad(initial[3:])
        # 不自动清错或重新使能；设备异常时应停下并人工检查。
        self._check_reset_result(self._checked_call(self.arm.set_mode, 0), "切换位置模式")
        self._check_reset_result(self._checked_call(self.arm.set_state, 0), "启动位置模式")
        self._check_reset_result(self._checked_call(self.arm.set_position,
            *initial, is_radian=True, wait=False,
            speed=config.SERVO_SPEED, mvacc=config.SERVO_ACCELERATION,
        ), "发送归位指令")

        deadline = time.monotonic() + 60.0
        while True:
            # 让接收循环及时消费消息，避免归位后回放积压的手柄动作。
            await asyncio.sleep(0.1)
            if self._fault_reason:
                raise RuntimeError(self._fault_reason)
            code, state = self.arm.get_state()
            self._check_reset_result(code, "读取机械臂状态")
            code, errors = self.arm.get_err_warn_code()
            self._check_reset_result(code, "读取机械臂错误")
            if errors[0] != 0 or state in (4, 5):
                raise RuntimeError(f"归位被中止: state={state}, error={errors[0]}")
            code, pose = self.arm.get_position(is_radian=True)
            self._check_reset_result(code, "读取归位位置")
            self._trace('home_feedback', tcp_pose_mm_radians=list(pose), controller_state=state)
            delta = np.asarray(pose, dtype=float) - initial
            angle_delta = (delta[3:] + np.pi) % (2 * np.pi) - np.pi
            if (state == 2 and np.max(np.abs(delta[:3])) <= 2.0
                    and np.max(np.abs(angle_delta)) <= np.deg2rad(0.5)):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("归位超过60秒，已停止，请检查机械臂")

        self._check_reset_result(self._checked_call(self.arm.set_mode, 1), "恢复伺服模式")
        self._check_reset_result(self._checked_call(self.arm.set_state, 0), "准备重新标定")
        self._trace('home_complete')

    async def _finish_episode_and_reset(self):
        """保存结束后归位；接收循环继续消费并丢弃期间的手柄消息。"""
        try:
            save_task = asyncio.create_task(asyncio.to_thread(self._save_finished_episode))
            try:
                await asyncio.shield(save_task)
            except asyncio.CancelledError:
                # 退出时等正在进行的保存完成，不再发起归位。
                await save_task
                raise

            if self._fault_reason:
                raise RuntimeError(self._fault_reason)
            print("录制已结束，正在归位；期间忽略手柄输入...")
            await self._return_to_saved_home()
            self._clear_teleop_reference()
            self.is_paused = False
            print("✓ 保存并归位完成。按B归位并自动标定后开始下一条；按A将刚保存的一条移到回收站。")
        except (Exception, asyncio.CancelledError) as exc:
            self._reset_failed = True
            self.is_paused = True
            self.is_calibrated = False
            try:
                self.arm.set_state(4)
            except Exception:
                pass
            print(f"归位未完成，控制保持停止，请检查后重新启动程序: {exc}")
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            self._resetting = False

    async def _start_episode_from_home(self):
        try:
            print('B开始：先归位，完成后自动标定并开始录制；归位期间可把手柄放到舒适位置。')
            await self._return_to_saved_home()
            if self._fault_reason:
                raise RuntimeError(self._fault_reason)
            self._clear_teleop_reference()
            latest = self._latest_right_input
            if latest is None or time.monotonic_ns() - latest[0] > 250_000_000:
                raise RuntimeError('归位后缺少新鲜手柄数据，未开始录制')
            received_ns, right = latest
            pos, ori = right['position'], right['orientation']
            if np.linalg.norm([ori[k] for k in ('x','y','z','w')]) < 1e-6:
                raise RuntimeError('归位后手柄姿态无效，未开始录制')
            if not self.calibrate(pos['x'], pos['y'], pos['z'], ori['x'], ori['y'], ori['z'], ori['w']):
                raise RuntimeError('自动标定失败，未开始录制')
            if time.monotonic_ns() - received_ns > 250_000_000:
                raise RuntimeError('自动标定期间手柄数据过期，未开始录制')
            self._require_trigger_release = False
            with self._episode_lock:
                if not self.data_collector.start_episode():
                    raise RuntimeError('未能开始新回合')
            self.is_paused = False
            self._trace('episode_start', automatic_calibration=True, anchor_input_ns=received_ns,
                        episode_index=self.data_collector.episode_index)
            print('🔴 已归位并自动标定，正式开始录制；现在可移动手柄完成任务。')
        except (Exception, asyncio.CancelledError) as exc:
            self._latch_fault(f'开始回合失败: {exc}')
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            self._resetting = False

    async def _trash_previous_episode(self):
        try:
            result = await asyncio.to_thread(self.data_collector.trash_last_episode)
            if result is None:
                print('本次启动没有可回收的上一条回合。')
            else:
                self._trace('episode_trashed', **result)
                print('✓ 整条回合已移到系统回收站，可从文件管理器恢复：' + result['original_episode'])
        except Exception as exc:
            try:
                self._trace('trash_error', error=str(exc))
            except Exception as trace_error:
                self._latch_fault(f'trash diagnostic: {trace_error}')
            print(f'回收操作未正常完成，请检查原目录和系统回收站；不会永久删除：{exc}')
        finally:
            self._trashing = False
            self.is_paused = bool(self._fault_reason)

    def _handle_episode_buttons(self, right):
        buttons = right.get('buttons', [])
        a = len(buttons) > 4 and bool(buttons[4].get('pressed', False))
        b = len(buttons) > 5 and bool(buttons[5].get('pressed', False))
        a_edge, b_edge = a and not self.last_a_button_state, b and not self.last_b_button_state
        self.last_a_button_state, self.last_b_button_state = a, b
        if self._resetting or self._trashing or self._reset_failed:
            if a_edge:
                print('正在保存/归位/故障处理中，忽略A；完成后松开再按。')
            return True
        if a and b:
            return True
        if a_edge:
            if self.data_collector.is_recording:
                print('正在录制，A不回收数据；先按B保存并等待归位。')
                return True
            if self.data_collector.last_saved_episode is None:
                print('本次启动没有可回收的上一条回合。')
                return True
            self.is_paused = True
            self._clear_teleop_reference()
            self._trashing = True
            self._trash_task = asyncio.create_task(self._trash_previous_episode())
            return True
        if b_edge:
            if (not self.data_collector.is_recording and self.data_collector.max_episodes is not None
                    and self.data_collector.retained_episode_count >= self.data_collector.max_episodes):
                print('练习配额已满；可按A回收刚保存的回合，或退出。')
                return True
            self.is_paused = True
            self.is_calibrated = False
            self._resetting = True
            action = (self._finish_episode_and_reset if self.data_collector.is_recording
                      else self._start_episode_from_home)
            self._reset_task = asyncio.create_task(action())
            return True
        return False

    async def handle_controller_data(self, websocket):
        """处理手柄数据 + 数据收集"""
        if self._client_active:
            await websocket.close(code=1013, reason='Single controller only')
            return
        self._client_active = True
        client_ip = websocket.remote_address[0]
        print(f"Quest 3 已连接: {client_ip}")

        if self.start_time is None:
            self.start_time = time.time()

        try:
            iterator = websocket.__aiter__()
            while True:
                try:
                    message = await asyncio.wait_for(iterator.__anext__(), timeout=1.0)
                except StopAsyncIteration:
                    break
                try:
                    received_ns = time.monotonic_ns()
                    input_interval_ns = (None if self._last_input_ns is None
                                         else received_ns - self._last_input_ns)
                    self._last_input_ns = received_ns
                    data = json.loads(message)
                    self._trace('input', receive_ns=received_ns, interval_ns=input_interval_ns,
                                packet=data, resetting=self._resetting,
                                faulted=bool(self._fault_reason), calibrated=self.is_calibrated,
                                paused=self.is_paused, recording=self.data_collector.is_recording)
                    if not (self._resetting or self._trashing or self._reset_failed):
                        self.frame_count += 1
                        self.data_collector.step_count += 1

                    if 'right' not in data or data['right'] is None:
                        self._latest_right_input = None
                        if self.is_calibrated:
                            self._latch_fault('right_controller_tracking_lost')
                        continue

                    right = data['right']
                    pos = right['position']
                    ori = right['orientation']

                    # 读取模拟量 trigger (0~1)，Unity 端发的是 right.trigger
                    buttons = right.get('buttons', [])
                    trigger_val = float(right.get('trigger', buttons[0].get('value', 0.0) if buttons else 0.0))
                    if not np.isfinite([*pos.values(), *ori.values(), trigger_val]).all():
                        raise ValueError('Nonfinite controller pose/trigger')
                    self._latest_right_input = (received_ns, right)
                    if self._handle_episode_buttons(right):
                        continue
                    trigger_val = np.clip(trigger_val, 0.0, 1.0)
                    if trigger_val < self.gripper_deadzone:
                        trigger_val = 0.0
                    
                    desired = 1.0 -trigger_val
                    gamma = 2.0
                    desired = desired ** gamma

                    # 简单低通滤波，让夹爪更稳
                    self.gripper_target = (1 - self.gripper_alpha) * self.gripper_target + self.gripper_alpha * desired

                    # 读取 Joystick（兼容 Unity "thumbstick" 和 WebXR "axes" 两种格式）
                    joy_x, joy_y = 0.0, 0.0
                    thumbstick = right.get('thumbstick')
                    if thumbstick is not None and isinstance(thumbstick, dict):
                        joy_x = float(thumbstick.get('x', 0.0))
                        joy_y = float(thumbstick.get('y', 0.0))
                    else:
                        axes = right.get('axes', [])
                        if len(axes) >= 4:
                            joy_x, joy_y = float(axes[2]), float(axes[3])
                        elif len(axes) >= 2:
                            joy_x, joy_y = float(axes[0]), float(axes[1])

                    if abs(joy_x) < self.JOYSTICK_DEADZONE:
                        joy_x = 0.0
                    if abs(joy_y) < self.JOYSTICK_DEADZONE:
                        joy_y = 0.0

                    # 调试输出
                    if self.frame_count % 50 == 0 and config.VERBOSE:
                        robot_pos = self.vr_to_robot_position(pos['x'], pos['y'], pos['z'])
                        print(f"\n[Frame {self.frame_count}]")
                        print(f"  VR位置: X={pos['x']:+.3f} Y={pos['y']:+.3f} Z={pos['z']:+.3f}")
                        print(f"  →Robot: X={robot_pos[0]:+.1f} Y={robot_pos[1]:+.1f} Z={robot_pos[2]:+.1f}")

                    # 归位后必须先松开Trigger，避免长按触发自动重新标定。
                    if self._require_trigger_release:
                        buttons = right.get('buttons', [])
                        if buttons and not buttons[0].get('pressed', False):
                            self._require_trigger_release = False
                        continue

                    # 标定检查
                    if not self.is_calibrated:
                        if 'buttons' in right and len(right['buttons']) > 0:
                            if right['buttons'][0].get('pressed', False):
                                self.calibrate(pos['x'], pos['y'], pos['z'],
                                             ori['x'], ori['y'], ori['z'], ori['w'])
                        continue

                    # 计算目标位姿
                    target_pose = self.compute_target_pose(
                        pos['x'], pos['y'], pos['z'],
                        ori['x'], ori['y'], ori['z'], ori['w']
                    )

                    if target_pose is None:
                        continue

                    # ========== 更新共享目标位姿（供独立录制线程使用） ==========
                    with self._target_pose_lock:
                        self._latest_target_pose = list(target_pose)

                    self._trace('target', receive_ns=received_ns,
                                desired_pose_mm_radians=list(target_pose), paused=self.is_paused)
                    now = time.monotonic()

                    # 频率控制
                    if (not self.is_paused) and (
                        self.last_command_time is None or
                        now - self.last_command_time >= 1.0/config.CONTROL_FREQUENCY
                    ):
                        control_dt = (1.0 / config.CONTROL_FREQUENCY if self.last_command_time is None
                                      else now - self.last_command_time)
                        desired_pose = list(target_pose)
                        target_pose = self.apply_filter(target_pose, control_dt)
                        with self._target_pose_lock:
                            self._latest_target_pose = list(target_pose)

                        # Joystick 增量累加到末端关节偏移
                        self.joystick_joint_offsets[4] += joy_y * self.JOYSTICK_SCALE
                        self.joystick_joint_offsets[5] += joy_x * self.JOYSTICK_SCALE

                        # Only publish SDK-accepted joint commands; never substitute unexecuted IK.
                        try:
                            ik_code, ik_angles = self.arm.get_inverse_kinematics(
                                target_pose, input_is_radian=True, return_is_radian=True)
                            self._check_reset_result(ik_code, 'IK')
                            modified_angles = list(ik_angles)
                            for i in range(6):
                                modified_angles[i] += self.joystick_joint_offsets[i]
                            if not np.isfinite(modified_angles).all():
                                raise ValueError('Nonfinite joint command')
                            submitted = time.monotonic_ns()
                            ret = self._set_joint_servo(modified_angles)
                            returned = time.monotonic_ns()
                            self._check_reset_result(ret, 'joint submission')
                            self._trace('command', receive_ns=received_ns,
                                        control_tick_ns=int(now * 1e9),
                                        desired_pose_mm_radians=desired_pose,
                                        target_pose_mm_radians=list(target_pose),
                                        target_joints_radians=modified_angles,
                                        submit_ns=submitted, return_ns=returned, sdk_code=ret,
                                        filter=self._last_filter_metrics, execution_confirmed=False,
                                        joystick_joint_offsets_radians=self.joystick_joint_offsets.tolist())
                            with self._target_joints_lock:
                                self._latest_target_joints = modified_angles
                                self._command_timing = {'submit_ns': submitted, 'return_ns': returned,
                                                        'sdk_code': ret, 'execution_confirmed': False}
                        except Exception as e:
                            self._latch_fault(f'command: {e}')
                            # Persist context only after requesting stop. This records
                            # submitted targets, not evidence of physical execution.
                            diagnostic = {
                                'reason': str(e), 'time_ns': time.time_ns(),
                                'target_pose_mm_radians': list(target_pose),
                                'last_accepted_pose_mm_radians': self.last_sent_pose,
                                'joystick_joint_offsets_radians': self.joystick_joint_offsets.tolist(),
                                'right_controller': right,
                                'execution_confirmed': False,
                            }
                            try:
                                detail = json.dumps(diagnostic, ensure_ascii=False, indent=2)
                                print('Command fault context: ' + detail, flush=True)
                                (self.data_collector.dataset_path / 'command_fault.json').write_text(
                                    detail + '\n', encoding='utf-8')
                            except Exception as diagnostic_error:
                                print(f'Cannot save command fault context: {diagnostic_error}', flush=True)
                            continue

                        self.last_command_time = now
                        self.last_sent_pose = target_pose
                    
                    # ================== 夹爪命令频率限制（30Hz） ==================
                   
                    if (not self.is_paused) and (
                        self.last_gripper_cmd_time is None or
                        now - self.last_gripper_cmd_time >= 1.0 / self.GRIPPER_CMD_FREQ
                    ):
                        if (self.last_sent_gripper is None or
                                abs(self.gripper_target - self.last_sent_gripper) >= self.GRIPPER_HOLD_DEADBAND):
                            self.set_gripper_continuous(self.gripper_target)
                        self.last_gripper_cmd_time = now


                    # # Trigger控制夹爪
                    # if 'buttons' in right and len(right['buttons']) > 0:
                    #     trigger = right['buttons'][0].get('pressed', False)
                    #     if trigger and not self.last_trigger_state:
                    #         self.gripper_open = not self.gripper_open
                    #         self.control_gripper()
                    #     self.last_trigger_state = trigger

                except Exception as e:
                    self._latch_fault(f"controller: {e}")
                    if config.VERBOSE:
                        import traceback
                        traceback.print_exc()

        except asyncio.TimeoutError:
            self._latch_fault('quest_message_timeout_1s')
        except websockets.exceptions.ConnectionClosed:
            print(f"Quest 3 断开连接")
        finally:
            self._client_active = False
            self._latch_fault("quest_disconnected")
            if self._reset_task and not self._reset_task.done():
                self._reset_task.cancel()

    async def run_server(self):
        """运行WebSocket服务器"""
        port = 8765
        print(f"\nWebSocket服务器启动: 端口 {port}")

        if not self.is_calibrated:
            print("\n等待B开始：先归位，再自动标定并录制。")
            print("Trigger仍可用于先手动标定试动，但B开始不需要预先标定。\n")

        print("数据收集说明:")
        print("  • 标定后，按B键开始录制Episode")
        print("  • 执行任务操作")
        print("  • 再次按B键结束录制并归位；松开再按Trigger重新标定")
        print("  • Joystick X/Y: 增减末端关节 (Joint 5/6)\n")

        async with websockets.serve(self.handle_controller_data, "0.0.0.0", port):
            await asyncio.Future()

    def shutdown(self):
        """关闭"""
        self._latch_fault(self._fault_reason or "shutdown")
        # 停止后台线程
        self._record_running = False
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=2.0)
            print("✓ 录制线程已停止")
        if self._state_ik_thread and self._state_ik_thread.is_alive():
            self._state_ik_thread.join(timeout=2.0)
            print("✓ 状态缓存线程已停止")

        self._monitor_running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
            print("✓ 调试监控线程已停止")

        self._camera_running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
            print("✓ 相机采集线程已停止")

        # 停止录制
        if self.data_collector.is_recording:
            print("\n检测到正在录制，自动保存...")
            try:
                with self._episode_lock:
                    self.data_collector.stop_episode(success=None, reason=self._fault_reason or "interrupted")
            except Exception as exc:
                print(f"Save failed; retained memory and partial bundle: {exc}", flush=True)

        # 打印统计
        self.data_collector.print_statistics()

        # 释放相机
        if self.camera_manager:
            self.camera_manager.release()

        # 停止机械臂
        if self.arm:
            try:
                self.arm.set_state(4)
                print("机械臂已暂停")
            except:
                pass

        if self.start_time:
            duration = time.time() - self.start_time
            print(f"运行时长: {duration:.1f}s, 帧数: {self.frame_count}")
        if self._diagnostics is not None:
            try:
                self._diagnostics.close()
            except Exception as exc:
                print(f'Diagnostic close failed: {exc}', flush=True)


async def main():
    """主函数"""
    import sys

    import os
    if os.environ.get('UF850_MANUAL_COLLECTION') != '1':
        raise RuntimeError('Use the reviewed manual launch.py collection entry')
    # 解析命令行参数
    robot_ip = sys.argv[1] if len(sys.argv) > 1 else None
    dataset_path = sys.argv[2] if len(sys.argv) > 2 else None

    controller = UF850WebXRControlWithDataCollection(
        robot_ip=robot_ip,
        dataset_path=dataset_path,
        preparation=json.loads(os.environ.get('UF850_PREPARATION_PROFILE', 'null')),
    )

    try:
        controller.start_diagnostics()
        if not await controller.initialize_robot():
            return 1
        await controller.run_server()
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        try:
            if controller._reset_task and not controller._reset_task.done():
                controller._reset_task.cancel()
                try:
                    await controller._reset_task
                except asyncio.CancelledError:
                    pass
            if controller._trash_task is not None:
                await asyncio.shield(controller._trash_task)
        finally:
            controller.shutdown()

    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n退出")
