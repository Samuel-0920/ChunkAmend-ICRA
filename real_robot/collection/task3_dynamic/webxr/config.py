#!/usr/bin/env python3
"""
UF850 VR 遥操作配置文件
"""

import os

# ==================== 机械臂连接配置 ====================
ROBOT_IP = "192.168.1.237"

# ==================== 控制模式配置 ====================
# 1: Servo motion mode (伺服模式，实时控制) - 推荐用于遥操作
# 7: Cartesian online trajectory planning mode (在线轨迹规划模式)
ROBOT_MODE = 1  # 修改为伺服模式,更适合实时遥操作

# ==================== 工作空间限制（相对于 base，单位：毫米） ====================
# 根据实际情况调整，确保安全
WORKSPACE_LIMITS = {
    'x': (200, 700),     # 前后范围：200-700mm
    'y': (-400, 400),    # 左右范围：±400mm
    'z': (100, 600)      # 上下范围：100-600mm
}

# ==================== 安全参数 ====================
# 旧版本兼容参数；当前遥操使用下方按时间计算的TELEOP_MAX_LINEAR_SPEED_MM_S。
MAX_DELTA_POSITION = 8.0  # 从 30 降到 8，防止一下移动太远

# 旧版本参数；当前控制链不使用此项作为旋转速度上限。
MAX_DELTA_ROTATION = 0.1  # 从 0.2 降到 0.1，减少旋转幅度

# 紧急停止按钮组合（同时按下 grip 和 trigger）
EMERGENCY_STOP_BUTTONS = ['grip', 'trigger']

# ==================== 控制参数 ====================
# 控制频率（Hz）- 伺服模式推荐 50-100Hz
CONTROL_FREQUENCY = 100

# 伺服模式参数 - 降低速度和加速度，更平滑、更安全
SERVO_SPEED = 200      # mm/s (从 500 降到 200)
SERVO_ACCELERATION = 150  # mm/s² (从 300 降到 150)

# 轨迹规划模式参数（降低速度以提高安全性）
TRAJECTORY_SPEED = 300    # mm/s（从 1000 降到 300）
TRAJECTORY_ACCELERATION = 1000  # mm/s²（从 5000 降到 1000）

# ==================== VR 参数 ====================
# VR 到机械臂空间的缩放因子
# 数值越大，VR 中需要更大的移动才对应真实空间的位移（更精细）
#SCALE_FACTOR = 5
SCALE_FACTOR = 1.0 / 1.2  # 平移1.2倍：手移动10cm，对应目标12cm

# 遥操目标平移限速与按时间滤波；不等于硬件实测速度保证。
TELEOP_MAX_LINEAR_SPEED_MM_S = 200.0
TELEOP_POSITION_FILTER_TAU_S = 0.05
TELEOP_ROTATION_FILTER_TAU_S = 0.05
TELEOP_MAX_CONTROL_DT_S = 0.05  # 消息停顿不积累大步长
TELEOP_TCP_DIAGNOSTIC_HZ = 30.0

# 旧版本兼容参数；当前遥操使用TELEOP_POSITION_FILTER_TAU_S。
# 0: 完全平滑（延迟大），1: 无平滑（可能抖动）
POSITION_SMOOTHING = 0.5  # 从 0.3 增加到 0.5，更平滑的运动

# ==================== 夹爪配置 ====================
# 0: 无夹爪
# 1: xArm Gripper (标准夹爪)
# 2: xArm Gripper G2 (第二代夹爪)
# 3: BIO Gripper G2 (仿生夹爪)
GRIPPER_TYPE = 1  # 修改为你的夹爪类型 (1, 2, 或 3)

# ==================== 相机配置 ====================
MAIN_CAM_SERIAL = os.environ.get("MAIN_RS_SERIAL")
WRIST_CAM_SERIAL = os.environ.get("WRIST_RS_SERIAL")

# 兼容现有 RealSense 采集代码中的变量名
MAIN_RS_SERIAL = MAIN_CAM_SERIAL
WRIST_RS_SERIAL = WRIST_CAM_SERIAL

# OpenCV / V4L2 fallback: RealSense RGB color nodes
MAIN_CAM_ID = os.environ.get("MAIN_CAM_DEVICE")   # 启动器按已核验序列号注入前置彩色节点
WRIST_CAM_ID = os.environ.get("WRIST_CAM_DEVICE")  # 启动器按已核验序列号注入腕部彩色节点

# ==================== fsr配置 ====================
FSR_PORT = '/dev/ttyACM0'
FSR_MODE = 1
FSR_K_PER_STEP = 5

# ==================== 初始位置配置 ====================
# 机械臂初始安全位置 [x, y, z, roll, pitch, yaw]
# 单位：毫米和度
INITIAL_POSITION = [332.808868, 69.979057, 450.30603, 179.994118, -0.00149, 0.000115]
# 用户2026-09-09查看任务一、二起点后选择任务一起点；见 ../TASK_HOME.json。

# ==================== 调试配置 ====================
# 是否打印详细日志
VERBOSE = True

# 是否显示实时位姿
SHOW_REALTIME_POSE = False

# ==================== 标定配置 ====================
# 标定数据保存路径
CALIBRATION_FILE = "calibration_data_uf850.json"
