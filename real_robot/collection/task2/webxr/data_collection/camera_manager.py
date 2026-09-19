#!/usr/bin/env python3
"""
相机管理模块 - 支持普通 USB 摄像头 (通过 OpenCV VideoCapture)
"""

import numpy as np
import cv2
import time


class CameraManager:
    """
    USB 摄像头管理器

    通过 cv2.VideoCapture 驱动普通 USB 摄像头 (如 Logitech C930e / C310)。
    接口与旧版 RealSense 管理器保持一致。
    """

    def __init__(self, camera_ids=None, width=640, height=480, fps=30):
        """
        Args:
            camera_ids: dict, 相机名称 → 设备ID 的映射
                        例如 {'wrist': 0, 'front': 2}
                        若为 None 则默认使用 {0, 2}
            width: 图像宽度
            height: 图像高度
            fps: 目标帧率
        """
        if camera_ids is None:
            raise ValueError('Explicit, identity-checked front and wrist devices required')

        if set(camera_ids) != {'front', 'wrist'} or len(set(map(str, camera_ids.values()))) != 2:
            raise ValueError('Two distinct front and wrist devices required')
        self.last_timing = {}
        self.sequence = 0
        self.camera_ids = camera_ids
        self.width = width
        self.height = height
        self.fps = fps

        self.camera_names = list(camera_ids.keys())
        self.caps = {}

        try:
            self._initialize_cameras()
        except Exception:
            self.release()
            raise

    def _initialize_cameras(self):
        """通过 OpenCV 打开所有 USB 摄像头"""
        print(f"\n{'='*60}")
        print(f"初始化 {len(self.camera_ids)} 个 USB 摄像头")

        for name, dev_id in self.camera_ids.items():
            try:
                dev_id_int = int(dev_id)
            except (ValueError, TypeError):
                dev_id_int = dev_id

            # CAP_ANY can select FFmpeg, which ignores V4L2 format setters.
            cap = cv2.VideoCapture(dev_id_int, cv2.CAP_V4L2)
            self.caps[name] = cap
            if not cap.isOpened():
                print(f"✗ 相机 [{name}] 设备 {dev_id} 打开失败")
                raise RuntimeError(f'Camera {name} failed to open')

            settings = (
                (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'), 'YUYV'),
                (cv2.CAP_PROP_FRAME_WIDTH, self.width, 'width'),
                (cv2.CAP_PROP_FRAME_HEIGHT, self.height, 'height'),
                (cv2.CAP_PROP_FPS, self.fps, 'fps'),
            )
            for prop, value, label in settings:
                if not cap.set(prop, value):
                    raise RuntimeError(f'Camera {name}: V4L2 rejected {label}={value}')

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            if ((actual_w, actual_h) != (self.width, self.height)
                    or not np.isfinite(actual_fps) or abs(actual_fps - self.fps) > 0.5):
                raise RuntimeError(
                    f'Camera {name}: requested {self.width}x{self.height}@{self.fps}, '
                    f'got {actual_w}x{actual_h}@{actual_fps}')

            for _ in range(10):
                ret, frame = cap.read()
                if (ret and frame is not None and frame.dtype == np.uint8
                        and frame.shape == (self.height, self.width, 3)):
                    break
            else:
                raise RuntimeError(f'Camera {name}: no valid frame during startup')

            print(f"✓ 相机 [{name}]: {dev_id}  "
                  f"{actual_w}x{actual_h} @ {actual_fps:g}fps (V4L2)")

        print(f"{'='*60}\n")

        if len(self.caps) == 0:
            raise RuntimeError(
                "❌ 所有摄像头打开失败！\n"
                "   请检查:\n"
                "   1. 设备是否已连接 (ls /dev/video*)\n"
                "   2. 设备ID是否正确\n"
                "   3. 是否有其他程序占用摄像头\n"
            )

    def get_frames(self):
        """Sequential host reads; timestamps are NOT hardware exposure times."""
        frames, timing = {}, {}
        for name, cap in self.caps.items():
            started = time.monotonic_ns()
            ret, frame = cap.read()
            ended = time.monotonic_ns()
            if (not ret or frame is None or frame.dtype != np.uint8
                    or frame.shape != (self.height, self.width, 3)):
                raise RuntimeError(f'Invalid/missing camera frame: {name}')
            frames[name] = frame
            timing[name] = {'read_start_ns': started, 'read_end_ns': ended,
                            'sequence': self.sequence}
        if set(frames) != {'front', 'wrist'}:
            raise RuntimeError('Missing required camera')
        self.last_timing = timing
        self.sequence += 1
        return frames

    def get_num_cameras(self):
        """获取实际可用的相机数量"""
        return len(self.caps)

    def release(self):
        """释放所有摄像头资源"""
        for name, cap in self.caps.items():
            try:
                cap.release()
            except Exception:
                pass
        self.caps.clear()
        print("✓ 相机资源已释放")

    def __del__(self):
        self.release()


if __name__ == '__main__':
    print("测试CameraManager (USB摄像头)...")

    cam_manager = CameraManager(
        camera_ids={'wrist': 0, 'front': 2},
        fps=15,
    )

    print(f"\n可用相机数: {cam_manager.get_num_cameras()}")

    print("\n测试采集10帧...")
    for i in range(10):
        start = time.time()
        frames = cam_manager.get_frames()
        elapsed = time.time() - start

        print(f"帧{i}: ", end='')
        for name, frame in frames.items():
            print(f"{name}={frame.shape} ", end='')
        print(f"耗时={elapsed*1000:.1f}ms")

        time.sleep(0.067)

    cam_manager.release()
    print("\n测试完成！")
