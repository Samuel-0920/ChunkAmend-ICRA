#!/usr/bin/env python3
"""
Episode数据保存模块 - LeRobot V2.0格式
"""

import json
import numpy as np
import cv2
from pathlib import Path


class EpisodeRecorder:
    """
    负责将episode数据保存为LeRobot V2.0格式

    包括:
    1. Parquet文件 (低维数据: state, action, timestamp等)
    2. MP4视频 (高维数据: images)
    3. Meta文件 (episodes.jsonl, tasks.jsonl, info.json)
    """

    def __init__(self, dataset_path, episode_index, fps=30):
        """
        初始化Episode记录器

        Args:
            dataset_path: 数据集根目录
            episode_index: Episode索引号
            fps: 视频帧率 (默认30)
        """
        self.dataset_path = Path(dataset_path)
        self.episode_index = episode_index
        self.fps = fps

    def save(self, episode_data):
        """
        保存完整episode

        Args:
            episode_data: dict包含:
                - 'states': List[np.ndarray] 每个(7,)
                - 'actions': List[np.ndarray] 每个(7,)
                - 'timestamps': List[float]
                - 'frames_wrist': List[np.ndarray] 每个(H,W,3)
                - 'frames_front': List[np.ndarray] 每个(H,W,3)
                - 'task_description': str
                - 'task_index': int
                - 'tactile_left':List[np.ndarray] 每个(1280,)
                - 'tactile_right':List[np.ndarray] 每个(1280,)
        """
        print(f"  正在保存Episode {self.episode_index}...")

        # 1. 保存Parquet文件
        self._save_parquet(episode_data)

        # 2. 保存MP4视频 (non-fatal: parquet is already saved)
        try:
            self._save_videos(episode_data)
        except Exception as e:
            print(f"    ⚠ 视频保存失败 (parquet已保存): {e}")

        # 3. 更新meta文件
        self._update_meta_files(episode_data)

        print(f"  ✅ Episode {self.episode_index} 保存完成")

    def _save_parquet(self, episode_data):
        """保存Parquet文件 (低维数据)"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("需要安装pandas: pip install pandas")

        num_frames = len(episode_data['states'])

        # Ensure tactile lists match num_frames (pad with zeros if short)
        tac_left = episode_data.get('tactile_left', [])
        tac_right = episode_data.get('tactile_right', [])
        if len(tac_left) < num_frames:
            pad = [np.zeros(1280, dtype=np.float32)] * (num_frames - len(tac_left))
            tac_left = list(tac_left) + pad
        if len(tac_right) < num_frames:
            pad = [np.zeros(1280, dtype=np.float32)] * (num_frames - len(tac_right))
            tac_right = list(tac_right) + pad

        # 构建DataFrame
        df_data = {
            'observation.state': episode_data['states'],
            'action': episode_data['actions'],
            'timestamp': episode_data['timestamps'],
            'observation.tactile_left': tac_left[:num_frames],
            'observation.tactile_right': tac_right[:num_frames],
            'task_index': [episode_data['task_index']] * num_frames,
            'annotation.human.action.task_description': [episode_data['task_index']] * num_frames,
            'annotation.human.validity': [1] * num_frames,  # 1=valid
            'episode_index': [self.episode_index] * num_frames,
            'frame_index': list(range(num_frames)),
            'index': list(range(num_frames)),  # 简化版，后续可以全局编号
            'rewards':[float(x) for x in episode_data.get('rewards', [0.0] * num_frames)],
            'dones':[bool(x) for x in episode_data.get('dones', [False] * (num_frames-1)+[True])],
            'masks':[float(x) for x in episode_data.get('masks', [1.0] * num_frames)],
            'mc_returns':[float(x) for x in episode_data.get('mc_returns', [0.0] * num_frames)],
            'is_intervention': [bool(x) for x in episode_data.get('is_intervention', [False] * num_frames)],
            'next.reward': [0.0] * num_frames,
            'next.done': [False] * (num_frames - 1) + [True]  # 最后一帧done=True
        }

        df = pd.DataFrame(df_data)

        # 保存路径
        parquet_path = (self.dataset_path / 'data' / 'chunk-000' /
                       f'episode_{self.episode_index:06d}.parquet')

        # 保存
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        print(f"    ✓ Parquet: {parquet_path.name} ({num_frames}行)")

    def _save_videos(self, episode_data):
        """保存MP4视频 (高维数据)"""
        for camera_name in ['wrist', 'front']:
            frames_key = f'frames_{camera_name}'

            if frames_key not in episode_data or len(episode_data[frames_key]) == 0:
                print(f"    ⚠ 跳过{camera_name}相机 (无数据)")
                continue

            frames = episode_data[frames_key]

            # 视频保存路径
            video_dir = (self.dataset_path / 'videos' / 'chunk-000' /
                        f'observation.images.{camera_name}')
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f'episode_{self.episode_index:06d}.mp4'

            # 获取帧尺寸
            height, width = frames[0].shape[:2]

            # 创建VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(video_path),
                fourcc,
                self.fps,
                (width, height)
            )

            if not out.isOpened():
                print(f"    ✗ 无法创建视频文件: {video_path}")
                continue

            # 写入所有帧
            for frame in frames:
                out.write(frame)  # BGR格式

            out.release()

            # 获取文件大小
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"    ✓ 视频: {camera_name} {len(frames)}帧 ({file_size_mb:.2f}MB)")

    def _update_meta_files(self, episode_data):
        """更新meta文件"""
        # 1. 更新episodes.jsonl
        self._update_episodes_jsonl(episode_data)

        # 2. 更新tasks.jsonl
        self._update_tasks_jsonl(episode_data)

        # 3. 更新或创建info.json
        self._update_info_json()

        # 4. 创建或更新modality.json (仅首次)
        if self.episode_index == 0:
            self._create_modality_json()

    def _update_episodes_jsonl(self, episode_data):
        """更新episodes.jsonl"""
        episodes_path = self.dataset_path / 'meta' / 'episodes.jsonl'

        # 读取已有episodes
        episodes = []
        if episodes_path.exists():
            with open(episodes_path, 'r') as f:
                for line in f:
                    episodes.append(json.loads(line))

        # 添加新episode
        new_episode = {
            'episode_index': self.episode_index,
            'tasks': [episode_data['task_description'], 'valid'],
            'length': len(episode_data['states'])
        }
        episodes.append(new_episode)

        # 保存
        with open(episodes_path, 'w') as f:
            for ep in episodes:
                f.write(json.dumps(ep) + '\n')

    def _update_tasks_jsonl(self, episode_data):
        """更新tasks.jsonl"""
        tasks_path = self.dataset_path / 'meta' / 'tasks.jsonl'

        # 读取已有tasks
        tasks = {}
        if tasks_path.exists():
            with open(tasks_path, 'r') as f:
                for line in f:
                    task = json.loads(line)
                    tasks[task['task_index']] = task['task']

        # 添加新任务 (如果不存在)
        task_index = episode_data['task_index']
        task_desc = episode_data['task_description']

        if task_index not in tasks:
            tasks[task_index] = task_desc

        # 添加"valid"标签 (如果不存在)
        if 'valid' not in tasks.values():
            tasks[len(tasks)] = 'valid'

        # 保存
        with open(tasks_path, 'w') as f:
            for idx, task in sorted(tasks.items()):
                f.write(json.dumps({'task_index': idx, 'task': task}) + '\n')

    def _update_info_json(self):
        """更新info.json"""
        info_path = self.dataset_path / 'meta' / 'info.json'

        # 统计所有episodes
        data_dir = self.dataset_path / 'data' / 'chunk-000'
        episodes = list(data_dir.glob('episode_*.parquet'))

        # 统计总帧数
        total_frames = 0
        try:
            import pandas as pd
            for ep_file in episodes:
                df = pd.read_parquet(ep_file)
                total_frames += len(df)
        except:
            total_frames = len(episodes) * 500  # 估算

        # 统计任务数
        tasks_path = self.dataset_path / 'meta' / 'tasks.jsonl'
        total_tasks = 0
        if tasks_path.exists():
            with open(tasks_path, 'r') as f:
                total_tasks = sum(1 for _ in f)

        # 构建info数据
        info = {
            'codebase_version': 'v2.0',
            'robot_type': 'UF850',
            'total_episodes': len(episodes),
            'total_frames': total_frames,
            'total_tasks': total_tasks,
            'total_videos': len(episodes) * 2,  # 2个相机
            'total_chunks': 1,
            'chunks_size': 1000,
            'fps': float(self.fps),
            'splits': {'train': '0:100'},
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
            'features': self._generate_features_schema()
        }

        # 保存
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

    def _generate_features_schema(self):
        """生成features schema"""
        return {
            'observation.images.wrist': {
                'dtype': 'video',
                'shape': [480, 640, 3],
                'names': ['height', 'width', 'channel'],
                'video_info': {
                    'video.fps': float(self.fps),
                    'video.codec': 'h264',
                    'video.pix_fmt': 'yuv420p',
                    'video.is_depth_map': False,
                    'has_audio': False
                }
            },
            'observation.images.front': {
                'dtype': 'video',
                'shape': [480, 640, 3],
                'names': ['height', 'width', 'channel'],
                'video_info': {
                    'video.fps': float(self.fps),
                    'video.codec': 'h264',
                    'video.pix_fmt': 'yuv420p',
                    'video.is_depth_map': False,
                    'has_audio': False
                }
            },
            'observation.state': {
                'dtype': 'float64',
                'shape': [8],
                'names': ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
            },
            'observation.tactile_left': {
                'dtype': 'float32',
                'shape': [1280],
                'description': 'Flat FSR sequence from left finger'
            },
            'observation.tactile_right': {
                'dtype': 'float32',
                'shape': [1280],
                'description': 'Flat FSR sequence from right finger'
            },
            'action': {
                'dtype': 'float64',
                'shape': [8],
                'names': ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
            },
            'timestamp': {
                'dtype': 'float64',
                'shape': [1]
            },
            'annotation.human.action.task_description': {
                'dtype': 'int64',
                'shape': [1]
            },
            'task_index': {
                'dtype': 'int64',
                'shape': [1]
            },
            'annotation.human.validity': {
                'dtype': 'int64',
                'shape': [1]
            },
            'episode_index': {
                'dtype': 'int64',
                'shape': [1]
            },
            'frame_index': {
                'dtype': 'int64',
                'shape': [1]
            },
            'index': {
                'dtype': 'int64',
                'shape': [1]
            },
            'rewards': {
                'dtype': 'float32',
                'shape': [1]
            },
            'dones': {
                'dtype': 'bool',
                'shape': [1]
            },
            'masks': {
                'dtype': 'float32',
                'shape': [1]
            },
            'mc_returns': {
                'dtype': 'float32',
                'shape': [1]
            },
            'is_intervention': {
                'dtype': 'bool',
                'shape': [1],
                'description': 'True if expert teleop, False if policy'
            },
            'next.reward': {
                'dtype': 'float64',
                'shape': [1]
            },
            'next.done': {
                'dtype': 'bool',
                'shape': [1]
            }
        }

    def _create_modality_json(self):
        """创建modality.json (GR00T特有配置)"""
        modality_path = self.dataset_path / 'meta' / 'modality.json'

        if modality_path.exists():
            return  # 已存在，不覆盖

        modality = {
            'state': {
                'single_arm': {
                    'start': 0,
                    'end': 7
                },
                'gripper': {
                    'start': 7,
                    'end': 8
                }
            },
            'action': {
                'single_arm': {
                    'start': 0,
                    'end': 7,
                    'absolute': True  # 绝对目标位置（非delta）
                },
                'gripper': {
                    'start': 7,
                    'end': 8,
                    'absolute': True
                }
            },
            'video': {
                'wrist': {
                    'original_key': 'observation.images.wrist'
                },
                'front': {
                    'original_key': 'observation.images.front'
                }
            },
            'tactile': {
                'left': {
                    'original_key': 'observation.tactile_left'
                },
                'right': {
                    'original_key': 'observation.tactile_right'
                },   
            },
            'annotation': {
                'human.task_description': {
                    'original_key': 'task_index'
                }
            }
        }

        with open(modality_path, 'w') as f:
            json.dump(modality, f, indent=2)

        print(f"    ✓ 创建modality.json")
