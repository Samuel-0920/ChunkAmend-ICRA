#!/usr/bin/env python3
"""
数据收集核心模块
"""

import time
import os
import numpy as np
from pathlib import Path
from .raw_recorder import RawEpisodeRecorder as EpisodeRecorder


class DataCollector:
    """
    LeRobot格式数据收集器
    核心职责:
    1. 管理episode录制状态
    2. 按30Hz下采样收集数据 (从100Hz控制循环)
    3. 内存缓存episode数据
    4. 调用EpisodeRecorder保存
    """

    # 预定义任务列表
    PREDEFINED_TASKS = [
        "pick the cup and place it on the plate",
    ]

    def __init__(self, dataset_path, record_freq=20, control_freq=100,
                 reward_neg: float = -0.05,
                 reward_success: float = 10.0,
                 discount: float = 0.98, collection_metadata=None, max_episodes=None):
        """
        初始化数据收集器

        Args:
            dataset_path: 数据集保存路径
            record_freq: 数据记录频率 (Hz, 默认30, 接近对齐: 100/3≈33Hz)
            control_freq: 控制循环频率 (Hz, 默认100)
            reward_neg: 负奖励值
            reward_success: 成功奖励值
            discount: 折扣因子
        """
        self.dataset_path = Path(dataset_path)
        self.collection_metadata = dict(collection_metadata or {})
        self.max_episodes = max_episodes
        self.record_freq = record_freq
        self.control_freq = control_freq
        self.reward_neg = float(reward_neg)
        self.reward_success = float(reward_success)
        self.discount = float(discount)
        #self.record_interval = control_freq // record_freq  # 每3个控制周期记录1次 (100//30=3)

        # Episode状态
        self.save_failed = False
        self.is_recording = False
        self.current_episode_data = None
        self.episode_index = 0
        self.retained_episode_count = 0
        self.last_saved_episode = None
        self.step_count = 0  # 全局step计数 (用于下采样)
        self.episode_start_time = None
        self.episode_step_count = 0  # 当前episode内的step数
        self.last_record_time = 0.0

        # 任务管理
        self.current_task_index = 0
        self.tasks = [os.environ.get('UF850_TASK', self.PREDEFINED_TASKS[0])]

        # 初始化数据集目录结构
        self._init_dataset_structure()

        # 加载已有episode数量
        self._load_existing_episodes()

        print(f"\n{'='*60}")
        print("数据收集器初始化完成")
        print(f"  数据集路径: {self.dataset_path}")
        print(f"  控制频率: {self.control_freq} Hz")
        print(f"  记录频率: {self.record_freq} Hz")
        print(f"  当前Episode索引: {self.episode_index}")
        print(f"  预定义任务数: {len(self.tasks)}")
        print(f"{'='*60}\n")

    def _init_dataset_structure(self):
        """创建LeRobot数据集目录结构"""
        # meta目录
        (self.dataset_path / 'meta').mkdir(parents=True, exist_ok=True)

        # data目录
        (self.dataset_path / 'data' / 'chunk-000').mkdir(parents=True, exist_ok=True)

        # videos目录
        video_base = self.dataset_path / 'videos' / 'chunk-000'
        (video_base / 'observation.images.wrist').mkdir(parents=True, exist_ok=True)
        (video_base / 'observation.images.front').mkdir(parents=True, exist_ok=True)

        print(f"✓ 数据集目录结构已创建: {self.dataset_path}")

    def _load_existing_episodes(self):
        """加载已有的episode数量，从现有基础上继续"""
        data_dir = self.dataset_path / 'episodes'
        if data_dir.exists():
            existing = list(data_dir.glob('episode_*/manifest.json'))
            self.retained_episode_count = len(existing)
            indices = [int(p.parent.name.split('_')[1]) for p in existing]
            indices += [int(p.name.removeprefix('.reserved_'))
                        for p in data_dir.glob('.reserved_*')
                        if p.name.removeprefix('.reserved_').isdigit()]
            if indices:
                self.episode_index = max(indices) + 1
                print(f"✓ 保留{len(existing)}条回合，下一编号{self.episode_index}；不会复用已回收编号")

    def should_record_this_step(self):
        """
        判断当前是否需要记录数据（按时间戳控制录制频率）
        """
        if not self.is_recording:
            return False

        now = time.time()
        if now - self.last_record_time >= 1.0 / self.record_freq:
            self.last_record_time = now
            return True
        return False

    def get_current_task(self):
        """获取当前任务描述"""
        return self.tasks[self.current_task_index]

    def set_task_by_index(self, task_index):
        """
        手动设置任务索引

        Args:
            task_index: 任务索引 (0到len(tasks)-1)
        """
        if 0 <= task_index < len(self.tasks):
            self.current_task_index = task_index
            print(f"\n✓ 已设置任务: [{task_index}] {self.get_current_task()}\n")
        else:
            print(f"⚠ 无效的任务索引 {task_index}，有效范围: 0-{len(self.tasks)-1}")

    def start_episode(self, task_description=None):
        """
        开始录制新episode

        Args:
            task_description: 任务描述，如果为None则使用当前任务
        """
        if self.save_failed:
            raise RuntimeError("Previous save failed; buffer retained, restart only after recovery")
        if self.is_recording:
            print("⚠ 已在录制中，请先结束当前episode")
            return False
        if self.max_episodes is not None and self.retained_episode_count >= self.max_episodes:
            print("✓ 已达到本轮练习剩余配额，拒绝新增回合。全部练习排除训练。")
            return False

        episode_metadata = dict(self.collection_metadata)
        if os.environ.get('UF850_RELOCATION_PLAN'):
            from .relocation_plan import next_assignment
            assignment = next_assignment(os.environ['UF850_RELOCATION_PLAN'], self.dataset_path.parent)
            if assignment is None:
                print('35条换位示教已保存，不再新增；请退出并审核。')
                return False
            episode_metadata.update(assignment)
            direction = '左' if assignment['planned_destination_direction'] == 'left' else '右'
            print(f"本条换位计划 {assignment['relocation_plan_index']}/35：杯子中位起始，向{direction}移动一次。终点未标记，撤手后继续放置。", flush=True)

        # 确定任务描述
        if task_description is None:
            task_description = self.get_current_task()

        # 初始化episode数据buffer
        self.current_episode_data = {
            'collection_metadata': episode_metadata,
            'timing': [],
            'termination_reason': None,
            'task_success': None,
            'states': [],           # observation.state (关节角度 + 夹爪)
            'actions': [],          # action (目标关节角度 + 夹爪)
            'timestamps': [],       # timestamp (秒)
            'frames_wrist': [],     # 手腕相机帧
            'frames_front': [],     # 正面相机帧
            'task_description': task_description,
            'task_index': self.current_task_index,
            'tactile_left': [],      # list of np.ndarray [k, 16, 16] per step
            'tactile_right': [],      # list of np.ndarray [k, 16, 16] per step (mode=1 才有)
            'rewards': [],          # reward for each step
            'dones': [],             # done flag for each step
            'masks': [],
            'mc_returns': [],
            'is_intervention': [],  # True = expert teleop, False = policy
        }

        self.is_recording = True
        self.episode_start_time = time.time()
        self.episode_step_count = 0
        self.last_record_time = 0.0

        print(f"\n{'='*60}")
        print(f"🔴 开始录制 Episode {self.episode_index}")
        print(f"   任务: [{self.current_task_index}] {task_description}")
        print(f"   采样频率: {self.record_freq} Hz")
        print(f"{'='*60}\n")

        return True

    def record_step(self, state, action, frames, timestamp, fsr_seq=None,
                    is_intervention=False, timing=None):
        """
        记录单步数据并处理 FSR 信号断流

        Args:
            is_intervention: True if this step is expert teleop, False if policy
        """
        if not self.is_recording:
            return

        # Validate before changing any buffer. Missing evidence is not a valid sample.
        if set(frames) != {'wrist', 'front'}:
            raise ValueError('Both real camera frames required')
        for frame in frames.values():
            if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
                raise ValueError('Expected 640x480 BGR uint8')
        if state.shape != action.shape or state.shape not in ((7,), (8,)):
            raise ValueError('Raw SDK vectors must have equal 7/8 widths; no column dropping')
        if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isfinite(timestamp):
            raise ValueError('Nonfinite sample')
        if timing is None:
            raise ValueError('Missing sample timing evidence')
        if self.current_episode_data['timestamps'] and timestamp <= self.current_episode_data['timestamps'][-1]:
            raise ValueError('Nonmonotonic sample time')
        self.current_episode_data['timing'].append(timing)
        # 1. 保存基础状态数据
        self.current_episode_data['states'].append(state.copy())
        self.current_episode_data['actions'].append(action.copy())
        self.current_episode_data['timestamps'].append(timestamp)

        if 'wrist' in frames:
            self.current_episode_data['frames_wrist'].append(frames['wrist'].copy())
        if 'front' in frames:
            self.current_episode_data['frames_front'].append(frames['front'].copy())

        # 2. --- FSR (Tactile) 处理与填充逻辑 ---
        tactile_left_data = None
        tactile_right_data = None

        # 判断条件：如果 fsr_seq 有效且长度符合要求(k=5)
        if fsr_seq is not None and len(fsr_seq) == 5:
            # 提取并堆叠
            left_stack = np.stack([x["fsr1"] for x in fsr_seq], axis=0).astype(np.float32)
            tactile_left_data = left_stack.flatten() # (1280,)

            right_list = [x.get("fsr2") for x in fsr_seq]
            if all(v is not None for v in right_list):
                right_stack = np.stack(right_list, axis=0).astype(np.float32)
                tactile_right_data = right_stack.flatten()
            else:
                tactile_right_data = np.zeros(1280, dtype=np.float32)
        
        # --- 核心改进：前向填充 (Forward Fill) ---
        # 如果当前这一步没拿到数据，或者拿到的全是 0 (np.allclose 检查)
        is_empty = (tactile_left_data is None) or np.allclose(tactile_left_data, 0)
        
        if is_empty:
            # 尝试获取上一帧保存的数据
            if len(self.current_episode_data['tactile_left']) > 0:
                tactile_left_data = self.current_episode_data['tactile_left'][-1].copy()
                tactile_right_data = self.current_episode_data['tactile_right'][-1].copy()
            else:
                # 如果是第一帧就没读到，只能填 0
                tactile_left_data = np.zeros(1280, dtype=np.float32)
                tactile_right_data = np.zeros(1280, dtype=np.float32)

        # 存入列表
        self.current_episode_data['tactile_left'].append(tactile_left_data)
        self.current_episode_data['tactile_right'].append(tactile_right_data)
        
        #rl labels
        self.current_episode_data['rewards'].append(np.float32(self.reward_neg))
        self.current_episode_data['dones'].append(False)
        self.current_episode_data['is_intervention'].append(bool(is_intervention))

        self.episode_step_count += 1

        # 每100帧打印一次进度和信号检查
        if self.episode_step_count % 100 == 0:
            elapsed = time.time() - self.episode_start_time
            fps = self.episode_step_count / elapsed
            max_l = np.max(tactile_left_data)
            print(f"  录制中... 已记录{self.episode_step_count}帧 ({fps:.1f}Hz), "
                  f"当前左手峰值压力: {max_l:.4f}")
    
    def _finalize_rl_labels(self, success: bool):
        """
        在 stop_episode 时调用：
        - 如果 success=True：把最后一步 reward 改成 10，done=True
        - 计算 masks
        - 计算 mc_returns（折扣回报）
        """
        n = len(self.current_episode_data['rewards'])
        if n == 0:
            return

        rewards = np.asarray(self.current_episode_data['rewards'], dtype=np.float32)
        dones = np.asarray(self.current_episode_data['dones'], dtype=bool)

        if success:
            rewards[-1] = np.float32(self.reward_success)
            dones[-1] = True
        else:
            # 如果你允许“失败结束”，可以把最后一步 done=True（可选）
            # dones[-1] = True
            pass

        masks = (1.0 - dones.astype(np.float32)).astype(np.float32)

        # mc_returns: G_t = r_t + gamma * mask_t * G_{t+1}
        mc = np.zeros_like(rewards, dtype=np.float32)
        G = np.float32(0.0)
        gamma = np.float32(self.discount)
        for t in range(n - 1, -1, -1):
            G = rewards[t] + gamma * masks[t] * G
            mc[t] = G

        self.current_episode_data['rewards'] = rewards.tolist()
        self.current_episode_data['dones'] = dones.tolist()
        self.current_episode_data['masks'] = masks.tolist()
        self.current_episode_data['mc_returns'] = mc.tolist()
            
    def stop_episode(self, success=None, reason="interrupted"):
        """结束当前episode并保存"""
        if not self.is_recording:
            print("⚠ 当前未在录制")
            return

        self.is_recording = False
        duration = time.time() - self.episode_start_time
        num_frames = len(self.current_episode_data['states'])
        
        self.current_episode_data['termination_reason'] = reason
        self.current_episode_data['task_success'] = success
        self._finalize_rl_labels(success=success is True)

        print(f"\n{'='*60}")
        print(f"⏹ 停止录制 Episode {self.episode_index}")
        print(f"   帧数: {num_frames}")
        print(f"   时长: {duration:.2f}s")
        if duration > 0:
            print(f"   实际频率: {num_frames/duration:.1f} Hz")
        print(f"{'='*60}\n")

        # 保存数据
        print("💾 保存数据中...")
        try:
            recorder = EpisodeRecorder(
                dataset_path=self.dataset_path,
                episode_index=self.episode_index,
                fps=self.record_freq
            )
            recorder.save(self.current_episode_data)
            self.last_saved_episode = self.dataset_path / 'episodes' / f'episode_{self.episode_index:06d}'
            self.retained_episode_count += 1

            print(f"✅ Episode {self.episode_index} 保存完成!\n")

            # 递增episode索引
            self.episode_index += 1
            if self.max_episodes is not None:
                before = self.collection_metadata.get('completed_before', 0)
                print(f"准备阶段进度：{before + self.retained_episode_count}/20 条保留（不计入训练）")

        except Exception as e:
            print(f"❌ 保存失败: {e}")
            import traceback
            traceback.print_exc()
            self.save_failed = True
            raise

        # 清空buffer
        self.current_episode_data = None
        self.episode_step_count = 0

    def trash_last_episode(self):
        from .episode_trash import trash_episode
        if self.is_recording or self.current_episode_data is not None or self.save_failed:
            raise RuntimeError('录制或保存未完成，不能回收上一条')
        path = self.last_saved_episode
        if path is None:
            return None
        if not path.is_dir():
            raise RuntimeError('上一条回合已不在原目录，请核对回收站；不自动选择更早回合')
        try:
            return trash_episode(path)
        finally:
            if not path.exists():
                # Never reuse an episode id or let repeated A presses delete older data.
                self.last_saved_episode = None
                self.retained_episode_count = max(0, self.retained_episode_count - 1)

    def get_statistics(self):
        """获取数据收集统计信息"""
        data_dir = self.dataset_path / 'episodes'
        episodes = list(data_dir.glob('episode_*')) if data_dir.exists() else []

        stats = {
            'total_episodes': len(episodes),
            'next_episode_index': self.episode_index,
            'is_recording': self.is_recording,
            'current_task': self.get_current_task(),
            'current_task_index': self.current_task_index
        }

        return stats

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()

        print(f"\n{'='*60}")
        print("数据收集统计")
        print(f"{'='*60}")
        print(f"  已收集Episodes: {stats['total_episodes']}")
        print(f"  下一个Episode索引: {stats['next_episode_index']}")
        print(f"  录制状态: {'🔴 录制中' if stats['is_recording'] else '⚪ 未录制'}")
        print(f"  当前任务: [{stats['current_task_index']}] {stats['current_task']}")
        print(f"{'='*60}\n")
