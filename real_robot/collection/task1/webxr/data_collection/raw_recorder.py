"""Atomic per-episode raw bundles. Not a LeRobot training dataset."""
import hashlib
import json
import os
from pathlib import Path
import uuid

import cv2
import numpy as np


class RawEpisodeRecorder:
    def __init__(self, dataset_path, episode_index, fps=30):
        self.root = Path(dataset_path)
        self.index = episode_index
        self.fps = fps

    def save(self, data):
        import pandas as pd
        n = len(data['states'])
        keys = ['actions', 'timestamps', 'frames_front', 'frames_wrist', 'timing']
        if any(len(data[k]) != n for k in keys):
            raise ValueError('Mismatched sample/video/timing lengths')
        for key in ['states', 'actions']:
            a = np.asarray(data[key])
            if n and (a.ndim != 2 or a.shape[1] not in (7, 8) or not np.isfinite(a).all()):
                raise ValueError('Invalid raw SDK vector')
        if n and np.asarray(data['states']).shape != np.asarray(data['actions']).shape:
            raise ValueError('State/action widths differ')
        if n and (not np.isfinite(data['timestamps']).all() or
                  np.any(np.diff(data['timestamps']) <= 0)):
            raise ValueError('Invalid monotonic sample timestamps')
        root = self.root / 'episodes'
        root.mkdir(exist_ok=True)
        target = root / f'episode_{self.index:06d}'
        # Reservation prevents competing writers and protects failed attempts from reuse.
        reservation = root / f'.reserved_{self.index:06d}'
        with reservation.open('x') as stream:
            stream.write('Reserved; retained on failure. See .partial directories.\n')
        if target.exists():
            raise FileExistsError(target)
        stage = root / f'.partial_{self.index:06d}_{uuid.uuid4().hex}'
        stage.mkdir()
        columns = {'observation.state': data['states'], 'action': data['actions'],
                   'sample_monotonic_s': data['timestamps'], 'frame_index': list(range(n)),
                   'timing_json': [json.dumps(x, allow_nan=False) for x in data['timing']]}
        pd.DataFrame(columns).to_parquet(stage / 'samples.parquet', index=False)
        for role in ['front', 'wrist']:
            if n:
                self._video(stage / f'{role}.mp4', data[f'frames_{role}'])
        manifest = {
            'format': 'uf850_raw_capture_v1', 'episode_index': self.index, 'samples': n,
            'nominal_video_fps': self.fps, 'codec': 'mpeg4/mp4v',
            'task': data['task_description'], 'task_index': data['task_index'],
            'termination_reason': data.get('termination_reason'),
            'task_success': data.get('task_success'), 'success_source': 'unverified_operator' if data.get('task_success') is not None else None,
            'training_ready': False, 'hardware_synchronized': False,
            'short_episode': n < 10, 'raw_vector_width': len(data['states'][0]) if n else None,
            'state_semantics': 'SDK joint payload rad + gripper pulse/850; effective axes require verification',
            'action_semantics': 'last SDK-accepted joint payload rad + last SDK-accepted rounded gripper pulse/850; execution not confirmed',
            'tactile_available': False, 'intervention_source': 'human_teleop',
            'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir()},
        }
        metadata = data.get('collection_metadata', {})
        for key in ('collection_purpose', 'preparation_batch', 'training_eligible', 'translation_gain',
                    'exclude_from_training', 'initial_pose_mm_degrees',
                    'initial_pose_source', 'completed_before', 'task_home_id',
                    'task_home_evidence', 'observed_initial_pose_mm_degrees'):
            if key in metadata:
                manifest[key] = metadata[key]
        (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
        for path in stage.iterdir():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        self._sync_dir(stage)
        stage.rename(target)
        self._sync_dir(root)
        return target

    @staticmethod
    def _sync_dir(path):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _video(self, path, frames):
        for frame in frames:
            if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
                raise ValueError('Video requires 640x480 BGR uint8 frames')
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (640, 480))
        try:
            if not writer.isOpened():
                raise OSError(f'VideoWriter failed: {path}')
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        # Decode only the just-written local file, never a camera device.
        reader = cv2.VideoCapture(str(path))
        count = 0
        try:
            if not reader.isOpened():
                raise OSError(f'Cannot decode {path}')
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                if frame.shape != (480, 640, 3):
                    raise OSError('Decoded dimensions differ')
                count += 1
        finally:
            reader.release()
        if count != len(frames):
            raise OSError(f'Video truncated: expected {len(frames)}, decoded {count}')
