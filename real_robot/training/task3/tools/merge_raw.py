"""Merge the two retained task-three batches offline; preserve raw payload bytes."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    (ROOT.parent / 'uf850_task3_precision_placement_v001/captures', 98),
    (ROOT.parent / 'uf850_task3_dynamic_collection_v001/captures', 35),
]
TASK = 'Pick up the lemon and place it into the goblet.'
REMOVED = {
    'collection_condition', 'relocation_plan_id', 'relocation_plan_sha256',
    'relocation_plan_index', 'planned_destination_direction', 'relocation_trigger',
    'destination_markers_present', 'actual_destination_position',
    'relocation_event_verified',
}
PAYLOADS = {'samples.parquet', 'front.mp4', 'wrist.mp4'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sources():
    result = []
    for root, expected in SOURCES:
        batch = sorted(root.glob('session_*/episodes/episode_*/manifest.json'))
        if len(batch) != expected:
            raise ValueError(f'Expected {expected} committed episodes: {root}; got {len(batch)}')
        result.extend(batch)
    return result


def check(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_payload(path, manifest):
    data = pq.read_table(path / 'samples.parquet').to_pydict()
    n = manifest['samples']
    check(n > 1 and all(len(v) == n for v in data.values()), f'Sample count: {path}')
    check(np.array_equal(data['frame_index'], np.arange(n)), f'Frame indices: {path}')
    for key in ['observation.state', 'action']:
        a = np.asarray(data[key])
        check(a.shape == (n, 8) and np.isfinite(a).all(), f'Invalid {key}: {path}')
        check(np.all(a[:, 6] == 0), f'Nonzero SDK padding: {path}')
    t = np.asarray(data['sample_monotonic_s'])
    check(np.isfinite(t).all() and np.all(np.diff(t) > 0), f'Nonmonotonic clock: {path}')
    timing = [json.loads(x) for x in data['timing_json']]
    check(np.allclose(t, [x['sample_ns'] / 1e9 for x in timing], rtol=0, atol=1e-8),
          f'Timestamp mismatch: {path}')
    for row in timing:
        for key in ['command', 'gripper_command']:
            check(row[key]['sdk_code'] == 0 and row[key]['return_ns'] <= row['sample_ns'],
                  f'Rejected or future command: {path}')
    videos = {}
    for role in ['front', 'wrist']:
        cap = cv2.VideoCapture(str(path / f'{role}.mp4'))
        try:
            check(cap.isOpened(), f'Unreadable {role}: {path}')
            fps = cap.get(cv2.CAP_PROP_FPS)
            check(abs(fps - manifest['nominal_video_fps']) < 1e-4, f'Video FPS: {path}')
            count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                check(frame.shape == (480, 640, 3), f'Video shape: {path}')
                count += 1
            check(count == n, f'{role}: decoded {count}, expected {n}: {path}')
            videos[role] = {'decoded_frames': count, 'nominal_fps': fps}
        finally:
            cap.release()
    return {'samples': n, 'observed_sample_hz': float((n-1) / (t[-1]-t[0])),
            'max_sample_interval_ms': float(np.diff(t).max()*1000), 'videos': videos}


def main():
    cv2.setNumThreads(2)
    check(not (ROOT / 'captures').exists(), 'Refusing to overwrite published merged data')
    original = sources()
    snapshot = [{'path': str(p), 'sha256': digest(p)} for p in original]
    write(ROOT / 'evidence/source_manifest_snapshot.json', snapshot)
    stage = ROOT / 'captures.partial'
    stage.mkdir(exist_ok=False)
    stamp = dt.datetime.now(dt.timezone.utc)
    session = 'session_' + stamp.strftime('%Y%m%d_%H%M%S_%f')
    mapping, audit = [], []
    signatures = set()
    for index, mf in enumerate(original):
        m = json.loads(mf.read_text())
        check(m['format'] == 'uf850_raw_capture_v1' and m['task'] == TASK, f'Task/schema: {mf}')
        check(m['collection_purpose'] == 'formal_demonstration' and m['training_eligible']
              and not m['exclude_from_training'] and m['termination_reason'] == 'operator_b_end',
              f'Eligibility mismatch: {mf}')
        check(set(m['files']) == PAYLOADS, f'Payload list mismatch: {mf}')
        signature = tuple(m['files'][k] for k in sorted(PAYLOADS))
        check(signature not in signatures, f'Duplicate payload: {mf}')
        signatures.add(signature)
        relative = Path(session) / 'episodes' / f'episode_{index:06d}'
        dest = stage / relative
        dest.mkdir(parents=True)
        for name in sorted(PAYLOADS):
            src = mf.parent / name
            check(digest(src) == m['files'][name], f'Source hash mismatch: {src}')
            shutil.copy2(src, dest / name)
            check(digest(dest / name) == m['files'][name], f'Copy hash mismatch: {dest/name}')
        checked = inspect_payload(dest, m)
        normalized = {k: v for k, v in m.items() if k not in REMOVED}
        normalized['episode_index'] = index
        write(dest / 'manifest.json', normalized)
        archive = ROOT / 'provenance/original_manifests' / f'episode_{index:06d}.json'
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mf, archive)
        check(digest(archive) == snapshot[index]['sha256'], f'Source changed during copy: {mf}')
        mapping.append({'episode_index': index, 'merged_episode': str(Path('captures') / relative),
                        'source_episode': str(mf.parent), 'source_manifest_sha256': snapshot[index]['sha256'],
                        'merged_manifest_sha256': digest(dest/'manifest.json'),
                        'payload_sha256': m['files'], 'samples': m['samples'],
                        'removed_historical_metadata_keys': sorted(REMOVED.intersection(m))})
        audit.append({'episode_index': index, **checked})
        if (index+1) % 10 == 0 or index+1 == len(original):
            print(f'{index+1}/{len(original)} copied, hashes checked, both videos fully decoded', flush=True)
    check([{'path': str(p), 'sha256': digest(p)} for p in sources()] == snapshot,
          'Source selection or manifests changed during merge')
    for row in mapping:
        for name, expected in row['payload_sha256'].items():
            check(digest(Path(row['source_episode']) / name) == expected, 'Source payload changed')
    spec = {'task_id': 'lemon_in_goblet', 'task_name_en': 'Precision Lemon Placement in a Goblet',
            'task_prompt_en': TASK, 'total_episodes': len(mapping),
            'success_condition': '松爪后柠檬进入高脚杯内即可，无需额外停留',
            'stable_retention_seconds': 0,
            'disturbance_annotation_policy': 'No per-episode disturbance direction, time, endpoint or count labels; user reports random disturbance.'}
    write(ROOT / 'TASK_SPEC.json', spec)
    write(stage / session / 'session.json', {'status': 'offline_merged', 'task_spec': spec,
          'created_at_utc': stamp.isoformat(), 'source_sessions': sorted({str(p.parents[2]) for p in original})})
    write(ROOT / 'provenance/source_map.json', mapping)
    write(ROOT / 'evidence/payload_audit.json', audit)
    stage.rename(ROOT / 'captures')
    report = {'status': 'merged_raw_verified', 'episodes': len(mapping),
              'source_batch_counts': [98, 35], 'samples': sum(x['samples'] for x in audit),
              'videos': 2*len(mapping), 'payload_bytes_unchanged': True,
              'all_videos_fully_decoded': True, 'source_manifests_unchanged': True,
              'duplicate_payloads': 0, 'disturbance_annotations_present': False,
              'task_success_labels_inferred': False, 'training_format_conversion_done': False,
              'nominal_video_fps': 30,
              'observed_episode_sample_hz_min_max': [min(x['observed_sample_hz'] for x in audit), max(x['observed_sample_hz'] for x in audit)],
              'dataset_captures': str(ROOT/'captures')}
    write(ROOT / 'MERGE_REPORT.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
