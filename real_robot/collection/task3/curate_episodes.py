#!/usr/bin/env python3
"""Review committed formal episodes without deleting or rewriting raw data."""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parent / 'captures'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def episode_path(root, session, episode):
    if not re.fullmatch(r'session_\d{8}_\d{6}_\d+', session):
        raise ValueError('请使用完整session目录名')
    if not re.fullmatch(r'episode_\d{6}', episode):
        raise ValueError('请使用episode_000000这样的回合目录名')
    path = root / session / 'episodes' / episode
    if path.resolve() != path.absolute() or not (path/'manifest.json').is_file():
        raise ValueError('不是正式目录内已经保存完成的回合')
    return path


def manifest(path):
    m = json.loads((path/'manifest.json').read_text())
    if m.get('exclude_from_training') or m.get('training_eligible') is False or m.get('preparation_batch'):
        raise ValueError('预备练习不能改成正式训练数据')
    return m


def validate_files(path, m):
    if m.get('samples', 0) <= 0:
        raise ValueError('空回合不能纳入训练候选')
    required = {'front.mp4', 'wrist.mp4', 'samples.parquet'}
    if not required <= m.get('files', {}).keys():
        raise ValueError('缺少双路视频或状态动作文件')
    for name, expected in m['files'].items():
        p = path/name
        if p.parent != path or p.resolve() != p.absolute() or not p.is_file() or digest(p) != expected:
            raise ValueError(f'原始数据校验失败: {name}')


def review(path):
    p = path/'review.json'
    return json.loads(p.read_text()) if p.exists() else {'status': 'unreviewed', 'history': []}


def atomic_json(path, data):
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def selection(root):
    accepted = []
    for p in sorted(root.glob('session_*/episodes/episode_*')):
        path = episode_path(root, p.parents[1].name, p.name)
        m, r = manifest(path), review(path)
        if r['status'] != 'accepted':
            continue
        if r.get('manifest_sha256') != digest(path/'manifest.json'):
            raise ValueError(f'审核后manifest发生变化: {path}')
        validate_files(path, m)
        accepted.append({'episode': str(path.resolve()), 'manifest_sha256': r['manifest_sha256'],
                         'review_sha256': digest(path/'review.json')})
    return {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'training_ready': False, 'selection_is_snapshot': True,
            'policy': 'only human-accepted formal episodes; converter must revalidate reviews and hashes',
            'episodes': accepted}


def set_review(root, session, episode, status, reason):
    if status not in ('accepted', 'rejected', 'unreviewed'):
        raise ValueError('Invalid review status')
    if status == 'rejected' and not reason.strip():
        raise ValueError('排除回合时请填写--reason')
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.review.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = episode_path(root, session, episode)
        m = manifest(path)
        if status == 'accepted':
            validate_files(path, m)
        old = review(path)
        event = {'status': status, 'reason': reason,
                 'reviewed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        atomic_json(path/'review.json', {**event, 'manifest_sha256': digest(path/'manifest.json'),
                                        'history': old.get('history', [])+[event]})
        try:
            selected = selection(root)
        except Exception as exc:
            atomic_json(root/'accepted_episodes.json', {'training_ready': False, 'episodes': [],
                                                       'error': str(exc)})
            raise RuntimeError('审核标记已保存，但清单校验失败，已清空候选清单；原件保留') from exc
        atomic_json(root/'accepted_episodes.json', selected)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['list','accept','reject','reset'])
    parser.add_argument('session', nargs='?')
    parser.add_argument('episode', nargs='?')
    parser.add_argument('--reason', default='')
    args = parser.parse_args()
    if args.operation == 'list':
        count = 0
        for mpath in sorted(ROOT.glob('session_*/episodes/episode_*/manifest.json')):
            p = mpath.parent
            m = manifest(p)
            print(p.parents[1].name, p.name, review(p)['status'], 'samples='+str(m['samples']))
            count += 1
        print(f'正式采集已保存: {count} 条；未审核及排除回合均不进入审核通过清单。')
        return
    if not args.session or not args.episode:
        parser.error('需要session目录名和episode目录名')
    status = {'accept':'accepted','reject':'rejected','reset':'unreviewed'}[args.operation]
    path = set_review(ROOT, args.session, args.episode, status, args.reason)
    print(f'{status}: {path}；原始文件保留。审核通过清单: {ROOT / "accepted_episodes.json"}')


if __name__ == '__main__':
    main()
