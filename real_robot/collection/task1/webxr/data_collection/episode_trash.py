"""Move a complete committed episode to the desktop trash, never unlink it."""
import datetime
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import subprocess


def trash_episode(path):
    path = Path(path).absolute()
    if (path.resolve() != path or path.parent.name != 'episodes'
            or not re.fullmatch(r'episode_\d{6}', path.name)
            or not (path/'manifest.json').is_file()):
        raise ValueError('Only a complete, non-symlink episode can be trashed')
    session = path.parent.parent
    root = session.parent
    with (root/'.review.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # gio has no permanent-delete fallback; unsupported trash is an error.
        subprocess.run(['/usr/bin/gio', 'trash', '--', str(path)],
                       check=True, capture_output=True, text=True, timeout=10)
        if path.exists():
            raise RuntimeError('gio reported success but episode is still present')
        event = {'original_episode': str(path), 'method': 'gio trash',
                 'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        with (session/'trash_events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False)+'\n')
        # Rebuild the formal selection so a previously accepted episode is removed.
        if root.name == 'captures':
            spec = importlib.util.spec_from_file_location('trash_curation',
                Path(__file__).resolve().parents[2]/'curate_episodes.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            try:
                selected = module.selection(root)
            except Exception as exc:
                selected = {'training_ready': False, 'episodes': [], 'error': str(exc)}
            module.atomic_json(root/'accepted_episodes.json', selected)
    return event
