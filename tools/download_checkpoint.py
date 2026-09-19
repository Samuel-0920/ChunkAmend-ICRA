"""Download and SHA-256 verify one public UF850 deployment checkpoint."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024**2): h.update(block)
    return h.hexdigest()


def download(item, base_url, output):
    name = item['path']
    if Path(name).name != name: raise ValueError('Archive must be a filename')
    if sum(p['size'] for p in item['parts']) != item['size']:
        raise ValueError('Invalid part sizes')
    output.mkdir(parents=True, exist_ok=True)
    target = output / name
    if target.exists():
        if target.stat().st_size == item['size'] and digest(target) == item['sha256']:
            print('Already verified:', target)
            return target
        raise FileExistsError('Existing archive differs; refusing to overwrite: ' + str(target))
    pending = target.with_suffix(target.suffix + '.partial')
    full = hashlib.sha256()
    with pending.open('wb') as out:
        for i, part in enumerate(item['parts'], 1):
            if Path(part['name']).name != part['name']: raise ValueError('Invalid part name')
            size, h = 0, hashlib.sha256()
            url = base_url + '/' + urllib.parse.quote(part['name'], safe='')
            with urllib.request.urlopen(url, timeout=180) as response:
                while block := response.read(8 * 1024**2):
                    size += len(block)
                    if size > part['size']: raise ValueError('Oversized part')
                    h.update(block)
                    full.update(block)
                    out.write(block)
            if size != part['size'] or h.hexdigest() != part['sha256']:
                raise ValueError('Checkpoint part integrity failure: ' + part['name'])
            print(f'Verified {i}/{len(item["parts"])} parts', flush=True)
    if pending.stat().st_size != item['size'] or full.hexdigest() != item['sha256']:
        raise ValueError('Checkpoint archive integrity failure')
    pending.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('task1', 'task2', 'task3'))
    parser.add_argument('--output', type=Path, default=Path('checkpoints/downloads'))
    args = parser.parse_args()
    manifest = json.loads((ROOT/'checkpoints/manifest.json').read_text())
    item = next(f for f in manifest['files'] if f['task'] == args.task)
    base = f'https://github.com/{manifest["repository"]}/releases/download/{manifest["tag"]}'
    # Carry model-use terms with every downloaded checkpoint.
    args.output.mkdir(parents=True, exist_ok=True)
    for filename in ('LICENSE', 'LICENSE_GEMMA.txt', 'NOTICE.txt', 'MODEL_LICENSE.md'):
        (args.output/filename).write_bytes((ROOT/filename).read_bytes())
    path = download(item, base, args.output)
    print('Verified archive:', path)
    print('Contains 9999/params and matching normalization statistics; no dataset. No files were extracted.')


if __name__ == '__main__':
    main()
