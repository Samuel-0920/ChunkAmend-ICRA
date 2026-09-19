"""Maintainer-only one-shot release packer; input URLs are temporary secrets.

Streams a hash-pinned source archive and publishes only explicitly listed model
members. Dataset entries, optimizer state and source logs are never written out.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import base64
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import threading
import urllib.request

REPO = os.environ['GITHUB_REPOSITORY']
TAG = 'uf850-checkpoints-v1'
JOB = json.loads(gzip.decompress(base64.b64decode(os.environ['MODEL_ARCHIVE'])))
TASK = JOB['task']
CHUNK = 512 * 1024**2
POOL = ThreadPoolExecutor(max_workers=2)
SLOTS = threading.Semaphore(3)
FUTURES = []
TEMP = tempfile.TemporaryDirectory(prefix='model-release-')


def upload(path):
    subprocess.run(['gh', 'release', 'upload', TAG, '--repo', REPO, str(path)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    path.unlink()


class Source(io.RawIOBase):
    def __init__(self):
        self.index = 0
        self.response = None
        self.total_hash = hashlib.sha256()
        self.total_size = 0
    def readable(self): return True
    def read(self, n=-1):
        if n < 0: raise ValueError('Bounded reads required')
        if n == 0: return b''
        out = bytearray()
        while len(out) < n:
            if self.response is None:
                if self.index == len(JOB['parts']): break
                self.part = JOB['parts'][self.index]
                self.response = urllib.request.urlopen(self.part['url'], timeout=180)
                self.part_hash = hashlib.sha256()
                self.part_size = 0
            block = self.response.read(min(n-len(out), 8*1024**2))
            if block:
                out.extend(block)
                self.part_hash.update(block)
                self.total_hash.update(block)
                self.part_size += len(block)
                self.total_size += len(block)
            else:
                self.response.close()
                self.response = None
                if self.part_size != self.part['size'] or self.part_hash.hexdigest() != self.part['sha256']:
                    raise RuntimeError('Source part integrity failure')
                self.index += 1
        return bytes(out)


class Parts:
    def __init__(self):
        self.file = None
        self.parts = []
        self.size = 0
        self.hash = hashlib.sha256()
    def write(self, data):
        self.size += len(data)
        self.hash.update(data)
        original = len(data)
        data = memoryview(data)
        while data:
            if self.file is None:
                SLOTS.acquire()
                for future in FUTURES:
                    if future.done(): future.result()
                self.path = Path(TEMP.name)/f'{TASK}-step9999-deploy.tar.part{len(self.parts):03d}'
                self.file = self.path.open('wb')
                self.part_size = 0
                self.part_hash = hashlib.sha256()
            block = data[:CHUNK-self.part_size]
            self.file.write(block)
            self.part_hash.update(block)
            self.part_size += len(block)
            data = data[len(block):]
            if self.part_size == CHUNK: self.finish()
        return original
    def finish(self):
        if self.file is None: return
        self.file.close()
        self.parts.append(dict(name=self.path.name, size=self.part_size, sha256=self.part_hash.hexdigest()))
        future = POOL.submit(upload, self.path)
        future.add_done_callback(lambda _: SLOTS.release())
        FUTURES.append(future)
        self.file = None


source, sink = Source(), Parts()
expected = {m['path']: m for m in JOB['members']}
assert all(p.startswith('9999/params/') or p == JOB['normalization_path'] or p == '9999/_CHECKPOINT_METADATA' for p in expected)
seen = {}
with tarfile.open(fileobj=source, mode='r|', bufsize=1024*1024) as src, tarfile.open(fileobj=sink, mode='w|', format=tarfile.PAX_FORMAT) as dst:
    for member in src:
        name = member.name.removeprefix('./')
        if name not in expected: continue
        if not member.isfile() or name in seen or member.size != expected[name]['size']:
            raise RuntimeError('Invalid or duplicate selected model member')
        info = tarfile.TarInfo(name)
        info.size, info.mode, info.mtime = member.size, 0o644, 0
        digest = hashlib.sha256()
        class HashedReader:
            def __init__(self, stream): self.stream = stream
            def read(self, n):
                b = self.stream.read(n)
                digest.update(b)
                return b
        with src.extractfile(member) as stream:
            dst.addfile(info, HashedReader(stream))
        if digest.hexdigest() != expected[name]['sha256']:
            raise RuntimeError('Model differs from verified local deployment')
        seen[name] = dict(path=name, size=member.size, sha256=digest.hexdigest())
while source.read(8*1024**2): pass
if source.total_size != JOB['size'] or source.total_hash.hexdigest() != JOB['sha256']:
    raise RuntimeError('Full source archive integrity failure')
if set(seen) != set(expected): raise RuntimeError('Missing selected model members')
sink.finish()
for future in FUTURES: future.result()
POOL.shutdown()
manifest = dict(task=TASK, prompt=JOB['prompt'], dataset_id=JOB['dataset_id'], policy_config=JOB['policy_config'],
                path=f'{TASK}-step9999-deploy.tar', size=sink.size, sha256=sink.hash.hexdigest(),
                parts=sink.parts, members=list(seen.values()))
path = Path(TEMP.name)/f'{TASK}-checkpoint.json'
path.write_text(json.dumps(manifest, indent=2)+'\n')
upload(path)
TEMP.cleanup()
print('Verified model-only release:', TASK, len(seen), 'members,', sink.size, 'bytes')
