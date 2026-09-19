"""Bounded asynchronous JSONL diagnostics; no hardware access in this module."""
import json
from pathlib import Path
import queue
import threading
import time


class TeleopDiagnostics:
    def __init__(self, directory, metadata, capacity=4096):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / 'metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        self._queue = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._error = None
        self._closed = False
        self._thread = threading.Thread(target=self._write, name='TeleopDiagnostics', daemon=True)
        self._thread.start()

    def emit(self, kind, **fields):
        if self._error is not None:
            raise RuntimeError(f'Diagnostic writer failed: {self._error}')
        if self._closed:
            raise RuntimeError('Diagnostic writer closed')
        # Serialize before queueing so later mutations cannot change the event.
        line = json.dumps({'event': kind, 'event_ns': time.monotonic_ns(), **fields},
                          ensure_ascii=False, allow_nan=False) + '\n'
        try:
            self._queue.put_nowait(line)
        except queue.Full as exc:
            raise RuntimeError('Diagnostic queue full; cannot preserve input trace') from exc

    def _write(self):
        try:
            with (self.directory / 'events.jsonl').open('x', encoding='utf-8') as stream:
                flushed = time.monotonic()
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        stream.write(self._queue.get(timeout=0.1))
                    except queue.Empty:
                        pass
                    if time.monotonic() - flushed >= 1:
                        stream.flush()
                        flushed = time.monotonic()
        except Exception as exc:
            self._error = exc

    def close(self):
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=3)
        if self._thread.is_alive():
            raise RuntimeError('Diagnostic writer did not finish within 3 seconds')
        if self._error is not None:
            raise RuntimeError(f'Diagnostic writer failed: {self._error}')
