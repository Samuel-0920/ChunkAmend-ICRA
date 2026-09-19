#!/usr/bin/env python3
"""Local dual-camera preview only; no robot SDK, model, or recording."""
import fcntl
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import signal
import threading
import time

BASE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>UF850 双相机预览</title><style>
body{background:#111827;color:#eee;font:18px system-ui;margin:24px}
.views{display:flex;gap:16px;flex-wrap:wrap}.view{flex:1;min-width:320px}
img{width:100%;max-width:640px;background:#000}button{font:inherit;padding:10px}
#status{color:#fbbf24}h2{font-size:20px}</style>
<h1>前置＋腕部相机</h1><p>仅预览，不控制机械臂、不录制。启动练习前请停止预览。</p>
<div class="views"><div class="view"><h2>前置 Front</h2><img id="front"></div>
<div class="view"><h2>腕部 Wrist</h2><img id="wrist"></div></div>
<p id="status">连接中…</p><button onclick="stopPreview()">停止预览并释放相机</button>
<script>let stopped=false;const statusEl=document.getElementById('status');
for(const role of ['front','wrist']){const im=document.getElementById(role);
function update(){if(!stopped)im.src='/frame/'+role+'.jpg?t='+Date.now()}
im.onload=()=>setTimeout(update,80);im.onerror=()=>{statusEl.textContent='画面读取失败或预览已停止';setTimeout(update,1000)};update()}
async function stopPreview(){stopped=true;try{await fetch('/stop',{method:'POST'});statusEl.textContent='预览已停止；相机正在释放'}catch(e){statusEl.textContent='预览连接已关闭'}}
setInterval(async()=>{if(stopped)return;try{const s=await(await fetch('/status')).json();
statusEl.textContent=s.error||('双路画面已更新 · 帧组 '+s.sequence+' · 不录制');}catch(e){statusEl.textContent='预览连接已关闭'}},1000);
</script></html>'''


def main():
    import cv2
    launcher = load('preview_launcher', BASE / 'launch.py')
    camera_module = load('preview_camera_manager', BASE / 'webxr/data_collection/camera_manager.py')
    with (BASE / '.camera-owner.lock').open('a') as ownership:
        try:
            fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('相机正被预览或采集入口使用，请先退出占用程序') from exc
        env = launcher.camera_environment()
        state = {'frames': {}, 'sequence': 0, 'updated': 0.0, 'error': None}
        mutex, stop = threading.Lock(), threading.Event()
        manager = None
        worker = None
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                path = self.path.split('?')[0]
                if path == '/':
                    body, content_type = PAGE.encode(), 'text/html; charset=utf-8'
                elif path == '/status':
                    with mutex:
                        body = json.dumps({'sequence': state['sequence'], 'error': state['error']}).encode()
                    content_type = 'application/json'
                elif path in ('/frame/front.jpg', '/frame/wrist.jpg'):
                    role = path.split('/')[2].split('.')[0]
                    with mutex:
                        body = state['frames'].get(role)
                        stale = time.monotonic() - state['updated'] > 1.0
                    if not body or stale:
                        self.send_error(503, 'No fresh frame')
                        return
                    content_type = 'image/jpeg'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            def do_POST(self):
                if self.path != '/stop':
                    self.send_error(404)
                    return
                self.send_response(200)
                self.end_headers()
                self.server.shutdown()
        server = ThreadingHTTPServer(('127.0.0.1', 8082), Handler)
        try:
            manager = camera_module.CameraManager(
                {'front': env['MAIN_CAM_DEVICE'], 'wrist': env['WRIST_CAM_DEVICE']}, fps=30)
            def capture():
                try:
                    while not stop.is_set():
                        frames = manager.get_frames()
                        encoded = {}
                        for role, frame in frames.items():
                            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if not ok:
                                raise RuntimeError('JPEG encoding failed')
                            encoded[role] = jpg.tobytes()
                        with mutex:
                            state.update(frames=encoded, sequence=state['sequence'] + 1,
                                         updated=time.monotonic(), error=None)
                except Exception as exc:
                    with mutex:
                        state.update(frames={}, error=str(exc))
            worker = threading.Thread(target=capture, daemon=True)
            worker.start()
            print('双相机预览：http://127.0.0.1:8082/（无机器人连接、无录制）', flush=True)
            server.serve_forever(poll_interval=0.1)
        finally:
            stop.set()
            server.server_close()
            if worker:
                worker.join(timeout=3)
            if manager:
                manager.release()
            print('预览已退出，相机资源已释放。', flush=True)


if __name__ == '__main__':
    def interrupt(*args):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        main()
    except KeyboardInterrupt:
        pass
