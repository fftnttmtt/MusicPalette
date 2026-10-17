"""Local-only SPECTRA server. Uses the project's existing CUDA environment."""
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, unquote
import argparse
import errno
import hashlib
import json
import re
import tempfile
import threading
import time
import uuid
import webbrowser

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 1024 * 1024 * 1024
UI_BUILD = 'local-epic3-r32'


def frontend_metadata(web_root):
    web_root = Path(web_root).resolve()
    html = (web_root / 'index.html').read_text(encoding='utf-8')
    visuals = (web_root / 'visuals.js').read_text(encoding='utf-8') if (web_root / 'visuals.js').exists() else ''
    build = re.search(r'<meta name="spectra-build" content="([^"]+)"', html)
    gap = re.search(r'EPIC_OUTLINE_GAP\s*=\s*(\d+)', visuals)
    digest = hashlib.sha256()
    for path in sorted(web_root.iterdir()):
        if path.is_file() and path.suffix in ('.html', '.js', '.css'):
            digest.update(path.name.encode('utf-8'))
            digest.update(path.read_bytes())
    return dict(build_id=build.group(1) if build else None,
        epic_outline_gap_css_px=int(gap.group(1)) if gap else None,
        asset_revision=digest.hexdigest()[:12], web_directory=str(web_root))


class Jobs:
    def __init__(self, predictor=None):
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.jobs = {}
        self.resources = {}
        self.predictor = predictor

    def reserve(self):
        with self.lock:
            if any(j['status'] in ('uploading', 'running') for j in self.jobs.values()):
                raise ValueError('上一首音乐仍在分析，请等待完成后再选择。')
            # Keep at most two recent results, avoiding unbounded RAM growth.
            while len(self.jobs) >= 2:
                self.jobs.pop(next(iter(self.jobs)))
            key = uuid.uuid4().hex
            self.jobs[key] = {'status': 'uploading', 'message': '接收音乐…', 'progress': 0.}
            return key

    def snapshot(self, key):
        with self.lock:
            return dict(self.jobs[key]) if key in self.jobs else None

    def update(self, key, **values):
        with self.lock:
            self.jobs[key].update(values)

    def submit(self, key, path, directory, unit, name, digest):
        self.update(key, status='running')
        self.executor.submit(self.run, key, path, directory, unit, name, digest)

    def run(self, key, path, directory, unit, name, digest):
        try:
            if self.predictor is None:
                from spectra_prediction import predict_file
                predictor = predict_file
            else:
                predictor = self.predictor
            bundle = predictor(path, prediction_unit=unit, resources=self.resources,
                audio_name=name, sha256=digest,
                progress=lambda message, fraction: self.update(key, message=message, progress=fraction))
            self.update(key, status='done', progress=1., message='预测完成', result=bundle)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.update(key, status='error', message=f'{type(exc).__name__}: {exc}', progress=0.)
        finally:
            directory.cleanup()


def make_server(port=8765, *, jobs=None, web_root=None):
    jobs = jobs or Jobs()
    web_root = Path(web_root or ROOT / 'spectra_web')

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_root), **kwargs)

        def allowed_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

        def end_headers(self):
            # Local updates must not keep old HTML or renderer modules alive.
            self.send_header('Cache-Control', 'no-store, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('X-Spectra-Build', UI_BUILD)
            super().end_headers()

        def send_head(self):
            for header in ('If-Modified-Since', 'If-None-Match'):
                if header in self.headers:
                    del self.headers[header]
            return super().send_head()

        def send_json(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.allowed_host():
                return self.send_json({'error': '不支持这个访问地址。'}, 403)
            path = urlsplit(self.path).path
            if path == '/api/config':
                from spectra_prediction import DEFAULTS
                return self.send_json({'engine': 'spectra-local-v1', 'defaults': DEFAULTS,
                    'frontend': frontend_metadata(web_root), 'project_directory': str(ROOT)})
            if path.startswith('/api/jobs/'):
                value = jobs.snapshot(path.rsplit('/', 1)[-1])
                return self.send_json(value or {'error': '任务不存在。'}, 200 if value else 404)
            if path.startswith('/api/'):
                return self.send_json({'error': '接口不存在。'}, 404)
            # Static assets only; never expose the project's weights or Python source.
            return super().do_GET()

        def do_POST(self):
            origin = self.headers.get('Origin')
            allowed_origins = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
            if not self.allowed_host() or (origin and origin not in allowed_origins) or self.headers.get('X-Spectra-Client') != '1':
                return self.send_json({'error': '请从本机 SPECTRA 页面选择音乐。'}, 403)
            parsed = urlsplit(self.path)
            if parsed.path != '/api/analyze':
                return self.send_json({'error': '接口不存在。'}, 404)
            unit = parse_qs(parsed.query).get('unit', ['bar'])[0]
            if unit not in ('bar', 'beat'):
                return self.send_json({'error': '无效预测单位。'}, 400)
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_UPLOAD:
                return self.send_json({'error': '音频为空或超过 1 GB。'}, 413)
            try:
                key = jobs.reserve()
            except ValueError as exc:
                return self.send_json({'error': str(exc)}, 409)
            name = Path(unquote(self.headers.get('X-Audio-Name', 'audio.wav')).replace('\\', '/')).name
            suffix = Path(name).suffix.lower()
            if suffix not in ('.mp3', '.wav', '.wave', '.flac', '.ogg', '.oga', '.m4a', '.aac', '.opus', '.aif', '.aiff', '.webm', '.mp4'):
                suffix = '.audio'
            directory = tempfile.TemporaryDirectory(prefix='spectra_')
            path = Path(directory.name) / ('input' + suffix)
            try:
                digest = hashlib.sha256()
                remaining = length
                with path.open('wb') as out:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError('上传未完成，请重试。')
                        out.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                jobs.submit(key, path, directory, unit, name, digest.hexdigest())
                self.send_json({'job_id': key}, 202)
            except Exception as exc:
                directory.cleanup()
                jobs.update(key, status='error', message=str(exc))
                self.send_json({'error': str(exc)}, 400)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.jobs = jobs
    return server


def make_available_server(port=8765, **kwargs):
    # Keep existing servers/jobs untouched; launch this folder on a free port.
    for candidate in ([0] if port == 0 else range(port, min(65535, port + 20) + 1)):
        try:
            return make_server(candidate, **kwargs)
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, 10048) and getattr(exc, 'winerror', None) != 10048:
                raise
    raise OSError('附近端口均被占用，请使用 --port 指定其他端口。')


def main():
    parser = argparse.ArgumentParser(description='SPECTRA 本地 CUDA 音乐情绪几何')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--convert', type=Path, help='把原完整预测 JSON 转成网站导入格式，无需重新预测')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.convert:
        from spectra_prediction import make_spectra_bundle
        bundle = make_spectra_bundle(json.loads(args.convert.read_text(encoding='utf-8-sig')))
        destination = args.output or args.convert.with_suffix('.spectra.json')
        destination.write_text(json.dumps(bundle, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        print(destination)
        return
    metadata = frontend_metadata(ROOT / 'spectra_web')
    if metadata['build_id'] != UI_BUILD or metadata['epic_outline_gap_css_px'] != 3:
        raise SystemExit('页面文件未完整更新：请将 spectra_web 文件夹与 spectra_server.py 一起覆盖到同一项目目录。')
    server = make_available_server(args.port)
    url = f'http://127.0.0.1:{server.server_port}/?build={UI_BUILD}&session={uuid.uuid4().hex[:8]}'
    if args.port and server.server_port != args.port:
        print(f'Port {args.port} is occupied; this instance uses {server.server_port}.', flush=True)
    print(f'SPECTRA: {url}\nBuild: {UI_BUILD} / epic gap: 3 CSS px\nProject: {ROOT}\nPage files: {metadata["web_directory"]}\nCUDA / Beat This! final0 / Music2Emo · Ctrl+C to exit', flush=True)
    if not args.no_browser:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.jobs.executor.shutdown(wait=True)


if __name__ == '__main__':
    main()
