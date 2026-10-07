"""Local web UI. Listens on 127.0.0.1 only."""

from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from seedregress.config import load_config, save_config
from seedregress.errors import SeedRegressError
from seedregress.paths import package_root
from seedregress.runner import cancel_mine, execute, plan_suite
from seedregress.suite import load_suite

_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json",
}


class App:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.root = package_root()
        self.lock = threading.Lock()

    def examples(self) -> list[dict]:
        folder = self.root / "examples"
        found = []
        if folder.is_dir():
            for path in sorted(folder.glob("*.suite.json")):
                found.append({"name": path.stem, "path": str(path)})
        return found


def make_server(data_dir: Path, port: int = 0) -> ThreadingHTTPServer:
    data_dir.mkdir(parents=True, exist_ok=True)
    app = App(data_dir)
    handler = _handler_class(app)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.app = app  # type: ignore[attr-defined]
    return server


def serve(*, data_dir: Path, port: int = 0, open_browser: bool = True) -> None:
    httpd = make_server(data_dir, port)
    bound_host, bound_port = httpd.server_address[:2]
    url = f"http://{bound_host}:{bound_port}/"
    print(f"SeedRegress listening on {url}")
    print("Dry-run is the default. Nothing is queued until you press Run on GPU.")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


def _handler_class(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path in {"/", "/index.html"}:
                    self._file(app.root / "seedregress" / "static" / "index.html")
                elif path == "/favicon.ico":
                    self._file(app.root / "assets" / "icon.ico")
                elif path.startswith("/assets/"):
                    self._file(_safe(app.root / "assets", path[len("/assets/") :]))
                elif path.startswith("/static/"):
                    self._file(_safe(app.root / "seedregress" / "static", path[len("/static/") :]))
                elif path.startswith("/media/"):
                    self._file(_safe(app.data_dir, path[len("/media/") :]))
                elif path == "/api/config":
                    self._json(load_config(app.data_dir))
                elif path == "/api/examples":
                    self._json({"examples": app.examples()})
                else:
                    self._json({"error": "Not found"}, status=404)
            except (OSError, ValueError) as exc:
                self._json({"error": str(exc)}, status=400)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                body = self._body()
                if parsed.path == "/api/config":
                    self._json(save_config(app.data_dir, body))
                elif parsed.path == "/api/plan":
                    self._json(self._plan(body).to_dict(app.data_dir))
                elif parsed.path == "/api/run":
                    self._json(self._run(body))
                elif parsed.path == "/api/cancel":
                    self._json(self._cancel(body))
                else:
                    self._json({"error": "Not found"}, status=404)
            except (SeedRegressError, OSError, ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, status=400)

        def _plan(self, body: dict):
            suite, config, host = self._context(body)
            return plan_suite(
                suite,
                data_dir=app.data_dir,
                host=host,
                seconds_per_step=float(config["seconds_per_step"]),
            )

        def _run(self, body: dict) -> dict:
            confirm = body.get("confirm_gpu") is True
            suite, config, host = self._context(body)
            models_root = str(body.get("models_root") or config["models_root"])
            if not confirm:
                outcome = plan_suite(
                    suite,
                    data_dir=app.data_dir,
                    host=host,
                    seconds_per_step=float(config["seconds_per_step"]),
                )
                return outcome.to_dict(app.data_dir)
            if not app.lock.acquire(blocking=False):
                return {
                    "mode": "refused",
                    "message": "A run is already in progress. SeedRegress does not queue a second one.",
                    "prompts_submitted": 0,
                    "queue_checked": False,
                }
            try:
                outcome = execute(
                    suite,
                    confirm_gpu=True,
                    data_dir=app.data_dir,
                    host=host,
                    models_root=models_root,
                    seconds_per_step=float(config["seconds_per_step"]),
                    case_timeout=float(config["case_timeout_seconds"]),
                    poll_interval=float(config["poll_interval_seconds"]),
                    update_baseline=body.get("update_baseline") is True,
                )
                return outcome.to_dict(app.data_dir)
            finally:
                app.lock.release()

        def _cancel(self, body: dict) -> dict:
            if body.get("confirm_cancel") is not True:
                return {
                    "ok": False,
                    "contacted": False,
                    "message": "Cancel was not confirmed. Nothing was contacted.",
                }
            config = load_config(app.data_dir)
            host = str(body.get("host") or config["comfy_host"]).strip()
            return cancel_mine(confirm_gpu=True, data_dir=app.data_dir, host=host)

        def _context(self, body: dict):
            suite_path = body.get("suite_path")
            if not suite_path:
                raise ValueError("Choose a suite file")
            suite = load_suite(Path(str(suite_path)))
            config = load_config(app.data_dir)
            host = str(body.get("host") if body.get("host") is not None else "").strip()
            if not host:
                host = (suite.comfy_host or config["comfy_host"]).strip()
            return suite, config, host

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or "0")
            if length > 2_000_000:
                raise ValueError("Request is too large")
            raw = self.rfile.read(length) if length else b"{}"
            parsed = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(parsed, dict):
                raise ValueError("JSON body must be an object")
            return parsed

        def _file(self, path: Path) -> None:
            if not path.is_file():
                self._json({"error": "Not found"}, status=404)
                return
            data = path.read_bytes()
            kind = _TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
            self._bytes(data, kind or "application/octet-stream")

        def _json(self, payload: dict, status: int = 200) -> None:
            self._bytes(json.dumps(payload).encode("utf-8"), "application/json", status=status)

        def _bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _safe(root: Path, relative: str) -> Path:
    root = root.resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Path escapes the allowed folder")
    return target
