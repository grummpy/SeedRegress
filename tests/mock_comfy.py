"""A tiny ComfyUI stand-in: /prompt, /history, /view, /queue, /system_stats, /interrupt."""

from __future__ import annotations

import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import parse_qs, urlparse

from PIL import Image


def render_png(seed: int, salt: int = 0) -> bytes:
    image = Image.new("RGB", (64, 64))
    pixels = image.load()
    rng = random.Random(int(seed) + salt * 10000)
    for y in range(64):
        for x in range(64):
            pixels[x, y] = (rng.randint(10, 50), rng.randint(20, 70), rng.randint(50, 110))
    shift = 18 * (salt % 3)
    for y in range(8, 28):
        for x in range(6 + shift, 26 + shift):
            if x < 64:
                pixels[x, y] = (240, 30, 140) if salt else (24, 18, 48)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class MockState:
    def __init__(self):
        self.prompts = {}
        self.history = {}
        self.images = {}
        self.queue_running = []
        self.queue_pending = []
        self.deleted = []
        self.interrupted = 0
        self.cleared = False
        self.calls = []
        self.bodies = []
        self.salt = 0
        self.system_stats = {
            "system": {
                "comfyui_version": "0.3.43",
                "python_version": "3.12.3",
                "pytorch_version": "2.5.1",
                "argv": ["should-not-be-stored"],
            },
            "devices": [{"name": "Mock GPU", "type": "cuda"}],
        }


class MockComfy:
    def __init__(self):
        self.state = MockState()
        handler = _handler(self.state)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> str:
        self.thread.start()
        port = self.httpd.server_address[1]
        return f"127.0.0.1:{port}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def prompt_posts(self) -> list:
        return [body for method, path, body in self.state.bodies if method == "POST" and path == "/prompt"]


def _handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            state.calls.append(("GET", parsed.path))
            if parsed.path == "/queue":
                self._json(
                    {"queue_running": state.queue_running, "queue_pending": state.queue_pending}
                )
            elif parsed.path == "/system_stats":
                self._json(state.system_stats)
            elif parsed.path.startswith("/history/"):
                prompt_id = parsed.path.rsplit("/", 1)[-1]
                entry = state.history.get(prompt_id)
                self._json({prompt_id: entry} if entry else {})
            elif parsed.path == "/view":
                name = parse_qs(parsed.query).get("filename", [""])[0]
                blob = state.images.get(name)
                if blob is None:
                    self._bytes(b"missing", "text/plain", 404)
                else:
                    self._bytes(blob, "image/png")
            else:
                self._bytes(b"no", "text/plain", 404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
            state.calls.append(("POST", parsed.path))
            state.bodies.append(("POST", parsed.path, body))
            if parsed.path == "/prompt":
                prompt_id = f"prompt-{len(state.prompts) + 1}"
                seed = int(body["prompt"]["3"]["inputs"]["seed"])
                filename = f"{prompt_id}.png"
                state.images[filename] = render_png(seed, state.salt)
                state.history[prompt_id] = {
                    "outputs": {
                        "9": {
                            "images": [{"filename": filename, "subfolder": "", "type": "output"}]
                        }
                    },
                    "status": {"completed": True, "status_str": "success"},
                }
                state.prompts[prompt_id] = body
                self._json({"prompt_id": prompt_id, "number": len(state.prompts)})
            elif parsed.path == "/interrupt":
                state.interrupted += 1
                self._json({"ok": True})
            elif parsed.path == "/queue":
                if "clear" in body:
                    state.cleared = True
                delete = list(body.get("delete") or [])
                state.deleted.extend(delete)
                state.queue_pending = [
                    item
                    for item in state.queue_pending
                    if (item[1] if isinstance(item, list) else item.get("prompt_id")) not in delete
                ]
                self._json({"ok": True})
            else:
                self._bytes(b"no", "text/plain", 404)

        def _json(self, payload: dict, status: int = 200) -> None:
            self._bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

        def _bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler
