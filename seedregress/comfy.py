"""ComfyUI HTTP and websocket client.

Every method contacts the network. Constructing the client requires
``confirmed=True``, which only the explicit Run-on-GPU path sets.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode, urlparse

from seedregress.errors import ComfyError, GpuConfirmationRequired

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _urlopen(request: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(request, timeout=timeout)


def _create_connection(*args, **kwargs):
    return socket.create_connection(*args, **kwargs)


def normalize_base(host: str) -> str:
    raw = (host or "").strip()
    if not raw:
        raise ComfyError("ComfyUI host is empty. Set it in the app before contacting the machine.")
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ComfyError("ComfyUI host must start with http:// or https://")
    if not parsed.hostname:
        raise ComfyError("ComfyUI host is missing a hostname")
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 8188
    return f"{parsed.scheme}://{parsed.hostname}:{port}"


class ComfyClient:
    def __init__(
        self,
        host: str,
        *,
        confirmed: bool,
        install_id: str,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
        use_websocket: bool = True,
        client_id: str | None = None,
    ):
        if confirmed is not True:
            raise GpuConfirmationRequired(
                "Refusing to contact ComfyUI without an explicit GPU confirmation."
            )
        self.base = normalize_base(host)
        self.install_id = install_id
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.use_websocket = use_websocket
        self.client_id = client_id or install_id

    def system_stats(self) -> dict:
        payload = self._request("GET", "/system_stats")
        if not isinstance(payload, dict):
            raise ComfyError("ComfyUI /system_stats did not return JSON")
        return payload

    def get_queue(self) -> dict:
        payload = self._request("GET", "/queue")
        if not isinstance(payload, dict):
            raise ComfyError("ComfyUI /queue did not return JSON")
        payload.setdefault("queue_running", [])
        payload.setdefault("queue_pending", [])
        return payload

    def submit(self, graph: dict, *, run_id: str, case_id: str) -> str:
        body = {
            "prompt": graph,
            "client_id": self.client_id,
            "extra_data": {
                "seedregress_install": self.install_id,
                "seedregress_run": run_id,
                "seedregress_case": case_id,
            },
        }
        payload = self._request("POST", "/prompt", body)
        if not isinstance(payload, dict) or "prompt_id" not in payload:
            message = payload.get("error") if isinstance(payload, dict) else payload
            raise ComfyError(f"ComfyUI rejected the prompt: {message}")
        return str(payload["prompt_id"])

    def history(self, prompt_id: str) -> dict:
        payload = self._request("GET", f"/history/{prompt_id}")
        if not isinstance(payload, dict):
            raise ComfyError("ComfyUI /history did not return JSON")
        return payload

    def view(self, filename: str, subfolder: str = "", kind: str = "output") -> bytes:
        query = urlencode({"filename": filename, "subfolder": subfolder, "type": kind})
        payload = self._request("GET", f"/view?{query}")
        if not isinstance(payload, (bytes, bytearray)) or not payload:
            raise ComfyError("ComfyUI /view did not return an image")
        return bytes(payload)

    def delete_queue_items(self, prompt_ids: list[str]) -> None:
        if not prompt_ids:
            return
        # Delete by id only. Never send "clear", which would drop every job.
        self._request("POST", "/queue", {"delete": list(prompt_ids)})

    def interrupt(self) -> None:
        self._request("POST", "/interrupt", {})

    def wait_for_image(self, prompt_id: str, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        ws = self._open_websocket() if self.use_websocket else None
        try:
            while True:
                if ws is not None:
                    self._drain_websocket(ws, 0.05)
                entry = self.history(prompt_id).get(prompt_id)
                image = image_from_history(entry)
                if image is not None:
                    return image
                error = error_from_history(entry)
                if error:
                    raise ComfyError(error)
                if time.monotonic() >= deadline:
                    raise ComfyError(
                        f"Timed out after {timeout:.0f}s waiting for ComfyUI prompt {prompt_id}"
                    )
                time.sleep(self.poll_interval)
        finally:
            if ws is not None:
                ws.close()

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None):
        url = self.base + path
        data = None
        headers = {"User-Agent": "SeedRegress", "Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with _urlopen(request, timeout=timeout or self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ComfyError(f"ComfyUI {method} {path} failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise ComfyError(f"Could not reach ComfyUI at {self.base}: {exc.reason}") from exc
        if path.startswith("/view"):
            return raw
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ComfyError(f"ComfyUI {path} returned non-JSON") from exc

    def _open_websocket(self) -> socket.socket | None:
        parsed = urlparse(self.base)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            sock = _create_connection((parsed.hostname, port), timeout=2)
        except OSError:
            return None
        try:
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            host = f"{parsed.hostname}:{port}"
            path = f"/ws?clientId={self.client_id}"
            request = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {host}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n"
                "\r\n"
            )
            sock.sendall(request.encode("ascii"))
            header, pending = _read_headers(sock)
            if b" 101 " not in header.split(b"\r\n", 1)[0]:
                sock.close()
                return None
            accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
            if accept.encode() not in header:
                sock.close()
                return None
            sock.settimeout(0.05)
            return _SocketBuffer(sock, pending)
        except OSError:
            sock.close()
            return None

    def _drain_websocket(self, sock: socket.socket, budget: float) -> None:
        end = time.monotonic() + budget
        while time.monotonic() < end:
            try:
                opcode, payload = read_frame(sock)
            except (TimeoutError, socket.timeout, OSError):
                return
            if opcode == 0x9:  # ping
                try:
                    send_frame(sock, 0xA, payload)
                except OSError:
                    return
            elif opcode == 0x8:
                return


def read_frame(sock: socket.socket) -> tuple[int, bytes]:
    header = _recv_exact(sock, 2)
    opcode = header[0] & 0x0F
    masked = (header[1] & 0x80) != 0
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack(">H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack(">Q", _recv_exact(sock, 8))[0]
    mask = _recv_exact(sock, 4) if masked else b""
    payload = _recv_exact(sock, length) if length else b""
    if masked:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return opcode, payload


def send_frame(sock: socket.socket, opcode: int, payload: bytes) -> None:
    mask = os.urandom(4)
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack(">H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack(">Q", length))
    sock.sendall(bytes(header) + mask + masked)


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ComfyError("WebSocket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _SocketBuffer:
    """Socket plus bytes already pulled off while reading the handshake."""

    def __init__(self, sock: socket.socket, pending: bytes = b""):
        self._sock = sock
        self._pending = pending

    def recv(self, size: int) -> bytes:
        if self._pending:
            chunk = self._pending[:size]
            self._pending = self._pending[size:]
            return chunk
        return self._sock.recv(size)

    def sendall(self, data: bytes) -> None:
        self._sock.sendall(data)

    def settimeout(self, value: float | None) -> None:
        self._sock.settimeout(value)

    def close(self) -> None:
        self._sock.close()


def _read_headers(sock: socket.socket) -> tuple[bytes, bytes]:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(1024)
        if not chunk:
            break
        data += chunk
        if len(data) > 65536:
            break
    header, marker, rest = data.partition(b"\r\n\r\n")
    if not marker:
        return data, b""
    return header + marker, rest


def image_from_history(entry: dict | None) -> dict | None:
    if not isinstance(entry, dict):
        return None
    outputs = entry.get("outputs") or {}
    for node in outputs.values():
        images = node.get("images") if isinstance(node, dict) else None
        if images:
            image = images[0]
            if image.get("filename"):
                return {
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder") or "",
                    "type": image.get("type") or "output",
                }
    return None


def error_from_history(entry: dict | None) -> str | None:
    if not isinstance(entry, dict):
        return None
    status = entry.get("status") or {}
    status_str = str(status.get("status_str") or "")
    if status_str == "error":
        messages = status.get("messages") or []
        return f"ComfyUI reported an error: {messages or status_str}"
    if status.get("completed") and image_from_history(entry) is None:
        return "ComfyUI finished the prompt without an image"
    return None


def queue_counts(payload: dict) -> tuple[int, int]:
    running = payload.get("queue_running") or []
    pending = payload.get("queue_pending") or []
    return len(running), len(pending)


def parse_queue_item(item) -> tuple[str, dict]:
    if isinstance(item, dict):
        extra = item.get("extra_data") or {}
        return str(item.get("prompt_id") or ""), extra if isinstance(extra, dict) else {}
    if isinstance(item, list) and len(item) >= 2:
        extra = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
        return str(item[1]), extra
    return "", {}


def item_is_ours(prompt_id: str, extra: dict, owned_ids: set[str], install_id: str) -> bool:
    if prompt_id and prompt_id in owned_ids:
        return True
    if extra.get("seedregress_install") == install_id:
        return True
    nested = extra.get("extra_pnginfo")
    if isinstance(nested, dict) and nested.get("seedregress_install") == install_id:
        return True
    return False
