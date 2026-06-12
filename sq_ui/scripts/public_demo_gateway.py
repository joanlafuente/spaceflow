#!/usr/bin/env python3
"""Authenticated public gateway for a controlled SpaceFlow demo link."""

from __future__ import annotations

import base64
import hmac
import http.client
import http.server
import json
import mimetypes
import os
import posixpath
import shutil
from pathlib import Path
from urllib.parse import unquote, urlsplit


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
HOST = os.environ.get("SQ_PUBLIC_HOST", "127.0.0.1").strip() or "127.0.0.1"
PORT = int(os.environ.get("SQ_PUBLIC_PORT", "11481"))
STATIC_ROOT = Path(
    os.environ.get("SQ_PUBLIC_STATIC_ROOT", str(REPO_ROOT / "sq_ui" / "app" / "dist"))
).expanduser().resolve()
BACKEND_URL = os.environ.get("SQ_PUBLIC_BACKEND", "http://127.0.0.1:11480").strip()
PUBLIC_USER = os.environ.get("SQ_PUBLIC_USER", "spaceflow")
PUBLIC_PASSWORD = os.environ.get("SQ_PUBLIC_PASSWORD", "")
MAX_UPLOAD_MB = float(os.environ.get("SQ_PUBLIC_MAX_UPLOAD_MB", "64") or "64")
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

BACKEND = urlsplit(BACKEND_URL)
if BACKEND.scheme not in {"http", "https"}:
    raise SystemExit("SQ_PUBLIC_BACKEND must be an http:// or https:// URL")

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
ALLOWED_SPACEFLOW_ROUTES = {
    "/spaceflow/runs/status": {"GET"},
    "/spaceflow/runs/log": {"GET"},
    "/spaceflow/runs/file": {"GET"},
    "/spaceflow/runs/start": {"POST"},
    "/spaceflow/runs/stop": {"POST"},
}


class PayloadTooLargeError(Exception):
    pass


class ThreadingHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class PublicDemoGateway(http.server.BaseHTTPRequestHandler):
    server_version = "SpaceFlowPublicGateway/0.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[sq-public-gateway] {fmt % args}", flush=True)

    def do_OPTIONS(self) -> None:
        if not self._require_auth():
            return
        self.send_response(204)
        self._security_headers()
        self.send_header("Allow", "GET,POST,OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:
        if not self._require_auth():
            return
        self._route(send_body=False)

    def do_GET(self) -> None:
        if not self._require_auth():
            return
        self._route(send_body=True)

    def do_POST(self) -> None:
        if not self._require_auth():
            return
        if self._request_body_too_large():
            self._send_json(
                413,
                {"error": {"message": f"Upload is larger than {MAX_UPLOAD_MB:g} MB."}},
            )
            self.close_connection = True
            return
        self._route(send_body=True)

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        body = b"Authentication required.\n"
        self.send_response(401)
        self._security_headers()
        self.send_header("WWW-Authenticate", 'Basic realm="SpaceFlow Demo"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        return False

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "basic" or not token:
            return False
        try:
            decoded = base64.b64decode(token, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return False
        username, sep, password = decoded.partition(":")
        if not sep:
            return False
        return (
            hmac.compare_digest(username, PUBLIC_USER)
            and hmac.compare_digest(password, PUBLIC_PASSWORD)
        )

    def _request_body_too_large(self) -> bool:
        raw = self.headers.get("Content-Length")
        if raw is None:
            return False
        try:
            return int(raw) > MAX_UPLOAD_BYTES
        except ValueError:
            return True

    def _route(self, *, send_body: bool) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path or "/"
        if path == "/spaceflow/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "spaceflow-public-demo",
                    "backend": "private",
                    "max_upload_mb": MAX_UPLOAD_MB,
                },
                send_body=send_body,
            )
            return
        if path.startswith("/spaceflow/assets/"):
            self._send_json(
                403,
                {"error": {"message": "Saved asset endpoints are disabled in public demo mode."}},
                send_body=send_body,
            )
            return
        if path.startswith("/spaceflow/"):
            allowed_methods = ALLOWED_SPACEFLOW_ROUTES.get(path)
            if allowed_methods is None or self.command not in allowed_methods:
                self._send_json(404, {"error": {"message": "Public demo route is not available."}}, send_body=send_body)
                return
            self._proxy_spaceflow(send_body=send_body)
            return
        if self.command not in {"GET", "HEAD"}:
            self._send_json(405, {"error": {"message": "Method not allowed."}}, send_body=send_body)
            return
        self._serve_static(path, send_body=send_body)

    def _proxy_spaceflow(self, *, send_body: bool) -> None:
        try:
            body = self._read_request_body()
        except PayloadTooLargeError:
            self._send_json(
                413,
                {"error": {"message": f"Upload is larger than {MAX_UPLOAD_MB:g} MB."}},
                send_body=send_body,
            )
            self.close_connection = True
            return

        connection_cls = http.client.HTTPSConnection if BACKEND.scheme == "https" else http.client.HTTPConnection
        host = BACKEND.hostname or "127.0.0.1"
        port = BACKEND.port or (443 if BACKEND.scheme == "https" else 80)
        backend_prefix = BACKEND.path.rstrip("/")
        target = f"{backend_prefix}{self.path}" if backend_prefix else self.path
        headers = self._proxy_request_headers()

        try:
            conn = connection_cls(host, port, timeout=300)
            conn.request(self.command, target, body=body, headers=headers)
            response = conn.getresponse()
            payload = response.read()
        except OSError as exc:
            self._send_json(502, {"error": {"message": f"Private SpaceFlow service is unavailable: {exc}"}}, send_body=send_body)
            return
        finally:
            try:
                conn.close()
            except UnboundLocalError:
                pass

        self.send_response(response.status, response.reason)
        self._security_headers()
        for key, value in response.getheaders():
            lower = key.lower()
            if lower in HOP_BY_HOP_HEADERS or lower in {"content-length", "cache-control"}:
                continue
            if lower.startswith("access-control-"):
                continue
            self.send_header(key, value)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if send_body:
            self.wfile.write(payload)

    def _read_request_body(self) -> bytes | None:
        content_length = self.headers.get("Content-Length")
        if content_length:
            length = int(content_length)
            if length > MAX_UPLOAD_BYTES:
                raise PayloadTooLargeError()
            return self.rfile.read(length)

        transfer_encoding = self.headers.get("Transfer-Encoding", "").lower()
        if "chunked" not in transfer_encoding:
            return None

        chunks: list[bytes] = []
        total = 0
        while True:
            line = self.rfile.readline()
            if not line:
                break
            size_raw = line.split(b";", 1)[0].strip()
            try:
                size = int(size_raw, 16)
            except ValueError:
                break
            if size == 0:
                while True:
                    trailer = self.rfile.readline()
                    if trailer in {b"", b"\r\n", b"\n"}:
                        break
                break
            total += size
            if total > MAX_UPLOAD_BYTES:
                raise PayloadTooLargeError()
            chunks.append(self.rfile.read(size))
            self.rfile.read(2)
        return b"".join(chunks)

    def _proxy_request_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        for key, value in self.headers.items():
            lower = key.lower()
            if lower in HOP_BY_HOP_HEADERS or lower in {"authorization", "content-length"}:
                continue
            headers[key] = value
        backend_host = BACKEND.netloc
        headers["Host"] = backend_host
        headers["X-Forwarded-Host"] = self.headers.get("Host", "")
        headers["X-Forwarded-Proto"] = "https"
        return headers

    def _serve_static(self, request_path: str, *, send_body: bool) -> None:
        target = self._static_target(request_path)
        if target is None:
            self._send_json(403, {"error": {"message": "Forbidden."}}, send_body=send_body)
            return
        if not target.is_file():
            if "." in Path(request_path).name:
                self._send_json(404, {"error": {"message": "Not found."}}, send_body=send_body)
                return
            target = STATIC_ROOT / "index.html"
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        size = target.stat().st_size
        self.send_response(200)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        if target.name == "index.html":
            self.send_header("Cache-Control", "no-store")
        else:
            self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        if send_body:
            with target.open("rb") as fh:
                shutil.copyfileobj(fh, self.wfile)

    def _static_target(self, request_path: str) -> Path | None:
        decoded = unquote(request_path.split("?", 1)[0])
        if decoded in {"", "/"}:
            rel = "index.html"
        else:
            rel = posixpath.normpath(decoded.lstrip("/"))
        if rel.startswith("../") or rel == "..":
            return None
        target = (STATIC_ROOT / rel).resolve()
        if target != STATIC_ROOT and STATIC_ROOT not in target.parents:
            return None
        if target.is_dir():
            target = target / "index.html"
        return target

    def _send_json(self, code: int, payload: object, *, send_body: bool = True) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self._security_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")


def main() -> None:
    if not PUBLIC_PASSWORD:
        raise SystemExit("Set SQ_PUBLIC_PASSWORD before starting the public demo gateway.")
    if not (STATIC_ROOT / "index.html").is_file():
        raise SystemExit(
            f"Missing built UI at {STATIC_ROOT}. Run: cd sq_ui/app && VITE_PUBLIC_DEMO=1 npm run build"
        )
    server = ThreadingHTTPServer((HOST, PORT), PublicDemoGateway)
    print(
        f"[sq-public-gateway] Listening on http://{HOST}:{PORT}\n"
        f"[sq-public-gateway] Static root: {STATIC_ROOT}\n"
        f"[sq-public-gateway] Backend: {BACKEND_URL}\n"
        f"[sq-public-gateway] Max upload: {MAX_UPLOAD_MB:g} MB\n",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
