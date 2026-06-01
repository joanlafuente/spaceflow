#!/usr/bin/env python3
"""
Ollama-compatible HTTP proxy for the SQ Editor "AI Generate" button.

  - Default: POST /api/chat runs ollama_infer.sh on a GPU node via Slurm (srun).
  - Optional: set SQ_OLLAMA_FORWARD (e.g. http://127.0.0.1:11436) to forward
    to a local ollama serve instead (no GPU job).

Environment (after setup_ollama.sh installs this script):
  OLLAMA_BASE   — your Ollama install root (substituted by setup script)
  SQ_SLURM_PARTITION, SQ_SLURM_ACCOUNT, SQ_SLURM_GPUS, SQ_SLURM_TIME — Slurm (--gpus only)
  SQ_PROXY_PORT — listen port (default 11434)
  SQ_OLLAMA_FORWARD — if set, forward instead of srun
"""

from __future__ import annotations

import http.server
import json
import os
import os.path
import subprocess
import tempfile
import urllib.error
import urllib.request

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "__OLLAMA_BASE__")
INFER_SCRIPT = os.path.join(OLLAMA_BASE, "scripts", "ollama_infer.sh")
FORWARD = os.environ.get("SQ_OLLAMA_FORWARD", "").strip().rstrip("/")

PARTITION = os.environ.get("SQ_SLURM_PARTITION", "interactive")
ACCOUNT = os.environ.get("SQ_SLURM_ACCOUNT", "3dv")
GPUS = os.environ.get("SQ_SLURM_GPUS", "1").strip()
TIME_LIMIT = os.environ.get("SQ_SLURM_TIME", "00:15:00")
PORT = int(os.environ.get("SQ_PROXY_PORT", "11434"))
SRUN_TIMEOUT = int(os.environ.get("SQ_SRUN_TIMEOUT_SEC", "180"))
EXTRA_ARGS = os.environ.get("SQ_SLURM_EXTRA_ARGS", "").strip()


def _split_args(s: str) -> list[str]:
    # Tiny shell-like splitter for env-provided args (no quotes support).
    return [tok for tok in s.split() if tok]


def _drop_gres_tokens(tokens: list[str]) -> list[str]:
    """Strip --gres; cluster uses --gpus only."""
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t.startswith("--gres="):
            i += 1
            continue
        if t == "--gres":
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def _append_srun_extras(cmd: list[str], extra: str) -> None:
    if not extra.strip():
        return
    raw = _split_args(extra)
    cleaned = _drop_gres_tokens(raw)
    if len(cleaned) < len(raw):
        print(
            "[sq-ollama-proxy] Removed --gres from SQ_SLURM_EXTRA_ARGS; this site uses --gpus only.",
            flush=True,
        )
    cmd.extend(cleaned)


class Handler(http.server.BaseHTTPRequestHandler):
    def _cors(self) -> None:
        # Allow the UI (dev server on :5173) to call the proxy on :11434.
        # Without these headers, browsers block the request (CORS) and the UI reports "Cannot reach Ollama".
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[sq-ollama-proxy] {fmt % args}", flush=True)

    def do_OPTIONS(self) -> None:
        # CORS preflight for POST /api/chat
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/", ""):
            self.send_response(200)
            self._cors()
            self.end_headers()
            self.wfile.write(b"Ollama is running")
            return
        if self.path == "/api/tags":
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"models":[]}')
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"

        if FORWARD:
            self._forward(body)
            return

        if not os.path.isfile(INFER_SCRIPT):
            self._send_json(
                500,
                {
                    "error": {
                        "message": f"Missing infer script: {INFER_SCRIPT}. Run bash sq_ui/setup_ollama.sh"
                    }
                },
            )
            return

        log_dir = os.path.join(OLLAMA_BASE, "logs")
        os.makedirs(log_dir, exist_ok=True)
        fd, req_path = tempfile.mkstemp(suffix=".json", dir=log_dir, text=False)
        try:
            os.write(fd, body)
        finally:
            os.close(fd)

        try:
            env = os.environ.copy()
            env["OLLAMA_BASE"] = OLLAMA_BASE
            cmd: list[str] = [
                "srun",
                f"--partition={PARTITION}",
                f"--account={ACCOUNT}",
                f"--time={TIME_LIMIT}",
                "--job-name=sq_ollama",
                f"--gpus={GPUS or '1'}",
            ]
            _append_srun_extras(cmd, EXTRA_ARGS)
            cmd.extend([INFER_SCRIPT, req_path])
            p = subprocess.run(cmd, capture_output=True, env=env, timeout=SRUN_TIMEOUT)
            if p.returncode != 0:
                err = p.stderr.decode("utf-8", errors="replace")[:8000]
                out = p.stdout.decode("utf-8", errors="replace")[:2000]
                self._send_json(
                    500,
                    {
                        "error": {
                            "message": f"srun failed ({p.returncode}). stderr: {err}\nstdout: {out}"
                        }
                    },
                )
                return
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(p.stdout)
        except subprocess.TimeoutExpired:
            self._send_json(504, {"error": {"message": "GPU inference / srun timed out"}})
        except FileNotFoundError:
            self._send_json(
                500,
                {"error": {"message": "srun not found — is this a Slurm login node?"}},
            )
        finally:
            try:
                os.unlink(req_path)
            except OSError:
                pass

    def _forward(self, body: bytes) -> None:
        url = FORWARD.rstrip("/") + "/api/chat"
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                rbody = resp.read()
                self.send_response(resp.status)
                self._cors()
                ct = resp.headers.get("Content-Type", "application/json")
                self.send_header("Content-Type", ct)
                self.end_headers()
                self.wfile.write(rbody)
        except urllib.error.HTTPError as e:
            err_body = e.read() if e.fp else b""
            self.send_response(e.code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err_body if err_body else json.dumps({"error": str(e)}).encode())
        except urllib.error.URLError as e:
            self._send_json(502, {"error": {"message": f"Forward failed: {e}"}})

    def _send_json(self, code: int, obj: object) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    os.makedirs(os.path.join(OLLAMA_BASE, "logs"), exist_ok=True)
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(
        f"[sq-ollama-proxy] OLLAMA_BASE={OLLAMA_BASE}\n"
        f"[sq-ollama-proxy] Listening on 0.0.0.0:{PORT}  (POST /api/chat)\n"
        f"[sq-ollama-proxy] Forward mode: {FORWARD or 'off (Slurm GPU)'}\n",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
