#!/usr/bin/env python3
"""Print a Superquadric Editor URL for a local .npz file."""

from __future__ import annotations

import argparse
import http.client
from pathlib import Path
from urllib.parse import quote


def editor_alive(port: int) -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=0.35)
        conn.request("HEAD", "/")
        resp = conn.getresponse()
        return 200 <= resp.status < 500
    except OSError:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def detect_port(start: int = 5173, end: int = 5199) -> int:
    active = [port for port in range(start, end + 1) if editor_alive(port)]
    if not active:
        raise SystemExit(
            "Could not find a running Vite editor on ports 5173-5199. "
            "Start it with: cd sq_ui/app && npm run dev -- --host 0.0.0.0"
        )
    return max(active)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="Path to a .npz file")
    parser.add_argument("--port", type=int, default=0, help="Vite port; auto-detects by default")
    parser.add_argument("--host", default="localhost", help="Browser host; default: localhost")
    args = parser.parse_args()

    npz = args.npz.expanduser().resolve()
    if npz.suffix.lower() != ".npz":
        raise SystemExit(f"Expected a .npz file, got: {npz}")
    if not npz.is_file():
        raise SystemExit(f"File not found: {npz}")

    port = args.port or detect_port()
    url = f"http://{args.host}:{port}/?npz={quote(str(npz), safe='')}"
    print(url)


if __name__ == "__main__":
    main()
