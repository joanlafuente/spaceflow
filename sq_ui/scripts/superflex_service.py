#!/usr/bin/env python3
"""HTTP service for SuperFlex-backed superquadric generation."""

from __future__ import annotations

import cgi
import http.server
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
SUPERFLEX_BASE = Path(os.environ.get("SUPERFLEX_BASE", str(REPO_ROOT / "superflex_ui"))).expanduser()
PORT = int(os.environ.get("SQ_SUPERFLEX_PORT", "11436"))
RUN_TIMEOUT = int(os.environ.get("SQ_SUPERFLEX_TIMEOUT_SEC", "900"))
FORCE_LOCAL = os.environ.get("SQ_SUPERFLEX_FORCE_LOCAL", "").strip() == "1"
PARTITION = os.environ.get("SQ_SUPERFLEX_SLURM_PARTITION", "interactive")
ACCOUNT = os.environ.get("SQ_SUPERFLEX_SLURM_ACCOUNT", "3dv")
GPUS = os.environ.get("SQ_SUPERFLEX_SLURM_GPUS", "1").strip()
TIME_LIMIT = os.environ.get("SQ_SUPERFLEX_SLURM_TIME", "00:20:00")
EXTRA_ARGS = os.environ.get("SQ_SUPERFLEX_SLURM_EXTRA_ARGS", "").strip()
TORCH_CUDA_ARCH_LIST = os.environ.get("SQ_SUPERFLEX_TORCH_CUDA_ARCH_LIST", "7.5;8.9+PTX").strip()


def _default_python_bin() -> str:
    candidates = [
        REPO_ROOT / "envs" / "guideflow3d" / "bin" / "python",
        SUPERFLEX_BASE / "venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


PYTHON_BIN = os.environ.get("SQ_SUPERFLEX_PYTHON", _default_python_bin())
CHECKPOINT_DIR = Path(
    os.environ.get(
        "SQ_SUPERFLEX_CHECKPOINT_DIR",
        str(REPO_ROOT / "superflex" / "weights"),
    )
).expanduser()
WORK_DIR = Path(os.environ.get("SQ_SUPERFLEX_WORK_DIR", str(REPO_ROOT / "superflex"))).expanduser()
INFER_SCRIPT = Path(os.environ.get("SQ_SUPERFLEX_INFER", str(SCRIPT_PATH.parent / "superflex_infer.py"))).expanduser()
RUNS_DIR = Path(os.environ.get("SQ_SUPERFLEX_RUNS", str(SUPERFLEX_BASE / "runs"))).expanduser()
TMP_DIR = Path(os.environ.get("SQ_SUPERFLEX_TMP", str(SUPERFLEX_BASE / "tmp"))).expanduser()
LOGS_DIR = Path(os.environ.get("SQ_SUPERFLEX_LOGS", str(SUPERFLEX_BASE / "logs"))).expanduser()


def _build_child_env() -> dict[str, str]:
    env = os.environ.copy()
    python_path = Path(PYTHON_BIN)
    venv_bin = python_path.parent
    existing_path = env.get("PATH", "")
    env["PATH"] = f"{venv_bin}:{existing_path}" if existing_path else str(venv_bin)
    env.setdefault("VIRTUAL_ENV", str(venv_bin.parent))
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{WORK_DIR}:{existing_pythonpath}" if existing_pythonpath else str(WORK_DIR)
    arch_tag = TORCH_CUDA_ARCH_LIST.replace(";", "_").replace("+", "p").replace(".", "_").replace(" ", "") or "default"
    env.setdefault("TORCH_EXTENSIONS_DIR", str(TMP_DIR / f"torch_extensions_{arch_tag}"))
    if TORCH_CUDA_ARCH_LIST:
        env.setdefault("TORCH_CUDA_ARCH_LIST", TORCH_CUDA_ARCH_LIST)
    return env


def _split_args(s: str) -> list[str]:
    return [tok for tok in s.split() if tok]


def _drop_gres_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
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


def _should_use_srun() -> bool:
    if FORCE_LOCAL or os.environ.get("SLURM_JOB_ID"):
        return False
    return shutil.which("srun") is not None


def _wrap_with_srun(cmd: list[str]) -> list[str]:
    srun_cmd = [
        "srun",
        f"--partition={PARTITION}",
        f"--account={ACCOUNT}",
        f"--time={TIME_LIMIT}",
        "--job-name=sq_superflex",
        f"--gpus={GPUS or '1'}",
    ]
    srun_cmd.extend(_drop_gres_tokens(_split_args(EXTRA_ARGS)))
    srun_cmd.extend(cmd)
    return srun_cmd


def _parse_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "SuperFlexService/0.1"

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[sq-superflex] {fmt % args}", flush=True)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"", "/", "/superflex/health"}:
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "superflex",
                    "python": PYTHON_BIN,
                    "checkpoint_dir": str(CHECKPOINT_DIR),
                    "work_dir": str(WORK_DIR),
                    "infer_script": str(INFER_SCRIPT),
                    "runs_dir": str(RUNS_DIR),
                    "uses_srun": _should_use_srun(),
                },
            )
            return
        if parsed.path == "/superflex/result":
            query = parse_qs(parsed.query)
            run_id = (query.get("run_id") or [""])[0]
            if not run_id:
                self._send_json(400, {"error": {"message": "Missing run_id"}})
                return
            self._serve_result(run_id)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/superflex/generate":
            self.send_error(404)
            return
        try:
            self._handle_generate()
        except Exception as exc:
            self._send_json(500, {"error": {"message": str(exc)}})

    def _handle_generate(self) -> None:
        ctype, pdict = cgi.parse_header(self.headers.get("Content-Type", ""))
        if ctype != "multipart/form-data" or "boundary" not in pdict:
            self._send_json(400, {"error": {"message": "Expected multipart/form-data"}})
            return

        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
        file_item = form["file"] if "file" in form else None
        if file_item is None or not getattr(file_item, "filename", ""):
            self._send_json(400, {"error": {"message": "Missing uploaded file"}})
            return

        input_name = Path(file_item.filename).name
        stem = Path(input_name).stem
        run_id = f"{int(time.time())}_{stem}"
        input_dir = RUNS_DIR / run_id / "input"
        output_dir = RUNS_DIR / run_id / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        uploaded_path = input_dir / input_name
        with uploaded_path.open("wb") as fh:
            shutil.copyfileobj(file_item.file, fh)

        name = form.getfirst("name", stem) or stem
        z_up = _parse_bool(form.getfirst("zUp", "false"), False)
        normalize = _parse_bool(form.getfirst("normalize", "true"), True)
        lm_optimization = _parse_bool(form.getfirst("lmOptimization", "false"), False)
        max_primitives = int(form.getfirst("maxPrimitives", "0") or "0")
        exist_threshold = float(form.getfirst("existThreshold", "0.5") or "0.5")

        output_npz = output_dir / "superflex_editor.npz"
        output_meta = output_dir / "superflex_meta.json"
        log_path = LOGS_DIR / f"{run_id}.log"
        cmd = [
            PYTHON_BIN,
            str(INFER_SCRIPT),
            "--input",
            str(uploaded_path),
            "--output-npz",
            str(output_npz),
            "--output-meta",
            str(output_meta),
            "--checkpoint-dir",
            str(CHECKPOINT_DIR),
            "--name",
            name,
            "--z-up",
            "true" if z_up else "false",
            "--normalize",
            "true" if normalize else "false",
            "--lm-optimization",
            "true" if lm_optimization else "false",
            "--exist-threshold",
            str(exist_threshold),
            "--max-primitives",
            str(max_primitives),
        ]
        final_cmd = _wrap_with_srun(cmd) if _should_use_srun() else cmd

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                final_cmd,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=RUN_TIMEOUT,
                check=False,
                env=_build_child_env(),
            )

        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            self._send_json(500, {"error": {"message": f"SuperFlex inference failed ({proc.returncode}).", "log_tail": tail}})
            return
        if not output_npz.is_file() or not output_meta.is_file():
            self._send_json(500, {"error": {"message": "Inference finished without expected output files"}})
            return

        meta = json.loads(output_meta.read_text(encoding="utf-8"))
        self._send_json(
            200,
            {
                "status": "ok",
                "run_id": run_id,
                "primitive_count": meta.get("primitive_count", 0),
                "names": meta.get("names", []),
                "download_url": f"/superflex/result?run_id={run_id}",
                "metadata": meta,
            },
        )

    def _serve_result(self, run_id: str) -> None:
        npz_path = RUNS_DIR / run_id / "output" / "superflex_editor.npz"
        if not npz_path.is_file():
            self._send_json(404, {"error": {"message": f"Unknown run_id: {run_id}"}})
            return
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{npz_path.name}"')
        self.end_headers()
        with npz_path.open("rb") as fh:
            shutil.copyfileobj(fh, self.wfile)

    def _send_json(self, code: int, obj: object) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    for path in (SUPERFLEX_BASE, RUNS_DIR, TMP_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(
        f"[sq-superflex] SUPERFLEX_BASE={SUPERFLEX_BASE}\n"
        f"[sq-superflex] Listening on 0.0.0.0:{PORT}\n"
        f"[sq-superflex] Work dir/PYTHONPATH: {WORK_DIR}\n"
        f"[sq-superflex] Checkpoints: {CHECKPOINT_DIR}\n"
        f"[sq-superflex] Python: {PYTHON_BIN}\n"
        f"[sq-superflex] Slurm: uses_srun={_should_use_srun()} gpu_flag=--gpus={GPUS or '1'}\n",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
