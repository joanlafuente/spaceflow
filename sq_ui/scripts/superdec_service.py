#!/usr/bin/env python3
"""HTTP service for SuperDec-backed superquadric generation."""

from __future__ import annotations

import cgi
import http.server
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse


SCRIPT_PATH = Path(__file__).resolve()
PLACEHOLDER_BASE = "__SUPERDEC_BASE__"


def _default_base() -> Path:
    return SCRIPT_PATH.parents[2] / "superdec_ui"


def _resolve_superdec_base() -> Path:
    env_base = os.environ.get("SUPERDEC_BASE", "").strip()
    if env_base:
        return Path(env_base).expanduser()

    # When started from an installed copy, setup_superdec.sh replaces the
    # placeholder with the concrete install path. When started from the repo copy,
    # the placeholder remains literal and we should fall back to the default shared
    # course install location instead of treating "__SUPERDEC_BASE__" as a real directory.
    template_base = PLACEHOLDER_BASE
    if template_base and template_base != PLACEHOLDER_BASE:
        return Path(template_base).expanduser()

    # If the script itself lives inside the install, use that layout.
    script_parent = SCRIPT_PATH.parent
    if script_parent.name == "scripts" and script_parent.parent.name == "superdec_ui":
        return script_parent.parent

    return _default_base()


SUPERDEC_BASE = _resolve_superdec_base()
PORT = int(os.environ.get("SQ_SUPERDEC_PORT", "11435"))
RUN_TIMEOUT = int(os.environ.get("SQ_SUPERDEC_TIMEOUT_SEC", "900"))
FORWARD = os.environ.get("SQ_SUPERDEC_FORWARD", "").strip().rstrip("/")
FORCE_LOCAL = os.environ.get("SQ_SUPERDEC_FORCE_LOCAL", "").strip() == "1"
PARTITION = os.environ.get("SQ_SUPERDEC_SLURM_PARTITION", "interactive")
ACCOUNT = os.environ.get("SQ_SUPERDEC_SLURM_ACCOUNT", "3dv")
GPUS = os.environ.get("SQ_SUPERDEC_SLURM_GPUS", "1").strip()
TIME_LIMIT = os.environ.get("SQ_SUPERDEC_SLURM_TIME", "00:20:00")
EXTRA_ARGS = os.environ.get("SQ_SUPERDEC_SLURM_EXTRA_ARGS", "").strip()
TORCH_CUDA_ARCH_LIST = os.environ.get("SQ_SUPERDEC_TORCH_CUDA_ARCH_LIST", "7.5;8.9+PTX").strip()
PYTHON_BIN = os.environ.get("SQ_SUPERDEC_PYTHON", str(SUPERDEC_BASE / "venv" / "bin" / "python"))
CHECKPOINT_DIR = Path(
    os.environ.get("SQ_SUPERDEC_CHECKPOINT_DIR", str(SUPERDEC_BASE / "weights" / "normalized"))
)
WORK_DIR = Path(os.environ.get("SQ_SUPERDEC_WORK_DIR", str(SUPERDEC_BASE / "repo")))
INFER_SCRIPT = Path(os.environ.get("SQ_SUPERDEC_INFER", str(SUPERDEC_BASE / "scripts" / "superdec_infer.py")))
RUNS_DIR = Path(os.environ.get("SQ_SUPERDEC_RUNS", str(SUPERDEC_BASE / "runs")))
TMP_DIR = Path(os.environ.get("SQ_SUPERDEC_TMP", str(SUPERDEC_BASE / "tmp")))
LOGS_DIR = Path(os.environ.get("SQ_SUPERDEC_LOGS", str(SUPERDEC_BASE / "logs")))


def _resolve_work_dir() -> Path:
    if WORK_DIR.exists():
        return WORK_DIR

    # When running from the repo copy during development, allow the checked-out
    # repository to serve as the working directory without requiring the installed
    # wrapper script.
    repo_root = SCRIPT_PATH.parents[2]
    if (repo_root / "sq_ui").exists() and (repo_root / "run.py").exists():
        return repo_root

    return WORK_DIR


def _resolve_infer_script() -> Path:
    sibling = SCRIPT_PATH.parent / "superdec_infer.py"
    if sibling.exists():
        return sibling

    if INFER_SCRIPT.exists():
        return INFER_SCRIPT

    return INFER_SCRIPT


WORK_DIR = _resolve_work_dir()
INFER_SCRIPT = _resolve_infer_script()


def _build_child_env() -> dict[str, str]:
    env = os.environ.copy()
    python_path = Path(PYTHON_BIN)
    venv_bin = python_path.parent
    existing_path = env.get("PATH", "")
    env["PATH"] = f"{venv_bin}:{existing_path}" if existing_path else str(venv_bin)
    env.setdefault("VIRTUAL_ENV", str(venv_bin.parent))
    arch_tag = (
    TORCH_CUDA_ARCH_LIST.replace(";", "_")
    .replace("+", "p")
    .replace(".", "_")
    .replace(" ", "")
    ) or "default"
    env.setdefault("TORCH_EXTENSIONS_DIR", str(TMP_DIR / f"torch_extensions_{arch_tag}"))

    if TORCH_CUDA_ARCH_LIST:
        env.setdefault("TORCH_CUDA_ARCH_LIST", TORCH_CUDA_ARCH_LIST)
    return env


def _split_args(s: str) -> list[str]:
    return [tok for tok in s.split() if tok]


def _drop_gres_tokens(tokens: list[str]) -> list[str]:
    """This cluster rejects --gres; only --gpus is used. Strip gres from user-provided extras."""
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


def _append_srun_extras(srun_cmd: list[str], extra: str, log_prefix: str) -> None:
    if not extra.strip():
        return
    raw = _split_args(extra)
    cleaned = _drop_gres_tokens(raw)
    if len(cleaned) < len(raw):
        print(
            f"{log_prefix} Removed --gres from SQ_SUPERDEC_SLURM_EXTRA_ARGS; this site uses --gpus only.",
            flush=True,
        )
    srun_cmd.extend(cleaned)


def _should_use_srun() -> bool:
    if FORCE_LOCAL:
        return False
    if os.environ.get("SLURM_JOB_ID"):
        return False
    return shutil.which("srun") is not None


def _wrap_with_srun(cmd: list[str]) -> list[str]:
    srun_cmd: list[str] = [
        "srun",
        f"--partition={PARTITION}",
        f"--account={ACCOUNT}",
        f"--time={TIME_LIMIT}",
        "--job-name=sq_superdec",
        f"--gpus={GPUS or '1'}",
    ]
    _append_srun_extras(srun_cmd, EXTRA_ARGS, "[sq-superdec]")
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
    server_version = "SuperDecService/0.1"

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[sq-superdec] {fmt % args}", flush=True)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"", "/"}:
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "superdec",
                    "work_dir": str(WORK_DIR),
                    "checkpoint_dir": str(CHECKPOINT_DIR),
                    "runs_dir": str(RUNS_DIR),
                    "uses_srun": _should_use_srun(),
                    "torch_cuda_arch_list": TORCH_CUDA_ARCH_LIST,
                },
            )
            return
        if parsed.path == "/superdec/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "python": PYTHON_BIN,
                    "checkpoint_dir": str(CHECKPOINT_DIR),
                    "forward_mode": bool(FORWARD),
                    "infer_script": str(INFER_SCRIPT),
                    "uses_srun": _should_use_srun(),
                    "torch_cuda_arch_list": TORCH_CUDA_ARCH_LIST,
                },
            )
            return
        if parsed.path == "/superdec/result":
            query = parse_qs(parsed.query)
            run_id = (query.get("run_id") or [""])[0]
            if not run_id:
                self._send_json(400, {"error": {"message": "Missing run_id"}})
                return
            self._serve_result(run_id)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/superdec/generate":
            self.send_error(404)
            return
        if FORWARD:
            self._send_json(501, {"error": {"message": "Forward mode is not implemented for SuperDec"}})
            return
        try:
            self._handle_generate()
        except Exception as exc:  # noqa: BLE001 - return as HTTP error
            self._send_json(500, {"error": {"message": str(exc)}})

    def _handle_generate(self) -> None:
        ctype, pdict = cgi.parse_header(self.headers.get("Content-Type", ""))
        if ctype != "multipart/form-data":
            self._send_json(400, {"error": {"message": "Expected multipart/form-data"}})
            return
        if "boundary" not in pdict:
            self._send_json(400, {"error": {"message": "Malformed multipart/form-data payload"}})
            return

        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
        file_item = form["file"] if "file" in form else None
        if file_item is None or not getattr(file_item, "filename", ""):
            self._send_json(400, {"error": {"message": "Missing uploaded file"}})
            return

        pointcloud_name = Path(file_item.filename).name
        stem = Path(pointcloud_name).stem
        run_id = f"{int(time.time())}_{stem}"
        run_dir = RUNS_DIR / run_id
        input_dir = run_dir / "input"
        output_dir = run_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        uploaded_path = input_dir / pointcloud_name

        with uploaded_path.open("wb") as fh:
            shutil.copyfileobj(file_item.file, fh)

        name = form.getfirst("name", stem) or stem
        z_up = _parse_bool(form.getfirst("zUp", "false"), False)
        normalize = _parse_bool(form.getfirst("normalize", "true"), True)
        lm_optimization = _parse_bool(form.getfirst("lmOptimization", "false"), False)
        max_primitives = int(form.getfirst("maxPrimitives", "0") or "0")
        exist_threshold = float(form.getfirst("existThreshold", "0.5") or "0.5")

        output_npz = output_dir / "superquadrics_editor.npz"
        output_meta = output_dir / "superquadrics_meta.json"
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
                cwd=WORK_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=RUN_TIMEOUT,
                check=False,
                env=_build_child_env(),
            )

        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            self._send_json(
                500,
                {
                    "error": {
                        "message": f"SuperDec inference failed ({proc.returncode}).",
                        "log_tail": tail,
                    }
                },
            )
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
                "download_url": f"/superdec/result?run_id={run_id}",
                "metadata": meta,
            },
        )

    def _serve_result(self, run_id: str) -> None:
        npz_path = RUNS_DIR / run_id / "output" / "superquadrics_editor.npz"
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
    for path in (SUPERDEC_BASE, RUNS_DIR, TMP_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    uses_srun = _should_use_srun()
    print(
        f"[sq-superdec] SUPERDEC_BASE={SUPERDEC_BASE}\n"
        f"[sq-superdec] Listening on 0.0.0.0:{PORT}\n"
        f"[sq-superdec] Work dir: {WORK_DIR}\n"
        f"[sq-superdec] Checkpoints: {CHECKPOINT_DIR}\n"
        f"[sq-superdec] Slurm: uses_srun={uses_srun} gpu_flag=--gpus={GPUS or '1'} (no --gres)\n",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
