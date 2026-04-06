#!/usr/bin/env python3
"""HTTP service for TRELLIS-backed text-to-pointcloud generation."""

from __future__ import annotations

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
REPO_ROOT = Path(os.environ.get("SQ_TRELLIS_REPO_ROOT", str(SCRIPT_PATH.parents[2])))
TRELLIS_SCRATCH = Path(
    os.environ.get("SQ_TRELLIS_SCRATCH", str(Path("/work/scratch") / os.environ.get("USER", "user") / "spaceflow" / "trellis_ui"))
)


def _default_python_bin() -> str:
    env_override = os.environ.get("SQ_TRELLIS_PYTHON", "").strip()
    if env_override:
        return env_override

    candidates = [
        REPO_ROOT / "envs" / "guideflow3d" / "bin" / "python",
        REPO_ROOT.parent / "guideflow3d" / "envs" / "guideflow3d" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return sys.executable


PORT = int(os.environ.get("SQ_TRELLIS_PORT", "11437"))
RUN_TIMEOUT = int(os.environ.get("SQ_TRELLIS_TIMEOUT_SEC", "1800"))
FORWARD = os.environ.get("SQ_TRELLIS_FORWARD", "").strip().rstrip("/")
FORCE_LOCAL = os.environ.get("SQ_TRELLIS_FORCE_LOCAL", "").strip() == "1"
PARTITION = os.environ.get("SQ_TRELLIS_SLURM_PARTITION", "interactive")
ACCOUNT = os.environ.get("SQ_TRELLIS_SLURM_ACCOUNT", "3dv")
GPUS = os.environ.get("SQ_TRELLIS_SLURM_GPUS", "1").strip()
TIME_LIMIT = os.environ.get("SQ_TRELLIS_SLURM_TIME", "00:30:00")
EXTRA_ARGS = os.environ.get("SQ_TRELLIS_SLURM_EXTRA_ARGS", "").strip()
PYTHON_BIN = _default_python_bin()
RUNS_DIR = Path(os.environ.get("SQ_TRELLIS_RUNS", str(REPO_ROOT / "sq_ui" / "scripts" / "__TRELLIS_RUNS__")))
LOGS_DIR = Path(os.environ.get("SQ_TRELLIS_LOGS", str(REPO_ROOT / "sq_ui" / "scripts" / "__TRELLIS_LOGS__")))
WORK_DIR = Path(os.environ.get("SQ_TRELLIS_WORK_DIR", str(REPO_ROOT)))
INFER_SCRIPT = Path(os.environ.get("SQ_TRELLIS_INFER", str(SCRIPT_PATH.parent / "trellis_infer.py")))
CACHE_ROOT = Path(os.environ.get("SQ_TRELLIS_CACHE_ROOT", str(TRELLIS_SCRATCH / "cache")))


def _build_child_env() -> dict[str, str]:
    env = os.environ.copy()
    xdg_cache = Path(env.get("XDG_CACHE_HOME", str(CACHE_ROOT)))
    hf_home = Path(env.get("HF_HOME", str(CACHE_ROOT / "huggingface")))
    torch_home = Path(env.get("TORCH_HOME", str(CACHE_ROOT / "torch")))

    env["XDG_CACHE_HOME"] = str(xdg_cache)
    env["HF_HOME"] = str(hf_home)
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    env.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    env["TORCH_HOME"] = str(torch_home)

    for path in [
        TRELLIS_SCRATCH,
        CACHE_ROOT,
        xdg_cache,
        hf_home,
        Path(env["HUGGINGFACE_HUB_CACHE"]),
        Path(env["TRANSFORMERS_CACHE"]),
        torch_home,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return env


def _split_args(s: str) -> list[str]:
    return [tok for tok in s.split() if tok]


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
        "--job-name=sq_trellis",
        f"--gpus={GPUS or '1'}",
    ]
    if EXTRA_ARGS:
        srun_cmd.extend(_split_args(EXTRA_ARGS))
    srun_cmd.extend(cmd)
    return srun_cmd


def _sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name).strip("_")
    return safe or "trellis"


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "TrellisService/0.1"

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[sq-trellis] {fmt % args}", flush=True)

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
                    "service": "trellis",
                    "work_dir": str(WORK_DIR),
                    "runs_dir": str(RUNS_DIR),
                    "uses_srun": _should_use_srun(),
                    "infer_script": str(INFER_SCRIPT),
                },
            )
            return
        if parsed.path == "/trellis/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "python": PYTHON_BIN,
                    "forward_mode": bool(FORWARD),
                    "uses_srun": _should_use_srun(),
                    "infer_script": str(INFER_SCRIPT),
                },
            )
            return
        if parsed.path == "/trellis/result":
            query = parse_qs(parsed.query)
            run_id = (query.get("run_id") or [""])[0]
            if not run_id:
                self._send_json(400, {"error": {"message": "Missing run_id"}})
                return
            self._serve_result(run_id)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/trellis/generate":
            self.send_error(404)
            return
        if FORWARD:
            self._send_json(501, {"error": {"message": "Forward mode is not implemented for TRELLIS"}})
            return
        try:
            self._handle_generate()
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": {"message": str(exc)}})

    def _handle_generate(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            self._send_json(400, {"error": {"message": "Expected JSON request body"}})
            return

        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": {"message": f"Invalid JSON body: {exc}"}})
            return

        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            self._send_json(400, {"error": {"message": "Missing prompt"}})
            return

        name = _sanitize_name(str(payload.get("name", "")).strip() or prompt)
        seed = int(payload.get("seed", 1) or 1)
        point_count = int(payload.get("pointCount", 4096) or 4096)
        normalize = bool(payload.get("normalize", True))
        prefer_mesh = bool(payload.get("preferMesh", False))
        sparse_steps = payload.get("sparseSteps")
        slat_steps = payload.get("slatSteps")
        cfg_strength = payload.get("cfgStrength")
        slat_cfg_strength = payload.get("slatCfgStrength")

        run_id = f"{int(time.time())}_{name}"
        run_dir = RUNS_DIR / run_id
        output_dir = run_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_ply = output_dir / f"{name}.ply"
        output_meta = output_dir / "trellis_meta.json"
        log_path = LOGS_DIR / f"{run_id}.log"

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        if _should_use_srun():
            cmd = [
                PYTHON_BIN,
                str(INFER_SCRIPT),
                "--prompt",
                prompt,
                "--output-ply",
                str(output_ply),
                "--output-meta",
                str(output_meta),
                "--name",
                name,
                "--seed",
                str(seed),
                "--point-count",
                str(point_count),
                "--normalize",
                "true" if normalize else "false",
                "--prefer-mesh",
                "true" if prefer_mesh else "false",
            ]
            if sparse_steps is not None:
                cmd.extend(["--sparse-steps", str(int(sparse_steps))])
            if slat_steps is not None:
                cmd.extend(["--slat-steps", str(int(slat_steps))])
            if cfg_strength is not None:
                cmd.extend(["--cfg-strength", str(float(cfg_strength))])
            if slat_cfg_strength is not None:
                cmd.extend(["--slat-cfg-strength", str(float(slat_cfg_strength))])

            final_cmd = _wrap_with_srun(cmd)
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
                            "message": f"TRELLIS inference failed ({proc.returncode}).",
                            "log_tail": tail,
                        }
                    },
                )
                return
        else:
            started = time.time()
            try:
                import trellis_infer

                points, meta = trellis_infer.generate_point_cloud(
                    prompt,
                    seed=seed,
                    point_count=point_count,
                    normalize=normalize,
                    prefer_mesh=prefer_mesh,
                    sparse_steps=int(sparse_steps) if sparse_steps is not None else 12,
                    slat_steps=int(slat_steps) if slat_steps is not None else 12,
                    cfg_strength=float(cfg_strength) if cfg_strength is not None else 7.5,
                    slat_cfg_strength=float(slat_cfg_strength) if slat_cfg_strength is not None else 7.5,
                )
                trellis_infer.save_ply(points, output_ply)
                meta = {
                    "name": name,
                    **meta,
                    "output_ply": str(output_ply),
                    "elapsed_sec": round(time.time() - started, 3),
                }
                output_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                log_path.write_text(
                    f"[sq-trellis] generated {meta['point_count']} points from {meta['source']}\n",
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                self._send_json(
                    500,
                    {
                        "error": {
                            "message": f"TRELLIS inference failed: {exc}",
                            "log_tail": log_path.read_text(encoding='utf-8', errors='replace'),
                        }
                    },
                )
                return

        if not output_ply.is_file() or not output_meta.is_file():
            self._send_json(500, {"error": {"message": "Inference finished without expected output files"}})
            return

        meta = json.loads(output_meta.read_text(encoding="utf-8"))
        self._send_json(
            200,
            {
                "status": "ok",
                "run_id": run_id,
                "point_count": meta.get("point_count", point_count),
                "filename": output_ply.name,
                "download_url": f"/trellis/result?run_id={run_id}",
                "metadata": meta,
            },
        )

    def _serve_result(self, run_id: str) -> None:
        run_dir = RUNS_DIR / run_id / "output"
        matches = sorted(run_dir.glob("*.ply"))
        if not matches:
            self._send_json(404, {"error": {"message": f"Unknown run_id: {run_id}"}})
            return
        ply_path = matches[0]
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{ply_path.name}"')
        self.end_headers()
        with ply_path.open("rb") as fh:
            shutil.copyfileobj(fh, self.wfile)

    def _send_json(self, code: int, obj: object) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    child_env = _build_child_env()
    os.environ.update({
        "XDG_CACHE_HOME": child_env["XDG_CACHE_HOME"],
        "HF_HOME": child_env["HF_HOME"],
        "HUGGINGFACE_HUB_CACHE": child_env["HUGGINGFACE_HUB_CACHE"],
        "TRANSFORMERS_CACHE": child_env["TRANSFORMERS_CACHE"],
        "TORCH_HOME": child_env["TORCH_HOME"],
    })

    for path in (TRELLIS_SCRATCH, CACHE_ROOT, RUNS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(
        f"[sq-trellis] Listening on 0.0.0.0:{PORT}\n"
        f"[sq-trellis] Work dir: {WORK_DIR}\n"
        f"[sq-trellis] Infer script: {INFER_SCRIPT}\n",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
