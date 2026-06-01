#!/usr/bin/env python3
"""HTTP service for saving SQ editor assets and launching SpaceFlow runs."""

from __future__ import annotations

import http.server
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="'cgi' is deprecated.*", category=DeprecationWarning)
    import cgi


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
PORT = int(os.environ.get("SQ_SPACEFLOW_PORT", "11438"))
USER = os.environ.get("USER", "user")
TEAM_STORAGE_ROOT = Path(
    os.environ.get("SQ_SPACEFLOW_STORAGE_ROOT", "/work/courses/3dv/team3/spaceflow_runtime")
).expanduser()
SAVE_ROOT = Path(
    os.environ.get("SQ_SPACEFLOW_ASSET_ROOT", str(TEAM_STORAGE_ROOT / "sq_ui_assets"))
).expanduser()
RUN_ROOT = Path(
    os.environ.get("SQ_SPACEFLOW_RUN_ROOT", str(TEAM_STORAGE_ROOT / "sq_ui_runs"))
).expanduser()
RUN_TIMEOUT = int(os.environ.get("SQ_SPACEFLOW_TIMEOUT_SEC", "7200"))
FORCE_LOCAL = os.environ.get("SQ_SPACEFLOW_FORCE_LOCAL", "").strip() == "1"
PARTITION = "interactive" # os.environ.get("SQ_SPACEFLOW_SLURM_PARTITION", "interactive")
ACCOUNT = os.environ.get("SQ_SPACEFLOW_SLURM_ACCOUNT", "3dv")
GPUS = os.environ.get("SQ_SPACEFLOW_SLURM_GPUS", "1").strip()
CONSTRAINT = os.environ.get("SQ_SPACEFLOW_SLURM_CONSTRAINT", "5060ti").strip()
TIME_LIMIT = os.environ.get("SQ_SPACEFLOW_SLURM_TIME", "00:30:00")
EXTRA_ARGS = os.environ.get("SQ_SPACEFLOW_SLURM_EXTRA_ARGS", "").strip()
RUN_SCRIPT = Path(os.environ.get("SQ_SPACEFLOW_RUN_SCRIPT", str(REPO_ROOT / "run_local_tau.py"))).expanduser()
CACHE_ROOT = Path(
    os.environ.get("SQ_SPACEFLOW_CACHE_ROOT", str(TEAM_STORAGE_ROOT / "huggingface"))
).expanduser()
XDG_CACHE_ROOT = Path(
    os.environ.get("SQ_SPACEFLOW_XDG_CACHE_ROOT", str(TEAM_STORAGE_ROOT / "xdg_cache"))
).expanduser()


def _default_python_bin() -> str:
    env_override = os.environ.get("SQ_SPACEFLOW_PYTHON", "").strip()
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


PYTHON_BIN = _default_python_bin()
RUNS: dict[str, subprocess.Popen[bytes]] = {}
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
KNOWN_OUTPUTS = [
    "sample.glb",
    "struct_mesh_zup.glb",
    "struct_mesh.glb",
    "denoising_evolution.mp4",
    "spatial_control_mesh.ply",
    "high_control_spatial_control_mesh.ply",
    "low_control_superquadric_mask.ply",
    "struct_renders/000.png",
    "struct_renders/mesh.ply",
    "voxels/struct_voxels.ply",
    "app_mesh_zup.glb",
    "app_mesh.glb",
    "app_image.png",
    "app_renders/000.png",
    "voxels/app_voxels.ply",
]


def _send_file(res: http.server.BaseHTTPRequestHandler, file_path: Path) -> None:
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    res.send_response(200)
    res._cors()
    res.send_header("Content-Type", content_type)
    res.send_header("Content-Length", str(file_path.stat().st_size))
    res.send_header("Content-Disposition", f'inline; filename="{file_path.name}"')
    res.end_headers()
    with file_path.open("rb") as fh:
        shutil.copyfileobj(fh, res.wfile)


def _sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name).strip("_")
    return safe or "superquadrics"


def _utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


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
        "--job-name=sq_spaceflow",
        "--ntasks=1",
        "--export=ALL",
        f"--gpus={GPUS or '1'}",
    ]
    if CONSTRAINT:
        srun_cmd.append(f"--constraint={CONSTRAINT}")
    srun_cmd.extend(_drop_gres_tokens(_split_args(EXTRA_ARGS)))
    srun_cmd.extend(cmd)
    return srun_cmd


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


def _build_run_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(CACHE_ROOT))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_ROOT / "hub"))
    env.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "transformers"))
    env.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_ROOT))
    env.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
    env.setdefault("TMPDIR", str(RUN_ROOT / "tmp"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TORCH_HOME", "TMPDIR"):
        Path(env[key]).expanduser().mkdir(parents=True, exist_ok=True)
    return env


def _read_json_field(form: cgi.FieldStorage, name: str, default: object) -> object:
    raw = form.getfirst(name)
    if not raw:
        return default
    return json.loads(raw)


def _ensure_inside_root(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    root_resolved = root.expanduser().resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Path is outside allowed root: {resolved}")
    return resolved


def _copy_uploaded(form: cgi.FieldStorage, field: str, target: Path) -> None:
    item = form[field] if field in form else None
    if item is None or not getattr(item, "filename", ""):
        raise ValueError(f"Missing uploaded file field: {field}")
    with target.open("wb") as fh:
        shutil.copyfileobj(item.file, fh)


def _history_path() -> Path:
    return SAVE_ROOT / "index.json"


def _read_history() -> dict[str, object]:
    path = _history_path()
    if not path.is_file():
        return {"entries": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data
    except json.JSONDecodeError:
        pass
    return {"entries": []}


def _write_history(data: dict[str, object]) -> None:
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    path = _history_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_history(entry: dict[str, object]) -> None:
    data = _read_history()
    entries = data.setdefault("entries", [])
    if not isinstance(entries, list):
        entries = []
        data["entries"] = entries
    entries.insert(0, entry)
    del entries[250:]
    _write_history(data)


def _save_bundle_from_form(form: cgi.FieldStorage, run_id: str | None = None) -> dict[str, object]:
    project_name = _sanitize_name(form.getfirst("projectName", "superquadrics") or "superquadrics")
    timestamp = _utc_timestamp()
    save_id = run_id or f"{timestamp}_{project_name}"
    asset_dir = SAVE_ROOT / project_name / save_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "all": asset_dir / "all.npz",
        "high_control": asset_dir / "high_control.npz",
        "low_control_bbox": asset_dir / "low_control_bbox.npz",
    }
    _copy_uploaded(form, "all", paths["all"])
    _copy_uploaded(form, "high_control", paths["high_control"])
    _copy_uploaded(form, "low_control_bbox", paths["low_control_bbox"])

    manifest = _read_json_field(form, "manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    manifest = {
        **manifest,
        "project_name": project_name,
        "saved_at": timestamp,
        "save_id": save_id,
        "paths": {key: str(path) for key, path in paths.items()},
    }
    manifest_path = asset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    counts = manifest.get("counts", {})
    entry = {
        "id": save_id,
        "project_name": project_name,
        "saved_at": timestamp,
        "asset_dir": str(asset_dir),
        "manifest_path": str(manifest_path),
        "paths": {key: str(path) for key, path in paths.items()},
        "counts": counts,
    }
    _append_history(entry)
    return entry


def _run_meta_path(run_id: str) -> Path:
    return RUN_ROOT / run_id / "run_meta.json"


def _write_run_meta(run_id: str, meta: dict[str, object]) -> None:
    path = _run_meta_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _read_run_meta(run_id: str) -> dict[str, object] | None:
    path = _run_meta_path(run_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _log_tail(log_path: str | Path, n: int = 6000) -> str:
    path = Path(log_path)
    if not path.is_file():
        return ""
    return _clean_log_text(path.read_text(encoding="utf-8", errors="replace")[-n:])


def _clean_log_text(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "\n")
    lines: list[str] = []
    previous_blank = False
    for line in text.splitlines():
        clean = line.rstrip()
        stripped = clean.strip()
        if "%|" in stripped and ("it/s" in stripped or "B/s" in stripped):
            continue
        if stripped.startswith(("Sampling:", "Rendering:")) and "it/s" in stripped:
            continue
        is_blank = not clean
        if is_blank and previous_blank:
            continue
        lines.append(clean)
        previous_blank = is_blank
    return "\n".join(lines)


def _file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".glb", ".gltf", ".obj"}:
        return "mesh"
    if suffix in {".ply", ".npz", ".npy"}:
        return "data"
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix in {".mp4", ".webm", ".mov"}:
        return "video"
    if suffix in {".log", ".txt"}:
        return "log"
    return "file"


def _run_file_url(run_id: str, relative_path: str) -> str:
    return f"/spaceflow/runs/file?run_id={quote(run_id)}&path={quote(relative_path)}"


def _list_output_files(meta: dict[str, object]) -> list[dict[str, object]]:
    output_dir_raw = str(meta.get("output_dir") or "")
    if not output_dir_raw:
        return []
    output_dir = Path(output_dir_raw)
    if not output_dir.is_dir():
        return []

    run_id = str(meta.get("run_id") or "")
    seen: set[str] = set()
    files: list[dict[str, object]] = []

    def add(path: Path) -> None:
        if not path.is_file():
            return
        try:
            rel = path.relative_to(output_dir).as_posix()
        except ValueError:
            return
        if rel in seen:
            return
        seen.add(rel)
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "relative_path": rel,
                "kind": _file_kind(path),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "url": _run_file_url(run_id, rel) if run_id else "",
            }
        )

    for rel in KNOWN_OUTPUTS:
        add(output_dir / rel)

    allowed_suffixes = {".glb", ".ply", ".mp4", ".png", ".jpg", ".jpeg", ".webp", ".npz", ".json", ".log", ".txt"}
    for path in sorted(output_dir.rglob("*")):
        if len(files) >= 80:
            break
        if path.suffix.lower() in allowed_suffixes:
            add(path)

    return files


def _run_with_outputs(meta: dict[str, object]) -> dict[str, object]:
    run = dict(meta)
    run["output_files"] = _list_output_files(run)
    return run


def _reconcile_untracked_run(meta: dict[str, object]) -> tuple[dict[str, object], bool]:
    if meta.get("status") != "running":
        return meta, False
    log_text = _log_tail(str(meta.get("log_path", "")), 12000)
    lower = log_text.lower()
    failure_markers = [
        "force terminated",
        "allocation",
        "revoked",
        "traceback",
        "runtimeerror",
        "error:",
        "failed",
    ]
    if any(marker in lower for marker in failure_markers):
        meta = dict(meta)
        meta["status"] = "failed"
        meta.setdefault("returncode", -1)
        return meta, True
    success_markers = [
        "structure-only mode complete",
    ]
    if any(marker in lower for marker in success_markers) and _list_output_files(meta):
        meta = dict(meta)
        meta["status"] = "succeeded"
        meta.setdefault("returncode", 0)
        return meta, True
    return meta, False


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "SpaceFlowUIService/0.1"

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[sq-spaceflow] {fmt % args}", flush=True)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"", "/", "/spaceflow/health"}:
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "spaceflow",
                    "asset_root": str(SAVE_ROOT),
                    "run_root": str(RUN_ROOT),
                    "cache_root": str(CACHE_ROOT),
                    "python": PYTHON_BIN,
                    "run_script": str(RUN_SCRIPT),
                    "uses_srun": _should_use_srun(),
                    "slurm_constraint": CONSTRAINT,
                },
            )
            return
        if parsed.path == "/spaceflow/assets/history":
            self._handle_history(parsed.query)
            return
        if parsed.path == "/spaceflow/assets/open":
            self._handle_open_asset(parsed.query)
            return
        if parsed.path == "/spaceflow/runs/status":
            self._handle_run_status(parsed.query)
            return
        if parsed.path == "/spaceflow/runs/log":
            self._handle_run_log(parsed.query)
            return
        if parsed.path == "/spaceflow/runs/file":
            self._handle_run_file(parsed.query)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/spaceflow/assets/save":
            self._handle_save()
            return
        if self.path == "/spaceflow/runs/start":
            self._handle_run_start()
            return
        self.send_error(404)

    def _multipart_form(self) -> cgi.FieldStorage:
        ctype, pdict = cgi.parse_header(self.headers.get("Content-Type", ""))
        if ctype != "multipart/form-data" or "boundary" not in pdict:
            raise ValueError("Expected multipart/form-data")
        return cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})

    def _handle_save(self) -> None:
        try:
            form = self._multipart_form()
            entry = _save_bundle_from_form(form)
            self._send_json(200, {"status": "ok", "entry": entry})
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": {"message": str(exc)}})

    def _handle_history(self, query: str) -> None:
        params = parse_qs(query)
        limit = max(1, min(250, int((params.get("limit") or ["50"])[0] or "50")))
        data = _read_history()
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            entries = []
        self._send_json(200, {"status": "ok", "entries": entries[:limit]})

    def _handle_open_asset(self, query: str) -> None:
        params = parse_qs(query)
        raw_path = (params.get("path") or [""])[0]
        if not raw_path:
            self._send_json(400, {"error": {"message": "Missing path"}})
            return
        try:
            file_path = _ensure_inside_root(Path(raw_path), SAVE_ROOT)
        except ValueError as exc:
            self._send_json(403, {"error": {"message": str(exc)}})
            return
        if file_path.suffix.lower() != ".npz" or not file_path.is_file():
            self._send_json(404, {"error": {"message": f"NPZ file not found: {file_path}"}})
            return
        _send_file(self, file_path)

    def _handle_run_start(self) -> None:
        try:
            form = self._multipart_form()
            run_config = _read_json_field(form, "runConfig", {})
            if not isinstance(run_config, dict):
                raise ValueError("runConfig must be a JSON object")

            project_name = _sanitize_name(form.getfirst("projectName", "superquadrics") or "superquadrics")
            output_name = _sanitize_name(str(run_config.get("outputName") or project_name))
            run_id = f"{_utc_timestamp()}_{output_name}"
            asset_entry = _save_bundle_from_form(form, run_id=run_id)
            run_dir = RUN_ROOT / run_id
            output_dir = run_dir / "output"
            log_path = run_dir / "spaceflow.log"
            run_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            text_prompt = str(run_config.get("textPrompt", "")).strip()
            if not text_prompt:
                raise ValueError("Missing text prompt")
            low_tau = float(run_config.get("lowTau", 3.0))
            high_tau = float(run_config.get("highTau", 10.0))
            if high_tau <= low_tau:
                raise ValueError("High tau must be greater than low tau")
            polyak = float(run_config.get("polyakTau", 0.18))
            appearance_mode = str(run_config.get("appearanceMode", "text")).strip().lower()
            convert_yup_to_zup = _parse_bool(str(run_config.get("convertYupToZup", True)), True)
            dry_run = _parse_bool(str(run_config.get("dryRun", False)), False)

            asset_paths = asset_entry["paths"]
            assert isinstance(asset_paths, dict)
            cmd = [
                PYTHON_BIN,
                str(RUN_SCRIPT),
                "--guidance_mode",
                "similarity",
                "--output_dir",
                str(output_dir),
                "--shape_superquadric_path",
                str(asset_paths["all"]),
                "--shape_tau",
                str(low_tau),
                "--shape_superquadric_high_control_path",
                str(asset_paths["high_control"]),
                "--shape_tau_high_control",
                str(high_tau),
                "--low_control_superquadric_mask_path",
                str(asset_paths["low_control_bbox"]),
                "--local_tau_mode",
                "low_control_mask",
                "--polyak_update_tau",
                str(polyak),
                "--text_prompt",
                text_prompt,
            ]
            if convert_yup_to_zup:
                cmd.append("--convert_yup_to_zup")

            if appearance_mode == "image":
                uploaded = form["appearance_image"] if "appearance_image" in form else None
                appearance_image_path = str(run_config.get("appearanceImagePath") or "").strip()
                if uploaded is not None and getattr(uploaded, "filename", ""):
                    target = run_dir / Path(uploaded.filename).name
                    with target.open("wb") as fh:
                        shutil.copyfileobj(uploaded.file, fh)
                    appearance_image_path = str(target)
                if not appearance_image_path:
                    raise ValueError("Image appearance mode requires an image path or upload")
                cmd.extend(["--appearance_image", appearance_image_path])
            else:
                appearance_text = str(run_config.get("appearanceText") or text_prompt).strip()
                cmd.extend(["--appearance_text", appearance_text])

            final_cmd = _wrap_with_srun(cmd) if _should_use_srun() else cmd
            meta = {
                "run_id": run_id,
                "status": "dry_run" if dry_run else "running",
                "project_name": project_name,
                "created_at": _utc_timestamp(),
                "asset_entry": asset_entry,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
                "command": final_cmd,
                "run_config": run_config,
                "pipeline_stage": "structure_only",
            }
            _write_run_meta(run_id, meta)

            if dry_run:
                log_path.write_text("Dry run command:\n" + " ".join(final_cmd) + "\n", encoding="utf-8")
            else:
                log_file = log_path.open("wb")
                try:
                    proc = subprocess.Popen(
                        final_cmd,
                        cwd=REPO_ROOT,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        env=_build_run_env(),
                    )
                finally:
                    log_file.close()
                RUNS[run_id] = proc

            self._send_json(200, {"status": "ok", "run_id": run_id, "run": _run_with_outputs(meta)})
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": {"message": str(exc)}})

    def _handle_run_status(self, query: str) -> None:
        params = parse_qs(query)
        run_id = (params.get("run_id") or [""])[0]
        if not run_id:
            self._send_json(400, {"error": {"message": "Missing run_id"}})
            return
        meta = _read_run_meta(run_id)
        if not meta:
            self._send_json(404, {"error": {"message": f"Unknown run_id: {run_id}"}})
            return
        proc = RUNS.get(run_id)
        if proc is not None:
            code = proc.poll()
            if code is None:
                meta["status"] = "running"
            else:
                meta["status"] = "succeeded" if code == 0 else "failed"
                meta["returncode"] = code
                RUNS.pop(run_id, None)
                _write_run_meta(run_id, meta)
        else:
            meta, changed = _reconcile_untracked_run(meta)
            if changed:
                _write_run_meta(run_id, meta)
        log_tail = _log_tail(str(meta.get("log_path", "")))
        self._send_json(200, {"status": "ok", "run": _run_with_outputs(meta), "log_tail": log_tail})

    def _handle_run_log(self, query: str) -> None:
        params = parse_qs(query)
        run_id = (params.get("run_id") or [""])[0]
        meta = _read_run_meta(run_id) if run_id else None
        if not meta:
            self._send_json(404, {"error": {"message": f"Unknown run_id: {run_id}"}})
            return
        self._send_json(200, {"status": "ok", "log_tail": _log_tail(str(meta.get("log_path", "")), 20000)})

    def _handle_run_file(self, query: str) -> None:
        params = parse_qs(query)
        run_id = (params.get("run_id") or [""])[0]
        rel_path = (params.get("path") or [""])[0]
        meta = _read_run_meta(run_id) if run_id else None
        if not meta:
            self._send_json(404, {"error": {"message": f"Unknown run_id: {run_id}"}})
            return
        if not rel_path:
            self._send_json(400, {"error": {"message": "Missing path"}})
            return
        try:
            output_dir = Path(str(meta.get("output_dir") or "")).expanduser().resolve()
            file_path = (output_dir / rel_path).expanduser().resolve()
            if file_path != output_dir and output_dir not in file_path.parents:
                raise ValueError("Requested file is outside the run output directory")
        except ValueError as exc:
            self._send_json(403, {"error": {"message": str(exc)}})
            return
        if not file_path.is_file():
            self._send_json(404, {"error": {"message": f"Run output file not found: {rel_path}"}})
            return
        _send_file(self, file_path)

    def _send_json(self, code: int, obj: object) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    for path in (SAVE_ROOT, RUN_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(
        f"[sq-spaceflow] Listening on 0.0.0.0:{PORT}\n"
        f"[sq-spaceflow] Asset root: {SAVE_ROOT}\n"
        f"[sq-spaceflow] Run root: {RUN_ROOT}\n"
        f"[sq-spaceflow] Cache root: {CACHE_ROOT}\n"
        f"[sq-spaceflow] Python: {PYTHON_BIN}\n"
        f"[sq-spaceflow] Run script: {RUN_SCRIPT}\n"
        f"[sq-spaceflow] Slurm: uses_srun={_should_use_srun()} gpu_flag=--gpus={GPUS or '1'} constraint={CONSTRAINT or 'none'}\n",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
