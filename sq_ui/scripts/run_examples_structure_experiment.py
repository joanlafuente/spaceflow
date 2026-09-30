#!/usr/bin/env python3
"""Batch launcher for the 20 example structure experiments.

The script prepares the same input bundle the UI would create for each example:
all superquadrics, the high-control subset, and a low-control bounding-box
primitive. It then runs the existing in-process experiment runner once across
all selected examples so the TRELLIS pipeline can be reused across variants.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_EXAMPLES_DIR = REPO_ROOT / "examples"
DEFAULT_PROMPTS_PATH = DEFAULT_EXAMPLES_DIR / "text_prompts.txt"
DEFAULT_RUN_ROOT = REPO_ROOT / "spaceflow_runtime" / "sq_ui_runs"
DEFAULT_ASSET_ROOT = REPO_ROOT / "spaceflow_runtime" / "sq_ui_assets" / "spaceflow_examples_structure"
EXPERIMENT_RUNNER = REPO_ROOT / "sq_ui" / "scripts" / "run_spaceflow_experiment.py"
COMPARISON_RENDERER = REPO_ROOT / "sq_ui" / "scripts" / "render_spaceflow_experiment_comparison.py"

LOW_TAU = 3.0
HIGH_TAU = 10.0
POLYAK_TAU = 0.18
REPAINT_STEPS = 10
TEXTURE_OPTIM_STEPS = 2
BBOX_MARGIN_FRACTION = 0.07
BBOX_MIN_HALF_EXTENT = 1e-4
BBOX_RESOLUTION = 32
CUBE_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
        [0.0, np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)],
    ],
    dtype=np.float64,
)


def _utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _local_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _sanitize_name(name: str, fallback: str = "example") -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    safe = re.sub(r"_+", "_", safe)
    return safe or fallback


def _num_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


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


def _parse_only(raw: str | None) -> list[int]:
    if not raw:
        return list(range(1, 21))
    selected: list[int] = []
    for token in raw.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid --only range: {token}")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(token))
    deduped = sorted(set(selected))
    invalid = [value for value in deduped if value < 1 or value > 20]
    if invalid:
        raise ValueError(f"--only values must be in 1..20, got {invalid}")
    if not deduped:
        raise ValueError("--only did not select any examples")
    return deduped


def _parse_prompts(path: Path) -> dict[int, str]:
    prompts: dict[int, str] = {}
    pattern = re.compile(r"^\s*(\d+)\.\s*(.*?)\s*$")
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        match = pattern.match(line)
        if not match:
            raise ValueError(f"Could not parse prompt line {line_no}: {line!r}")
        index = int(match.group(1))
        prompt = match.group(2).strip()
        if not prompt:
            raise ValueError(f"Prompt line {line_no} is empty")
        if index in prompts:
            raise ValueError(f"Duplicate prompt index {index} in {path}")
        prompts[index] = prompt
    return prompts


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    required = ("scales", "shapes", "translations", "rotations", "control_levels")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"{path} missing required array(s): {missing}")
    count = int(arrays["scales"].shape[0])
    expected_shapes = {
        "scales": (count, 3),
        "shapes": (count, 2),
        "translations": (count, 3),
        "rotations": (count, 3, 3),
        "control_levels": (count,),
    }
    for key, expected in expected_shapes.items():
        if tuple(arrays[key].shape) != expected:
            raise ValueError(f"{path}:{key} expected shape {expected}, got {arrays[key].shape}")
    if "tapering" in arrays and tuple(arrays["tapering"].shape) != (count, 2):
        raise ValueError(f"{path}:tapering expected shape {(count, 2)}, got {arrays['tapering'].shape}")
    if "bending" in arrays and tuple(arrays["bending"].shape) != (count, 6):
        raise ValueError(f"{path}:bending expected shape {(count, 6)}, got {arrays['bending'].shape}")
    return arrays


def _save_subset_npz(path: Path, arrays: dict[str, np.ndarray], mask: np.ndarray) -> None:
    keys = ["scales", "shapes", "translations", "rotations"]
    if "tapering" in arrays:
        keys.append("tapering")
    if "bending" in arrays:
        keys.append("bending")
    keys.append("control_levels")
    payload = {key: np.asarray(arrays[key][mask], dtype=np.float64) for key in keys}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _signed_power_sin(values: np.ndarray, exponent: float) -> np.ndarray:
    sine = np.sin(values)
    return np.sign(sine) * np.abs(sine) ** exponent


def _signed_power_cos(values: np.ndarray, exponent: float) -> np.ndarray:
    cosine = np.cos(values)
    return np.sign(cosine) * np.abs(cosine) ** exponent


def _apply_taper(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    c: float,
    kx: float,
    ky: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = c if abs(c) > 1e-8 else 1e-8
    z_norm = z / c
    return x * (kx * z_norm + 1.0), y * (ky * z_norm + 1.0), z


def _apply_bending_axis(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    kb: float,
    alpha: float,
    axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(kb) < 1e-3:
        return x, y, z
    if axis == "z":
        u, v, w = x.copy(), y.copy(), z.copy()
    elif axis == "x":
        u, v, w = y.copy(), z.copy(), x.copy()
    elif axis == "y":
        u, v, w = z.copy(), x.copy(), y.copy()
    else:
        raise ValueError(axis)

    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    beta = np.arctan2(v, u)
    radius = np.sqrt(u * u + v * v) * np.cos(alpha - beta)
    inv_kb = 1.0 / kb
    gamma = w * kb
    rho = inv_kb - radius
    bent_radius = inv_kb - rho * np.cos(gamma)
    expr = bent_radius - radius
    u = u + expr * cos_alpha
    v = v + expr * sin_alpha
    w = rho * np.sin(gamma)

    if axis == "z":
        return u, v, w
    if axis == "x":
        return w, u, v
    return v, w, u


def _superquadric_vertices(
    scale: np.ndarray,
    shape: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    tapering: np.ndarray,
    bending: np.ndarray,
    resolution: int,
) -> np.ndarray:
    a, b, c = [float(value) for value in scale]
    e1, e2 = [float(value) for value in shape]
    u = np.linspace(-np.pi, np.pi, resolution, endpoint=True)
    v = np.linspace(-np.pi / 2.0, np.pi / 2.0, resolution, endpoint=True)
    u = np.tile(u, resolution)
    v = np.repeat(v, resolution)
    if np.linalg.det(rotation) < 0:
        u = u[::-1]

    x = a * _signed_power_cos(v, e1) * _signed_power_cos(u, e2)
    y = b * _signed_power_cos(v, e1) * _signed_power_sin(u, e2)
    z = c * _signed_power_sin(v, e1)

    x[:resolution] = 0.0
    x[-resolution:] = 0.0

    x, y, z = _apply_taper(x, y, z, c, float(tapering[0]), float(tapering[1]))
    x, y, z = _apply_bending_axis(x, y, z, float(bending[4]), float(bending[5]), "y")
    x, y, z = _apply_bending_axis(x, y, z, float(bending[2]), float(bending[3]), "x")
    x, y, z = _apply_bending_axis(x, y, z, float(bending[0]), float(bending[1]), "z")

    local_vertices = np.stack([x, y, z], axis=1)
    return local_vertices @ rotation.T + translation


def _low_control_bbox(arrays: dict[str, np.ndarray], low_mask: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    count = int(arrays["scales"].shape[0])
    tapering = arrays.get("tapering", np.zeros((count, 2), dtype=np.float64))
    bending = arrays.get("bending", np.zeros((count, 6), dtype=np.float64))

    vertices = []
    for index in np.flatnonzero(low_mask):
        vertices.append(
            _superquadric_vertices(
                np.asarray(arrays["scales"][index], dtype=np.float64),
                np.asarray(arrays["shapes"][index], dtype=np.float64),
                np.asarray(arrays["translations"][index], dtype=np.float64),
                np.asarray(arrays["rotations"][index], dtype=np.float64),
                np.asarray(tapering[index], dtype=np.float64),
                np.asarray(bending[index], dtype=np.float64),
                BBOX_RESOLUTION,
            )
        )
    if not vertices:
        raise ValueError("Cannot build low-control bbox without low-control primitives")

    all_vertices = np.concatenate(vertices, axis=0)
    bbox_min = all_vertices.min(axis=0)
    bbox_max = all_vertices.max(axis=0)
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    pad = diagonal * BBOX_MARGIN_FRACTION
    padded_min = bbox_min - pad
    padded_max = bbox_max + pad
    center = (padded_min + padded_max) / 2.0
    half_extents = np.maximum((padded_max - padded_min) / 2.0, BBOX_MIN_HALF_EXTENT)
    hx, hy, hz = [float(value) for value in half_extents]

    bbox_arrays = {
        "scales": np.asarray([[hx, hz, hy]], dtype=np.float64),
        "shapes": np.asarray([[0.05, 0.05]], dtype=np.float64),
        "translations": center.reshape(1, 3).astype(np.float64),
        "rotations": CUBE_ROTATION.reshape(1, 3, 3).astype(np.float64),
        "control_levels": np.asarray([0.0], dtype=np.float64),
    }
    bbox_meta = {
        "min": padded_min.tolist(),
        "max": padded_max.tolist(),
        "center": center.tolist(),
        "halfExtents": half_extents.tolist(),
    }
    return bbox_arrays, bbox_meta


def _save_low_control_bbox(path: Path, arrays: dict[str, np.ndarray], low_mask: np.ndarray) -> dict[str, Any]:
    bbox_arrays, bbox_meta = _low_control_bbox(arrays, low_mask)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **bbox_arrays)
    return bbox_meta


def _variant_argv(
    asset_paths: dict[str, Path],
    output_dir: Path,
    prompt: str,
    *,
    low_tau: float,
    high_tau: float | None,
    polyak_tau: float,
) -> list[str]:
    argv = [
        "--guidance_mode",
        "similarity",
        "--output_dir",
        str(output_dir),
        "--shape_superquadric_path",
        str(asset_paths["all"]),
        "--shape_tau",
        str(low_tau),
        "--polyak_update_tau",
        str(polyak_tau),
        "--n_repaint_steps",
        str(REPAINT_STEPS),
        "--texture_optim_steps",
        str(TEXTURE_OPTIM_STEPS),
        "--text_prompt",
        prompt,
    ]
    if high_tau is not None:
        argv.extend(
            [
                "--shape_superquadric_high_control_path",
                str(asset_paths["high_control"]),
                "--shape_tau_high_control",
                str(high_tau),
                "--low_control_superquadric_mask_path",
                str(asset_paths["low_control_bbox"]),
                "--local_tau_mode",
                "low_control_mask",
            ]
        )
    argv.extend(
        [
            "--convert_yup_to_zup",
            "--full_pipeline",
            "--appearance_text",
            prompt,
        ]
    )
    return argv


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": f"01_local_tau3_tau10_polyak{_num_tag(POLYAK_TAU)}",
            "mode": "local_tau",
            "low_tau": LOW_TAU,
            "high_tau": HIGH_TAU,
            "polyak_tau": POLYAK_TAU,
        },
        {
            "name": "02_global_tau3_polyak0",
            "mode": "global_tau",
            "low_tau": LOW_TAU,
            "high_tau": None,
            "polyak_tau": 0.0,
        },
        {
            "name": "03_global_tau10_polyak0",
            "mode": "global_tau",
            "low_tau": HIGH_TAU,
            "high_tau": None,
            "polyak_tau": 0.0,
        },
    ]


def _manifest_variants(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in variant.items() if key != "argv"} for variant in variants]


def _experiment_runner_config(run_dir: Path, variants: list[dict[str, Any]], comparison_enabled: bool) -> dict[str, Any]:
    return {
        "spaceflow_config": "config/default.yaml",
        "run_dir": str(run_dir),
        "variants": variants,
        "comparison": {
            "enabled": comparison_enabled,
            "output_name": "output/variant_comparison_lower_camera.png",
            "azim": 0.0,
            "elev": 55.0,
        },
        "texture_optim_steps": TEXTURE_OPTIM_STEPS,
    }


def _variant_status(output_dir: Path) -> str:
    status_path = output_dir / "status.txt"
    if not status_path.is_file():
        return "pending"
    return status_path.read_text(encoding="utf-8", errors="replace").strip() or "unknown"


def _example_status(variants: list[dict[str, Any]]) -> str:
    statuses = [_variant_status(Path(str(variant["output_dir"]))) for variant in variants]
    if all(status == "succeeded" for status in statuses):
        return "succeeded"
    if any(status.startswith("failed") for status in statuses):
        return "failed"
    return "pending"


def _run_config(prompt: str, output_name: str) -> dict[str, Any]:
    return {
        "textPrompt": prompt,
        "appearanceMode": "text",
        "appearanceText": prompt,
        "appearanceImagePath": "",
        "textureMode": "text",
        "globalTextureText": prompt,
        "globalTextureImagePath": "",
        "localTextureTexts": [],
        "localTextureImagePaths": [],
        "lowTau": LOW_TAU,
        "highTau": HIGH_TAU,
        "polyakTau": POLYAK_TAU,
        "repaintSteps": REPAINT_STEPS,
        "textureOptimSteps": TEXTURE_OPTIM_STEPS,
        "outputName": output_name,
        "convertYupToZup": True,
        "lowControlBBoxMargin": BBOX_MARGIN_FRACTION,
        "dryRun": False,
        "experimentMode": True,
        "experimentType": "geometry",
    }


def _texture_guidance(prompt: str, primitive_count: int) -> dict[str, Any]:
    return {
        "mode": "text",
        "saved_uploads": {},
        "global_text": prompt,
        "local_override_count": 0,
        "local_text_prompts": [""] * primitive_count,
    }


def _write_variant_run_configs(variants: list[dict[str, Any]]) -> None:
    for variant in variants:
        output_dir = Path(str(variant["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        _json_write(
            output_dir / "run_config.json",
            {key: value for key, value in variant.items() if key != "argv"},
        )


def _prepare_example(
    index: int,
    prompt: str,
    source_npz: Path,
    batch_run_id: str,
    run_dir: Path,
    asset_root: Path,
    created_at: str,
) -> dict[str, Any]:
    arrays = _load_npz_arrays(source_npz)
    control_levels = np.asarray(arrays["control_levels"], dtype=np.float64)
    high_mask = control_levels >= 0.5
    low_mask = control_levels < 0.5
    high_count = int(high_mask.sum())
    low_count = int(low_mask.sum())
    primitive_count = int(control_levels.shape[0])
    if high_count == 0:
        raise ValueError(f"{source_npz} has no high-control primitives")
    if low_count == 0:
        raise ValueError(f"{source_npz} has no low-control primitives")

    slug = _sanitize_name(prompt.lower(), fallback=f"example_{index:02d}")
    output_name = f"{index:02d}_{slug}_experiment"
    example_run_id = f"{batch_run_id}_{output_name}"
    example_run_dir = run_dir / output_name
    output_dir = example_run_dir / "output"
    asset_dir = asset_root / batch_run_id / f"{index:02d}_{slug}"
    asset_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    asset_paths = {
        "all": asset_dir / "all.npz",
        "high_control": asset_dir / "high_control.npz",
        "low_control_bbox": asset_dir / "low_control_bbox.npz",
    }
    shutil.copy2(source_npz, asset_paths["all"])
    _save_subset_npz(asset_paths["high_control"], arrays, high_mask)
    bbox = _save_low_control_bbox(asset_paths["low_control_bbox"], arrays, low_mask)

    manifest = {
        "project_name": output_name,
        "saved_at": created_at,
        "save_id": example_run_id,
        "source_npz": str(source_npz),
        "prompt": prompt,
        "counts": {
            "all": primitive_count,
            "high": high_count,
            "low": low_count,
        },
        "low_tau": LOW_TAU,
        "high_tau": HIGH_TAU,
        "bbox_margin_fraction": BBOX_MARGIN_FRACTION,
        "bbox": bbox,
        "paths": {key: str(path) for key, path in asset_paths.items()},
        "primitives": [
            {
                "index": primitive_index,
                "name": f"SQ {primitive_index + 1}",
                "controlLevel": "low" if float(control_levels[primitive_index]) < 0.5 else "high",
                "visible": True,
            }
            for primitive_index in range(primitive_count)
        ],
    }
    manifest_path = asset_dir / "manifest.json"
    _json_write(manifest_path, manifest)

    asset_entry = {
        "id": example_run_id,
        "project_name": output_name,
        "saved_at": created_at,
        "asset_dir": str(asset_dir),
        "manifest_path": str(manifest_path),
        "paths": {key: str(path) for key, path in asset_paths.items()},
        "counts": manifest["counts"],
    }

    variants: list[dict[str, Any]] = []
    for spec in _variant_specs():
        variant_name = _sanitize_name(str(spec["name"]))
        variant_output_dir = output_dir / variant_name
        variant = {
            "name": variant_name,
            "output_dir": str(variant_output_dir),
            "argv": _variant_argv(
                asset_paths,
                variant_output_dir,
                prompt,
                low_tau=float(spec["low_tau"]),
                high_tau=None if spec["high_tau"] is None else float(spec["high_tau"]),
                polyak_tau=float(spec["polyak_tau"]),
            ),
            "mode": spec["mode"],
            "low_tau": spec["low_tau"],
            "high_tau": spec["high_tau"],
            "polyak_tau": spec["polyak_tau"],
            "n_repaint_steps": REPAINT_STEPS,
            "texture_optim_steps": TEXTURE_OPTIM_STEPS,
            "example_index": index,
            "prompt": prompt,
        }
        variants.append(variant)

    _json_write(output_dir / "experiment_manifest.json", {"variants": _manifest_variants(variants)})
    _write_variant_run_configs(variants)
    _json_write(example_run_dir / "experiment_runner_config.json", _experiment_runner_config(example_run_dir, variants, True))

    run_config = _run_config(prompt, output_name)
    run_meta = {
        "run_id": example_run_id,
        "status": "prepared",
        "project_name": "spaceflow_examples_structure",
        "created_at": created_at,
        "asset_entry": asset_entry,
        "output_dir": str(output_dir),
        "log_path": str(example_run_dir / "spaceflow.log"),
        "command": [],
        "run_config": run_config,
        "pipeline_stage": "full_pipeline",
        "launch_mode": "batch_local",
        "experiment_mode": True,
        "experiment_type": "geometry",
        "texture_guidance": _texture_guidance(prompt, primitive_count),
        "experiment_runner_config": str(example_run_dir / "experiment_runner_config.json"),
        "experiment_variants": _manifest_variants(variants),
        "source_npz": str(source_npz),
        "example_index": index,
    }
    _json_write(example_run_dir / "run_meta.json", run_meta)

    return {
        "index": index,
        "prompt": prompt,
        "slug": slug,
        "source_npz": str(source_npz),
        "run_dir": str(example_run_dir),
        "output_dir": str(output_dir),
        "asset_dir": str(asset_dir),
        "asset_entry": asset_entry,
        "manifest_path": str(manifest_path),
        "counts": manifest["counts"],
        "variants": variants,
        "run_meta": run_meta,
    }


def _update_example_run_meta(
    example: dict[str, Any],
    *,
    status: str,
    command: list[str],
    returncode: int | None,
    comparison_path: Path | None,
) -> None:
    run_meta = dict(example["run_meta"])
    run_meta["status"] = status
    run_meta["command"] = command
    if returncode is not None:
        run_meta["returncode"] = returncode
    if comparison_path is not None:
        run_meta["comparison_path"] = str(comparison_path)
    run_meta["experiment_variants"] = _manifest_variants(example["variants"])
    _json_write(Path(str(example["run_dir"])) / "run_meta.json", run_meta)
    example["run_meta"] = run_meta


def _batch_manifest(
    *,
    batch_run_id: str,
    created_at: str,
    run_dir: Path,
    asset_root: Path,
    selected: list[int],
    examples: list[dict[str, Any]],
    batch_config_path: Path,
    command: list[str],
    status: str,
    returncode: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": batch_run_id,
        "status": status,
        "created_at": created_at,
        "updated_at": _utc_timestamp(),
        "run_dir": str(run_dir),
        "asset_root": str(asset_root),
        "selected_examples": selected,
        "batch_config": str(batch_config_path),
        "command": command,
        "defaults": {
            "low_tau": LOW_TAU,
            "high_tau": HIGH_TAU,
            "polyak_tau": POLYAK_TAU,
            "repaint_steps": REPAINT_STEPS,
            "texture_optim_steps": TEXTURE_OPTIM_STEPS,
            "bbox_margin_fraction": BBOX_MARGIN_FRACTION,
            "full_pipeline": True,
            "convert_yup_to_zup": True,
        },
        "examples": [
            {
                "index": example["index"],
                "prompt": example["prompt"],
                "source_npz": example["source_npz"],
                "run_dir": example["run_dir"],
                "asset_dir": example["asset_dir"],
                "counts": example["counts"],
                "status": example.get("status", "prepared"),
                "comparison_path": example.get("comparison_path"),
                "variants": _manifest_variants(example["variants"]),
            }
            for example in examples
        ],
    }
    if returncode is not None:
        payload["returncode"] = returncode
    return payload


def _render_example(python_bin: str, example: dict[str, Any]) -> Path | None:
    run_dir = Path(str(example["run_dir"]))
    output_path = run_dir / "output" / "variant_comparison_lower_camera.png"
    command = [
        python_bin,
        str(COMPARISON_RENDERER),
        str(run_dir),
        "--output-name",
        "output/variant_comparison_lower_camera.png",
        "--azim",
        "0.0",
        "--elev",
        "55.0",
    ]
    print(f"{_local_timestamp()} [examples-structure] rendering comparison: {run_dir}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=False)
    return output_path if output_path.is_file() else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 20 example SpaceFlow structure experiments")
    parser.add_argument("--examples-dir", type=Path, default=DEFAULT_EXAMPLES_DIR)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--python-bin", default=_default_python_bin())
    parser.add_argument("--only", default=None, help="Comma-separated examples to run, e.g. 1,12,20 or 1-3")
    parser.add_argument("--run-id", default=None, help="Optional deterministic batch run id")
    parser.add_argument(
        "--texture-optim-steps",
        type=int,
        default=TEXTURE_OPTIM_STEPS,
        help="Similarity/texture optimization steps passed to each SpaceFlow variant (default: 2)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate assets/configs without launching SpaceFlow")
    parser.add_argument("--skip-render", action="store_true", help="Skip per-example comparison rendering after execution")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global TEXTURE_OPTIM_STEPS

    args = parse_args(argv)
    if args.texture_optim_steps < 2:
        raise ValueError("--texture-optim-steps must be at least 2")
    TEXTURE_OPTIM_STEPS = int(args.texture_optim_steps)
    examples_dir = args.examples_dir.expanduser().resolve()
    prompts_path = args.prompts.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    python_bin = str(Path(args.python_bin).expanduser()) if "/" in str(args.python_bin) else str(args.python_bin)
    selected = _parse_only(args.only)
    prompts = _parse_prompts(prompts_path)
    missing_prompts = [index for index in selected if index not in prompts]
    if missing_prompts:
        raise ValueError(f"Missing prompts for example(s): {missing_prompts}")

    created_at = _utc_timestamp()
    batch_run_id = args.run_id or f"{created_at}_examples_structure_experiment"
    run_dir = run_root / batch_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    asset_root.mkdir(parents=True, exist_ok=True)

    print(
        f"{_local_timestamp()} [examples-structure] preparing {len(selected)} example(s) "
        f"under {run_dir}",
        flush=True,
    )

    examples: list[dict[str, Any]] = []
    all_variants: list[dict[str, Any]] = []
    seen_output_dirs: set[str] = set()
    for index in selected:
        source_npz = examples_dir / f"{index}.npz"
        if not source_npz.is_file():
            raise FileNotFoundError(f"Missing example NPZ: {source_npz}")
        example = _prepare_example(
            index,
            prompts[index],
            source_npz,
            batch_run_id,
            run_dir,
            asset_root,
            created_at,
        )
        for variant in example["variants"]:
            output_dir = str(variant["output_dir"])
            if output_dir in seen_output_dirs:
                raise ValueError(f"Duplicate variant output_dir: {output_dir}")
            seen_output_dirs.add(output_dir)
            all_variants.append(variant)
        examples.append(example)
        print(
            f"{_local_timestamp()} [examples-structure] prepared example {index:02d}: "
            f"{prompts[index]} ({example['counts']['high']} high, {example['counts']['low']} low)",
            flush=True,
        )

    batch_config_path = run_dir / "batch_experiment_runner_config.json"
    _json_write(batch_config_path, _experiment_runner_config(run_dir, all_variants, False))
    command = [python_bin, str(EXPERIMENT_RUNNER), "--config", str(batch_config_path)]

    manifest_path = run_dir / "batch_manifest.json"
    for example in examples:
        example["status"] = "dry_run" if args.dry_run else "prepared"
        _update_example_run_meta(
            example,
            status=example["status"],
            command=command,
            returncode=None,
            comparison_path=None,
        )
    _json_write(
        manifest_path,
        _batch_manifest(
            batch_run_id=batch_run_id,
            created_at=created_at,
            run_dir=run_dir,
            asset_root=asset_root,
            selected=selected,
            examples=examples,
            batch_config_path=batch_config_path,
            command=command,
            status="dry_run" if args.dry_run else "prepared",
            returncode=None,
        ),
    )

    if args.dry_run:
        print(f"{_local_timestamp()} [examples-structure] dry run complete: {manifest_path}", flush=True)
        return 0

    print(
        f"{_local_timestamp()} [examples-structure] launching {len(all_variants)} variants "
        f"with shared runner",
        flush=True,
    )
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    returncode = int(result.returncode)

    render_failures: list[int] = []
    for example in examples:
        status = _example_status(example["variants"])
        comparison_path = None
        if status == "succeeded" and not args.skip_render:
            comparison_path = _render_example(python_bin, example)
            if comparison_path is None:
                render_failures.append(int(example["index"]))
        example["status"] = status
        if comparison_path is not None:
            example["comparison_path"] = str(comparison_path)
        _update_example_run_meta(
            example,
            status=status,
            command=command,
            returncode=returncode,
            comparison_path=comparison_path,
        )

    if returncode != 0:
        batch_status = "failed"
    elif render_failures:
        batch_status = "failed_render"
    elif all(example.get("status") == "succeeded" for example in examples):
        batch_status = "succeeded"
    else:
        batch_status = "failed"

    _json_write(
        manifest_path,
        _batch_manifest(
            batch_run_id=batch_run_id,
            created_at=created_at,
            run_dir=run_dir,
            asset_root=asset_root,
            selected=selected,
            examples=examples,
            batch_config_path=batch_config_path,
            command=command,
            status=batch_status,
            returncode=returncode,
        ),
    )
    print(f"{_local_timestamp()} [examples-structure] batch status: {batch_status}", flush=True)
    print(f"{_local_timestamp()} [examples-structure] manifest: {manifest_path}", flush=True)
    return 0 if batch_status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
