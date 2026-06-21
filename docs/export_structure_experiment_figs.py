#!/usr/bin/env python3
"""Export transparent structure-experiment variant renders by asset and view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
DOCS_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(DOCS_DIR) not in sys.path:
    sys.path.insert(0, str(DOCS_DIR))

from export_tau10_distance_parts import (  # noqa: E402
    UI_SQ_HIGH_COLOR,
    UI_SQ_LOW_COLOR,
    _rasterize_mesh_rgba,
    _save_rgba,
)
from render_latent_control_figure import render_utils  # noqa: E402
from render_tau10_distance_figure import (  # noqa: E402
    FIVE_VIEW_SPECS,
    SINGLE_VIEW_SPEC,
    _batch_run_dirs,
    _parse_only,
    _run_output_name,
    _status_for,
)


DEFAULT_BATCH_ROOT = REPO_ROOT / "spatial_control_user_study"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "structure_experiment_figs"
STRUCTURE_VARIANTS = [
    {
        "key": "local_tau",
        "dir": "01_local_tau3_tau10_polyak0p18",
        "label": "Local tau, low 3 / high 10",
    },
    {
        "key": "tau3",
        "dir": "02_global_tau3_polyak0",
        "label": "Global tau 3",
    },
    {
        "key": "tau10",
        "dir": "03_global_tau10_polyak0",
        "label": "Global tau 10",
    },
]
SUPERQUADRICS_ITEM = {
    "key": "superquadrics",
    "label": "Input colored superquadrics",
}
OUTPUT_ITEMS = [SUPERQUADRICS_ITEM, *STRUCTURE_VARIANTS]
TEXTURE_COMPARISON_VARIANTS = [
    {
        "key": "spaceflow_local_texture_routing",
        "dir": "01_spaceflow_local_texture_routing",
        "label": "SpaceFlow local texture routing",
    },
    {
        "key": "trellis_raw_flat_prompt",
        "dir": "02_trellis_raw_flat_prompt",
        "label": "TRELLIS raw flat prompt",
        "render_yaw_deg": 180.0,
    },
    {
        "key": "fixed_structure_appearance_fm",
        "dir": "03_fixed_structure_appearance_fm",
        "label": "SpaceFlow structure, TRELLIS appearance FM",
    },
    {
        "key": "fixed_structure_guideflow_appearance_fm",
        "dir": "04_fixed_structure_guideflow_appearance_fm",
        "label": "SpaceFlow structure, GuideFlow appearance FM",
    },
]
TEXTURE_COMPARISON_ITEMS = [SUPERQUADRICS_ITEM, *TEXTURE_COMPARISON_VARIANTS]

TEN_VIEW_SPECS = [
    {
        "name": f"view{i + 1:02d}",
        "azim": azim,
        "elev": 80.0,
    }
    for i, azim in enumerate([0.0, 30.0, 60.0, 90.0, 135.0, 180.0, 210.0, 240.0, 270.0, 315.0])
]


def _view_specs(view_set: str) -> list[dict[str, float | str]]:
    if view_set == "single":
        return [SINGLE_VIEW_SPEC]
    if view_set == "five":
        return FIVE_VIEW_SPECS
    if view_set == "ten":
        return TEN_VIEW_SPECS
    raise ValueError(f"Unknown view set: {view_set}")


def _is_single_run(run_dir: Path) -> bool:
    return not (run_dir / "output" / str(STRUCTURE_VARIANTS[0]["dir"])).is_dir()


def _single_run_file(run_dir: Path, filename: str) -> Path:
    """Return a file from either a standalone variant dir or nested output dir."""
    candidates = [
        run_dir / filename,
        run_dir / "output" / filename,
        run_dir / "output" / "01_spaceflow_local_texture_routing" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[-1]


def _single_run_manifest(run_dir: Path) -> dict[str, object]:
    npz_path = _single_run_file(run_dir, "input_superquadrics_all.npz")
    if not npz_path.is_file():
        return {}
    data = np.load(npz_path)
    control_levels = np.asarray(
        data["control_levels"] if "control_levels" in data.files else [],
        dtype=np.float64,
    )
    return {
        "primitives": [
            {
                "index": index,
                "name": f"SQ {index + 1}",
                "controlLevel": "high" if float(level) >= 0.5 else "low",
            }
            for index, level in enumerate(control_levels)
        ]
    }


def _get_output_items(run_dir: Path) -> list[dict[str, str]]:
    if not _is_single_run(run_dir):
        return OUTPUT_ITEMS
    return [
        SUPERQUADRICS_ITEM,
        {
            "key": "generated_asset",
            "label": "Generated asset",
        }
    ]


def _variant_statuses_succeeded(run_dir: Path) -> tuple[bool, dict[str, str | None]]:
    if _is_single_run(run_dir):
        has_sq = _single_run_file(run_dir, "spatial_control_mesh.ply").is_file()
        return has_sq, {"single_run": "succeeded" if has_sq else "missing_superquadrics"}
    statuses = {
        str(variant["key"]): _status_for(run_dir, str(variant["key"]).replace("local_tau", "local"))
        for variant in STRUCTURE_VARIANTS
    }
    return all(status == "succeeded" for status in statuses.values()), statuses


def _variant_mesh_path(run_dir: Path, variant: dict[str, object], mesh_file: str) -> Path:
    return run_dir / "output" / str(variant["dir"]) / mesh_file


def _load_variant_meshes(run_dir: Path, mesh_file: str) -> dict[str, dict[str, object]]:
    if _is_single_run(run_dir):
        meshes = {"superquadrics": _load_superquadric_mesh(run_dir)}
        path = _single_run_file(run_dir, mesh_file)
        if not path.is_file():
            raise FileNotFoundError(path)
        meshes["generated_asset"] = render_utils.load_mesh(path)
    else:
        meshes = {"superquadrics": _load_superquadric_mesh(run_dir)}
        for variant in STRUCTURE_VARIANTS:
            path = _variant_mesh_path(run_dir, variant, mesh_file)
            if not path.is_file():
                raise FileNotFoundError(path)
            meshes[str(variant["key"])] = render_utils.load_mesh(path)
    return meshes


def _load_superquadric_mesh(run_dir: Path) -> dict[str, object]:
    meta = render_utils.run_meta(run_dir)
    manifest = render_utils.asset_manifest(meta)
    if _is_single_run(run_dir):
        path = _single_run_file(run_dir, "spatial_control_mesh.ply")
        if not render_utils.primitive_rows(manifest):
            manifest = _single_run_manifest(run_dir)
    else:
        path = run_dir / "output" / str(STRUCTURE_VARIANTS[0]["dir"]) / "spatial_control_mesh.ply"
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = render_utils.load_sq_mesh(path, manifest)
    primitives = render_utils.primitive_rows(manifest)
    if primitives:
        faces = np.asarray(mesh["faces"], dtype=np.int64)
        vertices = np.asarray(mesh["vertices"], dtype=np.float64)
        vertices_per_primitive = len(vertices) // len(primitives)
        if vertices_per_primitive > 0:
            primitive_colors = np.array([
                UI_SQ_LOW_COLOR if primitive.get("controlLevel") == "low" else UI_SQ_HIGH_COLOR
                for primitive in primitives
            ])
            face_primitive_indices = np.clip(faces.min(axis=1) // vertices_per_primitive, 0, len(primitives) - 1)
            mesh["face_colors"] = primitive_colors[face_primitive_indices]
    return mesh


def _shared_scale(meshes: dict[str, dict[str, object]]) -> float:
    max_extent = 0.0
    for mesh in meshes.values():
        vertices = np.asarray(mesh["vertices"], dtype=np.float64)
        max_extent = max(max_extent, float(np.max(vertices.max(axis=0) - vertices.min(axis=0))))
    return 1.0 / max(max_extent, 1e-8)


def _project_mesh_for_view(
    mesh_info: dict[str, object],
    rotation: np.ndarray,
    scale: float,
) -> dict[str, object]:
    render_yaw_deg = float(mesh_info.get("render_yaw_deg") or 0.0)
    mesh_rotation = render_utils.camera_rotation(render_yaw_deg, 0.0) if render_yaw_deg else np.eye(3)
    vertices_camera = ((np.asarray(mesh_info["vertices"], dtype=np.float64) * scale) @ mesh_rotation.T) @ rotation.T
    vertices_camera[:, 1] *= -1.0
    normals_camera = (np.asarray(mesh_info["vertex_normals"], dtype=np.float64) @ mesh_rotation.T) @ rotation.T
    return {
        **mesh_info,
        "vertices": vertices_camera,
        "vertex_normals": normals_camera,
    }


def _render_view_variants(
    meshes: dict[str, dict[str, object]],
    azim: float,
    elev: float,
) -> dict[str, np.ndarray]:
    rotation = render_utils.camera_rotation(azim, elev)
    scale = _shared_scale(meshes)
    projected = {
        key: _project_mesh_for_view(mesh, rotation, scale)
        for key, mesh in meshes.items()
    }
    max_abs = np.array([0.0, 0.0], dtype=np.float64)
    for mesh in projected.values():
        vertices = np.asarray(mesh["vertices"], dtype=np.float64)
        max_abs = np.maximum(max_abs, np.max(np.abs(vertices[:, :2]), axis=0))
    lim = float(max(max_abs[0] * 1.12, max_abs[1] * 1.18, 1e-6))

    light_dir = np.array([-0.35, -0.45, 0.82], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    return {
        key: _rasterize_mesh_rgba(mesh, lim, lim, light_dir)
        for key, mesh in projected.items()
    }


def export_asset(
    run_dir: Path,
    output_dir: Path,
    view_specs: list[dict[str, float | str]],
    mesh_file: str,
    crop: bool,
) -> dict[str, object]:
    meshes = _load_variant_meshes(run_dir, mesh_file)
    asset_name = _run_output_name(run_dir)
    asset_dir = output_dir / asset_name
    views = []
    items = _get_output_items(run_dir)
    for view_spec in view_specs:
        view_name = str(view_spec["name"])
        azim = float(view_spec["azim"])
        elev = float(view_spec["elev"])
        view_dir = asset_dir / view_name
        rendered = _render_view_variants(meshes, azim, elev)
        variant_paths = {}
        for item in items:
            key = str(item["key"])
            path = view_dir / f"{key}.png"
            _save_rgba(path, rendered[key], crop=crop)
            variant_paths[key] = str(path)
        views.append({
            "name": view_name,
            "azim": azim,
            "elev": elev,
            "variants": variant_paths,
        })

    return {
        "run_dir": str(run_dir),
        "asset_dir": str(asset_dir),
        "mesh_file": mesh_file,
        "view_count": len(views),
        "views": views,
    }


def _texture_variant_mesh_path(run_dir: Path, variant: dict[str, object], mesh_file: str) -> Path:
    return run_dir / "output" / str(variant["dir"]) / mesh_file


def _load_texture_comparison_meshes(run_dir: Path, mesh_file: str) -> dict[str, dict[str, object]]:
    meshes = {"superquadrics": _load_superquadric_mesh(run_dir)}
    for variant in TEXTURE_COMPARISON_VARIANTS:
        path = _texture_variant_mesh_path(run_dir, variant, mesh_file)
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh = render_utils.load_mesh(path)
        render_yaw_deg = float(variant.get("render_yaw_deg") or 0.0)
        if render_yaw_deg:
            mesh["render_yaw_deg"] = render_yaw_deg
        meshes[str(variant["key"])] = mesh
    return meshes


def export_texture_comparisons(
    run_dir: Path,
    output_dir: Path,
    view_specs: list[dict[str, float | str]],
    mesh_file: str,
    crop: bool,
) -> dict[str, object]:
    meshes = _load_texture_comparison_meshes(run_dir, mesh_file)
    asset_name = _run_output_name(run_dir)
    comparison_dir = output_dir / asset_name / "comparisons"
    views = []
    for view_spec in view_specs:
        view_name = str(view_spec["name"])
        azim = float(view_spec["azim"])
        elev = float(view_spec["elev"])
        view_dir = comparison_dir / view_name
        rendered = _render_view_variants(meshes, azim, elev)
        variant_paths = {}
        for item in TEXTURE_COMPARISON_ITEMS:
            key = str(item["key"])
            path = view_dir / f"{key}.png"
            _save_rgba(path, rendered[key], crop=crop)
            variant_paths[key] = str(path)
        views.append({
            "name": view_name,
            "azim": azim,
            "elev": elev,
            "variants": variant_paths,
        })

    manifest = {
        "run_dir": str(run_dir),
        "comparison_dir": str(comparison_dir),
        "mesh_file": mesh_file,
        "crop": crop,
        "items": TEXTURE_COMPARISON_ITEMS,
        "view_count": len(views),
        "views": views,
    }
    manifest_path = comparison_dir / "comparison_parts_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def export_batch(
    batch_root: Path,
    output_dir: Path,
    only: set[str] | None,
    view_specs: list[dict[str, float | str]],
    mesh_file: str,
    crop: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    skipped = []
    for run_dir in _batch_run_dirs(batch_root, only):
        ok, statuses = _variant_statuses_succeeded(run_dir)
        if not ok:
            skipped.append({
                "run_dir": str(run_dir),
                "reason": "local, tau3, and tau10 variants are required",
                "statuses": statuses,
            })
            print(f"skip {run_dir.name}: {statuses}", flush=True)
            continue
        try:
            result = export_asset(run_dir, output_dir, view_specs, mesh_file, crop)
        except Exception as exc:  # noqa: BLE001
            skipped.append({
                "run_dir": str(run_dir),
                "reason": f"export failed: {exc}",
                "statuses": statuses,
            })
            print(f"failed {run_dir.name}: {exc}", flush=True)
            continue
        rendered.append(result)
        print(f"exported {run_dir.name}", flush=True)

    manifest = {
        "batch_root": str(batch_root),
        "output_dir": str(output_dir),
        "mesh_file": mesh_file,
        "crop": crop,
        "variants": STRUCTURE_VARIANTS,
        "items": OUTPUT_ITEMS,
        "superquadric_colors": {
            "high": "#f59e0b",
            "low": "#f8fafc",
        },
        "view_specs": view_specs,
        "rendered_asset_count": len(rendered),
        "rendered_part_count": sum(int(item["view_count"]) * len(item["views"][0]["variants"]) for item in rendered),
        "skipped_count": len(skipped),
        "rendered": rendered,
        "skipped": skipped,
    }
    manifest_path = output_dir / "structure_experiment_figs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, help="Export one experiment directory instead of batch mode.")
    parser.add_argument("--batch-run-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", default="", help="Comma-separated run names or numeric prefixes.")
    parser.add_argument("--view-set", choices=["single", "five", "ten"], default="five")
    parser.add_argument("--mesh-file", default="out_sim.glb", help="Variant mesh filename to render.")
    parser.add_argument("--texture-comparisons", action="store_true", help="Export all texture-experiment variants under comparisons/.")
    parser.add_argument("--keep-canvas", action="store_true", help="Do not crop transparent whitespace.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    view_specs = _view_specs(args.view_set)
    crop = not args.keep_canvas
    if args.run_dir:
        if args.texture_comparisons:
            result = export_texture_comparisons(args.run_dir.resolve(), output_dir, view_specs, args.mesh_file, crop)
            print(f"exported {result['comparison_dir']}")
            print(f"wrote {result['manifest_path']}")
            return 0
        result = export_asset(args.run_dir.resolve(), output_dir, view_specs, args.mesh_file, crop)
        print(f"exported {result['asset_dir']}")
        return 0

    manifest = export_batch(
        args.batch_run_root.resolve(),
        output_dir,
        _parse_only(args.only),
        view_specs,
        args.mesh_file,
        crop,
    )
    print(f"exported {manifest['rendered_asset_count']} assets")
    print(f"wrote {manifest['rendered_part_count']} transparent renders")
    print(f"skipped {manifest['skipped_count']} runs")
    print(f"wrote {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
