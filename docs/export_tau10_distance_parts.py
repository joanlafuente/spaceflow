#!/usr/bin/env python3
"""Export transparent A/B/C tau10-distance figure parts by view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
DOCS_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(DOCS_DIR) not in sys.path:
    sys.path.insert(0, str(DOCS_DIR))

from render_latent_control_figure import (  # noqa: E402
    _camera_project,
    _low_control_mask,
    _variant_data,
    render_utils,
)
from render_tau10_distance_figure import (  # noqa: E402
    FIVE_VIEW_SPECS,
    SINGLE_VIEW_SPEC,
    _batch_orange_threshold_stats,
    _batch_run_dirs,
    _distance_summary,
    _orange_threshold_stats,
    _parse_only,
    _required_variants_succeeded,
    _run_output_name,
    _tau10_distance,
)


DEFAULT_BATCH_ROOT = REPO_ROOT / "spatial_control_user_study"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "tau10_distance_user_study_parts"
UI_SQ_HIGH_COLOR = np.array([0xf5, 0x9e, 0x0b], dtype=np.float64) / 255.0
UI_SQ_LOW_COLOR = np.array([0xf8, 0xfa, 0xfc], dtype=np.float64) / 255.0
TAU10_DISTANCE_LOW_COLOR = "#1a9850"
TAU10_DISTANCE_MID_COLOR = "#f4f4f4"
TAU10_DISTANCE_HIGH_COLOR = "#d73027"


def _view_specs(view_set: str) -> list[dict[str, float | str]]:
    if view_set == "single":
        return [SINGLE_VIEW_SPEC]
    if view_set == "five":
        return FIVE_VIEW_SPECS
    raise ValueError(f"Unknown view set: {view_set}")


def _crop_alpha(image: np.ndarray, pad: int = 12) -> np.ndarray:
    alpha = image[:, :, 3]
    ys, xs = np.where(alpha > 1e-4)
    if len(xs) == 0 or len(ys) == 0:
        return image
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, image.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, image.shape[1])
    return image[y0:y1, x0:x1]


def _save_rgba(path: Path, image: np.ndarray, crop: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if crop:
        image = _crop_alpha(image)
    rgba = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path)


def _rasterize_mesh_rgba(mesh_info: dict[str, object], lim_x: float, lim_y: float, light_dir: np.ndarray) -> np.ndarray:
    resolution = render_utils.PANEL_RENDER_SIZE
    image = np.zeros((resolution, resolution, 4), dtype=np.float64)
    zbuffer = np.full((resolution, resolution), -np.inf, dtype=np.float64)

    vertices = np.asarray(mesh_info["vertices"], dtype=np.float64)
    faces = np.asarray(mesh_info["faces"], dtype=np.int64)
    normals = np.asarray(mesh_info["vertex_normals"], dtype=np.float64)
    face_colors = np.asarray(mesh_info["face_colors"], dtype=np.float64)
    uv = mesh_info.get("uv")
    uv = np.asarray(uv, dtype=np.float64) if uv is not None else None
    texture = mesh_info.get("texture")
    texture = np.asarray(texture, dtype=np.float64) if texture is not None else None
    material_color = np.asarray(mesh_info["material_color"], dtype=np.float64)[:3]

    screen_x = (vertices[:, 0] + lim_x) / max(2.0 * lim_x, 1e-8) * (resolution - 1)
    screen_y = (lim_y - vertices[:, 1]) / max(2.0 * lim_y, 1e-8) * (resolution - 1)
    screen_vertices = np.column_stack([screen_x, screen_y, vertices[:, 2]])

    for face_index, face in enumerate(faces):
        tri = screen_vertices[face]
        min_x = max(int(np.floor(tri[:, 0].min())), 0)
        max_x = min(int(np.ceil(tri[:, 0].max())), resolution - 1)
        min_y = max(int(np.floor(tri[:, 1].min())), 0)
        max_y = min(int(np.ceil(tri[:, 1].max())), resolution - 1)
        if min_x > max_x or min_y > max_y:
            continue

        x0, y0 = tri[0, :2]
        x1, y1 = tri[1, :2]
        x2, y2 = tri[2, :2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-10:
            continue

        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        px = xx.astype(np.float64) + 0.5
        py = yy.astype(np.float64) + 0.5
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        w2 = 1.0 - w0 - w1
        mask = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not np.any(mask):
            continue

        depth = w0 * tri[0, 2] + w1 * tri[1, 2] + w2 * tri[2, 2]
        local_zbuffer = zbuffer[min_y : max_y + 1, min_x : max_x + 1]
        update = mask & (depth > local_zbuffer)
        if not np.any(update):
            continue

        weights = np.stack([w0[update], w1[update], w2[update]], axis=1)
        if texture is not None and uv is not None:
            colors = render_utils._sample_texture(texture, weights @ uv[face])[:, :3] * material_color[None, :]
        else:
            colors = np.tile(face_colors[face_index], (len(weights), 1))

        pixel_normals = weights @ normals[face]
        pixel_normals /= np.maximum(np.linalg.norm(pixel_normals, axis=1, keepdims=True), 1e-8)
        intensity = np.clip(0.68 + 0.32 * np.abs(pixel_normals @ light_dir), 0.50, 1.0)
        colors = np.clip(colors * intensity[:, None], 0.0, 1.0)

        patch = image[min_y : max_y + 1, min_x : max_x + 1]
        patch[update, :3] = colors
        patch[update, 3] = 1.0
        local_zbuffer[update] = depth[update]

    return image


def _render_mesh_rgba(mesh_info: dict[str, object], azim: float, elev: float) -> np.ndarray:
    rotation = render_utils.camera_rotation(azim, elev)
    light_dir = np.array([-0.35, -0.45, 0.82], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    vertices = np.asarray(mesh_info["vertices"], dtype=np.float64)
    max_extent = max(float(np.max(vertices.max(axis=0) - vertices.min(axis=0))), 1e-8)
    scale = 1.0 / max_extent
    vertices_camera = (vertices * scale) @ rotation.T
    vertices_camera[:, 1] *= -1.0
    normals_camera = np.asarray(mesh_info["vertex_normals"], dtype=np.float64) @ rotation.T
    projected_info = {
        **mesh_info,
        "vertices": vertices_camera,
        "vertex_normals": normals_camera,
    }
    max_abs = np.max(np.abs(vertices_camera[:, :2]), axis=0)
    lim = float(max(max_abs[0] * 1.12, max_abs[1] * 1.18, 1e-6))
    return _rasterize_mesh_rgba(projected_info, lim, lim, light_dir)


def _tau10_distance_cmap(orange_threshold: float, vmax: float) -> mcolors.Colormap:
    display_threshold = float(np.clip(orange_threshold, 0.0, vmax))
    transition = float(np.clip(display_threshold / vmax if vmax > 0 else 0.5, 1e-6, 1.0 - 1e-6))
    return mcolors.LinearSegmentedColormap.from_list(
        "tau10_distance",
        [
            (0.0, TAU10_DISTANCE_LOW_COLOR),
            (transition, TAU10_DISTANCE_MID_COLOR),
            (1.0, TAU10_DISTANCE_HIGH_COLOR),
        ],
        N=256,
    )


def _render_distance_rgba(
    points: np.ndarray,
    distance: np.ndarray,
    orange_threshold: float,
    vmax: float,
    azim: float,
    elev: float,
) -> np.ndarray:
    projected = _camera_project(points, azim=azim, elev=elev)
    order = np.argsort(projected[:, 2])
    x = projected[order, 0]
    y = projected[order, 1]
    values = distance[order]

    fig = plt.figure(figsize=(4.0, 4.0), dpi=300)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.scatter(
        x,
        y,
        c=values,
        cmap=_tau10_distance_cmap(orange_threshold, vmax),
        norm=mcolors.Normalize(vmin=0.0, vmax=vmax),
        s=2.3,
        linewidths=0,
        alpha=0.94,
    )
    pad_x = max(float(x.max() - x.min()) * 0.035, 1e-6)
    pad_y = max(float(y.max() - y.min()) * 0.035, 1e-6)
    ax.set_xlim(float(x.min() - pad_x), float(x.max() + pad_x))
    ax.set_ylim(float(y.min() - pad_y), float(y.max() + pad_y))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.float64) / 255.0
    plt.close(fig)
    return image


def _load_asset_inputs(run_dir: Path) -> dict[str, object]:
    data = {key: _variant_data(run_dir, key) for key in ["local", "tau10"]}
    local_points = np.asarray(data["local"]["points"])
    distance = _tau10_distance(data["local"], data["tau10"])
    low_mask = _low_control_mask(Path(data["local"]["dir"]), local_points)

    meta = render_utils.run_meta(run_dir)
    manifest = render_utils.asset_manifest(meta)
    control_mesh = render_utils.load_sq_mesh(
        Path(data["local"]["dir"]) / "spatial_control_mesh.ply",
        manifest,
    )
    primitives = render_utils.primitive_rows(manifest)
    if primitives:
        faces = np.asarray(control_mesh["faces"], dtype=np.int64)
        vertices = np.asarray(control_mesh["vertices"], dtype=np.float64)
        vertices_per_primitive = len(vertices) // len(primitives)
        if vertices_per_primitive > 0:
            primitive_colors = np.array([
                UI_SQ_LOW_COLOR if primitive.get("controlLevel") == "low" else UI_SQ_HIGH_COLOR
                for primitive in primitives
            ])
            face_primitive_indices = np.clip(faces.min(axis=1) // vertices_per_primitive, 0, len(primitives) - 1)
            control_mesh["face_colors"] = primitive_colors[face_primitive_indices]
    result_mesh = render_utils.load_mesh(Path(data["local"]["dir"]) / "out_sim.glb")
    return {
        "local_points": local_points,
        "distance": distance,
        "low_mask": low_mask,
        "control_mesh": control_mesh,
        "result_mesh": result_mesh,
    }


def export_asset_parts(
    run_dir: Path,
    output_dir: Path,
    view_specs: list[dict[str, float | str]],
    distance_vmax: float,
    orange_threshold: float | None,
) -> dict[str, object]:
    inputs = _load_asset_inputs(run_dir)
    local_points = np.asarray(inputs["local_points"])
    distance = np.asarray(inputs["distance"])
    low_mask = np.asarray(inputs["low_mask"])
    threshold_stats = _orange_threshold_stats(distance, low_mask, distance_vmax)
    if orange_threshold is None:
        orange_threshold = float(threshold_stats["threshold"])
        threshold_source = "auto_run_balanced_accuracy"
    else:
        orange_threshold = float(np.clip(orange_threshold, 0.0, distance_vmax))
        threshold_source = "fixed_or_batch"

    asset_name = _run_output_name(run_dir)
    asset_dir = output_dir / asset_name
    views = []
    for view_spec in view_specs:
        view_name = str(view_spec["name"])
        azim = float(view_spec["azim"])
        elev = float(view_spec["elev"])
        view_dir = asset_dir / view_name
        paths = {
            "a": view_dir / "a.png",
            "b": view_dir / "b.png",
            "c": view_dir / "c.png",
        }
        _save_rgba(paths["a"], _render_mesh_rgba(inputs["control_mesh"], azim, elev))
        _save_rgba(paths["b"], _render_mesh_rgba(inputs["result_mesh"], azim, elev))
        _save_rgba(
            paths["c"],
            _render_distance_rgba(local_points, distance, orange_threshold, distance_vmax, azim, elev),
        )
        views.append({
            "name": view_name,
            "azim": azim,
            "elev": elev,
            "a": str(paths["a"]),
            "b": str(paths["b"]),
            "c": str(paths["c"]),
        })

    return {
        "run_dir": str(run_dir),
        "asset_dir": str(asset_dir),
        "view_count": len(views),
        "views": views,
        "distance_vmax": distance_vmax,
        "orange_threshold": orange_threshold,
        "orange_threshold_source": threshold_source,
        "orange_threshold_stats": threshold_stats,
        "summary": _distance_summary(distance, low_mask),
    }


def export_batch_parts(
    batch_root: Path,
    output_dir: Path,
    only: set[str] | None,
    view_specs: list[dict[str, float | str]],
    distance_vmax: float,
    orange_threshold: float | None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    skipped = []
    eligible_run_dirs = []
    for run_dir in _batch_run_dirs(batch_root, only):
        ok, statuses = _required_variants_succeeded(run_dir)
        if not ok:
            skipped.append({
                "run_dir": str(run_dir),
                "reason": "local and tau10 variants are required",
                "statuses": statuses,
            })
            print(f"skip {run_dir.name}: {statuses}", flush=True)
            continue
        eligible_run_dirs.append(run_dir)

    if orange_threshold is None:
        threshold_stats = _batch_orange_threshold_stats(eligible_run_dirs, distance_vmax)
        orange_threshold = float(threshold_stats["threshold"])
        threshold_source = "auto_batch_balanced_accuracy"
        print(f"auto orange threshold: {orange_threshold:.4f}", flush=True)
    else:
        orange_threshold = float(np.clip(orange_threshold, 0.0, distance_vmax))
        threshold_source = "fixed"
        threshold_stats = {
            "threshold": orange_threshold,
            "method": "fixed",
            "balanced_accuracy": None,
            "high_count": None,
            "low_count": None,
            "high_mean": None,
            "low_mean": None,
        }

    for run_dir in eligible_run_dirs:
        try:
            result = export_asset_parts(run_dir, output_dir, view_specs, distance_vmax, orange_threshold)
        except Exception as exc:  # noqa: BLE001
            skipped.append({
                "run_dir": str(run_dir),
                "reason": f"export failed: {exc}",
            })
            print(f"failed {run_dir.name}: {exc}", flush=True)
            continue
        rendered.append(result)
        print(f"exported {run_dir.name}", flush=True)

    manifest = {
        "batch_root": str(batch_root),
        "output_dir": str(output_dir),
        "view_specs": view_specs,
        "distance_vmax": distance_vmax,
        "orange_threshold": orange_threshold,
        "orange_threshold_source": threshold_source,
        "orange_threshold_stats": threshold_stats,
        "rendered_asset_count": len(rendered),
        "rendered_part_count": sum(int(item["view_count"]) * 3 for item in rendered),
        "skipped_count": len(skipped),
        "rendered": rendered,
        "skipped": skipped,
        "superquadric_colors": {
            "high": "#f59e0b",
            "low": "#f8fafc",
        },
    }
    manifest_path = output_dir / "tau10_distance_parts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, help="Export one experiment directory instead of batch mode.")
    parser.add_argument("--batch-run-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", default="", help="Comma-separated run names or numeric prefixes.")
    parser.add_argument("--view-set", choices=["single", "five"], default="five")
    parser.add_argument("--distance-vmax", type=float, default=1.0)
    parser.add_argument("--orange-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    view_specs = _view_specs(args.view_set)
    if args.run_dir:
        result = export_asset_parts(
            args.run_dir.resolve(),
            output_dir,
            view_specs,
            args.distance_vmax,
            args.orange_threshold,
        )
        print(f"exported {result['asset_dir']}")
        return 0

    manifest = export_batch_parts(
        args.batch_run_root.resolve(),
        output_dir,
        _parse_only(args.only),
        view_specs,
        args.distance_vmax,
        args.orange_threshold,
    )
    print(f"exported {manifest['rendered_asset_count']} assets")
    print(f"wrote {manifest['rendered_part_count']} transparent parts")
    print(f"skipped {manifest['skipped_count']} runs")
    print(f"wrote {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
