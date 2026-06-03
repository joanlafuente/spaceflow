#!/usr/bin/env python3
"""Render shared-view comparison plots for SpaceFlow experiment variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection
from trimesh.visual.color import uv_to_interpolated_color

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_RUN_ROOT = Path("/work/courses/3dv/team3/spaceflow_runtime/sq_ui_runs")

VARIANTS = [
    (
        "tau-by-parts\nlow 3 / high 10",
        [
            "output/tau3_tau10_polyak0p18/out_sim.glb",
            "output/01_local_tau3_tau10_polyak0p18/out_sim.glb",
            "output/tau3_tau10_polyak0p18/out_sim_geometry.glb",
            "output/01_local_tau3_tau10_polyak0p18/out_sim_geometry.glb",
        ],
    ),
    (
        "global low tau\n3",
        [
            "output/tau3_polyak0/out_sim.glb",
            "output/02_global_tau3_polyak0/out_sim.glb",
            "output/tau3_polyak0/out_sim_geometry.glb",
            "output/02_global_tau3_polyak0/out_sim_geometry.glb",
        ],
    ),
    (
        "global high tau\n10",
        [
            "output/tau10_polyak0/out_sim.glb",
            "output/03_global_tau10_polyak0/out_sim.glb",
            "output/tau10_polyak0/out_sim_geometry.glb",
            "output/03_global_tau10_polyak0/out_sim_geometry.glb",
        ],
    ),
]


def first_existing(run_dir: Path, rel_paths: list[str]) -> Path | None:
    for rel_path in rel_paths:
        path = run_dir / rel_path
        if path.is_file():
            return path
    return None


def complete_experiment_paths(run_dir: Path) -> list[tuple[str, Path]] | None:
    paths: list[tuple[str, Path]] = []
    for label, rel_paths in VARIANTS:
        path = first_existing(run_dir, rel_paths)
        if path is None:
            return None
        paths.append((label, path))
    return paths


def discover_experiments(root: Path) -> list[Path]:
    candidates = sorted(path for path in root.glob("*_experiment") if path.is_dir())
    return [path for path in candidates if complete_experiment_paths(path) is not None]


def title_for(run_dir: Path) -> str:
    meta_path = run_dir / "run_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            prompt = str(meta.get("run_config", {}).get("textPrompt") or "").strip()
            if prompt:
                return f"Textured comparison: {prompt}"
        except json.JSONDecodeError:
            pass
    name = run_dir.name
    if "_" in name:
        name = name.split("_", 1)[1]
    return "Textured comparison: " + name.replace("_experiment", "").replace("_", " ")


def camera_rotation(azim_deg: float, elev_deg: float) -> np.ndarray:
    azim = np.deg2rad(azim_deg)
    elev = np.deg2rad(elev_deg)
    rz = np.array(
        [
            [np.cos(azim), -np.sin(azim), 0.0],
            [np.sin(azim), np.cos(azim), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elev), -np.sin(elev)],
            [0.0, np.sin(elev), np.cos(elev)],
        ]
    )
    return rx @ rz


def _material_image(mesh: trimesh.Trimesh):
    material = getattr(getattr(mesh, "visual", None), "material", None)
    if material is None:
        return None
    for attr in ("baseColorTexture", "image"):
        image = getattr(material, attr, None)
        if image is not None:
            return image
    return None


def _material_color(mesh: trimesh.Trimesh) -> np.ndarray:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    color = getattr(material, "main_color", None) if material is not None else None
    if color is None:
        return np.array([0.68, 0.73, 0.76], dtype=np.float64)
    color = np.asarray(color, dtype=np.float64)
    if color.max(initial=0.0) > 1.0:
        color = color / 255.0
    return color[:3]


def _face_colors(mesh: trimesh.Trimesh, faces: np.ndarray) -> np.ndarray:
    visual = getattr(mesh, "visual", None)
    image = _material_image(mesh)
    uv = getattr(visual, "uv", None)
    if image is not None and uv is not None:
        vertex_colors = uv_to_interpolated_color(np.asarray(uv)[faces].reshape(-1, 2), image)
        colors = vertex_colors.reshape((-1, 3, 4)).astype(np.float64).mean(axis=1)[:, :3] / 255.0
        return colors * _material_color(mesh)[None, :]

    if visual is not None and hasattr(visual, "face_colors"):
        colors = np.asarray(visual.face_colors, dtype=np.float64)
        if len(colors) == len(faces):
            if colors.max(initial=0.0) > 1.0:
                colors = colors / 255.0
            return colors[:, :3]

    if visual is not None and hasattr(visual, "vertex_colors"):
        colors = np.asarray(visual.vertex_colors, dtype=np.float64)
        if len(colors) == len(mesh.vertices):
            if colors.max(initial=0.0) > 1.0:
                colors = colors / 255.0
            return colors[faces, :3].mean(axis=1)

    return np.tile(_material_color(mesh), (len(faces), 1))


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError(f"Empty mesh: {path}")
    face_colors = _face_colors(mesh, faces)
    vertices = vertices - (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    return vertices, faces, face_colors


def render_comparison(run_dir: Path, output_name: str, azim: float, elev: float) -> Path:
    variant_paths = complete_experiment_paths(run_dir)
    if variant_paths is None:
        raise RuntimeError(f"Missing one or more variants: {run_dir}")

    rotation = camera_rotation(azim, elev)
    light_dir = np.array([-0.35, -0.45, 0.82])
    light_dir /= np.linalg.norm(light_dir)

    meshes = []
    max_extent = 0.0
    for label, path in variant_paths:
        vertices, faces, face_colors = load_mesh(path)
        max_extent = max(max_extent, float(np.max(vertices.max(axis=0) - vertices.min(axis=0))))
        meshes.append((label, vertices, faces, face_colors))

    scale = 1.0 / max(max_extent, 1e-8)
    projected = []
    max_abs = np.array([0.0, 0.0])
    for label, vertices, faces, face_colors in meshes:
        vertices = (vertices * scale) @ rotation.T
        projected.append((label, vertices, faces, face_colors))
        max_abs = np.maximum(max_abs, np.max(np.abs(vertices[:, :2]), axis=0))

    lim_x = float(max_abs[0] * 1.10)
    lim_y = float(max_abs[1] * 1.16)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=220)
    fig.patch.set_facecolor("white")

    for ax, (label, vertices, faces, face_colors) in zip(axes, projected):
        tri = vertices[faces]
        polys = tri[:, :, :2]
        depths = tri[:, :, 2].mean(axis=1)
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        intensity = np.clip(0.42 + 0.58 * np.abs(normals @ light_dir), 0.25, 1.0)
        colors = np.clip(face_colors * intensity[:, None], 0, 1)
        order = np.argsort(depths)
        ax.add_collection(
            PolyCollection(
                polys[order],
                facecolors=colors[order],
                edgecolors=(0.45, 0.49, 0.52, 0.20),
                linewidths=0.15,
                antialiaseds=True,
            )
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-lim_x, lim_x)
        ax.set_ylim(-lim_y, lim_y)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(label, fontsize=13, pad=6)

    fig.suptitle(title_for(run_dir), fontsize=16, y=0.98)
    plt.subplots_adjust(left=0.015, right=0.985, top=0.79, bottom=0.02, wspace=0.03)
    output_path = run_dir / output_name
    fig.savefig(output_path, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", type=Path, help="Experiment run directories. Defaults to all complete experiments.")
    parser.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-name", default="variant_comparison.png")
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--elev", type=float, default=20.0)
    args = parser.parse_args()

    run_dirs = args.runs or discover_experiments(args.root)
    if not run_dirs:
        raise SystemExit("No complete experiment runs found.")

    for run_dir in run_dirs:
        variants = complete_experiment_paths(run_dir)
        if variants is None:
            print(f"skip missing variants: {run_dir}")
            continue
        try:
            output_path = render_comparison(run_dir, args.output_name, args.azim, args.elev)
        except Exception as exc:  # noqa: BLE001
            print(f"failed {run_dir}: {exc}")
            continue
        print(f"rendered {output_path}")


if __name__ == "__main__":
    main()
