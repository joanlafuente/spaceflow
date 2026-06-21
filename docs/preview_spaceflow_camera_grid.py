#!/usr/bin/env python3
"""Render a fast camera-angle contact sheet for one SpaceFlow mesh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


SCRIPT_PATH = Path(__file__).resolve()
DOCS_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(DOCS_DIR) not in sys.path:
    sys.path.insert(0, str(DOCS_DIR))

from export_structure_experiment_figs import _render_view_variants, render_utils  # noqa: E402


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "spaceflow_runtime"
    / "sq_ui_runs"
    / "20260608T234518Z_A_trophy_texture_experiment"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "figures"
    / "full_pipeline_examples"
    / "20260608T234518Z_A_trophy_texture_experiment"
    / "camera_preview_grid.png"
)


def _parse_floats(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated number")
    return values


def _parse_rotation(value: str) -> tuple[float, float, float]:
    values = _parse_floats(value)
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated degrees: x,y,z")
    return values[0], values[1], values[2]


def _rotation_matrix_xyz(rotation_deg: tuple[float, float, float]) -> np.ndarray:
    rx_deg, ry_deg, rz_deg = rotation_deg
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rx), -np.sin(rx)],
            [0.0, np.sin(rx), np.cos(rx)],
        ],
        dtype=np.float64,
    )
    rot_y = np.array(
        [
            [np.cos(ry), 0.0, np.sin(ry)],
            [0.0, 1.0, 0.0],
            [-np.sin(ry), 0.0, np.cos(ry)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0.0],
            [np.sin(rz), np.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def _rotate_mesh(mesh: dict[str, object], rotation_deg: tuple[float, float, float]) -> dict[str, object]:
    if all(abs(value) < 1e-8 for value in rotation_deg):
        return mesh
    rotation = _rotation_matrix_xyz(rotation_deg)
    rotated = dict(mesh)
    rotated["vertices"] = np.asarray(mesh["vertices"], dtype=np.float64) @ rotation.T
    rotated["vertex_normals"] = np.asarray(mesh["vertex_normals"], dtype=np.float64) @ rotation.T
    return rotated


def _crop_alpha(image: Image.Image, pad: int = 8) -> Image.Image:
    bbox = image.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    return image.crop((
        max(left - pad, 0),
        max(top - pad, 0),
        min(right + pad, image.width),
        min(bottom + pad, image.height),
    ))


def _mesh_path(run_dir: Path, variant_dir: str, mesh_file: str, mesh: Path | None) -> Path:
    if mesh is not None:
        return mesh if mesh.is_absolute() else (REPO_ROOT / mesh)
    return run_dir / "output" / variant_dir / mesh_file


def _tile(rendered: np.ndarray, label: str, thumb_size: int, flip_y: bool, image_rotate: int) -> Image.Image:
    rgba = Image.fromarray(np.clip(rendered * 255.0, 0, 255).astype(np.uint8), mode="RGBA")
    rgba = _crop_alpha(rgba)
    if flip_y:
        rgba = ImageOps.flip(rgba)
    if image_rotate:
        rgba = rgba.rotate(image_rotate, expand=True)
    rgba.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)

    label_h = 34
    margin = 14
    tile = Image.new("RGBA", (thumb_size + margin * 2, thumb_size + label_h + margin * 2), (255, 255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.text((margin, 8), label, fill=(0, 0, 0, 255))
    tile.alpha_composite(rgba, (margin + (thumb_size - rgba.width) // 2, label_h + margin + (thumb_size - rgba.height) // 2))
    return tile.convert("RGB")


def render_grid(
    mesh_path: Path,
    output_path: Path,
    azims: list[float],
    elevs: list[float],
    render_size: int,
    thumb_size: int,
    mesh_rotation: tuple[float, float, float],
    flip_y: bool,
    image_rotate: int,
) -> dict[str, object]:
    render_utils.PANEL_RENDER_SIZE = render_size
    mesh = _rotate_mesh(render_utils.load_mesh(mesh_path), mesh_rotation)

    tiles: list[Image.Image] = []
    views: list[dict[str, float | str | int]] = []
    for row, elev in enumerate(elevs):
        for col, azim in enumerate(azims):
            index = row * len(azims) + col + 1
            name = f"v{index:02d}"
            rendered = _render_view_variants({"mesh": mesh}, azim, elev)["mesh"]
            tiles.append(_tile(rendered, f"{name}: az={azim:g}, elev={elev:g}", thumb_size, flip_y, image_rotate))
            views.append({"index": index, "name": name, "azim": azim, "elev": elev})

    if not tiles:
        raise RuntimeError("no views rendered")

    tile_w, tile_h = tiles[0].size
    sheet = Image.new("RGB", (tile_w * len(azims), tile_h * len(elevs)), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % len(azims)) * tile_w, (index // len(azims)) * tile_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)

    manifest = {
        "mesh": str(mesh_path),
        "output": str(output_path),
        "render_size": render_size,
        "thumb_size": thumb_size,
        "mesh_rotation_xyz": list(mesh_rotation),
        "flip_y": flip_y,
        "image_rotate": image_rotate,
        "azims": azims,
        "elevs": elevs,
        "views": views,
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--variant-dir", default="01_spaceflow_local_texture_routing")
    parser.add_argument("--mesh-file", default="out_sim.glb")
    parser.add_argument("--mesh", type=Path, help="Explicit mesh path. Overrides --run-dir/--variant-dir/--mesh-file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--azims", type=_parse_floats, default=_parse_floats("0,90,180,270,320"))
    parser.add_argument("--elevs", type=_parse_floats, default=_parse_floats("60,75,90,105"))
    parser.add_argument("--mesh-rotation", type=_parse_rotation, default=(0.0, 0.0, 0.0), help="Pre-rotate mesh in degrees as x,y,z.")
    parser.add_argument("--flip-y", action="store_true", help="Flip each preview tile vertically after rendering.")
    parser.add_argument("--image-rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Rotate each preview tile after rendering.")
    parser.add_argument("--render-size", type=int, default=360)
    parser.add_argument("--thumb-size", type=int, default=260)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    mesh_path = _mesh_path(run_dir, args.variant_dir, args.mesh_file, args.mesh).resolve()
    output_path = args.output.resolve()
    if not mesh_path.is_file():
        raise SystemExit(f"Missing mesh: {mesh_path}")

    manifest = render_grid(
        mesh_path,
        output_path,
        args.azims,
        args.elevs,
        args.render_size,
        args.thumb_size,
        args.mesh_rotation,
        args.flip_y,
        args.image_rotate,
    )
    print(f"wrote {manifest['output']}")
    print(f"wrote {manifest['manifest']}")
    print(f"rendered {len(manifest['views'])} views")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
