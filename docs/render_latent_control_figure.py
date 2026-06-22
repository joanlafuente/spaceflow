#!/usr/bin/env python3
"""Render a paper-style latent/control comparison figure from completed variants."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sq_ui.scripts import render_spaceflow_experiment_comparison as render_utils  # noqa: E402


RUN_DIR = (
    REPO_ROOT
    / "spaceflow_runtime"
    / "sq_ui_runs"
    / "20260605T185622Z_examples_structure_texture150_experiment"
    / "12_pool_table_experiment"
)
OUTPUT_DIR = REPO_ROOT / "docs" / "figures"
BATCH_OUTPUT_DIR = OUTPUT_DIR / "latent_control_texture150_examples"
VARIANTS = {
    "local": {
        "dir": "01_local_tau3_tau10_polyak0p18",
        "label": "Local tau\nlow 3 / high 10",
        "color": "#6f4bb1",
    },
    "tau3": {
        "dir": "02_global_tau3_polyak0",
        "label": "Global tau\n3",
        "color": "#de7c1b",
    },
    "tau10": {
        "dir": "03_global_tau10_polyak0",
        "label": "Global tau\n10",
        "color": "#188a9e",
    },
}


def _read_ply_vertices(path: Path) -> np.ndarray:
    type_map = {
        "char": "i1",
        "uchar": "u1",
        "int8": "i1",
        "uint8": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int16": "<i2",
        "uint16": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "int32": "<i4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    with path.open("rb") as fh:
        header_lines: list[bytes] = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines).decode("ascii", errors="replace")
        vertex_count = None
        fmt = None
        vertex_props: list[tuple[str, str]] = []
        in_vertex = False
        for line in header.splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[:2] == ["format", "binary_little_endian"]:
                fmt = "binary_little_endian"
            elif parts[:2] == ["format", "ascii"]:
                fmt = "ascii"
            elif parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts[0] == "element":
                in_vertex = False
            elif in_vertex and parts[0] == "property" and parts[1] != "list":
                vertex_props.append((parts[-1], parts[1]))

        if vertex_count is None:
            raise ValueError(f"PLY missing vertex count: {path}")
        if [name for name, _kind in vertex_props[:3]] != ["x", "y", "z"]:
            raise ValueError(f"Expected first PLY vertex properties to be x/y/z: {path}")

        if fmt == "binary_little_endian":
            dtype = np.dtype([
                (f"p{i}_{name}", type_map[kind])
                for i, (name, kind) in enumerate(vertex_props)
            ])
            values = np.frombuffer(fh.read(vertex_count * dtype.itemsize), dtype=dtype, count=vertex_count)
            xyz = np.column_stack([
                values[dtype.names[0]],
                values[dtype.names[1]],
                values[dtype.names[2]],
            ])
            return xyz.astype(np.float64)
        if fmt == "ascii":
            values = []
            for _ in range(vertex_count):
                values.append([float(v) for v in fh.readline().split()[: len(vertex_props)]])
            return np.asarray(values, dtype=np.float64)[:, :3]
    raise ValueError(f"Unsupported PLY format {fmt!r}: {path}")


def _read_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    type_map = {
        "char": "i1",
        "uchar": "u1",
        "int8": "i1",
        "uint8": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int16": "<i2",
        "uint16": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "int32": "<i4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    with path.open("rb") as fh:
        header_lines: list[bytes] = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines).decode("ascii", errors="replace")
        vertex_count = None
        face_count = 0
        fmt = None
        vertex_props: list[tuple[str, str]] = []
        face_list_prop: tuple[str, str] | None = None
        section: str | None = None
        for line in header.splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[:2] == ["format", "binary_little_endian"]:
                fmt = "binary_little_endian"
            elif parts[:2] == ["format", "ascii"]:
                fmt = "ascii"
            elif parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                section = "vertex"
            elif parts[:2] == ["element", "face"]:
                face_count = int(parts[2])
                section = "face"
            elif parts[0] == "element":
                section = None
            elif section == "vertex" and parts[0] == "property" and parts[1] != "list":
                vertex_props.append((parts[-1], parts[1]))
            elif section == "face" and parts[:2] == ["property", "list"] and face_list_prop is None:
                face_list_prop = (parts[2], parts[3])

        if vertex_count is None:
            raise ValueError(f"PLY missing vertex count: {path}")
        if [name for name, _kind in vertex_props[:3]] != ["x", "y", "z"]:
            raise ValueError(f"Expected first PLY vertex properties to be x/y/z: {path}")

        if fmt == "binary_little_endian":
            dtype = np.dtype([
                (f"p{i}_{name}", type_map[kind])
                for i, (name, kind) in enumerate(vertex_props)
            ])
            values = np.frombuffer(fh.read(vertex_count * dtype.itemsize), dtype=dtype, count=vertex_count)
            vertices = np.column_stack([
                values[dtype.names[0]],
                values[dtype.names[1]],
                values[dtype.names[2]],
            ]).astype(np.float64)
            if face_count == 0:
                return vertices, np.empty((0, 3), dtype=np.int64)
            if face_list_prop is None:
                raise ValueError(f"PLY missing face vertex_indices list: {path}")
            count_dtype = np.dtype(type_map[face_list_prop[0]])
            index_dtype = np.dtype(type_map[face_list_prop[1]])
            faces: list[np.ndarray] = []
            for _ in range(face_count):
                count_raw = fh.read(count_dtype.itemsize)
                if len(count_raw) != count_dtype.itemsize:
                    raise ValueError(f"Truncated PLY face data: {path}")
                count = int(np.frombuffer(count_raw, dtype=count_dtype, count=1)[0])
                index_raw = fh.read(count * index_dtype.itemsize)
                if len(index_raw) != count * index_dtype.itemsize:
                    raise ValueError(f"Truncated PLY face indices: {path}")
                indices = np.frombuffer(index_raw, dtype=index_dtype, count=count).astype(np.int64)
                if count < 3:
                    continue
                for i in range(1, count - 1):
                    faces.append(np.array([indices[0], indices[i], indices[i + 1]], dtype=np.int64))
            return vertices, np.asarray(faces, dtype=np.int64).reshape(-1, 3)

        if fmt == "ascii":
            values = []
            for _ in range(vertex_count):
                values.append([float(v) for v in fh.readline().split()[: len(vertex_props)]])
            vertices = np.asarray(values, dtype=np.float64)[:, :3]
            faces = []
            for _ in range(face_count):
                parts = fh.readline().split()
                if not parts:
                    continue
                count = int(parts[0])
                indices = np.asarray([int(value) for value in parts[1 : 1 + count]], dtype=np.int64)
                if count < 3:
                    continue
                for i in range(1, count - 1):
                    faces.append(np.array([indices[0], indices[i], indices[i + 1]], dtype=np.int64))
            return vertices, np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    raise ValueError(f"Unsupported PLY format {fmt!r}: {path}")


def _control_surface_samples(local_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    mesh_path = local_dir / "spatial_control_mesh.ply"
    npz_path = local_dir / "input_superquadrics_all.npz"
    if not mesh_path.exists() or not npz_path.exists():
        return None

    with np.load(npz_path, allow_pickle=False) as data:
        if "control_levels" not in data:
            return None
        control_levels = np.asarray(data["control_levels"], dtype=np.float64)
    if control_levels.size == 0:
        return None

    vertices, faces = _read_ply_mesh(mesh_path)
    if vertices.size == 0 or faces.size == 0 or len(vertices) % len(control_levels) != 0:
        return None

    vertices_per_primitive = len(vertices) // len(control_levels)
    vertex_primitive = np.clip(
        np.arange(len(vertices), dtype=np.int64) // vertices_per_primitive,
        0,
        len(control_levels) - 1,
    )
    face_primitive = np.clip(
        faces.min(axis=1) // vertices_per_primitive,
        0,
        len(control_levels) - 1,
    )
    triangles = vertices[faces]
    face_centers = triangles.mean(axis=1)
    edge_centers = np.concatenate(
        [
            (triangles[:, 0] + triangles[:, 1]) * 0.5,
            (triangles[:, 1] + triangles[:, 2]) * 0.5,
            (triangles[:, 2] + triangles[:, 0]) * 0.5,
        ],
        axis=0,
    )
    samples = np.concatenate([vertices, face_centers, edge_centers], axis=0)
    sample_low = np.concatenate(
        [
            control_levels[vertex_primitive] < 0.5,
            control_levels[face_primitive] < 0.5,
            np.tile(control_levels[face_primitive] < 0.5, 3),
        ],
        axis=0,
    )
    finite = np.isfinite(samples).all(axis=1)
    if not np.any(finite):
        return None
    return samples[finite], sample_low[finite]


def _sample_triplane_features(part_planes: np.ndarray, voxel_points: np.ndarray) -> np.ndarray:
    coords = ((voxel_points + 0.5) * 64.0).astype(np.int64)
    voxel_coords = ((coords + 0.5) / 64.0) - 0.5
    bbmin = voxel_coords.min(axis=0)
    bbmax = voxel_coords.max(axis=0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * 0.9 / max(float((bbmax - bbmin).max()), 1e-8)
    normalized = (voxel_coords - center) * scale

    planes = torch.from_numpy(part_planes.astype(np.float32, copy=False))
    positions = torch.from_numpy(normalized.astype(np.float32))[None, :, :]

    def grid_sample(plane: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        grid = xy.unsqueeze(1)
        return torch.nn.functional.grid_sample(
            plane,
            grid,
            padding_mode="border",
            align_corners=True,
        )

    tri_plane = torch.unbind(planes, dim=1)
    xy = torch.cat([positions[:, :, 0:1], positions[:, :, 1:2]], dim=-1)
    yz = torch.cat([positions[:, :, 1:2], positions[:, :, 2:3]], dim=-1)
    xz = torch.cat([positions[:, :, 0:1], positions[:, :, 2:3]], dim=-1)
    with torch.no_grad():
        feats = grid_sample(tri_plane[0], xy) + grid_sample(tri_plane[1], yz) + grid_sample(tri_plane[2], xz)
    feats = feats.squeeze(2).permute(0, 2, 1).reshape(-1, planes.shape[2]).numpy()
    feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
    return feats.astype(np.float32, copy=False)


def _variant_data(run_dir: Path, variant_key: str) -> dict[str, object]:
    variant_dir = run_dir / "output" / VARIANTS[variant_key]["dir"]
    points = _read_ply_vertices(variant_dir / "voxels" / "struct_voxels.ply")
    part_planes = np.load(variant_dir / "partfield" / "part_feat_mesh_batch_part_plane.npy")
    features = _sample_triplane_features(part_planes, points)
    return {
        "dir": variant_dir,
        "points": points,
        "features": features,
    }


def _low_control_mask(local_dir: Path, points: np.ndarray) -> np.ndarray:
    surface_samples = _control_surface_samples(local_dir)
    if surface_samples is not None:
        samples, sample_low = surface_samples
        _distance, nearest = cKDTree(samples).query(points, k=1)
        return sample_low[nearest]

    # Legacy fallback for older runs that do not carry the original SQ control levels.
    low_mesh = _read_ply_vertices(local_dir / "low_control_superquadric_mask.ply")
    lo = low_mesh.min(axis=0)
    hi = low_mesh.max(axis=0)
    margin = 0.02
    return np.all((points >= lo - margin) & (points <= hi + margin), axis=1)


def _camera_project(points: np.ndarray, azim: float = -35.0, elev: float = 28.0) -> np.ndarray:
    rotation = render_utils.camera_rotation(azim, elev)
    projected = points @ rotation.T
    projected[:, 1] *= -1.0
    return projected


def _render_mesh_panel(mesh_info: dict[str, object], azim: float = 0.0, elev: float = 55.0) -> np.ndarray:
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
    return render_utils._rasterize_panel(projected_info, lim, lim, light_dir)


def _panel_title(label: str, title: str) -> str:
    return f"{label}  {title}"


def _add_mesh_panel(ax: plt.Axes, image: np.ndarray, title: str, label: str) -> None:
    ax.imshow(image)
    ax.set_title(_panel_title(label, title), fontsize=9.3, pad=4, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_representation_map(
    ax: plt.Axes,
    points: np.ndarray,
    preference: np.ndarray,
    low_mask: np.ndarray,
    cbar_ax: plt.Axes | None = None,
    label: str = "C",
) -> None:
    projected = _camera_project(points)
    order = np.argsort(projected[:, 2])
    x = projected[order, 0]
    y = projected[order, 1]
    pref = preference[order]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "tau_pref",
        ["#188a9e", "#f4f4f4", "#de7c1b"],
        N=256,
    )
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    ax.scatter(x, y, c=pref, cmap=cmap, norm=norm, s=2.3, linewidths=0, alpha=0.94)
    pad_x = max(float(x.max() - x.min()) * 0.035, 1e-6)
    pad_y = max(float(y.max() - y.min()) * 0.035, 1e-6)
    ax.set_xlim(float(x.min() - pad_x), float(x.max() + pad_x))
    ax.set_ylim(float(y.min() - pad_y), float(y.max() + pad_y))
    ax.set_aspect("equal")
    ax.set_title(_panel_title(label, "Voxel feature preference"), fontsize=9.3, pad=4, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("closer to tau 10        preference        closer to tau 3", fontsize=7.5, labelpad=2)


def _add_bar_panel(ax: plt.Axes, preference: np.ndarray, low_mask: np.ndarray, label: str = "D") -> None:
    groups = [
        ("High", ~low_mask, "#188a9e"),
        ("Low", low_mask, "#de7c1b"),
    ]
    means = []
    errors = []
    colors = []
    labels = []
    for group_label, mask, color in groups:
        values = preference[mask]
        mean, stderr, _count = _mean_stderr(values)
        means.append(mean)
        errors.append(stderr)
        colors.append(color)
        labels.append(group_label)
    y = np.arange(len(labels))[::-1]
    ax.axvline(0.0, color="#24272e", linewidth=0.8)
    valid = np.asarray([mean is not None for mean in means], dtype=bool)
    if np.any(valid):
        ax.barh(
            y[valid],
            [mean for mean in means if mean is not None],
            xerr=[error for error in errors if error is not None],
            color=[color for color, is_valid in zip(colors, valid) if is_valid],
            height=0.46,
            capsize=2,
        )
    for yi, mean in zip(y, means):
        if mean is None:
            ax.text(0.03, yi, "n/a", ha="left", va="center", fontsize=7.5, color="#555555")
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("mean preference", fontsize=8.2)
    ax.set_title(_panel_title(label, "Region mean"), fontsize=9.3, pad=4, loc="left")
    ax.set_xlim(-0.75, 0.55)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _status_for(run_dir: Path, variant_key: str) -> str | None:
    path = run_dir / "output" / VARIANTS[variant_key]["dir"] / "status.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _required_variants_succeeded(run_dir: Path) -> tuple[bool, dict[str, str | None]]:
    statuses = {key: _status_for(run_dir, key) for key in VARIANTS}
    return all(status == "succeeded" for status in statuses.values()), statuses


def _figure_stem(run_dir: Path, output_stem: str | None) -> str:
    if output_stem:
        return output_stem
    if run_dir == RUN_DIR:
        return "pool_table_latent_control_figure"
    return f"{run_dir.name}_latent_control_figure"


def _mean_stderr(values: np.ndarray) -> tuple[float | None, float | None, int]:
    if len(values) == 0:
        return None, None, 0
    return (
        float(np.mean(values)),
        float(np.std(values) / math.sqrt(len(values))),
        int(len(values)),
    )


def _region_summary(preference: np.ndarray, low_mask: np.ndarray) -> dict[str, float | int | None]:
    high_values = preference[~low_mask]
    low_values = preference[low_mask]
    high_mean, high_stderr, high_count = _mean_stderr(high_values)
    low_mean, low_stderr, low_count = _mean_stderr(low_values)
    return {
        "high_mean": high_mean,
        "high_stderr": high_stderr,
        "high_count": high_count,
        "low_mean": low_mean,
        "low_stderr": low_stderr,
        "low_count": low_count,
    }


def render_figure(run_dir: Path, output_dir: Path, output_stem: str | None = None) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {key: _variant_data(run_dir, key) for key in VARIANTS}
    local_points = np.asarray(data["local"]["points"])
    local_features = np.asarray(data["local"]["features"])

    tree_tau3 = cKDTree(np.asarray(data["tau3"]["points"]))
    tree_tau10 = cKDTree(np.asarray(data["tau10"]["points"]))
    _, idx_tau3 = tree_tau3.query(local_points, k=1)
    _, idx_tau10 = tree_tau10.query(local_points, k=1)
    dist_tau3 = 1.0 - np.sum(local_features * np.asarray(data["tau3"]["features"])[idx_tau3], axis=1)
    dist_tau10 = 1.0 - np.sum(local_features * np.asarray(data["tau10"]["features"])[idx_tau10], axis=1)
    preference = (dist_tau10 - dist_tau3) / np.maximum(dist_tau10 + dist_tau3, 1e-8)
    preference = np.clip(preference, -1.0, 1.0)
    low_mask = _low_control_mask(Path(data["local"]["dir"]), local_points)
    summary = _region_summary(preference, low_mask)

    meta = render_utils.run_meta(run_dir)
    manifest = render_utils.asset_manifest(meta)
    control_mesh = render_utils.load_sq_mesh(
        Path(data["local"]["dir"]) / "spatial_control_mesh.ply",
        manifest,
    )
    mesh_panels = [
        ("Control map\nhigh tau 10 / low tau 3", _render_mesh_panel(control_mesh)),
        ("Local-tau result\ngenerated mesh", _render_mesh_panel(render_utils.load_mesh(Path(data["local"]["dir"]) / "out_sim.glb"))),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10.5,
        "axes.labelsize": 8.5,
        "savefig.dpi": 320,
    })
    fig = plt.figure(figsize=(9.2, 4.8), constrained_layout=False)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.05, 1.65, 0.92],
        height_ratios=[1.0, 1.0],
        wspace=0.30,
        hspace=0.18,
    )
    for i, (title, image) in enumerate(mesh_panels):
        ax = fig.add_subplot(gs[i, 0])
        _add_mesh_panel(ax, image, title, chr(ord("A") + i))

    map_gs = gs[:, 1].subgridspec(2, 1, height_ratios=[1.0, 0.07], hspace=0.08)
    ax_map = fig.add_subplot(map_gs[0, 0])
    cbar_ax = fig.add_subplot(map_gs[1, 0])
    _add_representation_map(ax_map, local_points, preference, low_mask, cbar_ax, "C")
    bar_gs = gs[:, 2].subgridspec(3, 1, height_ratios=[0.24, 1.0, 0.24])
    ax_bar = fig.add_subplot(bar_gs[1, 0])
    _add_bar_panel(ax_bar, preference, low_mask, "D")
    fig.text(
        0.5,
        0.035,
        "Preference is computed per generated voxel in PartField feature space: "
        "orange/positive is closer to the global tau 3 baseline, teal/negative is closer to global tau 10.",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#30343b",
    )
    fig.subplots_adjust(left=0.045, right=0.985, top=0.91, bottom=0.15)
    stem = _figure_stem(run_dir, output_stem)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return {
        "run_dir": str(run_dir),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "summary": summary,
    }


def _batch_run_dirs(batch_root: Path, only: set[str] | None = None) -> list[Path]:
    dirs = [
        path
        for path in sorted(batch_root.iterdir())
        if path.is_dir()
        and path.name.endswith("_experiment")
        and (path / "output").is_dir()
    ]
    if only is None:
        return dirs
    return [
        path
        for path in dirs
        if path.name in only or path.name.split("_", 1)[0] in only
    ]


def render_batch(batch_root: Path, output_dir: Path, only: set[str] | None = None) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for run_dir in _batch_run_dirs(batch_root, only):
        ok, statuses = _required_variants_succeeded(run_dir)
        if not ok:
            skipped.append({
                "run_dir": str(run_dir),
                "reason": "not all required variants succeeded",
                "statuses": statuses,
            })
            print(f"skip {run_dir.name}: {statuses}", flush=True)
            continue
        try:
            result = render_figure(run_dir, output_dir, f"{run_dir.name}_latent_control_figure")
        except Exception as exc:  # noqa: BLE001
            skipped.append({
                "run_dir": str(run_dir),
                "reason": f"render failed: {exc}",
                "statuses": statuses,
            })
            print(f"failed {run_dir.name}: {exc}", flush=True)
            continue
        rendered.append(result)
        print(f"rendered {run_dir.name}", flush=True)

    manifest = {
        "batch_root": str(batch_root),
        "output_dir": str(output_dir),
        "rendered_count": len(rendered),
        "skipped_count": len(skipped),
        "rendered": rendered,
        "skipped": skipped,
    }
    manifest_path = output_dir / "latent_control_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--batch-run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-stem")
    parser.add_argument("--only", help="Comma-separated run names or numeric prefixes for batch mode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_run_root:
        only = None
        if args.only:
            only = {part.strip() for part in args.only.split(",") if part.strip()}
        output_dir = (args.output_dir or BATCH_OUTPUT_DIR).resolve()
        manifest = render_batch(args.batch_run_root.resolve(), output_dir, only)
        print(f"rendered {manifest['rendered_count']} figures")
        print(f"skipped {manifest['skipped_count']} runs")
        print(f"wrote {manifest['manifest_path']}")
        return 0

    output_dir = (args.output_dir or OUTPUT_DIR).resolve()
    result = render_figure(args.run_dir.resolve(), output_dir, args.output_stem)
    print(f"wrote {result['png']}")
    print(f"wrote {result['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
