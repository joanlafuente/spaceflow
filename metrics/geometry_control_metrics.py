#!/usr/bin/env python3
"""Measure part-wise geometry preservation for local SpaceFlow control.

The main metric is control monotonicity:

    Spearman(control_tau, -part_chamfer)

where higher tau means stronger control and lower Chamfer means better
preservation of the input superquadric scaffold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]


@dataclass(frozen=True)
class VariantSpec:
    variant_dir: Path
    variant_name: str
    run_name: str
    mode: str
    low_tau: float | None
    high_tau: float | None
    polyak_tau: float | None


def _load_trimesh():
    try:
        import trimesh  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "geometry_control_metrics.py requires trimesh. Run it with the "
            "project environment, e.g. ../guideflow3d/envs/guideflow3d/bin/python."
        ) from exc
    return trimesh


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _num_from_tag(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


def _infer_variant_metadata(variant_dir: Path) -> dict[str, Any]:
    name = variant_dir.name
    meta: dict[str, Any] = {
        "name": name,
        "mode": "",
        "low_tau": None,
        "high_tau": None,
        "polyak_tau": None,
    }

    local_match = re.search(
        r"local_tau(?P<low>[0-9mp.]+)_tau(?P<high>[0-9mp.]+)_polyak(?P<polyak>[0-9mp.]+)",
        name,
    )
    if local_match:
        meta.update(
            {
                "mode": "local_tau",
                "low_tau": _num_from_tag(local_match.group("low")),
                "high_tau": _num_from_tag(local_match.group("high")),
                "polyak_tau": _num_from_tag(local_match.group("polyak")),
            }
        )
        return meta

    global_match = re.search(r"global_tau(?P<tau>[0-9mp.]+)_polyak(?P<polyak>[0-9mp.]+)", name)
    if global_match:
        meta.update(
            {
                "mode": "global_tau",
                "low_tau": _num_from_tag(global_match.group("tau")),
                "high_tau": None,
                "polyak_tau": _num_from_tag(global_match.group("polyak")),
            }
        )
    return meta


def _variant_from_manifest(variant_dir: Path) -> dict[str, Any]:
    manifest = _read_json(variant_dir.parent / "experiment_manifest.json")
    variants = manifest.get("variants")
    if not isinstance(variants, list):
        return {}

    resolved = variant_dir.resolve()
    for item in variants:
        if not isinstance(item, dict):
            continue
        output_dir = item.get("output_dir")
        if output_dir and Path(str(output_dir)).resolve() == resolved:
            return item
        if str(item.get("name") or "") == variant_dir.name:
            return item
    return {}


def _variant_spec(variant_dir: Path) -> VariantSpec:
    inferred = _infer_variant_metadata(variant_dir)
    manifest = _variant_from_manifest(variant_dir)
    merged = {**inferred, **{key: value for key, value in manifest.items() if value is not None}}
    run_name = variant_dir.parent.parent.name if variant_dir.parent.name == "output" else variant_dir.parent.name
    return VariantSpec(
        variant_dir=variant_dir,
        variant_name=str(merged.get("name") or variant_dir.name),
        run_name=run_name,
        mode=str(merged.get("mode") or inferred.get("mode") or ""),
        low_tau=None if merged.get("low_tau") is None else float(merged["low_tau"]),
        high_tau=None if merged.get("high_tau") is None else float(merged["high_tau"]),
        polyak_tau=None if merged.get("polyak_tau") is None else float(merged["polyak_tau"]),
    )


def _discover_variant_dirs(batch_root: Path, include_global: bool) -> list[Path]:
    candidates: list[Path] = []
    for mesh_path in batch_root.rglob("struct_renders/mesh.ply"):
        variant_dir = mesh_path.parents[1]
        if not (variant_dir / "input_superquadrics_all.npz").is_file():
            continue
        spec = _variant_spec(variant_dir)
        if spec.mode == "local_tau" or (include_global and spec.mode == "global_tau"):
            candidates.append(variant_dir)
    return sorted(set(candidates))


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


def _superquadric_mesh(
    scale: np.ndarray,
    shape: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    tapering: np.ndarray,
    bending: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
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

    vertices = np.column_stack([x, y, z])
    vertices = (rotation @ vertices.T).T + translation

    faces: list[list[int]] = []
    n = resolution
    for i in range(n - 1):
        for j in range(n - 1):
            faces.append([i * n + j, i * n + j + 1, (i + 1) * n + j])
            faces.append([(i + 1) * n + j, i * n + j + 1, (i + 1) * n + (j + 1)])
    for i in range(n - 1):
        faces.append([i * n + (n - 1), i * n, (i + 1) * n + (n - 1)])
        faces.append([(i + 1) * n + (n - 1), i * n, (i + 1) * n])
    faces.append([(n - 1) * n + (n - 1), (n - 1) * n, n - 1])
    faces.append([n - 1, (n - 1) * n, 0])
    return vertices, np.asarray(faces, dtype=np.int64)


def _load_sq_meshes(npz_path: Path, resolution: int):
    trimesh = _load_trimesh()
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}

    required = ["scales", "shapes", "translations", "rotations", "control_levels"]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"{npz_path} missing required arrays: {missing}")

    count = int(arrays["scales"].shape[0])
    tapering = arrays.get("tapering", np.zeros((count, 2), dtype=np.float64))
    bending = arrays.get("bending", np.zeros((count, 6), dtype=np.float64))

    raw: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(count):
        raw.append(
            _superquadric_mesh(
                arrays["scales"][index],
                arrays["shapes"][index],
                arrays["translations"][index],
                arrays["rotations"][index],
                tapering[index],
                bending[index],
                resolution,
            )
        )

    all_vertices = np.concatenate([vertices for vertices, _faces in raw], axis=0)
    bb_min = all_vertices.min(axis=0)
    bb_max = all_vertices.max(axis=0)
    center = (bb_min + bb_max) * 0.5
    scale = 1.0 / max(float((bb_max - bb_min).max()), 1e-8)

    meshes = []
    for vertices, faces in raw:
        normalized = (vertices - center) * scale
        meshes.append(trimesh.Trimesh(vertices=normalized, faces=faces, process=False))

    control_levels = np.asarray(arrays["control_levels"], dtype=np.float64)
    return meshes, control_levels


def _as_mesh(obj: Any):
    trimesh = _load_trimesh()
    if isinstance(obj, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(obj.dump()))
    else:
        mesh = obj
    return mesh


def _load_mesh_points(mesh_path: Path, sample_count: int, seed: int) -> np.ndarray:
    trimesh = _load_trimesh()
    np.random.seed(seed)
    loaded = trimesh.load(mesh_path, process=False)
    mesh = _as_mesh(loaded)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(getattr(mesh, "faces", []), dtype=np.int64)
    if vertices.size == 0:
        raise ValueError(f"No vertices in generated mesh: {mesh_path}")
    if faces.size == 0:
        indices = np.random.choice(len(vertices), size=sample_count, replace=len(vertices) < sample_count)
        return vertices[indices]
    return np.asarray(mesh.sample(sample_count), dtype=np.float64)


def _sample_mesh(mesh: Any, count: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        indices = np.random.choice(len(vertices), size=count, replace=len(vertices) < count)
        return vertices[indices]
    return np.asarray(mesh.sample(count), dtype=np.float64)


def _generated_mesh_path(variant_dir: Path) -> Path:
    candidates = [
        variant_dir / "struct_renders" / "mesh.ply",
        variant_dir / "voxels" / "struct_voxels.ply",
        variant_dir / "spatial_control_mesh.ply",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No generated structure mesh/points found in {variant_dir}")


def _assigned_tau(spec: VariantSpec, control_level: str) -> float | None:
    if spec.mode == "global_tau":
        return spec.low_tau
    if control_level == "high":
        return spec.high_tau if spec.high_tau is not None else spec.low_tau
    return spec.low_tau


def evaluate_variant(
    spec: VariantSpec,
    *,
    sq_resolution: int,
    sq_samples_per_primitive: int,
    generated_samples: int,
    seed: int,
    fscore_thresholds: tuple[float, ...] = (0.02, 0.04),
) -> list[dict[str, Any]]:
    npz_path = spec.variant_dir / "input_superquadrics_all.npz"
    meshes, control_levels = _load_sq_meshes(npz_path, sq_resolution)
    generated_points = _load_mesh_points(_generated_mesh_path(spec.variant_dir), generated_samples, seed)
    generated_tree = cKDTree(generated_points)

    primitive_samples = [
        _sample_mesh(mesh, sq_samples_per_primitive, seed + primitive_index + 1)
        for primitive_index, mesh in enumerate(meshes)
    ]
    all_sq_points = np.concatenate(primitive_samples, axis=0)
    primitive_ids = np.concatenate(
        [
            np.full(len(points), primitive_index, dtype=np.int64)
            for primitive_index, points in enumerate(primitive_samples)
        ]
    )
    _, nearest_sq = cKDTree(all_sq_points).query(generated_points, k=1)
    generated_part_ids = primitive_ids[nearest_sq]

    rows: list[dict[str, Any]] = []
    for primitive_index, sq_points in enumerate(primitive_samples):
        control_level = "low" if float(control_levels[primitive_index]) < 0.5 else "high"
        control_tau = _assigned_tau(spec, control_level)
        sq_to_generated = generated_tree.query(sq_points, k=1)[0]
        generated_part = generated_points[generated_part_ids == primitive_index]
        if len(generated_part) > 0:
            generated_to_sq = cKDTree(sq_points).query(generated_part, k=1)[0]
            generated_to_sq_mean = float(np.mean(generated_to_sq))
            chamfer_mean = float((np.mean(sq_to_generated) + generated_to_sq_mean) * 0.5)
            assigned_generated_samples = int(len(generated_part))
        else:
            generated_to_sq_mean = math.nan
            chamfer_mean = math.nan
            assigned_generated_samples = 0

        row = {
            "run_name": spec.run_name,
            "variant_name": spec.variant_name,
            "variant_dir": str(spec.variant_dir),
            "mode": spec.mode,
            "primitive_index": primitive_index,
            "control_level": control_level,
            "control_tau": control_tau,
            "low_tau": spec.low_tau,
            "high_tau": spec.high_tau,
            "polyak_tau": spec.polyak_tau,
            "sq_to_generated_mean": float(np.mean(sq_to_generated)),
            "sq_to_generated_median": float(np.median(sq_to_generated)),
            "sq_to_generated_p95": float(np.percentile(sq_to_generated, 95)),
            "generated_to_sq_mean": generated_to_sq_mean,
            "chamfer_mean": chamfer_mean,
            "assigned_generated_samples": assigned_generated_samples,
            "sq_samples": int(len(sq_points)),
            "generated_samples": int(len(generated_points)),
        }
        for threshold in fscore_thresholds:
            tag = _threshold_tag(threshold)
            recall = float(np.mean(sq_to_generated <= threshold))
            if len(generated_part) > 0:
                precision = float(np.mean(generated_to_sq <= threshold))
                denom = precision + recall
                fscore = float(2.0 * precision * recall / denom) if denom > 0.0 else 0.0
            else:
                precision = math.nan
                fscore = math.nan
            row[f"recall_{tag}"] = recall
            row[f"precision_{tag}"] = precision
            row[f"fscore_{tag}"] = fscore
        rows.append(row)
    return rows


def _threshold_tag(threshold: float) -> str:
    return f"{threshold:g}".replace("-", "m").replace(".", "p")


def _finite_metric_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        tau = row.get("control_tau")
        value = row.get(metric)
        if tau is None or value is None:
            continue
        tau_f = float(tau)
        value_f = float(value)
        if np.isfinite(tau_f) and np.isfinite(value_f):
            out.append(row)
    return out


def _summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    finite = _finite_metric_rows(rows, metric)
    if len(finite) >= 2:
        tau = np.asarray([float(row["control_tau"]) for row in finite], dtype=np.float64)
        preservation = -np.asarray([float(row[metric]) for row in finite], dtype=np.float64)
        rho, p_value = spearmanr(tau, preservation)
    else:
        rho, p_value = math.nan, math.nan

    by_tau: dict[str, dict[str, Any]] = {}
    for row in finite:
        key = f"{float(row['control_tau']):g}"
        by_tau.setdefault(key, {"count": 0, "values": []})
        by_tau[key]["count"] += 1
        by_tau[key]["values"].append(float(row[metric]))
    for key, item in by_tau.items():
        values = np.asarray(item.pop("values"), dtype=np.float64)
        item["mean"] = float(np.mean(values))
        item["median"] = float(np.median(values))
        item["stderr"] = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0

    by_level: dict[str, dict[str, Any]] = {}
    for row in finite:
        key = str(row["control_level"])
        by_level.setdefault(key, {"count": 0, "values": []})
        by_level[key]["count"] += 1
        by_level[key]["values"].append(float(row[metric]))
    for key, item in by_level.items():
        values = np.asarray(item.pop("values"), dtype=np.float64)
        item["mean"] = float(np.mean(values))
        item["median"] = float(np.median(values))
        item["stderr"] = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0

    return {
        "metric": metric,
        "spearman_control_tau_vs_negative_error": {
            "rho": None if not np.isfinite(rho) else float(rho),
            "p_value": None if not np.isfinite(p_value) else float(p_value),
            "n": len(finite),
        },
        "by_tau": by_tau,
        "by_control_level": by_level,
        "variant_count": len({row["variant_dir"] for row in rows}),
        "primitive_row_count": len(rows),
        "finite_row_count": len(finite),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _exclude_outliers(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    top_k: int,
    above: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    excluded_indices: set[int] = set()
    finite = []
    for index, row in enumerate(rows):
        value = row.get(metric)
        if value is None:
            continue
        value_f = float(value)
        if not np.isfinite(value_f):
            continue
        finite.append((index, value_f))
        if above is not None and value_f > above:
            excluded_indices.add(index)

    if top_k > 0:
        for index, _value in sorted(finite, key=lambda item: item[1], reverse=True)[:top_k]:
            excluded_indices.add(index)

    kept = [row for index, row in enumerate(rows) if index not in excluded_indices]
    excluded = [row for index, row in enumerate(rows) if index in excluded_indices]
    return kept, excluded


def _excluded_summary(excluded: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return {
        "count": len(excluded),
        "rows": [
            {
                "run_name": row.get("run_name"),
                "variant_name": row.get("variant_name"),
                "primitive_index": row.get("primitive_index"),
                "control_level": row.get("control_level"),
                "control_tau": row.get("control_tau"),
                metric: row.get(metric),
            }
            for row in sorted(excluded, key=lambda item: float(item.get(metric, 0.0)), reverse=True)
        ],
    }


def _plot(rows: list[dict[str, Any]], summary: dict[str, Any], metric: str, output_path: Path) -> None:
    finite = _finite_metric_rows(rows, metric)
    if not finite:
        return

    rng = np.random.default_rng(7)
    tau = np.asarray([float(row["control_tau"]) for row in finite], dtype=np.float64)
    error = np.asarray([float(row[metric]) for row in finite], dtype=np.float64)
    levels = np.asarray([str(row["control_level"]) for row in finite])
    jitter_width = max((tau.max() - tau.min()) * 0.012, 0.015)
    x = tau + rng.normal(0.0, jitter_width, size=len(tau))

    colors = {"low": "#4e79a7", "high": "#f28e2b"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for level in ["low", "high"]:
        mask = levels == level
        if np.any(mask):
            ax.scatter(
                x[mask],
                error[mask],
                s=24,
                alpha=0.62,
                linewidths=0.25,
                edgecolors="white",
                color=colors[level],
                label=f"{level} control",
            )

    tau_values = sorted(set(float(value) for value in tau))
    means = []
    stderrs = []
    for value in tau_values:
        vals = error[tau == value]
        means.append(float(np.mean(vals)))
        stderrs.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    ax.errorbar(
        tau_values,
        means,
        yerr=stderrs,
        color="#222222",
        marker="o",
        markersize=4,
        linewidth=1.2,
        capsize=3,
        label="mean +/- stderr",
    )

    stat = summary["spearman_control_tau_vs_negative_error"]
    rho = stat["rho"]
    p_value = stat["p_value"]
    rho_text = "n/a" if rho is None else f"{rho:.3f}"
    p_text = "n/a" if p_value is None else f"{p_value:.2g}"
    ax.text(
        0.98,
        0.96,
        f"Spearman(tau, -Chamfer) = {rho_text}\np = {p_text}, n = {stat['n']}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.94},
    )
    ax.set_xlabel("Assigned control tau")
    ax.set_ylabel("Part Chamfer distance (lower is better)")
    ax.set_title("Control Monotonicity")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=[0.0, 0.07, 1.0, 1.0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-dir", type=Path, action="append", default=[], help="Variant output directory.")
    parser.add_argument("--batch-root", type=Path, action="append", default=[], help="Recursively scan for variants.")
    parser.add_argument("--include-global", action="store_true", help="Include global_tau variants when scanning.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "metrics" / "outputs" / "control_monotonicity")
    parser.add_argument("--metric", default="chamfer_mean", choices=["chamfer_mean", "sq_to_generated_mean"])
    parser.add_argument("--sq-resolution", type=int, default=72)
    parser.add_argument("--sq-samples-per-primitive", type=int, default=1600)
    parser.add_argument("--generated-samples", type=int, default=50000)
    parser.add_argument("--fscore-thresholds", default="0.02,0.04", help="Comma-separated F-score distance thresholds.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--exclude-top-k", type=int, default=0, help="Exclude the top-k rows by metric before writing outputs.")
    parser.add_argument("--exclude-above", type=float, default=None, help="Exclude rows with metric above this threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant_dirs = [path.resolve() for path in args.variant_dir]
    for batch_root in args.batch_root:
        variant_dirs.extend(_discover_variant_dirs(batch_root.resolve(), args.include_global))
    variant_dirs = sorted(set(variant_dirs))
    if not variant_dirs:
        raise SystemExit("No variant directories found. Pass --variant-dir or --batch-root.")

    rows: list[dict[str, Any]] = []
    fscore_thresholds = tuple(float(value) for value in args.fscore_thresholds.split(",") if value.strip())
    for index, variant_dir in enumerate(variant_dirs, start=1):
        spec = _variant_spec(variant_dir)
        print(f"[{index}/{len(variant_dirs)}] {spec.run_name}/{spec.variant_name}", flush=True)
        rows.extend(
            evaluate_variant(
                spec,
                sq_resolution=args.sq_resolution,
                sq_samples_per_primitive=args.sq_samples_per_primitive,
                generated_samples=args.generated_samples,
                seed=args.seed,
                fscore_thresholds=fscore_thresholds,
            )
        )

    rows, excluded = _exclude_outliers(
        rows,
        args.metric,
        top_k=max(0, int(args.exclude_top_k)),
        above=args.exclude_above,
    )
    summary = _summary(rows, args.metric)
    if excluded:
        summary["excluded_outliers"] = _excluded_summary(excluded, args.metric)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_primitive_geometry_control.csv"
    excluded_csv_path = args.output_dir / "excluded_outliers.csv"
    summary_path = args.output_dir / "control_monotonicity_summary.json"
    plot_path = args.output_dir / "control_monotonicity.png"
    _write_csv(csv_path, rows)
    if excluded:
        _write_csv(excluded_csv_path, excluded)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(rows, summary, args.metric, plot_path)

    print(json.dumps({"csv": str(csv_path), "summary": str(summary_path), "plot": str(plot_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
