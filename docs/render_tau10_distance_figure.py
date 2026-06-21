#!/usr/bin/env python3
"""Render local-tau distance-to-tau10 representation figures."""

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
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
DOCS_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(DOCS_DIR) not in sys.path:
    sys.path.insert(0, str(DOCS_DIR))

from render_latent_control_figure import (  # noqa: E402
    OUTPUT_DIR,
    VARIANTS,
    _camera_project,
    _figure_stem,
    _low_control_mask,
    _mean_stderr,
    _panel_title,
    _render_mesh_panel,
    _status_for,
    _variant_data,
    render_utils,
)


DEFAULT_BATCH_ROOT = (
    REPO_ROOT
    / "spaceflow_runtime"
    / "sq_ui_runs"
    / "20260605T185622Z_examples_structure_texture150_experiment"
)
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "tau10_distance_examples"
DEFAULT_ONLY = "11,12,17,19,20"
SINGLE_VIEW_SPEC = {
    "name": "default",
    "azim": -35.0,
    "elev": 28.0,
}
FIVE_VIEW_SPECS = [
    {
        "name": f"view{i + 1:02d}",
        "azim": SINGLE_VIEW_SPEC["azim"] + offset,
        "elev": SINGLE_VIEW_SPEC["elev"],
    }
    for i, offset in enumerate([0.0, 72.0, 144.0, 216.0, 288.0])
]
TAU10_DISTANCE_LOW_COLOR = "#1a9850"
TAU10_DISTANCE_MID_COLOR = "#f4f4f4"
TAU10_DISTANCE_HIGH_COLOR = "#d73027"


def _run_output_name(run_dir: Path) -> str:
    return run_dir.name.removesuffix(" copy")


def _batch_run_dirs(batch_root: Path, only: set[str] | None = None) -> list[Path]:
    dirs = [
        path
        for path in sorted(batch_root.iterdir())
        if path.is_dir()
        and "_experiment" in path.name
        and (path / "output").is_dir()
    ]
    if only is None:
        return dirs
    return [
        path
        for path in dirs
        if path.name in only
        or _run_output_name(path) in only
        or _run_output_name(path).split("_", 1)[0] in only
    ]


def _required_variants_succeeded(run_dir: Path) -> tuple[bool, dict[str, str | None]]:
    statuses = {key: _status_for(run_dir, key) for key in ["local", "tau10"]}
    return all(status == "succeeded" for status in statuses.values()), statuses


def _tau10_distance(local_data: dict[str, object], tau10_data: dict[str, object]) -> np.ndarray:
    local_points = np.asarray(local_data["points"])
    local_features = np.asarray(local_data["features"])
    tau10_points = np.asarray(tau10_data["points"])
    tau10_features = np.asarray(tau10_data["features"])

    _, idx_tau10 = cKDTree(tau10_points).query(local_points, k=1)
    cosine_similarity = np.sum(local_features * tau10_features[idx_tau10], axis=1)
    return np.maximum(1.0 - cosine_similarity, 0.0)


def _distance_summary(distance: np.ndarray, low_mask: np.ndarray) -> dict[str, float | int | None]:
    high_mean, high_stderr, high_count = _mean_stderr(distance[~low_mask])
    low_mean, low_stderr, low_count = _mean_stderr(distance[low_mask])
    return {
        "high_mean": high_mean,
        "high_stderr": high_stderr,
        "high_count": high_count,
        "low_mean": low_mean,
        "low_stderr": low_stderr,
        "low_count": low_count,
    }


def _orange_threshold_stats(
    distance: np.ndarray,
    low_mask: np.ndarray,
    vmax: float,
) -> dict[str, float | int | str | None]:
    finite = np.isfinite(distance)
    high = np.clip(distance[finite & ~low_mask], 0.0, vmax)
    low = np.clip(distance[finite & low_mask], 0.0, vmax)
    fallback_values = np.clip(distance[finite], 0.0, vmax)

    if high.size == 0 or low.size == 0:
        fallback_threshold = float(np.median(fallback_values)) if fallback_values.size else min(0.2, vmax)
        return {
            "threshold": fallback_threshold,
            "method": "median_fallback",
            "balanced_accuracy": None,
            "high_count": int(high.size),
            "low_count": int(low.size),
            "high_mean": float(np.mean(high)) if high.size else None,
            "low_mean": float(np.mean(low)) if low.size else None,
        }

    high_sorted = np.sort(high)
    low_sorted = np.sort(low)
    unique_values = np.unique(np.concatenate([high_sorted, low_sorted]))
    if unique_values.size > 1:
        midpoints = (unique_values[:-1] + unique_values[1:]) * 0.5
        candidates = np.unique(np.concatenate([[0.0, vmax], unique_values, midpoints]))
    else:
        candidates = np.array([unique_values[0]], dtype=float)

    high_correct = np.searchsorted(high_sorted, candidates, side="left") / high_sorted.size
    low_correct = (low_sorted.size - np.searchsorted(low_sorted, candidates, side="left")) / low_sorted.size
    balanced_accuracy = (high_correct + low_correct) * 0.5
    best = np.flatnonzero(balanced_accuracy == np.max(balanced_accuracy))
    threshold = float(np.median(candidates[best]))

    return {
        "threshold": threshold,
        "method": "max_balanced_accuracy_high_below_low_above",
        "balanced_accuracy": float(balanced_accuracy[best[0]]),
        "high_count": int(high.size),
        "low_count": int(low.size),
        "high_mean": float(np.mean(high)),
        "low_mean": float(np.mean(low)),
    }


def _add_mesh_panel(ax: plt.Axes, image: np.ndarray, title: str, label: str) -> None:
    ax.imshow(image)
    ax.set_title(_panel_title(label, title), fontsize=9.3, pad=4, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_distance_map(
    ax: plt.Axes,
    points: np.ndarray,
    distance: np.ndarray,
    cbar_ax: plt.Axes,
    vmax: float,
    orange_threshold: float,
    azim: float,
    elev: float,
) -> None:
    projected = _camera_project(points, azim=azim, elev=elev)
    order = np.argsort(projected[:, 2])
    x = projected[order, 0]
    y = projected[order, 1]
    values = distance[order]
    display_threshold = float(np.clip(orange_threshold, 0.0, vmax))
    transition = float(np.clip(display_threshold / vmax if vmax > 0 else 0.5, 1e-6, 1.0 - 1e-6))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "tau10_distance",
        [
            (0.0, TAU10_DISTANCE_LOW_COLOR),
            (transition, TAU10_DISTANCE_MID_COLOR),
            (1.0, TAU10_DISTANCE_HIGH_COLOR),
        ],
        N=256,
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    ax.scatter(x, y, c=values, cmap=cmap, norm=norm, s=2.3, linewidths=0, alpha=0.94)
    pad_x = max(float(x.max() - x.min()) * 0.035, 1e-6)
    pad_y = max(float(y.max() - y.min()) * 0.035, 1e-6)
    ax.set_xlim(float(x.min() - pad_x), float(x.max() + pad_x))
    ax.set_ylim(float(y.min() - pad_y), float(y.max() + pad_y))
    ax.set_aspect("equal")
    ax.set_title(_panel_title("C", "Distance from tau 10"), fontsize=9.3, pad=4, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, cax=cbar_ax, orientation="horizontal", extend="max")
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(
        f"tau10-like        cosine distance        far from tau10  (transition t={display_threshold:.2f})",
        fontsize=7.5,
        labelpad=2,
    )


def _add_distance_bar(ax: plt.Axes, distance: np.ndarray, low_mask: np.ndarray, vmax: float) -> None:
    groups = [
        ("High", ~low_mask, TAU10_DISTANCE_LOW_COLOR),
        ("Low", low_mask, TAU10_DISTANCE_HIGH_COLOR),
    ]
    means = []
    errors = []
    colors = []
    labels = []
    for group_label, mask, color in groups:
        mean, stderr, _count = _mean_stderr(distance[mask])
        means.append(mean)
        errors.append(stderr)
        colors.append(color)
        labels.append(group_label)

    y = np.arange(len(labels))[::-1]
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

    valid_means = [mean for mean in means if mean is not None]
    x_max = max(vmax, max(valid_means) * 1.15 if valid_means else vmax)
    ax.set_xlim(0.0, x_max)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("mean distance to tau 10", fontsize=8.2)
    ax.set_title(_panel_title("D", "Region mean"), fontsize=9.3, pad=4, loc="left")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def render_figure(
    run_dir: Path,
    output_dir: Path,
    output_stem: str | None,
    distance_vmax: float,
    orange_threshold: float | None,
    view_specs: list[dict[str, float | str]] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if view_specs is None:
        view_specs = [SINGLE_VIEW_SPEC]
    data = {key: _variant_data(run_dir, key) for key in ["local", "tau10"]}
    local_points = np.asarray(data["local"]["points"])
    distance = _tau10_distance(data["local"], data["tau10"])
    low_mask = _low_control_mask(Path(data["local"]["dir"]), local_points)
    summary = _distance_summary(distance, low_mask)
    threshold_stats = _orange_threshold_stats(distance, low_mask, distance_vmax)
    threshold_source = "auto_run_balanced_accuracy"
    if orange_threshold is None:
        orange_threshold = float(threshold_stats["threshold"])
    else:
        orange_threshold = float(np.clip(orange_threshold, 0.0, distance_vmax))
        threshold_source = "fixed"

    meta = render_utils.run_meta(run_dir)
    manifest = render_utils.asset_manifest(meta)
    control_mesh = render_utils.load_sq_mesh(
        Path(data["local"]["dir"]) / "spatial_control_mesh.ply",
        manifest,
    )
    result_mesh = render_utils.load_mesh(Path(data["local"]["dir"]) / "out_sim.glb")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10.5,
        "axes.labelsize": 8.5,
        "savefig.dpi": 320,
    })
    stem = _figure_stem(run_dir, output_stem).replace("_latent_control_figure", "_tau10_distance_figure")
    rendered_views: list[dict[str, object]] = []
    for view_spec in view_specs:
        view_name = str(view_spec["name"])
        azim = float(view_spec["azim"])
        elev = float(view_spec["elev"])
        mesh_panels = [
            (
                "Control map\nhigh tau 10 / low tau 3",
                _render_mesh_panel(control_mesh, azim=azim, elev=elev),
            ),
            (
                "Local-tau result\ngenerated mesh",
                _render_mesh_panel(result_mesh, azim=azim, elev=elev),
            ),
        ]

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
            _add_mesh_panel(fig.add_subplot(gs[i, 0]), image, title, chr(ord("A") + i))

        map_gs = gs[:, 1].subgridspec(2, 1, height_ratios=[1.0, 0.07], hspace=0.08)
        _add_distance_map(
            fig.add_subplot(map_gs[0, 0]),
            local_points,
            distance,
            fig.add_subplot(map_gs[1, 0]),
            distance_vmax,
            orange_threshold,
            azim,
            elev,
        )
        bar_gs = gs[:, 2].subgridspec(3, 1, height_ratios=[0.24, 1.0, 0.24])
        _add_distance_bar(fig.add_subplot(bar_gs[1, 0]), distance, low_mask, distance_vmax)

        fig.text(
            0.5,
            0.035,
            "Cosine distance to nearest spatial match in global tau-10 PartField features; lower is more tau10-like.",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#30343b",
        )
        fig.subplots_adjust(left=0.045, right=0.985, top=0.91, bottom=0.15)

        suffix = "" if len(view_specs) == 1 and view_name == "default" else f"_{view_name}"
        png_path = output_dir / f"{stem}{suffix}.png"
        pdf_path = output_dir / f"{stem}{suffix}.pdf"
        fig.savefig(png_path, facecolor="white")
        fig.savefig(pdf_path, facecolor="white")
        plt.close(fig)
        rendered_views.append({
            "name": view_name,
            "png": str(png_path),
            "pdf": str(pdf_path),
            "azim": azim,
            "elev": elev,
        })

    primary_view = rendered_views[0]
    return {
        "run_dir": str(run_dir),
        "png": primary_view["png"],
        "pdf": primary_view["pdf"],
        "views": rendered_views,
        "view_count": len(rendered_views),
        "distance_vmax": distance_vmax,
        "orange_threshold": orange_threshold,
        "orange_threshold_source": threshold_source,
        "orange_threshold_stats": threshold_stats,
        "summary": summary,
    }


def _batch_orange_threshold_stats(run_dirs: list[Path], distance_vmax: float) -> dict[str, float | int | str | None]:
    distances = []
    masks = []
    for run_dir in run_dirs:
        data = {key: _variant_data(run_dir, key) for key in ["local", "tau10"]}
        local_points = np.asarray(data["local"]["points"])
        distance = _tau10_distance(data["local"], data["tau10"])
        low_mask = _low_control_mask(Path(data["local"]["dir"]), local_points)
        distances.append(distance)
        masks.append(low_mask)

    if not distances:
        return {
            "threshold": min(0.2, distance_vmax),
            "method": "empty_batch_fallback",
            "balanced_accuracy": None,
            "high_count": 0,
            "low_count": 0,
            "high_mean": None,
            "low_mean": None,
        }

    stats = _orange_threshold_stats(np.concatenate(distances), np.concatenate(masks), distance_vmax)
    stats["method"] = f"batch_{stats['method']}"
    return stats


def render_batch(
    batch_root: Path,
    output_dir: Path,
    only: set[str] | None,
    distance_vmax: float,
    orange_threshold: float | None,
    view_specs: list[dict[str, float | str]],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    eligible_run_dirs: list[Path] = []
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

    threshold_source = "fixed"
    if orange_threshold is None:
        batch_threshold_stats = _batch_orange_threshold_stats(eligible_run_dirs, distance_vmax)
        orange_threshold = float(batch_threshold_stats["threshold"])
        threshold_source = "auto_batch_balanced_accuracy"
        print(f"auto orange threshold: {orange_threshold:.4f}", flush=True)
    else:
        orange_threshold = float(np.clip(orange_threshold, 0.0, distance_vmax))
        batch_threshold_stats = {
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
            result = render_figure(
                run_dir,
                output_dir,
                f"{_run_output_name(run_dir)}_tau10_distance_figure",
                distance_vmax,
                orange_threshold,
                view_specs,
            )
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
        "distance_vmax": distance_vmax,
        "orange_threshold": orange_threshold,
        "orange_threshold_source": threshold_source,
        "orange_threshold_stats": batch_threshold_stats,
        "rendered_count": len(rendered),
        "rendered_view_count": sum(int(result.get("view_count", 1)) for result in rendered),
        "view_specs": view_specs,
        "skipped_count": len(skipped),
        "rendered": rendered,
        "skipped": skipped,
    }
    manifest_path = output_dir / "tau10_distance_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    average_summary = write_average_summary_and_plot(manifest, output_dir)
    manifest["average_summary_path"] = average_summary["summary_path"]
    manifest["average_plot_png"] = average_summary["plot_png"]
    manifest["average_plot_pdf"] = average_summary["plot_pdf"]
    return manifest


def _region_rows(rendered: list[dict[str, object]], region_key: str) -> list[tuple[float, int, str]]:
    rows = []
    for result in rendered:
        summary = result["summary"]
        assert isinstance(summary, dict)
        mean_value = summary.get(f"{region_key}_mean")
        count = summary.get(f"{region_key}_count")
        if mean_value is not None and count:
            rows.append((float(mean_value), int(count), str(result["run_dir"])))
    return rows


def _unweighted_average(rows: list[tuple[float, int, str]]) -> float | None:
    if not rows:
        return None
    return float(sum(mean_value for mean_value, _count, _run_dir in rows) / len(rows))


def _weighted_average(rows: list[tuple[float, int, str]]) -> float | None:
    total_count = sum(count for _mean_value, count, _run_dir in rows)
    if total_count == 0:
        return None
    return float(sum(mean_value * count for mean_value, count, _run_dir in rows) / total_count)


def _plot_average_metrics(summary: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    unweighted = summary["unweighted_mean_of_region_means"]
    weighted = summary["voxel_weighted_mean"]
    assert isinstance(unweighted, dict)
    assert isinstance(weighted, dict)

    categories = ["High control", "Low control"]
    unweighted_values = [
        float(unweighted["high"]) if unweighted["high"] is not None else np.nan,
        float(unweighted["low"]) if unweighted["low"] is not None else np.nan,
    ]
    weighted_values = [
        float(weighted["high"]) if weighted["high"] is not None else np.nan,
        float(weighted["low"]) if weighted["low"] is not None else np.nan,
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "savefig.dpi": 320,
    })
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = np.arange(len(categories))
    width = 0.32
    bars_a = ax.bar(x - width / 2, unweighted_values, width, color="#9dbfc8", label="Mean of examples")
    bars_b = ax.bar(x + width / 2, weighted_values, width, color="#de7c1b", label="Voxel-weighted")
    ax.set_xticks(x, categories)
    ax.set_ylabel("mean distance to tau 10")
    ax.set_title("Average tau10-distance by control region", fontsize=10)
    ax.set_ylim(0.0, max(0.65, np.nanmax(unweighted_values + weighted_values) * 1.18))
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for bars in [bars_a, bars_b]:
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    png_path = output_dir / "tau10_distance_average_metrics.png"
    pdf_path = output_dir / "tau10_distance_average_metrics.pdf"
    fig.savefig(png_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def write_average_summary_and_plot(manifest: dict[str, object], output_dir: Path) -> dict[str, object]:
    rendered = manifest["rendered"]
    assert isinstance(rendered, list)
    high_rows = _region_rows(rendered, "high")
    low_rows = _region_rows(rendered, "low")
    all_rows = high_rows + low_rows

    summary = {
        "source_manifest": str(output_dir / "tau10_distance_figure_manifest.json"),
        "batch_root": manifest["batch_root"],
        "orange_threshold": manifest.get("orange_threshold"),
        "orange_threshold_source": manifest.get("orange_threshold_source"),
        "orange_threshold_stats": manifest.get("orange_threshold_stats"),
        "rendered_count": manifest["rendered_count"],
        "rendered_view_count": manifest.get("rendered_view_count", manifest["rendered_count"]),
        "view_specs": manifest.get("view_specs"),
        "skipped_count": manifest["skipped_count"],
        "skipped": manifest["skipped"],
        "unweighted_mean_of_region_means": {
            "high": _unweighted_average(high_rows),
            "low": _unweighted_average(low_rows),
            "all_regions": _unweighted_average(all_rows),
        },
        "voxel_weighted_mean": {
            "high": _weighted_average(high_rows),
            "low": _weighted_average(low_rows),
            "all_regions": _weighted_average(all_rows),
        },
        "counts": {
            "high_examples_with_values": len(high_rows),
            "low_examples_with_values": len(low_rows),
            "high_voxels": sum(count for _mean_value, count, _run_dir in high_rows),
            "low_voxels": sum(count for _mean_value, count, _run_dir in low_rows),
        },
    }
    summary_path = output_dir / "tau10_distance_average_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_png, plot_pdf = _plot_average_metrics(summary, output_dir)
    return {
        "summary_path": str(summary_path),
        "plot_png": str(plot_png),
        "plot_pdf": str(plot_pdf),
        "summary": summary,
    }


def _parse_only(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def _view_specs(view_set: str) -> list[dict[str, float | str]]:
    if view_set == "single":
        return [SINGLE_VIEW_SPEC]
    if view_set == "five":
        return FIVE_VIEW_SPECS
    raise ValueError(f"Unknown view set: {view_set}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, help="Render one experiment directory instead of batch mode.")
    parser.add_argument("--batch-run-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", default=DEFAULT_ONLY, help="Comma-separated run names or numeric prefixes.")
    parser.add_argument("--distance-vmax", type=float, default=1.0, help="Colorbar max for cosine distance.")
    parser.add_argument(
        "--view-set",
        choices=["single", "five"],
        default="single",
        help="Render the default single view or five yaw-rotated views per asset.",
    )
    parser.add_argument(
        "--orange-threshold",
        type=float,
        default=None,
        help="Fixed distance value for the white/orange transition. Omit to estimate it from high/low regions.",
    )
    parser.add_argument("--output-stem")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    view_specs = _view_specs(args.view_set)
    if args.run_dir:
        result = render_figure(
            args.run_dir.resolve(),
            output_dir,
            args.output_stem,
            args.distance_vmax,
            args.orange_threshold,
            view_specs,
        )
        for view in result["views"]:
            assert isinstance(view, dict)
            print(f"wrote {view['png']}")
            print(f"wrote {view['pdf']}")
        return 0

    manifest = render_batch(
        args.batch_run_root.resolve(),
        output_dir,
        _parse_only(args.only),
        args.distance_vmax,
        args.orange_threshold,
        view_specs,
    )
    print(f"rendered {manifest['rendered_count']} assets")
    print(f"wrote {manifest['rendered_view_count']} figures")
    print(f"skipped {manifest['skipped_count']} runs")
    print(f"wrote {manifest['manifest_path']}")
    print(f"wrote {manifest['average_summary_path']}")
    print(f"wrote {manifest['average_plot_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
