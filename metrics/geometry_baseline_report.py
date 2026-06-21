#!/usr/bin/env python3
"""Compute geometry-control metrics across local/global user-study variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

from geometry_control_metrics import (  # noqa: E402
    _threshold_tag,
    _variant_spec,
    evaluate_variant,
)


VARIANT_LABELS = {
    "local": "Local tau 3/10",
    "global_tau3": "Global tau 3",
    "global_tau10": "Global tau 10",
}


def _read_status(variant_dir: Path) -> str:
    status_path = variant_dir / "status.txt"
    if not status_path.is_file():
        return "missing"
    return status_path.read_text(encoding="utf-8", errors="replace").strip() or "unknown"


def _variant_key(variant_dir: Path) -> str | None:
    spec = _variant_spec(variant_dir)
    if spec.mode == "local_tau":
        return "local"
    if spec.mode == "global_tau" and spec.low_tau is not None:
        if abs(float(spec.low_tau) - 3.0) < 1e-6:
            return "global_tau3"
        if abs(float(spec.low_tau) - 10.0) < 1e-6:
            return "global_tau10"
    return None


def _discover_experiments(root: Path) -> list[Path]:
    experiments = []
    for output_dir in sorted(root.rglob("output")):
        if not output_dir.is_dir():
            continue
        local_dirs = list(output_dir.glob("*local_tau*"))
        if local_dirs:
            experiments.append(output_dir.parent)
    return experiments


def _discover_variant_dirs(experiment_dir: Path) -> dict[str, Path]:
    output_dir = experiment_dir / "output"
    variants: dict[str, Path] = {}
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        key = _variant_key(child)
        if key and _read_status(child) == "succeeded" and (child / "struct_renders" / "mesh.ply").is_file():
            variants[key] = child
    return variants


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _stderr(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _group_summary(rows: list[dict[str, Any]], group_keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)

    out = []
    for key, group_rows in sorted(groups.items()):
        item = {group_key: key[index] for index, group_key in enumerate(group_keys)}
        item["count"] = len(group_rows)
        for metric in metrics:
            values = [value for row in group_rows if (value := _float_or_none(row.get(metric))) is not None]
            item[f"{metric}_mean"] = _mean(values)
            item[f"{metric}_median"] = _median(values)
            item[f"{metric}_stderr"] = _stderr(values)
        out.append(item)
    return out


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


def _paired_rows(rows: list[dict[str, Any]], variant_a: str, variant_b: str, control_level: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        by_key[(str(row["run_name"]), int(row["primitive_index"]), str(row["variant_key"]))] = row

    pairs = []
    for row in rows:
        if row["variant_key"] != variant_a or row["control_level"] != control_level:
            continue
        other = by_key.get((str(row["run_name"]), int(row["primitive_index"]), variant_b))
        if other is not None:
            pairs.append((row, other))
    return pairs


def _paired_delta_summary(
    rows: list[dict[str, Any]],
    *,
    variant_a: str,
    variant_b: str,
    control_level: str,
    metric: str,
    interpretation: str,
) -> dict[str, Any]:
    pairs = _paired_rows(rows, variant_a, variant_b, control_level)
    deltas = []
    pct = []
    for row_a, row_b in pairs:
        value_a = _float_or_none(row_a.get(metric))
        value_b = _float_or_none(row_b.get(metric))
        if value_a is None or value_b is None:
            continue
        deltas.append(value_b - value_a)
        if abs(value_b) > 1e-12:
            pct.append((value_b - value_a) / value_b * 100.0)
    return {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "control_level": control_level,
        "metric": metric,
        "interpretation": interpretation,
        "n": len(deltas),
        "mean_delta_b_minus_a": _mean(deltas),
        "median_delta_b_minus_a": _median(deltas),
        "mean_percent_reduction_vs_b": _mean(pct),
    }


def _overall_variant_control_summary(rows: list[dict[str, Any]], fscore_tags: list[str]) -> list[dict[str, Any]]:
    metrics = ["chamfer_mean", "sq_to_generated_p95"] + [f"fscore_{tag}" for tag in fscore_tags]
    return _group_summary(rows, ["variant_key", "control_level"], metrics)


def _lookup_summary(summary_rows: list[dict[str, Any]], variant_key: str, control_level: str, metric: str) -> float | None:
    for row in summary_rows:
        if row["variant_key"] == variant_key and row["control_level"] == control_level:
            return _float_or_none(row.get(f"{metric}_mean"))
    return None


def _ratio_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    local_low = _lookup_summary(summary_rows, "local", "low", "chamfer_mean")
    local_high = _lookup_summary(summary_rows, "local", "high", "chamfer_mean")
    return {
        "local_selectivity_low_over_high": None if not local_low or not local_high else float(local_low / local_high),
        "local_leakage_high_over_low": None if not local_low or not local_high else float(local_high / local_low),
        "local_low_chamfer_mean": local_low,
        "local_high_chamfer_mean": local_high,
    }


def _plot_chamfer_bars(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    variants = ["global_tau3", "local", "global_tau10"]
    levels = ["low", "high"]
    colors = {"low": "#4e79a7", "high": "#f28e2b"}
    x = np.arange(len(variants))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for offset_index, level in enumerate(levels):
        means = []
        errors = []
        for variant in variants:
            means.append(_lookup_summary(summary_rows, variant, level, "chamfer_mean") or np.nan)
            row = next((item for item in summary_rows if item["variant_key"] == variant and item["control_level"] == level), None)
            errors.append(np.nan if row is None else row.get("chamfer_mean_stderr") or 0.0)
        offset = (offset_index - 0.5) * width
        ax.bar(x + offset, means, width, yerr=errors, capsize=3, color=colors[level], alpha=0.84, label=level)

    ax.set_xticks(x, [VARIANT_LABELS[key] for key in variants])
    ax.set_ylabel("Mean part Chamfer distance")
    ax.set_title("Geometry Preservation By Control Region")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, title="Control level")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, facecolor="white")
    plt.close(fig)


def _plot_fscore_bars(summary_rows: list[dict[str, Any]], output_path: Path, tag: str) -> None:
    variants = ["global_tau3", "local", "global_tau10"]
    levels = ["low", "high"]
    colors = {"low": "#4e79a7", "high": "#f28e2b"}
    x = np.arange(len(variants))
    width = 0.34
    metric = f"fscore_{tag}"

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for offset_index, level in enumerate(levels):
        means = []
        errors = []
        for variant in variants:
            means.append(_lookup_summary(summary_rows, variant, level, metric) or np.nan)
            row = next((item for item in summary_rows if item["variant_key"] == variant and item["control_level"] == level), None)
            errors.append(np.nan if row is None else row.get(f"{metric}_stderr") or 0.0)
        offset = (offset_index - 0.5) * width
        ax.bar(x + offset, means, width, yerr=errors, capsize=3, color=colors[level], alpha=0.84, label=level)

    ax.set_xticks(x, [VARIANT_LABELS[key] for key in variants])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(f"F-score at distance {tag.replace('p', '.')}")
    ax.set_title("Near-Surface Preservation F-score")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, title="Control level")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, facecolor="white")
    plt.close(fig)


def _fmt(value: Any, digits: int = 4) -> str:
    value_f = _float_or_none(value)
    if value_f is None:
        return "n/a"
    return f"{value_f:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(
    path: Path,
    *,
    root: Path,
    summary_rows: list[dict[str, Any]],
    ratios: dict[str, Any],
    paired: dict[str, Any],
    counts: dict[str, Any],
    plot_paths: dict[str, str],
) -> None:
    variant_rows = []
    for variant in ["global_tau3", "local", "global_tau10"]:
        for level in ["low", "high"]:
            row = next((item for item in summary_rows if item["variant_key"] == variant and item["control_level"] == level), None)
            if row is None:
                continue
            variant_rows.append(
                [
                    VARIANT_LABELS[variant],
                    level,
                    str(row["count"]),
                    _fmt(row["chamfer_mean_mean"]),
                    _fmt(row["sq_to_generated_p95_mean"]),
                    _fmt(row.get("fscore_0p02_mean")),
                    _fmt(row.get("fscore_0p04_mean")),
                ]
            )

    paired_rows = [
        [
            "High-control preservation vs global tau 3",
            str(paired["high_improvement_vs_global_tau3"]["n"]),
            _fmt(paired["high_improvement_vs_global_tau3"]["mean_delta_b_minus_a"]),
            _fmt(paired["high_improvement_vs_global_tau3"]["median_delta_b_minus_a"]),
        ],
        [
            "Low-control freedom vs global tau 10",
            str(paired["low_freedom_vs_global_tau10"]["n"]),
            _fmt(paired["low_freedom_vs_global_tau10"]["mean_delta_b_minus_a"]),
            _fmt(paired["low_freedom_vs_global_tau10"]["median_delta_b_minus_a"]),
        ],
    ]

    text = f"""# Geometry Baseline Metrics

Source root: `{root}`

Evaluated `{counts["experiment_count"]}` experiments and `{counts["variant_count"]}` succeeded variants.

## Main Takeaways

- Local high-control parts have mean Chamfer `{_fmt(ratios["local_high_chamfer_mean"])}`.
- Local low-control parts have mean Chamfer `{_fmt(ratios["local_low_chamfer_mean"])}`.
- Selectivity ratio, low/high Chamfer: `{_fmt(ratios["local_selectivity_low_over_high"], 2)}x`.
- Leakage ratio, high/low Chamfer: `{_fmt(ratios["local_leakage_high_over_low"], 3)}`.

## Variant Summary

{_markdown_table(["Variant", "Region", "N", "Chamfer mean", "SQ-to-gen p95", "F@0.02", "F@0.04"], variant_rows)}

## Paired Baseline Comparisons

Positive `delta` means the desired effect is present: local has lower high-control error than global tau 3, or local has higher low-control error than global tau 10.

{_markdown_table(["Comparison", "N", "Mean delta", "Median delta"], paired_rows)}

## Plots

- Chamfer bars: `{plot_paths["chamfer_bars"]}`
- F-score bars: `{plot_paths["fscore_bars"]}`
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "spatial_control_user_study")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "metrics" / "outputs" / "geometry_baseline_report_user_study")
    parser.add_argument("--sq-resolution", type=int, default=72)
    parser.add_argument("--sq-samples-per-primitive", type=int, default=1600)
    parser.add_argument("--generated-samples", type=int, default=50000)
    parser.add_argument("--fscore-thresholds", default="0.02,0.04")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fscore_thresholds = tuple(float(value) for value in args.fscore_thresholds.split(",") if value.strip())
    fscore_tags = [_threshold_tag(value) for value in fscore_thresholds]

    experiments = _discover_experiments(args.root)
    all_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    variant_count = 0
    for experiment_index, experiment_dir in enumerate(experiments, start=1):
        variants = _discover_variant_dirs(experiment_dir)
        print(f"[{experiment_index}/{len(experiments)}] {experiment_dir.name}: {', '.join(sorted(variants))}", flush=True)
        if "local" not in variants:
            skipped.append({"experiment": str(experiment_dir), "reason": "missing succeeded local variant"})
            continue
        for key, variant_dir in variants.items():
            spec = _variant_spec(variant_dir)
            variant_count += 1
            rows = evaluate_variant(
                spec,
                sq_resolution=args.sq_resolution,
                sq_samples_per_primitive=args.sq_samples_per_primitive,
                generated_samples=args.generated_samples,
                seed=args.seed,
                fscore_thresholds=fscore_thresholds,
            )
            for row in rows:
                row["variant_key"] = key
                row["variant_label"] = VARIANT_LABELS[key]
                row["experiment_dir"] = str(experiment_dir)
            all_rows.extend(rows)

    summary_rows = _overall_variant_control_summary(all_rows, fscore_tags)
    ratios = _ratio_summary(summary_rows)
    paired = {
        "high_improvement_vs_global_tau3": _paired_delta_summary(
            all_rows,
            variant_a="local",
            variant_b="global_tau3",
            control_level="high",
            metric="chamfer_mean",
            interpretation="positive means local high-control Chamfer is lower than global tau 3",
        ),
        "low_freedom_vs_global_tau10": _paired_delta_summary(
            all_rows,
            variant_a="global_tau10",
            variant_b="local",
            control_level="low",
            metric="chamfer_mean",
            interpretation="positive means local low-control Chamfer is higher than global tau 10",
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_primitive_csv = args.output_dir / "per_primitive_all_variants.csv"
    summary_json = args.output_dir / "geometry_baseline_summary.json"
    report_md = args.output_dir / "geometry_baseline_report.md"
    chamfer_plot = args.output_dir / "chamfer_by_variant_region.png"
    fscore_plot = args.output_dir / "fscore_0p02_by_variant_region.png"
    _write_csv(per_primitive_csv, all_rows)
    _write_csv(args.output_dir / "variant_region_summary.csv", summary_rows)
    _plot_chamfer_bars(summary_rows, chamfer_plot)
    _plot_fscore_bars(summary_rows, fscore_plot, fscore_tags[0])

    counts = {
        "experiment_count": len(experiments),
        "variant_count": variant_count,
        "primitive_row_count": len(all_rows),
        "skipped": skipped,
    }
    payload = {
        "source_root": str(args.root),
        "counts": counts,
        "variant_region_summary": summary_rows,
        "ratios": ratios,
        "paired_comparisons": paired,
        "outputs": {
            "per_primitive_csv": str(per_primitive_csv),
            "variant_region_summary_csv": str(args.output_dir / "variant_region_summary.csv"),
            "report_md": str(report_md),
            "chamfer_bars": str(chamfer_plot),
            "fscore_bars": str(fscore_plot),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(
        report_md,
        root=args.root,
        summary_rows=summary_rows,
        ratios=ratios,
        paired=paired,
        counts=counts,
        plot_paths=payload["outputs"],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
