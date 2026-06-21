#!/usr/bin/env python3
"""Measure PartField voxel distance to a fixed global tau baseline.

This computes the same clipped cosine distance used by the tau10-distance
figures, but lets multiple source variants be compared to one fixed target:

    max(1 - dot(source_partfield_feature, target_partfield_feature), 0)

For each source voxel, the target feature is taken from the nearest target
voxel in spatial coordinates. Features are the L2-normalized PartField
triplane features sampled by docs/render_latent_control_figure.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DOCS_DIR = REPO_ROOT / "docs"
if str(DOCS_DIR) not in sys.path:
    sys.path.insert(0, str(DOCS_DIR))

from render_latent_control_figure import (  # noqa: E402
    VARIANTS,
    _low_control_mask,
    _status_for,
    _variant_data,
)


DEFAULT_BATCH_ROOT = REPO_ROOT / "spatial_control_evaluations"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "metrics" / "outputs" / "partfield_tau10_distance_spatial_control_evaluations"
DEFAULT_SOURCES = ("local", "tau3")
DEFAULT_TARGET = "tau10"
DEFAULT_SPLIT_UNIFORM_SOURCES = True
SOURCE_REGION_TAU = {
    "local": {"high": 10.0, "low": 3.0},
    "tau3": {"high": 3.0, "low": 3.0},
    "tau10": {"high": 10.0, "low": 10.0},
}


def _run_output_name(run_dir: Path) -> str:
    return run_dir.name.removesuffix(" copy")


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


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


def _mean_stderr(values: np.ndarray) -> tuple[float | None, float | None, int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None, 0
    stderr = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return float(np.mean(values)), stderr, int(values.size)


def _distance_to_target(source_data: dict[str, object], target_data: dict[str, object]) -> np.ndarray:
    source_points = np.asarray(source_data["points"])
    source_features = np.asarray(source_data["features"])
    target_points = np.asarray(target_data["points"])
    target_features = np.asarray(target_data["features"])

    _, target_indices = cKDTree(target_points).query(source_points, k=1)
    cosine_similarity = np.sum(source_features * target_features[target_indices], axis=1)
    return np.maximum(1.0 - cosine_similarity, 0.0)


def _distance_summary(distance: np.ndarray, low_mask: np.ndarray) -> dict[str, float | int | None]:
    all_mean, all_stderr, all_count = _mean_stderr(distance)
    high_mean, high_stderr, high_count = _mean_stderr(distance[~low_mask])
    low_mean, low_stderr, low_count = _mean_stderr(distance[low_mask])
    return {
        "all_mean": all_mean,
        "all_stderr": all_stderr,
        "all_count": all_count,
        "all_cosine_similarity_mean": None if all_mean is None else 1.0 - all_mean,
        "high_mean": high_mean,
        "high_stderr": high_stderr,
        "high_count": high_count,
        "high_cosine_similarity_mean": None if high_mean is None else 1.0 - high_mean,
        "low_mean": low_mean,
        "low_stderr": low_stderr,
        "low_count": low_count,
        "low_cosine_similarity_mean": None if low_mean is None else 1.0 - low_mean,
    }


def _distance_summary_all(distance: np.ndarray) -> dict[str, float | int | None]:
    all_mean, all_stderr, all_count = _mean_stderr(distance)
    return {
        "all_mean": all_mean,
        "all_stderr": all_stderr,
        "all_count": all_count,
        "all_cosine_similarity_mean": None if all_mean is None else 1.0 - all_mean,
        "high_mean": None,
        "high_stderr": None,
        "high_count": None,
        "high_cosine_similarity_mean": None,
        "low_mean": None,
        "low_stderr": None,
        "low_count": None,
        "low_cosine_similarity_mean": None,
    }


def _is_uniform_source(source: str) -> bool:
    tau = SOURCE_REGION_TAU.get(source)
    if not tau:
        return False
    high = tau.get("high")
    low = tau.get("low")
    return high is not None and low is not None and abs(float(high) - float(low)) < 1e-8


def _threshold_high_below_low_above(
    distance: np.ndarray,
    low_mask: np.ndarray,
    *,
    distance_vmax: float,
) -> dict[str, float | int | str | None]:
    """Descriptive high-vs-low separation, oriented as high closer to tau10."""

    finite = np.isfinite(distance)
    high = np.clip(distance[finite & ~low_mask], 0.0, distance_vmax)
    low = np.clip(distance[finite & low_mask], 0.0, distance_vmax)
    fallback_values = np.clip(distance[finite], 0.0, distance_vmax)

    payload: dict[str, float | int | str | None] = {
        "orientation": "high_below_low_above",
        "high_count": int(high.size),
        "low_count": int(low.size),
        "high_mean": float(np.mean(high)) if high.size else None,
        "low_mean": float(np.mean(low)) if low.size else None,
    }
    if high.size == 0 or low.size == 0:
        fallback_threshold = float(np.median(fallback_values)) if fallback_values.size else min(0.2, distance_vmax)
        payload.update(
            {
                "threshold": fallback_threshold,
                "method": "median_fallback",
                "balanced_accuracy": None,
            }
        )
        return payload

    high_sorted = np.sort(high)
    low_sorted = np.sort(low)
    unique_values = np.unique(np.concatenate([high_sorted, low_sorted]))
    if unique_values.size > 1:
        midpoints = (unique_values[:-1] + unique_values[1:]) * 0.5
        candidates = np.unique(np.concatenate([[0.0, distance_vmax], unique_values, midpoints]))
    else:
        candidates = np.array([unique_values[0]], dtype=float)

    high_correct = np.searchsorted(high_sorted, candidates, side="left") / high_sorted.size
    low_correct = (low_sorted.size - np.searchsorted(low_sorted, candidates, side="left")) / low_sorted.size
    balanced_accuracy = (high_correct + low_correct) * 0.5
    best = np.flatnonzero(balanced_accuracy == np.max(balanced_accuracy))
    threshold = float(np.median(candidates[best]))

    payload.update(
        {
            "threshold": threshold,
            "method": "max_balanced_accuracy_high_below_low_above",
            "balanced_accuracy": float(balanced_accuracy[best[0]]),
        }
    )
    return payload


def _unweighted_average(rows: list[tuple[float, int, str]]) -> float | None:
    if not rows:
        return None
    return float(sum(mean_value for mean_value, _count, _run_dir in rows) / len(rows))


def _weighted_average(rows: list[tuple[float, int, str]]) -> float | None:
    total_count = sum(count for _mean_value, count, _run_dir in rows)
    if total_count == 0:
        return None
    return float(sum(mean_value * count for mean_value, count, _run_dir in rows) / total_count)


def _region_rows(per_run: list[dict[str, Any]], source: str, region: str) -> list[tuple[float, int, str]]:
    rows = []
    for row in per_run:
        if row["source"] != source:
            continue
        mean_value = row.get(f"{region}_mean")
        count = row.get(f"{region}_count")
        if mean_value is not None and count:
            rows.append((float(mean_value), int(count), str(row["run_dir"])))
    return rows


def _all_rows(per_run: list[dict[str, Any]], source: str) -> list[tuple[float, int, str]]:
    rows = []
    for row in per_run:
        if row["source"] != source:
            continue
        mean_value = row.get("all_mean")
        count = row.get("all_count")
        if mean_value is not None and count:
            rows.append((float(mean_value), int(count), str(row["run_dir"])))
    return rows


def _source_summary(
    *,
    source: str,
    target: str,
    per_run: list[dict[str, Any]],
    distances: list[np.ndarray],
    masks: list[np.ndarray],
    skipped: list[dict[str, Any]],
    distance_vmax: float,
    split_uniform_sources: bool,
) -> dict[str, Any]:
    split_regions = split_uniform_sources or not _is_uniform_source(source)
    high_rows = _region_rows(per_run, source, "high")
    low_rows = _region_rows(per_run, source, "low")
    all_rows = _all_rows(per_run, source)

    if not split_regions:
        separation = None
    elif distances:
        separation = _threshold_high_below_low_above(
            np.concatenate(distances),
            np.concatenate(masks),
            distance_vmax=distance_vmax,
        )
    else:
        separation = _threshold_high_below_low_above(
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=bool),
            distance_vmax=distance_vmax,
        )

    unweighted = {
        "high": _unweighted_average(high_rows),
        "low": _unweighted_average(low_rows),
        "all_regions": _unweighted_average(all_rows),
    }
    weighted = {
        "high": _weighted_average(high_rows),
        "low": _weighted_average(low_rows),
        "all_regions": _weighted_average(all_rows),
    }
    return {
        "source": source,
        "source_dir": VARIANTS[source]["dir"],
        "target": target,
        "target_dir": VARIANTS[target]["dir"],
        "source_region_tau": SOURCE_REGION_TAU.get(source),
        "region_split_enabled": split_regions,
        "metric": "clipped_cosine_distance=max(1-cosine_similarity,0)",
        "region_separation_high_below_low_above": separation,
        "measured_count": len({row["run_dir"] for row in per_run if row["source"] == source}),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "unweighted_mean_of_region_means": unweighted,
        "unweighted_mean_cosine_similarity": {
            key: None if value is None else 1.0 - value
            for key, value in unweighted.items()
        },
        "voxel_weighted_mean": weighted,
        "voxel_weighted_cosine_similarity": {
            key: None if value is None else 1.0 - value
            for key, value in weighted.items()
        },
        "counts": {
            "high_examples_with_values": len(high_rows),
            "low_examples_with_values": len(low_rows),
            "all_examples_with_values": len(all_rows),
            "high_voxels": sum(count for _mean_value, count, _run_dir in high_rows),
            "low_voxels": sum(count for _mean_value, count, _run_dir in low_rows),
            "all_voxels": sum(count for _mean_value, count, _run_dir in all_rows),
        },
    }


def _paired_source_delta(
    per_run: list[dict[str, Any]],
    *,
    source_a: str,
    source_b: str,
    region: str,
) -> dict[str, Any]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in per_run:
        by_key[(str(row["run_name"]), str(row["source"]))] = row

    deltas = []
    cosine_deltas = []
    for row_a in per_run:
        if row_a["source"] != source_a:
            continue
        row_b = by_key.get((str(row_a["run_name"]), source_b))
        if row_b is None:
            continue
        value_a = row_a.get(f"{region}_mean")
        value_b = row_b.get(f"{region}_mean")
        if value_a is None or value_b is None:
            continue
        value_a_f = float(value_a)
        value_b_f = float(value_b)
        if not np.isfinite(value_a_f) or not np.isfinite(value_b_f):
            continue
        deltas.append(value_a_f - value_b_f)
        cosine_deltas.append((1.0 - value_a_f) - (1.0 - value_b_f))
    return {
        "source_a": source_a,
        "source_b": source_b,
        "region": region,
        "metric": "distance",
        "interpretation": "delta = source_a distance - source_b distance; negative means source_a is closer to target",
        "n": len(deltas),
        "mean_delta_a_minus_b": None if not deltas else float(np.mean(deltas)),
        "median_delta_a_minus_b": None if not deltas else float(np.median(deltas)),
        "mean_cosine_similarity_delta_a_minus_b": None if not cosine_deltas else float(np.mean(cosine_deltas)),
    }


def _paired_source_delta_all(
    per_run: list[dict[str, Any]],
    *,
    source_a: str,
    source_b: str,
    region_a: str,
) -> dict[str, Any]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in per_run:
        by_key[(str(row["run_name"]), str(row["source"]))] = row

    deltas = []
    cosine_deltas = []
    for row_a in per_run:
        if row_a["source"] != source_a:
            continue
        row_b = by_key.get((str(row_a["run_name"]), source_b))
        if row_b is None:
            continue
        value_a = row_a.get(f"{region_a}_mean")
        value_b = row_b.get("all_mean")
        if value_a is None or value_b is None:
            continue
        value_a_f = float(value_a)
        value_b_f = float(value_b)
        if not np.isfinite(value_a_f) or not np.isfinite(value_b_f):
            continue
        deltas.append(value_a_f - value_b_f)
        cosine_deltas.append((1.0 - value_a_f) - (1.0 - value_b_f))
    return {
        "source_a": source_a,
        "source_b": source_b,
        "region_a": region_a,
        "region_b": "all",
        "metric": "distance",
        "interpretation": "delta = source_a region distance - source_b all distance; negative means source_a region is closer to target",
        "n": len(deltas),
        "mean_delta_a_minus_b": None if not deltas else float(np.mean(deltas)),
        "median_delta_a_minus_b": None if not deltas else float(np.median(deltas)),
        "mean_cosine_similarity_delta_a_minus_b": None if not cosine_deltas else float(np.mean(cosine_deltas)),
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


def evaluate_batch(
    batch_root: Path,
    output_dir: Path,
    *,
    sources: list[str],
    target: str,
    only: set[str] | None,
    distance_vmax: float,
    split_uniform_sources: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run: list[dict[str, Any]] = []
    per_source_distances: dict[str, list[np.ndarray]] = {source: [] for source in sources}
    per_source_masks: dict[str, list[np.ndarray]] = {source: [] for source in sources}
    skipped: dict[str, list[dict[str, Any]]] = {source: [] for source in sources}
    run_dirs = _batch_run_dirs(batch_root, only)

    for run_dir in run_dirs:
        statuses = {key: _status_for(run_dir, key) for key in [*sources, target]}
        if statuses[target] != "succeeded":
            for source in sources:
                skipped[source].append(
                    {
                        "run_dir": str(run_dir),
                        "reason": f"{target} target variant is required",
                        "statuses": statuses,
                    }
                )
            print(f"skip {run_dir.name}: {target}={statuses[target]}", flush=True)
            continue

        try:
            target_data = _variant_data(run_dir, target)
        except Exception as exc:  # noqa: BLE001
            for source in sources:
                skipped[source].append(
                    {
                        "run_dir": str(run_dir),
                        "reason": f"target data failed: {exc}",
                        "statuses": statuses,
                    }
                )
            print(f"failed target data {run_dir.name}: {exc}", flush=True)
            continue

        for source in sources:
            if statuses[source] != "succeeded":
                skipped[source].append(
                    {
                        "run_dir": str(run_dir),
                        "reason": f"{source} source variant is required",
                        "statuses": statuses,
                    }
                )
                print(f"skip {run_dir.name}/{source}: {statuses[source]}", flush=True)
                continue
            try:
                source_data = _variant_data(run_dir, source)
                source_points = np.asarray(source_data["points"])
                distance = _distance_to_target(source_data, target_data)
                split_regions = split_uniform_sources or not _is_uniform_source(source)
                if not split_regions:
                    low_mask = None
                    summary = _distance_summary_all(distance)
                    separation = None
                else:
                    low_mask = _low_control_mask(Path(source_data["dir"]), source_points)
                    summary = _distance_summary(distance, low_mask)
                    separation = _threshold_high_below_low_above(
                        distance,
                        low_mask,
                        distance_vmax=distance_vmax,
                    )
            except Exception as exc:  # noqa: BLE001
                skipped[source].append(
                    {
                        "run_dir": str(run_dir),
                        "reason": f"metric failed: {exc}",
                        "statuses": statuses,
                    }
                )
                print(f"failed {run_dir.name}/{source}: {exc}", flush=True)
                continue

            per_source_distances[source].append(distance)
            if low_mask is not None:
                per_source_masks[source].append(low_mask)
            row = {
                "run_name": _run_output_name(run_dir),
                "run_dir": str(run_dir),
                "source": source,
                "source_dir": VARIANTS[source]["dir"],
                "target": target,
                "target_dir": VARIANTS[target]["dir"],
                "source_high_tau": (SOURCE_REGION_TAU.get(source) or {}).get("high"),
                "source_low_tau": (SOURCE_REGION_TAU.get(source) or {}).get("low"),
                "region_split_enabled": split_regions,
                "all_mean": summary["all_mean"],
                "all_stderr": summary["all_stderr"],
                "all_count": summary["all_count"],
                "all_cosine_similarity_mean": summary["all_cosine_similarity_mean"],
                "high_mean": summary["high_mean"],
                "high_stderr": summary["high_stderr"],
                "high_count": summary["high_count"],
                "high_cosine_similarity_mean": summary["high_cosine_similarity_mean"],
                "low_mean": summary["low_mean"],
                "low_stderr": summary["low_stderr"],
                "low_count": summary["low_count"],
                "low_cosine_similarity_mean": summary["low_cosine_similarity_mean"],
                "high_below_low_threshold": None if separation is None else separation["threshold"],
                "high_below_low_balanced_accuracy": None if separation is None else separation["balanced_accuracy"],
            }
            per_run.append(row)
            print(
                f"measured {run_dir.name}/{source}-> {target}: "
                f"all={summary['all_mean']} high={summary['high_mean']} low={summary['low_mean']}",
                flush=True,
            )

    source_summaries = {
        source: _source_summary(
            source=source,
            target=target,
            per_run=per_run,
            distances=per_source_distances[source],
            masks=per_source_masks[source],
            skipped=skipped[source],
            distance_vmax=distance_vmax,
            split_uniform_sources=split_uniform_sources,
        )
        for source in sources
    }

    paired: dict[str, Any] = {}
    if "local" in sources and "tau3" in sources:
        if split_uniform_sources:
            paired = {
                "local_vs_tau3_high_region": _paired_source_delta(
                    per_run,
                    source_a="local",
                    source_b="tau3",
                    region="high",
                ),
                "local_vs_tau3_low_region": _paired_source_delta(
                    per_run,
                    source_a="local",
                    source_b="tau3",
                    region="low",
                ),
                "local_all_vs_tau3_all": _paired_source_delta(
                    per_run,
                    source_a="local",
                    source_b="tau3",
                    region="all",
                ),
            }
        else:
            paired = {
                "local_high_vs_tau3_all": _paired_source_delta_all(
                    per_run,
                    source_a="local",
                    source_b="tau3",
                    region_a="high",
                ),
                "local_low_vs_tau3_all": _paired_source_delta_all(
                    per_run,
                    source_a="local",
                    source_b="tau3",
                    region_a="low",
                ),
                "local_all_vs_tau3_all": _paired_source_delta(
                    per_run,
                    source_a="local",
                    source_b="tau3",
                    region="all",
                ),
            }

    per_run_csv = output_dir / "per_run_partfield_tau10_distance.csv"
    summary_path = output_dir / "partfield_tau10_distance_summary.json"
    _write_csv(per_run_csv, per_run)
    payload = {
        "batch_root": str(batch_root),
        "output_dir": str(output_dir),
        "sources": sources,
        "target": target,
        "split_uniform_sources": split_uniform_sources,
        "distance_vmax": distance_vmax,
        "run_count": len(run_dirs),
        "per_run_csv": str(per_run_csv),
        "source_summaries": source_summaries,
        "paired_comparisons": paired,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["summary_path"] = str(summary_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-run-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help="Comma-separated source variants, e.g. local,tau3.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Fixed target variant, usually tau10.")
    parser.add_argument("--only", default="", help="Comma-separated run names or numeric prefixes.")
    parser.add_argument("--distance-vmax", type=float, default=1.0)
    parser.add_argument(
        "--no-split-uniform-sources",
        action="store_true",
        help="For uniform sources such as global tau3, report only all-voxel metrics instead of high/low regions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = _parse_csv(args.sources)
    unknown_sources = [source for source in sources if source not in VARIANTS]
    if unknown_sources:
        raise SystemExit(f"Unknown sources: {unknown_sources}. Known: {sorted(VARIANTS)}")
    if args.target not in VARIANTS:
        raise SystemExit(f"Unknown target: {args.target}. Known: {sorted(VARIANTS)}")
    if args.target in sources:
        raise SystemExit("--target should not also be listed in --sources")

    payload = evaluate_batch(
        args.batch_run_root.resolve(),
        args.output_dir.resolve(),
        sources=sources,
        target=args.target,
        only=set(_parse_csv(args.only)) or None,
        distance_vmax=args.distance_vmax,
        split_uniform_sources=DEFAULT_SPLIT_UNIFORM_SOURCES and not args.no_split_uniform_sources,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
