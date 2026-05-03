"""Diagnostic exports for the SUPERDEC correspondence pipeline.

Saves three artefacts under ``<output_dir>/superdec/`` for visual inspection:

- ``superdec_segments_q.ply`` and ``superdec_segments_a.ply``
    Per-shape voxel point cloud where each voxel is coloured by its
    SUPERDEC primitive index ``s_q(i)`` / ``s_a(j)``. Distinct primitives
    get distinct colours; non-existing primitives get black.

- ``segment_correspondence_q.ply`` and ``segment_correspondence_a.ply``
    Same point clouds, but voxels in matched primitives share a colour
    across the two shapes (so it is easy to verify that a chair leg in Q
    has been assigned to the corresponding leg in A). Voxels whose input
    primitive is unmatched (``tau == UNMATCHED``) appear as grey on the
    structure side; the appearance side keeps its native primitive colours.

- ``superdec_summary.json``
    A small JSON dump of P_q, P_a, mean / per-segment confidence, voxel
    counts, the tau map, and the size of L_q. Useful when an output looks
    bad and you want to know whether the segmenter or the matcher is at
    fault before re-running the full optimisation.

Open3D is already a dependency of GuideFlow3D; we use the same
``open3d_pycg`` import as ``run.py``.
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
from typing import Optional, Sequence

import numpy as np
import open3d_pycg as o3d
import torch

from .superdec_match import UNMATCHED


log = logging.getLogger(__name__)


def _generate_palette(n: int, seed: int = 0) -> np.ndarray:
    """Distinct, deterministic RGB colours for n primitives (in [0, 1]).

    Uses a golden-ratio hue sweep for good visual separation; seed only
    affects the starting hue for variety across runs.
    """
    n = max(int(n), 1)
    rng = np.random.default_rng(seed)
    base_hue = float(rng.random())
    golden_ratio_conj = 0.6180339887498949
    hues = ((base_hue + np.arange(n) * golden_ratio_conj) % 1.0)
    rgb = np.zeros((n, 3), dtype=np.float32)
    for i, h in enumerate(hues):
        # Simple HSV->RGB with full S/V.
        c = 1.0
        x = c * (1.0 - abs(((h * 6.0) % 2.0) - 1.0))
        if h < 1 / 6.0:
            r, g, b = c, x, 0.0
        elif h < 2 / 6.0:
            r, g, b = x, c, 0.0
        elif h < 3 / 6.0:
            r, g, b = 0.0, c, x
        elif h < 4 / 6.0:
            r, g, b = 0.0, x, c
        elif h < 5 / 6.0:
            r, g, b = x, 0.0, c
        else:
            r, g, b = c, 0.0, x
        rgb[i] = (r, g, b)
    return rgb


def _save_ply_with_colors(
    points: np.ndarray, colors: np.ndarray, ply_path: str
) -> None:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pc.colors = o3d.utility.Vector3dVector(
        np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
    )
    o3d.io.write_point_cloud(ply_path, pc, write_ascii=False)


def save_segment_visualisations(
    voxels_q_xyz: torch.Tensor,
    voxels_a_xyz: torch.Tensor,
    s_q: torch.Tensor,
    s_a: torch.Tensor,
    tau: torch.Tensor,
    output_dir: str,
    *,
    palette_seed: int = 0,
) -> None:
    """Save colour-coded segment + correspondence point clouds for QC."""
    out = osp.join(output_dir, 'superdec')
    os.makedirs(out, exist_ok=True)

    voxels_q = voxels_q_xyz.detach().cpu().numpy()
    voxels_a = voxels_a_xyz.detach().cpu().numpy()
    s_q_np = s_q.detach().cpu().numpy().astype(np.int64)
    s_a_np = s_a.detach().cpu().numpy().astype(np.int64)
    tau_np = tau.detach().cpu().numpy().astype(np.int64)

    n_pq = max(int(s_q_np.max()) + 1 if s_q_np.size else 0, int(tau.shape[0]))
    n_pa = max(int(s_a_np.max()) + 1 if s_a_np.size else 0, 1)

    palette_q = _generate_palette(n_pq, seed=palette_seed)
    palette_a = _generate_palette(n_pa, seed=palette_seed + 17)

    # 1. Segment-only PLYs (each shape coloured by its own primitive index).
    colors_q_seg = palette_q[s_q_np]
    colors_a_seg = palette_a[s_a_np]
    _save_ply_with_colors(voxels_q, colors_q_seg, osp.join(out, 'superdec_segments_q.ply'))
    _save_ply_with_colors(voxels_a, colors_a_seg, osp.join(out, 'superdec_segments_a.ply'))

    # 2. Correspondence PLYs (matched pairs share a colour drawn from
    #    palette_a; unmatched input primitives -> grey on the Q side).
    grey = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    matched_palette_q = np.zeros((n_pq, 3), dtype=np.float32)
    for p in range(n_pq):
        r = int(tau_np[p]) if p < tau_np.shape[0] else UNMATCHED
        if r == UNMATCHED or r < 0 or r >= palette_a.shape[0]:
            matched_palette_q[p] = grey
        else:
            matched_palette_q[p] = palette_a[r]
    colors_q_corr = matched_palette_q[s_q_np]
    _save_ply_with_colors(voxels_q, colors_q_corr, osp.join(out, 'segment_correspondence_q.ply'))
    _save_ply_with_colors(voxels_a, colors_a_seg, osp.join(out, 'segment_correspondence_a.ply'))


def save_summary(
    output_dir: str,
    *,
    n_pq: int,
    n_pa: int,
    n_voxels_q: int,
    n_voxels_a: int,
    tau: torch.Tensor,
    conf: torch.Tensor,
    valid_mask: torch.Tensor,
    nn_dist: torch.Tensor,
    mass_q: Optional[torch.Tensor] = None,
    mass_a: Optional[torch.Tensor] = None,
) -> None:
    """Dump a JSON summary of correspondence quality."""
    out = osp.join(output_dir, 'superdec')
    os.makedirs(out, exist_ok=True)

    tau_list = tau.detach().cpu().tolist()
    conf_list = conf.detach().cpu().tolist()
    valid_count = int(valid_mask.sum().item())
    valid_dist = nn_dist[valid_mask] if valid_mask.any() else None

    summary = {
        "P_q": int(n_pq),
        "P_a": int(n_pa),
        "n_voxels_q": int(n_voxels_q),
        "n_voxels_a": int(n_voxels_a),
        "n_unmatched_segments": int(sum(1 for r in tau_list if r == UNMATCHED)),
        "L_q_size": int(valid_count),
        "L_q_fraction": float(valid_count) / max(int(n_voxels_q), 1),
        "tau": tau_list,
        "confidence_per_segment": conf_list,
        "mean_confidence": float(np.mean(conf_list)) if conf_list else 0.0,
        "mean_nn_dist_in_primitive_frame": (
            float(valid_dist.mean().item()) if valid_dist is not None and valid_dist.numel() > 0 else None
        ),
        "median_nn_dist_in_primitive_frame": (
            float(valid_dist.median().item()) if valid_dist is not None and valid_dist.numel() > 0 else None
        ),
    }
    if mass_q is not None:
        summary["mass_q"] = mass_q.detach().cpu().tolist()
    if mass_a is not None:
        summary["mass_a"] = mass_a.detach().cpu().tolist()

    with open(osp.join(out, 'superdec_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


__all__: Sequence[str] = ("save_segment_visualisations", "save_summary")
