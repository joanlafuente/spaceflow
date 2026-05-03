"""SUPERDEC-driven correspondence for the GuideFlow3D appearance pipeline.

Implements the five steps of the plan, in order:

1. ``project_assignments_to_voxels`` — map SUPERDEC's per-input-point soft
   assignment matrix onto GuideFlow's active voxel grid by NN back-projection,
   with optional implicit-SDF boundary refinement that sharpens seam voxels
   using the explicit superquadric implicits.
2. ``segment_descriptors`` — pool a frame-invariant geometric descriptor per
   primitive from SUPERDEC parameters and primitive-frame voxel statistics.
   No PartField, no DINOv2, no learned per-shape features.
3. ``match_segments`` — Sinkhorn entropy-regularised transport between the two
   primitive descriptor sets with mass marginals from voxel counts; reduce
   the transport plan to a many-to-one ``tau`` map with confidence scores
   and an explicit unmatched bucket.
4. ``primitive_frame_coords`` — map every active voxel to its primitive's
   canonical [-1, 1]^3 box via ``f = R^T (x - t) / s``.
5. ``match_voxels`` — for each input voxel, hard nearest-neighbour search
   inside the matched appearance primitive's voxel set, in primitive-frame
   coordinates.

All functions are pure (no I/O) and operate on torch tensors so the matcher
can stay on the GPU alongside the SLAT optimisation. Inputs come from the
NPZ produced by ``predict_superdec``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


log = logging.getLogger(__name__)


UNMATCHED: int = -1
"""Sentinel value used in the ``tau`` map for input primitives that have no
sufficiently confident appearance partner; voxels carrying this label are
masked out of the appearance loss."""


# -----------------------------------------------------------------------------
# 1. Voxel projection
# -----------------------------------------------------------------------------


def _superquadric_implicit(
    voxels_xyz: torch.Tensor,            # (N, 3)
    scale: torch.Tensor,                 # (P, 3)
    shape: torch.Tensor,                 # (P, 2)
    rotate: torch.Tensor,                # (P, 3, 3)
    trans: torch.Tensor,                 # (P, 3)
    eps_floor: float = 1e-4,
) -> torch.Tensor:
    """Evaluate the superquadric inside-outside function for every (voxel, primitive).

    Returns ``e[N, P]`` where ``e < 0`` denotes inside the primitive surface
    and ``e > 0`` outside; ``argmin_p e[i, p]`` is the primitive whose
    surface contains (or is closest to containing) voxel ``i``. Mirrors the
    formula in ``superdec.utils.predictions_handler.PredictionHandler.get_occupancy``.
    """
    # Broadcast voxels to (N, 1, 3) and transform into each primitive's frame:
    # x' = R^T (x - t)
    centered = voxels_xyz.unsqueeze(1) - trans.unsqueeze(0)        # (N, P, 3)
    rotate_t = rotate.transpose(-1, -2).unsqueeze(0)               # (1, P, 3, 3)
    x_prim = torch.einsum('npij,npj->npi', rotate_t.expand(centered.shape[0], -1, -1, -1), centered)

    s = scale.clamp_min(eps_floor).unsqueeze(0)                    # (1, P, 3)
    eps1 = shape[..., 0].clamp_min(eps_floor).unsqueeze(0)         # (1, P)
    eps2 = shape[..., 1].clamp_min(eps_floor).unsqueeze(0)         # (1, P)

    # Formula matches superdec.utils.predictions_handler.PredictionHandler.get_occupancy:
    # a = ((x/sx)^(2/eps1) + (y/sy)^(2/eps2))^(eps1/(eps1+eps2))
    # b = (z/sz)^(2/eps1)
    # impl = a + b - 1   ; impl < 0 ⇔ inside the primitive surface.
    abs_xs = (x_prim.abs() / s).clamp_min(eps_floor)
    a1 = torch.pow(abs_xs[..., 0] ** 2, 1.0 / eps1)
    a2 = torch.pow(abs_xs[..., 1] ** 2, 1.0 / eps2)
    a = torch.pow(a1 + a2, eps1 / (eps1 + eps2))
    b = torch.pow(abs_xs[..., 2] ** 2, 1.0 / eps1)
    return a + b - 1.0


@dataclass
class VoxelProjection:
    s_hard: torch.Tensor       # (M,) long, primitive index per active voxel
    p_soft: torch.Tensor       # (M, P) float, soft membership per voxel
    boundary_conf: torch.Tensor  # (M,) float in [0,1], 1.0 = deep inside one primitive
    exist_mask: torch.Tensor   # (P,) bool, primitives kept by exist threshold


def project_assignments_to_voxels(
    voxels_xyz: torch.Tensor,           # (M, 3)
    superdec: dict,
    exist_threshold: float = 0.5,
    refine_boundary: bool = True,
    boundary_softmax_beta: float = 50.0,
) -> VoxelProjection:
    """Map SUPERDEC per-point soft assignments onto GuideFlow active voxels.

    Step 1: nearest-neighbour back-projection from each active voxel to the
    closest SUPERDEC input point, copying its assignment row.

    Step 2 (optional): for voxels whose top-2 soft probabilities are within
    ``boundary_thresh`` of each other (a fuzzy seam), recompute the soft
    membership analytically from the superquadric implicit values evaluated
    at that voxel's position. This sharpens boundaries without needing to
    reproduce SUPERDEC's internal pre-processing for every voxel.

    Parameters
    ----------
    voxels_xyz: (M, 3) float tensor on any device.
    superdec: dict loaded from the NPZ saved by ``predict_superdec``. Keys
        used: ``input_points`` (N,3), ``assign_matrix`` (N,P), ``scale``
        (P,3), ``shape`` (P,2), ``rotate`` (P,3,3), ``trans`` (P,3),
        ``exist`` (P,).
    exist_threshold: keep primitives with ``exist > threshold``; primitives
        below the threshold get ``-inf`` membership and never become the
        argmax winner.
    refine_boundary: if True (default), apply implicit-SDF refinement.
    boundary_softmax_beta: temperature for the implicit-based softmax used
        in the refinement. Higher -> sharper boundaries.

    Returns
    -------
    VoxelProjection
    """
    device = voxels_xyz.device
    voxels_xyz = voxels_xyz.float()
    input_points = torch.as_tensor(superdec['input_points'], dtype=torch.float32, device=device)
    assign_matrix = torch.as_tensor(superdec['assign_matrix'], dtype=torch.float32, device=device)
    scale = torch.as_tensor(superdec['scale'], dtype=torch.float32, device=device)
    shape = torch.as_tensor(superdec['shape'], dtype=torch.float32, device=device)
    rotate = torch.as_tensor(superdec['rotate'], dtype=torch.float32, device=device)
    trans = torch.as_tensor(superdec['trans'], dtype=torch.float32, device=device)
    exist = torch.as_tensor(superdec['exist'], dtype=torch.float32, device=device).reshape(-1)

    n_p = scale.shape[0]
    exist_mask = exist > exist_threshold
    if not exist_mask.any():
        # Fall back to the single most confident primitive so downstream code
        # still has something to work with; this matches superdec_infer.py's
        # behaviour.
        argmax_exist = int(exist.argmax().item())
        exist_mask = torch.zeros_like(exist_mask)
        exist_mask[argmax_exist] = True

    # ------- Step 1: NN back-projection -------
    # For each voxel find its nearest input point (chunked to bound memory).
    chunk = 4096
    nn_idx = torch.empty(voxels_xyz.shape[0], dtype=torch.long, device=device)
    for i in range(0, voxels_xyz.shape[0], chunk):
        d2 = torch.cdist(voxels_xyz[i:i + chunk], input_points)
        nn_idx[i:i + chunk] = d2.argmin(dim=1)
    p_soft = assign_matrix[nn_idx].clone()                          # (M, P)

    # Mask non-existing primitives so they never become the argmax winner.
    minus_inf = torch.tensor(-1e30, device=device, dtype=p_soft.dtype)
    p_soft = torch.where(exist_mask.unsqueeze(0).expand_as(p_soft), p_soft, minus_inf)
    # Renormalise into a proper distribution over the kept primitives.
    p_soft = torch.softmax(p_soft.clamp_min(-1e30), dim=1)

    # ------- Step 2: implicit-SDF boundary refinement -------
    if refine_boundary:
        # Voxels whose top-1 soft prob is "weak" -> seam candidates. Pick a
        # generous threshold (top1 < 0.6 OR top1 - top2 < 0.1).
        top2_vals, _ = p_soft.topk(2, dim=1)
        weak = (top2_vals[:, 0] < 0.6) | ((top2_vals[:, 0] - top2_vals[:, 1]) < 0.1)
        if weak.any():
            weak_idx = torch.where(weak)[0]
            v = voxels_xyz[weak_idx]
            with torch.no_grad():
                e = _superquadric_implicit(v, scale, shape, rotate, trans)  # (W, P)
            # Mask non-existing primitives with a large positive implicit so they
            # never become the argmin / argmax of the softmax.
            big = torch.tensor(1e6, device=device, dtype=e.dtype)
            e = torch.where(exist_mask.unsqueeze(0).expand_as(e), e, big)
            # Implicit value < 0 ⇒ inside; we want larger softmax weight for
            # smaller implicit, so use logits = -beta * e.
            new_soft = torch.softmax(-boundary_softmax_beta * e, dim=1)
            p_soft = p_soft.clone()
            p_soft[weak_idx] = new_soft

    # Hard label = argmax of soft membership over existing primitives.
    s_hard = p_soft.argmax(dim=1)

    # Boundary confidence: gap between top-1 and top-2 soft probabilities,
    # rescaled to [0, 1]. Voxels deep inside one primitive -> confidence ~ 1;
    # voxels on a seam -> confidence ~ 0.
    top2_vals, _ = p_soft.topk(2, dim=1)
    boundary_conf = (top2_vals[:, 0] - top2_vals[:, 1]).clamp(0.0, 1.0)

    return VoxelProjection(
        s_hard=s_hard,
        p_soft=p_soft,
        boundary_conf=boundary_conf,
        exist_mask=exist_mask,
    )


# -----------------------------------------------------------------------------
# 2. Segment descriptors (geometric, frame-invariant)
# -----------------------------------------------------------------------------


def segment_descriptors(
    voxels_xyz: torch.Tensor,        # (M, 3)
    s_hard: torch.Tensor,            # (M,) long
    superdec: dict,
    exist_mask: torch.Tensor,        # (P,) bool
    log_eps: float = 1e-6,
    include_translation_descriptor: bool = False,
) -> torch.Tensor:
    """Pool a frame-invariant per-primitive descriptor.

    Default descriptor (16-d, no PartField, no DINO):

    - ``log_volume``                                                     (1)
    - sorted ``log(scale_min, scale_med, scale_max)``                    (3)
    - shape exponents ``(eps1, eps2)``                                   (2)
    - log-mass = log(voxel_count + 1)                                    (1)
    - primitive-frame mean of voxel positions = mean(R^T (x-t)/s)        (3)
    - primitive-frame std  of voxel positions = std(R^T (x-t)/s)         (3)
    - shape ratio sorted (s_min/s_max, s_med/s_max)                      (2)
    - sphericity proxy = std/scale_max                                   (1)

    If ``include_translation_descriptor`` is True (off by default), also
    appends 4-d frame-dependent placement features (normalised translation
    + radial distance). Use only when both shapes share a canonical frame.

    Returns
    -------
    g: (P, D) float tensor, L2-normalised. Rows for non-existing primitives
    are zero.
    """
    device = voxels_xyz.device
    voxels_xyz = voxels_xyz.float()
    scale = torch.as_tensor(superdec['scale'], dtype=torch.float32, device=device)
    shape = torch.as_tensor(superdec['shape'], dtype=torch.float32, device=device)
    rotate = torch.as_tensor(superdec['rotate'], dtype=torch.float32, device=device)
    trans = torch.as_tensor(superdec['trans'], dtype=torch.float32, device=device)
    n_p = scale.shape[0]

    feats = []

    # log volume (∝ s_x s_y s_z up to a shape-dependent constant we ignore)
    log_vol = (scale.clamp_min(log_eps).log().sum(dim=-1, keepdim=True))      # (P, 1)
    feats.append(log_vol)

    # sorted log scales
    sorted_log_scale, _ = scale.clamp_min(log_eps).log().sort(dim=-1)         # (P, 3)
    feats.append(sorted_log_scale)

    # shape exponents
    feats.append(shape)                                                       # (P, 2)

    # voxel-mass (per primitive)
    mass = torch.zeros(n_p, dtype=torch.float32, device=device)
    mass.scatter_add_(0, s_hard, torch.ones_like(s_hard, dtype=torch.float32))
    feats.append(torch.log(mass + 1.0).unsqueeze(-1))                         # (P, 1)

    # Primitive-frame statistics: f = R^T (x - t) / s for voxels in this primitive.
    centered = voxels_xyz.unsqueeze(1) - trans.unsqueeze(0)                   # (M, P, 3)
    rotate_t = rotate.transpose(-1, -2).unsqueeze(0).expand(centered.shape[0], -1, -1, -1)
    x_prim = torch.einsum('npij,npj->npi', rotate_t, centered)                # (M, P, 3)
    f_all = x_prim / scale.clamp_min(log_eps).unsqueeze(0)                    # (M, P, 3)

    pf_mean = torch.zeros(n_p, 3, dtype=torch.float32, device=device)
    pf_sq = torch.zeros(n_p, 3, dtype=torch.float32, device=device)
    # Only voxels with s_hard == p contribute.
    one_hot = F.one_hot(s_hard, num_classes=n_p).float()                      # (M, P)
    f_each_voxel = f_all.gather(1, s_hard.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)  # (M, 3)
    pf_mean.scatter_add_(0, s_hard.unsqueeze(-1).expand(-1, 3), f_each_voxel)
    pf_sq.scatter_add_(0, s_hard.unsqueeze(-1).expand(-1, 3), f_each_voxel ** 2)
    counts = mass.clamp_min(1.0).unsqueeze(-1)
    pf_mean = pf_mean / counts
    pf_var = (pf_sq / counts) - pf_mean ** 2
    pf_std = pf_var.clamp_min(0.0).sqrt()

    feats.append(pf_mean)                                                     # (P, 3)
    feats.append(pf_std)                                                      # (P, 3)

    # sorted shape ratios (pose-invariant aspect ratio summary)
    s_sorted, _ = scale.clamp_min(log_eps).sort(dim=-1)                       # (P, 3)
    s_max = s_sorted[..., 2:3]                                                # (P, 1)
    ratio_min_max = s_sorted[..., 0:1] / s_max
    ratio_med_max = s_sorted[..., 1:2] / s_max
    feats.append(torch.cat([ratio_min_max, ratio_med_max], dim=-1))           # (P, 2)

    # sphericity proxy: std along principal axis vs scale magnitude
    sphericity = (pf_std.norm(dim=-1, keepdim=True) /
                  s_max.clamp_min(log_eps))                                   # (P, 1)
    feats.append(sphericity)

    if include_translation_descriptor:
        bbox = trans[exist_mask]
        if bbox.numel() == 0:
            t_norm = trans
        else:
            tmin = bbox.min(dim=0).values
            tmax = bbox.max(dim=0).values
            extent = (tmax - tmin).clamp_min(log_eps)
            t_norm = (trans - tmin) / extent                                  # (P, 3)
        radial = trans.norm(dim=-1, keepdim=True)
        feats.append(t_norm)                                                  # (P, 3)
        feats.append(radial)                                                  # (P, 1)

    g = torch.cat(feats, dim=-1)                                              # (P, D)

    # Replace NaN/Inf (e.g. for primitives with zero mass) with 0 before normalisation.
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    # Zero out rows for non-existing primitives.
    g = g * exist_mask.unsqueeze(-1).to(g.dtype)

    # L2 normalise (kept rows only). Empty rows stay all-zero.
    norm = g.norm(dim=-1, keepdim=True).clamp_min(log_eps)
    g = g / norm
    g = g * exist_mask.unsqueeze(-1).to(g.dtype)
    return g


# -----------------------------------------------------------------------------
# 3. Segment matching with confidence + unmatched bucket
# -----------------------------------------------------------------------------


def _sinkhorn(cost: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor,
              eps: float, n_iters: int) -> torch.Tensor:
    """Entropy-regularised Sinkhorn transport with given mass marginals.

    Returns a coupling matrix ``M`` with row sums ≤ ``mu`` and column sums ≤
    ``nu`` (approximately equal at convergence).
    """
    K = torch.exp(-cost / max(eps, 1e-8))
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)
    safe_eps = 1e-30
    for _ in range(n_iters):
        v = nu / (K.t() @ u + safe_eps)
        u = mu / (K @ v + safe_eps)
    return u.unsqueeze(1) * K * v.unsqueeze(0)


@dataclass
class SegmentMatch:
    tau: torch.Tensor         # (P_q,) long, appearance primitive index or UNMATCHED
    conf: torch.Tensor        # (P_q,) float in [0, 1]
    M: torch.Tensor           # (P_q, P_a) float, normalised soft transport plan
    cost: torch.Tensor        # (P_q, P_a) float, descriptor distance


def match_segments(
    g_q: torch.Tensor,                       # (P_q, D)
    g_a: torch.Tensor,                       # (P_a, D)
    mass_q: torch.Tensor,                    # (P_q,) float >= 0
    mass_a: torch.Tensor,                    # (P_a,) float >= 0
    exist_q: torch.Tensor,                   # (P_q,) bool
    exist_a: torch.Tensor,                   # (P_a,) bool
    eps: float = 0.05,
    n_iters: int = 30,
    conf_threshold: float = 0.15,
) -> SegmentMatch:
    """Match input primitives to appearance primitives via Sinkhorn transport.

    Output ``tau`` is a function ``{1..P_q} -> {1..P_a} ∪ {UNMATCHED}``;
    many-to-one is allowed. ``conf`` is the row-normalised top-1 mass and
    drives the unmatched threshold.
    """
    device = g_q.device
    n_q, n_a = g_q.shape[0], g_a.shape[0]

    cost = torch.cdist(g_q, g_a, p=2) ** 2                                    # (P_q, P_a)
    # Mask non-existing primitives by an enormous cost so they never win.
    big = torch.tensor(1e3, device=device, dtype=cost.dtype)
    cost = torch.where(exist_q.view(-1, 1).expand_as(cost), cost, big)
    cost = torch.where(exist_a.view(1, -1).expand_as(cost), cost, big)

    mass_q = mass_q.clamp_min(0.0)
    mass_a = mass_a.clamp_min(0.0)
    mass_q = mass_q * exist_q.to(mass_q.dtype)
    mass_a = mass_a * exist_a.to(mass_a.dtype)
    mu = mass_q / mass_q.sum().clamp_min(1e-8)
    nu = mass_a / mass_a.sum().clamp_min(1e-8)
    # Replace fully-empty marginals with uniform so Sinkhorn doesn't divide by 0.
    if not torch.isfinite(mu).all() or mu.sum() <= 0:
        mu = exist_q.to(mu.dtype) / exist_q.to(mu.dtype).sum().clamp_min(1.0)
    if not torch.isfinite(nu).all() or nu.sum() <= 0:
        nu = exist_a.to(nu.dtype) / exist_a.to(nu.dtype).sum().clamp_min(1.0)

    M = _sinkhorn(cost, mu, nu, eps=eps, n_iters=n_iters)                     # (P_q, P_a)

    # Row-normalise so each input primitive's row sums to 1 (over kept
    # appearance primitives) — this gives a clean confidence in [0, 1].
    row_sum = M.sum(dim=1, keepdim=True).clamp_min(1e-12)
    M_row = M / row_sum

    # Top-1 with explicit unmatched bucket.
    conf, r_star = M_row.max(dim=1)                                           # (P_q,), (P_q,)
    # Set UNMATCHED for non-existing input primitives or low-confidence rows.
    bad = (~exist_q) | (conf < conf_threshold)
    tau = torch.where(bad, torch.full_like(r_star, UNMATCHED), r_star)

    return SegmentMatch(tau=tau, conf=conf, M=M_row, cost=cost)


# -----------------------------------------------------------------------------
# 4. Voxel-level descriptor: primitive-frame coordinates
# -----------------------------------------------------------------------------


def primitive_frame_coords(
    voxels_xyz: torch.Tensor,        # (M, 3)
    s_hard: torch.Tensor,            # (M,) long
    superdec: dict,
    log_eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-voxel ``f = R^T (x - t) / s`` for each voxel's primitive.

    Returns ``(M, 3)`` float tensor; voxels whose primitive is non-existing
    return zero (they are masked out by the matcher anyway).
    """
    device = voxels_xyz.device
    voxels_xyz = voxels_xyz.float()
    scale = torch.as_tensor(superdec['scale'], dtype=torch.float32, device=device)
    rotate = torch.as_tensor(superdec['rotate'], dtype=torch.float32, device=device)
    trans = torch.as_tensor(superdec['trans'], dtype=torch.float32, device=device)
    s_clamped = scale.clamp_min(log_eps)

    # gather per-voxel primitive params
    t_per_voxel = trans[s_hard]                                  # (M, 3)
    R_per_voxel = rotate[s_hard]                                 # (M, 3, 3)
    s_per_voxel = s_clamped[s_hard]                              # (M, 3)

    centered = voxels_xyz - t_per_voxel                          # (M, 3)
    f = torch.einsum('mji,mj->mi', R_per_voxel, centered)        # (M, 3) (R^T x = einsum 'mji,mj->mi')
    f = f / s_per_voxel
    f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    return f


# -----------------------------------------------------------------------------
# 5. Voxel matching inside a matched primitive pair
# -----------------------------------------------------------------------------


@dataclass
class VoxelMatch:
    m: torch.Tensor           # (M_q,) long, appearance voxel index, -1 if invalid
    valid: torch.Tensor       # (M_q,) bool, True iff matched
    nn_dist: torch.Tensor     # (M_q,) float, primitive-frame distance to chosen match


def match_voxels(
    f_q: torch.Tensor,           # (M_q, 3)
    s_q: torch.Tensor,           # (M_q,) long
    f_a: torch.Tensor,           # (M_a, 3)
    s_a: torch.Tensor,           # (M_a,) long
    tau: torch.Tensor,           # (P_q,) long, UNMATCHED for unmatched
) -> VoxelMatch:
    """For each input voxel, hard NN to appearance voxels in tau(s_q[i]).

    Loops over input primitives that have a valid ``tau`` partner. For
    primitives mapped to UNMATCHED (or for which the appearance side has no
    voxels), ``valid`` is False and ``m`` = -1.
    """
    device = f_q.device
    n_q = f_q.shape[0]
    m = torch.full((n_q,), -1, dtype=torch.long, device=device)
    valid = torch.zeros(n_q, dtype=torch.bool, device=device)
    nn_dist = torch.full((n_q,), float('inf'), dtype=torch.float32, device=device)

    # Precompute per-primitive index lists on the appearance side.
    n_pa = int(tau.max().item()) + 1 if (tau >= 0).any() else 0
    n_pa = max(n_pa, int(s_a.max().item()) + 1) if s_a.numel() > 0 else n_pa
    a_indices_per_prim: list[Optional[torch.Tensor]] = [None] * max(n_pa, 1)
    for r in range(len(a_indices_per_prim)):
        idx = (s_a == r).nonzero(as_tuple=False).flatten()
        a_indices_per_prim[r] = idx if idx.numel() > 0 else None

    # Iterate over input primitives that actually appear on the input side.
    unique_q = torch.unique(s_q)
    for p in unique_q.tolist():
        r = int(tau[p].item()) if 0 <= p < tau.shape[0] else UNMATCHED
        if r == UNMATCHED:
            continue
        if r < 0 or r >= len(a_indices_per_prim) or a_indices_per_prim[r] is None:
            continue
        q_idx = (s_q == p).nonzero(as_tuple=False).flatten()
        if q_idx.numel() == 0:
            continue
        a_idx = a_indices_per_prim[r]

        # Chunked NN to bound memory if a primitive has many voxels.
        chunk = 4096
        for cs in range(0, q_idx.numel(), chunk):
            ce = min(cs + chunk, q_idx.numel())
            sub = q_idx[cs:ce]
            d2 = torch.cdist(f_q[sub], f_a[a_idx])  # (cs, |a_idx|)
            best = d2.argmin(dim=1)
            best_dist = d2.gather(1, best.unsqueeze(1)).squeeze(1)
            m[sub] = a_idx[best]
            valid[sub] = True
            nn_dist[sub] = best_dist

    return VoxelMatch(m=m, valid=valid, nn_dist=nn_dist)


# -----------------------------------------------------------------------------
# Convenience: end-to-end matcher for the appearance loss
# -----------------------------------------------------------------------------


@dataclass
class AppearanceCorrespondence:
    m: torch.Tensor                # (M_q,) long, m(i) — index into appearance voxels
    valid: torch.Tensor            # (M_q,) bool, True iff i ∈ L_q
    tau: torch.Tensor              # (P_q,) long
    conf: torch.Tensor             # (P_q,) float
    s_q: torch.Tensor              # (M_q,) long
    s_a: torch.Tensor              # (M_a,) long
    boundary_conf_q: torch.Tensor  # (M_q,) float
    nn_dist: torch.Tensor          # (M_q,) float


def build_correspondence(
    voxels_q_xyz: torch.Tensor,      # (M_q, 3)
    voxels_a_xyz: torch.Tensor,      # (M_a, 3)
    superdec_q: dict,
    superdec_a: dict,
    *,
    exist_threshold: float = 0.5,
    refine_boundary: bool = True,
    boundary_softmax_beta: float = 50.0,
    sinkhorn_eps: float = 0.05,
    sinkhorn_iters: int = 30,
    conf_threshold: float = 0.15,
    inside_threshold: float = 0.0,
    include_translation_descriptor: bool = False,
) -> AppearanceCorrespondence:
    """Run the full matcher pipeline on a pair of shapes."""
    proj_q = project_assignments_to_voxels(
        voxels_q_xyz, superdec_q,
        exist_threshold=exist_threshold,
        refine_boundary=refine_boundary,
        boundary_softmax_beta=boundary_softmax_beta,
    )
    proj_a = project_assignments_to_voxels(
        voxels_a_xyz, superdec_a,
        exist_threshold=exist_threshold,
        refine_boundary=refine_boundary,
        boundary_softmax_beta=boundary_softmax_beta,
    )

    g_q = segment_descriptors(
        voxels_q_xyz, proj_q.s_hard, superdec_q, proj_q.exist_mask,
        include_translation_descriptor=include_translation_descriptor,
    )
    g_a = segment_descriptors(
        voxels_a_xyz, proj_a.s_hard, superdec_a, proj_a.exist_mask,
        include_translation_descriptor=include_translation_descriptor,
    )

    # Voxel-mass marginals.
    n_pq = g_q.shape[0]
    n_pa = g_a.shape[0]
    mass_q = torch.zeros(n_pq, dtype=torch.float32, device=g_q.device)
    mass_q.scatter_add_(0, proj_q.s_hard, torch.ones_like(proj_q.s_hard, dtype=torch.float32))
    mass_a = torch.zeros(n_pa, dtype=torch.float32, device=g_a.device)
    mass_a.scatter_add_(0, proj_a.s_hard, torch.ones_like(proj_a.s_hard, dtype=torch.float32))

    seg = match_segments(
        g_q, g_a, mass_q, mass_a, proj_q.exist_mask, proj_a.exist_mask,
        eps=sinkhorn_eps, n_iters=sinkhorn_iters, conf_threshold=conf_threshold,
    )

    f_q = primitive_frame_coords(voxels_q_xyz, proj_q.s_hard, superdec_q)
    f_a = primitive_frame_coords(voxels_a_xyz, proj_a.s_hard, superdec_a)

    vm = match_voxels(f_q, proj_q.s_hard, f_a, proj_a.s_hard, seg.tau)

    valid = vm.valid.clone()
    if inside_threshold > 0.0:
        valid = valid & (proj_q.boundary_conf >= inside_threshold)

    return AppearanceCorrespondence(
        m=vm.m,
        valid=valid,
        tau=seg.tau,
        conf=seg.conf,
        s_q=proj_q.s_hard,
        s_a=proj_a.s_hard,
        boundary_conf_q=proj_q.boundary_conf,
        nn_dist=vm.nn_dist,
    )


__all__ = [
    "UNMATCHED",
    "VoxelProjection",
    "VoxelMatch",
    "SegmentMatch",
    "AppearanceCorrespondence",
    "project_assignments_to_voxels",
    "segment_descriptors",
    "match_segments",
    "primitive_frame_coords",
    "match_voxels",
    "build_correspondence",
]
