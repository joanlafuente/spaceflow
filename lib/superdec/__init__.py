"""SUPERDEC-driven appearance correspondence for GuideFlow3D.

This package replaces the original PartField + k-means co-segmentation in
``lib/opt/appearance.py`` with a two-stage SUPERDEC-only correspondence:

1. Per-shape SUPERDEC segmentation (``predict_superdec``).
2. Voxel projection from SUPERDEC's per-point assignments to GuideFlow's
   active voxels (``project_assignments_to_voxels``).
3. Segment descriptors built from SUPERDEC primitive parameters and pooled
   voxel-position statistics (``segment_descriptors``).
4. Sinkhorn-based segment matching with confidence scores and an explicit
   unmatched bucket (``match_segments``).
5. Hard NN voxel matching in primitive-frame normalised coordinates
   (``primitive_frame_coords`` + ``match_voxels``).

See plan.md / README for the full design.
"""

from .diagnostics import save_segment_visualisations, save_summary
from .superdec import load_superdec_npz, predict_superdec
from .superdec_match import (
    UNMATCHED,
    AppearanceCorrespondence,
    SegmentMatch,
    VoxelMatch,
    VoxelProjection,
    build_correspondence,
    match_segments,
    match_voxels,
    primitive_frame_coords,
    project_assignments_to_voxels,
    segment_descriptors,
)

__all__ = [
    "predict_superdec",
    "load_superdec_npz",
    "project_assignments_to_voxels",
    "segment_descriptors",
    "match_segments",
    "primitive_frame_coords",
    "match_voxels",
    "build_correspondence",
    "save_segment_visualisations",
    "save_summary",
    "UNMATCHED",
    "VoxelProjection",
    "SegmentMatch",
    "VoxelMatch",
    "AppearanceCorrespondence",
]
