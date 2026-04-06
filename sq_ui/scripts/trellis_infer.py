#!/usr/bin/env python3
"""Generate a point cloud from text with TRELLIS and export it as .ply."""

from __future__ import annotations

import argparse
import gc
import json
import struct
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from third_party.TRELLIS.trellis.pipelines import TrellisTextTo3DPipeline
from third_party.TRELLIS.trellis.utils import postprocessing_utils


DEFAULT_PIPELINE_PATH = "gui"
_PIPELINE_LOCK = threading.Lock()
_PIPELINE_CACHE: dict[str, TrellisTextTo3DPipeline] = {}


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_points(points: np.ndarray) -> np.ndarray:
    aabb = np.stack([points.min(axis=0), points.max(axis=0)])
    center = (aabb[0] + aabb[1]) / 2.0
    scale = float(np.max(aabb[1] - aabb[0]))
    if scale <= 1e-8:
        return points - center
    return (points - center) / scale


def coords_to_points(coords: torch.Tensor) -> np.ndarray:
    xyz = coords[:, 1:].detach().cpu().numpy().astype(np.float32)
    return (xyz + 0.5) / 64.0 - 0.5


def mesh_extract_to_trimesh(mesh: Any) -> Any:
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    vertices, faces = postprocessing_utils.postprocess_mesh(
        vertices,
        faces,
        simplify=True,
        simplify_ratio=0.9,
        fill_holes=False,
        verbose=False,
    )
    import trimesh

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def sample_points_from_mesh(mesh: Any, count: int) -> np.ndarray:
    sampled, _ = mesh.sample(count, return_index=True)
    return np.asarray(sampled, dtype=np.float32)


def save_ply(points: np.ndarray, path: Path) -> None:
    pts = np.asarray(points, dtype=np.float32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as fh:
        fh.write(header)
        for x, y, z in pts:
            fh.write(struct.pack("<fff", float(x), float(y), float(z)))


def load_pipeline(pipeline_path: str = DEFAULT_PIPELINE_PATH) -> TrellisTextTo3DPipeline:
    if not torch.cuda.is_available():
        raise RuntimeError("TRELLIS generation requires CUDA")
    with _PIPELINE_LOCK:
        cached = _PIPELINE_CACHE.get(pipeline_path)
        if cached is not None:
            return cached
        pipeline = TrellisTextTo3DPipeline.from_pretrained(pipeline_path)
        pipeline.cuda()
        _PIPELINE_CACHE[pipeline_path] = pipeline
        return pipeline


def generate_point_cloud(
    prompt: str,
    *,
    seed: int = 1,
    point_count: int = 4096,
    normalize: bool = True,
    prefer_mesh: bool = False,
    sparse_steps: int = 12,
    slat_steps: int = 12,
    cfg_strength: float = 7.5,
    slat_cfg_strength: float = 7.5,
    pipeline_path: str = DEFAULT_PIPELINE_PATH,
) -> tuple[np.ndarray, dict[str, Any]]:
    pipeline = load_pipeline(pipeline_path)

    with _PIPELINE_LOCK:
        cond = pipeline.get_cond_text([prompt])
        torch.manual_seed(seed)
        np.random.seed(seed)
        coords = pipeline.sample_sparse_structure(
            cond,
            num_samples=1,
            sampler_params={
                "steps": int(sparse_steps),
                "cfg_strength": float(cfg_strength),
            },
        )

        source = "sparse_structure"
        points = coords_to_points(coords)

        mesh_face_count = 0
        mesh_vertex_count = 0
        if prefer_mesh:
            try:
                slat = pipeline.sample_slat(
                    cond,
                    coords,
                    sampler_params={
                        "steps": int(slat_steps),
                        "cfg_strength": float(slat_cfg_strength),
                    },
                )
                outputs = pipeline.decode_slat(slat, formats=["mesh"])
                mesh = mesh_extract_to_trimesh(outputs["mesh"][0])
                mesh_vertex_count = int(len(mesh.vertices))
                mesh_face_count = int(len(mesh.faces))
                if mesh_vertex_count > 0 and mesh_face_count > 0:
                    points = sample_points_from_mesh(mesh, int(point_count))
                    source = "mesh_surface"
            except Exception as exc:  # noqa: BLE001
                source = f"sparse_structure_fallback:{type(exc).__name__}"

    if points.shape[0] != int(point_count):
        replace = points.shape[0] < int(point_count)
        idx = np.random.choice(points.shape[0], int(point_count), replace=replace)
        points = points[idx]

    if normalize:
        points = normalize_points(points)

    meta = {
        "prompt": prompt,
        "seed": int(seed),
        "point_count": int(points.shape[0]),
        "normalize": bool(normalize),
        "prefer_mesh": bool(prefer_mesh),
        "source": source,
        "sparse_steps": int(sparse_steps),
        "slat_steps": int(slat_steps),
        "cfg_strength": float(cfg_strength),
        "slat_cfg_strength": float(slat_cfg_strength),
        "mesh_vertex_count": mesh_vertex_count,
        "mesh_face_count": mesh_face_count,
        "pipeline_path": pipeline_path,
    }
    gc.collect()
    torch.cuda.empty_cache()
    return points, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TRELLIS point cloud from text")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--output-ply", required=True, help="Output PLY path")
    parser.add_argument("--output-meta", required=True, help="Output metadata JSON path")
    parser.add_argument("--name", default="trellis", help="Base name for metadata")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--point-count", type=int, default=4096, help="Number of output points")
    parser.add_argument("--normalize", type=parse_bool, default=True, help="Normalize output points")
    parser.add_argument("--prefer-mesh", type=parse_bool, default=True, help="Prefer mesh decode over sparse voxels")
    parser.add_argument("--sparse-steps", type=int, default=12, help="Sparse structure sampling steps")
    parser.add_argument("--slat-steps", type=int, default=12, help="SLAT sampling steps")
    parser.add_argument("--cfg-strength", type=float, default=7.5, help="CFG strength for sparse structure")
    parser.add_argument("--slat-cfg-strength", type=float, default=7.5, help="CFG strength for SLAT decoding")
    parser.add_argument(
        "--pipeline-path",
        default=DEFAULT_PIPELINE_PATH,
        help="Local pipeline dir or HF model id for TrellisTextTo3DPipeline",
    )
    args = parser.parse_args()

    output_ply = Path(args.output_ply)
    output_meta = Path(args.output_meta)
    ensure_parent(output_ply)
    ensure_parent(output_meta)

    points, meta = generate_point_cloud(
        args.prompt,
        seed=args.seed,
        point_count=args.point_count,
        normalize=bool(args.normalize),
        prefer_mesh=bool(args.prefer_mesh),
        sparse_steps=args.sparse_steps,
        slat_steps=args.slat_steps,
        cfg_strength=args.cfg_strength,
        slat_cfg_strength=args.slat_cfg_strength,
        pipeline_path=args.pipeline_path,
    )
    save_ply(points, output_ply)

    meta = {
        "name": args.name,
        **meta,
        "output_ply": str(output_ply),
    }
    output_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
