#!/usr/bin/env python3
"""Run SuperDec on one point cloud and save the *full* inference output.

Differs from sq_ui/scripts/superdec_infer.py in two ways:

1. We keep all 16 query slots (no exist-threshold filtering) so that the
   downstream matcher in lib/util/superdec_match.py can decide what to keep
   per-shape, without having to reconcile a per-shape filter mask.
2. We save the raw soft assign_matrix and the input points used by the
   model, both in the *input* coordinate frame (no Z-up rotation, no
   normalisation). This lets the matcher do a single nearest-neighbour
   look-up from each GuideFlow active voxel into input_points and copy the
   corresponding assign row, with no need to reproduce SuperDec's internal
   pre-processing.

This script must be invoked with the superdec_ui venv (see lib/util/superdec.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch
from omegaconf import OmegaConf

from superdec.data.dataloader import denormalize_outdict, normalize_points
from superdec.data.transform import rotate_around_axis
from superdec.lm_optimization.lm_optimizer import LMOptimizer
from superdec.superdec import SuperDec


POINT_COUNT = 4096


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_point_cloud(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in {".xyz", ".xyzn", ".xyzrgb", ".pcd", ".pts", ".ply"}:
        pc = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pc.points, dtype=np.float32)
        if points.size == 0:
            raise ValueError(f"Point cloud is empty or unreadable: {path}")
        return points
    mesh = o3d.io.read_triangle_mesh(str(path))
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if verts.size == 0:
        raise ValueError(f"Unsupported or empty input file: {path}")
    return verts


def prepare_zup_input_for_superdec(points: np.ndarray) -> np.ndarray:
    return rotate_around_axis(
        points,
        axis=(1, 0, 0),
        angle=-np.pi / 2,
        center_point=np.zeros(3),
    )


def sample_points(points: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    if points.shape[0] == count:
        return points
    rng = np.random.default_rng(seed)
    replace = points.shape[0] < count
    idxs = rng.choice(points.shape[0], count, replace=replace)
    return points[idxs]


def to_numpy_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: to_numpy_tree(v) for k, v in obj.items()}
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SuperDec and save full inference output.")
    parser.add_argument("--input", required=True, help="Point cloud or mesh file path")
    parser.add_argument("--output-npz", required=True, help="Output NPZ path")
    parser.add_argument("--output-meta", required=False, default="", help="Optional metadata JSON path")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory with ckpt.pt and config.yaml")
    parser.add_argument("--name", default="superdec", help="Base name for log lines")
    parser.add_argument("--z-up", type=parse_bool, default=False, help="Input is authored in Z-up")
    parser.add_argument("--normalize", type=parse_bool, default=True, help="Normalise point cloud before inference")
    parser.add_argument("--lm-optimization", type=parse_bool, default=False, help="Enable LM refinement")
    parser.add_argument("--seed", type=int, default=0, help="Sampling RNG seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_npz = Path(args.output_npz)
    checkpoint_dir = Path(args.checkpoint_dir)
    ckpt_path = checkpoint_dir / "ckpt.pt"
    config_path = checkpoint_dir / "config.yaml"

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not ckpt_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint artifacts in {checkpoint_dir}; expected ckpt.pt and config.yaml"
        )

    ensure_parent(output_npz)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    configs = OmegaConf.load(config_path)

    model = SuperDec(configs.superdec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.lm_optimization = bool(args.lm_optimization)
    if args.lm_optimization:
        model.lm_optimizer = LMOptimizer().to(device)
    model.eval()

    raw_points = load_point_cloud(input_path)
    sampled_user_frame = sample_points(raw_points, POINT_COUNT, seed=args.seed)

    # Apply z_up→y_up rotation only for the model's input. We keep
    # sampled_user_frame untouched so the saved per-point order stays in the
    # caller's original coordinate system, which is what GuideFlow's voxels are
    # also expressed in.
    if args.z_up:
        sampled_for_sd = prepare_zup_input_for_superdec(sampled_user_frame.copy())
    else:
        sampled_for_sd = sampled_user_frame

    if args.normalize:
        model_points, translation, scale = normalize_points(sampled_for_sd)
    else:
        model_points = sampled_for_sd
        translation = np.zeros(3, dtype=np.float32)
        scale = 1.0

    points_tensor = torch.from_numpy(model_points).unsqueeze(0).to(device).float()
    translation_batch = torch.as_tensor(np.array([translation], dtype=np.float32), device=device)
    scale_batch = torch.as_tensor(np.array([scale], dtype=np.float32), device=device)

    if args.lm_optimization:
        outdict = model(points_tensor)
    else:
        with torch.no_grad():
            outdict = model(points_tensor)

    # Denormalise primitive params back to the input coordinate frame (Z-up if
    # z_up=True). assign_matrix is *not* touched by denormalisation; it is
    # per-point (one row per element of model_points, which has the same row
    # order as sampled_user_frame), so the per-row correspondence to
    # sampled_user_frame is preserved.
    outdict = denormalize_outdict(outdict, translation_batch, scale_batch, args.z_up)
    out_np = to_numpy_tree(outdict)

    scale_pp = np.asarray(out_np["scale"][0], dtype=np.float32)        # (P, 3)
    shape_pp = np.asarray(out_np["shape"][0], dtype=np.float32)        # (P, 2)
    rotate_pp = np.asarray(out_np["rotate"][0], dtype=np.float32)      # (P, 3, 3)
    trans_pp = np.asarray(out_np["trans"][0], dtype=np.float32)        # (P, 3)
    exist_pp = np.asarray(out_np["exist"][0], dtype=np.float32).reshape(-1)  # (P,)
    assign_matrix = np.asarray(out_np["assign_matrix"][0], dtype=np.float32)  # (N, P)

    np.savez_compressed(
        output_npz,
        input_points=sampled_user_frame.astype(np.float32),
        assign_matrix=assign_matrix,
        scale=scale_pp,
        shape=shape_pp,
        rotate=rotate_pp,
        trans=trans_pp,
        exist=exist_pp,
        z_up=np.array(bool(args.z_up)),
    )

    if args.output_meta:
        Path(args.output_meta).write_text(
            json.dumps(
                {
                    "name": args.name,
                    "primitive_count": int(scale_pp.shape[0]),
                    "input_path": str(input_path),
                    "output_npz": str(output_npz),
                    "checkpoint_dir": str(checkpoint_dir),
                    "device": device,
                    "z_up_input": bool(args.z_up),
                    "normalize": bool(args.normalize),
                    "lm_optimization": bool(args.lm_optimization),
                    "n_input_points": int(sampled_user_frame.shape[0]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        f"[superdec_full_infer] {args.name}: saved {output_npz}  "
        f"(P={scale_pp.shape[0]}, N={sampled_user_frame.shape[0]}, "
        f"exist>0.5={int((exist_pp > 0.5).sum())})"
    )


if __name__ == "__main__":
    main()
