#!/usr/bin/env python3
"""Run SuperDec on a single point cloud and emit editor-friendly artifacts.

Outputs:
  - pipeline-compatible NPZ with keys: scales, shapes, translations, rotations
  - JSON metadata with stable primitive names and run information
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


S_ZUP_TO_YUP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def prepare_zup_input_for_superdec(points: np.ndarray) -> np.ndarray:
    # Mirror SuperDec's demo path for z_up=True: rotate the raw point cloud by
    # -90deg around X before inference, then convert the fitted poses back later.
    return rotate_around_axis(
        points,
        axis=(1, 0, 0),
        angle=-np.pi / 2,
        center_point=np.zeros(3),
    )


def convert_pose_zup_to_yup(
    translations: np.ndarray,
    rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        translations @ S_ZUP_TO_YUP.T,
        np.einsum("ij,njk->nik", S_ZUP_TO_YUP, rotations),
    )


def sample_points(points: np.ndarray, count: int) -> np.ndarray:
    if points.shape[0] == count:
        return points
    replace = points.shape[0] < count
    idxs = np.random.choice(points.shape[0], count, replace=replace)
    return points[idxs]


def to_numpy_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: to_numpy_tree(v) for k, v in obj.items()}
    return obj


def build_names(base_name: str, count: int) -> list[str]:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in base_name).strip("_")
    prefix = safe or "superdec"
    return [f"{prefix}_part_{i + 1:02d}" for i in range(count)]


def squeeze_primitive_axes(array: np.ndarray, trailing_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    arr = np.squeeze(arr)
    if trailing_shape:
        if arr.ndim < len(trailing_shape):
            raise ValueError(
                f"Expected trailing shape {trailing_shape}, got {arr.shape}"
            )
        if tuple(arr.shape[-len(trailing_shape):]) != trailing_shape:
            raise ValueError(
                f"Expected trailing shape {trailing_shape}, got {arr.shape}"
            )
        arr = arr.reshape(-1, *trailing_shape)
    else:
        arr = arr.reshape(-1)
    if arr.shape[1:] != trailing_shape:
        raise ValueError(
            f"Expected trailing shape {trailing_shape}, got {arr.shape}"
        )
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SuperDec on one point cloud.")
    parser.add_argument("--input", required=True, help="Point cloud or mesh file path")
    parser.add_argument("--output-npz", required=True, help="Output NPZ path")
    parser.add_argument("--output-meta", required=True, help="Output metadata JSON path")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory containing ckpt.pt and config.yaml")
    parser.add_argument("--name", default="superdec", help="Base name for generated primitives")
    parser.add_argument("--z-up", type=parse_bool, default=False, help="Input is authored in Z-up")
    parser.add_argument("--normalize", type=parse_bool, default=True, help="Normalize point cloud before inference")
    parser.add_argument("--lm-optimization", type=parse_bool, default=False, help="Enable LM optimization on model")
    parser.add_argument("--exist-threshold", type=float, default=0.5, help="Primitive existence threshold")
    parser.add_argument("--max-primitives", type=int, default=0, help="Optional cap after thresholding; 0 keeps all")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_npz = Path(args.output_npz)
    output_meta = Path(args.output_meta)
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
    ensure_parent(output_meta)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    configs = OmegaConf.load(config_path)

    model = SuperDec(configs.superdec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Upstream __init__ sets lm_optimization=False and only builds lm_optimizer when True at
    # construction time, so lm_optimizer is never created. Enable LM after load + attach module.
    model.lm_optimization = bool(args.lm_optimization)
    if args.lm_optimization:
        model.lm_optimizer = LMOptimizer().to(device)
    model.eval()

    raw_points = load_point_cloud(input_path)
    if args.z_up:
        raw_points = prepare_zup_input_for_superdec(raw_points)
    sampled = sample_points(raw_points, POINT_COUNT)

    if args.normalize:
        model_points, translation, scale = normalize_points(sampled)
    else:
        model_points = sampled
        translation = np.zeros(3, dtype=np.float32)
        scale = 1.0

    points_tensor = torch.from_numpy(model_points).unsqueeze(0).to(device).float()
    translation_batch = torch.as_tensor(
        np.array([translation], dtype=np.float32),
        device=device,
    )
    scale_batch = torch.as_tensor(
        np.array([scale], dtype=np.float32),
        device=device,
    )

    # LM refinement uses functorch jacfwd; must not run the whole forward under inference_mode/no_grad.
    if args.lm_optimization:
        outdict = model(points_tensor)
    else:
        with torch.no_grad():
            outdict = model(points_tensor)
    outdict = denormalize_outdict(outdict, translation_batch, scale_batch, args.z_up)
    out_np = to_numpy_tree(outdict)

    scales = squeeze_primitive_axes(out_np["scale"][0], (3,))
    shapes = squeeze_primitive_axes(out_np["shape"][0], (2,))
    translations = squeeze_primitive_axes(out_np["trans"][0], (3,))
    rotations = squeeze_primitive_axes(out_np["rotate"][0], (3, 3))
    exists = squeeze_primitive_axes(out_np["exist"][0], ())

    if args.z_up:
        translations, rotations = convert_pose_zup_to_yup(translations, rotations)

    keep = np.where(exists > args.exist_threshold)[0]
    if keep.size == 0:
        keep = np.array([int(np.argmax(exists))], dtype=np.int64)
    keep = keep[np.argsort(exists[keep])[::-1]]
    if args.max_primitives > 0:
        keep = keep[: args.max_primitives]

    out_scales = scales[keep]
    out_shapes = shapes[keep]
    out_translations = translations[keep]
    out_rotations = rotations[keep]
    out_exists = exists[keep]

    np.savez_compressed(
        output_npz,
        scales=out_scales,
        shapes=out_shapes,
        translations=out_translations,
        rotations=out_rotations,
    )

    meta = {
        "name": args.name,
        "primitive_count": int(len(keep)),
        "names": build_names(args.name, int(len(keep))),
        "confidence": out_exists.tolist(),
        "checkpoint_dir": str(checkpoint_dir),
        "input_path": str(input_path),
        "output_npz": str(output_npz),
        "device": device,
        "source_basis": "y-up",
        "normalize": bool(args.normalize),
        "lm_optimization": bool(args.lm_optimization),
        "z_up_input": bool(args.z_up),
    }
    output_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
