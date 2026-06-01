#!/usr/bin/env python3
"""Run SuperFlex on one point cloud/mesh and save full assignment output.

The local SuperFlex checkout still uses the package name ``superdec``. This
script is intended to be launched with ``PYTHONPATH`` pointing at the
SuperFlex repo so it imports the deformable implementation, not the old
SuperDEC package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch
import trimesh
from omegaconf import OmegaConf

from superdec.superdec import SuperDec


DEFAULT_POINT_COUNT = 4096


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def rotate_around_x(points: np.ndarray, angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
    return np.asarray(points, dtype=np.float32) @ rot.T


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    translation = points.mean(axis=0).astype(np.float32)
    centered = points - translation
    scale = float(max(2.0 * np.max(np.abs(centered)), 1e-4))
    return (centered / scale).astype(np.float32), translation, scale


def denormalize_outdict(outdict: dict, translation: torch.Tensor, scale: torch.Tensor, z_up: bool) -> dict:
    scale_b = scale[:, None, None]
    translation_b = translation[:, None, :]
    outdict["scale"] = outdict["scale"] * scale_b
    outdict["trans"] = outdict["trans"] * scale_b + translation_b
    if "bending_k" in outdict:
        outdict["bending_k"] = outdict["bending_k"] / scale_b
    if z_up:
        trans_np = rotate_around_x(outdict["trans"].detach().cpu().numpy().reshape(-1, 3), np.pi / 2.0)
        outdict["trans"] = torch.as_tensor(
            trans_np.reshape(tuple(outdict["trans"].shape)),
            dtype=outdict["trans"].dtype,
            device=outdict["trans"].device,
        )
        rot_x_90 = torch.as_tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=outdict["rotate"].dtype,
            device=outdict["rotate"].device,
        )
        outdict["rotate"] = torch.einsum("ij,bpjk->bpik", rot_x_90, outdict["rotate"])
    return outdict


def sample_surface_random(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float32)


def sample_surface_even(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface_even(mesh, count, seed=seed)
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] >= count:
        return points[:count]

    # `sample_surface_even` is rejection-based and can return fewer samples.
    # Keep its more regular coverage, then top up to the requested tensor size.
    missing = count - points.shape[0]
    extra = sample_surface_random(mesh, missing, seed + 7919)
    if points.shape[0] == 0:
        return extra
    return np.concatenate([points, extra], axis=0)


def load_input_points(path: Path, count: int, seed: int, sample_mode: str) -> np.ndarray:
    ext = path.suffix.lower()
    rng = np.random.default_rng(seed)
    if sample_mode in {"surface", "surface_even"}:
        mesh = trimesh.load(path, force="mesh")
        if getattr(mesh, "faces", None) is not None and len(mesh.faces) > 0:
            if sample_mode == "surface_even":
                return sample_surface_even(mesh, count, seed)
            return sample_surface_random(mesh, count, seed)
        # Fall through to point sampling if the input is a point cloud.

    if ext in {".xyz", ".xyzn", ".xyzrgb", ".pcd", ".pts", ".ply"}:
        pc = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pc.points, dtype=np.float32)
    else:
        mesh = trimesh.load(path, force="mesh")
        points = np.asarray(mesh.vertices, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Could not load a non-empty point set from {path}")
    if points.shape[0] == count:
        return points
    idx = rng.choice(points.shape[0], count, replace=points.shape[0] < count)
    return points[idx]


def to_numpy_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: to_numpy_tree(v) for k, v in obj.items()}
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-meta", default="")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--name", default="superflex")
    parser.add_argument("--z-up", type=parse_bool, default=False)
    parser.add_argument("--normalize", type=parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-count", type=int, default=DEFAULT_POINT_COUNT)
    parser.add_argument("--sample-mode", choices=("surface", "surface_even", "points"), default="surface")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_npz = Path(args.output_npz)
    checkpoint_dir = Path(args.checkpoint_dir)
    ckpt_path = checkpoint_dir / "ckpt.pt"
    config_path = checkpoint_dir / "config.yaml"

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not ckpt_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Missing ckpt.pt/config.yaml in {checkpoint_dir}")

    ensure_parent(output_npz)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    configs = OmegaConf.load(config_path)
    model = SuperDec(configs.superdec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sampled_user_frame = load_input_points(input_path, args.point_count, args.seed, args.sample_mode)
    sampled_for_model = rotate_around_x(sampled_user_frame, -np.pi / 2.0) if args.z_up else sampled_user_frame

    if args.normalize:
        model_points, translation, scale = normalize_points(sampled_for_model)
    else:
        model_points = sampled_for_model.astype(np.float32)
        translation = np.zeros(3, dtype=np.float32)
        scale = 1.0

    points_tensor = torch.from_numpy(model_points).unsqueeze(0).to(device).float()
    translation_batch = torch.as_tensor(np.array([translation], dtype=np.float32), device=device)
    scale_batch = torch.as_tensor(np.array([scale], dtype=np.float32), device=device)

    with torch.no_grad():
        outdict = model(points_tensor)
    outdict = denormalize_outdict(outdict, translation_batch, scale_batch, bool(args.z_up))
    out_np = to_numpy_tree(outdict)

    scale_pp = np.asarray(out_np["scale"][0], dtype=np.float32)
    shape_pp = np.asarray(out_np["shape"][0], dtype=np.float32)
    rotate_pp = np.asarray(out_np["rotate"][0], dtype=np.float32)
    trans_pp = np.asarray(out_np["trans"][0], dtype=np.float32)
    exist_pp = np.asarray(out_np["exist"][0], dtype=np.float32).reshape(-1)
    assign_matrix = np.asarray(out_np["assign_matrix"][0], dtype=np.float32)

    n_prim = scale_pp.shape[0]
    taper = np.asarray(out_np.get("tapering", np.zeros((1, n_prim, 2), dtype=np.float32))[0], dtype=np.float32)
    if "bending" in out_np:
        bending = np.asarray(out_np["bending"][0], dtype=np.float32)
    elif "bending_k" in out_np and "bending_a" in out_np:
        bk = np.asarray(out_np["bending_k"][0], dtype=np.float32)
        ba = np.asarray(out_np["bending_a"][0], dtype=np.float32)
        bending = np.stack([bk[:, 0], ba[:, 0], bk[:, 1], ba[:, 1], bk[:, 2], ba[:, 2]], axis=1)
    else:
        bending = np.zeros((n_prim, 6), dtype=np.float32)

    np.savez_compressed(
        output_npz,
        input_points=sampled_user_frame.astype(np.float32),
        assign_matrix=assign_matrix,
        scale=scale_pp,
        shape=shape_pp,
        rotate=rotate_pp,
        trans=trans_pp,
        exist=exist_pp,
        tapering=taper,
        bending=bending,
        z_up=np.array(bool(args.z_up)),
        backend=np.array("superflex"),
        sample_mode=np.array(args.sample_mode),
    )

    if args.output_meta:
        Path(args.output_meta).write_text(
            json.dumps(
                {
                    "name": args.name,
                    "backend": "superflex",
                    "input_path": str(input_path),
                    "checkpoint_dir": str(checkpoint_dir),
                    "device": device,
                    "z_up_input": bool(args.z_up),
                    "normalize": bool(args.normalize),
                    "sample_mode": args.sample_mode,
                    "n_input_points": int(sampled_user_frame.shape[0]),
                    "primitive_count": int(n_prim),
                    "extended": bool("tapering" in out_np or "bending_k" in out_np or "bending" in out_np),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        f"[superflex_full_infer] {args.name}: saved {output_npz} "
        f"(P={n_prim}, N={sampled_user_frame.shape[0]}, exist>0.5={int((exist_pp > 0.5).sum())}, "
        f"extended={bool(np.any(np.abs(taper) > 1e-7) or np.any(np.abs(bending) > 1e-7))})"
    )


if __name__ == "__main__":
    main()
