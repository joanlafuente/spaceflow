#!/usr/bin/env python3
"""Helpers for texture-focused TRELLIS experiment variants."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
import utils3d

import run_local_tau
from lib.util import generation


def _ensure_text_pipeline_ready(pipeline) -> None:
    pipeline.cuda()
    run_local_tau.move_trellis_text_conditioner(pipeline, "cuda")


def _write_metadata(output_dir: Path, metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "variant_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _coords_from_struct_voxels(path: Path) -> torch.Tensor:
    struct_coords = utils3d.io.read_ply(str(path))[0]
    struct_coords = torch.from_numpy(struct_coords).float().cuda()
    struct_coords = ((struct_coords + 0.5) * 64).long()
    zeros = torch.zeros(
        (struct_coords.size(0), 1),
        dtype=struct_coords.dtype,
        device=struct_coords.device,
    )
    return torch.cat([zeros, struct_coords], dim=1).int()


def _export_sparse_structure(coords: torch.Tensor, output_dir: Path) -> None:
    coords_np = coords.detach().cpu().numpy()
    filtered_coords = coords_np[:, 1:]
    run_local_tau.sparse_voxels_to_glb(
        filtered_coords,
        grid_size=64,
        output_filename=str(output_dir / "sample.glb"),
    )


def _decode_text_slat(
    pipeline,
    output_dir: Path,
    prompt: str,
    coords: torch.Tensor,
    *,
    seed: int,
    reset_seed: bool = True,
) -> None:
    _ensure_text_pipeline_ready(pipeline)
    cond = pipeline.get_cond_text([prompt])
    if reset_seed:
        torch.manual_seed(seed)
    slat = pipeline.sample_slat(cond, coords, {})
    generation.decode_slat(
        pipeline,
        slat.feats,
        slat.coords,
        str(output_dir / "out_sim.glb"),
        None,
        texture=True,
    )


def run_trellis_raw_text_variant(
    pipeline,
    output_dir: Path,
    prompt: str,
    *,
    seed: int = 1,
) -> None:
    """Run unconstrained TRELLIS text-to-3D from the flattened prompt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(
        output_dir,
        {
            "runner": "trellis_raw_text",
            "prompt": prompt,
            "seed": seed,
        },
    )
    _ensure_text_pipeline_ready(pipeline)
    Path("debug").mkdir(exist_ok=True)
    cond = pipeline.get_cond_text([prompt])
    torch.manual_seed(seed)
    coords = pipeline.sample_sparse_structure(cond, num_samples=1, sampler_params={})
    _export_sparse_structure(coords, output_dir)
    _decode_text_slat(pipeline, output_dir, prompt, coords, seed=seed, reset_seed=False)


def run_fixed_structure_appearance_fm_variant(
    pipeline,
    output_dir: Path,
    prompt: str,
    structure_voxels_path: Path,
    *,
    seed: int = 1,
) -> None:
    """Run normal TRELLIS text SLAT sampling on a fixed SpaceFlow structure."""
    if not structure_voxels_path.is_file():
        raise FileNotFoundError(f"Missing source structure voxels: {structure_voxels_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(
        output_dir,
        {
            "runner": "fixed_structure_appearance_fm",
            "prompt": prompt,
            "seed": seed,
            "structure_voxels_path": str(structure_voxels_path),
        },
    )

    source_dir = structure_voxels_path.parent.parent
    _copy_if_exists(structure_voxels_path, output_dir / "voxels" / "struct_voxels.ply")
    _copy_if_exists(source_dir / "struct_mesh.glb", output_dir / "struct_mesh.glb")
    _copy_if_exists(source_dir / "struct_mesh_zup.glb", output_dir / "struct_mesh_zup.glb")
    _copy_if_exists(source_dir / "spatial_control_mesh.ply", output_dir / "spatial_control_mesh.ply")

    coords = _coords_from_struct_voxels(structure_voxels_path)
    _decode_text_slat(pipeline, output_dir, prompt, coords, seed=seed)
