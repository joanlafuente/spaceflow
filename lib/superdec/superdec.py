"""SUPERDEC inference plumbing for the GuideFlow3D appearance pipeline.

The SuperDec package (with its PVCNN-style C++/CUDA extensions) is not
installed in the GuideFlow3D Python environment; it lives in a separate venv
at ``$SQ_SUPERDEC_VENV`` (default
``/work/scratch/$USER/spaceflow/superdec_ui/venv``). To avoid coupling the two
environments we run inference via subprocess against
``scripts/superdec_full_infer.py`` and read back the saved NPZ.

The NPZ format is the contract consumed by ``superdec_match.py``:

- ``input_points``  : (N, 3) float32, in the *caller's* coord frame (typically
  GuideFlow's [-0.5, 0.5]^3 voxel frame). Rows correspond to ``assign_matrix``.
- ``assign_matrix`` : (N, P) float32, softmax over P primitives per point.
- ``scale``         : (P, 3) float32, primitive scales (caller's frame).
- ``shape``         : (P, 2) float32, superquadric exponents (eps1, eps2).
- ``rotate``        : (P, 3, 3) float32, primitive rotation matrices.
- ``trans``         : (P, 3) float32, primitive translations (caller's frame).
- ``exist``         : (P,) float32, per-primitive existence probabilities.
- ``z_up``          : 0-d bool, whether the caller's frame is Z-up.
"""

from __future__ import annotations

import logging
import os
import os.path as osp
import shutil
import subprocess
import tempfile
from typing import Optional, Sequence, Union

import numpy as np
import utils3d


log = logging.getLogger(__name__)


def _default_superdec_base() -> str:
    user = os.environ.get("USER", "nedela")
    return f"/work/scratch/{user}/spaceflow/superdec_ui"


def _resolve_venv_dir(venv_dir: Optional[Union[str, os.PathLike]]) -> str:
    if venv_dir is None:
        venv_dir = os.environ.get(
            "SQ_SUPERDEC_VENV",
            osp.join(_default_superdec_base(), "venv"),
        )
    venv_dir = str(venv_dir)
    if not osp.isdir(venv_dir):
        raise FileNotFoundError(
            f"SuperDec venv not found: {venv_dir}. "
            f"Set SQ_SUPERDEC_VENV or run sq_ui/setup_superdec.sh."
        )
    return venv_dir


def _resolve_venv_python(venv_dir: Union[str, os.PathLike]) -> str:
    candidate = osp.join(str(venv_dir), "bin", "python")
    if not osp.isfile(candidate):
        raise FileNotFoundError(
            f"Could not find {candidate}. Set SQ_SUPERDEC_VENV to the SuperDec venv root."
        )
    return candidate


def _resolve_checkpoint_dir(checkpoint_dir: Optional[Union[str, os.PathLike]]) -> str:
    if checkpoint_dir is None:
        checkpoint_dir = os.environ.get(
            "SQ_SUPERDEC_CHECKPOINT_DIR",
            osp.join(_default_superdec_base(), "weights", "normalized"),
        )
    checkpoint_dir = str(checkpoint_dir)
    if not osp.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"SuperDec checkpoint directory not found: {checkpoint_dir}. "
            f"Set SQ_SUPERDEC_CHECKPOINT_DIR or run sq_ui/setup_superdec.sh."
        )
    return checkpoint_dir


def _write_voxels_ply(voxels_xyz: np.ndarray, ply_path: str) -> None:
    if voxels_xyz.ndim != 2 or voxels_xyz.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) voxel array, got {voxels_xyz.shape}")
    utils3d.io.write_ply(ply_path, np.asarray(voxels_xyz, dtype=np.float32))


def predict_superdec(
    voxels_xyz: np.ndarray,
    output_npz: str,
    name: str,
    z_up: bool = False,
    lm_optimization: bool = False,
    seed: int = 0,
    checkpoint_dir: Optional[str] = None,
    venv_dir: Optional[str] = None,
    extra_env: Optional[dict] = None,
    keep_input_ply: bool = False,
) -> dict:
    """Run SuperDec on a (N, 3) point cloud and write its full output to NPZ.

    Returns the loaded NPZ as a dict of numpy arrays.
    """
    output_npz = str(output_npz)
    os.makedirs(osp.dirname(output_npz) or ".", exist_ok=True)

    venv_dir = _resolve_venv_dir(venv_dir)
    venv_python = _resolve_venv_python(venv_dir)
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)

    # Repo root: lib/superdec/superdec.py -> lib/superdec -> lib -> repo_root
    repo_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    script_path = osp.join(repo_root, "scripts", "superdec_full_infer.py")
    if not osp.isfile(script_path):
        raise FileNotFoundError(f"superdec_full_infer.py not found at {script_path}")

    tmpdir = tempfile.mkdtemp(prefix="superdec_full_")
    ply_path = osp.join(tmpdir, f"{name}_voxels.ply")
    _write_voxels_ply(voxels_xyz, ply_path)

    cmd = [
        venv_python,
        script_path,
        "--input", ply_path,
        "--output-npz", output_npz,
        "--checkpoint-dir", checkpoint_dir,
        "--name", name,
        "--z-up", "true" if z_up else "false",
        "--normalize", "true",
        "--lm-optimization", "true" if lm_optimization else "false",
        "--seed", str(int(seed)),
    ]

    env = os.environ.copy()
    env["PATH"] = osp.join(venv_dir, "bin") + os.pathsep + env.get("PATH", "")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9;9.0")
    if extra_env:
        env.update(extra_env)

    log.info(
        "[predict_superdec] %s: %d voxels, z_up=%s, ckpt=%s, venv=%s",
        name, int(voxels_xyz.shape[0]), z_up, checkpoint_dir, venv_dir,
    )
    try:
        proc = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
    finally:
        if not keep_input_ply:
            shutil.rmtree(tmpdir, ignore_errors=True)

    if proc.returncode != 0:
        log.error("[predict_superdec] %s: subprocess failed (rc=%d)", name, proc.returncode)
        log.error("[predict_superdec] stdout:\n%s", proc.stdout)
        log.error("[predict_superdec] stderr:\n%s", proc.stderr)
        raise RuntimeError(
            f"SuperDec inference failed for {name!r} (rc={proc.returncode}). "
            f"See logs above; cmd was: {' '.join(cmd)}"
        )

    if proc.stdout:
        for line in proc.stdout.splitlines():
            log.info("[predict_superdec:%s] %s", name, line)

    if not osp.isfile(output_npz):
        raise FileNotFoundError(
            f"SuperDec subprocess returned 0 but {output_npz} was not written."
        )

    return dict(np.load(output_npz, allow_pickle=False))


def load_superdec_npz(npz_path: str) -> dict:
    """Lightweight loader for the NPZ saved by ``predict_superdec``."""
    return dict(np.load(npz_path, allow_pickle=False))


__all__: Sequence[str] = ("predict_superdec", "load_superdec_npz")
