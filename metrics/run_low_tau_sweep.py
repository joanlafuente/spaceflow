#!/usr/bin/env python3
"""Launch structure-only SpaceFlow variants with several low-control tau values."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SQ_UI_SCRIPTS = REPO_ROOT / "sq_ui" / "scripts"
if str(SQ_UI_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SQ_UI_SCRIPTS))

from run_examples_structure_experiment import (  # noqa: E402
    _json_write,
    _load_npz_arrays,
    _num_tag,
    _parse_prompts,
    _sanitize_name,
    _save_low_control_bbox,
    _save_subset_npz,
)


DEFAULT_RUN_ROOT = REPO_ROOT / "spaceflow_runtime" / "metrics_low_tau_sweep"
DEFAULT_ASSET_ROOT = REPO_ROOT / "spaceflow_runtime" / "metrics_low_tau_assets"
DEFAULT_EXAMPLES_DIR = REPO_ROOT / "examples"
EXPERIMENT_RUNNER = REPO_ROOT / "sq_ui" / "scripts" / "run_spaceflow_experiment.py"


def _utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _default_python_bin() -> str:
    env_override = os.environ.get("SQ_SPACEFLOW_PYTHON", "").strip()
    if env_override:
        return env_override
    candidates = [
        REPO_ROOT / "envs" / "guideflow3d" / "bin" / "python",
        REPO_ROOT.parent / "guideflow3d" / "envs" / "guideflow3d" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    return sorted(set(values))


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(token) for token in raw.replace(" ", "").split(",") if token]


def _variant_argv(
    asset_paths: dict[str, Path],
    output_dir: Path,
    prompt: str,
    *,
    low_tau: float,
    high_tau: float,
    polyak_tau: float,
    repaint_steps: int,
    texture_optim_steps: int,
    full_pipeline: bool,
) -> list[str]:
    argv = [
        "--guidance_mode",
        "similarity",
        "--output_dir",
        str(output_dir),
        "--shape_superquadric_path",
        str(asset_paths["all"]),
        "--shape_tau",
        str(low_tau),
        "--polyak_update_tau",
        str(polyak_tau),
        "--n_repaint_steps",
        str(repaint_steps),
        "--texture_optim_steps",
        str(texture_optim_steps),
        "--text_prompt",
        prompt,
        "--shape_superquadric_high_control_path",
        str(asset_paths["high_control"]),
        "--shape_tau_high_control",
        str(high_tau),
        "--low_control_superquadric_mask_path",
        str(asset_paths["low_control_bbox"]),
        "--local_tau_mode",
        "low_control_mask",
        "--convert_yup_to_zup",
        "--appearance_text",
        prompt,
    ]
    if full_pipeline:
        argv.append("--full_pipeline")
    return argv


def _prepare_assets(source_npz: Path, asset_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    arrays = _load_npz_arrays(source_npz)
    control_levels = np.asarray(arrays["control_levels"], dtype=np.float64)
    high_mask = control_levels >= 0.5
    low_mask = control_levels < 0.5
    if int(high_mask.sum()) == 0:
        raise ValueError(f"{source_npz} has no high-control primitives")
    if int(low_mask.sum()) == 0:
        raise ValueError(f"{source_npz} has no low-control primitives")

    asset_dir.mkdir(parents=True, exist_ok=True)
    asset_paths = {
        "all": asset_dir / "all.npz",
        "high_control": asset_dir / "high_control.npz",
        "low_control_bbox": asset_dir / "low_control_bbox.npz",
    }
    shutil.copy2(source_npz, asset_paths["all"])
    _save_subset_npz(asset_paths["high_control"], arrays, high_mask)
    bbox = _save_low_control_bbox(asset_paths["low_control_bbox"], arrays, low_mask)
    manifest = {
        "source_npz": str(source_npz),
        "counts": {
            "all": int(control_levels.shape[0]),
            "high": int(high_mask.sum()),
            "low": int(low_mask.sum()),
        },
        "bbox": bbox,
        "paths": {key: str(path) for key, path in asset_paths.items()},
        "primitives": [
            {
                "index": index,
                "name": f"SQ {index + 1}",
                "controlLevel": "low" if float(value) < 0.5 else "high",
                "visible": True,
            }
            for index, value in enumerate(control_levels)
        ],
    }
    _json_write(asset_dir / "manifest.json", manifest)
    return asset_paths, manifest


def _prepare_example(
    *,
    example_index: int,
    prompt: str,
    source_npz: Path,
    run_dir: Path,
    asset_root: Path,
    low_taus: list[float],
    high_tau: float,
    polyak_tau: float,
    repaint_steps: int,
    texture_optim_steps: int,
    full_pipeline: bool,
) -> dict[str, Any]:
    slug = _sanitize_name(prompt.lower(), fallback=f"example_{example_index:02d}")
    example_name = f"{example_index:02d}_{slug}_experiment"
    example_run_dir = run_dir / example_name
    output_dir = example_run_dir / "output"
    asset_dir = asset_root / run_dir.name / f"{example_index:02d}_{slug}"
    asset_paths, asset_manifest = _prepare_assets(source_npz, asset_dir)

    variants: list[dict[str, Any]] = []
    for variant_index, low_tau in enumerate(low_taus, start=1):
        variant_name = (
            f"{variant_index:02d}_local_tau{_num_tag(low_tau)}"
            f"_tau{_num_tag(high_tau)}_polyak{_num_tag(polyak_tau)}"
        )
        variant_output_dir = output_dir / _sanitize_name(variant_name)
        variants.append(
            {
                "name": variant_name,
                "output_dir": str(variant_output_dir),
                "argv": _variant_argv(
                    asset_paths,
                    variant_output_dir,
                    prompt,
                    low_tau=low_tau,
                    high_tau=high_tau,
                    polyak_tau=polyak_tau,
                    repaint_steps=repaint_steps,
                    texture_optim_steps=texture_optim_steps,
                    full_pipeline=full_pipeline,
                ),
                "mode": "local_tau",
                "low_tau": low_tau,
                "high_tau": high_tau,
                "polyak_tau": polyak_tau,
                "n_repaint_steps": repaint_steps,
                "texture_optim_steps": texture_optim_steps,
                "example_index": example_index,
                "prompt": prompt,
            }
        )

    runner_config = {
        "spaceflow_config": "config/default.yaml",
        "run_dir": str(example_run_dir),
        "variants": variants,
        "comparison": {"enabled": False},
        "experiment_type": "geometry",
    }
    _json_write(output_dir / "experiment_manifest.json", {"variants": [{k: v for k, v in item.items() if k != "argv"} for item in variants]})
    _json_write(example_run_dir / "experiment_runner_config.json", runner_config)
    _json_write(
        example_run_dir / "run_meta.json",
        {
            "run_id": f"{run_dir.name}_{example_name}",
            "status": "prepared",
            "project_name": "metrics_low_tau_sweep",
            "output_dir": str(output_dir),
            "log_path": str(example_run_dir / "spaceflow.log"),
            "experiment_mode": True,
            "experiment_type": "geometry",
            "asset_entry": {
                "asset_dir": str(asset_dir),
                "manifest_path": str(asset_dir / "manifest.json"),
                "paths": {key: str(path) for key, path in asset_paths.items()},
                "counts": asset_manifest["counts"],
            },
            "experiment_runner_config": str(example_run_dir / "experiment_runner_config.json"),
            "experiment_variants": [{k: v for k, v in item.items() if k != "argv"} for item in variants],
            "source_npz": str(source_npz),
            "example_index": example_index,
        },
    )
    return {"example_run_dir": example_run_dir, "config": example_run_dir / "experiment_runner_config.json"}


def _srun_prefix(time_limit: str, partition: str, account: str, gpus: str, constraint: str) -> list[str]:
    command = [
        "srun",
        f"--partition={partition}",
        f"--account={account}",
        f"--time={time_limit}",
        "--ntasks=1",
        "--export=ALL",
        f"--gpus={gpus}",
    ]
    if constraint:
        command.append(f"--constraint={constraint}")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default="1,2", help="Comma-separated examples from examples/*.npz.")
    parser.add_argument("--examples-dir", type=Path, default=DEFAULT_EXAMPLES_DIR)
    parser.add_argument("--prompts-path", type=Path, default=DEFAULT_EXAMPLES_DIR / "text_prompts.txt")
    parser.add_argument("--low-taus", default="1,3,6", help="Comma-separated low-control tau values.")
    parser.add_argument("--high-tau", type=float, default=10.0)
    parser.add_argument("--polyak-tau", type=float, default=0.18)
    parser.add_argument("--repaint-steps", type=int, default=10)
    parser.add_argument("--texture-optim-steps", type=int, default=2)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--run-id", default="", help="Optional fixed run id.")
    parser.add_argument("--full-pipeline", action="store_true", help="Run appearance optimization too. Off by default.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare configs but do not execute.")
    parser.add_argument("--use-srun", action="store_true", help="Execute each example through srun.")
    parser.add_argument("--srun-time", default="02:00:00")
    parser.add_argument("--srun-partition", default=os.environ.get("SQ_SPACEFLOW_SLURM_PARTITION", "interactive"))
    parser.add_argument("--srun-account", default=os.environ.get("SQ_SPACEFLOW_SLURM_ACCOUNT", "3dv"))
    parser.add_argument("--srun-gpus", default=os.environ.get("SQ_SPACEFLOW_SLURM_GPUS", "1"))
    parser.add_argument("--srun-constraint", default=os.environ.get("SQ_SPACEFLOW_SLURM_CONSTRAINT", "5060ti"))
    parser.add_argument("--python-bin", default=_default_python_bin())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    example_indices = _parse_csv_ints(args.examples)
    low_taus = _parse_csv_floats(args.low_taus)
    if not low_taus:
        raise SystemExit("--low-taus must include at least one value")
    if any(args.high_tau < low_tau for low_tau in low_taus):
        raise SystemExit("--high-tau must be >= every low tau")

    run_id = args.run_id.strip() or f"{_utc_timestamp()}_low_tau_sweep"
    run_dir = args.run_root / _sanitize_name(run_id)
    prompts = _parse_prompts(args.prompts_path)
    prepared = []
    for index in example_indices:
        source_npz = args.examples_dir / f"{index}.npz"
        if not source_npz.is_file():
            raise FileNotFoundError(source_npz)
        prompt = prompts.get(index)
        if not prompt:
            raise ValueError(f"No prompt for example {index} in {args.prompts_path}")
        prepared.append(
            _prepare_example(
                example_index=index,
                prompt=prompt,
                source_npz=source_npz,
                run_dir=run_dir,
                asset_root=args.asset_root,
                low_taus=low_taus,
                high_tau=args.high_tau,
                polyak_tau=args.polyak_tau,
                repaint_steps=args.repaint_steps,
                texture_optim_steps=args.texture_optim_steps,
                full_pipeline=args.full_pipeline,
            )
        )

    manifest = {
        "run_dir": str(run_dir),
        "examples": example_indices,
        "low_taus": low_taus,
        "high_tau": args.high_tau,
        "polyak_tau": args.polyak_tau,
        "full_pipeline": bool(args.full_pipeline),
        "prepared": [{"example_run_dir": str(item["example_run_dir"]), "config": str(item["config"])} for item in prepared],
    }
    _json_write(run_dir / "low_tau_sweep_manifest.json", manifest)

    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return

    for item in prepared:
        config = Path(item["config"])
        command = [args.python_bin, str(EXPERIMENT_RUNNER), "--config", str(config)]
        if args.use_srun:
            command = _srun_prefix(
                args.srun_time,
                args.srun_partition,
                args.srun_account,
                args.srun_gpus,
                args.srun_constraint,
            ) + command
        print(f"Running: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
