#!/usr/bin/env python3
"""Run multiple SpaceFlow variants in one Python process.

This keeps the TRELLIS pipeline loaded across variants while preserving the
per-variant logs and status files produced by the shell experiment wrapper.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import shutil
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from omegaconf import OmegaConf


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_local_tau  # noqa: E402
from trellis_texture_variants import (  # noqa: E402
    run_fixed_structure_appearance_fm_variant,
    run_fixed_structure_guideflow_appearance_fm_variant,
    run_trellis_raw_text_variant,
)

LOGGER = logging.getLogger("spaceflow_experiment")


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.2f}s"


def _experiment_log(message: str) -> None:
    LOGGER.info("[experiment] %s", message)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.streams[0].fileno()


@contextlib.contextmanager
def tee_variant_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        tee_stdout = TeeStream(old_stdout, log_file)
        tee_stderr = TeeStream(old_stderr, log_file)
        handler_streams = []

        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        for handler in logging.getLogger().handlers:
            if hasattr(handler, "stream"):
                handler_streams.append((handler, handler.stream))
                handler.stream = tee_stderr
        try:
            yield
        finally:
            for handler, stream in handler_streams:
                handler.stream = stream
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a SpaceFlow experiment in one process")
    parser.add_argument("--config", required=True, help="Path to experiment runner JSON")
    return parser.parse_args(argv)


def _variant_argv(variant: dict[str, object]) -> list[str]:
    argv = variant.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("Each experiment variant must provide an argv string list")
    return argv


def _variant_runner(variant: dict[str, object]) -> str:
    return str(variant.get("runner") or "spaceflow").strip() or "spaceflow"


def _variant_output_dir(variant: dict[str, object], variant_args=None) -> Path:
    if variant_args is not None:
        return Path(variant_args.output_dir)
    output_dir = variant.get("output_dir")
    if not output_dir:
        raise ValueError("Each experiment variant must provide output_dir")
    return Path(str(output_dir))


def _copy_input_superquadrics_glb(variant: dict[str, object], output_dir: Path) -> None:
    source_raw = str(variant.get("input_superquadrics_glb_path") or "").strip()
    if not source_raw:
        return
    source = Path(source_raw)
    if not source.is_file():
        _experiment_log(f"input superquadrics GLB not available for copy yet: {source}")
        return
    target = output_dir / "input_superquadrics_colored.glb"
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    _experiment_log(f"copied input superquadrics GLB: {source} -> {target}")


def _load_pipeline_for_experiment(pipeline, cfg, variant_args=None):
    if pipeline is not None:
        _experiment_log("reusing shared TRELLIS pipeline")
        return pipeline
    args = variant_args or SimpleNamespace(trellis_pipeline_path=None)
    return run_local_tau.load_trellis_pipeline(args, cfg)


def _parse_variants(variants: list[dict[str, object]]):
    parsed = []
    for variant in variants:
        runner = _variant_runner(variant)
        variant_args = run_local_tau.init_args(_variant_argv(variant)) if runner == "spaceflow" else None
        parsed.append((variant, runner, variant_args))
    return parsed


def run_variant(
    variant: dict[str, object],
    runner: str,
    variant_args,
    cfg,
    pipeline,
):
    if runner == "spaceflow":
        pipeline = _load_pipeline_for_experiment(pipeline, cfg, variant_args)
        run_local_tau.run(variant_args, cfg=cfg, generation_pipeline=pipeline)
        return pipeline

    pipeline = _load_pipeline_for_experiment(pipeline, cfg)
    output_dir = _variant_output_dir(variant, variant_args)
    _copy_input_superquadrics_glb(variant, output_dir)
    prompt = str(variant.get("prompt") or variant.get("flattened_prompt") or "").strip()
    if not prompt:
        raise ValueError(f"Variant {variant.get('name') or runner} is missing prompt")
    seed = int(variant.get("seed") or 1)

    if runner == "trellis_raw_text":
        run_trellis_raw_text_variant(pipeline, output_dir, prompt, seed=seed)
        return pipeline

    if runner == "fixed_structure_appearance_fm":
        structure_voxels_raw = str(variant.get("structure_voxels_path") or "").strip()
        if not structure_voxels_raw:
            raise ValueError("fixed_structure_appearance_fm variant is missing structure_voxels_path")
        structure_voxels_path = Path(structure_voxels_raw)
        run_fixed_structure_appearance_fm_variant(
            pipeline,
            output_dir,
            prompt,
            structure_voxels_path,
            seed=seed,
        )
        return pipeline

    if runner == "fixed_structure_guideflow_appearance_fm":
        structure_voxels_raw = str(variant.get("structure_voxels_path") or "").strip()
        if not structure_voxels_raw:
            raise ValueError("fixed_structure_guideflow_appearance_fm variant is missing structure_voxels_path")
        structure_voxels_path = Path(structure_voxels_raw)
        run_fixed_structure_guideflow_appearance_fm_variant(
            pipeline,
            cfg,
            output_dir,
            prompt,
            structure_voxels_path,
            seed=seed,
        )
        return pipeline

    raise ValueError(f"Unknown experiment variant runner: {runner}")


def render_experiment_comparison(config: dict[str, object], config_path: Path) -> None:
    comparison = config.get("comparison")
    if isinstance(comparison, dict) and comparison.get("enabled") is False:
        _experiment_log("comparison render disabled")
        return

    run_dir = Path(str(config.get("run_dir") or config_path.parent))
    output_name = "output/variant_comparison_lower_camera.png"
    azim = 0.0
    elev = 55.0
    if isinstance(comparison, dict):
        output_name = str(comparison.get("output_name") or output_name)
        azim = float(comparison.get("azim", azim))
        elev = float(comparison.get("elev", elev))

    render_start = time.perf_counter()
    _experiment_log("rendering variant comparison")
    from render_spaceflow_experiment_comparison import render_comparison

    output_path = render_comparison(run_dir, output_name, azim, elev)
    _experiment_log(f"rendered variant comparison in {_elapsed(render_start)}: {output_path}")


def main(argv=None):
    args = parse_args(argv)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    variants = config.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Experiment config must contain a non-empty variants list")

    cfg_path = str(config.get("spaceflow_config") or "config/default.yaml")
    cfg = OmegaConf.load(cfg_path)
    texture_optim_steps = config.get("texture_optim_steps")
    if texture_optim_steps is not None:
        texture_optim_steps = int(texture_optim_steps)
        if texture_optim_steps < 2:
            raise ValueError("texture_optim_steps must be an integer at least 2")
        cfg.sim_guidance.steps = texture_optim_steps
        _experiment_log(f"overriding texture optimization steps: {texture_optim_steps}")
    parsed_variants = _parse_variants(variants)

    experiment_start = time.perf_counter()
    _experiment_log(f"starting SpaceFlow experiment with {len(parsed_variants)} variants")
    experiment_status = 0
    pipeline = None

    try:
        for index, (variant, runner, variant_args) in enumerate(parsed_variants, start=1):
            name = str(variant.get("name") or f"variant_{index}")
            output_dir = _variant_output_dir(variant, variant_args)
            log_path = output_dir / "spaceflow.log"
            status_path = output_dir / "status.txt"
            output_dir.mkdir(parents=True, exist_ok=True)

            code = 0
            variant_start = time.perf_counter()
            with tee_variant_log(log_path):
                _experiment_log(
                    f"variant {index}/{len(parsed_variants)} started: "
                    f"{name} ({runner}); output_dir={output_dir}"
                )
                try:
                    pipeline = run_variant(variant, runner, variant_args, cfg, pipeline)
                except Exception:  # noqa: BLE001 - mirror shell wrapper behavior
                    code = 1
                    traceback.print_exc()
                finally:
                    cleanup_start = time.perf_counter()
                    if pipeline is not None:
                        run_local_tau.offload_trellis_pipeline(pipeline)
                    gc.collect()
                    try:
                        run_local_tau.torch.cuda.empty_cache()
                    except Exception:
                        pass
                    _experiment_log(
                        f"variant {index}/{len(parsed_variants)} cleanup completed "
                        f"in {_elapsed(cleanup_start)}"
                    )

                if code == 0:
                    status_path.write_text("succeeded\n", encoding="utf-8")
                    _experiment_log(
                        f"variant {index}/{len(parsed_variants)} succeeded: "
                        f"{name} in {_elapsed(variant_start)}"
                    )
                else:
                    status_path.write_text(f"failed:{code}\n", encoding="utf-8")
                    _experiment_log(
                        f"variant {index}/{len(parsed_variants)} failed with code {code}: "
                        f"{name} after {_elapsed(variant_start)}"
                    )
                    if experiment_status == 0:
                        experiment_status = code
    finally:
        if pipeline is not None:
            del pipeline
        gc.collect()
        try:
            run_local_tau.torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        render_experiment_comparison(config, config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[experiment] comparison failed: {exc}")

    _experiment_log(f"completed SpaceFlow experiment in {_elapsed(experiment_start)}")
    return experiment_status


if __name__ == "__main__":
    raise SystemExit(main())
