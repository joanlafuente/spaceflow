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
from pathlib import Path
import sys
import traceback

from omegaconf import OmegaConf


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_local_tau  # noqa: E402


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


def main(argv=None):
    args = parse_args(argv)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    variants = config.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Experiment config must contain a non-empty variants list")

    cfg_path = str(config.get("spaceflow_config") or "config/default.yaml")
    cfg = OmegaConf.load(cfg_path)
    parsed_variants = [(variant, run_local_tau.init_args(_variant_argv(variant))) for variant in variants]

    print("[experiment] starting SpaceFlow experiment")
    experiment_status = 0
    pipeline = None

    try:
        for index, (variant, variant_args) in enumerate(parsed_variants, start=1):
            name = str(variant.get("name") or f"variant_{index}")
            output_dir = Path(variant_args.output_dir)
            log_path = output_dir / "spaceflow.log"
            status_path = output_dir / "status.txt"
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[experiment] variant {index}/{len(parsed_variants)}: {name}")

            code = 0
            with tee_variant_log(log_path):
                try:
                    if pipeline is None:
                        pipeline = run_local_tau.load_trellis_pipeline(variant_args, cfg)
                    run_local_tau.run(variant_args, cfg=cfg, generation_pipeline=pipeline)
                except Exception:  # noqa: BLE001 - mirror shell wrapper behavior
                    code = 1
                    traceback.print_exc()
                finally:
                    if pipeline is not None:
                        run_local_tau.offload_trellis_pipeline(pipeline)
                    gc.collect()
                    try:
                        run_local_tau.torch.cuda.empty_cache()
                    except Exception:
                        pass

            if code == 0:
                status_path.write_text("succeeded\n", encoding="utf-8")
                print(f"[experiment] variant {index}/{len(parsed_variants)} succeeded: {name}")
            else:
                status_path.write_text(f"failed:{code}\n", encoding="utf-8")
                print(f"[experiment] variant {index}/{len(parsed_variants)} failed with code {code}: {name}")
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

    print("[experiment] completed SpaceFlow experiment")
    return experiment_status


if __name__ == "__main__":
    raise SystemExit(main())
