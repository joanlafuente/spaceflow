#!/usr/bin/env python3
"""Run SuperFlex and convert its full output into the editor NPZ layout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_INFER = REPO_ROOT / "scripts" / "superflex_full_infer.py"


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_names(base_name: str, count: int) -> list[str]:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in base_name).strip("_")
    prefix = safe or "superflex"
    return [f"{prefix}_part_{i + 1:02d}" for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-meta", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--name", default="superflex")
    parser.add_argument("--z-up", type=parse_bool, default=False)
    parser.add_argument("--normalize", type=parse_bool, default=True)
    parser.add_argument("--lm-optimization", type=parse_bool, default=False)
    parser.add_argument("--exist-threshold", type=float, default=0.5)
    parser.add_argument("--max-primitives", type=int, default=0)
    args = parser.parse_args()

    output_npz = Path(args.output_npz)
    output_meta = Path(args.output_meta)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sq_superflex_") as tmp:
        raw_npz = Path(tmp) / "raw_superflex.npz"
        raw_meta = Path(tmp) / "raw_superflex_meta.json"
        cmd = [
            sys.executable,
            str(FULL_INFER),
            "--input",
            args.input,
            "--output-npz",
            str(raw_npz),
            "--output-meta",
            str(raw_meta),
            "--checkpoint-dir",
            args.checkpoint_dir,
            "--name",
            args.name,
            "--z-up",
            "true" if args.z_up else "false",
            "--normalize",
            "true" if args.normalize else "false",
        ]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

        data = np.load(raw_npz)
        scales = np.asarray(data["scale"], dtype=np.float32)
        shapes = np.asarray(data["shape"], dtype=np.float32)
        translations = np.asarray(data["trans"], dtype=np.float32)
        rotations = np.asarray(data["rotate"], dtype=np.float32)
        exists = np.asarray(data["exist"], dtype=np.float32).reshape(-1)
        tapering = np.asarray(data.get("tapering", np.zeros((scales.shape[0], 2), dtype=np.float32)), dtype=np.float32)
        bending = np.asarray(data.get("bending", np.zeros((scales.shape[0], 6), dtype=np.float32)), dtype=np.float32)

        keep = np.where(exists > args.exist_threshold)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(exists))], dtype=np.int64)
        keep = keep[np.argsort(exists[keep])[::-1]]
        if args.max_primitives > 0:
            keep = keep[: args.max_primitives]

        np.savez_compressed(
            output_npz,
            scales=scales[keep],
            shapes=shapes[keep],
            translations=translations[keep],
            rotations=rotations[keep],
            tapering=tapering[keep],
            bending=bending[keep],
            confidence=exists[keep],
        )

        raw = json.loads(raw_meta.read_text(encoding="utf-8")) if raw_meta.is_file() else {}
        meta = {
            **raw,
            "name": args.name,
            "backend": "superflex",
            "primitive_count": int(len(keep)),
            "names": build_names(args.name, int(len(keep))),
            "confidence": exists[keep].astype(float).tolist(),
            "output_npz": str(output_npz),
            "lm_optimization_requested": bool(args.lm_optimization),
        }
        output_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
