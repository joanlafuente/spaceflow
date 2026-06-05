#!/usr/bin/env python3
"""
generate_trials.py
──────────────────
Run this script whenever you add or update renders in static/data/.
It scans the folder structure and writes trials.json for the study page.

Usage:
    python generate_trials.py

Expected folder structure per scene:
    static/data/
    └── chair_01/
        ├── metadata.json          ← required: prompt + part labels
        ├── reference_a.webp       ← required: reference image for part A
        ├── reference_b.webp       ← required: reference image for part B
        ├── ours/
        │   └── output.webp        ← your method's render
        ├── sc/
        │   └── output.webp        ← SpaceControl baseline
        └── sf_global/
            └── output.webp        ← SpaceFlow global-reference baseline

metadata.json format:
    {
      "prompt":      "A wooden chair with red velvet seat and oak legs",
      "ref_a_label": "seat (red velvet)",
      "ref_b_label": "legs (oak wood)"
    }
"""

import json
import os
import random
from pathlib import Path

# ── CONFIGURE THESE ────────────────────────────────────────────────────────
DATA_ROOT    = Path("static/data")
OUTPUT_FILE  = Path("trials.json")

# All method folder names. "ours" must be present in each scene.
# Each scene will produce one trial per baseline (ours vs sc, ours vs sf_global, etc.)
METHODS      = ["ours", "sc", "sf_global"]

# The filename(s) inside each method folder to display.
# Can be a list if you want to show multiple images per side, e.g. front+back.
OUTPUT_FILES = ["output.webp"]

# ── GENERATION ────────────────────────────────────────────────────────────
def generate():
    if not DATA_ROOT.exists():
        print(f"ERROR: {DATA_ROOT} not found.")
        print("Create it and add scene subfolders before running this script.")
        return

    scenes = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir()])
    if not scenes:
        print(f"No scene folders found in {DATA_ROOT}.")
        return

    print(f"Found {len(scenes)} scenes: {scenes}\n")
    trials = []

    for scene_id in scenes:
        scene_path = DATA_ROOT / scene_id
        issues = []

        # ── check metadata ────────────────────────────────────────────
        meta_path = scene_path / "metadata.json"
        if not meta_path.exists():
            issues.append("missing metadata.json")
        else:
            with open(meta_path) as f:
                meta = json.load(f)
            required_keys = ["prompt", "ref_a_label", "ref_b_label"]
            missing = [k for k in required_keys if k not in meta]
            if missing:
                issues.append(f"metadata.json missing keys: {missing}")

        # ── check reference images ────────────────────────────────────
        def find_reference(scene_path, name):
            for ext in ['.webp', '.png', '.jpg', '.jpeg']:
                p = scene_path / f"{name}{ext}"
                if p.exists():
                    return p
            return None

        ref_a = find_reference(scene_path, "reference_a")
        ref_b = find_reference(scene_path, "reference_b")

        if not ref_a.exists():
            issues.append("missing reference_a.webp")
        if not ref_b.exists():
            issues.append("missing reference_b.webp")

        # ── check "ours" method exists ────────────────────────────────
        if not (scene_path / "ours").is_dir():
            issues.append("missing 'ours/' folder")

        if issues:
            print(f"  SKIP {scene_id}: {'; '.join(issues)}")
            continue

        # ── check output files for each method ────────────────────────
        available_methods = []
        for method in METHODS:
            method_dir = scene_path / method
            if not method_dir.is_dir():
                continue
            missing_outputs = [f for f in OUTPUT_FILES
                               if not (method_dir / f).exists()]
            if missing_outputs:
                print(f"  WARN {scene_id}/{method}: missing {missing_outputs} — skipping method")
                continue
            available_methods.append(method)

        if "ours" not in available_methods:
            print(f"  SKIP {scene_id}: 'ours/' folder has no valid outputs")
            continue

        baselines = [m for m in available_methods if m != "ours"]
        if not baselines:
            print(f"  SKIP {scene_id}: no baseline methods available")
            continue

        # ── generate one trial per (ours vs baseline) pair ────────────
        for baseline in baselines:
            # randomly assign which side "ours" appears on
            if random.random() < 0.5:
                method_a, method_b = "ours", baseline
            else:
                method_a, method_b = baseline, "ours"

            trial = {
                "scene_id":    scene_id,
                "mapping":     {"A": method_a, "B": method_b},
                "outputs_a":   [f"static/data/{scene_id}/{method_a}/{f}"
                                for f in OUTPUT_FILES],
                "outputs_b":   [f"static/data/{scene_id}/{method_b}/{f}"
                                for f in OUTPUT_FILES],
                "ref_a":       f"static/data/{scene_id}/reference_a.webp",
                "ref_b":       f"static/data/{scene_id}/reference_b.webp",
                "prompt":      meta["prompt"],
                "ref_a_label": meta["ref_a_label"],
                "ref_b_label": meta["ref_b_label"],
            }
            trials.append(trial)
            print(f"  + {scene_id}: ours vs {baseline}  "
                  f"(A={method_a}, B={method_b})")

    if not trials:
        print("\nNo trials generated. Check the warnings above.")
        return

    # shuffle so the order is different from the data folder order
    random.shuffle(trials)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(trials, f, indent=2)

    print(f"\n✓ Wrote {len(trials)} trials to {OUTPUT_FILE}")
    print(f"  Each participant sees up to TRIALS_PER_PARTICIPANT"
          f" (set in main.js) randomly chosen trials.")


if __name__ == "__main__":
    generate()
