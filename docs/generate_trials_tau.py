#!/usr/bin/env python3
"""
generate_trials_tau.py
──────────────────────
Generates trials_tau.json from the tau control study data.

Expected structure under static/data/tau/:
    local_tau/
        1_chair.glb
        2_sofa.glb
        ...
    tau_3/
        1_chair.glb
        2_sofa.glb
        ...
    tau_10/
        1_chair.glb
        2_sofa.glb
        ...
    sq_priors_images/
        1_chair_sq.png      (any name, matched by number prefix)
        2_sofa_sq.png
        ...

Edit tau_metadata.json to set prompts for each numbered example.

Usage:
    python generate_trials_tau.py
"""

import json, os, random, re
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("static/data/tau")
OUTPUT_FILE  = Path("trials_tau.json")
METADATA     = Path("tau_metadata.json")

METHOD_DIRS  = {
    "local_tau": DATA_ROOT / "local_tau",
    "tau_3":     DATA_ROOT / "tau_3",      # uniform low τ
    "tau_10":    DATA_ROOT / "tau_10",     # uniform high τ
}
SQ_DIR       = DATA_ROOT / "sq_priors_glbs"

# The three pairwise comparisons to generate per scene
PAIRS = [
    ("local_tau", "tau_10"),   # ours vs uniform high
    ("local_tau", "tau_3"),    # ours vs uniform low
]

# ── HELPERS ───────────────────────────────────────────────────────────────
def get_number(filename):
    """Extract leading number from filename: '1_chair.glb' → 1"""
    m = re.match(r'^(\d+)', filename)
    return int(m.group(1)) if m else None

def find_file(directory, number):
    """Find any file in directory whose name starts with '{number}_'."""
    if not directory.exists():
        return None
    for f in directory.iterdir():
        if get_number(f.name) == number:
            return f
    return None

# ── MAIN ──────────────────────────────────────────────────────────────────
def generate():
    # Load metadata (prompts)
    if not METADATA.exists():
        print(f"ERROR: {METADATA} not found.")
        print("Create it using tau_metadata.json template.")
        return

    with open(METADATA) as f:
        meta = json.load(f)

    # Find all scene numbers from the local_tau folder
    if not METHOD_DIRS["local_tau"].exists():
        print(f"ERROR: {METHOD_DIRS['local_tau']} not found.")
        return

    glb_files   = [f for f in METHOD_DIRS["local_tau"].iterdir()
                   if f.suffix.lower() in ('.glb', '.gltf')]
    numbers     = sorted(set(filter(None, [get_number(f.name) for f in glb_files])))

    if not numbers:
        print(f"No GLB files found in {METHOD_DIRS['local_tau']}")
        return

    print(f"Found {len(numbers)} scenes: {numbers}\n")
    trials = []

    for num in numbers:
        num_str = str(num)

        # Check metadata
        if num_str not in meta:
            print(f"  SKIP scene {num}: no entry in {METADATA}")
            continue

        scene_meta = meta[num_str]
        if "prompt" not in scene_meta:
            print(f"  SKIP scene {num}: missing 'prompt' in metadata")
            continue

        # Find SQ prior image
        sq_file = find_file(SQ_DIR, num)
        if sq_file is None:
            print(f"  WARN scene {num}: no SQ prior image found in {SQ_DIR}")
            ref_path = ""
        else:
            ref_path = str(sq_file).replace("\\", "/")

        # Find GLB for each method
        method_files = {}
        skip = False
        for method, directory in METHOD_DIRS.items():
            f = find_file(directory, num)
            if f is None:
                print(f"  SKIP scene {num}: no file for method '{method}' in {directory}")
                skip = True
                break
            method_files[method] = str(f).replace("\\", "/")

        if skip:
            continue

        # Generate one trial per pair
        scene_id = f"scene_{num:02d}"
        for method_a, method_b in PAIRS:
            # Randomize which side A/B appears on
            if random.random() < 0.5:
                method_a, method_b = method_b, method_a

            trial = {
                "scene_id":    scene_id,
                "pair":        f"{method_a}_vs_{method_b}",
                "mapping":     {"A": method_a, "B": method_b},
                "outputs_a":   [method_files[method_a]],
                "outputs_b":   [method_files[method_b]],
                "ref":         ref_path,
                "prompt":      scene_meta["prompt"],
            }
            trials.append(trial)
            print(f"  + scene {num}: {method_a} vs {method_b}")

    if not trials:
        print("\nNo trials generated.")
        return

    random.shuffle(trials)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(trials, f, indent=2)

    print(f"\n✓ Wrote {len(trials)} trials to {OUTPUT_FILE}")
    print(f"  ({len(numbers)} scenes × 3 pairs)")
    print(f"  Each participant sees up to TRIALS_PER_PARTICIPANT (set in main_tau.js)")

if __name__ == "__main__":
    generate()