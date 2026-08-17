#!/usr/bin/env python3
"""
generate_trials_tau.py
──────────────────────
Generates trials_tau.json from the new 45-scene experiment directory.

Expected structure under static/data/tau/:
    local_tau/          acoustic_guitar.glb, airplane_bent_tapered.glb, ...
    tau_3/              same names
    tau_10/             same names
    sq_priors_glbs/     acoustic_guitar_sq.glb, ...

File names are derived from the experiment folder name by stripping _full_experiment.
Prompts are read from each scene's output/experiment_manifest.json.

Usage:
    cd docs/
    python generate_trials_tau.py
"""

import json, random
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("static/data/tau")
OUTPUT_FILE = Path("trials_tau.json")
SRC_ROOT    = Path("/work/courses/3dv/team3/spaceflow-minimal/83_final_examples")

METHOD_DIRS = {
    "local_tau": DATA_ROOT / "local_tau",
    "tau_3":     DATA_ROOT / "tau_3",
    "tau_10":    DATA_ROOT / "tau_10",
}
SQ_DIR = DATA_ROOT / "sq_priors_glbs"

LOCAL_TAU_SUBDIR = "01_local_tau3_tau10_polyak0p18"

PAIRS = [
    ("local_tau", "tau_10"),   # ours vs uniform high τ
    ("local_tau", "tau_3"),    # ours vs uniform low τ
]

# ── HELPERS ───────────────────────────────────────────────────────────────
def get_prompt(scene_name: str) -> str | None:
    """Extract prompt from the experiment manifest for the local_tau variant."""
    # Try exact match first, then glob (handles non-standard dir names like the monitor)
    exact = SRC_ROOT / f"{scene_name}_full_experiment"
    matches = [exact] if exact.exists() else list(SRC_ROOT.glob(f"{scene_name}_full_experiment*"))
    scene_dir = matches[0] if matches else exact
    manifest = scene_dir / "output" / "experiment_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        variants = data.get("variants", [])
        for v in variants:
            if v.get("name") == LOCAL_TAU_SUBDIR:
                prompt = v.get("prompt")
                if prompt:
                    return prompt
        # Fallback: use first variant's prompt if it has one
        if variants and variants[0].get("prompt"):
            return variants[0]["prompt"]

    # Fallback for old-format scenes: read inputs/prompt.txt
    prompt_txt = scene_dir / "inputs" / "prompt.txt"
    if prompt_txt.exists():
        for line in prompt_txt.read_text().splitlines():
            if line.startswith("Shape prompt:"):
                return line[len("Shape prompt:"):].strip()

    return None

# ── MAIN ──────────────────────────────────────────────────────────────────
def generate():
    # Discover scenes from what was actually copied into local_tau/
    glb_files = sorted(METHOD_DIRS["local_tau"].glob("*.glb"))
    if not glb_files:
        print(f"ERROR: No GLB files found in {METHOD_DIRS['local_tau']}")
        return

    scene_names = [f.stem for f in glb_files]   # e.g. "acoustic_guitar"
    print(f"Found {len(scene_names)} scenes\n")

    trials = []

    for scene_name in scene_names:
        scene_id = f"scene_{scene_name}"

        # Prompt from experiment manifest
        prompt = get_prompt(scene_name)
        if not prompt:
            print(f"  SKIP {scene_name}: no prompt found in manifest")
            continue

        # Collect available GLB paths per method
        method_files = {}
        for method, directory in METHOD_DIRS.items():
            glb = directory / f"{scene_name}.glb"
            if glb.exists():
                method_files[method] = str(glb).replace("\\", "/")

        # SQ prior
        sq_glb = SQ_DIR / f"{scene_name}_sq.glb"
        ref_path = str(sq_glb).replace("\\", "/") if sq_glb.exists() else ""
        if not sq_glb.exists():
            print(f"  WARN {scene_name}: no SQ prior at {sq_glb}")

        # One trial per pair — skip pair if either method GLB is missing
        for method_a, method_b in PAIRS:
            if method_a not in method_files or method_b not in method_files:
                print(f"  SKIP {scene_name}: {method_a} vs {method_b} (missing GLB)")
                continue
            if random.random() < 0.5:
                method_a, method_b = method_b, method_a
            trials.append({
                "scene_id":  scene_id,
                "pair":      f"{method_a}_vs_{method_b}",
                "mapping":   {"A": method_a, "B": method_b},
                "outputs_a": [method_files[method_a]],
                "outputs_b": [method_files[method_b]],
                "ref":       ref_path,
                "prompt":    prompt,
            })
            print(f"  + {scene_name}: {method_a} vs {method_b}")

    if not trials:
        print("\nNo trials generated.")
        return

    random.shuffle(trials)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(trials, f, indent=2)

    n_scenes = len(scene_names)
    print(f"\n✓ Wrote {len(trials)} trials to {OUTPUT_FILE}")
    print(f"  ({n_scenes} scenes × {len(PAIRS)} pairs)")
    print(f"  Each participant sees SCENES_PER_PAIR×2 trials (set in main_tau.js)")

if __name__ == "__main__":
    generate()
