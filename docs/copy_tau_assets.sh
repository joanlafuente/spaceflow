#!/usr/bin/env bash
# Copies GLB assets from quality-checked experiment directory into docs/static/data/tau/.
# Run from the docs/ directory: bash copy_tau_assets.sh
set -euo pipefail

SRC="/work/courses/3dv/team3/spaceflow-minimal/83_final_examples"
DST="$(cd "$(dirname "$0")" && pwd)/static/data/tau"

echo "==> Clearing old GLBs from $DST"
rm -f "$DST/local_tau/"*.glb "$DST/tau_3/"*.glb "$DST/tau_10/"*.glb "$DST/sq_priors_glbs/"*.glb

mkdir -p "$DST/local_tau" "$DST/tau_3" "$DST/tau_10" "$DST/sq_priors_glbs"

ok=0; missing_scenes=()

for scene_dir in "$SRC"/*_full_experiment; do
  base=$(basename "$scene_dir" _full_experiment)
  out="$scene_dir/output"

  m1="$out/01_local_tau3_tau10_polyak0p18/out_sim_geometry.glb"
  m2="$out/02_global_tau3_polyak0/out_sim_geometry.glb"
  m3="$out/03_global_tau10_polyak0/out_sim_geometry.glb"
  sq="$out/01_local_tau3_tau10_polyak0p18/input_superquadrics_colored.glb"

  any_missing=0
  for f in "$m1" "$m2" "$m3" "$sq"; do
    if [[ ! -f "$f" ]]; then
      echo "  MISSING: $f"
      any_missing=1
    fi
  done
  if [[ $any_missing -eq 1 ]]; then
    missing_scenes+=("$base")
    continue
  fi

  cp "$m1" "$DST/local_tau/${base}.glb"
  cp "$m2" "$DST/tau_3/${base}.glb"
  cp "$m3" "$DST/tau_10/${base}.glb"
  cp "$sq" "$DST/sq_priors_glbs/${base}_sq.glb"
  echo "  OK  $base"
  ok=$((ok+1))
done

# Monitor special case: non-standard dir name, no tau_3 GLB needed
monitor_dir="$SRC/19_monitor_with_an_attached_keyboard_full_experiment_only_t10_and_tbyparts"
monitor_base="19_monitor_with_an_attached_keyboard"
if [[ -f "$monitor_dir/output/01_local_tau3_tau10_polyak0p18/out_sim_geometry.glb" &&
      -f "$monitor_dir/output/03_global_tau10_polyak0/out_sim_geometry.glb" &&
      -f "$monitor_dir/output/01_local_tau3_tau10_polyak0p18/input_superquadrics_colored.glb" ]]; then
  cp "$monitor_dir/output/01_local_tau3_tau10_polyak0p18/out_sim_geometry.glb"       "$DST/local_tau/${monitor_base}.glb"
  cp "$monitor_dir/output/03_global_tau10_polyak0/out_sim_geometry.glb"               "$DST/tau_10/${monitor_base}.glb"
  cp "$monitor_dir/output/01_local_tau3_tau10_polyak0p18/input_superquadrics_colored.glb" "$DST/sq_priors_glbs/${monitor_base}_sq.glb"
  echo "  OK  $monitor_base (special case, no tau_3 appearance output)"
  ok=$((ok+1))
else
  echo "  MISSING: monitor GLBs not found at $monitor_dir"
  missing_scenes+=("$monitor_base")
fi

echo ""
echo "==> Done: $ok scenes copied (${#missing_scenes[@]} skipped)"
if [[ ${#missing_scenes[@]} -gt 0 ]]; then
  echo "    Skipped: ${missing_scenes[*]}"
fi
