#!/usr/bin/env bash
#
# A/B test: run the SUPERDEC-driven appearance correspondence
# (feature/superdec-correspondence) and the original PartField + k-means
# correspondence (main) on the same inputs, then render side-by-side
# spinning videos of out_app.glb so the human can decide merge / abandon.
#
# Usage:
#   bash scripts/ab_compare_superdec_vs_partfield.sh \
#       --text_prompt "A wooden chair" \
#       --appearance_mesh examples/lamp.glb \
#       --shape_superquadric examples/superquadrics/chair_sq.npz \
#       --shape_tau 6.0 \
#       --convert_yup_to_zup \
#       --out_root /work/scratch/$USER/spaceflow/ab_runs/lamp_to_chair
#
# Notes on inputs:
#   * --shape_superquadric is the *structure* shape (the geometric
#     scaffolding that drives generation together with the text prompt);
#     it is independent of the appearance mesh. Pre-built files under
#     examples/superquadrics/: chair_sq.npz, plane_sq.npz, sofa_sq.npz,
#     car_sq.npz (no lamp_sq.npz exists yet — generate one with
#     sq_ui/scripts/superdec_infer.py if you want a lamp structure).
#   * The appearance mesh contributes only material/texture features;
#     mismatching geometry between structure and appearance is OK.
#
# Required env (defaults below assume the cluster layout used by
# sq_ui/setup_superdec.sh):
#   SQ_SUPERDEC_VENV            = /work/scratch/$USER/spaceflow/superdec_ui/venv
#   SQ_SUPERDEC_CHECKPOINT_DIR  = $SQ_SUPERDEC_VENV/../weights/normalized
#
# This script:
#   1) Confirms we are on feature/superdec-correspondence.
#   2) Runs ./run.py with --guidance_mode appearance into <out_root>/superdec/.
#   3) Removes TRELLIS's repo-root merged_mesh_voxelized.ply, stashes *tracked*
#      local edits only (not `stash -u`, which breaks on third_party perms),
#      switches to main, runs the same command into <out_root>/partfield/.
#   4) Switches back and pops the stash.
#   5) Prints the paths to the two out_app.glb files plus the SUPERDEC
#      diagnostics so the human can render both and compare boundary
#      sharpness + part bleeding.
#
# Decision rule of thumb (see plan §Success criteria):
#   • Sharper part boundaries + less cross-part texture bleeding in the
#     SUPERDEC output  -> merge feature/superdec-correspondence.
#   • Mean confidence < 0.2 or unmatched_segments / P_q > 0.4 from
#     superdec/superdec_summary.json -> matcher is unstable on this
#     input; first try raising superdec_match_iters or lowering
#     superdec_conf_threshold before deciding.
#   • SUPERDEC output looks visibly worse (e.g. material leaks across a
#     primitive boundary that SUPERDEC mis-segments) -> abandon, keep
#     feature branch unmerged for follow-up.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEXT_PROMPT=""
APP_PATH=""
OUT_ROOT=""
SHAPE_SQ_PATH=""
SHAPE_TAU="6.0"
CONVERT_YUP_TO_ZUP=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --text|--text_prompt) TEXT_PROMPT="$2"; shift 2 ;;
    --appearance|--appearance_mesh) APP_PATH="$2"; shift 2 ;;
    --shape_superquadric|--shape_superquadric_path) SHAPE_SQ_PATH="$2"; shift 2 ;;
    --shape_tau) SHAPE_TAU="$2"; shift 2 ;;
    --convert_yup_to_zup) CONVERT_YUP_TO_ZUP=1; shift ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,40p' "$0" | sed 's/^# //'
      exit 0
      ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

usage() {
  cat <<USAGE >&2
Usage: $0 \\
    --text_prompt "<prompt>" \\
    --appearance_mesh <path/to/app_mesh.glb> \\
    --shape_superquadric <path/to/structure_sq.npz> \\
    [--shape_tau 6.0] \\
    [--convert_yup_to_zup] \\
    --out_root <dir> \\
    [extra run.py args ...]

Pre-built superquadric files under examples/superquadrics/:
  chair_sq.npz, plane_sq.npz, sofa_sq.npz, car_sq.npz, ...
  (no lamp_sq.npz exists; generate one via sq_ui/scripts/superdec_infer.py
   if you need a lamp structure).
USAGE
}

if [[ -z "$TEXT_PROMPT" || -z "$APP_PATH" || -z "$OUT_ROOT" || -z "$SHAPE_SQ_PATH" ]]; then
  usage
  exit 2
fi
if [[ ! -f "$APP_PATH" ]]; then
  echo "appearance mesh not found: $APP_PATH" >&2
  exit 2
fi
if [[ ! -f "$SHAPE_SQ_PATH" ]]; then
  echo "shape superquadric NPZ not found: $SHAPE_SQ_PATH" >&2
  echo "  available files:" >&2
  ls "$REPO_ROOT/examples/superquadrics" 2>/dev/null | sed 's/^/    /' >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"/{superdec,partfield}
SUPERDEC_DIR="$OUT_ROOT/superdec"
PARTFIELD_DIR="$OUT_ROOT/partfield"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/envs/guideflow3d/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "GuideFlow3D venv python not found: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN explicitly if your venv is elsewhere." >&2
  exit 3
fi

echo "============================================================"
echo "SUPERDEC arm  -> $SUPERDEC_DIR"
echo "PartField arm -> $PARTFIELD_DIR"
echo "============================================================"

# --- 1. Confirm we are on feature/superdec-correspondence ---
ORIG_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$ORIG_BRANCH" != "feature/superdec-correspondence" ]]; then
  echo "Expected branch feature/superdec-correspondence, got $ORIG_BRANCH" >&2
  echo "Run: git checkout feature/superdec-correspondence" >&2
  exit 4
fi
CURRENT_BRANCH="$ORIG_BRANCH"

# --- 2. SUPERDEC arm ---
RUN_ARGS=(
  --text_prompt "$TEXT_PROMPT"
  --appearance_mesh "$APP_PATH"
  --shape_superquadric_path "$SHAPE_SQ_PATH"
  --shape_tau "$SHAPE_TAU"
  --guidance_mode appearance
)
if [[ "$CONVERT_YUP_TO_ZUP" == "1" ]]; then
  RUN_ARGS+=(--convert_yup_to_zup)
fi

echo
echo ">>> [SUPERDEC arm] running run.py on $CURRENT_BRANCH"
"$PYTHON_BIN" run.py \
  --output_dir "$SUPERDEC_DIR" \
  "${RUN_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

echo ">>> [SUPERDEC arm] done. out_app.glb: $SUPERDEC_DIR/out_app.glb"
echo ">>> [SUPERDEC arm] diagnostics:"
ls -la "$SUPERDEC_DIR/superdec/" 2>/dev/null || true

# --- 3. PartField arm: stash, switch, run, restore ---
# run.py / TRELLIS may touch tracked files in the repo root (e.g. .gitignore).
# TRELLIS also dumps merged_mesh_voxelized.ply in the repo cwd — remove it so
# it does not block checkout and so we never need `git stash -u`.
#
# Do NOT use `git stash push -u` on shared checkouts: it tries to stash *all*
# untracked files and then delete them from the tree; removal fails with
# "Permission denied" under third_party/ if those paths are not owned by you,
# and the whole stash aborts (exit 6).
rm -f "$REPO_ROOT/merged_mesh_voxelized.ply"

# IMPORTANT: `git stash create` + `git stash store` does NOT modify the working
# tree — use `git stash push` for tracked/index changes only.
STASHED=0
if [[ -n "$(git status --porcelain)" ]]; then
  echo
  echo ">>> stashing local tracked changes before switching branches"
  if git stash push -m "ab_compare_pre_main_${USER}_$(date +%s)"; then
    STASHED=1
  else
    echo "[ab] error: git stash push failed — cannot switch to main cleanly." >&2
    echo "[ab] hint: fix permissions or commit manually, then rerun." >&2
    exit 6
  fi
fi

trap 'set +e
echo "[ab] restoring branch + stash..."
git checkout "$ORIG_BRANCH" >/dev/null 2>&1
if [[ "$STASHED" == "1" ]]; then
  git stash pop >/dev/null 2>&1 || git stash pop
fi' EXIT

git checkout main
echo
echo ">>> [PartField arm] running run.py on main"
"$PYTHON_BIN" run.py \
  --output_dir "$PARTFIELD_DIR" \
  "${RUN_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
echo ">>> [PartField arm] done. out_app.glb: $PARTFIELD_DIR/out_app.glb"

# --- 4 + 5: trap restores branch + stash ---

cat <<INFO
============================================================
A/B comparison ready. Inspect:

  SUPERDEC arm  out_app   : $SUPERDEC_DIR/out_app.glb
  SUPERDEC arm  diagnostics: $SUPERDEC_DIR/superdec/
                              - superdec_segments_q.ply / _a.ply
                              - segment_correspondence_q.ply / _a.ply
                              - superdec_summary.json (P_q, P_a, tau, conf, |L_q|)
  PartField arm out_app   : $PARTFIELD_DIR/out_app.glb

Optional: also compare $SUPERDEC_DIR/out_gaussian_app.mp4 vs
          $PARTFIELD_DIR/out_gaussian_app.mp4 for spinning previews.

Decision rules — see header of this script.
============================================================
INFO
