#!/usr/bin/env bash
#
# A/B test: run the SUPERDEC-driven appearance correspondence
# (feature/superdec-correspondence) and the original PartField + k-means
# correspondence (main) on the same inputs, then render side-by-side
# spinning videos of out_app.glb so the human can decide merge / abandon.
#
# Usage:
#   bash scripts/ab_compare_superdec_vs_partfield.sh \
#       --text "wooden desk lamp" \
#       --appearance assets/lamp_appearance.glb \
#       --out_root /work/scratch/$USER/spaceflow/ab_runs/lamp
#
# Required env (defaults below assume the cluster layout used by
# sq_ui/setup_superdec.sh):
#   SQ_SUPERDEC_VENV            = /work/scratch/$USER/spaceflow/superdec_ui/venv
#   SQ_SUPERDEC_CHECKPOINT_DIR  = $SQ_SUPERDEC_VENV/../weights/normalized
#
# This script:
#   1) Confirms we are on feature/superdec-correspondence.
#   2) Runs ./run.py with --guidance_mode appearance into <out_root>/superdec/.
#   3) Stashes any local changes, switches to main, runs the same command into
#      <out_root>/partfield/.
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
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --text|--text_prompt) TEXT_PROMPT="$2"; shift 2 ;;
    --appearance|--appearance_mesh) APP_PATH="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$TEXT_PROMPT" || -z "$APP_PATH" || -z "$OUT_ROOT" ]]; then
  echo "Usage: $0 --text \"<prompt>\" --appearance <path/to/app_mesh.glb> --out_root <dir> [run.py extra args...]" >&2
  exit 2
fi
if [[ ! -f "$APP_PATH" ]]; then
  echo "appearance mesh not found: $APP_PATH" >&2
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
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "feature/superdec-correspondence" ]]; then
  echo "Expected branch feature/superdec-correspondence, got $CURRENT_BRANCH" >&2
  echo "Run: git checkout feature/superdec-correspondence" >&2
  exit 4
fi

# --- 2. SUPERDEC arm ---
echo
echo ">>> [SUPERDEC arm] running run.py on $CURRENT_BRANCH"
"$PYTHON_BIN" run.py \
  --text_prompt "$TEXT_PROMPT" \
  --appearance_mesh "$APP_PATH" \
  --output_dir "$SUPERDEC_DIR" \
  --guidance_mode appearance \
  "${EXTRA_ARGS[@]}"

echo ">>> [SUPERDEC arm] done. out_app.glb: $SUPERDEC_DIR/out_app.glb"
echo ">>> [SUPERDEC arm] diagnostics:"
ls -la "$SUPERDEC_DIR/superdec/" 2>/dev/null || true

# --- 3. PartField arm: stash, switch, run, restore ---
if [[ -n "$(git status --porcelain)" ]]; then
  echo
  echo ">>> stashing local working-tree changes before switching branches"
  STASH_REF=$(git stash create "ab_compare_${USER}_$(date +%s)")
  git stash store -m "ab_compare_pre_main" "$STASH_REF" || true
  STASHED=1
else
  STASHED=0
fi

trap 'echo "[ab] restoring branch + stash..."; git checkout - >/dev/null 2>&1 || true; if [[ "$STASHED" == "1" ]]; then git stash pop >/dev/null 2>&1 || true; fi' EXIT

git checkout main
echo
echo ">>> [PartField arm] running run.py on main"
"$PYTHON_BIN" run.py \
  --text_prompt "$TEXT_PROMPT" \
  --appearance_mesh "$APP_PATH" \
  --output_dir "$PARTFIELD_DIR" \
  --guidance_mode appearance \
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
